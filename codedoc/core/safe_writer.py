"""
Live JSON backup writer for the codedoc pipeline.

This writer is the **default** crash-safety mechanism for every run —
no longer opt-in via ``--safe-mode``.

Crash-recovery data is kept in its own dedicated file named
``crash_recovery.json``, **never** the
stable output.  The recovery file is created before AI work begins and updated
after each completed file.  The stable completed output (the final JSON for
``json``/``both`` or the Markdown for ``md``/``both``) is written only once, on
clean completion, by ``write_project_outputs`` — it is not opened, truncated, or
mutated while a run is in progress.  After the stable output is written,
``SafeWriter.delete()`` removes the recovery file.

Format behaviour
----------------
All formats
    The recovery file is the exact ``crash_recovery.json`` in the output
    directory.  On clean completion the stable output is written first and the
    recovery file is then deleted by ``SafeWriter.delete()``.  On interrupt or
    failure the stable output is left intact and the recovery file is preserved
    so the next run can resume compatible ordinary files. Completed fresh-split
    files are preserved at file level but deliberately re-documented under the
    active fresh-execution policy.

Banner
------
While the run is incomplete the JSON contains a top-level ``_crash_safety``
key as the first entry, clearly marking it as a crash-recovery backup.
``write_project_outputs`` writes the final clean JSON without this key.

Queue-order writes
------------------
The ``files`` array in every flush follows the topological processing order
provided via ``set_queue_order()``, not arbitrary path sorting.  Files that
completed out of order are stored in memory keyed by path and re-sorted on
every flush.

Thread safety
-------------
``record()`` and ``has_record()`` acquire a lock — safe to call from multiple
parallel worker threads simultaneously.

Atomic writes
-------------
Each flush writes to a ``.tmp`` sibling first, then renames into place via
``Path.replace``, so a crash mid-write never corrupts the output file.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from codedoc.core.block_manager import atomic_write_text
from codedoc.core.db import compute_file_hash
from codedoc.core.io_diagnostics import (
    category_reason,
    classify_os_error,
    describe_cause,
)
from codedoc.core.project_view import clean_file_record
from codedoc.core.file_division import SPLIT_PARTIAL_SCHEMA_VERSION, ReductionNodeState, SplitTreeState
from codedoc.utils.errors import LiveBackupWriteError, OutputError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_STATUS_IN_PROGRESS = "in_progress"

_CRASH_SAFETY_BANNER = (
    "INCOMPLETE RUN - codedoc is still generating or stopped before completion. "
    "This JSON is a crash-recovery backup containing only files that were "
    "successfully documented so far. Re-run the same command to continue: "
    "compatible completed ordinary files can resume, while completed fresh-split "
    "files are deliberately re-documented from scratch."
)


class SafeWriter:
    """
    Live-output writer that persists every completed file result immediately.

    Typical pipeline usage::

        recovery_path = output_dir / RECOVERY_FILENAME
        sw = SafeWriter(recovery_path, output_format, entry_rel, file_map, identity)
        sw.set_queue_order(ordered_selected_paths)
        sw.load()              # pre-populate from any existing records + ownership check
        sw.initialize_empty()  # flush empty in-progress banner before AI starts
        # ... process files ...
        sw.record(rel_path, result, file_hash)  # called in worker thread after each file
        # ... write_project_outputs writes the final clean stable output ...
        sw.delete()            # removes the dedicated recovery file (all formats)

    Attributes
    ----------
    path : Path
        Absolute path to the dedicated crash-recovery file.
    size : int
        Number of file results currently recorded in memory.
    """

    def __init__(
        self,
        backup_path: Path,
        output_format: str,
        entry_file: str | None,
        file_map: dict[str, dict],
        recovery_identity: dict | None = None,
    ) -> None:
        self._path: Path = backup_path
        self._output_format: str = output_format
        self._entry_file: str | None = entry_file
        self._file_map: dict[str, dict] = file_map
        # Versioned run identity emitted inside the ``_codedoc`` block on every
        # flush, so a partially written recovery file already carries the identity
        # used to decide whether a later identical run may resume from it.
        self._recovery_identity: dict | None = recovery_identity
        self._lock: threading.Lock = threading.Lock()
        self._clean_records: dict[str, dict] = {}
        self._tree_states: dict[str, SplitTreeState] = {}
        # rel_paths actually written via record() during THIS run, as opposed to
        # records preloaded from a prior output by load().  Used to decide whether
        # a rate-limited file may be treated as "already done" (only if a worker
        # recorded it this run) or must be retried (A4).
        self._recorded_this_run: set[str] = set()
        self._queue_order: list[str] = []
        self._created_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_queue_order(self, ordered_paths: list[str]) -> None:
        """Set the topological processing order for the ``files`` array.

        Must be called before ``initialize_empty()`` and before any
        ``record()`` calls so every flush produces a correctly ordered
        ``files`` array.  Safe to call from the main thread before any
        workers start.
        """
        with self._lock:
            self._queue_order = list(ordered_paths)

    def load(
        self,
        preloaded: dict[str, dict] | None = None,
        *,
        include_partial_files: bool = False,
        preloaded_partials: Mapping[str, SplitTreeState] | None = None,
    ) -> dict[str, dict]:
        """
        Pre-populate in-memory state from an existing recovery file or output.

        Parameters
        ----------
        preloaded:
            The merged reuse record set computed by the canonical resume
            boundary (stable completed output overlaid with any compatible
            exact ``crash_recovery.json`` records). Seeding the writer with this
            set means every partial flush of the dedicated recovery file is a
            self-contained snapshot, so an interrupted run resumes correctly
            even if the prior stable output later changes or disappears. This
            replaces the older behaviour where recovery state *was* the stable
            output and ``load()`` re-read it directly.
        include_partial_files:
            Whether split-only checkpoints may be re-read from the document at
            this boundary.  Disabled by default, so an existing split recovery
            cannot be re-flushed by a truncate run.  Production code leaves this
            off and supplies ``preloaded_partials`` instead; every partial that
            the document contains is *parseable*, but only the pipeline knows
            which are *retainable*.
        preloaded_partials:
            The exact pipeline-authorized retention set, keyed by normalized
            ``rel_path``.  Copied into writer state without flushing, so loading
            never writes an empty checkpoint.  Seeding from this set rather than
            from every parsed partial is what keeps rejected, stale-hash,
            unselected, and unrelated checkpoints from becoming deletion
            authority in :meth:`has_partial_state`.

        Ownership guard
        ---------------
        If a file exists at the target recovery path but is not a codedoc output
        (unreadable / malformed JSON or no ``_codedoc`` metadata block) a
        :class:`ConfigError` is raised before any LLM work begins.  In normal
        operation the active recovery path was already validated by the exact
        recovery-reader preflight, so this guard is defensive.

        A valid in-progress recovery file (``_codedoc.status = "in_progress"``)
        is fully accepted — this is the normal resume path.

        Returns ``{}`` — routing is handled by ``_load_existing_file_docs``
        in the pipeline, not here.
        """
        if preloaded:
            for rel_path, record in preloaded.items():
                if isinstance(record, dict) and record.get("path"):
                    self._clean_records[rel_path] = dict(record)

        # Seed the authorized retention set before the existence check so the
        # mapping is honoured verbatim; nothing is flushed here.
        for rel_path, state in (preloaded_partials or {}).items():
            if rel_path != state.rel_path:
                raise ValueError(
                    "preloaded partial key does not match its normalized "
                    f"rel_path: {rel_path!r} != {state.rel_path!r}"
                )
            self._tree_states[rel_path] = state

        if not self._path.exists():
            return {}

        from codedoc.core.document import read_codedoc_document
        from codedoc.utils.errors import ConfigError

        def _foreign_file_error() -> ConfigError:
            return ConfigError(
                f"'{self._path.name}' already exists but does not appear to be a "
                "codedoc output file.\n"
                "codedoc will not overwrite it to protect your data.\n\n"
                "To resolve this, choose one of:\n"
                f"  • Use a different output directory:   codedoc --output my_docs/\n"
                f"  • Delete or rename the conflicting file:  {self._path}"
            )

        # Ownership + parsing via the centralized reader.  A foreign or
        # malformed file raises before any LLM work begins.
        try:
            document = read_codedoc_document(
                self._path,
                include_partial_files=include_partial_files,
            )
        except (ConfigError, FileNotFoundError):
            raise _foreign_file_error()

        meta = document.metadata
        self._created_at = (
            meta.get("created_at")
            or meta.get("generated_at")
            or self._created_at
        )

        for f in document.files:
            if isinstance(f, dict) and f.get("path"):
                self._clean_records[f["path"]] = dict(f)
        for partial in document.partial_files:
            self._tree_states[partial.rel_path] = partial

        if self._clean_records:
            logger.info(
                "LiveBackup: loaded %d existing file record(s) from '%s' — "
                "they will be preserved in every partial write.",
                len(self._clean_records),
                self._path.name,
            )

        return {}

    def initialize_empty(self) -> None:
        """Flush the empty in-progress banner to disk before AI work starts.

        This ensures crash_recovery.json exists even if the process is killed
        before the first file finishes.  When records were pre-loaded from a
        previous run the flush includes those records (not truly empty), which
        is the correct behaviour — the banner is the important part.
        """
        with self._lock:
            self._flush_locked()

    def record(self, rel_path: str, result: dict, file_hash: str = "") -> None:
        """Clean and persist one completed file result immediately.

        Thread-safe — may be called concurrently from multiple worker threads.

        Parameters
        ----------
        rel_path:
            Project-relative file path (used as the record key).
        result:
            The full result dict returned by the orchestrator for this file.
        file_hash:
            Optional pre-computed SHA-256 hex digest.  Computed from
            ``file_map`` when omitted or empty.
        """
        if not file_hash:
            descriptor = self._file_map.get(rel_path, {})
            try:
                file_hash = (
                    compute_file_hash(descriptor["path"])
                    if descriptor.get("path")
                    else ""
                )
            except Exception:
                file_hash = ""

        raw_record = {
            "hash": file_hash,
            "file_path": rel_path,
            "language": result.get("language", ""),
            "documentation": result,
        }
        clean = clean_file_record(raw_record)

        with self._lock:
            # Transactional: persist to disk first, then keep the in-memory
            # markers only if the flush succeeded.  After a failed flush,
            # size/get_record/has_record/recorded_this_run must expose exactly
            # the state that existed before this call.
            had_record = rel_path in self._clean_records
            prior_record = self._clean_records.get(rel_path)
            was_recorded_this_run = rel_path in self._recorded_this_run

            self._clean_records[rel_path] = clean
            prior_partial = self._tree_states.pop(rel_path, None)
            self._recorded_this_run.add(rel_path)
            try:
                self._flush_locked()
            except LiveBackupWriteError:
                if had_record:
                    self._clean_records[rel_path] = prior_record
                else:
                    self._clean_records.pop(rel_path, None)
                if prior_partial is not None:
                    self._tree_states[rel_path] = prior_partial
                if not was_recorded_this_run:
                    self._recorded_this_run.discard(rel_path)
                raise

    def discard(self, rel_path: str) -> None:
        """Transactionally remove a preloaded or newly recorded file result."""
        with self._lock:
            if rel_path not in self._clean_records:
                return
            prior_record = self._clean_records[rel_path]
            was_recorded_this_run = rel_path in self._recorded_this_run
            self._clean_records.pop(rel_path)
            self._recorded_this_run.discard(rel_path)
            try:
                self._flush_locked()
            except LiveBackupWriteError:
                self._clean_records[rel_path] = prior_record
                if was_recorded_this_run:
                    self._recorded_this_run.add(rel_path)
                raise

    def record_tree_node(self, rel_path: str, node: ReductionNodeState) -> None:
        """Transactionally append one dependency-validated split-tree node
        checkpoint (split-partial schema version 2; see D11/D12/section 12).

        Every node for one file shares that file's `SplitTreeState` container;
        this appends *node* to the existing container (creating one if this is
        the file's first checkpointed node) and re-persists the whole
        container atomically.  Re-checkpointing an already-present node ID
        replaces it in place rather than duplicating it.
        """
        with self._lock:
            if rel_path in self._recorded_this_run:
                raise LiveBackupWriteError(
                    str(self._path),
                    "refusing to store a split node for a file completed in this run "
                    f"('{rel_path}').",
                )
            prior = self._tree_states.get(rel_path)
            existing_nodes = tuple(prior.nodes) if prior is not None else ()
            if any(existing.node_id == node.node_id for existing in existing_nodes):
                updated_nodes = tuple(
                    node if existing.node_id == node.node_id else existing
                    for existing in existing_nodes
                )
            else:
                updated_nodes = existing_nodes + (node,)
            self._tree_states[rel_path] = SplitTreeState(
                schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
                owner="codedoc-ai",
                rel_path=rel_path,
                content_hash=node.content_hash,
                division_plan_digest=node.division_plan_digest,
                reduction_tree_digest=node.reduction_tree_digest,
                nodes=updated_nodes,
            )
            try:
                self._flush_locked()
            except LiveBackupWriteError:
                if prior is None:
                    self._tree_states.pop(rel_path, None)
                else:
                    self._tree_states[rel_path] = prior
                raise

    def get_tree_state(self, rel_path: str) -> SplitTreeState | None:
        """Return the validated split-tree state for *rel_path*, if any."""
        with self._lock:
            return self._tree_states.get(rel_path)

    def has_partial_state(self) -> bool:
        """Whether any pipeline-authorized split checkpoint remains retained.

        This is the recovery-deletion authority.  It is meaningful only because
        writer partial state is seeded from the pipeline's retention set: it
        answers "is there checkpointed work still worth resuming?", not "did the
        recovery file happen to contain a parseable partial?".
        """
        with self._lock:
            return bool(self._tree_states)

    def clear_tree_state(self, rel_path: str) -> None:
        """Transactionally remove one file's split-tree checkpoint."""
        with self._lock:
            prior = self._tree_states.pop(rel_path, None)
            if prior is None:
                return
            try:
                self._flush_locked()
            except LiveBackupWriteError:
                self._tree_states[rel_path] = prior
                raise

    def has_record(self, rel_path: str) -> bool:
        """Return True if *rel_path* is already recorded in memory.

        Used by the ladder retry logic to avoid re-submitting a file that
        a worker successfully recorded before batch cancellation.
        """
        with self._lock:
            return rel_path in self._clean_records

    def recorded_this_run(self, rel_path: str) -> bool:
        """Return True only if *rel_path* was written via ``record()`` this run.

        Unlike :meth:`has_record`, this excludes records preloaded by
        :meth:`load` from a prior output.  The rate-limit retry logic uses this
        so a *changed* file that was merely preloaded (stale) is retried rather
        than silently restored from old documentation (A4).
        """
        with self._lock:
            return rel_path in self._recorded_this_run

    def get_record(self, rel_path: str) -> dict | None:
        """Return a copy of the clean record for *rel_path*, or ``None``.

        Lets the pipeline recover the already-persisted result for a file that
        was recorded by a worker but whose future later surfaced a (now moot)
        rate-limit error — so the final output reflects the real record instead
        of an empty placeholder.
        """
        with self._lock:
            record = self._clean_records.get(rel_path)
            return dict(record) if record is not None else None

    def delete(self) -> None:
        """Remove the dedicated crash-recovery file after a clean completion.

        The recovery file is the exact ``crash_recovery.json``
        for **every** output format, never the stable output.  Clean completion
        writes the stable output first (via ``write_project_outputs``) and only
        then calls this to remove the recovery file.  Deletion is therefore a
        required part of clean completion for ``json``, ``both``, and ``md``
        alike.

        The guard ``status == "in_progress"`` ensures this only ever unlinks an
        active in-progress recovery file we own — a completed CodeDoc JSON that
        happens to share the name is left intact.  A missing file is a no-op
        (nothing was initialized, e.g. an all-reused run).

        If the validated active recovery file exists and cannot be unlinked, the
        stable output may already hold the completed document; that is *not*
        rolled back.  Instead the recovery file is preserved and an
        :class:`OutputError` naming the path is raised so the run is reported as
        unsuccessful and the next invocation can finalize again.
        """
        if not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            meta = data.get("_codedoc", {})
            if not isinstance(meta, dict) or meta.get("status") != _STATUS_IN_PROGRESS:
                return
        except Exception:
            return

        try:
            self._path.unlink()
            logger.info(
                "LiveBackup: removed crash-recovery file '%s' after clean completion.",
                self._path.name,
            )
        except OSError as exc:
            raise OutputError(
                str(self._path),
                f"could not remove crash-recovery file '{self._path.name}' after "
                "writing the completed stable output; the stable output is intact "
                "and the recovery file was preserved — re-run to finalize again.",
            ) from exc

    @property
    def path(self) -> Path:
        """Absolute path to the dedicated crash-recovery file."""
        return self._path

    @property
    def size(self) -> int:
        """Number of results currently recorded in memory."""
        with self._lock:
            return len(self._clean_records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """Serialize current state to disk atomically.

        Must be called while ``self._lock`` is held.  Delegates the physical
        write to the canonical :func:`atomic_write_text` helper (unique temp
        sibling, flush + close, then rename) so a crash mid-write never leaves a
        corrupt output file.  A write failure is fatal: the prior valid backup
        is preserved and a :class:`LiveBackupWriteError` is raised so the
        pipeline stops rather than continue under a false crash-safety
        guarantee.

        Files in ``files`` are written in queue/topological order when
        ``set_queue_order()`` has been called.  Completed records whose
        rel_path is not yet in the queue order (should not happen in practice)
        are appended at the end in path-sorted order.

        A path completed in *this* run may never also be serialized as a split
        partial: the completed record supersedes it, so emitting both would
        publish a checkpoint that is already obsolete.  The collision is refused
        here, before serialization, so the prior recovery file is left intact
        and ``record()``/``record_tree_node()`` roll their in-memory state back.
        A preloaded stable record plus a current partial for the same path is
        legitimate and is not rejected.
        """
        collisions = sorted(self._recorded_this_run & set(self._tree_states))
        if collisions:
            raise LiveBackupWriteError(
                str(self._path),
                "refusing to serialize split partial state for file(s) already "
                f"completed in this run: {', '.join(collisions)}.",
            )
        if self._queue_order:
            ordered = []
            unsorted_keys = set(self._clean_records.keys())
            for rel_path in self._queue_order:
                if rel_path in self._clean_records:
                    ordered.append(self._clean_records[rel_path])
                    unsorted_keys.discard(rel_path)
            # Append any records not in the queue order (fallback)
            for rel_path in sorted(unsorted_keys):
                ordered.append(self._clean_records[rel_path])
            files_list = ordered
        else:
            files_list = sorted(
                self._clean_records.values(),
                key=lambda f: f.get("path", ""),
            )

        now = datetime.now(timezone.utc).isoformat()
        codedoc_block: dict = {
            "entry_file": self._entry_file,
            "created_at": self._created_at,
            "updated_at": now,
            "status": _STATUS_IN_PROGRESS,
            "live_backup": True,
        }
        if self._recovery_identity is not None:
            codedoc_block["recovery_identity"] = self._recovery_identity
        if self._tree_states:
            codedoc_block["partial_files"] = {
                rel_path: _tree_state_to_json(self._tree_states[rel_path])
                for rel_path in sorted(self._tree_states)
            }
        payload: dict = {
            "_crash_safety": _CRASH_SAFETY_BANNER,
            "_codedoc": codedoc_block,
            "files": files_list,
        }

        try:
            atomic_write_text(
                self._path,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )
        except Exception as exc:
            # The prior valid backup is left intact by atomic_write_text.  A
            # serialization or persistence failure is fatal — surface it so
            # execution stops and record() can roll back its in-memory markers.
            # Include the sanitized OS cause/category and an actionable
            # resume note (no secrets — only the local path and OS metadata).
            category = classify_os_error(exc)
            cause = describe_cause(exc)
            detail = f"{category_reason(category)} ({cause})" if cause else category_reason(category)
            raise LiveBackupWriteError(
                str(self._path),
                f"could not update crash recovery file '{self._path.name}': {detail}. "
                "Close any program viewing the file and rerun the same command; "
                "recovery data persisted before this failed write remains preserved, "
                "but the result being written is not guaranteed saved. Compatible "
                "completed ordinary files can resume; completed fresh-split files "
                "are deliberately re-documented from scratch.",
            ) from exc


def _tree_state_to_json(state: SplitTreeState) -> dict:
    """Serialize one node-keyed split-partial container (schema version 2)."""
    return {
        "schema_version": state.schema_version,
        "owner": state.owner,
        "rel_path": state.rel_path,
        "content_hash": state.content_hash,
        "division_plan_digest": state.division_plan_digest,
        "reduction_tree_digest": state.reduction_tree_digest,
        "nodes": {
            node.node_id: {
                "node_type": node.node_type,
                "content_hash": node.content_hash,
                "division_plan_digest": node.division_plan_digest,
                "reduction_tree_digest": node.reduction_tree_digest,
                "execution_identity_digest": node.execution_identity_digest,
                "unit_id": node.unit_id,
                "child_ids": list(node.child_ids),
                "coverage_leaf_ids": list(node.coverage_leaf_ids),
                "result_json": node.result_json,
            }
            for node in state.nodes
        },
    }
