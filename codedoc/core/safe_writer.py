"""
Live JSON backup writer for the codedoc pipeline.

0.8.0: This writer is now the **default** crash-safety mechanism for every
run — no longer opt-in via ``--safe-mode``.  A visible ``codedoc.json`` (or
the appropriate sibling) is created before AI work begins, updated after each
completed file, and finalized (banner removed) by ``write_project_outputs``
at the end of a clean run.

Format behaviour
----------------
JSON / both
    The live backup IS ``codedoc.json`` (or the named JSON file).
    ``write_project_outputs`` overwrites it cleanly at the end.  No explicit
    cleanup needed.

MD only
    The live backup is a JSON sibling of the requested MD file:
    ``codedoc.json`` (for ``--format md``) or ``report.json`` (for
    ``--output docs/report.md``).  After a clean MD conversion,
    ``SafeWriter.delete()`` removes the sibling so only ``codedoc.md``
    remains.  On interrupt the sibling is preserved and the next run resumes
    from it.

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

from codedoc.core.db import compute_file_hash
from codedoc.core.project_view import SCHEMA_VERSION, clean_file_record
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_STATUS_IN_PROGRESS = "in_progress"

_CRASH_SAFETY_BANNER = (
    "INCOMPLETE RUN - codedoc is still generating or stopped before completion. "
    "This JSON is a crash-recovery backup containing only files that were "
    "successfully documented so far. Re-run the same command to resume and "
    "produce the final clean output."
)


class SafeWriter:
    """
    Live-output writer that persists every completed file result immediately.

    Typical pipeline usage::

        backup_path = _resolve_live_backup_path(output_dir, fmt, json_fn, md_fn)
        sw = SafeWriter(backup_path, output_format, entry_rel, file_map)
        sw.set_queue_order(ordered_selected_paths)
        sw.load()              # pre-populate from any existing records + ownership check
        sw.initialize_empty()  # flush empty in-progress banner before AI starts
        # ... process files ...
        sw.record(rel_path, result, file_hash)  # called in worker thread after each file
        # ... write_project_outputs writes the final clean output ...
        sw.delete()            # removes live backup for MD-only runs

    Attributes
    ----------
    path : Path
        Absolute path to the live JSON backup file.
    size : int
        Number of file results currently recorded in memory.
    """

    def __init__(
        self,
        backup_path: Path,
        output_format: str,
        entry_file: str | None,
        file_map: dict[str, dict],
    ) -> None:
        self._path: Path = backup_path
        self._output_format: str = output_format
        self._entry_file: str | None = entry_file
        self._file_map: dict[str, dict] = file_map
        self._lock: threading.Lock = threading.Lock()
        self._clean_records: dict[str, dict] = {}
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

    def load(self) -> dict[str, dict]:
        """
        Pre-populate in-memory state from an existing live backup or output file.

        Ownership guard
        ---------------
        If a file exists at the target path but is not a codedoc output
        (unreadable / malformed JSON or no ``_codedoc`` metadata block) a
        :class:`ConfigError` is raised before any LLM work begins.

        A valid in-progress live backup (``_codedoc.status = "in_progress"``)
        is fully accepted — this is the normal resume path.

        Returns ``{}`` — routing is handled by ``_load_existing_file_docs``
        in the pipeline, not here.
        """
        if not self._path.exists():
            return {}

        from codedoc.utils.errors import ConfigError

        def _foreign_file_error() -> ConfigError:
            return ConfigError(
                f"'{self._path.name}' already exists but does not appear to be a "
                "codedoc output file.\n"
                "codedoc will not overwrite it to protect your data.\n\n"
                "To resolve this, choose one of:\n"
                f"  • Use a different output directory:   codedoc run --output my_docs/\n"
                f"  • Delete or rename the conflicting file:  {self._path}"
            )

        try:
            data = json.loads(self._path.read_text(encoding="utf-8-sig"))
        except Exception:
            raise _foreign_file_error()

        if not isinstance(data, dict) or not isinstance(data.get("_codedoc"), dict):
            raise _foreign_file_error()

        meta = data["_codedoc"]
        self._created_at = (
            meta.get("generated_at") or meta.get("created_at") or self._created_at
        )

        for f in data.get("files", []):
            if isinstance(f, dict) and f.get("path"):
                self._clean_records[f["path"]] = f

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

        This ensures the live backup exists even if the process is killed
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
            self._clean_records[rel_path] = clean
            self._flush_locked()

    def has_record(self, rel_path: str) -> bool:
        """Return True if *rel_path* is already recorded in memory.

        Used by the ladder retry logic to avoid re-submitting a file that
        a worker successfully recorded before batch cancellation.
        """
        with self._lock:
            return rel_path in self._clean_records

    def delete(self) -> None:
        """Remove the live JSON backup for MD-only runs after clean conversion.

        For JSON / both format, ``write_project_outputs`` already overwrote
        the live backup with the final clean output — nothing to remove.

        For MD-only runs the live backup is a JSON sibling (e.g. ``codedoc.json``
        next to ``codedoc.md``).  It is deleted here after a clean MD write so
        only the Markdown file remains.  If deletion fails (permission error,
        file locked on Windows) a warning is logged and the path is reported so
        the user knows the leftover file is safe to delete manually.

        The guard ``status == "in_progress"`` prevents accidentally deleting a
        completed JSON from a previous JSON-format run that happens to share the
        same filename.
        """
        if self._output_format != "md":
            return

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
                "LiveBackup: removed live JSON backup '%s' after successful MD write.",
                self._path.name,
            )
        except Exception as exc:
            logger.warning(
                "LiveBackup: could not remove live backup '%s' — it is safe to delete "
                "manually.  Path: %s  Cause: %s",
                self._path.name,
                self._path,
                exc,
            )

    @property
    def path(self) -> Path:
        """Absolute path to the live JSON backup file."""
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

        Must be called while ``self._lock`` is held.
        Writes to a ``.tmp`` sibling first, then renames into place so a
        crash mid-write never leaves a corrupt output file.

        Files in ``files`` are written in queue/topological order when
        ``set_queue_order()`` has been called.  Completed records whose
        rel_path is not yet in the queue order (should not happen in practice)
        are appended at the end in path-sorted order.
        """
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
        payload: dict = {
            "_crash_safety": _CRASH_SAFETY_BANNER,
            "_codedoc": {
                "entry_file": self._entry_file,
                "schema_version": SCHEMA_VERSION,
                "generated_at": self._created_at,
                "updated_at": now,
                "status": _STATUS_IN_PROGRESS,
                "live_backup": True,
            },
            "files": files_list,
        }

        tmp: Path = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning(
                "LiveBackup: flush failed — partial results may not be saved. "
                "Cause: %s",
                exc,
            )
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
