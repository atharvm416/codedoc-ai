"""
Safe-mode incremental output writer for the codedoc pipeline.

When ``--safe-mode`` is active, completed file results are written directly
to the real output file (``codedoc.json``) after every file, instead of being
held in RAM until the run finishes.  This means a Ctrl-C or crash always
leaves a readable partial output file on disk — no results are lost.

Format behaviour
----------------
JSON / both
    ``codedoc.json`` is written after every completed file and contains
    whatever has been documented so far.  The final ``write_project_outputs``
    call at the end of the pipeline overwrites it with the complete, polished
    output.  No explicit cleanup is required.

MD only
    ``.codedoc_build.json`` (the internal build file) is used as a temporary
    intermediate while the run is in progress (written after every file).
    After a successful run, ``write_project_outputs`` writes the final
    ``codedoc.md`` and then ``SafeWriter.delete()`` removes the intermediate.
    If the run fails before the MD conversion completes, the intermediate is
    preserved — the user still has partial output, and the next run resumes
    automatically.

Resume on re-run
----------------
Because the partial output is a structurally valid ``codedoc.json``, the
existing incremental logic in ``_load_existing_file_docs`` picks it up
automatically on the next run.  Files already documented are detected as
unchanged (same hash) and skipped; only the remaining files are sent to the
LLM.

Thread safety
-------------
``record()`` acquires a lock before mutating state — safe to call from
multiple parallel worker threads simultaneously.

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
from codedoc.core.output import BUILD_FILENAME
from codedoc.core.project_view import SCHEMA_VERSION, clean_file_record
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel embedded in the partial JSON so SafeWriter can identify its own
# intermediate files and avoid accidentally deleting completed outputs.
_STATUS_IN_PROGRESS = "in_progress"


class SafeWriter:
    """
    Incremental live-output writer for ``--safe-mode`` runs.

    After every file completes, the result is cleaned and written directly to
    the real JSON output file so the user always has partial output on disk.

    Typical pipeline usage::

        sw = SafeWriter(output_dir, "codedoc.json", "json", entry_rel, file_map)
        sw.load()                    # pre-populate with any in-progress entries
        # ... process files ...
        sw.record(rel_path, result)  # called after each file completes
        # ... write_project_outputs writes the final clean output ...
        sw.delete()                  # removes intermediate JSON for MD-only runs

    Attributes
    ----------
    path : Path
        Absolute path to the live output JSON file.
    size : int
        Number of file results currently recorded in memory.
    """

    def __init__(
        self,
        output_dir: Path,
        json_filename: str,
        output_format: str,
        entry_file: str | None,
        file_map: dict[str, dict],
    ) -> None:
        # For MD-only runs the per-file intermediate is the internal build
        # file (.codedoc_build.json) — the same file that write_project_outputs
        # uses — so the two codepaths share a single durable intermediate.
        # This avoids any collision with a user-owned codedoc.json and keeps
        # the intermediate under a clearly system-managed name.
        if output_format == "md":
            self._path: Path = output_dir / BUILD_FILENAME
        else:
            self._path = output_dir / json_filename
        self._output_format: str = output_format
        self._entry_file: str | None = entry_file
        self._file_map: dict[str, dict] = file_map
        self._lock: threading.Lock = threading.Lock()
        # rel_path → clean public file entry (written to the partial JSON)
        self._clean_records: dict[str, dict] = {}
        self._created_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, dict]:
        """
        Pre-populate in-memory state from an existing codedoc output file.

        Any file records already on disk — whether from an interrupted
        ``status = "in_progress"`` run *or* from a previously completed codedoc
        output — are loaded into memory so subsequent ``record()`` +
        ``_flush_locked()`` calls preserve them.  This is what guarantees that
        interrupting a safe-mode run never erases work that was already on disk:
        the first flush re-writes the prior records alongside the new ones.

        Ownership guard
        ---------------
        If a file exists at the target path but is **not** a codedoc output —
        unreadable / malformed JSON, a non-object JSON value, or a JSON object
        with no ``_codedoc`` metadata block — a :class:`ConfigError` is raised.
        SafeWriter refuses to flush over a foreign file at startup, before any
        LLM work begins.  This mirrors ``_check_file_ownership`` in
        ``output.py`` (which treats malformed files as foreign too).

        Returns an empty dict — SafeWriter does not use the checkpoint-results
        routing loop.  The existing ``_load_existing_file_docs`` logic already
        reads the partial JSON and handles unchanged files via hash comparison.
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
            # Unreadable / malformed file — treat as foreign and refuse to
            # overwrite it (consistent with output._check_file_ownership).
            raise _foreign_file_error()

        # A codedoc output is always a JSON object carrying a "_codedoc" block.
        # Anything else (a list, a number, an object without metadata) is foreign.
        if not isinstance(data, dict) or not isinstance(data.get("_codedoc"), dict):
            raise _foreign_file_error()

        meta = data["_codedoc"]

        # Restore the original generation timestamp so it stays stable across
        # resumes (falls back to "created_at" for older intermediates).
        self._created_at = meta.get("generated_at") or meta.get("created_at") or self._created_at

        # Pre-load every existing record — both in-progress intermediates and
        # completed outputs — so the first flush preserves prior work instead of
        # overwriting the file with only the records processed in this run.
        for f in data.get("files", []):
            if isinstance(f, dict) and f.get("path"):
                self._clean_records[f["path"]] = f

        if self._clean_records:
            logger.info(
                "SafeWriter: loaded %d existing file record(s) from '%s' — "
                "they will be preserved in every partial write and the final output.",
                len(self._clean_records),
                self._path.name,
            )

        # Return {} — routing is handled by _load_existing_file_docs, not here.
        return {}

    def record(self, rel_path: str, result: dict, file_hash: str = "") -> None:
        """
        Clean and persist one completed file result immediately.

        Thread-safe — may be called concurrently from multiple worker threads.

        Parameters
        ----------
        rel_path:
            Project-relative file path (used as the record key).
        result:
            The full result dict returned by the orchestrator for this file.
        file_hash:
            Optional pre-computed SHA-256 hex digest of the file.  When
            omitted or empty, the hash is computed from ``file_map``.
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

    def delete(self) -> None:
        """
        Remove the intermediate JSON for MD-only runs after a clean completion.

        For JSON / both formats, ``write_project_outputs`` already overwrites
        the file with the final output, so this method is a no-op.

        For MD-only runs, the intermediate ``.codedoc_build.json`` is deleted
        here — but only if it still carries ``status = "in_progress"``.  This
        guard prevents accidentally deleting a complete JSON output written by a
        previous run in a different format.

        If no intermediate file exists, this method returns silently.
        """
        if self._output_format != "md":
            return  # JSON / both: write_project_outputs already overwrote the file.

        if not self._path.exists():
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            meta = data.get("_codedoc", {})
            if not isinstance(meta, dict) or meta.get("status") != _STATUS_IN_PROGRESS:
                # Not our intermediate — could be a completed JSON from a prior run.
                return
        except Exception:
            return

        try:
            self._path.unlink()
            logger.info(
                "SafeWriter: removed intermediate JSON after successful MD write."
            )
        except Exception as exc:
            logger.warning(
                "SafeWriter: could not remove intermediate JSON '%s': %s",
                self._path.name,
                exc,
            )

    @property
    def path(self) -> Path:
        """Absolute path to the live output file."""
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
        """
        Serialize current state to disk atomically.

        Must be called while ``self._lock`` is held.
        Writes to a ``.tmp`` sibling first, then renames into place so a
        crash mid-write never leaves a corrupt output file.
        """
        files_list = sorted(
            self._clean_records.values(),
            key=lambda f: f.get("path", ""),
        )
        now = datetime.now(timezone.utc).isoformat()
        payload: dict = {
            "_codedoc": {
                "entry_file": self._entry_file,
                "schema_version": SCHEMA_VERSION,
                "generated_at": self._created_at,
                "updated_at": now,
                "status": _STATUS_IN_PROGRESS,
            },
            "files": files_list,
        }

        tmp: Path = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._path)  # atomic on POSIX; best-effort on Windows
        except Exception as exc:
            logger.warning(
                "SafeWriter: flush failed — partial results may not be saved. "
                "Cause: %s",
                exc,
            )
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
