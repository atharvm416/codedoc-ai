"""
Incremental progress checkpoint for the codedoc pipeline.

After every file completes, the result is written to
``.codedoc_progress.json`` inside the output directory.  If the run is
interrupted (Ctrl-C, network failure, crash) the user can simply re-run
the same command — previously finished files are loaded from the
checkpoint and skipped, so only the remaining files are sent to the LLM.

The checkpoint is deleted automatically after a clean, successful run.

Thread safety
-------------
``record()`` acquires a lock before mutating state, so it is safe to call
from multiple worker threads simultaneously (parallel file processing).

Atomic writes
-------------
The file is written to a ``.tmp`` sibling first, then renamed into place.
On POSIX systems ``rename`` is atomic; on Windows it is best-effort
(Python 3.3+ ``Path.replace`` is used which handles the Windows case).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Public constants — importable by callers that need the filename.
CHECKPOINT_FILENAME = ".codedoc_progress.json"
_CHECKPOINT_VERSION = "1"


class Checkpoint:
    """
    Thread-safe incremental progress checkpoint.

    Typical usage inside the pipeline::

        cp = Checkpoint(output_dir, entry_file="src/main.py")
        prior = cp.load()           # {rel_path: result} from interrupted run
        # ... build agent_rels, skipping keys already in prior ...
        cp.record(rel_path, result) # called after each file completes
        cp.delete()                 # called once on clean completion

    Attributes
    ----------
    path : Path
        Absolute path to the checkpoint file on disk.
    size : int
        Number of file results currently recorded in memory.
    """

    def __init__(self, output_dir: Path, entry_file: str | None = None) -> None:
        self._path: Path = output_dir / CHECKPOINT_FILENAME
        self._entry_file: str | None = entry_file
        self._lock: threading.Lock = threading.Lock()
        self._results: dict[str, dict] = {}
        self._created_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, dict]:
        """
        Load results from an existing checkpoint file.

        Returns
        -------
        dict[str, dict]
            Mapping of ``rel_path → result`` for files already processed
            in a prior interrupted run.  Returns an empty dict when no
            checkpoint exists or when the file cannot be parsed.
        """
        if not self._path.exists():
            return {}

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Checkpoint at '%s' could not be read — starting fresh. Cause: %s",
                self._path.name,
                exc,
            )
            return {}

        if not data.get("codedoc_checkpoint"):
            logger.debug(
                "'%s' exists but is not a codedoc checkpoint — ignoring.",
                self._path.name,
            )
            return {}

        self._results = data.get("results", {})
        self._created_at = data.get("created_at", self._created_at)

        count = len(self._results)
        if count:
            logger.info(
                "Checkpoint loaded: %d previously completed file(s) will be resumed "
                "(created %s).",
                count,
                data.get("created_at", "unknown"),
            )

        return dict(self._results)

    def record(self, rel_path: str, result: dict, file_hash: str = "") -> None:
        """
        Record one completed file result and flush to disk immediately.

        Thread-safe — may be called concurrently from worker threads.

        Parameters
        ----------
        rel_path:
            Project-relative file path (used as the record key).
        result:
            The full result dict returned by the orchestrator for this file.
        file_hash:
            SHA-256 hex digest of the file at the time it was processed.
            When provided, it is embedded in the stored entry under the
            reserved key ``"_checkpoint_hash"`` so that the routing loop can
            detect whether the file was modified between the interrupted run
            and the next resume attempt.
        """
        entry = dict(result)
        if file_hash:
            entry["_checkpoint_hash"] = file_hash
        with self._lock:
            self._results[rel_path] = entry
            self._flush_locked()

    def delete(self) -> None:
        """
        Remove the checkpoint file after a clean run completes.

        Safe to call even when the file does not exist.
        """
        try:
            if self._path.exists():
                self._path.unlink()
                logger.debug("Checkpoint removed after clean run.")
        except Exception as exc:
            logger.warning(
                "Could not remove checkpoint file '%s': %s",
                self._path.name,
                exc,
            )

    @property
    def path(self) -> Path:
        """Absolute path to the checkpoint file."""
        return self._path

    @property
    def size(self) -> int:
        """Number of results currently recorded in memory."""
        with self._lock:
            return len(self._results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """
        Serialize state to disk atomically.

        Must be called while ``self._lock`` is held.
        Writes to a ``.tmp`` sibling first, then renames into place so a
        crash mid-write never leaves a corrupt checkpoint.
        """
        payload: dict = {
            "codedoc_checkpoint": True,
            "version": _CHECKPOINT_VERSION,
            "created_at": self._created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "entry_file": self._entry_file,
            "total_recorded": len(self._results),
            "results": self._results,
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
                "Checkpoint write failed — progress may be lost on interrupt. Cause: %s",
                exc,
            )
            # Best-effort cleanup of the temp file.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
