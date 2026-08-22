"""
Processing queue for codedoc.

Tracks which files are pending (unchecked) and which have been
documented (checked). Supports dependency-aware ordering —
files whose imports have already been processed are prioritised.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# File status constants
STATUS_UNCHECKED = "unchecked"
STATUS_IN_PROGRESS = "in_progress"
STATUS_CHECKED = "checked"
STATUS_FAILED = "failed"
STATUS_SKIPPED_INSUFFICIENT_SOURCE = "skipped_insufficient_source"


class ProcessingQueue:
    """
    Manages the ordered list of files to document.

    Usage:
        q = ProcessingQueue()
        q.add(file_descriptor)          # enqueue a file
        item = q.next()                 # pop next pending file
        q.mark_checked(rel_path)        # mark as done
        q.mark_failed(rel_path, reason) # mark as failed (won't retry)
    """

    def __init__(self) -> None:
        self._queue: deque[dict] = deque()
        self._state: dict[str, dict] = {}  # rel_path → {status, descriptor, error}

    # ------------------------------------------------------------------
    # Adding files
    # ------------------------------------------------------------------

    def add(self, descriptor: dict) -> None:
        """
        Add a file to the queue if it hasn't been seen before.
        descriptor must contain at least 'rel_path'.
        """
        rel = descriptor["rel_path"]
        if rel in self._state:
            return  # already known — don't add twice
        self._state[rel] = {
            "status": STATUS_UNCHECKED,
            "descriptor": descriptor,
            "error": None,
        }
        self._queue.append(descriptor)
        logger.debug("Queued: %s", rel)

    def add_many(self, descriptors: list[dict]) -> None:
        for d in descriptors:
            self.add(d)

    # ------------------------------------------------------------------
    # Consuming files
    # ------------------------------------------------------------------

    def next(self) -> dict | None:
        """
        Return the next unchecked file descriptor, or None if empty.
        Skips anything already checked / failed / in-progress.
        """
        while self._queue:
            item = self._queue.popleft()
            rel = item["rel_path"]
            status = self._state[rel]["status"]
            if status == STATUS_UNCHECKED:
                self._state[rel]["status"] = STATUS_IN_PROGRESS
                return item
        return None

    def has_pending(self) -> bool:
        return any(
            s["status"] in (STATUS_UNCHECKED, STATUS_IN_PROGRESS)
            for s in self._state.values()
        )

    # ------------------------------------------------------------------
    # Status updates
    # ------------------------------------------------------------------

    def mark_checked(self, rel_path: str) -> None:
        if rel_path in self._state:
            self._state[rel_path]["status"] = STATUS_CHECKED
            logger.debug("Checked: %s", rel_path)

    def mark_failed(self, rel_path: str, reason: str = "") -> None:
        """Store and render an already-bounded caller-owned failure reason.

        Provider/user-derived exceptions are projected by the execution
        boundary before reaching this queue. The queue deliberately neither
        reclassifies nor expands the reason, so its WARNING log and
        :meth:`iter_failed` expose the same bounded text.
        """
        if rel_path in self._state:
            self._state[rel_path]["status"] = STATUS_FAILED
            self._state[rel_path]["error"] = reason
            logger.warning("Failed: %s — %s", rel_path, reason)

    def mark_skipped_insufficient_source(self, rel_path: str, reason: str) -> None:
        if rel_path in self._state:
            self._state[rel_path]["status"] = STATUS_SKIPPED_INSUFFICIENT_SOURCE
            self._state[rel_path]["error"] = reason
            logger.debug("Skipped insufficient source: %s — %s", rel_path, reason)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status_of(self, rel_path: str) -> str:
        return self._state.get(rel_path, {}).get("status", "unknown")

    def all_checked(self) -> bool:
        return all(
            s["status"]
            in (STATUS_CHECKED, STATUS_FAILED, STATUS_SKIPPED_INSUFFICIENT_SOURCE)
            for s in self._state.values()
        )

    def stats(self) -> dict:
        counts = {STATUS_UNCHECKED: 0, STATUS_IN_PROGRESS: 0,
                  STATUS_CHECKED: 0, STATUS_FAILED: 0,
                  STATUS_SKIPPED_INSUFFICIENT_SOURCE: 0}
        for s in self._state.values():
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        return counts

    def iter_checked(self) -> Iterator[dict]:
        for s in self._state.values():
            if s["status"] == STATUS_CHECKED:
                yield s["descriptor"]

    def iter_failed(self) -> Iterator[dict]:
        for s in self._state.values():
            if s["status"] == STATUS_FAILED:
                yield {**s["descriptor"], "error": s["error"]}

    def snapshot(self) -> dict[str, str]:
        """Return {rel_path: status} for all known files."""
        return {rel: s["status"] for rel, s in self._state.items()}
