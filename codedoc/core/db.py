"""File hash and source-read utilities for codedoc."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Canonical source-read encoding.  Must stay in sync with the read in
# ``codedoc/core/execution.py`` so a character count computed during planning is
# identical to ``len(content)`` seen by the orchestrator.
SOURCE_TEXT_ENCODING = "utf-8-sig"


def compute_file_hash(file_path: Path) -> str:
    """Compute a SHA256 hash of file content."""
    hash_sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()


def read_source_text(file_path: Path) -> str:
    """Read a source file as text using the canonical pipeline encoding.

    Mirrors the read in ``codedoc/core/execution.py`` (``utf-8-sig`` with
    ``errors="replace"``) so the character count used for cache identity during
    planning matches ``len(content)`` seen by the orchestrator at processing
    time.
    """
    return Path(file_path).read_text(encoding=SOURCE_TEXT_ENCODING, errors="replace")


def source_char_count(file_path: Path, *, ceiling: int) -> int:
    """Source character count for the truncation decision, cheaply bounded.

    A file whose byte size is at most *ceiling* can never have more than
    *ceiling* characters (UTF-8 yields at most one character per byte, and
    ``errors="replace"`` at most one replacement character per undecodable byte),
    so it is never truncated and the exact count is irrelevant; the byte size
    (already ``<= ceiling``) is returned without reading the file.  Only larger
    files are decoded for the exact count.  The return value is guaranteed correct
    only relative to *ceiling* — ``value > ceiling`` iff the true character count
    exceeds *ceiling* — which is all the truncation decision needs.
    """
    path = Path(file_path)
    size = path.stat().st_size
    if size <= ceiling:
        return size
    return len(read_source_text(path))
