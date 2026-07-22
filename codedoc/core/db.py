"""File hash and source-read utilities for codedoc."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Canonical source-read encoding.  Must stay in sync with the read in
# ``read_source_snapshot`` so a character count computed during planning is
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

    Mirrors :func:`read_source_snapshot` (``utf-8-sig`` with
    ``errors="replace"`` and text-mode universal-newline translation) so the
    character count used for cache identity during planning matches
    ``len(content)`` seen by the orchestrator at processing time.
    """
    return Path(file_path).read_text(encoding=SOURCE_TEXT_ENCODING, errors="replace")


def decode_source_bytes(raw: bytes) -> str:
    """Decode raw source bytes exactly as :func:`read_source_text` would.

    ``Path.read_text()`` opens in text mode, so besides ``utf-8-sig`` decoding
    with ``errors="replace"`` it applies Python's universal-newline translation
    (``\\r\\n`` and ``\\r`` become ``\\n``).  That translation is part of the
    canonical text policy — prompt bytes, the ``len(content)`` truncation
    decision, and the ``_max_context_revision`` cache-identity key all derive
    from this text — so it is reproduced here rather than dropped.  It is
    applied *after* the raw-byte hash is taken, keeping the two representations
    deliberately distinct.
    """
    return (
        raw.decode(SOURCE_TEXT_ENCODING, errors="replace")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def read_source_snapshot(file_path: Path) -> tuple[str, str]:
    """Read raw bytes once; return ``(content_hash, content)`` from that read.

    ``content_hash`` is the SHA-256 of the raw bytes — byte-identical to
    :func:`compute_file_hash`. ``content`` is those same bytes decoded through
    :func:`decode_source_bytes`, so it is character-identical to
    :func:`read_source_text` for every input, including CRLF- and
    CR-terminated sources. The hash is never reconstructed by re-encoding the
    decoded string: BOM-bearing, invalid-UTF-8, and CRLF inputs are not
    reversible through that decode, so both values must come from the one raw
    read performed here.
    """
    raw = Path(file_path).read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    return content_hash, decode_source_bytes(raw)


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
