"""Atomic text writing for codedoc completed output and the live backup.

``atomic_write_text`` is the single canonical helper for replacing a file's
contents without ever truncating the existing target in place.  It writes to a
uniquely named temporary sibling in the same directory, flushes and closes it,
then atomically renames it over the target via :meth:`Path.replace`.

Contract
--------
- UTF-8 encoding.
- Uniquely named temporary sibling in the target directory (concurrent writers
  never collide, and the final rename stays on one filesystem so it is atomic).
- The temporary file is explicitly flushed and closed before the rename; a
  flush or close failure is treated as a write failure.
- The existing target is never truncated or altered before the replacement.
- Only the temporary file created by this operation is cleaned up on failure.
- ``OSError`` propagates with its original cause available.

This is the only atomic-write implementation in the codebase; both the
completed-output writers and the live-backup writer delegate to it.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from codedoc.core.io_diagnostics import (
    WINDOWS_TRANSIENT_LOCK_ERRORS,
    is_transient_lock,
)
from codedoc.utils.errors import ConfigError

# Bounded transient-lock retry budget for the final atomic replacement step
# (Workstream B).  The first attempt is immediate; these are the sleeps *between*
# subsequent retries, so the total added wait is deliberately below one second.
# Only a Windows sharing/lock violation (see ``WINDOWS_TRANSIENT_LOCK_ERRORS``)
# is retried; every other failure raises immediately.
ATOMIC_REPLACE_RETRY_DELAYS_S: tuple[float, ...] = (0.05, 0.15, 0.30)

__all__ = [
    "ATOMIC_REPLACE_RETRY_DELAYS_S",
    "WINDOWS_TRANSIENT_LOCK_ERRORS",
    "BlockError",
    "atomic_write_text",
    "merge_managed_block",
    "write_owned_block",
]


class BlockError(ConfigError):
    """Existing owned-block markers are malformed or unsafe."""


def _validate_block_arguments(
    interior_lines: list[str], start_marker: str, end_marker: str
) -> None:
    for name, marker in (("start_marker", start_marker), ("end_marker", end_marker)):
        if not isinstance(marker, str) or not marker or "\n" in marker or "\r" in marker:
            raise ValueError(f"{name} must be one non-empty logical line")
    if start_marker == end_marker:
        raise ValueError("start_marker and end_marker must be distinct")
    if not isinstance(interior_lines, list):
        raise ValueError("interior_lines must be a list of logical lines")
    for line in interior_lines:
        if (
            not isinstance(line, str)
            or "\n" in line
            or "\r" in line
            or line in {start_marker, end_marker}
        ):
            raise ValueError("interior_lines must contain valid non-marker logical lines")


def merge_managed_block(
    existing_text: str | None,
    interior_lines: list[str],
    start_marker: str,
    end_marker: str,
) -> str:
    """Return text with exactly one validated owned block merged into it."""
    _validate_block_arguments(interior_lines, start_marker, end_marker)
    newline = "\r\n" if existing_text and "\r\n" in existing_text else "\n"
    block = newline.join([start_marker, *interior_lines, end_marker]) + newline
    if not existing_text:
        return block

    lines = existing_text.splitlines(keepends=True)
    logical = [line.rstrip("\r\n") for line in lines]
    starts = [index for index, line in enumerate(logical) if line == start_marker]
    ends = [index for index, line in enumerate(logical) if line == end_marker]
    if not starts and not ends:
        return existing_text.rstrip("\r\n") + newline * 2 + block
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise BlockError("Existing managed-block markers are malformed")

    start_index, end_index = starts[0], ends[0]
    for line in logical[start_index + 1 : end_index]:
        if line in {start_marker, end_marker}:
            raise BlockError("Existing managed-block markers are nested")

    prefix = "".join(lines[: start_index + 1])
    if not prefix.endswith(("\n", "\r")):
        prefix += newline
    suffix = "".join(lines[end_index:])
    replacement = newline.join(interior_lines)
    if replacement:
        replacement += newline
    merged = prefix + replacement + suffix
    return existing_text if merged == existing_text else merged


def write_owned_block(
    path: Path,
    interior_lines: list[str],
    start_marker: str,
    end_marker: str,
) -> None:
    """Strictly read and atomically update one owned block in *path*."""
    _validate_block_arguments(interior_lines, start_marker, end_marker)
    path = Path(path)
    if path.is_symlink():
        raise BlockError(f"Owned-block target '{path}' is a symbolic link")
    if path.exists() and path.is_dir():
        raise BlockError(f"Owned-block target '{path}' is a directory")

    existing: str | None = None
    if path.exists():
        try:
            existing = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BlockError(f"Owned-block target '{path}' is not valid UTF-8") from exc
    merged = merge_managed_block(existing, interior_lines, start_marker, end_marker)
    if existing != merged:
        atomic_write_text(path, merged)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write *text* to *path* (UTF-8), never truncating in place.

    Raises
    ------
    OSError
        If the temporary file cannot be created, written, flushed, closed, or
        renamed over the target.  The existing target is left intact and the
        temporary file (if any) is removed.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    # mkstemp guarantees a unique name inside ``directory`` — the temporary
    # target can never resolve outside the target directory.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(directory)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
            # Flush + fsync surface any deferred write failure here, before the
            # rename, so a half-written temp file never replaces the target.
            handle.flush()
            os.fsync(handle.fileno())
        # mkstemp creates the temp file 0600; relax it to the umask-respecting
        # default so the result matches what a normal write_text would have
        # produced (the previous writer's behaviour) instead of silently making
        # output owner-only.  A no-op on platforms that ignore POSIX modes.
        _relax_to_default_mode(tmp)
        _replace_with_lock_retry(tmp, path)
    except Exception as exc:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        _silent_unlink(tmp)
        if isinstance(exc, OSError):
            raise
        raise OSError(f"Could not write UTF-8 text to '{path}'") from exc


def _replace_with_lock_retry(tmp: Path, path: Path) -> None:
    """Atomically rename *tmp* over *path*, retrying only a transient Windows lock.

    The first ``replace`` attempt is immediate.  When — and only when — the
    failure is a Windows sharing/lock violation (``winerror`` 32/33) the same
    already-written, flushed, fsynced, and closed temporary file is reused after
    a bounded sleep and the rename is retried.  Every other failure (permission
    denial without a lock code, ``ENOSPC``, read-only media, missing parent,
    directory collision, generic I/O) raises immediately with its cause intact.
    The provider is never re-contacted; this is a pure local filesystem retry.

    ``time.sleep`` is the only wait; tests monkeypatch it (and ``Path.replace``)
    so the suite never actually sleeps.
    """
    attempt = 0
    while True:
        try:
            tmp.replace(path)
            return
        except OSError as exc:
            if attempt >= len(ATOMIC_REPLACE_RETRY_DELAYS_S) or not is_transient_lock(exc):
                raise
            time.sleep(ATOMIC_REPLACE_RETRY_DELAYS_S[attempt])
            attempt += 1


def _relax_to_default_mode(path: Path) -> None:
    """Set *path* to the process default file mode (``0666`` masked by umask)."""
    try:
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(path, 0o666 & ~current_umask)
    except OSError:
        pass


def _silent_unlink(path: Path) -> None:
    """Remove *path* if present, swallowing cleanup-time OSErrors."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
