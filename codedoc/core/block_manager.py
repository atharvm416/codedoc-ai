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
from pathlib import Path


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
        tmp.replace(path)
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
