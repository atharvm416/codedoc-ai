"""Atomic text writing for completed output and crash recovery.

``atomic_write_text`` is the single canonical helper for replacing a file's
contents without ever truncating the existing target in place.  It writes to a
uniquely named temporary sibling in the same directory, flushes and closes it,
then atomically renames it over the target via :meth:`Path.replace`.

Contract
--------
- UTF-8 encoding with LF line endings on every platform.
- Uniquely named temporary sibling in the target directory (concurrent writers
  never collide, and the final rename stays on one filesystem so it is atomic).
- The temporary file is explicitly flushed and closed before the rename; a
  flush or close failure is treated as a write failure.
- The existing target is never truncated or altered before the replacement.
- Only the temporary file created by this operation is cleaned up on failure.
- ``OSError`` propagates with its original cause available.

This is the only atomic-write implementation in the codebase; both the
completed-output and crash-recovery writers delegate to it.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from codedoc.core.io_diagnostics import (
    WINDOWS_TRANSIENT_LOCK_ERRORS,
)
from codedoc.utils.errors import ConfigError

# Bounded transient-lock retry budget for the final atomic replacement step.
# The first attempt is immediate; these are the sleeps *between* subsequent
# retries. Only the context-specific Windows codes in
# ``ATOMIC_REPLACE_RETRYABLE_WINERRORS`` are retried; every other failure
# raises immediately.
ATOMIC_REPLACE_RETRY_DELAYS_S: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
# WinError 5 (ERROR_ACCESS_DENIED) can be a short-lived replace collision on
# Windows even though it is not globally a lock diagnostic. Retry it only at
# this atomic-replacement boundary; if it persists, the original exception
# escapes and downstream diagnostics correctly classify it as permission
# denied. Codes 32/33 are the ordinary sharing/lock violations.
ATOMIC_REPLACE_RETRYABLE_WINERRORS: frozenset[int] = frozenset(
    {5, *WINDOWS_TRANSIENT_LOCK_ERRORS}
)

__all__ = [
    "ATOMIC_REPLACE_RETRY_DELAYS_S",
    "ATOMIC_REPLACE_RETRYABLE_WINERRORS",
    "WINDOWS_TRANSIENT_LOCK_ERRORS",
    "BlockError",
    "atomic_write_text",
    "create_text_exclusive",
    "replace_text_atomic_no_backup",
]


class BlockError(ConfigError):
    """Existing owned-block markers are malformed or unsafe."""


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
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
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


def create_text_exclusive(path: Path, text: str) -> None:
    """Create *path* with *text* (UTF-8), refusing any existing target.

    Uses ``O_CREAT | O_EXCL`` so creation is atomic and race-safe: if anything
    already exists at *path* — a regular file, a directory, or a symlink — the
    operation raises :class:`BlockError` rather than overwriting it.  Used by the
    config generator (``--init-config``) for the
    no-``--force`` no-overwrite creation path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    except FileExistsError as exc:
        raise BlockError(
            f"refusing to overwrite the existing path '{path}'. "
            "Choose a different path or pass --force."
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        _silent_unlink(path)
        raise


def replace_text_atomic_no_backup(path: Path, text: str) -> None:
    """Atomically replace a regular-file *path* with *text*, keeping no backup.

    The ``--force`` path for the config generator (``--init-config``). The
    product contract permits only
    one active config/support file, so ``--force`` is the user's explicit
    permission to discard the old bytes — no timestamped ``.bak-`` sibling is
    written.

    Only a regular file may be replaced: a symlink or directory at *path* raises
    :class:`BlockError`.  An absent target is created exclusively.  The current
    bytes are captured on entry and re-read immediately before the atomic
    replacement; a concurrent change in that window aborts with :class:`BlockError`
    rather than being silently overwritten.  On any failure the original file is
    left byte-identical (``atomic_write_text`` never truncates in place).
    """
    path = Path(path)
    if path.is_symlink():
        raise BlockError(f"refusing to replace symlink '{path}' with --force.")
    if path.exists() and path.is_dir():
        raise BlockError(f"refusing to replace directory '{path}' with --force.")
    if not path.exists():
        create_text_exclusive(path, text)
        return
    if not path.is_file():
        raise BlockError(f"refusing to replace non-regular file '{path}' with --force.")

    original = path.read_bytes()
    # Concurrent-change guard: re-read immediately before the atomic replace so a
    # change that slipped in since inspection is not silently discarded.
    if path.read_bytes() != original:
        raise BlockError(
            f"target '{path}' changed during inspection; aborting --force replacement."
        )
    atomic_write_text(path, text)


def _replace_with_lock_retry(tmp: Path, path: Path) -> None:
    """Atomically rename *tmp* over *path*, retrying only a transient Windows lock.

    The first ``replace`` attempt is immediate.  When — and only when — the
    failure is a Windows sharing/lock code (``winerror`` 32/33), or a
    potentially transient ``ERROR_ACCESS_DENIED`` from this exact replace
    operation (``winerror`` 5), the same already-written, flushed, fsynced, and
    closed temporary file is reused after a bounded sleep and the rename is
    retried. Every other failure (``ENOSPC``, read-only media, missing parent,
    directory collision, generic I/O) raises immediately with its cause intact.
    A persistent retryable failure still raises once the bounded budget is
    exhausted; a persistent winerror 5 is then diagnosed as permission denied,
    not as a lock.
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
            retryable = getattr(exc, "winerror", None) in ATOMIC_REPLACE_RETRYABLE_WINERRORS
            if attempt >= len(ATOMIC_REPLACE_RETRY_DELAYS_S) or not retryable:
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
