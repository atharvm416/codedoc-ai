"""Private OS-error classification and sanitized diagnostic formatting (0.10.1).

This module turns a raw local I/O failure into a small, stable *category* and a
concise, secret-free, user-facing message.  It is pure: it never reads file
contents, never touches the network, and never inspects credentials, prompts,
or provider responses.  Only the failure metadata the operating system already
provides (exception class, ``winerror``, ``errno``, ``strerror``) and the
caller-supplied local path appear in any message.

Categories are intentionally private — used only for messages and tests.  They
do not change CLI exit-code contracts.
"""

from __future__ import annotations

import errno
from pathlib import Path

# Windows sharing/lock violation codes: ERROR_SHARING_VIOLATION (32) and
# ERROR_LOCK_VIOLATION (33).  These are the only transient-lock codes the
# bounded atomic-replace retry will act on.
WINDOWS_TRANSIENT_LOCK_ERRORS: frozenset[int] = frozenset({32, 33})

# Stable private failure categories (messages/tests only).
CATEGORY_LOCKED = "locked"
CATEGORY_PERMISSION = "permission"
CATEGORY_MISSING_PARENT = "missing_parent"
CATEGORY_IS_DIRECTORY = "is_directory"
CATEGORY_NO_SPACE = "no_space"
CATEGORY_READ_ONLY = "read_only"
CATEGORY_IO = "io"
CATEGORY_SERIALIZATION = "serialization"

# Short human phrase per category, used as the lead reason in messages.
_CATEGORY_REASON: dict[str, str] = {
    CATEGORY_LOCKED: "file is temporarily locked",
    CATEGORY_PERMISSION: "permission denied",
    CATEGORY_MISSING_PARENT: "the parent directory is missing or invalid",
    CATEGORY_IS_DIRECTORY: "the target is a directory, not a file",
    CATEGORY_NO_SPACE: "no space left on device or quota exceeded",
    CATEGORY_READ_ONLY: "the filesystem or media is read-only",
    CATEGORY_IO: "an I/O error occurred",
    CATEGORY_SERIALIZATION: "the content could not be serialized",
}

# ``EDQUOT`` is not defined on every platform/Python build.
_ENOSPC_CODES = {errno.ENOSPC}
_EDQUOT = getattr(errno, "EDQUOT", None)
if _EDQUOT is not None:
    _ENOSPC_CODES.add(_EDQUOT)


def find_os_error(exc: BaseException | None) -> OSError | None:
    """Return the nearest :class:`OSError` in the cause/context chain, or ``None``.

    Walks ``__cause__`` then ``__context__`` defensively (guarding against
    cycles) so a wrapped failure still surfaces the underlying OS reason.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError):
            return current
        current = current.__cause__ or current.__context__
    return None


def classify_os_error(exc: BaseException | None) -> str:
    """Classify *exc* (or its nearest underlying OS error) into a stable category.

    A non-OSError serialization fault (``TypeError`` / ``ValueError`` /
    ``UnicodeError`` with no OSError in the chain) is ``serialization``; any
    other non-OSError is ``io``.
    """
    os_err = find_os_error(exc)
    if os_err is None:
        if isinstance(exc, (TypeError, ValueError, UnicodeError)):
            return CATEGORY_SERIALIZATION
        return CATEGORY_IO

    # Windows raises a sharing violation as PermissionError with winerror 32/33,
    # so the lock check must precede the generic permission check.
    winerror = getattr(os_err, "winerror", None)
    if winerror in WINDOWS_TRANSIENT_LOCK_ERRORS:
        return CATEGORY_LOCKED

    eno = os_err.errno
    if isinstance(os_err, IsADirectoryError) or eno == errno.EISDIR:
        return CATEGORY_IS_DIRECTORY
    if eno in _ENOSPC_CODES:
        return CATEGORY_NO_SPACE
    if eno == errno.EROFS:
        return CATEGORY_READ_ONLY
    if isinstance(os_err, PermissionError) or eno in (errno.EACCES, errno.EPERM):
        return CATEGORY_PERMISSION
    if isinstance(os_err, FileNotFoundError) or isinstance(os_err, NotADirectoryError):
        return CATEGORY_MISSING_PARENT
    if eno in (errno.ENOENT, errno.ENOTDIR):
        return CATEGORY_MISSING_PARENT
    return CATEGORY_IO


def is_transient_lock(exc: BaseException | None) -> bool:
    """True only for a Windows sharing/lock violation (winerror 32 or 33)."""
    return classify_os_error(exc) == CATEGORY_LOCKED


def describe_cause(exc: BaseException | None) -> str:
    """Return a concise parenthetical-free cause fragment for *exc*.

    Examples: ``"PermissionError, WinError 32"`` or ``"OSError, errno 28: no
    space left on device"``.  Only OS-provided metadata appears — never a path,
    source content, prompt, or provider response.  Returns ``""`` when there is
    nothing useful to add.
    """
    os_err = find_os_error(exc)
    if os_err is None:
        return type(exc).__name__ if exc is not None else ""
    parts: list[str] = [type(os_err).__name__]
    winerror = getattr(os_err, "winerror", None)
    if winerror:
        parts.append(f"WinError {winerror}")
    if os_err.errno:
        parts.append(f"errno {os_err.errno}")
    head = ", ".join(parts)
    reason = (os_err.strerror or "").strip()
    if reason:
        return f"{head}: {reason}"
    return head


def category_reason(category: str) -> str:
    """Return the short human phrase for a private *category*."""
    return _CATEGORY_REASON.get(category, _CATEGORY_REASON[CATEGORY_IO])


def diagnose(exc: BaseException | None) -> tuple[str, str]:
    """Return ``(category, cause_fragment)`` for *exc*."""
    return classify_os_error(exc), describe_cause(exc)


def format_local_io_error(
    lead: str,
    path: str | Path,
    exc: BaseException | None,
    action: str = "",
) -> str:
    """Compose a sanitized, actionable local I/O error message.

    Layout::

        <lead> '<path>': <reason> (<cause fragment>).
        <action>

    *lead* names the operation (e.g. ``"Could not update crash recovery file"``).
    *action* is an optional second line of guidance.  The category reason and
    cause fragment are derived only from OS metadata; only the local *path* and
    these reasons appear.
    """
    category = classify_os_error(exc)
    reason = category_reason(category)
    cause = describe_cause(exc)
    detail = f"{reason} ({cause})" if cause else reason
    message = f"{lead} '{path}': {detail}."
    if action:
        message = f"{message}\n{action}"
    return message
