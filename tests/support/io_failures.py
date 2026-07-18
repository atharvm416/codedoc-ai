"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _oserror(klass=OSError, *, errno_=None, winerror=None, strerror="simulated"):
    exc = klass(strerror)
    if errno_ is not None:
        exc.errno = errno_
    if strerror is not None:
        exc.strerror = strerror
    if winerror is not None:
        # Settable on every platform; classify_os_error reads it via getattr.
        exc.winerror = winerror
    return exc
