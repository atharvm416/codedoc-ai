"""Tests organized by feature ownership."""

from __future__ import annotations

import errno
from codedoc.core.io_diagnostics import (
    CATEGORY_IO,
    CATEGORY_IS_DIRECTORY,
    CATEGORY_LOCKED,
    CATEGORY_NO_SPACE,
    CATEGORY_PERMISSION,
    CATEGORY_READ_ONLY,
    CATEGORY_SERIALIZATION,
    classify_os_error,
    describe_cause,
    format_local_io_error,
    is_transient_lock,
)
from tests.support.io_failures import _oserror

def test_winerror_lock_codes_classify_as_locked():
    assert classify_os_error(_oserror(PermissionError, winerror=32)) == CATEGORY_LOCKED
    assert classify_os_error(_oserror(PermissionError, winerror=33)) == CATEGORY_LOCKED
    assert is_transient_lock(_oserror(PermissionError, winerror=32))

def test_plain_permission_is_not_a_transient_lock():
    exc = _oserror(PermissionError, errno_=errno.EACCES)
    assert classify_os_error(exc) == CATEGORY_PERMISSION
    assert not is_transient_lock(exc)

def test_distinct_categories():
    assert classify_os_error(_oserror(errno_=errno.ENOSPC)) == CATEGORY_NO_SPACE
    assert classify_os_error(_oserror(errno_=errno.EROFS)) == CATEGORY_READ_ONLY
    assert classify_os_error(_oserror(IsADirectoryError, errno_=errno.EISDIR)) == CATEGORY_IS_DIRECTORY
    assert classify_os_error(_oserror(errno_=errno.EIO)) == CATEGORY_IO
    # A serialization fault has no OSError in the chain.
    assert classify_os_error(TypeError("not serializable")) == CATEGORY_SERIALIZATION

def test_describe_cause_includes_metadata_but_no_path():
    exc = _oserror(PermissionError, errno_=13, winerror=32, strerror="being used")
    cause = describe_cause(exc)
    assert "PermissionError" in cause
    assert "WinError 32" in cause
    assert "errno 13" in cause
    assert "being used" in cause

def test_format_local_io_error_includes_path_and_action_only():
    msg = format_local_io_error(
        "Cannot write output directory",
        "/out/docs",
        _oserror(PermissionError, errno_=13),
        action="No provider was contacted.",
    )
    assert "/out/docs" in msg
    assert "permission denied" in msg
    assert "No provider was contacted." in msg

def test_nearest_oserror_is_found_through_cause_chain():
    root = _oserror(PermissionError, winerror=32)
    try:
        try:
            raise root
        except OSError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as wrapped:
        assert classify_os_error(wrapped) == CATEGORY_LOCKED
