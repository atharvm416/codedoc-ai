"""0.10.1 — output diagnostics, Windows write resilience, and log lifecycle.

Covers OS-error classification + sanitized diagnostics, the bounded
transient-lock retry on atomic replace, and the output accessibility preflight.
"""
from __future__ import annotations

import errno
from pathlib import Path

import pytest

import codedoc.core.block_manager as block_manager
from codedoc.core.block_manager import ATOMIC_REPLACE_RETRY_DELAYS_S, atomic_write_text
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
from codedoc.core.output import preflight_output_accessibility
from codedoc.utils.errors import OutputError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _tmp_files(directory: Path) -> list[Path]:
    return list(directory.glob(".*.tmp"))


# ---------------------------------------------------------------------------
# A. OS-error classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# B. Bounded atomic-replace retry
# ---------------------------------------------------------------------------

def _patch_replace(monkeypatch, fail_times: int, winerror: int | None = 32,
                   errno_: int | None = None):
    real_replace = Path.replace
    state = {"n": 0}

    def fake_replace(self, target):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise _oserror(PermissionError, winerror=winerror, errno_=errno_)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fake_replace)
    return state


def _patch_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(block_manager.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_immediate_replace_success_performs_no_sleeps(tmp_path, monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=0)
    target = tmp_path / "out.txt"
    atomic_write_text(target, "data")
    assert target.read_text(encoding="utf-8") == "data"
    assert sleeps == []
    assert _tmp_files(tmp_path) == []


def test_winerror_32_succeeds_after_one_retry(tmp_path, monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=1, winerror=32)
    target = tmp_path / "out.txt"
    atomic_write_text(target, "ok")
    assert target.read_text(encoding="utf-8") == "ok"
    assert sleeps == [ATOMIC_REPLACE_RETRY_DELAYS_S[0]]
    assert _tmp_files(tmp_path) == []


def test_winerror_33_succeeds_on_final_bounded_retry(tmp_path, monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=len(ATOMIC_REPLACE_RETRY_DELAYS_S), winerror=33)
    target = tmp_path / "out.txt"
    atomic_write_text(target, "ok")
    assert target.read_text(encoding="utf-8") == "ok"
    assert sleeps == list(ATOMIC_REPLACE_RETRY_DELAYS_S)
    assert _tmp_files(tmp_path) == []


def test_exhausted_lock_retries_preserve_target_and_raise_cause(tmp_path, monkeypatch):
    target = tmp_path / "out.txt"
    target.write_text("OLD", encoding="utf-8")
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=99, winerror=32)
    with pytest.raises(OSError) as excinfo:
        atomic_write_text(target, "NEW")
    assert getattr(excinfo.value, "winerror", None) == 32  # original cause intact
    assert sleeps == list(ATOMIC_REPLACE_RETRY_DELAYS_S)  # bounded
    assert target.read_text(encoding="utf-8") == "OLD"  # prior target preserved
    assert _tmp_files(tmp_path) == []  # temp cleaned up


def test_plain_permission_error_is_not_retried(tmp_path, monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=99, winerror=None, errno_=errno.EACCES)
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "out.txt", "x")
    assert sleeps == []
    assert _tmp_files(tmp_path) == []


def test_enospc_is_not_retried(tmp_path, monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=99, winerror=None, errno_=errno.ENOSPC)
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "out.txt", "x")
    assert sleeps == []


# ---------------------------------------------------------------------------
# C. Output accessibility preflight
# ---------------------------------------------------------------------------

def test_preflight_passes_for_writable_dir_and_leaves_no_probes(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    preflight_output_accessibility(out)
    assert list(out.glob(".codedoc_preflight_*")) == []


def test_preflight_creates_absent_directory(tmp_path):
    out = tmp_path / "nested" / "docs"
    assert not out.exists()
    preflight_output_accessibility(out)
    assert out.is_dir()
    assert list(out.glob(".codedoc_preflight_*")) == []


def test_preflight_classifies_failure_and_cleans_probes(tmp_path, monkeypatch):
    import codedoc.core.output as output_mod

    out = tmp_path / "docs"
    out.mkdir()

    def boom(src, dst):
        raise _oserror(PermissionError, errno_=errno.EACCES)

    monkeypatch.setattr(output_mod.os, "replace", boom)
    with pytest.raises(OutputError) as excinfo:
        preflight_output_accessibility(out)
    assert "No provider was contacted" in str(excinfo.value)
    assert str(out) in excinfo.value.file_path
    assert list(out.glob(".codedoc_preflight_*")) == []
