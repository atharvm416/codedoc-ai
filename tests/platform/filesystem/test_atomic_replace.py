"""Tests organized by feature ownership."""

from __future__ import annotations

import errno
from pathlib import Path
import pytest
from codedoc.core.block_manager import ATOMIC_REPLACE_RETRY_DELAYS_S, atomic_write_text
from codedoc.core.io_diagnostics import CATEGORY_PERMISSION, classify_os_error
from tests.support.clocks import capture_sleeps
from tests.support.io_failures import _oserror

pytestmark = pytest.mark.platform

def _tmp_files(directory: Path) -> list[Path]:
    return list(directory.glob(".*.tmp"))

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
    return capture_sleeps(monkeypatch, "codedoc.core.block_manager.time.sleep")

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

def test_winerror_5_succeeds_after_one_retry(tmp_path, monkeypatch):
    """A repeated-replacement stress probe (SafeWriter checkpointing several
    hundred sequential split nodes onto the same crash-recovery path — the
    exact pattern a large split file's leaf/reduction nodes produce)
    reproducibly hit ERROR_ACCESS_DENIED a handful of times on an otherwise
    healthy, writable, exclusively-owned directory, and every occurrence
    succeeded on an immediate retry. WinError 5 is therefore retryable at this
    atomic-replacement boundary, while remaining a permission diagnostic if
    the bounded retry budget is exhausted."""
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=1, winerror=5)
    target = tmp_path / "out.txt"
    atomic_write_text(target, "ok")
    assert target.read_text(encoding="utf-8") == "ok"
    assert sleeps == [ATOMIC_REPLACE_RETRY_DELAYS_S[0]]
    assert _tmp_files(tmp_path) == []

def test_winerror_5_succeeds_on_final_bounded_retry(tmp_path, monkeypatch):
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=len(ATOMIC_REPLACE_RETRY_DELAYS_S), winerror=5)
    target = tmp_path / "out.txt"
    atomic_write_text(target, "ok")
    assert target.read_text(encoding="utf-8") == "ok"
    assert sleeps == list(ATOMIC_REPLACE_RETRY_DELAYS_S)
    assert _tmp_files(tmp_path) == []

def test_persistent_winerror_5_still_raises_after_exhausting_bounded_retries(
    tmp_path, monkeypatch
):
    """A genuinely persistent access-denied condition (not a transient
    background-reader collision) is never retried indefinitely: it still
    raises, with its cause intact, once the same fixed retry budget used for
    32/33 is exhausted — exactly the existing winerror-32 guarantee, now
    proven for winerror 5 too."""
    target = tmp_path / "out.txt"
    target.write_text("OLD", encoding="utf-8")
    sleeps = _patch_sleep(monkeypatch)
    _patch_replace(monkeypatch, fail_times=99, winerror=5)
    with pytest.raises(OSError) as excinfo:
        atomic_write_text(target, "NEW")
    assert getattr(excinfo.value, "winerror", None) == 5  # original cause intact
    assert classify_os_error(excinfo.value) == CATEGORY_PERMISSION
    assert sleeps == list(ATOMIC_REPLACE_RETRY_DELAYS_S)  # bounded, never indefinite
    assert target.read_text(encoding="utf-8") == "OLD"  # prior target preserved
    assert _tmp_files(tmp_path) == []  # temp cleaned up

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
