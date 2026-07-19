"""Tests organized by feature ownership."""

from __future__ import annotations

import errno
from pathlib import Path
import pytest
from codedoc.core.block_manager import ATOMIC_REPLACE_RETRY_DELAYS_S, atomic_write_text
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
