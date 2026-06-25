"""0.10.1 — output diagnostics, Windows write resilience, and log lifecycle.

Covers Workstreams A–E: OS-error classification + sanitized diagnostics, the
bounded transient-lock retry on atomic replace, the output accessibility
preflight, and the error.log ownership / stale-cleanup lifecycle.
"""
from __future__ import annotations

import errno
import json
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
from codedoc.utils.errors import (
    LOG_OWNERSHIP_MARKER,
    ErrorReporter,
    OutputError,
    is_codedoc_owned_log,
)


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


# ---------------------------------------------------------------------------
# D. error.log ownership + lifecycle
# ---------------------------------------------------------------------------

def test_ownership_marker_recognition(tmp_path):
    new_log = tmp_path / "new.log"
    new_log.write_text(f"{LOG_OWNERSHIP_MARKER} — 1 issue(s)\n...\n", encoding="utf-8")
    assert is_codedoc_owned_log(new_log)

    legacy_log = tmp_path / "legacy.log"
    legacy_log.write_text("codedoc issue log — 2 issue(s)\n...\n", encoding="utf-8")
    assert is_codedoc_owned_log(legacy_log)

    foreign = tmp_path / "foreign.log"
    foreign.write_text("important user notes\n", encoding="utf-8")
    assert not is_codedoc_owned_log(foreign)

    assert not is_codedoc_owned_log(tmp_path / "missing.log")


def test_flush_writes_marker_and_is_atomic(tmp_path):
    log = tmp_path / "error.log"
    reporter = ErrorReporter(log)
    reporter.record(RuntimeError("boom"), context="ctx", level="error")
    assert reporter.flush() is True
    assert reporter.has_persisted_log()
    text = log.read_text(encoding="utf-8")
    assert text.startswith(LOG_OWNERSHIP_MARKER)
    assert is_codedoc_owned_log(log)
    assert list(tmp_path.glob(".*.tmp")) == []  # atomic writer leaves no temp


def test_flush_does_not_overwrite_foreign_log(tmp_path):
    log = tmp_path / "error.log"
    log.write_text("user-owned content", encoding="utf-8")
    reporter = ErrorReporter(log)
    reporter.record(RuntimeError("boom"), level="error")
    assert reporter.flush() is False
    assert not reporter.has_persisted_log()
    assert log.read_text(encoding="utf-8") == "user-owned content"


def test_issue_stats_do_not_point_at_foreign_log(tmp_path):
    from codedoc.pipeline import _set_issue_stats

    log = tmp_path / "error.log"
    log.write_text("user-owned content", encoding="utf-8")
    reporter = ErrorReporter(log)
    reporter.record(RuntimeError("boom"), level="error")
    assert reporter.flush() is False

    stats = {}
    _set_issue_stats(stats, reporter, tmp_path / "crash_recovery_codedoc.json")
    assert stats["issues_recorded"] == 1
    assert stats["error_log"] is None
    assert "foreign file already exists" in stats["issue_log_warning"]


def test_run_summary_prints_issue_log_warning_when_log_not_persisted(capsys):
    from codedoc.cli.cli import _print_run_summary

    _print_run_summary(
        {
            "checked": 1,
            "failed": 0,
            "output_dir": "codedoc",
            "issues_recorded": 1,
            "error_log": None,
            "issue_log_warning": "1 issue was recorded but no log was written.",
            "rate_limit_warnings": [],
        }
    )
    out = capsys.readouterr().out
    assert "Issue log warning:" in out
    assert "no log was written" in out


def test_clear_stale_owned_log_removes_owned_and_keeps_foreign(tmp_path):
    owned = tmp_path / "error.log"
    owned.write_text(f"{LOG_OWNERSHIP_MARKER} — 1 issue(s)\n", encoding="utf-8")
    reporter = ErrorReporter(owned)  # no entries recorded => clean run
    assert reporter.clear_stale_owned_log() is None
    assert not owned.exists()

    foreign = tmp_path / "foreign" / "error.log"
    foreign.parent.mkdir()
    foreign.write_text("keep me", encoding="utf-8")
    reporter2 = ErrorReporter(foreign)
    assert reporter2.clear_stale_owned_log() is None
    assert foreign.read_text(encoding="utf-8") == "keep me"


def test_clear_stale_owned_log_noop_when_issues_recorded(tmp_path):
    owned = tmp_path / "error.log"
    owned.write_text(f"{LOG_OWNERSHIP_MARKER} — old\n", encoding="utf-8")
    reporter = ErrorReporter(owned)
    reporter.record(RuntimeError("current"), level="error")
    # With current issues, the stale-cleanup is a no-op (flush replaces instead).
    assert reporter.clear_stale_owned_log() is None
    assert owned.exists()


# ---------------------------------------------------------------------------
# Integration: clean run clears a stale owned log; foreign survives
# ---------------------------------------------------------------------------

class _CombinedProvider:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return json.dumps({"description": "Documented.", "role_in_system": "core"})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt)


def _run_clean(tmp_path, monkeypatch):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _CombinedProvider())
    from codedoc.pipeline import run_pipeline

    return run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "parallel_agents": False, "propagate_changes": False},
    )


def test_clean_run_removes_stale_owned_error_log(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "codedoc"
    out.mkdir()
    stale = out / "error.log"
    stale.write_text("codedoc issue log — 3 issue(s)\nold failure\n", encoding="utf-8")

    stats = _run_clean(tmp_path, monkeypatch)
    assert stats["checked"] == 1
    assert not stale.exists(), "stale CodeDoc-owned error.log must be cleared on a clean run"
    assert stats.get("error_log") is None


def test_clean_run_preserves_foreign_error_log(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "codedoc"
    out.mkdir()
    foreign = out / "error.log"
    foreign.write_text("my personal notes, not codedoc's\n", encoding="utf-8")

    _run_clean(tmp_path, monkeypatch)
    assert foreign.read_text(encoding="utf-8") == "my personal notes, not codedoc's\n"
