"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import time
import pytest
import codedoc.core.execution as ex
import codedoc.core.safe_writer as safe_writer_mod
from codedoc.core.safe_writer import SafeWriter
from codedoc.utils.errors import (
    ErrorReporter,
    InsufficientSourceError,
    LiveBackupWriteError,
    OutputError,
)
import errno
from pathlib import Path
from codedoc.core.output import preflight_output_accessibility
from tests.support.execution_requests import make_execution_requests
from tests.support.io_failures import _oserror
from codedoc.pipeline import run_pipeline
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _first_run
from tests.support.cross_format_runs import _forbid_provider
from tests.support.cross_format_runs import _write_compatible_md_recovery
from tests.support.recovery_runs import _run
from tests.support.recovery_runs import _write_codedoc_json

def _writer(tmp_path):
    return SafeWriter(tmp_path / "codedoc.json", "json", "a.py", {})

def _result(path="a.py"):
    return {"file_path": path, "language": "python", "description": "d"}

class _FakeLLM:
    provider_name = "fake"

class _FakeOrchestrator:
    llm = _FakeLLM()

    def process(self, request):
        return {"file_path": request.rel_path, "language": "python"}

class _FakeQueue:
    def __init__(self):
        self.checked: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def mark_checked(self, rel_path):
        self.checked.append(rel_path)

    def mark_failed(self, rel_path, reason):
        self.failed.append((rel_path, reason))

    def mark_skipped_insufficient_source(self, rel_path, reason):
        self.skipped.append((rel_path, reason))

class _CountingFailRecorder:
    """Recorder whose every record() fails fatally; counts attempts."""

    def __init__(self):
        self.record_calls = 0

    def record(self, rel_path, result, file_hash=""):
        self.record_calls += 1
        raise LiveBackupWriteError("codedoc.json", "could not persist live backup")

    def recorded_this_run(self, rel_path):
        return False

    def get_record(self, rel_path):
        return None

    def has_record(self, rel_path):
        return False

class _DiscardFailRecorder:
    """Recorder whose discard fails before a skip can be committed."""

    def __init__(self):
        self.discard_calls = 0

    def discard(self, rel_path):
        self.discard_calls += 1
        raise LiveBackupWriteError("codedoc.json", "could not persist discard")

    def record(self, rel_path, result, file_hash=""):
        raise AssertionError("insufficient source must not be recorded")

    def recorded_this_run(self, rel_path):
        return False

    def get_record(self, rel_path):
        return None

def _requests(tmp_path, count):
    return make_execution_requests(tmp_path, [f"f{i}.py" for i in range(count)])

def test_live_backup_write_error_is_output_error():
    assert issubclass(LiveBackupWriteError, OutputError)

def test_initialize_empty_raises_on_write_failure(tmp_path, monkeypatch):
    sw = _writer(tmp_path)

    def boom(path, text):
        raise OSError("no space")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", boom)
    with pytest.raises(LiveBackupWriteError) as excinfo:
        sw.initialize_empty()
    # Carries the target path, retains the original cause, leaks no source data.
    message = str(excinfo.value)
    assert "codedoc.json" in message
    assert "persisted before this failed write remains preserved" in message
    assert "result being written is not guaranteed saved" in message
    assert "completed ordinary files can resume" in message
    assert "completed fresh-split files are deliberately re-documented" in message
    assert isinstance(excinfo.value.__cause__, OSError)

def test_pipeline_initialization_failure_creates_no_provider(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    provider_calls = []

    def fail_write(_path, _text):
        raise OSError("no space")

    def create_provider(_config):
        provider_calls.append(True)
        raise AssertionError("provider creation must not be reached")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", fail_write)
    monkeypatch.setattr("codedoc.pipeline.create_provider", create_provider)

    with pytest.raises(LiveBackupWriteError):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "max_parallel_files": 1,
                "propagate_changes": False,
            },
        )

    assert provider_calls == []

def test_record_failure_rolls_back_all_markers(tmp_path, monkeypatch):
    sw = _writer(tmp_path)
    sw.initialize_empty()  # succeeds, empty backup on disk

    def boom(path, text):
        raise OSError("no space")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", boom)
    with pytest.raises(LiveBackupWriteError):
        sw.record("a.py", _result("a.py"), file_hash="h")

    # After a failed record(), all in-memory persistence markers expose exactly
    # the pre-call state.
    assert sw.size == 0
    assert sw.has_record("a.py") is False
    assert sw.recorded_this_run("a.py") is False
    assert sw.get_record("a.py") is None

def test_serialization_failure_rolls_back_all_markers(tmp_path, monkeypatch):
    sw = _writer(tmp_path)
    sw.initialize_empty()

    def boom(*args, **kwargs):
        raise TypeError("not JSON serializable")

    monkeypatch.setattr(safe_writer_mod.json, "dumps", boom)
    with pytest.raises(LiveBackupWriteError) as excinfo:
        sw.record("a.py", _result("a.py"), file_hash="h")

    assert isinstance(excinfo.value.__cause__, TypeError)
    assert sw.size == 0
    assert sw.has_record("a.py") is False
    assert sw.recorded_this_run("a.py") is False
    assert sw.get_record("a.py") is None

def test_failed_record_preserves_previously_persisted_records(tmp_path, monkeypatch):
    sw = _writer(tmp_path)
    sw.initialize_empty()
    sw.record("a.py", _result("a.py"), file_hash="h1")  # succeeds

    def boom(path, text):
        raise OSError("no space")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", boom)
    with pytest.raises(LiveBackupWriteError):
        sw.record("b.py", _result("b.py"), file_hash="h2")

    # The already-persisted record survives for resume; the failed one does not.
    assert sw.size == 1
    assert sw.has_record("a.py") is True
    assert sw.has_record("b.py") is False
    on_disk = (tmp_path / "codedoc.json").read_text(encoding="utf-8")
    assert "a.py" in on_disk
    assert "b.py" not in on_disk

def test_discard_removes_record_and_recorded_marker(tmp_path):
    sw = _writer(tmp_path)
    sw.initialize_empty()
    sw.record("a.py", _result("a.py"), file_hash="h")

    sw.discard("a.py")

    assert sw.size == 0
    assert sw.has_record("a.py") is False
    assert sw.recorded_this_run("a.py") is False
    payload = json.loads((tmp_path / "codedoc.json").read_text(encoding="utf-8"))
    assert payload["files"] == []

def test_discard_absent_record_does_not_flush(tmp_path, monkeypatch):
    sw = _writer(tmp_path)

    def unexpected_flush():
        raise AssertionError("absent discard must not flush")

    monkeypatch.setattr(sw, "_flush_locked", unexpected_flush)
    sw.discard("missing.py")

def test_discard_failure_restores_record_and_marker(tmp_path, monkeypatch):
    sw = _writer(tmp_path)
    sw.initialize_empty()
    sw.record("a.py", _result("a.py"), file_hash="h")

    def boom(path, text):
        raise OSError("no space")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", boom)
    with pytest.raises(LiveBackupWriteError):
        sw.discard("a.py")

    assert sw.size == 1
    assert sw.has_record("a.py") is True
    assert sw.recorded_this_run("a.py") is True
    assert sw.get_record("a.py") is not None

def test_sequential_persistence_failure_is_fatal_without_retry(tmp_path, monkeypatch):
    recorder = _CountingFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(LiveBackupWriteError):
        ex._process_files_sequentially(
            _requests(tmp_path, 1),
            _FakeOrchestrator(),
            queue,
            stats,
            reporter,
            retry_attempts=3,
            max_consecutive_failures=5,
            new_results={},
            recorder=recorder,
        )

    # Recorded exactly once — never retried — and never demoted to a per-file
    # failure in the stats or the queue.
    assert recorder.record_calls == 1
    assert stats["failed"] == 0
    assert queue.failed == []

def test_parallel_persistence_failure_is_fatal_without_retry(tmp_path, monkeypatch):
    recorder = _CountingFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(LiveBackupWriteError):
        ex._process_descriptor_batch(
            _requests(tmp_path, 4),
            _FakeOrchestrator(),
            queue,
            stats,
            reporter,
            max_workers=4,
            recorder=recorder,
        )

    # No file was reclassified as a rate-limit or ordinary failure.
    assert queue.failed == []
    assert stats["failed"] == 0

def test_parallel_persistence_failure_cancels_work_not_yet_started(
    tmp_path,
    monkeypatch,
):
    class SlowAfterFirstOrchestrator(_FakeOrchestrator):
        def __init__(self):
            self.process_calls = 0

        def process(self, request):
            self.process_calls += 1
            if self.process_calls > 1:
                time.sleep(0.1)
            return super().process(request)

    orchestrator = SlowAfterFirstOrchestrator()
    recorder = _CountingFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(LiveBackupWriteError):
        ex._process_descriptor_batch(
            _requests(tmp_path, 6),
            orchestrator,
            queue,
            stats,
            reporter,
            max_workers=1,
            recorder=recorder,
        )

    # One additional task may already be running when the first failed future is
    # observed, but queued work must be cancelled rather than all being started.
    assert orchestrator.process_calls <= 2
    assert recorder.record_calls <= 2

def test_sequential_skip_discard_failure_is_fatal_before_skip_accounting(
    tmp_path, monkeypatch
):
    request = _requests(tmp_path, 1)[0]
    monkeypatch.setattr(
        ex,
        "_process_one_file",
        lambda _request, _orchestrator: (_ for _ in ()).throw(
            InsufficientSourceError(
                request.rel_path, "empty_or_whitespace_only"
            )
        ),
    )
    recorder = _DiscardFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0, "skipped_insufficient_source": 0}

    with pytest.raises(LiveBackupWriteError):
        ex._process_files_sequentially(
            [request],
            _FakeOrchestrator(),
            queue,
            stats,
            ErrorReporter(),
            retry_attempts=3,
            max_consecutive_failures=5,
            new_results={},
            recorder=recorder,
        )

    assert recorder.discard_calls == 1
    assert stats["skipped_insufficient_source"] == 0
    assert queue.skipped == []
    assert queue.failed == []

def test_parallel_skip_discard_failure_uses_fatal_abort_protocol(
    tmp_path, monkeypatch
):
    request = _requests(tmp_path, 1)[0]
    monkeypatch.setattr(
        ex,
        "_process_one_file",
        lambda _request, _orchestrator: (_ for _ in ()).throw(
            InsufficientSourceError(
                request.rel_path, "empty_or_whitespace_only"
            )
        ),
    )
    recorder = _DiscardFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0, "skipped_insufficient_source": 0}

    with pytest.raises(LiveBackupWriteError):
        ex._process_descriptor_batch(
            [request],
            _FakeOrchestrator(),
            queue,
            stats,
            ErrorReporter(),
            max_workers=1,
            recorder=recorder,
        )

    assert recorder.discard_calls == 1
    assert stats["skipped_insufficient_source"] == 0
    assert queue.skipped == []
    assert queue.failed == []

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


def test_zero_source_directory_creation_failure_is_classified(tmp_path, monkeypatch):
    import codedoc.core.output as output_mod

    out = tmp_path / "docs"
    original_mkdir = output_mod.Path.mkdir

    def guarded_mkdir(path, *args, **kwargs):
        if path == out:
            raise _oserror(PermissionError, errno_=errno.EACCES)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(output_mod.Path, "mkdir", guarded_mkdir)

    with pytest.raises(OutputError) as excinfo:
        run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "output_dir": "docs",
            },
        )

    assert "No provider was contacted" in str(excinfo.value)
    assert excinfo.value.file_path == str(out)
    assert not out.exists()


def test_zero_call_conversion_write_failure_preserves_sibling_and_recovery(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    stable = tmp_path / "docs" / "codedoc.json"
    before = stable.read_bytes()
    recovery = _write_compatible_md_recovery(tmp_path, "newer recovery record")
    _forbid_provider(monkeypatch)

    import codedoc.pipeline as pipeline_mod

    def fail_write(*_args, **_kwargs):
        raise OutputError(str(tmp_path / "docs" / "codedoc.md"), "forced failure")

    monkeypatch.setattr(pipeline_mod, "write_project_outputs", fail_write)
    with pytest.raises(OutputError, match="forced failure"):
        run_pipeline(tmp_path, _config("md"))
    assert stable.read_bytes() == before
    assert recovery.exists()
    assert not (tmp_path / "docs" / "codedoc.md").exists()

def test_forced_stable_write_failure_preserves_recovery_and_prior_stable(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    stable = out / "codedoc.json"
    recovery = out / "crash_recovery.json"

    _write_codedoc_json(stable, [{"path": "old.py", "hash": "OLD", "language": "python"}])
    stable_before = stable.read_bytes()

    import codedoc.pipeline as pipeline_mod

    def boom(*a, **k):
        raise OutputError(str(stable), "forced stable-write failure")

    monkeypatch.setattr(pipeline_mod, "write_project_outputs", boom)

    with pytest.raises(OutputError):
        _run(tmp_path, monkeypatch, entry_file="main.py")

    # The stable output is left exactly as it was; the recovery file survives.
    assert stable.read_bytes() == stable_before
    assert recovery.exists()
    rec = json.loads(recovery.read_text(encoding="utf-8"))
    assert rec.get("_codedoc", {}).get("status") == "in_progress"

def test_forced_recovery_deletion_oserror_raises_outputerror_and_keeps_both(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    stable = out / "codedoc.json"
    recovery = out / "crash_recovery.json"

    real_unlink = Path.unlink

    def guarded_unlink(self, *args, **kwargs):
        if self.name == "crash_recovery.json":
            raise OSError("forced recovery deletion failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)

    with pytest.raises(OutputError) as excinfo:
        _run(tmp_path, monkeypatch, entry_file="main.py")

    # The completed stable output remains; the recovery file is preserved; the
    # error names the recovery path.
    assert stable.exists()
    completed = json.loads(stable.read_text(encoding="utf-8"))
    assert "_crash_safety" not in completed
    assert recovery.exists()
    assert "crash_recovery.json" in str(excinfo.value)
