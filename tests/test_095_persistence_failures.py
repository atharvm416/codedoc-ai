"""0.9.5 — fatal live-backup persistence semantics.

Covers:
- ``SafeWriter.record()`` rolling back every in-memory marker after a failed
  flush, while previously persisted records remain for resume;
- ``initialize_empty()`` and ``record()`` raising ``LiveBackupWriteError`` (an
  ``OutputError`` subtype) when the physical write fails;
- the execution layer treating a persistence failure as fatal: no provider
  retry, no rate-limit reclassification, immediate propagation on both the
  sequential and parallel paths.
"""
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


def _writer(tmp_path):
    return SafeWriter(tmp_path / "codedoc.json", "json", "a.py", {})


def _result(path="a.py"):
    return {"file_path": path, "language": "python", "description": "d"}


# ---------------------------------------------------------------------------
# SafeWriter failure semantics
# ---------------------------------------------------------------------------

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
    assert "codedoc.json" in str(excinfo.value)
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


# ---------------------------------------------------------------------------
# Execution layer: persistence failure is fatal (no retry, no reclassification)
# ---------------------------------------------------------------------------

class _FakeLLM:
    provider_name = "fake"


class _FakeOrchestrator:
    llm = _FakeLLM()

    def process(self, descriptor, content, imports):
        return {"file_path": descriptor["rel_path"], "language": "python"}


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


def _descriptors(tmp_path, count):
    descriptors = []
    for i in range(count):
        f = tmp_path / f"f{i}.py"
        f.write_text("x = 1\n", encoding="utf-8")
        descriptors.append({"rel_path": f"f{i}.py", "path": f, "language": "python"})
    return descriptors


def test_sequential_persistence_failure_is_fatal_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "parse_file", lambda descriptor: [])
    recorder = _CountingFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(LiveBackupWriteError):
        ex._process_files_sequentially(
            _descriptors(tmp_path, 1),
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
    monkeypatch.setattr(ex, "parse_file", lambda descriptor: [])
    recorder = _CountingFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(LiveBackupWriteError):
        ex._process_descriptor_batch(
            _descriptors(tmp_path, 4),
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
    monkeypatch.setattr(ex, "parse_file", lambda descriptor: [])

    class SlowAfterFirstOrchestrator(_FakeOrchestrator):
        def __init__(self):
            self.process_calls = 0

        def process(self, descriptor, content, imports):
            self.process_calls += 1
            if self.process_calls > 1:
                time.sleep(0.1)
            return super().process(descriptor, content, imports)

    orchestrator = SlowAfterFirstOrchestrator()
    recorder = _CountingFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(LiveBackupWriteError):
        ex._process_descriptor_batch(
            _descriptors(tmp_path, 6),
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
    descriptor = _descriptors(tmp_path, 1)[0]
    monkeypatch.setattr(
        ex,
        "_process_one_file",
        lambda _descriptor, _orchestrator: (_ for _ in ()).throw(
            InsufficientSourceError(
                descriptor["rel_path"], "empty_or_whitespace_only"
            )
        ),
    )
    recorder = _DiscardFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0, "skipped_insufficient_source": 0}

    with pytest.raises(LiveBackupWriteError):
        ex._process_files_sequentially(
            [descriptor],
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
    descriptor = _descriptors(tmp_path, 1)[0]
    monkeypatch.setattr(
        ex,
        "_process_one_file",
        lambda _descriptor, _orchestrator: (_ for _ in ()).throw(
            InsufficientSourceError(
                descriptor["rel_path"], "empty_or_whitespace_only"
            )
        ),
    )
    recorder = _DiscardFailRecorder()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0, "skipped_insufficient_source": 0}

    with pytest.raises(LiveBackupWriteError):
        ex._process_descriptor_batch(
            [descriptor],
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
