"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from codedoc.utils.errors import UnrecoverableProviderError
from tests.support.execution_requests import make_execution_requests
from tests.support.response_correction_cases import RoutingProvider
import time
import codedoc.core.execution as ex
from codedoc.core.safe_writer import SafeWriter
from codedoc.utils.errors import (
    ErrorReporter,
    LLMError,
)
from tests.support.provider_failures import provider_failure_error

def test_parallel_contract_failures_do_not_enter_sequential_retry(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "parallel_agents": True,
            "max_parallel_files": 2,
            "file_retry_attempts": 2,
            "response_correction_enabled": False,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["failed"] == 2
    assert stats["attempted_calls"] == 2
    assert stats["response_contract_failures"] == 2

TERMINAL_BILLING = provider_failure_error("openai", "provider-quota-exhausted", status=429)

GLOBAL_PERMANENT = provider_failure_error("openai", "provider-authentication-rejected", status=401)

INPUT_PERMANENT = provider_failure_error("anthropic", "provider-input-rejected", status=400)

class _LLM:
    def __init__(self, name="openai"):
        self.provider_name = name

class _RaisingOrch:
    """Orchestrator whose ``process`` raises a fixed exception and counts calls."""

    def __init__(self, exc, name="openai"):
        self.llm = _LLM(name)
        self._exc = exc
        self.calls = 0

    def process(self, request):
        self.calls += 1
        raise self._exc

class _FakeQueue:
    def __init__(self):
        self.checked: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def mark_checked(self, rel_path):
        self.checked.append(rel_path)

    def mark_failed(self, rel_path, reason):
        self.failed.append((rel_path, reason))

def _requests(tmp_path, count):
    return make_execution_requests(tmp_path, [f"f{i}.py" for i in range(count)])

def _terminal_provider():
    class _Provider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            # Route (a): the phrase is folded into the agent error message.
            raise provider_failure_error("openai", "provider-quota-exhausted", status=429)

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return _Provider()

@pytest.mark.parametrize(
    "exc",
    [TERMINAL_BILLING, GLOBAL_PERMANENT],
    ids=["terminal_billing", "global_permanent"],
)
def test_sequential_abort_without_retry(tmp_path, exc):
    orch = _RaisingOrch(exc)
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        ex._process_files_sequentially(
            _requests(tmp_path, 3),
            orch,
            queue,
            stats,
            reporter,
            retry_attempts=3,  # would retry 4× per file if not aborted
            max_consecutive_failures=5,
            new_results={},
            recorder=SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {}),
            split_execution_mode="recovery",
        )

    assert excinfo.value.category == "terminal"
    # Aborted on the FIRST file's first failure: exactly one provider call, no
    # per-file retries, and never demoted to an ordinary failure.
    assert orch.calls == 1
    assert stats["failed"] == 0
    assert queue.failed == []

def test_sequential_input_permanent_marks_failed_without_retry_and_continues(tmp_path):
    requests = _requests(tmp_path, 2)

    class _Orch:
        def __init__(self):
            self.llm = _LLM("anthropic")
            self.calls = 0

        def process(self, request):
            self.calls += 1
            if request.rel_path == "f0.py":
                raise INPUT_PERMANENT
            return {"file_path": request.rel_path, "language": "python"}

    orch = _Orch()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()
    new_results: dict = {}

    outcome = ex._process_files_sequentially(
        requests,
        orch,
        queue,
        stats,
        reporter,
        retry_attempts=3,  # input file must NOT consume these
        max_consecutive_failures=5,
        new_results=new_results,
        recorder=SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {}),
        split_execution_mode="recovery",
    )

    # f0 failed with exactly one attempt (no retry); f1 succeeded; run continued.
    assert stats["failed"] == 1
    assert stats["checked"] == 1
    assert [rel for rel, _reason in queue.failed] == ["f0.py"]
    assert "f1.py" in queue.checked
    assert orch.calls == 2  # f0 once + f1 once
    # The input failure is non-rate-limit, so the pass is not zero-progress.
    assert outcome.all_failures_rate_limited is False

def test_sequential_transient_error_still_retries_and_succeeds(tmp_path):
    class _FlakyOrch:
        def __init__(self):
            self.llm = _LLM("openai")
            self.calls = 0

        def process(self, request):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("openai", "temporary provider outage")
            return {"file_path": request.rel_path, "language": "python"}

    orch = _FlakyOrch()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()

    ex._process_files_sequentially(
        _requests(tmp_path, 1),
        orch,
        queue,
        stats,
        reporter,
        retry_attempts=1,
        max_consecutive_failures=5,
        new_results={},
        recorder=SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {}),
        split_execution_mode="recovery",
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert orch.calls == 2  # failed once, retried, succeeded

@pytest.mark.parametrize(
    "exc",
    [TERMINAL_BILLING, GLOBAL_PERMANENT],
    ids=["terminal_billing", "global_permanent"],
)
def test_parallel_abort_without_retry(tmp_path, exc):
    orch = _RaisingOrch(exc)
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()
    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        ex._process_descriptor_batch(
            _requests(tmp_path, 4),
            orch,
            queue,
            stats,
            reporter,
            max_workers=2,
            recorder=recorder,
            split_execution_mode="recovery",
        )

    assert excinfo.value.category == "terminal"
    # Never reclassified as a rate-limit or ordinary failure.
    assert queue.failed == []
    assert stats["failed"] == 0
    assert stats["checked"] == 0

def test_parallel_abort_cancels_pending_work(tmp_path):
    """The first terminal failure cancels queued descriptors rather than starting
    them all (mirrors the fatal-persistence cancellation contract)."""

    class _SlowAfterFirst(_RaisingOrch):
        def process(self, request):
            self.calls += 1
            if self.calls > 1:
                time.sleep(0.1)  # would-be later work; should be cancelled
            raise self._exc

    orch = _SlowAfterFirst(TERMINAL_BILLING)
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()
    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

    with pytest.raises(UnrecoverableProviderError):
        ex._process_descriptor_batch(
            _requests(tmp_path, 6),
            orch,
            queue,
            stats,
            reporter,
            max_workers=1,
            recorder=recorder,
            split_execution_mode="recovery",
        )

    # At most one extra task may already be running when the abort is observed;
    # the remaining queued requests must be cancelled, not all started.
    assert orch.calls <= 2

def test_parallel_input_permanent_is_recorded_without_sequential_retry(tmp_path):
    requests = _requests(tmp_path, 2)

    class _Orch:
        def __init__(self):
            self.llm = _LLM("anthropic")
            self.calls: dict[str, int] = {}

        def process(self, request):
            rel_path = request.rel_path
            self.calls[rel_path] = self.calls.get(rel_path, 0) + 1
            if rel_path == "f0.py":
                raise INPUT_PERMANENT
            return {"file_path": rel_path, "language": "python"}

    orch = _Orch()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter()
    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

    succeeded, rate_limited, retryable = ex._process_descriptor_batch(
        requests,
        orch,
        queue,
        stats,
        reporter,
        max_workers=2,
        recorder=recorder,
        split_execution_mode="recovery",
    )

    assert set(succeeded) == {"f1.py"}
    assert rate_limited == []
    assert retryable == []
    assert orch.calls == {"f0.py": 1, "f1.py": 1}
    assert stats == {"checked": 1, "failed": 1}
    assert [rel for rel, _reason in queue.failed] == ["f0.py"]

def test_pipeline_records_and_flushes_abort_and_preserves_backup(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "create_provider", lambda _cfg: _terminal_provider())

    wrote = {"called": False}
    real_write = pipeline.write_project_outputs
    monkeypatch.setattr(
        pipeline,
        "write_project_outputs",
        lambda *a, **k: wrote.__setitem__("called", True) or real_write(*a, **k),
    )

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        pipeline.run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "max_parallel_files": 1,
                "file_retry_attempts": 2,
                "propagate_changes": False,
            },
        )

    assert excinfo.value.category == "terminal"

    out_dir = tmp_path / "docs"
    # Diagnostics are terminal/in-memory only; no persistent issue log is written.
    assert not (out_dir / "error.log").exists()

    # The final output write was never reached, so the dedicated recovery file is
    # left intact and resumable (still the in-progress crash-safety file, not
    # deleted or overwritten with a "complete" final output).  0.9.8: the stable
    # output (docs/codedoc.json) was never created.
    assert wrote["called"] is False
    assert not (out_dir / "codedoc.json").exists()
    backup = out_dir / "crash_recovery.json"
    assert backup.exists()
    data = json.loads(backup.read_text(encoding="utf-8"))
    assert "_crash_safety" in data or data.get("_codedoc", {}).get("status") == "in_progress"
