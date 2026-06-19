"""0.9.7 — Workstreams B & D: fail-fast routing and operator-facing reporting.

All via injected fakes — no network or credentials.

Workstream B (fail-fast routing):
- terminal-billing and global-permanent errors abort via
  ``UnrecoverableProviderError`` in both sequential and parallel modes, after a
  single file's first failure, with no per-file retries;
- pending parallel work is cancelled on abort;
- an input-permanent error marks exactly that file failed without retrying it,
  and the run continues;
- a genuinely transient error still retries and can succeed.

Workstream D (reporting):
- the pipeline records + flushes the abort to error.log before re-raising and
  does NOT reach write_project_outputs / recorder.delete (live backup intact);
- the CLI exits 2 for a terminal abort and 1 for the bounded rate-limit stop,
  prints a resume hint, and does not print ``Fatal error:``.
"""

from __future__ import annotations

import json
import time

import pytest

import codedoc.core.execution as ex
from codedoc.core.safe_writer import SafeWriter
from codedoc.utils.errors import (
    ErrorReporter,
    LLMError,
    UnrecoverableProviderError,
)

TERMINAL_BILLING = LLMError("openai", "Your credit balance is too low to continue")
GLOBAL_PERMANENT = LLMError("openai", "Incorrect API key provided")
INPUT_PERMANENT = LLMError("anthropic", "prompt is too long: 250000 tokens > 200000")


class _LLM:
    def __init__(self, name="openai"):
        self.provider_name = name


class _RaisingOrch:
    """Orchestrator whose ``process`` raises a fixed exception and counts calls."""

    def __init__(self, exc, name="openai"):
        self.llm = _LLM(name)
        self._exc = exc
        self.calls = 0

    def process(self, descriptor, content, imports):
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


def _descriptors(tmp_path, count):
    descriptors = []
    for i in range(count):
        f = tmp_path / f"f{i}.py"
        f.write_text("x = 1\n", encoding="utf-8")
        descriptors.append(
            {"rel_path": f"f{i}.py", "path": f, "language": "python", "extension": ".py"}
        )
    return descriptors


@pytest.fixture(autouse=True)
def _no_parse(monkeypatch):
    monkeypatch.setattr(ex, "parse_file", lambda descriptor: [])


# ---------------------------------------------------------------------------
# Workstream B — sequential path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [TERMINAL_BILLING, GLOBAL_PERMANENT],
    ids=["terminal_billing", "global_permanent"],
)
def test_sequential_abort_without_retry(tmp_path, exc):
    orch = _RaisingOrch(exc)
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter(tmp_path / "error.log")

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        ex._process_files_sequentially(
            _descriptors(tmp_path, 3),
            orch,
            queue,
            stats,
            reporter,
            retry_attempts=3,  # would retry 4× per file if not aborted
            max_consecutive_failures=5,
            new_results={},
            recorder=SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {}),
        )

    assert excinfo.value.category == "terminal"
    # Aborted on the FIRST file's first failure: exactly one provider call, no
    # per-file retries, and never demoted to an ordinary failure.
    assert orch.calls == 1
    assert stats["failed"] == 0
    assert queue.failed == []


def test_sequential_input_permanent_marks_failed_without_retry_and_continues(tmp_path):
    descriptors = _descriptors(tmp_path, 2)

    class _Orch:
        def __init__(self):
            self.llm = _LLM("anthropic")
            self.calls = 0

        def process(self, descriptor, content, imports):
            self.calls += 1
            if descriptor["rel_path"] == "f0.py":
                raise INPUT_PERMANENT
            return {"file_path": descriptor["rel_path"], "language": "python"}

    orch = _Orch()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter(tmp_path / "error.log")
    new_results: dict = {}

    outcome = ex._process_files_sequentially(
        descriptors,
        orch,
        queue,
        stats,
        reporter,
        retry_attempts=3,  # input file must NOT consume these
        max_consecutive_failures=5,
        new_results=new_results,
        recorder=SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {}),
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

        def process(self, descriptor, content, imports):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("openai", "temporary provider outage")
            return {"file_path": descriptor["rel_path"], "language": "python"}

    orch = _FlakyOrch()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter(tmp_path / "error.log")

    ex._process_files_sequentially(
        _descriptors(tmp_path, 1),
        orch,
        queue,
        stats,
        reporter,
        retry_attempts=1,
        max_consecutive_failures=5,
        new_results={},
        recorder=SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {}),
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert orch.calls == 2  # failed once, retried, succeeded


# ---------------------------------------------------------------------------
# Workstream B — parallel path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [TERMINAL_BILLING, GLOBAL_PERMANENT],
    ids=["terminal_billing", "global_permanent"],
)
def test_parallel_abort_without_retry(tmp_path, exc):
    orch = _RaisingOrch(exc)
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter(tmp_path / "error.log")
    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        ex._process_descriptor_batch(
            _descriptors(tmp_path, 4),
            orch,
            queue,
            stats,
            reporter,
            max_workers=2,
            recorder=recorder,
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
        def process(self, descriptor, content, imports):
            self.calls += 1
            if self.calls > 1:
                time.sleep(0.1)  # would-be later work; should be cancelled
            raise self._exc

    orch = _SlowAfterFirst(TERMINAL_BILLING)
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter(tmp_path / "error.log")
    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

    with pytest.raises(UnrecoverableProviderError):
        ex._process_descriptor_batch(
            _descriptors(tmp_path, 6),
            orch,
            queue,
            stats,
            reporter,
            max_workers=1,
            recorder=recorder,
        )

    # At most one extra task may already be running when the abort is observed;
    # the remaining queued descriptors must be cancelled, not all started.
    assert orch.calls <= 2


def test_parallel_input_permanent_is_recorded_without_sequential_retry(tmp_path):
    descriptors = _descriptors(tmp_path, 2)

    class _Orch:
        def __init__(self):
            self.llm = _LLM("anthropic")
            self.calls: dict[str, int] = {}

        def process(self, descriptor, content, imports):
            rel_path = descriptor["rel_path"]
            self.calls[rel_path] = self.calls.get(rel_path, 0) + 1
            if rel_path == "f0.py":
                raise INPUT_PERMANENT
            return {"file_path": rel_path, "language": "python"}

    orch = _Orch()
    queue = _FakeQueue()
    stats = {"checked": 0, "failed": 0}
    reporter = ErrorReporter(tmp_path / "error.log")
    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

    succeeded, rate_limited, retryable = ex._process_descriptor_batch(
        descriptors,
        orch,
        queue,
        stats,
        reporter,
        max_workers=2,
        recorder=recorder,
    )

    assert set(succeeded) == {"f1.py"}
    assert rate_limited == []
    assert retryable == []
    assert orch.calls == {"f0.py": 1, "f1.py": 1}
    assert stats == {"checked": 1, "failed": 1}
    assert [rel for rel, _reason in queue.failed] == ["f0.py"]


# ---------------------------------------------------------------------------
# Workstream D — pipeline records + flushes, keeps the live backup intact
# ---------------------------------------------------------------------------

def _terminal_provider():
    class _Provider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            # Route (a): the phrase is folded into the agent error message.
            raise LLMError("openai", "Your credit balance is too low to continue")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return _Provider()


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
    # The pipeline recorded + flushed the abort to error.log before re-raising.
    error_log = out_dir / "error.log"
    assert error_log.exists()
    log_text = error_log.read_text(encoding="utf-8")
    assert "UnrecoverableProviderError" in log_text
    assert "provider abort" in log_text

    # The final output write was never reached, so the live JSON backup is left
    # intact and resumable (still the in-progress crash-safety file, not deleted
    # or overwritten with a "complete" final output).
    assert wrote["called"] is False
    backup = out_dir / "codedoc.json"
    assert backup.exists()
    data = json.loads(backup.read_text(encoding="utf-8"))
    assert "_crash_safety" in data or data.get("_codedoc", {}).get("status") == "in_progress"


# ---------------------------------------------------------------------------
# Workstream D — CLI presentation / exit codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "category, expected_code",
    [("terminal", 2), ("rate_limit_exhausted", 1)],
)
def test_cli_exit_codes_for_unrecoverable_provider_error(
    tmp_path, monkeypatch, capsys, category, expected_code
):
    from codedoc.cli.cli import run_cli

    def fake_pipeline(*args, **kwargs):
        raise UnrecoverableProviderError("openai", "stopped: doomed run", category)

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == expected_code

    err = capsys.readouterr().err
    # A safe-stop message, NOT the generic crash fallthrough.
    assert "Fatal error:" not in err
    # Resume hint is always printed.
    assert "re-run" in err.lower()
    assert "live json backup" in err.lower()
