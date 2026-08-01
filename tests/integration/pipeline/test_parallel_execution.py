"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_parallel_split_files_finalize_without_leaking_partial_state(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    for name in ("alpha.py", "beta.py"):
        (tmp_path / name).write_text(source, encoding="utf-8", newline="")
    provider = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 2,
            "parallel_agents": False,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    payload = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert stats["checked"] == 2
    assert stats["split_divided_files"] == 2
    assert {record["path"] for record in payload["files"]} == {
        "alpha.py",
        "beta.py",
    }
    assert all("division" not in record for record in payload["files"])
    assert all("documentation_units" not in record for record in payload["files"])
    assert all(
        record["_large_file_identity"].startswith("large-file-v2:")
        for record in payload["files"]
    )
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_pipeline_processes_files_with_bounded_parallelism(tmp_path, monkeypatch):
    import threading
    import time

    from codedoc.pipeline import run_pipeline

    for index in range(6):
        imports = ""
        if index == 0:
            imports = "".join(f"import file_{dep}\n" for dep in range(1, 6))
        (tmp_path / f"file_{index}.py").write_text(
            f"{imports}def func_{index}():\n    return {index}\n",
            encoding="utf-8",
        )

    class SlowProvider:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        @property
        def provider_name(self):
            return "SlowProvider"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                if "key_concepts" in prompt:
                    return json.dumps(
                        {
                            "description": "Documented file.",
                            "role_in_system": "Test role.",
                            "key_concepts": [],
                            "usage_example": "",
                        }
                    )
                if "dependencies_analysis" in prompt:
                    return json.dumps(
                        {
                            "dependencies_analysis": {
                                "internal": [],
                                "external": [],
                                "dependency_refs": [],
                                "catalog_updates": [],
                                "usage_notes": [],
                                "warnings": [],
                            }
                        }
                    )
                return json.dumps(
                    {
                        "description": "Structured file.",
                        "role_in_system": "Test role.",
                        "functions": [],
                        "classes": [],
                        "exports": [],
                    }
                )
            finally:
                with self.lock:
                    self.active -= 1

    provider = SlowProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "file_0.py",
            "parallel_agents": False,
            "max_parallel_files": 3,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 6
    assert stats["failed"] == 0
    assert 1 < provider.max_active <= 3
    assert (tmp_path / "docs_output" / "codedoc.json").exists()

def test_pipeline_retries_failed_file_before_marking_failed(tmp_path, monkeypatch):

    from codedoc.pipeline import run_pipeline

    source = tmp_path / "main.py"
    source.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    class FlakyProvider:
        def __init__(self):
            self.calls = 0

        @property
        def provider_name(self):
            return "FlakyProvider"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider outage")
            if "key_concepts" in prompt:
                return json.dumps(
                    {
                        "description": "Recovered file.",
                        "role_in_system": "Recovered role.",
                        "key_concepts": [],
                        "usage_example": "",
                    }
                )
            if "dependencies_analysis" in prompt:
                return json.dumps(
                    {
                        "dependencies_analysis": {
                            "internal": [],
                            "external": [],
                            "dependency_refs": [],
                            "catalog_updates": [],
                            "usage_notes": [],
                            "warnings": [],
                        }
                    }
                )
            return json.dumps(
                {
                    "description": "Recovered structure.",
                    "role_in_system": "Recovered role.",
                    "functions": [],
                    "classes": [],
                    "exports": [],
                }
            )

    provider = FlakyProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "main.py",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert provider.calls > 1
    assert "Recovered file." in (
        tmp_path / "docs_output" / "codedoc.json"
    ).read_text(encoding="utf-8")

def test_D8_process_batch_returns_exception_with_descriptor(tmp_path, monkeypatch):
    """D8: retry_rate_limited contains (request, exception) tuples."""
    from codedoc.utils.errors import LLMError
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.queue import ProcessingQueue
    from codedoc.core.safe_writer import SafeWriter
    from tests.support.execution_requests import make_execution_request

    (tmp_path / "main.py").write_text("x=1\n")
    descriptor = {
        "rel_path": "main.py",
        "path": tmp_path / "main.py",
        "language": "python",
    }
    request = make_execution_request(tmp_path, "main.py", "x=1\n", write=False)

    class RateLimitProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "429 rate_limit_exceeded tpm")
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    from codedoc.agents.orchestrator import Orchestrator
    orch = Orchestrator(RateLimitProvider(), parallel=False)

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0, "failed": 0}
    from codedoc.utils.errors import ErrorReporter
    error_reporter = ErrorReporter()

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {"main.py": descriptor})
    sw.set_queue_order(["main.py"])

    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    profile = get_rate_limit_profile("openai")

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [request], orch, queue, stats, error_reporter,
        max_workers=1, recorder=sw, profile=profile,
    )

    assert len(retry_rate_limited) == 1, "Rate-limited file must be in retry list"
    desc_back, exc_back = retry_rate_limited[0]
    assert desc_back is request, "Original request must be returned"
    # The exception may be AgentError wrapping LLMError — verify it carries
    # the rate-limit signal so _is_rate_limit_error can detect it via chain walk.
    assert exc_back is not None, "Exception must be preserved (not None)"
    assert "429" in str(exc_back), (
        f"Rate-limit signal must be somewhere in the exception string: {exc_back!r}"
    )


def test_keyboard_interrupt_stops_queued_parallel_work(tmp_path, monkeypatch):
    import codedoc.core.execution as execution
    from codedoc.core.queue import ProcessingQueue
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ErrorReporter
    from tests.support.execution_requests import make_execution_request

    requests = []
    descriptors = {}
    queue = ProcessingQueue()
    for index in range(6):
        rel_path = f"file_{index}.py"
        source = f"value = {index}\n"
        request = make_execution_request(
            tmp_path,
            rel_path,
            source,
            write=False,
        )
        requests.append(request)
        descriptor = {
            "rel_path": rel_path,
            "path": tmp_path / rel_path,
            "language": "python",
        }
        descriptors[rel_path] = descriptor
        queue.add(descriptor)

    started = []

    def interrupt_first(request, _orchestrator):
        started.append(request.rel_path)
        raise KeyboardInterrupt()

    monkeypatch.setattr(execution, "_process_one_file", interrupt_first)
    recorder = SafeWriter(
        tmp_path / "crash_recovery.json",
        "json",
        None,
        descriptors,
    )
    orchestrator = SimpleNamespace(
        llm=SimpleNamespace(provider_name="InterruptProvider")
    )

    with pytest.raises(KeyboardInterrupt):
        execution._process_descriptor_batch(
            requests,
            orchestrator,
            queue,
            {"checked": 0, "failed": 0},
            ErrorReporter(),
            max_workers=1,
            recorder=recorder,
            max_consecutive_failures=100,
        )

    assert started == ["file_0.py"]
    assert not recorder.recorded_this_run("file_0.py")


def test_inner_triple_interrupt_prevents_correction_and_preserves_interrupt(
    tmp_path,
):
    import threading

    import codedoc.core.execution as execution
    from codedoc.agents.orchestrator import Orchestrator
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.execution_model import (
        CallManifestTracker,
        build_call_manifest,
    )
    from codedoc.core.queue import ProcessingQueue
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ErrorReporter
    from tests.support.execution_requests import make_execution_request

    request = make_execution_request(
        tmp_path,
        "main.py",
        "value = 1\n",
        analysis_mode="triple",
        write=False,
    )
    manifest = build_call_manifest((), (request.rel_path,), "triple")
    stop_signalled = threading.Event()

    class ObservedTracker(CallManifestTracker):
        def signal_stop(self) -> None:
            super().signal_stop()
            stop_signalled.set()

    tracker = ObservedTracker(manifest)
    correction_ledger = CorrectionLedger(True)
    dependency_entered = threading.Event()
    trace = []
    trace_lock = threading.Lock()
    original_interrupt = KeyboardInterrupt("dependency interrupted")

    class InterruptingTripleProvider:
        provider_name = "InterruptingTripleProvider"

        def complete_json(self, _prompt, system=""):
            if "dependency analysis" in system:
                kind = "dependency"
            elif "senior software engineer analysing source code" in system:
                kind = "structure"
            elif "technical writer" in system:
                kind = "documentation"
            else:
                kind = "correction"
            with trace_lock:
                trace.append(kind)
            if kind == "dependency":
                dependency_entered.set()
                raise original_interrupt
            if kind == "structure":
                if not dependency_entered.wait(timeout=2):
                    raise RuntimeError("dependency call did not start")
                if not stop_signalled.wait(timeout=2):
                    raise RuntimeError("interrupt did not signal cancellation")
                return json.dumps({"exports": "not-a-list"})
            raise AssertionError(f"unexpected provider contact: {kind}")

    orchestrator = Orchestrator(
        InterruptingTripleProvider(),
        parallel=True,
        analysis_mode="triple",
        response_correction_enabled=True,
        correction_ledger=correction_ledger,
        call_tracker=tracker,
    )
    descriptor = {
        "rel_path": request.rel_path,
        "path": tmp_path / request.rel_path,
        "language": "python",
    }
    queue = ProcessingQueue()
    queue.add(descriptor)
    recorder = SafeWriter(
        tmp_path / "crash_recovery.json",
        "json",
        request.rel_path,
        {request.rel_path: descriptor},
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        execution._process_descriptor_batch(
            [request],
            orchestrator,
            queue,
            {"checked": 0, "failed": 0},
            ErrorReporter(),
            max_workers=1,
            recorder=recorder,
        )

    assert caught.value is original_interrupt
    assert sorted(trace) == ["dependency", "structure"]
    assert correction_ledger.snapshot()["response_correction_calls_attempted"] == 0
    assert not recorder.recorded_this_run(request.rel_path)
