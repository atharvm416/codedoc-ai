"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import logging
import threading

import pytest

import codedoc.pipeline as pipeline_mod
from codedoc.agents.base_agent import TRUNCATION_MARKER
from codedoc.core.execution import _process_files_sequentially, reconcile_planned_calls
from codedoc.core.queue import ProcessingQueue
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ErrorReporter, LLMError, UnrecoverableProviderError
from tests.support.execution_requests import make_execution_requests
from tests.support.pipeline_usage import write_py
from tests.support.providers import SmartFake


class _LLM:
    def __init__(self, name="openai"):
        self.provider_name = name


# ---------------------------------------------------------------------------
# Request construction is scoped exactly to plan.agent_rels
# ---------------------------------------------------------------------------


class TestRequestConstructionScope:
    def test_execution_requests_built_only_for_agent_rels(self, tmp_path, monkeypatch):
        """Unchanged same-path files and identical-content reuse never receive
        an execution request; only genuinely agent-routed files do."""
        write_py(tmp_path / "changed.py", "original\n")
        write_py(tmp_path / "unchanged.py", "stable content\n")

        class _Fake:
            provider_name = "fake"

            def complete_json(self, prompt, system=""):
                return json.dumps({"description": "d"})

            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)

        monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _Fake())
        base_config = {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "propagate_changes": False,
        }
        run_pipeline(tmp_path, dict(base_config))

        # Second run: "changed.py" is edited; "twin.py" is a brand-new file whose
        # content is identical to "unchanged.py"'s (identical-content reuse).
        write_py(tmp_path / "changed.py", "modified\n")
        write_py(tmp_path / "twin.py", "stable content\n")

        captured = {}
        real_build = pipeline_mod.build_pipeline_plan

        def _capture(*args, **kwargs):
            plan, materials = real_build(*args, **kwargs)
            captured["plan"] = plan
            captured["materials"] = materials
            return plan, materials

        monkeypatch.setattr("codedoc.pipeline.build_pipeline_plan", _capture)
        run_pipeline(tmp_path, dict(base_config))

        plan = captured["plan"]
        materials = captured["materials"]
        assert plan.agent_rels == frozenset({"changed.py"})
        assert plan.unchanged_rels == frozenset({"unchanged.py"})
        assert plan.identical_reuse_rels == frozenset({"twin.py"})
        assert set(materials.execution_requests) == {"changed.py"}


# ---------------------------------------------------------------------------
# Exact source carried through execution without a worker reread
# ---------------------------------------------------------------------------


class TestNoWorkerReread:
    def test_worker_never_rereads_source_after_the_request_is_built(
        self, tmp_path, monkeypatch
    ):
        """A file mutated on disk after its frozen request was built must not
        influence what the provider receives — the request is the sole input."""
        (tmp_path / "main.py").write_text("original-content\n", encoding="utf-8")

        real_build = pipeline_mod.build_pipeline_plan

        def _mutate_after_snapshot(*args, **kwargs):
            plan, materials = real_build(*args, **kwargs)
            (tmp_path / "main.py").write_text("mutated-content\n", encoding="utf-8")
            return plan, materials

        monkeypatch.setattr("codedoc.pipeline.build_pipeline_plan", _mutate_after_snapshot)

        captured_prompts = []

        class _Capturing:
            provider_name = "fake"

            def complete_json(self, prompt, system=""):
                captured_prompts.append(prompt)
                return json.dumps({"description": "d"})

            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)

        monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _Capturing())
        stats = run_pipeline(tmp_path, {"entry_file": "main.py"})

        assert stats["checked"] == 1
        assert any("original-content" in p for p in captured_prompts)
        assert not any("mutated-content" in p for p in captured_prompts)


# ---------------------------------------------------------------------------
# Deterministic commit ordering regardless of concurrent completion order
# ---------------------------------------------------------------------------


class TestDeterministicCommitOrder:
    def test_final_output_order_is_topological_not_completion_order(
        self, tmp_path, monkeypatch
    ):
        """b.py and c.py are submitted (topological order) before a.py, which
        depends on both. Force a.py to complete FIRST anyway; the committed
        output must still list dependencies before the dependent."""
        (tmp_path / "a.py").write_text("import b\nimport c\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "c.py").write_text("y = 2\n", encoding="utf-8")

        a_done = threading.Event()

        class _OrderScramblingProvider:
            provider_name = "fake"

            def complete_json(self, prompt, system=""):
                if "File: a.py" in prompt:
                    result = json.dumps({"description": "d"})
                    a_done.set()
                    return result
                # b.py and c.py wait for a.py (submitted last) to finish first,
                # forcing completion order to be the reverse of submission order.
                assert a_done.wait(timeout=5), "a.py never completed"
                return json.dumps({"description": "d"})

            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)

        monkeypatch.setattr(
            "codedoc.pipeline.create_provider", lambda _cfg: _OrderScramblingProvider()
        )
        stats = run_pipeline(
            tmp_path,
            {
                "entry_file": "a.py",
                "max_parallel_files": 3,
                "parallel_agents": False,
            },
        )

        assert stats["checked"] == 3
        payload = json.loads(
            (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
        )
        paths = [item["path"] for item in payload["files"]]
        assert paths.index("b.py") < paths.index("a.py")
        assert paths.index("c.py") < paths.index("a.py")


# ---------------------------------------------------------------------------
# Concurrency isolation: mixed truncation outcomes and profile scopes cannot
# leak context through the orchestrator/agents shared across worker threads.
# ---------------------------------------------------------------------------


class TestConcurrencyIsolation:
    def test_mixed_size_and_profile_scope_files_do_not_leak_context(
        self, tmp_path, monkeypatch
    ):
        write_py(tmp_path / "small.py", "x = 1\n")
        (tmp_path / "big.ts").write_text("y" * 5000 + "\n", encoding="utf-8")

        barrier = threading.Barrier(2)
        lock = threading.Lock()
        captured: dict[str, str] = {}

        class _SynchronizedProvider(SmartFake):
            def complete_json(self, prompt, system=""):
                if "File: " in prompt:
                    # Force both concurrently-scheduled documentation calls to be
                    # genuinely in-flight at the same time before either returns,
                    # maximizing exposure of any shared mutable per-call state on
                    # the orchestrator/agent instances both threads share.
                    barrier.wait(timeout=5)
                    with lock:
                        if "File: small.py" in prompt:
                            captured["small.py"] = prompt
                        elif "File: big.ts" in prompt:
                            captured["big.ts"] = prompt
                return super().complete_json(prompt, system)

        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda _cfg: _SynchronizedProvider(),
        )
        stats = run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "documentation_scope": "all",
                "max_parallel_files": 2,
                "max_content_chars": 1000,
                "file_retry_attempts": 0,
                "propagate_changes": False,
                "prompt_profiles": {
                    "single": {
                        "common": {
                            "fields": [
                                {
                                    "key": "description",
                                    "type": "string",
                                    "instruction": "default instruction",
                                }
                            ]
                        },
                        "per_extension": {
                            ".ts": {
                                "fields": [
                                    {
                                        "key": "description",
                                        "type": "string",
                                        "instruction": "typescript-only instruction",
                                    }
                                ]
                            }
                        },
                    }
                },
            },
        )

        assert stats["checked"] == 2
        assert set(captured) == {"small.py", "big.ts"}
        # Neither file's content leaked into the other's prompt.
        assert "y" * 100 not in captured["small.py"]
        assert "x = 1" in captured["small.py"]
        # Each file used its own resolved profile scope, not the other's.
        assert "default instruction" in captured["small.py"]
        assert "typescript-only instruction" not in captured["small.py"]
        assert "typescript-only instruction" in captured["big.ts"]
        assert "default instruction" not in captured["big.ts"]
        # The oversized file was truncated to the shared ceiling; the small one
        # was not — proving per-file truncation outcome under shared agents.
        # (The RULES text always mentions "[truncated]" as an instruction to the
        # model regardless of file size, so the real marker is checked instead.)
        assert TRUNCATION_MARKER in captured["big.ts"]
        assert TRUNCATION_MARKER not in captured["small.py"]


# ---------------------------------------------------------------------------
# Bounded observability: identifiers and outcomes only, never content
# ---------------------------------------------------------------------------


class TestBoundedDebugLogging:
    def test_debug_logs_carry_identifiers_and_outcomes_but_never_content(
        self, tmp_path, monkeypatch, caplog
    ):
        """Debug observability is limited to the plan digest, call IDs and
        categories, the request owner, and outcomes. Raw source, prompts,
        custom instructions, and provider responses must never reach a log."""
        write_py(tmp_path / "main.py", "SOURCE_MARKER_NOT_LOGGABLE = 1\n")

        class _Fake(SmartFake):
            def complete_json(self, prompt, system=""):
                if "standards/safety review" in prompt:
                    return super().complete_json(prompt, system)
                return json.dumps(
                    {
                        "description": "RESPONSE_MARKER_NOT_LOGGABLE",
                        "role_in_system": "r",
                    }
                )

        monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _Fake())
        with caplog.at_level(logging.DEBUG, logger="codedoc"):
            stats = run_pipeline(
                tmp_path,
                {
                    "entry_file": "main.py",
                    "log_level": "DEBUG",
                    "prompt_profiles": {
                        "single": {
                            "common": {
                                "fields": [
                                    {
                                        "key": "description",
                                        "type": "string",
                                        "instruction": "INSTRUCTION_MARKER_NOT_LOGGABLE",
                                    }
                                ]
                            }
                        }
                    },
                },
            )

        assert stats["checked"] == 1
        text = "\n".join(record.getMessage() for record in caplog.records)
        # Present: the bounded identifiers and the per-file outcome.
        assert stats["call_manifest_digest"] in text
        assert "category=prompt-review" in text
        assert "category=file-documentation" in text
        assert "Outcome for owner=main.py: state=checked" in text
        # Absent: everything that could leak content or credentials.
        for secret in (
            "SOURCE_MARKER_NOT_LOGGABLE",
            "RESPONSE_MARKER_NOT_LOGGABLE",
            "INSTRUCTION_MARKER_NOT_LOGGABLE",
        ):
            assert secret not in text


# ---------------------------------------------------------------------------
# Planned-versus-attempted-logical-call reconciliation
# ---------------------------------------------------------------------------


class TestPlannedIdReconciliation:
    def test_successful_retry_counts_as_an_additional_attempt(self, tmp_path, monkeypatch):
        write_py(tmp_path / "main.py")

        class _FlakyOnce:
            provider_name = "fake"

            def __init__(self):
                self.calls = 0

            def complete_json(self, prompt, system=""):
                self.calls += 1
                if self.calls == 1:
                    raise LLMError("openai", "temporary provider outage")
                return json.dumps({"description": "d"})

            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)

        provider = _FlakyOnce()
        monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
        stats = run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "file_retry_attempts": 1,
                "max_parallel_files": 1,
                "propagate_changes": False,
            },
        )

        assert stats["checked"] == 1
        assert stats["failed"] == 0
        assert provider.calls == 2
        assert stats["total_calls_planned"] == 1
        assert stats["attempted_logical_calls"] == 1
        assert stats["planned_calls_not_attempted"] == 0
        assert stats["additional_attempts"] == 1

    def test_allow_partial_completion_reconciles_with_nothing_unattempted(
        self, tmp_path, monkeypatch
    ):
        write_py(tmp_path / "good.py")
        write_py(tmp_path / "bad.py")

        class _OneFails:
            provider_name = "fake"

            def complete_json(self, prompt, system=""):
                if "File: bad.py" in prompt:
                    return "not json"
                return json.dumps({"description": "d"})

            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)

        monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _OneFails())
        stats = run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "documentation_scope": "all",
                "file_retry_attempts": 0,
                "allow_partial": True,
                "propagate_changes": False,
            },
        )

        assert stats["checked"] == 1
        assert stats["failed"] == 1
        assert stats["total_calls_planned"] == 2
        assert stats["attempted_logical_calls"] == 2
        assert stats["planned_calls_not_attempted"] == 0
        assert stats["additional_attempts"] == 0

    def test_terminal_failure_reconciles_retry_and_unattempted_ids_without_hiding_either(
        self, tmp_path
    ):
        """f0 needs one retry to succeed; f1 aborts the run with a terminal
        billing error; f2/f3 are never reached. additional_attempts must come
        from attempted logical calls actually reached, not the full planned
        total — proving 3 - 1 == 2, not 4 - 1 == 3."""
        requests = make_execution_requests(tmp_path, ["f0.py", "f1.py", "f2.py", "f3.py"])

        class _RetryThenTerminalOrch:
            def __init__(self):
                self.llm = _LLM("openai")
                self.calls_by_file: dict[str, int] = {}

            def process(self, request):
                rel = request.rel_path
                self.calls_by_file[rel] = self.calls_by_file.get(rel, 0) + 1
                if rel == "f0.py":
                    if self.calls_by_file[rel] == 1:
                        raise LLMError("openai", "temporary provider outage")
                    return {"file_path": rel, "language": "python"}
                if rel == "f1.py":
                    raise LLMError("openai", "Your credit balance is too low to continue")
                return {"file_path": rel, "language": "python"}

        orch = _RetryThenTerminalOrch()
        queue = ProcessingQueue()
        # Mirror execute_agent_files' upfront dequeue-all-before-processing step:
        # every queued request is already STATUS_IN_PROGRESS before the
        # sequential pass begins, exactly as it would be in a real run.
        for request in requests:
            queue.add({"rel_path": request.rel_path})
        for _ in requests:
            queue.next()
        stats = {"checked": 0, "failed": 0}
        reporter = ErrorReporter()
        recorder = SafeWriter(tmp_path / "codedoc.json", "json", "f0.py", {})

        with pytest.raises(UnrecoverableProviderError) as excinfo:
            _process_files_sequentially(
                requests,
                orch,
                queue,
                stats,
                reporter,
                retry_attempts=1,
                max_consecutive_failures=5,
                new_results={},
                recorder=recorder,
            )
        assert excinfo.value.category == "terminal"

        snapshot = queue.snapshot()
        assert snapshot["f0.py"] == "checked"
        assert snapshot["f1.py"] == "in_progress"
        assert snapshot["f2.py"] == "in_progress"
        assert snapshot["f3.py"] == "in_progress"

        # f0: one failed attempt + one successful retry = 2. f1: one attempt
        # before the abort = 1. Total actual attempts = 3.
        attempted_calls = sum(orch.calls_by_file.values())
        assert attempted_calls == 3

        result = reconcile_planned_calls(
            total_calls_planned=4,
            attempted_logical_calls=2,
            attempted_calls=attempted_calls,
        )
        assert result["attempted_logical_calls"] == 2
        assert result["planned_calls_not_attempted"] == 2
        assert result["additional_attempts"] == 1

    def test_pipeline_terminal_error_carries_manifest_reconciliation(
        self, tmp_path, monkeypatch
    ):
        for rel in ("f0.py", "f1.py", "f2.py"):
            write_py(tmp_path / rel)

        class _RetryThenTerminal:
            provider_name = "openai"

            def __init__(self):
                self.calls_by_file = {}

            def complete_json(self, prompt, system=""):
                rel = next(
                    name for name in ("f0.py", "f1.py", "f2.py")
                    if f"File: {name}" in prompt
                )
                self.calls_by_file[rel] = self.calls_by_file.get(rel, 0) + 1
                if rel == "f0.py" and self.calls_by_file[rel] == 1:
                    raise LLMError("openai", "temporary provider outage")
                if rel == "f1.py":
                    raise LLMError(
                        "openai", "Your credit balance is too low to continue"
                    )
                return json.dumps({"description": "d"})

            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt, system)

        provider = _RetryThenTerminal()
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider", lambda _cfg: provider
        )

        with pytest.raises(UnrecoverableProviderError) as caught:
            run_pipeline(
                tmp_path,
                {
                    "entry_file": None,
                    "auto_entry_candidates": [],
                    "documentation_scope": "all",
                    "propagate_changes": False,
                    "max_parallel_files": 1,
                    "file_retry_attempts": 1,
                },
            )

        stats = caught.value.stats
        assert stats["total_calls_planned"] == 3
        assert stats["attempted_calls"] == 3
        assert stats["attempted_logical_calls"] == 2
        assert stats["planned_calls_not_attempted"] == 1
        assert stats["additional_attempts"] == 1
