"""End-to-end split-mode pipeline behavior: leaf/reduction/final call
sequencing, retry/correction honesty, rate-limit step-down, terminal-failure
checkpointing, node-keyed recovery resumption, and the D8 whole-run-abort
contract for a genuine ``DivisionInternalDefect``.

Split is valid only in ``analysis_mode: 'single'`` (D2); every scenario here
uses single mode only.
"""

from __future__ import annotations

import hashlib
import json
import threading

import pytest
import codedoc.core.file_division as file_division
from codedoc.core.result_assembly import flat_combined_result
from codedoc.core.execution import _process_descriptor_batch
from codedoc.core.execution_model import build_call_manifest
from codedoc.core.file_division import (
    BLOCKED_REASON_ORDER,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    DivisionInternalDefect,
    SplitCapacityBlocked,
    SplitTreeState,
    build_division_plan,
    build_reduction_tree,
    canonical_json,
    final_execution_identity,
    leaf_execution_identity,
    provider_execution_identity,
    reduction_execution_identity,
    tree_node_state,
)
from codedoc.core.loader import load_config
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    resolve_profile_source,
)
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import RecoveryState, build_recovery_identity
from codedoc.core.safe_writer import SafeWriter
from codedoc.llm.factory import (
    ProviderExecutionDescriptor,
    attest_provider_execution,
)
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import (
    ConfigError,
    ErrorReporter,
    LLMError,
    UnrecoverableProviderError,
)
from tests.support.execution_requests import (
    make_execution_request,
    make_execution_requests,
)
from tests.support.providers import SmartFake
from tests.support.profiles import INLINE
from tests.support.structure_extra import requires_structure_pack

pytestmark = pytest.mark.future_split_execution


def _large_python_source(lines: int = 220) -> str:
    # Named function declarations, not bare top-level statements: under
    # syntax-mode parsing, adjacent bare statements with no intervening
    # declaration can merge into one shared "gap" unit spanning the whole
    # fixture, producing wildly different (and here, capacity-blocked or
    # retry-budget-exhausting) chunk counts than under lexical fallback. A
    # `def` is reliably its own semantic unit under both parsing modes.
    return "\n".join(f"def fn_{index}(): return {index}" for index in range(lines)) + "\n"


def _realistic_service_source(methods: int = 140) -> str:
    return (
        "class ApplicationService:\n"
        '    """Coordinates a representative application workflow."""\n\n'
        + "".join(
            (
                f"    def operation_{index:03d}(self, value: int) -> int:\n"
                f"        normalized = value + {index}\n"
                "        return normalized\n\n"
            )
            for index in range(methods)
        )
    )


def test_planning_preview_does_not_reuse_completed_split_output(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    provider = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
        },
    )

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("split planning preview created a provider"),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "dry_run": True,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
        },
    )

    assert stats["would_reuse"] == 0
    assert stats["split_divided_files"] == 1
    assert stats["unit_documentation_calls_planned"] > 0


def _reduction_total(tree) -> int:
    return len(tree.unit_consolidation_nodes) + len(tree.general_nodes)


def _provider_identity_for(tmp_path, overrides: dict) -> str:
    """The exact provider-free identity `run_pipeline` will compute for this
    config — resolved the same way planning resolves it, so a hand-built
    recovery checkpoint validates as current."""
    resolved = load_config(tmp_path, overrides)
    return provider_execution_identity(resolved)


def _fully_completed_tree_state(
    plan,
    tree,
    *,
    provider_identity: str,
    content_hash: str,
    prompt_profile_digest: str = NO_PROMPT_PROFILE_DIGEST,
    final_fields: dict | None = None,
) -> SplitTreeState:
    """A synthetic but dependency-valid SplitTreeState covering every leaf,
    reduction, and final node — as if the whole tree had already been paid
    for and checkpointed in an earlier run."""
    nodes = [
        tree_node_state(
            node_id=chunk.chunk_id,
            node_type="leaf",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            execution_identity_digest=leaf_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                provider_identity=provider_identity,
                chunk=chunk,
            ),
            unit_id=None,
            child_ids=(),
            coverage_leaf_ids=(chunk.chunk_id,),
            result={
                "description": f"restored {index}",
                "chunk_id": chunk.chunk_id,
                "unit_id": chunk.unit_id,
            },
        )
        for index, chunk in enumerate(plan.chunks)
    ]
    for node in tree.unit_consolidation_nodes + tree.general_nodes:
        nodes.append(
            tree_node_state(
                node_id=node.node_id,
                node_type=node.phase,
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                execution_identity_digest=reduction_execution_identity(
                    rel_path=plan.rel_path,
                    content_hash=content_hash,
                    division_plan_digest=plan.plan_digest,
                    reduction_tree_digest=tree.tree_digest,
                    provider_identity=provider_identity,
                    node=node,
                ),
                unit_id=node.unit_id,
                child_ids=node.child_ids,
                coverage_leaf_ids=node.leaf_ids,
                result={"narrative": "restored narrative"},
            )
        )
    final = tree.final_node
    nodes.append(
        tree_node_state(
            node_id=final.node_id,
            node_type="final",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            execution_identity_digest=final_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                prompt_profile_digest=prompt_profile_digest,
                node=final,
            ),
            unit_id=None,
            child_ids=final.child_ids,
            coverage_leaf_ids=final.leaf_ids,
            result=flat_combined_result(
                plan.rel_path,
                "python",
                [],
                final_fields or {"description": "restored complete file"},
            ),
        )
    )
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(nodes),
    )


def _one_leaf_completed_tree_state(plan, tree, *, provider_identity: str, content_hash: str) -> SplitTreeState:
    leaf_identity = leaf_execution_identity(
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        provider_identity=provider_identity,
        chunk=plan.chunks[0],
    )
    node = tree_node_state(
        node_id=plan.chunks[0].chunk_id,
        node_type="leaf",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        execution_identity_digest=leaf_identity,
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(plan.chunks[0].chunk_id,),
        result={
            "description": "restored",
            "chunk_id": plan.chunks[0].chunk_id,
            "unit_id": plan.chunks[0].unit_id,
        },
    )
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=(node,),
    )


def test_split_division_manifest_counts_leaves_reduction_and_synthesis() -> None:
    source = "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n"
    plan = build_division_plan(
        rel_path="src/large.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)

    manifest = build_call_manifest(
        [],
        ["src/large.py"],
        "single",
        division_plans={"src/large.py": plan},
        reduction_trees={"src/large.py": tree},
    )

    categories = [call.category for call in manifest.calls]
    assert categories.count("unit-documentation") == len(plan.chunks)
    assert categories.count("file-reduction") == _reduction_total(tree)
    assert categories.count("file-synthesis") == 1
    assert not any(category == "file-documentation" for category in categories)


def test_split_pipeline_documents_all_chunks_then_synthesizes(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)
    provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    output = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    record = output["files"][0]
    assert stats["failed"] == 0
    assert stats["checked"] == 1
    assert provider.doc_calls == stats["documentation_calls_attempted"]
    assert stats["unit_documentation_calls_planned"] == len(division.chunks)
    assert stats["file_reduction_calls_planned"] == _reduction_total(tree)
    assert stats["synthesis_calls_planned"] == 1
    assert stats["split_chunks"] == len(division.chunks)
    assert "division" not in record
    assert "documentation_units" not in record
    assert record["_large_file_identity"].startswith("large-file-v2:")
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


@requires_structure_pack
def test_realistic_large_class_plans_and_executes_with_proportional_calls(
    tmp_path, monkeypatch
) -> None:
    source = _realistic_service_source()
    (tmp_path / "service.py").write_text(
        source,
        encoding="utf-8",
        newline="",
    )
    config = {
        "entry_file": "service.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 12000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    division = build_division_plan(
        rel_path="service.py",
        language="python",
        content=source,
        source_budget_chars=12000,
    )
    tree = build_reduction_tree(division, max_content_chars=12000)

    assert len(source) > 12000
    assert 2 <= len(division.chunks) <= (len(source) + 11999) // 12000 + 1

    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})
    assert dry_stats["split_blocked_files"] == 0
    assert dry_stats["split_chunks"] == len(division.chunks)
    assert dry_stats["unit_documentation_calls_planned"] == len(
        division.chunks
    )
    assert not (tmp_path / "docs").exists()

    provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: provider,
    )
    stats = run_pipeline(tmp_path, config)

    assert stats["failed"] == 0
    assert stats["checked"] == 1
    assert stats["split_chunks"] == len(division.chunks)
    assert stats["file_reduction_calls_planned"] == _reduction_total(tree)
    assert stats["synthesis_calls_planned"] == 1
    assert provider.doc_calls == stats["documentation_calls_attempted"]


def test_constructed_provider_identity_mismatch_aborts_before_any_call(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "main.py").write_text(
        "def main():\n    return 1\n",
        encoding="utf-8",
        newline="",
    )
    provider = SmartFake()
    provider._codedoc_provider_execution_descriptor = (
        ProviderExecutionDescriptor(
            provider_kind="anthropic",
            model="different-model",
            endpoint_identity="provider-default",
        )
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: provider,
    )

    with pytest.raises(ConfigError, match="does not match the provider-free plan"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "analysis_mode": "single",
                "large_file_strategy": "split",
                "parallel_agents": False,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    assert provider.doc_calls == 0


def test_split_leaf_retry_repeats_only_the_incomplete_leaf(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)

    class FailSecondLeafOnce(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self.leaf_prompts: list[str] = []
            self.failed = False

        def complete_json(self, prompt, system=""):
            if "This is one bounded fragment of a larger" in prompt:
                self.leaf_prompts.append(prompt)
                if len(self.leaf_prompts) == 2 and not self.failed:
                    self.failed = True
                    self.doc_calls += 1
                    raise LLMError("openai", "temporary provider outage")
            return super().complete_json(prompt, system)

    provider = FailSecondLeafOnce()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert provider.leaf_prompts[1] == provider.leaf_prompts[2]
    assert provider.leaf_prompts.count(provider.leaf_prompts[0]) == 1
    assert provider.leaf_prompts.count(provider.leaf_prompts[1]) == 2
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned
    assert stats["failed_calls"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_split_synthesis_retry_does_not_repeat_completed_leaves(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)

    class FailSynthesisOnce(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self.leaf_prompts: list[str] = []
            self.synthesis_prompts: list[str] = []

        def complete_json(self, prompt, system=""):
            if "This is one bounded fragment of a larger" in prompt:
                self.leaf_prompts.append(prompt)
            if "Synthesize one final file-level documentation JSON object" in prompt:
                self.synthesis_prompts.append(prompt)
                if len(self.synthesis_prompts) == 1:
                    self.doc_calls += 1
                    raise LLMError("openai", "temporary provider outage")
            return super().complete_json(prompt, system)

    provider = FailSynthesisOnce()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert len(provider.leaf_prompts) == len(division.chunks)
    assert len(set(provider.leaf_prompts)) == len(division.chunks)
    assert len(provider.synthesis_prompts) == 2
    assert provider.synthesis_prompts[0] == provider.synthesis_prompts[1]
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned
    assert stats["failed_calls"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_split_response_correction_uses_the_originating_leaf_call(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)

    class CorrectFirstLeaf(SmartFake):
        def __init__(self) -> None:
            super().__init__()
            self.failed_initial = False
            self.correction_calls = 0

        def complete_json(self, prompt, system=""):
            if "Previous response (verbatim" in prompt:
                self.correction_calls += 1
                return super().complete_json(prompt, system)
            if (
                "This is one bounded fragment of a larger" in prompt
                and "File: main.py" in prompt
                and not self.failed_initial
            ):
                self.failed_initial = True
                self.doc_calls += 1
                return json.dumps({"functions": ["missing required description"]})
            return super().complete_json(prompt, system)

    provider = CorrectFirstLeaf()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    callback_calls = []
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "parallel_agents": False,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 1,
            "response_correction_enabled": True,
            "propagate_changes": False,
            "output_dir": "docs",
        },
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert provider.correction_calls == 1
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned + 1
    assert stats["failed_calls"] == 0
    assert stats["response_contract_failures"] == 1
    assert stats["response_correction_calls_attempted"] == 1
    assert stats["response_correction_calls_succeeded"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert callback_calls == []
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()
    assert json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]["description"] == "A file."


def test_mixed_ordinary_and_divided_files_resume_after_rate_limit_step_down(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "small.py").write_text(
        "def small_value():\n    return 1\n", encoding="utf-8"
    )
    # Smaller than the shared _large_python_source() default: this test
    # exercises genuine two-thread concurrency (large.py and small.py process
    # simultaneously before the rate-limit step-down), so it deliberately
    # minimizes large.py's own checkpoint-write volume — just enough to
    # require 2+ chunks under both parsing modes — rather than adding
    # unrelated I/O contention on top of the concurrency already under test.
    source = _large_python_source(120)
    (tmp_path / "large.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="large.py", language="python", content=source, source_budget_chars=2000
    )

    class RateLimitSecondLargeLeaf(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self.large_leaf_prompts: list[str] = []
            self.failed = False

        def complete_json(self, prompt, system=""):
            if (
                "This is one bounded fragment of a larger" in prompt
                and "File: large.py" in prompt
            ):
                with self._lock:
                    self.large_leaf_prompts.append(prompt)
                    should_fail = (
                        len(self.large_leaf_prompts) == 2 and not self.failed
                    )
                    if should_fail:
                        self.failed = True
                        self.doc_calls += 1
                if should_fail:
                    raise LLMError("openai", "429 rate_limit_exceeded")
            return super().complete_json(prompt, system)

    provider = RateLimitSecondLargeLeaf()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda _seconds: None)

    callback_calls = []
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "analysis_mode": "single",
            "parallel_agents": False,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 2,
            "rate_limit_adaptive": True,
            "file_retry_attempts": 0,
            "rate_limit_backoff_s": 0,
            "respect_retry_after": False,
            "propagate_changes": False,
            "output_dir": "docs",
        },
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    payload = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert {record["path"] for record in payload["files"]} == {
        "large.py",
        "small.py",
    }
    assert stats["checked"] == 2
    assert stats["failed"] == 0
    assert stats["split_ordinary_files"] == 1
    assert stats["split_divided_files"] == 1
    assert len(stats["rate_limit_warnings"]) == 1
    assert len(provider.large_leaf_prompts) == len(division.chunks) + 1
    assert provider.large_leaf_prompts.count(provider.large_leaf_prompts[0]) == 1
    assert provider.large_leaf_prompts.count(provider.large_leaf_prompts[1]) == 2
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert callback_calls == []
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_terminal_split_failure_cancels_pending_file_tasks_and_keeps_checkpoint(
    tmp_path,
) -> None:
    source = _large_python_source(120)
    split_request = make_execution_request(
        tmp_path,
        "a_large.py",
        source,
        analysis_mode="single",
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path=split_request.rel_path,
        language=split_request.language,
        content=source,
        source_budget_chars=2000,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    ordinary_requests = make_execution_requests(
        tmp_path,
        [f"ordinary_{index}.py" for index in range(5)],
    )

    class TerminalSplitOrchestrator:
        class _LLM:
            provider_name = "openai"

        llm = _LLM()

        def __init__(self) -> None:
            self.leaf_calls = 0
            self.ordinary_calls = 0
            self.ordinary_delay = threading.Event()

        def process_leaf_chunk(self, chunk_request):
            self.leaf_calls += 1
            if self.leaf_calls == 2:
                raise LLMError(
                    "openai", "Your credit balance is too low to continue"
                )
            return {"description": f"chunk {chunk_request.chunk_id}"}

        def process_reduction_node(self, _request):
            pytest.fail("terminal split task reached a reduction node")

        def synthesize_divided_file(self, _request, _digest, _manifest):
            pytest.fail("terminal split task reached synthesis")

        def process(self, request):
            self.ordinary_calls += 1
            self.ordinary_delay.wait(0.1)
            return {
                "file_path": request.rel_path,
                "language": request.language,
                "description": "ordinary",
            }

    class Queue:
        def __init__(self) -> None:
            self.checked: list[str] = []
            self.failed: list[tuple[str, str]] = []

        def mark_checked(self, rel_path):
            self.checked.append(rel_path)

        def mark_failed(self, rel_path, reason):
            self.failed.append((rel_path, reason))

    orchestrator = TerminalSplitOrchestrator()
    queue = Queue()
    stats = {"checked": 0, "failed": 0}
    writer = SafeWriter(
        tmp_path / "docs" / "crash_recovery.json",
        "json",
        None,
        {
            request.rel_path: {}
            for request in (split_request, *ordinary_requests)
        },
    )

    with pytest.raises(UnrecoverableProviderError):
        _process_descriptor_batch(
            [split_request, *ordinary_requests],
            orchestrator,
            queue,
            stats,
            ErrorReporter(),
            max_workers=1,
            recorder=writer,
            division_plans={split_request.rel_path: plan},
            reduction_trees={split_request.rel_path: tree},
            provider_identity="test-provider",
        )

    checkpoint = writer.get_tree_state(split_request.rel_path)
    assert checkpoint is not None
    completed = checkpoint.by_id()
    assert len(completed) == 1
    assert all(node.node_type == "leaf" for node in completed.values())
    assert orchestrator.leaf_calls == 2
    assert orchestrator.ordinary_calls <= 1
    assert writer.get_record(split_request.rel_path) is None
    assert queue.failed == []
    assert stats["failed"] == 0
    persisted = json.loads(writer.path.read_text(encoding="utf-8"))
    persisted_nodes = persisted["_codedoc"]["partial_files"][split_request.rel_path][
        "nodes"
    ]
    assert len(persisted_nodes) == 1
    assert next(iter(persisted_nodes.values()))["node_type"] == "leaf"


def test_fully_synthesized_split_recovery_finalizes_without_a_provider(
    tmp_path, monkeypatch
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_files": 1,
        "propagate_changes": False,
        "output_dir": "docs",
        "prompt_profiles": INLINE,
    }
    provider_identity = _provider_identity_for(tmp_path, config)
    resolved_config = load_config(tmp_path, config)
    profile_resolution = resolve_profile_source(
        resolved_config,
        tmp_path,
        known_extensions=frozenset(resolved_config["extension_language_map"]),
        active_mode="single",
    )
    profile_digest = ResolvedProfile(
        "single", profile_resolution.profile
    ).file_digest("main.py")
    recovered = _fully_completed_tree_state(
        division,
        tree,
        provider_identity=provider_identity,
        content_hash=content_hash,
        prompt_profile_digest=profile_digest,
        final_fields={
            "description": "restored complete file",
            "key_concepts": ["restored"],
        },
    )
    monkeypatch.setattr(
        "codedoc.pipeline.load_recovery_records_if_compatible",
        lambda *_args, **_kwargs: RecoveryState(
            records=(
                (
                    "main.py",
                    canonical_json({"path": "main.py", "hash": "stale"}),
                ),
            ),
            partial_files=(recovered,),
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("restored synthesis created a provider")
        ),
    )

    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})
    assert dry_stats["max_files_candidate_files"] == 1
    assert dry_stats["prompt_customization_security_review_calls_planned"] == 1

    callback_calls = []
    stats = run_pipeline(
        tmp_path,
        config,
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    output = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert stats["checked"] == 1
    assert stats["resumed"] == 1
    assert stats["total_calls_planned"] == 0
    assert stats["attempted_calls"] == 0
    assert stats["prompt_customization_security_review_calls_planned"] == 0
    assert stats["split_restored_complete_chunks"] == len(division.chunks)
    assert stats["split_restored_unit_consolidation_calls"] + stats[
        "split_restored_general_reduction_calls"
    ] == _reduction_total(tree)
    assert stats["split_restored_final_synthesis_calls"] == 1
    assert callback_calls == []
    assert output["files"][0]["description"] == "restored complete file"
    assert (
        output["last_run"]["split_restored_complete_chunks"]
        == len(division.chunks)
    )
    assert output["last_run"]["split_restored_final_synthesis_calls"] == 1
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_truncate_run_blocks_on_split_recovery_without_erasing_checkpoints(
    tmp_path, monkeypatch
) -> None:
    """A truncate run must fail closed on split recovery, not silently drop it.

    A truncate flush cannot read or rewrite ``partial_files``, so re-flushing one
    would erase already-paid node work.  ``large_file_strategy`` is therefore
    part of the recovery identity: the mismatch is refused before SafeWriter
    initialization and provider creation, the recovery file stays byte-for-byte
    intact, and a later split run still resumes the same checkpoint.
    """
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "truncate",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    provider_identity = _provider_identity_for(
        tmp_path, {**config, "large_file_strategy": "split"}
    )
    recovered = _one_leaf_completed_tree_state(
        division, tree, provider_identity=provider_identity, content_hash=content_hash
    )
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path,
        "json",
        "main.py",
        {},
        build_recovery_identity(
            project_root=tmp_path,
            json_target=tmp_path / "docs" / "codedoc.json",
            md_target=None,
            entry_file="main.py",
            documentation_scope="entry",
            analysis_mode="single",
            analysis_revision=ANALYSIS_REVISION,
            large_file_strategy="split",
        ),
    )
    for node in recovered.nodes:
        writer.record_tree_node("main.py", node)
    original_recovery = recovery_path.read_bytes()

    # A dry run stays non-mutating and provider-free: it neither counts nor
    # rewrites the split checkpoint, and emits no split observability.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("dry run created a provider"),
    )
    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})

    assert dry_stats["would_resume"] == 0
    assert not any(key.startswith("split_") for key in dry_stats)
    assert recovery_path.read_bytes() == original_recovery

    # The real truncate run blocks before SafeWriter or provider creation.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("incompatible recovery created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.load",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible recovery initialized the writer"
        ),
    )
    with pytest.raises(ConfigError) as blocked:
        run_pipeline(tmp_path, config)

    assert "large_file_strategy" in str(blocked.value)
    assert "'truncate'" in str(blocked.value)
    assert "'split'" in str(blocked.value)
    assert recovery_path.read_bytes() == original_recovery
    assert not (tmp_path / "docs" / "codedoc.json").exists()

    # The same checkpoint is still resumable by a split run.
    monkeypatch.undo()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda provider_config: attest_provider_execution(
            SmartFake(),
            provider_config,
        ),
    )
    stats = run_pipeline(tmp_path, {**config, "large_file_strategy": "split"})

    assert stats["resumed"] == 1
    assert stats["split_restored_complete_chunks"] == 1
    record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert "division" not in record
    assert "documentation_units" not in record
    assert record["_large_file_identity"].startswith("large-file-v2:")
    assert not recovery_path.exists()


def test_legacy_recovery_without_strategy_stays_compatible_with_truncate(
    tmp_path, monkeypatch
) -> None:
    """An identity written before this release has no strategy field.

    Absence normalizes to the default ``truncate``, so ordinary recovery keeps
    resuming and default identity bytes are unchanged.
    """
    (tmp_path / "main.py").write_bytes(b"VALUE = 1\n")
    legacy_identity = build_recovery_identity(
        project_root=tmp_path,
        json_target=tmp_path / "docs" / "codedoc.json",
        md_target=None,
        entry_file="main.py",
        documentation_scope="entry",
        analysis_mode="single",
        analysis_revision=ANALYSIS_REVISION,
    )

    assert "large_file_strategy" not in legacy_identity

    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path, "json", "main.py", {}, legacy_identity
    )
    writer.record(
        "main.py",
        {"file_path": "main.py", "language": "python", "description": "Legacy."},
        hashlib.sha256(b"VALUE = 1\n").hexdigest(),
    )

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda provider_config: attest_provider_execution(
            SmartFake(),
            provider_config,
        ),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    assert stats["resumed"] == 1
    assert not any(key.startswith("split_") for key in stats)


def test_split_dry_run_ignores_partial_recovery_and_reports_fresh_calls(
    tmp_path, monkeypatch
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    provider_identity = _provider_identity_for(tmp_path, config)
    recovered = _one_leaf_completed_tree_state(
        division, tree, provider_identity=provider_identity, content_hash=content_hash
    )
    recovery_options = []

    def load_recovery(*_args, **kwargs):
        recovery_options.append(kwargs)
        return RecoveryState(
            records=(
                (
                    "main.py",
                    canonical_json({"path": "main.py", "hash": "stale"}),
                ),
            ),
            partial_files=(recovered,),
        )

    monkeypatch.setattr(
        "codedoc.pipeline.load_recovery_records_if_compatible",
        load_recovery,
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("dry-run created a provider"),
    )

    stats = run_pipeline(tmp_path, {**config, "dry_run": True})

    reduction_total = _reduction_total(tree)
    assert recovery_options == [{"include_partial_files": False}]
    assert stats["would_resume"] == 0
    assert stats["split_restored_complete_chunks"] == 0
    assert stats["split_restored_unit_consolidation_calls"] == 0
    assert stats["split_restored_general_reduction_calls"] == 0
    assert stats["split_restored_final_synthesis_calls"] == 0
    assert stats["unit_documentation_calls_planned"] == len(division.chunks)
    assert stats["file_reduction_calls_planned"] == reduction_total
    assert stats["synthesis_calls_planned"] == 1
    assert stats["total_calls_planned"] == (
        len(division.chunks) + reduction_total + 1
    )
    assert stats["split_synthesis_input_estimate"] == "deterministic-worst-case-envelope"
    assert stats["estimated_input_tokens"] > 0
    assert not (tmp_path / "docs").exists()


def test_disconnected_split_call_count_uses_exact_chunk_manifest(
    tmp_path,
) -> None:
    (tmp_path / "main.py").write_bytes(b"ENTRY = True\n")
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "orphan.py").write_bytes(source.encode("utf-8"))

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "documentation_scope": "all",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
        },
    )

    assert stats["disconnected_paid_files"] == 1
    assert stats["disconnected_planned_calls"] == (
        stats["unit_documentation_calls_planned"]
        + stats["file_reduction_calls_planned"]
        + stats["synthesis_calls_planned"]
    )
    assert stats["disconnected_planned_calls"] > stats["initial_calls_per_file"]


def test_capacity_blocked_split_dry_run_reports_sorted_pairs_and_excludes_max_files(
    tmp_path, monkeypatch, capsys
) -> None:
    """D8: dry-run exposes every blocked path/reason, remains provider- and
    writer-free, and excludes blocked files from the paid-file safety cap."""
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8", newline="")
    oversized = _large_python_source(120)
    (tmp_path / "zeta.py").write_text(oversized, encoding="utf-8", newline="")
    (tmp_path / "alpha.py").write_text(oversized, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_files": 1,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }

    # Exercise the real division capacity check rather than manufacturing a
    # planning result. Each oversized fixture needs more than one chunk.
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 1)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("capacity-blocked dry run created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "capacity-blocked dry run constructed a writer"
        ),
    )

    stats = run_pipeline(tmp_path, {**config, "dry_run": True})

    expected_pairs = (
        ("alpha.py", "chunk-cap"),
        ("zeta.py", "chunk-cap"),
    )
    assert stats["split_blocked_files"] == 2
    assert stats["split_blocked_by_reason"] == {"chunk-cap": 2}
    assert stats["split_blocked_pairs"] == expected_pairs
    assert stats["max_files_candidate_files"] == 1
    assert stats["would_call_llm_for"] == 1
    assert stats["max_files_exceeded"] is False
    assert not (tmp_path / "docs").exists()

    # The public CLI must render the same sorted pairs and retain the frozen
    # dry-run exit code 0.
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(
            {
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "max_files": 1,
                "parallel_agents": False,
                "propagate_changes": False,
                "output_dir": "docs",
            }
        ),
        encoding="utf-8",
        newline="",
    )
    from codedoc.cli.cli import run_cli

    capsys.readouterr()
    exit_code = run_cli(
        [
            str(tmp_path),
            "--entry",
            "main.py",
            "--documentation-scope",
            "all",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Blocked path/reason pairs:" in output
    assert output.index("alpha.py (chunk-cap)") < output.index("zeta.py (chunk-cap)")
    assert not (tmp_path / "docs").exists()

    # The corresponding real run reports every pair in the same order and
    # aborts at the already-patched writer/provider boundary.
    with pytest.raises(ConfigError) as blocked:
        run_pipeline(tmp_path, config)
    message = str(blocked.value)
    assert message.index("alpha.py (chunk-cap)") < message.index(
        "zeta.py (chunk-cap)"
    )
    assert not (tmp_path / "docs").exists()


@pytest.mark.parametrize("reason", BLOCKED_REASON_ORDER)
def test_every_capacity_block_reason_aborts_before_provider_or_writer(
    tmp_path, monkeypatch, reason
) -> None:
    """Every frozen capacity reason follows the same no-truncate real-run
    boundary; none may be demoted to an ordinary provider action."""
    (tmp_path / "main.py").write_text(
        _large_python_source(120), encoding="utf-8", newline=""
    )
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            SplitCapacityBlocked(kwargs["rel_path"], reason)
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail(f"{reason} block created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_args, **_kwargs: pytest.fail(f"{reason} block constructed a writer"),
    )

    with pytest.raises(ConfigError, match=reason) as blocked:
        run_pipeline(tmp_path, config)

    assert "Split never falls back to truncation" in str(blocked.value)
    assert not (tmp_path / "docs").exists()


def test_new_capacity_block_cannot_be_bypassed_by_reuse_and_cli_exits_two(
    tmp_path, monkeypatch, capsys
) -> None:
    """D8: feasibility is recalculated before reuse. Even a current completed
    split record cannot hide a newly blocked plan; the real run and CLI both
    preserve prior stable output and fail before writer/provider creation."""
    source = _large_python_source(120)
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )
    first = run_pipeline(tmp_path, config)
    assert first["checked"] == 1

    stable_path = tmp_path / "docs" / "codedoc.json"
    stable_bytes = stable_path.read_bytes()
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    assert not recovery_path.exists()

    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            SplitCapacityBlocked(kwargs["rel_path"], "chunk-cap")
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("new capacity block created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "new capacity block constructed a writer"
        ),
    )

    with pytest.raises(ConfigError, match=r"main\.py \(chunk-cap\)"):
        run_pipeline(tmp_path, config)

    assert stable_path.read_bytes() == stable_bytes
    assert not recovery_path.exists()

    # Exercise the public real-run exit contract against the same still-valid
    # completed record. The config file is not a supported source extension.
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(config), encoding="utf-8", newline=""
    )
    from codedoc.cli.cli import run_cli

    capsys.readouterr()
    assert run_cli([str(tmp_path)]) == 2
    stderr = capsys.readouterr().err
    assert "main.py (chunk-cap)" in stderr
    assert stable_path.read_bytes() == stable_bytes
    assert not recovery_path.exists()


def test_division_internal_defect_aborts_the_whole_run_uncaught(
    tmp_path, monkeypatch
) -> None:
    """D8: a genuine DivisionInternalDefect is a programming-invariant failure,
    not a per-file outcome. It propagates uncaught out of planning and aborts
    the whole run — dry or real — before any provider or writer side effect,
    even when another ordinary file in the same run would otherwise succeed."""
    source = _large_python_source()
    (tmp_path / "big.py").write_bytes(source.encode("utf-8"))
    (tmp_path / "main.py").write_text(
        "import big\n\ndef tiny():\n    return 1\n", encoding="utf-8", newline=""
    )
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **_kwargs: (_ for _ in ()).throw(
            DivisionInternalDefect("forced division invariant")
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("division defect created a provider"),
    )

    with pytest.raises(DivisionInternalDefect, match="forced division invariant"):
        run_pipeline(tmp_path, {**config, "dry_run": True})

    assert not (tmp_path / "docs").exists()

    with pytest.raises(DivisionInternalDefect, match="forced division invariant"):
        run_pipeline(tmp_path, config)

    assert not (tmp_path / "docs" / "codedoc.json").exists()
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_division_internal_defect_leaves_prior_state_untouched_and_unblocked(
    tmp_path, monkeypatch
) -> None:
    """A defect discovered on a later run must not corrupt or block runs
    around it: prior stable output from an earlier successful run survives
    byte-for-byte, and a subsequent differently-configured run is not blocked
    by any residue (because the aborted run never wrote anything)."""
    source = _large_python_source()
    (tmp_path / "big.py").write_bytes(source.encode("utf-8"))
    (tmp_path / "main.py").write_text(
        "import big\n\ndef tiny():\n    return 1\n", encoding="utf-8", newline=""
    )
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )
    first = run_pipeline(tmp_path, config)
    assert first["checked"] == 2
    assert first["failed"] == 0
    stable_bytes = (tmp_path / "docs" / "codedoc.json").read_bytes()
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    assert not recovery_path.exists()

    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **_kwargs: (_ for _ in ()).throw(
            DivisionInternalDefect("forced division invariant")
        ),
    )
    with pytest.raises(DivisionInternalDefect):
        run_pipeline(tmp_path, {**config, "force_files": ["big.py"]})

    assert (tmp_path / "docs" / "codedoc.json").read_bytes() == stable_bytes
    assert not recovery_path.exists()

    monkeypatch.undo()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda provider_config: attest_provider_execution(
            SmartFake(),
            provider_config,
        ),
    )
    later = run_pipeline(tmp_path, {**config, "large_file_strategy": "truncate"})

    assert later["failed"] == 0
    assert not recovery_path.exists()


def test_rejected_recovery_partial_does_not_force_retention(
    tmp_path, monkeypatch
) -> None:
    """Parsed partials are not automatically retainable.

    ``SafeWriter.load()`` can import every structurally readable partial, while
    planning accepts only the subset passing ``validate_node_for_tree()``. The
    distinguishing case is a stale partial for a path that is *not* recorded
    this run — an unchanged/reused file never calls ``record()``, so nothing
    ever pops the checkpoint. Seeding the writer from every parsed partial would
    therefore keep ``crash_recovery.json`` alive permanently; seeding from the
    pipeline-authorized retention set removes it.
    """
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )
    first = run_pipeline(tmp_path, config)
    assert first["checked"] == 1

    # Rebuild an in-progress envelope: the completed record plus a stale,
    # execution-incompatible partial for that same now-unchanged path.
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, max_content_chars=2000)
    stale_hash = "9" * 64
    provider_identity = _provider_identity_for(tmp_path, config)
    stale = _one_leaf_completed_tree_state(
        division, tree, provider_identity=provider_identity, content_hash=stale_hash
    )
    stable_record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path,
        "json",
        "main.py",
        {},
        build_recovery_identity(
            project_root=tmp_path,
            json_target=tmp_path / "docs" / "codedoc.json",
            md_target=None,
            entry_file="main.py",
            documentation_scope="entry",
            analysis_mode="single",
            analysis_revision=ANALYSIS_REVISION,
            large_file_strategy="split",
        ),
    )
    writer.load(preloaded={"main.py": stable_record})
    for node in stale.nodes:
        writer.record_tree_node("main.py", node)
    assert recovery_path.exists()

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("unchanged file must not call a provider"),
    )

    second = run_pipeline(tmp_path, config)

    # The file is unchanged, so record() never runs and never pops the stale
    # checkpoint. It must still not keep the recovery file alive.
    assert second["checked"] == 0
    assert second["skipped"] == 1
    assert second["split_restored_complete_chunks"] == 0
    assert not recovery_path.exists()
