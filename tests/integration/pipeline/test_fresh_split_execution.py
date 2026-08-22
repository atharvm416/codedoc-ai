"""Public split execution contracts: real fresh execution, node checkpoint
writing, same-path completed reuse, and node-keyed partial recovery."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading

import pytest

from codedoc.core.execution import _process_one_file_with_retries
from codedoc.core.file_division import (
    build_division_plan,
    build_reduction_tree,
    leaf_execution_identity,
    leaf_input_digest,
    provider_execution_identity,
    tree_node_state,
)
from codedoc.core.loader import load_config
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import build_recovery_identity
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import LLMError, UnrecoverableProviderError
from tests.support.execution_requests import make_execution_request
from tests.support.provider_failures import provider_failure_error
from tests.support.providers import SmartFake


def _large_python_source(lines: int = 220) -> str:
    return "".join(f"def fn_{index}():\n    return {index}\n\n" for index in range(lines))


def _config() -> dict:
    return {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "max_parallel_files": 1,
        "file_retry_attempts": 0,
        "propagate_changes": False,
    }


def test_real_split_writes_node_checkpoints_and_second_run_is_zero_call_reuse(
    tmp_path, monkeypatch
) -> None:
    """Current release: split execution checkpoints every node as it completes
    (node_checkpoints=True), and an unchanged second run reuses the completed
    record with zero provider calls (completed_reuse=True) rather than
    re-executing fresh -- the opposite of the retired 0.14.1 fresh-only
    contract this test used to verify."""
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    providers: list[SmartFake] = []
    checkpointed_nodes: list[str] = []
    from codedoc.core.safe_writer import SafeWriter

    original_record_tree_node = SafeWriter.record_tree_node

    def recording_record_tree_node(self, rel_path, node, *, reduction_tree_digest):
        checkpointed_nodes.append(node.node_id)
        return original_record_tree_node(
            self, rel_path, node, reduction_tree_digest=reduction_tree_digest
        )

    def provider_factory(_config):
        provider = SmartFake()
        providers.append(provider)
        return provider

    monkeypatch.setattr("codedoc.pipeline.create_provider", provider_factory)
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.record_tree_node",
        recording_record_tree_node,
    )

    first = run_pipeline(tmp_path, _config())
    assert len(checkpointed_nodes) > 0, "the current release must checkpoint split nodes"
    second = run_pipeline(tmp_path, _config())

    assert first["checked"] == 1
    assert first["reused"] == 0
    assert first["resumed"] == 0
    # The unchanged second run is a completed-record reuse: it does not
    # re-execute (not "checked"), makes zero provider calls, and is not
    # double-counted as a node-level "resume" (that counter is for a
    # genuinely interrupted/partial file, not a fully completed one).
    assert second["checked"] == 0
    assert second["skipped"] == 1
    assert len(providers) == 1, "an unchanged completed split file must make zero provider calls"

    payload = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8"))
    record = payload["files"][0]
    assert "_split_reuse_contract" not in record
    assert record["_large_file_identity"].startswith("large-file-v")
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()


def test_under_threshold_split_selection_keeps_ordinary_reuse(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "main.py").write_text(
        "def main():\n    return 1\n", encoding="utf-8", newline=""
    )
    providers: list[SmartFake] = []

    def provider_factory(_config):
        provider = SmartFake()
        providers.append(provider)
        return provider

    monkeypatch.setattr("codedoc.pipeline.create_provider", provider_factory)

    first = run_pipeline(tmp_path, _config())
    second = run_pipeline(tmp_path, _config())

    assert first["checked"] == 1
    assert second["checked"] == 0
    assert second["skipped"] == 1
    assert second["documentation_calls_planned"] == 0
    assert len(providers) == 1
    payload = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert "_large_file_identity" not in payload["files"][0]
    assert "_split_reuse_contract" not in payload["files"][0]


def test_completed_split_reuse_does_not_propagate_to_unchanged_dependents(
    tmp_path, monkeypatch
) -> None:
    """Current release: an unchanged completed split file is itself reused
    (zero calls), and — exactly as under the retired fresh-only policy — that
    routing decision alone must still never propagate a redo to its unchanged
    dependent. Both files are skipped on the identical second run."""
    (tmp_path / "large.py").write_text(
        _large_python_source(), encoding="utf-8", newline=""
    )
    (tmp_path / "main.py").write_text(
        "import large\n\ndef main():\n    return large.fn_0()\n",
        encoding="utf-8",
        newline="",
    )
    providers: list[SmartFake] = []

    def provider_factory(_config):
        provider = SmartFake()
        providers.append(provider)
        return provider

    monkeypatch.setattr("codedoc.pipeline.create_provider", provider_factory)
    config = {
        **_config(),
        "documentation_scope": "all",
        "propagate_changes": True,
    }

    first = run_pipeline(tmp_path, config)
    second = run_pipeline(tmp_path, config)

    assert first["file_documentation_calls_planned"] == 1
    assert second["file_documentation_calls_planned"] == 0
    assert second["checked"] == 0
    assert second["skipped"] == 2
    assert len(providers) == 1, "the unchanged second run must make zero provider calls"


def test_fresh_sequential_retry_observes_stop_before_next_leaf(tmp_path) -> None:
    source = _large_python_source()
    request = make_execution_request(
        tmp_path,
        "main.py",
        source,
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    tree = build_reduction_tree(
        plan,
        max_content_chars=2000,
        language="python",
        imports=(),
    )
    stop_event = threading.Event()

    class StoppingOrchestrator:
        def __init__(self) -> None:
            self.calls = 0

        def process_leaf_chunk(self, _request):
            self.calls += 1
            stop_event.set()
            return {"description": "first leaf"}

        def process_reduction_node(self, _request):
            pytest.fail("reduction began after cancellation")

        def synthesize_divided_file(self, *_args):
            pytest.fail("synthesis began after cancellation")

    orchestrator = StoppingOrchestrator()
    with pytest.raises(concurrent.futures.CancelledError):
        _process_one_file_with_retries(
            request,
            orchestrator,
            retry_attempts=2,
            division_plan=plan,
            reduction_tree=tree,
            stop_event=stop_event,
            split_execution_mode="fresh",
        )

    assert orchestrator.calls == 1


def test_fresh_leaf_retry_repeats_only_the_failed_logical_call(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    division = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )

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
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    stats = run_pipeline(
        tmp_path,
        {**_config(), "file_retry_attempts": 1},
    )

    assert provider.leaf_prompts[1] == provider.leaf_prompts[2]
    assert provider.leaf_prompts.count(provider.leaf_prompts[0]) == 1
    assert provider.leaf_prompts.count(provider.leaf_prompts[1]) == 2
    assert len(provider.leaf_prompts) == len(division.chunks) + 1
    assert stats["additional_attempts"] == 1
    assert stats["failed_calls"] == 1


def test_fresh_synthesis_retry_does_not_repeat_completed_tree(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    division = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )

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
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    stats = run_pipeline(
        tmp_path,
        {**_config(), "file_retry_attempts": 1},
    )

    assert len(provider.leaf_prompts) == len(division.chunks)
    assert len(set(provider.leaf_prompts)) == len(division.chunks)
    assert len(provider.synthesis_prompts) == 2
    assert provider.synthesis_prompts[0] == provider.synthesis_prompts[1]
    assert stats["additional_attempts"] == 1
    assert stats["failed_calls"] == 1


def test_fresh_rate_limit_step_down_keeps_run_local_split_progress(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "small.py").write_text(
        "def small_value():\n    return 1\n", encoding="utf-8", newline=""
    )
    source = _large_python_source(120)
    (tmp_path / "large.py").write_text(source, encoding="utf-8", newline="")
    division = build_division_plan(
        rel_path="large.py",
        language="python",
        content=source,
        source_budget_chars=2000,
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
                    raise provider_failure_error("openai", "provider-rate-limited", status=429, limit_type="tpm")
            return super().complete_json(prompt, system)

    provider = RateLimitSecondLargeLeaf()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda _seconds: None)

    stats = run_pipeline(
        tmp_path,
        {
            **_config(),
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "max_parallel_files": 2,
            "file_retry_attempts": 0,
            "rate_limit_adaptive": True,
            "rate_limit_backoff_s": 0,
            "respect_retry_after": False,
        },
    )

    assert stats["checked"] == 2
    assert len(provider.large_leaf_prompts) == len(division.chunks) + 1
    assert provider.large_leaf_prompts.count(provider.large_leaf_prompts[0]) == 1
    assert provider.large_leaf_prompts.count(provider.large_leaf_prompts[1]) == 2
    assert stats["additional_attempts"] == 1


def test_split_dry_run_keeps_ordinary_recovery_and_matches_real_plan(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "a_small.py").write_text(
        "def small_value():\n    return 1\n", encoding="utf-8", newline=""
    )
    (tmp_path / "z_large.py").write_text(
        "import a_small\n\n" + _large_python_source(),
        encoding="utf-8",
        newline="",
    )
    config = {
        **_config(),
        "entry_file": None,
        "auto_entry_candidates": [],
        "documentation_scope": "all",
    }

    class StopOnLargeLeaf(SmartFake):
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            if (
                "This is one bounded fragment of a larger" in prompt
                and "File: z_large.py" in prompt
            ):
                self.doc_calls += 1
                raise provider_failure_error(
                    "openai", "provider-quota-exhausted", status=429
                )
            return super().complete_json(prompt, system)

    failing_provider = StopOnLargeLeaf()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: failing_provider
    )
    with pytest.raises(UnrecoverableProviderError):
        run_pipeline(tmp_path, config)

    recovery_path = tmp_path / "codedoc" / "crash_recovery.json"
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert [record["path"] for record in recovery["files"]] == ["a_small.py"]

    dry = run_pipeline(tmp_path, {**config, "dry_run": True})
    real_provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: real_provider
    )
    real = run_pipeline(tmp_path, config)

    comparable = (
        "documentation_calls_planned",
        "file_documentation_calls_planned",
        "unit_documentation_calls_planned",
        "file_reduction_calls_planned",
        "synthesis_calls_planned",
        "split_unit_consolidation_calls_planned",
        "split_general_reduction_calls_planned",
        "split_final_synthesis_calls_planned",
        "call_manifest_digest",
    )
    assert dry["would_resume"] == real["resumed"] == 1
    assert {key: dry[key] for key in comparable} == {
        key: real[key] for key in comparable
    }
    assert dry["file_documentation_calls_planned"] == 0
    assert real_provider.doc_calls == real["documentation_calls_planned"]


def test_current_release_resumes_an_existing_valid_node_checkpoint(
    tmp_path, monkeypatch
) -> None:
    """Current release: a pre-existing, dependency-valid node checkpoint is
    consumed by a real run rather than blocked -- the opposite of the retired
    0.14.1 fresh-only ConfigError this test used to verify. Only the
    remaining unpaid leaf/reduction/final nodes reach the provider."""
    source = _large_python_source()
    source_path = tmp_path / "main.py"
    source_path.write_text(source, encoding="utf-8", newline="")
    config = load_config(tmp_path, _config())
    output_dir = tmp_path / "codedoc"
    recovery_path = output_dir / "crash_recovery.json"
    identity = build_recovery_identity(
        project_root=tmp_path.resolve(),
        json_target=output_dir / "codedoc.json",
        md_target=None,
        entry_file="main.py",
        documentation_scope="entry",
        analysis_mode="single",
        analysis_revision=ANALYSIS_REVISION,
        large_file_strategy="split",
    )
    file_map = {
        "main.py": {
            "path": source_path,
            "rel_path": "main.py",
            "language": "python",
        }
    }
    writer = SafeWriter(recovery_path, "json", "main.py", file_map, identity)
    writer.set_queue_order(["main.py"])
    writer.initialize_empty()

    plan = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    tree = build_reduction_tree(
        plan,
        max_content_chars=2000,
        language="python",
        imports=(),
    )
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    chunk = plan.chunks[0]
    writer.record_tree_node(
        "main.py",
        tree_node_state(
            node_id=chunk.chunk_id,
            node_type="leaf",
            rel_path="main.py",
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=leaf_input_digest(
                rel_path=plan.rel_path,
                language="python",
                chunk=chunk,
                unit_indexes=plan.unit_positions(chunk),
                unit_count=len(plan.units),
            ),
            execution_identity_digest=leaf_execution_identity(
                rel_path="main.py",
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                provider_identity=provider_execution_identity(config),
                chunk=chunk,
            ),
            unit_id=None,
            child_ids=(),
            coverage_leaf_ids=(chunk.chunk_id,),
            # Must match exactly what the live leaf cleaner would produce, so
            # validate_node_for_tree accepts it as dependency-valid rather
            # than rejecting a checkpoint whose stored result no longer
            # matches the current cleaner/required-field contract.
            result={
                "description": "already paid",
                "chunk_id": chunk.chunk_id,
                "unit_id": chunk.unit_id,
            },
        ),
        reduction_tree_digest=tree.tree_digest,
    )

    leaf_prompts: list[str] = []
    provider = SmartFake()
    original_complete_json = provider.complete_json

    def counting_complete_json(prompt, system=""):
        if "This is one bounded fragment of a larger" in prompt:
            leaf_prompts.append(prompt)
        return original_complete_json(prompt, system)

    provider.complete_json = counting_complete_json
    provider_created = []
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda *_a, **_k: (provider_created.append(True), provider)[1],
    )

    stats = run_pipeline(tmp_path, _config())

    # The provider IS created (force bypasses nothing here; this is an
    # ordinary resumable run), and the pre-checkpointed leaf is never
    # re-requested: only the remaining leaves reach the provider.
    assert provider_created == [True]
    assert len(leaf_prompts) == len(plan.chunks) - 1
    assert stats["checked"] == 1
    assert not recovery_path.exists()
    assert (output_dir / "codedoc.json").exists()
