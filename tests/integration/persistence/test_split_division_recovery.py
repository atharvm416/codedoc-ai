"""Node-keyed split recovery (schema version 4 — D11/D12/section 9/section 12).

Every leaf, unit-consolidation, general-reduction, and final-synthesis node is
checkpointed independently and keyed by its own node ID; resuming a run
re-executes only the nodes missing from the recovered `SplitTreeState`, never
a whole file and never more than the unpaid remainder.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading

import pytest
from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import CorrectionLedger
from codedoc.core.document import read_codedoc_document
import codedoc.core.execution as execution
from codedoc.core.execution import (
    _process_divided_file,
    _process_one_file_with_retries,
)
import codedoc.core.file_division as file_division
import codedoc.core.record_meta as record_meta
import codedoc.core.safe_writer as safe_writer_mod
from codedoc.core.file_division import (
    SPLIT_PARTIAL_SCHEMA_VERSION,
    SplitTreeState,
    build_division_plan,
    build_reduction_tree,
    tree_node_state,
)
from codedoc.cli.cli import run_cli
from codedoc.core.execution_model import CallManifestTracker, build_call_manifest
from codedoc.core.project_view import json_from_view
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import build_recovery_identity
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import (
    ConfigError,
    InsufficientSourceError,
    LiveBackupWriteError,
    LLMError,
    UnrecoverableProviderError,
)
from tests.support.execution_requests import make_execution_request
from tests.support.fixture_paths import FIXTURES_ROOT
from tests.support.provider_failures import provider_failure_error
from tests.support.providers import SmartFake
from tests.support.run_metadata_cases import _view as run_metadata_view
from tests.support.structure_extra import requires_structure_pack

_CONTENT_HASH = "0" * 64
_PLAN_DIGEST = "division-plan:" + "1" * 64
_TREE_DIGEST = "reduction-tree:" + "2" * 64
_EXECUTION_ID = "division-execution:" + "3" * 64
_CHUNK_ID = "chunk_" + "4" * 64
_INPUT_DIGEST = "leaf-input:" + "5" * 64


def _leaf_node(rel_path: str):
    return tree_node_state(
        node_id=_CHUNK_ID,
        node_type="leaf",
        rel_path=rel_path,
        content_hash=_CONTENT_HASH,
        division_plan_digest=_PLAN_DIGEST,
        execution_identity_digest=_EXECUTION_ID,
        input_digest=_INPUT_DIGEST,
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(_CHUNK_ID,),
        result={"description": "ok"},
    )


def test_split_partial_is_internal_to_recovery_metadata(tmp_path) -> None:
    writer = SafeWriter(
        tmp_path / "crash_recovery.json",
        "json",
        None,
        {},
        {"version": 1},
    )
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )

    text = writer.path.read_text(encoding="utf-8")

    assert '"partial_files"' in text
    assert '"files": []' in text
    assert "src/large.py" in text


def test_schema_four_recovery_retains_valid_nodes_beside_malformed_siblings(
    tmp_path,
) -> None:
    writer = SafeWriter(
        tmp_path / "crash_recovery.json",
        "json",
        None,
        {},
        {"version": 1},
    )
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )
    payload = json.loads(writer.path.read_text(encoding="utf-8"))
    nodes = payload["_codedoc"]["partial_files"]["src/large.py"]["nodes"]
    valid_node = nodes[0]
    nodes.append({**valid_node, "node_id": "chunk_" + "5" * 64, "content_hash": "9" * 64})
    nodes.append({"node_id": "chunk_" + "6" * 64, "node_type": "leaf", "child_ids": 7})
    nodes.append(
        {
            **valid_node,
            "node_id": "chunk_" + "7" * 64,
            "reduction_tree_digest": _TREE_DIGEST,
        }
    )
    writer.path.write_text(json.dumps(payload), encoding="utf-8")

    document = read_codedoc_document(writer.path)

    assert len(document.partial_files) == 1
    assert tuple(document.partial_files[0].by_id()) == (_CHUNK_ID,)
    quarantine = document.partial_files[0].quarantine_by_id()
    assert set(quarantine) == {
        "chunk_" + "5" * 64,
        "chunk_" + "6" * 64,
        "chunk_" + "7" * 64,
    }
    assert quarantine["chunk_" + "5" * 64].reason == "stale-identity"
    assert quarantine["chunk_" + "6" * 64].reason == "live-schema-mismatch"
    assert quarantine["chunk_" + "7" * 64].reason == "live-schema-mismatch"


def test_released_schema_three_container_blocks_before_writer_or_provider(
    tmp_path, monkeypatch,
) -> None:
    """Section 20A items 4 and 6: this module's own named schema-3
    blocked-run assertion, proving the actual blocked pipeline/run boundary
    -- not merely that `read_codedoc_document` alone raises -- alongside
    `tests/integration/persistence/test_cross_version_split_state.py`'s own
    fixture-driven coverage of the same boundary (not substituted for by
    it: this file proves it independently, in this file's own established
    idiom for a real blocked `run_pipeline` call, matching
    `test_legacy_v1_split_partial_fails_closed_with_migration_guidance`
    above). Uses the frozen, real released-`0.14.2`-produced container
    (section 4A) via the canonical `FIXTURES_ROOT / "split_state"` path,
    not a hand-built placeholder."""
    fixture = json.loads(
        (FIXTURES_ROOT / "split_state" / "partial_schema3_0_14_2.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["schema_version"] == 3

    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "large.py").write_bytes(source.encode("utf-8"))
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path,
        "json",
        "src/large.py",
        {},
        build_recovery_identity(
            project_root=tmp_path,
            json_target=tmp_path / "docs" / "codedoc.json",
            md_target=None,
            entry_file="src/large.py",
            documentation_scope="entry",
            analysis_mode="single",
            analysis_revision=ANALYSIS_REVISION,
            large_file_strategy="split",
        ),
    )
    writer.initialize_empty()
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload["_codedoc"]["partial_files"] = {"src/large.py": fixture}
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    original = recovery_path.read_bytes()

    monkeypatch.setattr(
        "codedoc.pipeline.build_pipeline_plan",
        lambda *a, **k: pytest.fail(
            "released schema-3 container reached build_pipeline_plan"
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter",
        lambda *a, **k: pytest.fail(
            "released schema-3 container reached SafeWriter construction"
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail(
            "released schema-3 container reached provider construction"
        ),
    )

    with pytest.raises(ConfigError, match="unsupported"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "src/large.py",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    assert recovery_path.read_bytes() == original
    assert not (tmp_path / "docs" / "codedoc.json").exists()

    # Direct proof that the reader itself never returns a recovery state
    # for this document, independent of the full run_pipeline call chain.
    with pytest.raises(ConfigError, match="unsupported"):
        read_codedoc_document(recovery_path, include_partial_files=True)
    assert recovery_path.read_bytes() == original


def test_duplicate_key_json_is_fatal_and_preserves_the_recovery_bytes(tmp_path) -> None:
    writer = SafeWriter(tmp_path / "crash_recovery.json", "json", None, {}, {"version": 1})
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )
    text = writer.path.read_text(encoding="utf-8")
    needle = f'"node_id": "{_CHUNK_ID}"'
    duplicate = f'{needle},\n          {needle}'
    assert text.count(needle) == 1
    writer.path.write_text(text.replace(needle, duplicate), encoding="utf-8")
    before = writer.path.read_bytes()

    with pytest.raises(ConfigError, match="duplicate JSON key"):
        read_codedoc_document(writer.path)

    assert writer.path.read_bytes() == before


def test_unknown_schema_four_container_field_is_fatal(tmp_path) -> None:
    writer = SafeWriter(tmp_path / "crash_recovery.json", "json", None, {}, {"version": 1})
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )
    payload = json.loads(writer.path.read_text(encoding="utf-8"))
    payload["_codedoc"]["partial_files"]["src/large.py"]["future_field"] = True
    writer.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown or missing field"):
        read_codedoc_document(writer.path)


def test_duplicate_schema_four_node_id_is_fatal(tmp_path) -> None:
    writer = SafeWriter(tmp_path / "crash_recovery.json", "json", None, {}, {"version": 1})
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )
    payload = json.loads(writer.path.read_text(encoding="utf-8"))
    nodes = payload["_codedoc"]["partial_files"]["src/large.py"]["nodes"]
    nodes.append(dict(nodes[0]))
    writer.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate split-partial node ID"):
        read_codedoc_document(writer.path)


def test_completed_record_clears_same_path_split_partial(tmp_path) -> None:
    writer = SafeWriter(tmp_path / "crash_recovery.json", "json", None, {"src/large.py": {}})
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )

    writer.record(
        "src/large.py",
        {"file_path": "src/large.py", "language": "python", "description": "done"},
        _CONTENT_HASH,
    )

    assert writer.get_tree_state("src/large.py") is None
    assert '"partial_files"' not in writer.path.read_text(encoding="utf-8")


def test_flush_refuses_partial_state_for_a_file_completed_this_run(tmp_path) -> None:
    """The flush itself fails closed on a completed/partial collision.

    ``record()`` and ``record_tree_node()`` each guard the transition, so this
    drives the invariant directly to prove the serializer refuses the state
    instead of publishing an obsolete checkpoint alongside its own completed
    record — and that the prior recovery file survives the refusal intact.
    """
    writer = SafeWriter(tmp_path / "crash_recovery.json", "json", None, {})
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )
    intact = writer.path.read_bytes()

    # Simulate the forbidden state the public guards prevent.
    writer._recorded_this_run.add("src/large.py")

    with pytest.raises(LiveBackupWriteError) as blocked:
        writer._flush_locked()

    assert "src/large.py" in str(blocked.value)
    assert writer.path.read_bytes() == intact


def test_flush_allows_a_preloaded_stable_record_beside_a_current_partial(
    tmp_path,
) -> None:
    """A preloaded record is not a completion of *this* run, so it is allowed."""
    writer = SafeWriter(
        tmp_path / "crash_recovery.json",
        "json",
        None,
        {},
    )
    writer.load(preloaded={"src/large.py": {"path": "src/large.py"}})
    writer.record_tree_node(
        "src/large.py", _leaf_node("src/large.py"), reduction_tree_digest=_TREE_DIGEST
    )

    assert '"partial_files"' in writer.path.read_text(encoding="utf-8")


def test_resume_runs_only_unpaid_nodes_and_then_synthesis(tmp_path) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    request = make_execution_request(
        tmp_path,
        "src/large.py",
        source,
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path=request.rel_path,
        language=request.language,
        content=source,
        source_budget_chars=2000,
    )
    # Pair the tree with the request's carried synthesis budget (automatic
    # 12,000 floor), as production planning does; the deprecated alias would
    # carry the raw source value and be rejected by the execution guard.
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=request.context.synthesis_manifest_chars
    )
    reduction_total = len(tree.unit_consolidation_nodes) + len(tree.general_nodes)
    assert len(plan.chunks) >= 2
    assert reduction_total >= 1
    writer = SafeWriter(
        tmp_path / "docs" / "crash_recovery.json",
        "json",
        None,
        {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
    )

    class Interrupted:
        def __init__(self) -> None:
            self.leaf_calls = 0

        def process_leaf_chunk(self, _request):
            self.leaf_calls += 1
            if self.leaf_calls == 2:
                raise RuntimeError("interrupted")
            return {"description": f"leaf {self.leaf_calls}"}

    with pytest.raises(RuntimeError, match="interrupted"):
        _process_divided_file(
            request, plan, tree, "test-provider", Interrupted(), writer
        )

    checkpoint = writer.get_tree_state(request.rel_path)
    assert checkpoint is not None
    completed = checkpoint.by_id()
    assert len(completed) == 1
    assert plan.chunks[0].chunk_id in completed

    class Resumed:
        def __init__(self) -> None:
            self.leaf_calls = 0
            self.reduction_calls = 0
            self.synthesis_calls = 0

        def process_leaf_chunk(self, _request):
            self.leaf_calls += 1
            return {"description": f"resumed leaf {self.leaf_calls}"}

        def process_reduction_node(self, _request):
            self.reduction_calls += 1
            return {"narrative": f"resumed reduction {self.reduction_calls}"}

        def synthesize_divided_file(self, _request, _digest, _manifest, terminology_source=""):
            self.synthesis_calls += 1
            return {"description": "complete"}

    resumed = Resumed()
    result = _process_divided_file(
        request, plan, tree, "test-provider", resumed, writer
    )

    assert resumed.leaf_calls == len(plan.chunks) - 1
    assert resumed.reduction_calls == reduction_total
    assert resumed.synthesis_calls == 1
    assert "division" not in result
    assert "documentation_units" not in result
    assert result["_large_file_identity"].startswith("large-file-v3:")
    assert tree.final_node.node_id in writer.get_tree_state(request.rel_path).by_id()


@pytest.mark.parametrize("completed_before_terminal", [1, 2])
def test_terminal_split_failure_preserves_stable_output_and_resumes_only_unpaid_leaves(
    tmp_path, completed_before_terminal
) -> None:
    # Named function declarations, not bare top-level statements: see
    # test_split_division.py's _large_python_source for why (syntax-mode
    # "gap" unit merging produces a wildly different chunk count for many
    # bare statements). 220 functions clear every capacity cap under both
    # parsing modes at this budget (lexical packs them into a handful of
    # chunks; syntax treats each as its own unit) while comfortably exceeding
    # completed_before_terminal (max 2).
    source = "\n".join(f"def fn_{index}(): return {index}" for index in range(220)) + "\n"
    request = make_execution_request(
        tmp_path,
        "src/large.py",
        source,
        analysis_mode="single",
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path=request.rel_path,
        language=request.language,
        content=source,
        source_budget_chars=2000,
    )
    # Pair the tree with the request's carried synthesis budget (automatic
    # 12,000 floor), as production planning does; the deprecated alias would
    # carry the raw source value and be rejected by the execution guard.
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=request.context.synthesis_manifest_chars
    )
    assert len(plan.chunks) > completed_before_terminal
    stable = {
        "path": request.rel_path,
        "language": "python",
        "description": "prior stable output",
    }
    writer = SafeWriter(
        tmp_path / "docs" / "crash_recovery.json",
        "json",
        None,
        {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
    )
    writer.load({request.rel_path: stable})

    class TerminalAfterPrefix:
        class _LLM:
            provider_name = "openai"

        llm = _LLM()

        def __init__(self) -> None:
            self.completed = 0
            self.reduction_calls = 0
            self.synthesis_calls = 0

        def process_leaf_chunk(self, chunk_request):
            if self.completed == completed_before_terminal:
                raise provider_failure_error(
                    "openai", "provider-quota-exhausted", status=429
                )
            self.completed += 1
            return {"description": f"chunk {chunk_request.chunk_id}"}

        def process_reduction_node(self, _request):
            self.reduction_calls += 1
            return {"narrative": "combined"}

        def synthesize_divided_file(self, _request, _digest, _manifest, terminology_source=""):
            self.synthesis_calls += 1
            return {"description": "complete"}

    interrupted = TerminalAfterPrefix()
    with pytest.raises(UnrecoverableProviderError):
        _process_one_file_with_retries(
            request,
            interrupted,
            retry_attempts=2,
            division_plan=plan,
            reduction_tree=tree,
            provider_identity="test-provider",
            recorder=writer,
            split_execution_mode="recovery",
        )

    checkpoint = writer.get_tree_state(request.rel_path)
    assert checkpoint is not None
    completed_ids = checkpoint.by_id()
    assert len(completed_ids) == completed_before_terminal
    assert all(node.node_type == "leaf" for node in completed_ids.values())
    assert interrupted.completed == completed_before_terminal
    assert interrupted.reduction_calls == 0
    assert interrupted.synthesis_calls == 0
    assert writer.get_record(request.rel_path) == stable

    class CompleteDeterministically:
        def __init__(self) -> None:
            self.leaf_ids: list[str] = []
            self.reduction_calls = 0
            self.synthesis_calls = 0

        def process_leaf_chunk(self, chunk_request):
            self.leaf_ids.append(chunk_request.chunk_id)
            return {"description": f"chunk {chunk_request.chunk_id}"}

        def process_reduction_node(self, _request):
            self.reduction_calls += 1
            return {"narrative": "combined"}

        def synthesize_divided_file(self, _request, _digest, _manifest, terminology_source=""):
            self.synthesis_calls += 1
            return {"description": "complete"}

    resumed = CompleteDeterministically()
    resumed_result = _process_divided_file(
        request, plan, tree, "test-provider", resumed, writer
    )
    assert resumed.leaf_ids == [
        chunk.chunk_id for chunk in plan.chunks[completed_before_terminal:]
    ]
    assert resumed.synthesis_calls == 1

    uninterrupted_writer = SafeWriter(
        tmp_path / "uninterrupted" / "crash_recovery.json",
        "json",
        None,
        {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
    )
    uninterrupted = CompleteDeterministically()
    uninterrupted_result = _process_divided_file(
        request, plan, tree, "test-provider", uninterrupted, uninterrupted_writer
    )
    assert resumed_result == uninterrupted_result

    writer.record(request.rel_path, resumed_result, request.content_hash)
    assert writer.get_tree_state(request.rel_path) is None
    assert writer.get_record(request.rel_path)["description"] == "complete"


def test_stop_event_checkpoints_current_leaf_and_skips_remaining_provider_calls(
    tmp_path,
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    request = make_execution_request(
        tmp_path,
        "src/large.py",
        source,
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path=request.rel_path,
        language=request.language,
        content=source,
        source_budget_chars=2000,
    )
    # Pair the tree with the request's carried synthesis budget (automatic
    # 12,000 floor), as production planning does; the deprecated alias would
    # carry the raw source value and be rejected by the execution guard.
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=request.context.synthesis_manifest_chars
    )
    assert len(plan.chunks) >= 2
    writer = SafeWriter(
        tmp_path / "docs" / "crash_recovery.json",
        "json",
        None,
        {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
    )
    stop_event = threading.Event()

    class StopAfterFirstLeaf:
        def __init__(self) -> None:
            self.leaf_calls = 0

        def process_leaf_chunk(self, _request):
            self.leaf_calls += 1
            stop_event.set()
            return {"description": "paid leaf"}

    orchestrator = StopAfterFirstLeaf()
    with pytest.raises(concurrent.futures.CancelledError):
        _process_divided_file(
            request,
            plan,
            tree,
            "test-provider",
            orchestrator,
            writer,
            stop_event,
        )

    checkpoint = writer.get_tree_state(request.rel_path)
    assert checkpoint is not None
    assert orchestrator.leaf_calls == 1
    assert tuple(checkpoint.by_id()) == (plan.chunks[0].chunk_id,)


def test_stop_event_prevents_response_correction_provider_call(tmp_path) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    request = make_execution_request(
        tmp_path,
        "src/large.py",
        source,
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path=request.rel_path,
        language=request.language,
        content=source,
        source_budget_chars=2000,
    )
    # Pair the tree with the request's carried synthesis budget (automatic
    # 12,000 floor), as production planning does; the deprecated alias would
    # carry the raw source value and be rejected by the execution guard.
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=request.context.synthesis_manifest_chars
    )
    manifest = build_call_manifest(
        (),
        (request.rel_path,),
        "single",
        {request.rel_path: plan},
        {request.rel_path: tree},
    )
    tracker = CallManifestTracker(manifest)
    ledger = CorrectionLedger(True)
    stop_event = threading.Event()

    class StopWithInvalidResponse:
        provider_name = "StopWithInvalidResponse"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, _prompt, system=""):
            self.calls += 1
            if self.calls == 1:
                stop_event.set()
                return "{}"
            raise AssertionError("response correction reached the provider")

    provider = StopWithInvalidResponse()
    orchestrator = Orchestrator(
        provider,
        parallel=False,
        max_content_chars=2000,
        analysis_mode="single",
        response_correction_enabled=True,
        correction_ledger=ledger,
        call_tracker=tracker,
    )
    orchestrator.bind_stop_event(stop_event)
    writer = SafeWriter(
        tmp_path / "docs" / "crash_recovery.json",
        "json",
        None,
        {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
    )

    with pytest.raises(concurrent.futures.CancelledError):
        _process_divided_file(
            request,
            plan,
            tree,
            "test-provider",
            orchestrator,
            writer,
            stop_event,
        )

    assert provider.calls == 1
    assert writer.get_tree_state(request.rel_path) is None
    assert ledger.snapshot()["response_contract_failures"] == 1
    assert ledger.snapshot()["response_correction_calls_attempted"] == 0
    assert tracker.snapshot()["attempted_logical_calls"] == 1


def test_pipeline_terminal_chunk_failure_retains_resumable_checkpoint(
    tmp_path, monkeypatch
) -> None:
    """An exhausted leaf failure must not delete its own paid checkpoints.

    The existing terminal-failure coverage drives ``_process_divided_file`` and
    ``_process_descriptor_batch`` directly, so it proves a checkpoint is
    *written* but never that it *survives the end of a ``run_pipeline`` call*.
    """
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    plan = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    assert len(plan.chunks) >= 2
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "analysis_mode": "single",
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    original_leaf = Orchestrator.process_leaf_chunk
    attempted = {"count": 0}

    def fail_after_first_leaf(self, request):
        attempted["count"] += 1
        if attempted["count"] >= 2:
            raise LLMError("chunk provider failure exhausts this file")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_after_first_leaf)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )

    stats = run_pipeline(tmp_path, config)

    assert stats["checked"] == 0
    assert stats["failed"] == 1
    assert (tmp_path / "docs" / "codedoc.json").exists()
    assert recovery_path.exists()
    partials = json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]
    assert list(partials) == ["main.py"]
    leaf_nodes = [
        node for node in partials["main.py"]["nodes"]
        if node["node_type"] == "leaf"
    ]
    assert len(leaf_nodes) == 1

    resumed = {"count": 0}

    def counting_leaf(self, request):
        resumed["count"] += 1
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", counting_leaf)

    second = run_pipeline(tmp_path, config)

    assert second["checked"] == 1
    assert second["failed"] == 0
    assert second["resumed"] == 1
    assert second["split_restored_complete_chunks"] == 1
    assert resumed["count"] == len(plan.chunks) - 1
    assert not recovery_path.exists()
    record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert "division" not in record
    assert "documentation_units" not in record
    assert record["_large_file_identity"].startswith("large-file-v3:")
    assert record["description"]


def test_legacy_v1_split_partial_fails_closed_with_migration_guidance(
    tmp_path, monkeypatch
) -> None:
    """D11: a predecessor (schema version 1) ordered-prefix split partial is
    migration-readable only. It is never resumed, executed, or silently
    reinterpreted as node-keyed state — it blocks with a deterministic
    preserve-or-move-aside remedy, and prior stable output and the recovery
    file itself are left completely untouched."""
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
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
    writer.initialize_empty()

    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload["_codedoc"]["partial_files"] = {
        "main.py": {
            "schema_version": 1,
            "owner": "codedoc-ai",
            "rel_path": "main.py",
            "completed_chunks": [
                ["chunk_" + "a" * 58, '{"description": "legacy"}']
            ],
        }
    }
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    original_recovery = recovery_path.read_bytes()

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("legacy split partial created a provider"),
    )

    with pytest.raises(ConfigError) as blocked:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    message = str(blocked.value)
    assert "schema version 1" in message
    # Section 8: bounded, JSON-escaped rendering -- the one affected path is
    # a single ensure_ascii JSON string, with an exact total and a
    # full-stream digest, never the old raw single-quoted interpolation.
    assert json.dumps("main.py", ensure_ascii=True) in message
    assert "'main.py'" not in message
    assert "1 path" in message
    assert "sha256:" in message
    assert "crash_recovery.json" in message
    assert "predecessor" in message
    assert recovery_path.read_bytes() == original_recovery
    assert not (tmp_path / "docs" / "codedoc.json").exists()


@pytest.mark.parametrize(
    "malformation",
    ["unsupported-schema", "non-object-nodes", "foreign-owner", "invalid-digest"],
)
def test_malformed_schema_two_container_blocks_without_overwriting_recovery(
    tmp_path, monkeypatch, malformation
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
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
    writer.record_tree_node(
        "main.py", _leaf_node("main.py"), reduction_tree_digest=_TREE_DIGEST
    )
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    container = payload["_codedoc"]["partial_files"]["main.py"]
    if malformation == "unsupported-schema":
        # 2 is the dormant per-node-tree-digest-gated predecessor schema
        # (section 16): recognized only enough to fail closed, never executed.
        container["schema_version"] = 2
    elif malformation == "non-object-nodes":
        container["nodes"] = {}
    elif malformation == "foreign-owner":
        container["owner"] = "foreign"
    else:
        container["division_plan_digest"] = "division-plan:not-a-digest"
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    original_recovery = recovery_path.read_bytes()

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("malformed recovery created a provider"),
    )

    with pytest.raises(ConfigError) as blocked:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    assert "move 'crash_recovery.json' aside" in str(blocked.value)
    assert recovery_path.read_bytes() == original_recovery
    assert not (tmp_path / "docs" / "codedoc.json").exists()


@pytest.mark.parametrize("alias", ["src\\main.py", "src//main.py"])
def test_aliased_split_partial_paths_block_without_mutation_or_provider(
    tmp_path, monkeypatch, alias
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    source_path = tmp_path / "src" / "main.py"
    source_path.parent.mkdir()
    source_path.write_bytes(source.encode("utf-8"))
    output_dir = tmp_path / "docs"
    output_dir.mkdir()
    stable_path = output_dir / "codedoc.json"
    stable_view = run_metadata_view()
    stable_view["last_run"]["entry_file"] = "src/main.py"
    stable_view["files"][0]["path"] = "src/main.py"
    stable_path.write_text(json_from_view(stable_view), encoding="utf-8")
    stable_before = stable_path.read_bytes()
    recovery_path = output_dir / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path,
        "json",
        "src/main.py",
        {},
        build_recovery_identity(
            project_root=tmp_path,
            json_target=stable_path,
            md_target=None,
            entry_file="src/main.py",
            documentation_scope="entry",
            analysis_mode="single",
            analysis_revision=ANALYSIS_REVISION,
            large_file_strategy="split",
        ),
    )
    writer.record_tree_node(
        "src/main.py", _leaf_node("src/main.py"), reduction_tree_digest=_TREE_DIGEST
    )
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    partials = payload["_codedoc"]["partial_files"]
    partials[alias] = dict(partials["src/main.py"])
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    recovery_before = recovery_path.read_bytes()
    provider_creations = []
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: provider_creations.append(True),
    )

    with pytest.raises(ConfigError, match="aliased paths"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "src/main.py",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    assert provider_creations == []
    assert stable_path.read_bytes() == stable_before
    assert recovery_path.read_bytes() == recovery_before


def test_cross_plan_transition_preserves_then_transactionally_supersedes_recovery(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3 / 5.7, through the real ``run_pipeline`` / ``SafeWriter``
    path: a schema-4 checkpoint written by one run becomes a *cross-plan
    predecessor* for a later run whose reduction-tree digest moved (a
    ``REDUCTION_PACKING_REVISION`` advance -- same file content, same
    division-plan digest). The predecessor container is carried byte-for-byte
    while a replacement is incomplete, never overwritten by a new node
    checkpoint, preserved across a failed replacement together with the stable
    output, and only transactionally superseded -- with its recovery file
    removed -- after a clean replacement run completes. No paid predecessor
    node is silently deleted merely because it was detected as stale.
    """
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "analysis_mode": "single",
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    stable_path = tmp_path / "docs" / "codedoc.json"
    original_leaf = Orchestrator.process_leaf_chunk
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())

    # ---- Pass 1: interrupted run writes a genuine one-leaf checkpoint. ----
    first_calls = {"n": 0}

    def fail_after_first_leaf(self, request):
        first_calls["n"] += 1
        if first_calls["n"] >= 2:
            raise LLMError("interrupted after the first leaf")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_after_first_leaf)
    stats1 = run_pipeline(tmp_path, config)
    assert stats1["failed"] == 1
    assert recovery_path.exists()
    predecessor_partial = json.loads(recovery_path.read_text(encoding="utf-8"))[
        "_codedoc"
    ]["partial_files"]["main.py"]
    predecessor_node_ids = [node["node_id"] for node in predecessor_partial["nodes"]]
    assert len(predecessor_node_ids) == 1

    def _documented_paths():
        doc = json.loads(stable_path.read_text(encoding="utf-8"))
        return {record.get("path") for record in doc.get("files", [])}

    # main.py never completed, so it holds no published documentation record.
    assert "main.py" not in _documented_paths()

    # The exact pre-replacement bytes of BOTH persisted files.
    recovery_before = recovery_path.read_bytes()
    stable_before = stable_path.read_bytes()

    # ---- Advance the reduction-packing revision: the next run's tree digest
    # differs while content and the division-plan digest do not. ----
    bumped = "reduction-packing-v5-next"
    monkeypatch.setattr(file_division, "REDUCTION_PACKING_REVISION", bumped)
    if hasattr(record_meta, "REDUCTION_PACKING_REVISION"):
        monkeypatch.setattr(record_meta, "REDUCTION_PACKING_REVISION", bumped)

    # ---- Pass 2a: the failed/interrupted cross-plan replacement. ----
    def fail_every_leaf(self, request):
        raise LLMError("second run also interrupted")

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_every_leaf)
    stats2 = run_pipeline(tmp_path, config)
    assert stats2["failed"] == 1
    assert recovery_path.exists()

    # Exact-byte contract (section 6.3): a failed/interrupted fresh replacement
    # with carried predecessor state leaves the pre-run recovery file and the
    # pre-run stable output BYTE-IDENTICAL -- no wrapper churn, no telemetry
    # rewrite. Not json.loads equality, not selected-field equality, not
    # record-absence, not canonical-JSON equality.
    assert recovery_path.read_bytes() == recovery_before
    assert stable_path.read_bytes() == stable_before

    # Semantic guarantees retained:
    carried_partial = json.loads(recovery_path.read_text(encoding="utf-8"))[
        "_codedoc"
    ]["partial_files"]["main.py"]
    # predecessor partial remains present; its node IDs are exact; no current
    # replacement checkpoint was appended; no paid predecessor node vanished.
    assert carried_partial == predecessor_partial
    assert [node["node_id"] for node in carried_partial["nodes"]] == predecessor_node_ids
    assert len(carried_partial["nodes"]) == 1
    # The failed file published no replacement documentation record.
    assert "main.py" not in _documented_paths()

    # ---- Pass 2b: a clean replacement run. Carry state is transactionally
    # superseded and the recovery file is removed. ----
    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", original_leaf)
    stats3 = run_pipeline(tmp_path, config)
    assert stats3["checked"] == 1
    assert stats3["failed"] == 0
    assert not recovery_path.exists()
    record = json.loads(stable_path.read_text(encoding="utf-8"))["files"][0]
    assert record["_large_file_identity"].startswith("large-file-v3:")
    assert record["description"]


def test_mixed_run_defers_unrelated_work_until_cross_plan_replacement_succeeds(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3 mixed-file transaction boundary: a run that carries a
    cross-plan predecessor for one split file AND has a second reachable
    changed ordinary file treats the carried replacement as a prerequisite. If
    that replacement fails, the unrelated file is NOT attempted or charged, and
    BOTH the prior stable output and the recovery file stay BYTE-IDENTICAL. A
    later successful retry supersedes the carried state and only then processes
    the unrelated file normally, removing recovery on clean completion. No
    ``SafeWriter`` method changes; ordinary and non-carry runs are unchanged.
    """
    big = "import helper\n" + "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    (tmp_path / "main.py").write_text(big, encoding="utf-8", newline="")
    (tmp_path / "helper.py").write_text("HELPER = 1\n", encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "analysis_mode": "single",
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    stable_path = tmp_path / "docs" / "codedoc.json"
    original_leaf = Orchestrator.process_leaf_chunk
    original_process = Orchestrator.process
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())

    def _documented_paths():
        if not stable_path.exists():
            return set()
        doc = json.loads(stable_path.read_text(encoding="utf-8"))
        return {record.get("path") for record in doc.get("files", [])}

    # ---- Pass 1: interrupt main.py's split -> a genuine one-leaf checkpoint. --
    calls = {"leaf": 0}

    def fail_after_first_leaf(self, request):
        calls["leaf"] += 1
        if calls["leaf"] >= 2:
            raise LLMError("interrupted after the first leaf")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_after_first_leaf)
    stats1 = run_pipeline(tmp_path, config)
    assert stats1["failed"] == 1
    assert recovery_path.exists()
    predecessor_partial = json.loads(recovery_path.read_text(encoding="utf-8"))[
        "_codedoc"
    ]["partial_files"]["main.py"]
    assert len(predecessor_partial["nodes"]) == 1
    # helper.py was legitimately documented in this pass; its record is now
    # part of the stable output that pass 2 must not disturb.
    assert "helper.py" in _documented_paths()

    def _helper_record():
        doc = json.loads(stable_path.read_text(encoding="utf-8"))
        return next(r for r in doc["files"] if r.get("path") == "helper.py")

    helper_record_before = _helper_record()

    # ---- Make main.py's partial a CROSS-PLAN predecessor; change helper.py. ---
    monkeypatch.setattr(file_division, "REDUCTION_PACKING_REVISION", "rp-v5-next")
    if hasattr(record_meta, "REDUCTION_PACKING_REVISION"):
        monkeypatch.setattr(record_meta, "REDUCTION_PACKING_REVISION", "rp-v5-next")
    (tmp_path / "helper.py").write_text(
        "HELPER = 2  # changed\n", encoding="utf-8", newline=""
    )

    recovery_before = recovery_path.read_bytes()
    stable_before = stable_path.read_bytes() if stable_path.exists() else None

    helper_calls = {"n": 0}

    def count_helper(self, request):
        if getattr(request, "rel_path", None) == "helper.py":
            helper_calls["n"] += 1
        return original_process(self, request)

    monkeypatch.setattr(Orchestrator, "process", count_helper)

    # ---- Pass 2: the carried replacement fails; helper.py must NOT run. ------
    def fail_every_leaf(self, request):
        raise LLMError("cross-plan replacement interrupted")

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_every_leaf)
    stats2 = run_pipeline(tmp_path, config)

    assert stats2["failed"] == 1  # only main.py failed
    assert helper_calls["n"] == 0  # the unrelated file was never attempted/charged
    assert stats2["checked"] == 0
    # exact-byte preservation of BOTH persisted files
    assert recovery_path.read_bytes() == recovery_before
    assert stable_before is not None
    assert stable_path.read_bytes() == stable_before
    # helper.py's stale (pre-change) record is untouched -- not re-documented
    assert _helper_record() == helper_record_before
    # the carried predecessor container is still present and unchanged
    carried = json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]["main.py"]
    assert carried == predecessor_partial

    # ---- Pass 3: a clean retry supersedes the carried state, then helper.py. -
    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", original_leaf)
    stats3 = run_pipeline(tmp_path, config)
    assert stats3["failed"] == 0
    assert stats3["checked"] >= 2
    assert helper_calls["n"] == 1  # the deferred file is processed exactly once now
    assert not recovery_path.exists()
    assert {"main.py", "helper.py"} <= _documented_paths()


def test_non_carry_split_failed_run_still_writes_stable_output_and_checkpoint(
    tmp_path, monkeypatch
) -> None:
    """Isolation for section 6.3's exact-byte carry fix: a *non-carry* fresh
    split run that fails a leaf must still (re)write ``codedoc.json`` and its
    resumable ``crash_recovery.json`` checkpoint. The cross-plan-carry byte
    guard must not have disabled stable-output writing for ordinary failed
    split runs.
    """
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "analysis_mode": "single",
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    stable_path = tmp_path / "docs" / "codedoc.json"
    original_leaf = Orchestrator.process_leaf_chunk
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    calls = {"n": 0}

    def fail_after_first_leaf(self, request):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise LLMError("interrupted after the first leaf")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_after_first_leaf)
    stats = run_pipeline(tmp_path, config)

    assert stats["failed"] == 1
    assert stats["checked"] == 0
    # Non-carry failed split run: stable output IS written and the checkpoint
    # IS persisted (nothing about ordinary failed-run behaviour changed).
    assert stable_path.exists()
    assert recovery_path.exists()
    partials = json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]
    assert list(partials) == ["main.py"]
    assert len(partials["main.py"]["nodes"]) == 1


def test_ordinary_failed_run_still_writes_stable_output(tmp_path, monkeypatch) -> None:
    """Isolation for section 6.3's exact-byte carry fix: an *ordinary*
    (non-split) run whose only file fails must still write ``codedoc.json``.
    """
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8", newline="")
    stable_path = tmp_path / "docs" / "codedoc.json"

    class _BadJson:
        provider_name = "bad-json"

        def complete_json(self, prompt, system=""):
            return "not json at all"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: _BadJson())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "propagate_changes": False,
            "output_dir": "docs",
            "file_retry_attempts": 0,
            # Terminal ordinary failure is the subject; pin the correction
            # default (Section 10 / section 7.2.1).
            "response_correction_enabled": False,
        },
    )

    assert stats["failed"] == 1
    assert stats["checked"] == 0
    assert stable_path.exists()


def test_failed_completed_record_flush_rolls_back_carry_state_and_preserves_bytes(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3: a clean replacement whose completed-record flush itself
    fails must roll back the in-memory carry state (the predecessor stays
    available) and leave the previous recovery bytes intact -- the atomic
    writer never half-writes.
    """
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    carried = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=_CONTENT_HASH,
        division_plan_digest=_PLAN_DIGEST,
        reduction_tree_digest=_TREE_DIGEST,
        nodes=(_leaf_node("main.py"),),
    )
    writer = SafeWriter(recovery_path, "json", "main.py", {})
    writer.load(preloaded_carry_partials={"main.py": carried})
    writer.initialize_empty()
    recovery_before = recovery_path.read_bytes()
    assert writer.get_tree_state("main.py") is None  # carry never validated

    real_atomic = safe_writer_mod.atomic_write_text

    def _boom(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", _boom)
    with pytest.raises(LiveBackupWriteError):
        writer.record("main.py", {"description": "replacement result"}, "0" * 64)
    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", real_atomic)

    # In-memory carry state rolled back; predecessor still carried and resumable.
    assert writer.has_partial_state()
    # The previous recovery bytes are untouched -- the atomic writer never
    # renamed a partial file into place.
    assert recovery_path.read_bytes() == recovery_before
    on_disk = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert "main.py" in on_disk["_codedoc"]["partial_files"]
    assert len(on_disk["_codedoc"]["partial_files"]["main.py"]["nodes"]) == 1


def test_two_phase_deferred_insufficient_source_skip_excludes_stale_record(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3 two-phase boundary regression: a deferred file skipped in
    phase two must be excluded from the published output exactly as the
    single-phase path excludes an execution-time insufficient-source skip --
    never republished from its stale predecessor record.

    Phase one replaces a genuine cross-plan carried predecessor (``main.py``)
    successfully; phase two then defers ``helper.py``, which raises a typed
    ``InsufficientSourceError`` through the real execution routing and is
    marked skipped on the *phase-two* ``ProcessingQueue``. Before the fix,
    final assembly reads execution-time skip states from the *phase-one*
    queue only, so the phase-two skip is invisible and the stale ``helper.py``
    record is re-emitted into ``codedoc.json``. Ordinary single-phase skip
    exclusion is covered by ``test_insufficient_source.py`` and is unchanged.
    """
    big = "import helper\n" + "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    (tmp_path / "main.py").write_text(big, encoding="utf-8", newline="")
    (tmp_path / "helper.py").write_text("HELPER = 1\n", encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    stable_path = tmp_path / "docs" / "codedoc.json"
    original_leaf = Orchestrator.process_leaf_chunk
    original_process = Orchestrator.process
    real_process_one_file = execution._process_one_file
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())

    def _documented_paths():
        if not stable_path.exists():
            return set()
        doc = json.loads(stable_path.read_text(encoding="utf-8"))
        return {record.get("path") for record in doc.get("files", [])}

    # ---- Pass 1: interrupt main.py's split -> a genuine one-leaf checkpoint;
    #      helper.py is documented normally and enters stable output. ----
    calls = {"leaf": 0}

    def fail_after_first_leaf(self, request):
        calls["leaf"] += 1
        if calls["leaf"] >= 2:
            raise LLMError("interrupted after the first leaf")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_after_first_leaf)
    stats1 = run_pipeline(tmp_path, config)
    assert stats1["failed"] == 1
    assert recovery_path.exists()
    predecessor_partial = json.loads(recovery_path.read_text(encoding="utf-8"))[
        "_codedoc"
    ]["partial_files"]["main.py"]
    assert len(predecessor_partial["nodes"]) == 1
    assert "helper.py" in _documented_paths()

    # ---- Make main.py's partial a CROSS-PLAN predecessor; change helper.py so
    #      it is reachable agent work again (not a provider-free precheck skip). -
    monkeypatch.setattr(file_division, "REDUCTION_PACKING_REVISION", "rp-v5-next")
    if hasattr(record_meta, "REDUCTION_PACKING_REVISION"):
        monkeypatch.setattr(record_meta, "REDUCTION_PACKING_REVISION", "rp-v5-next")
    (tmp_path / "helper.py").write_text(
        "HELPER = 2  # changed\n", encoding="utf-8", newline=""
    )

    # ---- Pass 2: phase one replaces main.py; phase two defers helper.py, which
    #      raises a typed InsufficientSourceError through the real routing. ----
    def process_one_file(request, orchestrator):
        if request.rel_path == "helper.py":
            raise InsufficientSourceError("helper.py", "empty_or_whitespace_only")
        return real_process_one_file(request, orchestrator)

    helper_provider_calls = {"n": 0}

    def count_helper_process(self, request):
        if getattr(request, "rel_path", None) == "helper.py":
            helper_provider_calls["n"] += 1
        return original_process(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", original_leaf)
    monkeypatch.setattr(execution, "_process_one_file", process_one_file)
    monkeypatch.setattr(Orchestrator, "process", count_helper_process)

    stats2 = run_pipeline(tmp_path, config)

    # main.py's carried replacement completed; helper.py was skipped, not failed.
    assert stats2["failed"] == 0
    assert stats2["checked"] == 1
    assert stats2["skipped_insufficient_source"] == 1
    # No provider work for helper.py after the typed insufficient-source verdict.
    assert helper_provider_calls["n"] == 0

    documented = _documented_paths()
    # main.py's successful replacement is present with a fresh identity.
    assert "main.py" in documented
    main_record = next(
        r for r in json.loads(stable_path.read_text(encoding="utf-8"))["files"]
        if r.get("path") == "main.py"
    )
    assert main_record["_large_file_identity"].startswith("large-file-v3:")
    # The deferred insufficient-source file is excluded -- NOT republished from
    # its stale predecessor record (the pre-fix bug).
    assert "helper.py" not in documented

    # Selected-file completion accounting reconciles; last_run's
    # insufficient-source count is right.
    last_run = json.loads(stable_path.read_text(encoding="utf-8"))["last_run"]
    assert last_run["files_skipped_insufficient_source"] == 1
    assert last_run["files_failed"] == 0
    assert last_run["files_selected"] == sum(
        last_run[key]
        for key in (
            "files_documented_by_llm",
            "files_failed",
            "files_reused_unchanged",
            "files_reused_identical_content",
            "files_unattempted",
            "files_skipped_insufficient_source",
        )
    )
    assert stats2["unattempted_files"] == 0

    # Recovery cleanup follows the existing successful-with-skip behaviour:
    # main.py's carry was superseded by a clean replacement and nothing else
    # holds partial state, so recovery is removed.
    assert not recovery_path.exists()


def test_multi_carry_completed_path_persists_while_failed_path_predecessor_is_preserved(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3 multi-carry contract (per-path) -- contract-freezing.

    Recovery carries cross-plan predecessor partials for TWO oversized split
    files. In a deterministic sequential run the first replacement completes
    and the second fails. Plan sec 7.1 ("a completed replacement still flushes
    transactionally") and plan sec 12 (checkpointing over carried state is
    forbidden only *before* clean replacement) resolve the contract per carried
    path, not per whole recovery file:

      1. the failed path keeps its predecessor container unchanged;
      2. no replacement checkpoint is written for the failed path;
      3. the completed path transactionally supersedes its own predecessor and
         persists as a completed recovery record;
      4. that successfully paid work is not discarded;
      5. stable project output stays byte-identical while any carry is incomplete;
      6. recovery stays present while failed carried state remains;
      7. the next run reuses the completed path with no provider call and retries
         only the still-incomplete carried path;
      8. when the remaining path succeeds, stable output publishes and recovery
         is removed.

    Current behaviour already satisfies this per-path contract, so this freezes
    it rather than proving a fix.
    """
    helper_src = "\n".join(f"def h_{i}(): return {i}" for i in range(220)) + "\n"
    main_src = (
        "import helper\n"
        + "\n".join(f"def m_{i}(): return {i}" for i in range(220))
        + "\n"
    )
    (tmp_path / "helper.py").write_text(helper_src, encoding="utf-8", newline="")
    (tmp_path / "main.py").write_text(main_src, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    stable_path = tmp_path / "docs" / "codedoc.json"
    original_leaf = Orchestrator.process_leaf_chunk
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())

    # ---- Pass 1: interrupt the first leaf of EACH split file -> two genuine
    #      one-leaf predecessor containers. ----
    leaf_counts: dict = {}

    def fail_first_leaf_per_file(self, request):
        rel = getattr(request, "rel_path", None)
        leaf_counts[rel] = leaf_counts.get(rel, 0) + 1
        if leaf_counts[rel] >= 2:
            raise LLMError(f"interrupted after first leaf of {rel}")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_first_leaf_per_file)
    stats1 = run_pipeline(tmp_path, config)
    assert stats1["failed"] == 2
    partials1 = json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]
    assert set(partials1) == {"helper.py", "main.py"}
    assert len(partials1["helper.py"]["nodes"]) == 1
    assert len(partials1["main.py"]["nodes"]) == 1
    main_predecessor = partials1["main.py"]

    # ---- Cross-plan revision transition: both partials become predecessors. ----
    monkeypatch.setattr(file_division, "REDUCTION_PACKING_REVISION", "rp-v5-next")
    if hasattr(record_meta, "REDUCTION_PACKING_REVISION"):
        monkeypatch.setattr(record_meta, "REDUCTION_PACKING_REVISION", "rp-v5-next")

    stable_before = stable_path.read_bytes()

    # ---- Pass 2: deterministic order (helper.py before main.py -- main imports
    #      helper). helper.py replacement completes; main.py replacement fails. --
    def fail_only_main_leaves(self, request):
        if getattr(request, "rel_path", None) == "main.py":
            raise LLMError("main.py replacement interrupted")
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", fail_only_main_leaves)
    stats2 = run_pipeline(tmp_path, config)

    assert stats2["checked"] == 1  # helper.py
    assert stats2["failed"] == 1  # main.py
    # (5) stable project output byte-identical while a carry is incomplete
    assert stable_path.read_bytes() == stable_before
    # (6) recovery still present
    assert recovery_path.exists()
    recovery2 = json.loads(recovery_path.read_text(encoding="utf-8"))
    partials2 = recovery2["_codedoc"]["partial_files"]
    # (1) + (2) failed path predecessor unchanged; no replacement checkpoint
    assert list(partials2) == ["main.py"]
    assert partials2["main.py"] == main_predecessor
    assert len(partials2["main.py"]["nodes"]) == 1
    # (3) completed path persisted as a completed recovery record
    assert {r.get("path") for r in recovery2["files"]} == {"helper.py"}

    # ---- Pass 3: retry. (7) helper.py reused with no provider call; only main.py
    #      retried. (8) main.py succeeds -> stable publishes, recovery removed. --
    provider_leaves: dict = {}

    def count_leaves_per_file(self, request):
        rel = getattr(request, "rel_path", None)
        provider_leaves[rel] = provider_leaves.get(rel, 0) + 1
        return original_leaf(self, request)

    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", count_leaves_per_file)
    stats3 = run_pipeline(tmp_path, config)

    assert stats3["failed"] == 0
    # (7) completed carried path reused with no provider call; only the
    # still-incomplete carried path is retried.
    assert provider_leaves.get("helper.py", 0) == 0
    assert provider_leaves.get("main.py", 0) >= 1
    # (4) the paid helper.py work from pass 2 was not discarded / re-executed
    # (proved by the zero provider leaves above) and remains published.
    document3 = json.loads(stable_path.read_text(encoding="utf-8"))
    assert {r.get("path") for r in document3["files"]} == {"helper.py", "main.py"}
    for record in document3["files"]:
        assert record["_large_file_identity"].startswith("large-file-v3:")
    # (8) clean whole-run completion publishes stable output and removes recovery.
    assert not recovery_path.exists()


# ===========================================================================
# Section 5.8 / 6.3 correction round 2: dry-run is a read-only preview of the
# SAME payable work and the SAME recovery-transition classification as a real
# preflight, proven against a GENUINE on-disk crash_recovery.json produced by a
# real interrupted split run. For every recovery-bearing case: the dry-run
# leaves the recovery file BYTE-FOR-BYTE unchanged and writes / quarantines /
# checkpoints / deletes nothing; the dry and real preflight snapshots are
# field-for-field identical (recursively, minus the ``dry_run`` marker); and
# the three §6.3 counters -- persisted ``split_reexecuted_nodes`` and the two
# ephemeral ``split_recovery_discarded_predecessor_nodes`` /
# ``split_recovery_replacement_nodes_planned`` -- carry the exact expected
# integers.
# ===========================================================================


def _cr2_norm(value):
    from types import MappingProxyType

    if isinstance(value, (dict, MappingProxyType)):
        return {k: _cr2_norm(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_cr2_norm(v) for v in value]
    return value


class _Cr2Stop(Exception):
    pass


def _cr2_source(lines: int = 1000) -> str:
    return "\n".join(f"value_{i} = {i}" for i in range(lines)) + "\n"


def _cr2_current_node_count(source: str, budget: int) -> int:
    dp = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=budget
    )
    tr = build_reduction_tree(dp, synthesis_manifest_chars=12000)
    return len(dp.chunks) + len(tr.all_nodes)


def _cr2_write_interrupted_recovery(tmp_path, monkeypatch, *, budget, stop_after):
    """Run a real split pipeline that fails after ``stop_after`` leaf
    checkpoints, leaving a genuine on-disk crash_recovery.json. Returns
    (config_without_budget, recovery_path, checkpointed_node_ids)."""
    (tmp_path / "main.py").write_text(_cr2_source(), encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "analysis_mode": "single",
        "max_content_chars": budget,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    original_leaf = Orchestrator.process_leaf_chunk
    seen = {"n": 0}

    def _fail_after(self, request):
        if seen["n"] >= stop_after:
            raise LLMError("interrupted for the recovery fixture")
        seen["n"] += 1
        return original_leaf(self, request)

    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
        mp.setattr(Orchestrator, "process_leaf_chunk", _fail_after)
        stats = run_pipeline(tmp_path, config)
    assert stats["failed"] == 1
    assert recovery_path.exists()
    partial = json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]["main.py"]
    node_ids = [n["node_id"] for n in partial["nodes"]]
    assert len(node_ids) == stop_after
    return config, recovery_path, node_ids


def _cr2_capture(tmp_path, config, monkeypatch, *, dry):
    """Capture the provider-free preflight snapshot for one run. dry=True hard-
    fails on any provider/writer/output probe. dry=False allows a real completing
    run but still proves the report preceded provider construction."""
    events, snaps = [], []
    import codedoc.pipeline as _pl

    real_probe = _pl.preflight_output_accessibility

    def _prov(_c):
        events.append("provider")
        if dry:
            raise _Cr2Stop("provider constructed during dry-run preflight")
        return SmartFake()

    def _writer(*a, **k):
        events.append("writer")
        if dry:
            raise _Cr2Stop("SafeWriter constructed during dry-run preflight")
        return SafeWriter(*a, **k)

    def _probe(*a, **k):
        events.append("probe")
        if dry:
            raise _Cr2Stop("output probed during dry-run")
        return real_probe(*a, **k)

    err = None
    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", _prov)
        mp.setattr("codedoc.pipeline.SafeWriter", _writer)
        mp.setattr("codedoc.pipeline.preflight_output_accessibility", _probe)
        try:
            run_pipeline(
                tmp_path, {**config, "dry_run": dry},
                plan_reporter=lambda s: (events.append("report"), snaps.append(_cr2_norm(s))),
            )
        except (_Cr2Stop, ConfigError) as exc:
            err = exc
    snap = snaps[0] if snaps else None
    if snap is not None:
        snap.pop("dry_run", None)
    # An unexpected ConfigError (or a _Cr2Stop from a provider/writer/probe
    # that should never have been constructed) is surfaced, never swallowed.
    return snap, events, err


_CR2_TRANSITION_CASES = {
    # name: (predecessor_budget, stop_after, current_budget)
    "compatible_current_partial": (2000, 3, 2000),   # same plan -> resume unpaid
    "same_plan_stale_node": (2000, 3, 2000),         # (stale via bumped revision below)
    "cross_plan_expansion": (8000, 1, 2000),         # 1 discarded  < many replacement
    "cross_plan_contraction": (2000, 5, 8000),       # 5 discarded  > few replacement
}


@pytest.mark.parametrize("case", sorted(_CR2_TRANSITION_CASES))
def test_cr2_recovery_transition_dry_real_parity_on_disk(tmp_path, monkeypatch, case):
    pred_budget, stop_after, cur_budget = _CR2_TRANSITION_CASES[case]
    config, recovery_path, checkpointed = _cr2_write_interrupted_recovery(
        tmp_path, monkeypatch, budget=pred_budget, stop_after=stop_after
    )
    original_bytes = recovery_path.read_bytes()
    source = _cr2_source()

    is_cross_plan = case.startswith("cross_plan")
    if is_cross_plan:
        cur_node_ids = _cr2_current_node_count(source, cur_budget)
        expected_discarded = stop_after
        expected_replacement = cur_node_ids
        expected_reexecuted = 0            # plan digest moved -> chunk IDs differ
        expected_conflict = 1
    elif case == "same_plan_stale_node":
        # Bump the leaf prompt revision so the checkpointed leaves are stale
        # under a matching plan/tree -> quarantine + re-run (no cross-plan carry).
        monkeypatch.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-cr2-stale")
        if hasattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION"):
            monkeypatch.setattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-cr2-stale")
        expected_discarded = 0             # not a cross-plan transition
        expected_replacement = 0
        expected_reexecuted = stop_after   # the stale checkpointed leaves re-run
        expected_conflict = 1
    else:  # compatible_current_partial
        expected_discarded = 0
        expected_replacement = 0
        expected_reexecuted = 0
        expected_conflict = 0

    cfg = {**config, "max_content_chars": cur_budget}
    stable_path = tmp_path / "docs" / "codedoc.json"
    stable_before = stable_path.read_bytes() if stable_path.exists() else None
    dry_snap, dry_events, dry_err = _cr2_capture(tmp_path, cfg, monkeypatch, dry=True)

    # a resolved-valid compatible/stale/cross-plan recovery is NOT an error
    # scenario: neither mode raises, and no _Cr2Stop sentinel is swallowed.
    assert dry_err is None

    # BYTE preservation + no mutation after the dry run.
    assert recovery_path.read_bytes() == original_bytes
    assert (stable_path.read_bytes() if stable_path.exists() else None) == stable_before
    assert dry_events == ["report"]

    real_snap, real_events, real_err = _cr2_capture(tmp_path, cfg, monkeypatch, dry=False)
    assert real_err is None
    assert real_events[0] == "report"
    if "provider" in real_events:
        assert real_events.index("report") < real_events.index("provider")

    # FULL recursive snapshot parity, minus the mode marker.
    assert dry_snap is not None and real_snap is not None
    assert dry_snap == real_snap

    # exact §6.3 counter values on BOTH snapshots.
    for snap in (dry_snap, real_snap):
        assert snap["split_recovery_discarded_predecessor_nodes"] == expected_discarded
        assert snap["split_recovery_replacement_nodes_planned"] == expected_replacement
        assert snap["split_reexecuted_nodes"] == expected_reexecuted
        assert snap["split_recovery_conflict_files"] == expected_conflict
        assert snap["split_reexecuted_nodes"] <= snap["split_unpaid_nodes"]
        if case == "cross_plan_contraction":
            assert (
                snap["split_recovery_discarded_predecessor_nodes"]
                > snap["split_recovery_replacement_nodes_planned"]
            )
        # scenario-presence: prove the run actually entered this scenario.
        if is_cross_plan:
            assert snap["split_recovery_conflict_files"] == 1
            assert snap["split_divided_files"] == 1
            assert snap["total_calls_planned"] > 0
        elif case == "same_plan_stale_node":
            assert snap["split_quarantined_nodes"] >= stop_after
            assert snap["split_reexecuted_nodes"] == stop_after
        else:  # compatible_current_partial
            assert snap["split_partial_files_resumed"] == 1
            assert snap["split_restored_complete_chunks"] == stop_after
            assert snap["split_recovery_conflict_files"] == 0


def test_cli_prints_recovery_transition_counters_in_preflight_and_final_stats(
    tmp_path, monkeypatch, capsys
):
    """F1 regression: ten test files already asserted
    ``split_recovery_discarded_predecessor_nodes`` /
    ``split_recovery_replacement_nodes_planned`` on the snapshot *dict* --
    zero of them asserted the printed CLI text, so a presenter that silently
    dropped both counters would have left the whole suite green. This test
    exists to close exactly that hole: it asserts the terminal output of a
    real ``run_cli`` invocation, not the snapshot.

    Uses a genuine cross-plan *contraction* (predecessor budget 2000, 5
    checkpointed leaves; current budget 8000) so ``discarded`` (5) is
    strictly greater than ``replacement`` (fewer nodes fit the larger
    budget) -- proving the two counters are rendered as independent counts,
    never as a balanced pair."""
    config, _recovery_path, _node_ids = _cr2_write_interrupted_recovery(
        tmp_path, monkeypatch, budget=2000, stop_after=5
    )
    source = _cr2_source()
    cur_budget = 8000
    expected_discarded = 5
    expected_replacement = _cr2_current_node_count(source, cur_budget)
    assert expected_discarded > expected_replacement, (
        "fixture must exercise a genuine contraction (discarded > replacement)"
    )

    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({**config, "max_content_chars": cur_budget}), encoding="utf-8"
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())

    rc = run_cli([str(tmp_path), "--entry", "main.py"])
    out = capsys.readouterr().out
    assert rc == 0

    expected_line = (
        "Recovery transition (discarded/replacement): "
        f"{expected_discarded} / {expected_replacement}"
    )
    preflight = out.split("Planned provider work (before calls)", 1)[1].split(
        "\ncodedoc complete.", 1
    )[0]
    assert expected_line in preflight, "missing from the preflight summary"

    final = out.split("\ncodedoc complete.", 1)[1]
    assert expected_line in final, "missing from the final run summary"

    # The two counts are never presented as if they must reconcile.
    assert expected_discarded != expected_replacement


def test_cli_prints_sorted_nonzero_closure_reasons_before_provider_construction(
    tmp_path, monkeypatch, capsys
):
    """F1 regression (closure reasons half, F1b): section 5.8 builds six
    ``split_closures_*`` aggregates naming why each leaf closed; before this
    fix ``grep -c "split_closures" codedoc/cli/cli.py`` returned 0, so the
    data existed only in the snapshot dict. This test independently captures
    the genuine snapshot via ``plan_reporter`` (not a hand-typed guess) to
    know the true nonzero reasons, then asserts the CLI's printed line
    matches those exact counts, sorted, with zero-count reasons omitted --
    matching the established ``Blocked reasons`` style."""
    (tmp_path / "main.py").write_text(
        _cr2_source(lines=1000), encoding="utf-8", newline=""
    )
    cfg = {
        "entry_file": "main.py", "large_file_strategy": "split", "analysis_mode": "single",
        "max_content_chars": 2000, "parallel_agents": False, "propagate_changes": False,
        "output_dir": "docs",
    }
    snaps = []
    run_pipeline(
        tmp_path, {**cfg, "dry_run": True},
        plan_reporter=lambda s: snaps.append(s),
    )
    snap = snaps[0]
    closure_map = {
        "source-ceiling": snap["split_closures_source_ceiling"],
        "metadata-ceiling": snap["split_closures_metadata_ceiling"],
        "source-and-metadata-ceiling": snap["split_closures_source_and_metadata_ceiling"],
        "oversized-unit-isolation": snap["split_closures_oversized_unit_isolation"],
        "continuation": snap["split_closures_continuation"],
        "end-of-file": snap["split_closures_end_of_file"],
    }
    nonzero = {name: count for name, count in closure_map.items() if count}
    assert nonzero, "fixture produced no closures -- pick a genuine split scenario"
    expected_line = "Closure reasons               : " + ", ".join(
        f"{name}={count}" for name, count in sorted(nonzero.items())
    )

    (tmp_path / "codedoc.config.json").write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: (_ for _ in ()).throw(
            RuntimeError("provider must not be constructed")
        ),
    )

    rc = run_cli([str(tmp_path), "--entry", "main.py"])
    out = capsys.readouterr().out
    assert "Planned provider work (before calls)" in out
    assert expected_line in out
    for name, count in closure_map.items():
        if count == 0:
            assert f"{name}=" not in out, f"zero-count reason {name!r} was rendered"
    assert rc == 1


def test_cr2_no_conflict_fresh_and_completed_reuse_transition_counters_are_zero(
    tmp_path, monkeypatch
):
    """No-conflict fresh split work and a completed-current reuse both report
    zero recovery-transition activity, identically in dry and real preflight."""
    (tmp_path / "main.py").write_text(_cr2_source(), encoding="utf-8", newline="")
    cfg = {
        "entry_file": "main.py", "large_file_strategy": "split", "analysis_mode": "single",
        "max_content_chars": 2000, "parallel_agents": False, "propagate_changes": False,
        "output_dir": "docs",
    }
    # fresh
    d1, _e1, d1_err = _cr2_capture(tmp_path, cfg, monkeypatch, dry=True)
    r1, _e2, r1_err = _cr2_capture(tmp_path, cfg, monkeypatch, dry=False)
    assert d1_err is None and r1_err is None
    assert d1 == r1
    assert d1["split_recovery_discarded_predecessor_nodes"] == 0
    assert d1["split_recovery_replacement_nodes_planned"] == 0
    assert d1["split_reexecuted_nodes"] == 0
    assert d1["split_divided_files"] == 1              # scenario presence: genuinely fresh
    assert d1["total_calls_planned"] > 0

    # completed-current reuse: the fresh real run above already wrote codedoc.json.
    d2, _e3, d2_err = _cr2_capture(tmp_path, cfg, monkeypatch, dry=True)
    r2, _e4, r2_err = _cr2_capture(tmp_path, cfg, monkeypatch, dry=False)
    assert d2_err is None and r2_err is None
    assert d2 == r2
    assert d2["split_completed_files_reused"] == 1     # scenario presence: reuse happened
    assert d2["total_calls_planned"] == 0
    assert d2["split_recovery_discarded_predecessor_nodes"] == 0
    assert d2["split_recovery_replacement_nodes_planned"] == 0


def test_cr2_malformed_current_recovery_dry_real_error_text_is_identical_on_disk(
    tmp_path, monkeypatch
):
    """A genuine on-disk crash_recovery.json whose current-schema container is
    malformed: dry and real preflight raise the IDENTICAL ConfigError type and
    exact text, at the recovery-load boundary (before any snapshot); no
    reporter, no provider, no review, no writer, no output probe; dry leaves
    the recovery file byte-for-byte unchanged."""
    config, recovery_path, _ids = _cr2_write_interrupted_recovery(
        tmp_path, monkeypatch, budget=2000, stop_after=2
    )
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload["_codedoc"]["partial_files"]["main.py"]["nodes"] = "not-a-list"
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    original_bytes = recovery_path.read_bytes()

    def _run(dry, reports, events):
        with monkeypatch.context() as mp:
            mp.setattr("codedoc.pipeline.create_provider",
                       lambda _c: events.append("provider") or pytest.fail("provider"))
            mp.setattr("codedoc.pipeline.SafeWriter",
                       lambda *a, **k: events.append("writer") or pytest.fail("writer"))
            mp.setattr("codedoc.pipeline.preflight_output_accessibility",
                       lambda *a, **k: events.append("probe") or pytest.fail("probe"))
            with pytest.raises(ConfigError) as exc:
                run_pipeline(tmp_path, {**config, "dry_run": dry},
                             plan_reporter=lambda s: reports.append(s))
        return exc.value

    dry_reports, dry_events = [], []
    dry_err = _run(True, dry_reports, dry_events)
    assert recovery_path.read_bytes() == original_bytes
    assert dry_reports == [] and dry_events == []

    real_reports, real_events = [], []
    real_err = _run(False, real_reports, real_events)
    assert real_reports == [] and real_events == []

    assert type(dry_err) is type(real_err)
    assert str(dry_err) == str(real_err)
    assert recovery_path.read_bytes() == original_bytes


# ===========================================================================
# Defect 1 (P0): an oversized split file whose source later fits
# ``max_content_chars`` routes as an ordinary whole-file call. Its paid
# schema-4 checkpoints must still be carried byte-for-byte (planning Edit A),
# and a carried path this run cannot complete must not withhold stable output
# forever (pipeline Edit B). Every fixture below builds a GENUINE on-disk
# schema-4 crash_recovery.json through the real ``run_pipeline`` / ``SafeWriter``
# path -- no hand-authored recovery bytes.
# ===========================================================================

_D1_LOW_BUDGET = 2000    # the ~3.3k-char 220-line source splits at this ceiling
_D1_HIGH_BUDGET = 5000   # ... and routes as an ordinary whole-file call at this one


def _d1_source(prefix: str = "") -> str:
    return prefix + "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"


def _d1_write_split_partial(tmp_path, monkeypatch, *, prefix=""):
    """Real interrupted split run at ``_D1_LOW_BUDGET`` -> a genuine one-leaf
    on-disk crash_recovery.json. Returns
    (config_at_low_budget, recovery_path, stable_path, predecessor_partial)."""
    source = _d1_source(prefix)
    assert _D1_LOW_BUDGET < len(source) < _D1_HIGH_BUDGET
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "analysis_mode": "single",
        "max_content_chars": _D1_LOW_BUDGET,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    stable_path = tmp_path / "docs" / "codedoc.json"
    original_leaf = Orchestrator.process_leaf_chunk
    seen = {"n": 0}

    def _fail_after_first_leaf(self, request):
        seen["n"] += 1
        if seen["n"] >= 2:
            raise LLMError("interrupted after the first leaf (D1 fixture)")
        return original_leaf(self, request)

    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
        mp.setattr(Orchestrator, "process_leaf_chunk", _fail_after_first_leaf)
        stats = run_pipeline(tmp_path, config)
    assert stats["failed"] == 1
    assert recovery_path.exists()
    predecessor_partial = json.loads(recovery_path.read_text(encoding="utf-8"))[
        "_codedoc"
    ]["partial_files"]["main.py"]
    assert len(predecessor_partial["nodes"]) == 1
    return config, recovery_path, stable_path, predecessor_partial


def _d1_partial_nodes_on_disk(recovery_path):
    return json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]["main.py"]


def test_d1_under_threshold_provider_init_failure_preserves_recovery(
    tmp_path, monkeypatch
) -> None:
    """Ceiling raised so the file routes ordinary + the provider factory raises:
    the schema-4 recovery container is byte-identical and its partial is intact
    (before Edit A, ``initialize_empty()`` flushed the banner over it before a
    provider even existed)."""
    config, recovery_path, stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()
    stable_before = stable_path.read_bytes()

    class _InitSentinel(Exception):
        pass

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: (_ for _ in ()).throw(_InitSentinel("provider init sentinel")),
    )
    with pytest.raises(_InitSentinel):
        run_pipeline(tmp_path, {**config, "max_content_chars": _D1_HIGH_BUDGET})

    assert recovery_path.read_bytes() == recovery_before
    assert stable_path.read_bytes() == stable_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor


def test_d1_under_threshold_provider_call_failure_preserves_recovery(
    tmp_path, monkeypatch
) -> None:
    """Ceiling raised + the ordinary documentation call fails: the recovery file
    still exists and is byte-identical (before the fix it was deleted outright)."""
    config, recovery_path, _stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()

    class _BadJson:
        provider_name = "bad-json"

        def complete_json(self, prompt, system=""):
            return "not json at all"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: _BadJson())
    stats = run_pipeline(
        tmp_path,
        {
            **config,
            "max_content_chars": _D1_HIGH_BUDGET,
            # Terminal provider-call failure is the subject here; pin the
            # correction default so the flip adds no incidental repair call
            # (Section 10 / section 7.2.1).
            "response_correction_enabled": False,
        },
    )
    assert stats["failed"] == 1

    assert recovery_path.exists()
    assert recovery_path.read_bytes() == recovery_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor


def test_d1_under_threshold_keyboard_interrupt_preserves_recovery(
    tmp_path, monkeypatch
) -> None:
    """Ceiling raised + a ``KeyboardInterrupt`` mid-run: the recovery bytes are
    byte-identical and the partial survives."""
    config, recovery_path, _stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()

    def _interrupt(request, _orchestrator):
        raise KeyboardInterrupt()

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    monkeypatch.setattr(execution, "_process_one_file", _interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_pipeline(tmp_path, {**config, "max_content_chars": _D1_HIGH_BUDGET})

    assert recovery_path.read_bytes() == recovery_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor


def test_d1_under_threshold_failed_then_clean_replacement_supersedes(
    tmp_path, monkeypatch
) -> None:
    """A failed ordinary replacement leaves the carried predecessor bytes
    untouched; the subsequent clean ordinary replacement transactionally
    supersedes it and removes recovery, publishing a fresh ordinary record."""
    config, recovery_path, stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()
    stable_before = stable_path.read_bytes()
    high = {**config, "max_content_chars": _D1_HIGH_BUDGET}

    # ---- failed ordinary replacement: predecessor + stable output untouched. --
    def _interrupt(request, _orchestrator):
        raise KeyboardInterrupt()

    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
        mp.setattr(execution, "_process_one_file", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_pipeline(tmp_path, high)
    assert recovery_path.read_bytes() == recovery_before
    assert stable_path.read_bytes() == stable_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor

    # ---- clean ordinary replacement: superseded, recovery removed. -----------
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    stats = run_pipeline(tmp_path, high)
    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert not recovery_path.exists()
    record = next(
        r
        for r in json.loads(stable_path.read_text(encoding="utf-8"))["files"]
        if r.get("path") == "main.py"
    )
    assert record["description"]
    # An ordinary whole-file record, not a split-identity one.
    assert "_large_file_identity" not in record


def test_d1_under_threshold_mixed_run_preserves_carry_and_publishes_unrelated(
    tmp_path, monkeypatch
) -> None:
    """A carried under-threshold path alongside unrelated agent work: the
    predecessor container is never flushed over by ``initialize_empty()``, both
    files are documented, and recovery is removed once both records land."""
    config, recovery_path, stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch, prefix="import helper\n"
    )
    recovery_before = recovery_path.read_bytes()
    # helper.py is fresh, unrelated agent work for pass 2.
    (tmp_path / "helper.py").write_text(
        "HELPER = 2  # a real module\n", encoding="utf-8", newline=""
    )

    real_initialize_empty = SafeWriter.initialize_empty
    captured = {}

    def _capture_after_init(self):
        result = real_initialize_empty(self)
        captured["bytes_after_init"] = recovery_path.read_bytes()
        return result

    monkeypatch.setattr(SafeWriter, "initialize_empty", _capture_after_init)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    stats = run_pipeline(
        tmp_path, {**config, "max_content_chars": _D1_HIGH_BUDGET}
    )

    # initialize_empty() ran but did NOT rewrite the carried container.
    assert captured["bytes_after_init"] == recovery_before
    assert stats["failed"] == 0
    documented = {
        r.get("path")
        for r in json.loads(stable_path.read_text(encoding="utf-8"))["files"]
    }
    assert {"main.py", "helper.py"} <= documented
    assert not recovery_path.exists()


def test_d1_under_threshold_forced_routing_preserves_then_supersedes(
    tmp_path, monkeypatch
) -> None:
    """``force_files`` on the now-ordinary path: a failed forced run preserves
    the predecessor bytes; a clean forced run supersedes it and removes
    recovery."""
    config, recovery_path, stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()
    forced = {
        **config,
        "max_content_chars": _D1_HIGH_BUDGET,
        "force_files": ["main.py"],
    }

    def _interrupt(request, _orchestrator):
        raise KeyboardInterrupt()

    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
        mp.setattr(execution, "_process_one_file", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_pipeline(tmp_path, forced)
    assert recovery_path.read_bytes() == recovery_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    stats = run_pipeline(tmp_path, forced)
    assert stats["checked"] == 1
    assert not recovery_path.exists()


def test_d1_under_threshold_dry_and_real_preflight_parity(
    tmp_path, monkeypatch
) -> None:
    """The recovery-aware preflight snapshot is field-for-field identical in dry
    and real mode for the carried under-threshold path, and a dry run leaves the
    predecessor bytes untouched and constructs no provider / writer / probe."""
    config, recovery_path, stable_path, _predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    original_bytes = recovery_path.read_bytes()
    stable_before = stable_path.read_bytes()
    high = {**config, "max_content_chars": _D1_HIGH_BUDGET}

    dry_snap, dry_events, dry_err = _cr2_capture(tmp_path, high, monkeypatch, dry=True)
    assert dry_err is None
    assert dry_events == ["report"]
    assert recovery_path.read_bytes() == original_bytes
    assert stable_path.read_bytes() == stable_before

    real_snap, real_events, real_err = _cr2_capture(
        tmp_path, high, monkeypatch, dry=False
    )
    assert real_err is None
    assert real_events[0] == "report"
    if "provider" in real_events:
        assert real_events.index("report") < real_events.index("provider")
    assert dry_snap is not None and real_snap is not None
    assert dry_snap == real_snap


def test_d1_insufficient_source_carried_path_still_publishes_unrelated_work(
    tmp_path, monkeypatch
) -> None:
    """Edit B deadlock guard: a carried under-threshold path that this run
    classifies as insufficient-source is skipped and never enters
    ``new_results``. It must NOT withhold the run's stable output forever --
    unrelated agent work is still published -- while its recovery stays
    preserved."""
    config, recovery_path, stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch, prefix="import helper\n"
    )
    recovery_before = recovery_path.read_bytes()
    (tmp_path / "helper.py").write_text(
        "HELPER = 2  # changed\n", encoding="utf-8", newline=""
    )

    import codedoc.core.planning as planning
    real_insufficient = planning.insufficient_source

    def _force_main_insufficient(content):
        if content.startswith("import helper"):
            return True, "d1_forced_insufficient"
        return real_insufficient(content)

    monkeypatch.setattr(planning, "insufficient_source", _force_main_insufficient)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    stats = run_pipeline(
        tmp_path, {**config, "max_content_chars": _D1_HIGH_BUDGET}
    )

    assert stats["failed"] == 0
    assert stats["skipped_insufficient_source"] == 1
    documented = {
        r.get("path")
        for r in json.loads(stable_path.read_text(encoding="utf-8"))["files"]
    }
    # Unrelated work is published despite the un-completable carried path
    # (before Edit B, the carried path withheld stable output permanently).
    assert "helper.py" in documented
    assert "main.py" not in documented
    # The carried predecessor is preserved, not deleted: its checkpoint nodes
    # are still on disk unchanged (the recovery file is re-serialized as
    # helper.py's completion is flushed, but the carried main.py partial rides
    # through untouched). Before Edit A it was flushed away entirely.
    assert recovery_path.exists()
    assert _d1_partial_nodes_on_disk(recovery_path)["nodes"] == predecessor["nodes"]
    assert recovery_before  # captured a genuine one-node predecessor


def test_d1_negative_control_oversized_cross_plan_transition_is_unchanged(
    tmp_path, monkeypatch
) -> None:
    """Regression guard: a genuine oversized -> oversized cross-plan transition
    still runs through the existing conflict path (not the Edit A preservation
    sweep), reporting the three distinct §6.3 counters and
    ``recovery_conflict_files == 1``, and preserving predecessor bytes on a
    failed replacement."""
    # Source oversized at BOTH budgets -> a true cross-plan transition, never
    # the under-threshold ordinary route the Edit A sweep handles.
    config, recovery_path, _ids = _cr2_write_interrupted_recovery(
        tmp_path, monkeypatch, budget=8000, stop_after=1
    )
    original_bytes = recovery_path.read_bytes()
    current_nodes = _cr2_current_node_count(_cr2_source(), 2000)
    expanded = {**config, "max_content_chars": 2000}

    # Counter snapshot via a NON-mutating dry preflight: the genuine conflict
    # path (not the Edit A sweep) still reports the three distinct §6.3 counters.
    dry_snap, dry_events, dry_err = _cr2_capture(
        tmp_path, expanded, monkeypatch, dry=True
    )
    assert dry_err is None and dry_events == ["report"]
    assert dry_snap is not None
    assert dry_snap["split_recovery_conflict_files"] == 1
    assert dry_snap["split_recovery_discarded_predecessor_nodes"] == 1
    assert dry_snap["split_recovery_replacement_nodes_planned"] == current_nodes
    assert dry_snap["split_reexecuted_nodes"] == 0
    assert dry_snap["split_reexecuted_nodes"] <= dry_snap["split_unpaid_nodes"]
    assert recovery_path.read_bytes() == original_bytes

    # A failed real replacement leaves the predecessor container byte-identical.
    def _fail_every_leaf(self, request):
        raise LLMError("replacement interrupted")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    monkeypatch.setattr(Orchestrator, "process_leaf_chunk", _fail_every_leaf)
    stats = run_pipeline(tmp_path, expanded)
    assert stats["failed"] == 1
    assert recovery_path.read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# Section 6.3 lines 1611-1613 / 1641-1645: content hash is one of the three
# carry triggers, and preservation is not conditional on resumability. An
# EDITED source (content hash no longer matches the partial) must be carried
# byte-for-byte on BOTH routes -- the split branch already does this via
# ``cross_plan_conflict``; the ordinary-routed sweep must agree.
# ---------------------------------------------------------------------------

_D1_EDITED_SUFFIX = "sentinel_edit_marker = 1\n"


def test_d1_edited_source_under_threshold_provider_init_failure_preserves_recovery(
    tmp_path, monkeypatch
) -> None:
    """Regression: interrupted split leaves a 1-node partial; the source is then
    EDITED (content hash changes) and the ceiling raised so the file routes as
    an ordinary whole-file call; the second run fails at provider construction.
    The recovery container must be byte-identical with its partial intact --
    exactly as when the source is unchanged, and exactly as the still-oversized
    control below. Before the completed-record guard swap this partial was
    destroyed purely because the routing differed."""
    config, recovery_path, _stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()

    edited = _d1_source() + _D1_EDITED_SUFFIX
    assert _D1_LOW_BUDGET < len(edited) < _D1_HIGH_BUDGET
    (tmp_path / "main.py").write_text(edited, encoding="utf-8", newline="")

    class _InitSentinel(Exception):
        pass

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: (_ for _ in ()).throw(_InitSentinel("provider init sentinel")),
    )
    with pytest.raises(_InitSentinel):
        run_pipeline(tmp_path, {**config, "max_content_chars": _D1_HIGH_BUDGET})

    assert recovery_path.exists()
    assert recovery_path.read_bytes() == recovery_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor


def test_d1_edited_source_still_oversized_preserves_recovery(
    tmp_path, monkeypatch
) -> None:
    """Control: the identical source edit with the file left oversized (ceiling
    unchanged, so it routes split) is carried byte-for-byte on a failed second
    run -- the pre-existing ``cross_plan_conflict`` path. This is the route the
    regression above must now match: same edit, same paid nodes, same outcome
    regardless of routing."""
    config, recovery_path, _stable_path, predecessor = _d1_write_split_partial(
        tmp_path, monkeypatch
    )
    recovery_before = recovery_path.read_bytes()

    edited = _d1_source() + _D1_EDITED_SUFFIX
    assert len(edited) > _D1_LOW_BUDGET  # still oversized at the low ceiling
    (tmp_path / "main.py").write_text(edited, encoding="utf-8", newline="")

    class _InitSentinel(Exception):
        pass

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: (_ for _ in ()).throw(_InitSentinel("provider init sentinel")),
    )
    with pytest.raises(_InitSentinel):
        run_pipeline(tmp_path, config)  # low ceiling -> still split

    assert recovery_path.exists()
    assert recovery_path.read_bytes() == recovery_before
    assert _d1_partial_nodes_on_disk(recovery_path) == predecessor


# ===========================================================================
# 0.14.9 section 5.6 / section 13 step 11 -- the recovery & invalidation
# regression matrix for the two authorized revision advances:
#   LEAF_CAPSULE_SCHEMA_REVISION  leaf-capsule-v10  -> leaf-capsule-v11
#   REDUCER_PROMPT_REVISION       file-reduction-v3 -> file-reduction-v4
#
# Every predecessor node here is stamped by the *production* identity function
# with the constant patched back, never a hand-authored digest, and every
# quarantine reason / retained set / re-execution is read from the real
# validate_recovered_tree + build_pipeline_plan resume path.
# ===========================================================================

_S3_OLD_LEAF_REV = "leaf-capsule-v10"
_S3_OLD_REDUCER_REV = "file-reduction-v3"


def _s3_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _s3_source(units: int = 260) -> str:
    return "\n".join(f"value_{i} = {i}" for i in range(units)) + "\n"


def _s3_config(budget: int, *, propagate: bool = False) -> dict:
    return {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": budget,
        "max_parallel_files": 1,
        "parallel_agents": False,
        "propagate_changes": propagate,
        "output_dir": "docs",
        "file_retry_attempts": 0,
    }


def _s3_plan_tree(source: str, budget: int):
    plan = file_division.build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=budget
    )
    tree = file_division.build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(
            budget, file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
        ),
        language="python",
    )
    return plan, tree


def _s3_provider_identity(tmp_path, config: dict) -> str:
    from codedoc.core.loader import load_config

    return file_division.provider_execution_identity(load_config(tmp_path, config))


def _s3_leaf_node(plan, chunk, *, content_hash, provider_identity, index):
    return tree_node_state(
        node_id=chunk.chunk_id,
        node_type="leaf",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=file_division.leaf_input_digest(
            rel_path=plan.rel_path,
            language="python",
            chunk=chunk,
            unit_indexes=plan.unit_positions(chunk),
            unit_count=len(plan.units),
        ),
        execution_identity_digest=file_division.leaf_execution_identity(
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=chunk,
        ),
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(chunk.chunk_id,),
        result={"description": f"restored leaf {index}", "chunk_id": chunk.chunk_id,
                "unit_id": chunk.unit_id},
    )


def _s3_reducer_node(plan, tree, node, *, content_hash, provider_identity, child_results):
    raw = tuple(
        child_results[cid].get("narrative", child_results[cid].get("description", ""))
        for cid in node.child_ids
    )
    result = {"narrative": f"restored {node.phase} narrative"}
    red = tree_node_state(
        node_id=node.node_id,
        node_type=node.phase,
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=file_division.reduction_input_digest(
            rel_path=plan.rel_path,
            phase=node.phase,
            level=node.level,
            unit_id=node.unit_id,
            child_count=len(node.child_ids),
            ordered_child_narratives=file_division.refine_narrative_inputs(raw),
        ),
        execution_identity_digest=file_division.reduction_execution_identity(
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
        result=result,
    )
    return red, result


def _s3_final_result(rel_path, *, imports=()):
    """The exact post-cleaner combined object a live final-synthesis checkpoint
    stores, so a CURRENT synthetic final node passes
    ``_node_result_matches_live_schema`` and can be *retained* (not only
    dependency-pruned). Built through the same ``process_response`` +
    ``flat_combined_result`` path the recovery validator reapplies."""
    from codedoc.agents.response_cleaning import clean_combined_report
    from codedoc.agents.response_diagnostics import process_response
    from codedoc.core.prompt_profiles import ResolvedProfile
    from codedoc.core.result_assembly import flat_combined_result

    resolved_shape = ResolvedProfile("single", None).resolve_block(
        "combined", rel_path
    )
    cleaned = process_response(
        file_division.canonical_json(
            {"description": "A restored final synthesis narrative for the file."}
        ),
        mode="single", agent="combined", file_path=rel_path,
        clean_reporter=clean_combined_report, resolved_shape=resolved_shape,
    )
    return flat_combined_result(rel_path, "python", list(imports), cleaned)


def _s3_final_node(plan, tree, *, content_hash, provider_identity, results_by_id,
                   prompt_profile_digest, imports=()):
    final = tree.final_node
    imports_digest = file_division.deterministic_imports_digest(imports)
    ledger = file_division.build_fact_ledger(
        [results_by_id[c.chunk_id] for c in plan.chunks],
        language="python", chunks=plan.chunks, symbols=plan.symbols,
    )
    raw = tuple(
        results_by_id[cid].get("narrative", results_by_id[cid].get("description", ""))
        for cid in final.child_ids
    )
    manifest_json = file_division.final_synthesis_input(
        rel_path=plan.rel_path, language="python", imports=imports,
        root_narratives=file_division.refine_narrative_inputs(raw),
        root_coverage_leaf_ids=final.leaf_ids, ledger=ledger,
        max_chars=tree.synthesis_manifest_chars,
    )
    return tree_node_state(
        node_id=final.node_id,
        node_type="final",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=file_division.final_input_digest(
            imports_digest=imports_digest,
            resolved_shape_digest=prompt_profile_digest,
            manifest_json=manifest_json,
        ),
        execution_identity_digest=file_division.final_execution_identity(
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            provider_identity=provider_identity,
            prompt_profile_digest=prompt_profile_digest,
            imports_digest=imports_digest,
            node=final,
        ),
        unit_id=None,
        child_ids=final.child_ids,
        coverage_leaf_ids=final.leaf_ids,
        result=_s3_final_result(plan.rel_path, imports=imports),
    )


def _s3_completed_state(plan, tree, *, content_hash, provider_identity,
                        prompt_profile_digest, monkeypatch,
                        stale_leaves=frozenset(), stale_reducers=frozenset()):
    """A dependency-valid predecessor ``SplitTreeState`` over the CURRENT
    plan/tree, with selected leaf indexes / reducer node-ids stamped under the
    OLD revision (genuine predecessor identities via the production functions)."""
    import json as _json

    results_by_id: dict[str, dict] = {}
    leaf_nodes = []
    for index, chunk in enumerate(plan.chunks):
        if index in stale_leaves:
            with monkeypatch.context() as mp:
                mp.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", _S3_OLD_LEAF_REV)
                if hasattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION"):
                    mp.setattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION",
                               _S3_OLD_LEAF_REV)
                node = _s3_leaf_node(plan, chunk, content_hash=content_hash,
                                     provider_identity=provider_identity, index=index)
        else:
            node = _s3_leaf_node(plan, chunk, content_hash=content_hash,
                                 provider_identity=provider_identity, index=index)
        leaf_nodes.append(node)
        results_by_id[chunk.chunk_id] = _json.loads(node.result_json)
    reducer_nodes = []
    for node in tree.unit_consolidation_nodes + tree.general_nodes:
        child_results = {cid: results_by_id[cid] for cid in node.child_ids}
        if node.node_id in stale_reducers:
            with monkeypatch.context() as mp:
                mp.setattr(file_division, "REDUCER_PROMPT_REVISION", _S3_OLD_REDUCER_REV)
                if hasattr(record_meta, "REDUCER_PROMPT_REVISION"):
                    mp.setattr(record_meta, "REDUCER_PROMPT_REVISION",
                               _S3_OLD_REDUCER_REV)
                red, result = _s3_reducer_node(
                    plan, tree, node, content_hash=content_hash,
                    provider_identity=provider_identity, child_results=child_results)
        else:
            red, result = _s3_reducer_node(
                plan, tree, node, content_hash=content_hash,
                provider_identity=provider_identity, child_results=child_results)
        reducer_nodes.append(red)
        results_by_id[node.node_id] = result
    final_node = _s3_final_node(
        plan, tree, content_hash=content_hash, provider_identity=provider_identity,
        results_by_id=results_by_id, prompt_profile_digest=prompt_profile_digest,
    )
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(leaf_nodes + reducer_nodes + [final_node]),
    )


def _s3_validate_kwargs(plan, tree, content_hash, provider_identity):
    from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST, ResolvedProfile

    return dict(
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=NO_PROMPT_PROFILE_DIGEST,
        imports_digest=file_division.deterministic_imports_digest(()),
        language="python",
        resolved_shape=ResolvedProfile("single", None).resolve_block("combined", "main.py"),
    )


def _s3_build_plan(tmp_path, source, budget, recovered_state):
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan

    src = tmp_path / "main.py"
    src.write_text(source, encoding="utf-8", newline="")
    file_map = {
        "main.py": {"path": src, "rel_path": "main.py", "language": "python",
                    "extension": ".py"},
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split", "max_content_chars": budget,
        "truncation_head_ratio": 0.70,
    }
    return build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, [], config,
        recovered_partials={"main.py": recovered_state},
    )


def _s3_plan_tree_for(rel_path, source, budget):
    plan = file_division.build_division_plan(
        rel_path=rel_path, language="python", content=source,
        source_budget_chars=budget,
    )
    tree = file_division.build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(
            budget, file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
        ),
        language="python",
    )
    return plan, tree


def _s3_build_two_file_plan(tmp_path, files, budget):
    """``build_pipeline_plan`` over two independent split files, each with its
    own recovered partial. ``files`` maps rel_path -> (source, SplitTreeState)."""
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan

    file_map = {}
    graph = DependencyGraph()
    recovered = {}
    for rel, (source, state) in files.items():
        path = tmp_path / rel
        path.write_text(source, encoding="utf-8", newline="")
        file_map[rel] = {"path": path, "rel_path": rel, "language": "python",
                         "extension": ".py"}
        graph.add_file(rel)
        recovered[rel] = state
    entry = next(iter(files))
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split", "max_content_chars": budget,
        "truncation_head_ratio": 0.70,
    }
    return build_pipeline_plan(
        file_map, graph, set(files), entry, {}, [], config,
        recovered_partials=recovered,
    )


def test_s3_final_synthesis_and_split_checkpoints_outside_the_closure_are_preserved(
    tmp_path, monkeypatch
):
    """P2-4: a stale identity in ONE split file does not reach into an unrelated
    split file whose recovered partial is fully current and dependency-closed.

    ``affected.py`` carries a ``leaf-capsule-v10`` leaf -> that leaf plus its
    dependency-pruned reducer and final are invalidated. ``unrelated.py`` carries
    a wholly current partial -- every leaf, every reducer, AND the final-synthesis
    node. After the real resume:

    * every ``unrelated.py`` node -- leaves, reducers, and the final-synthesis
      checkpoint -- is retained, with zero quarantine;
    * no ``unrelated.py`` node (and not its ``file-synthesis`` call) appears in
      the unpaid call manifest; only ``affected.py``'s closure does;
    * ``split_reexecuted_nodes`` counts ``affected.py``'s closure only;
    * ``recovery_conflict_files`` is 1, not 2.

    Ordinary and truncate current-record preservation under the same two
    advances is covered by
    ``tests/integration/pipeline/test_cache_identity.py``
    ::``test_s3_current_identity_split_ordinary_and_truncate_records_still_reuse``."""
    budget = 1500
    aff_source = _s3_source(600)
    unr_source = _s3_source(620)
    aff_plan, aff_tree = _s3_plan_tree_for("affected.py", aff_source, budget)
    unr_plan, unr_tree = _s3_plan_tree_for("unrelated.py", unr_source, budget)
    assert aff_plan.plan_digest != unr_plan.plan_digest        # genuinely distinct
    assert len(aff_plan.chunks) >= 3 and len(unr_plan.chunks) >= 3

    provider_identity = _s3_provider_identity(tmp_path, _s3_config(budget))
    aff_hash = _s3_sha256(aff_source)
    unr_hash = _s3_sha256(unr_source)
    ppd = _s3_validate_kwargs(aff_plan, aff_tree, aff_hash, provider_identity)[
        "prompt_profile_digest"
    ]

    aff_state = _s3_completed_state(
        aff_plan, aff_tree, content_hash=aff_hash,
        provider_identity=provider_identity, prompt_profile_digest=ppd,
        monkeypatch=monkeypatch, stale_leaves={0},
    )
    unr_state = _s3_completed_state(
        unr_plan, unr_tree, content_hash=unr_hash,
        provider_identity=provider_identity, prompt_profile_digest=ppd,
        monkeypatch=monkeypatch,
    )

    res, materials = _s3_build_two_file_plan(
        tmp_path,
        {"affected.py": (aff_source, aff_state),
         "unrelated.py": (unr_source, unr_state)},
        budget,
    )

    unr_final_id = unr_tree.final_node.node_id
    unr_reducer_ids = {
        n.node_id for n in unr_tree.unit_consolidation_nodes + unr_tree.general_nodes
    }
    unr_leaf_ids = {c.chunk_id for c in unr_plan.chunks}
    unr_all_ids = unr_leaf_ids | unr_reducer_ids | {unr_final_id}

    # ---- unrelated.py: every node retained, including final-synthesis --------
    unr_ts = materials.tree_states["unrelated.py"]
    assert set(unr_ts.by_id()) == unr_all_ids
    assert unr_final_id in set(unr_ts.by_id())
    assert unr_reducer_ids <= set(unr_ts.by_id())
    assert list(unr_ts.quarantine) == []

    # ---- affected.py: only its own closure is invalidated -------------------
    aff_ts = materials.tree_states["affected.py"]
    aff_stale_leaf = aff_plan.chunks[0].chunk_id
    aff_reasons = {e.node_id: e.reason for e in aff_ts.quarantine}
    assert aff_reasons[aff_stale_leaf] == "stale-identity"
    assert set(aff_reasons.values()) == {"stale-identity", "input-digest-mismatch"}
    assert aff_stale_leaf not in set(aff_ts.by_id())

    assert materials.recovery_conflict_files == 1                 # not 2
    assert materials.reexecuted_nodes == len(aff_state.nodes) - len(list(aff_ts.by_id()))

    # ---- the unpaid manifest touches nothing in unrelated.py ---------------
    manifest = build_call_manifest(
        [], sorted(res.agent_rels), "single",
        division_plans=materials.division_plans,
        reduction_trees=materials.reduction_trees,
        tree_states=materials.tree_states,
    )
    owners = {call.owner for call in manifest.calls}
    assert owners.isdisjoint(unr_all_ids)                        # no retained node
    assert "unrelated.py" not in owners                          # nor its synthesis
    by_cat: dict[str, list[str]] = {}
    for call in manifest.calls:
        by_cat.setdefault(call.category, []).append(call.owner)
    assert by_cat["file-synthesis"] == ["affected.py"]           # only the affected final
    assert set(by_cat.get("unit-documentation", [])) == {aff_stale_leaf}
    assert manifest.digest == _s3_sha256(
        "\n".join(c.call_id for c in manifest.calls)
    )


def test_s3_isolated_stale_leaf_is_quarantined_stale_identity_and_re_executed(
    tmp_path, monkeypatch
):
    """Objective 2: one leaf stamped ``leaf-capsule-v10`` while every other node
    is current. The stale leaf is quarantined with the exact closed reason
    ``stale-identity`` and leaves ``completed_ids``; its dependency-pruned
    ancestors are ``input-digest-mismatch``; the unrelated current leaves stay
    retained (reusable)."""
    source = _s3_source(600)
    budget = 1500
    plan, tree = _s3_plan_tree(source, budget)
    assert len(plan.chunks) >= 3
    content_hash = _s3_sha256(source)
    provider_identity = _s3_provider_identity(tmp_path, _s3_config(budget))
    kw = _s3_validate_kwargs(plan, tree, content_hash, provider_identity)

    state = _s3_completed_state(
        plan, tree, content_hash=content_hash, provider_identity=provider_identity,
        prompt_profile_digest=kw["prompt_profile_digest"], stale_leaves={0},
        monkeypatch=monkeypatch,
    )

    retained, quarantine = file_division.validate_recovered_tree(state.nodes, **kw)
    reasons = {e.node_id: e.reason for e in quarantine}
    stale_leaf_id = plan.chunks[0].chunk_id
    retained_ids = {n.node_id for n in retained}

    assert reasons[stale_leaf_id] == "stale-identity"
    assert stale_leaf_id not in retained_ids
    current_leaf_ids = {c.chunk_id for c in plan.chunks[1:]}
    assert current_leaf_ids <= retained_ids
    leaf_ids = {c.chunk_id for c in plan.chunks}
    for node_id, reason in reasons.items():
        if node_id not in leaf_ids:
            assert reason == "input-digest-mismatch", node_id
    assert set(reasons.values()) == {"stale-identity", "input-digest-mismatch"}
    assert len(quarantine) <= file_division.MAX_QUARANTINE_ENTRIES_PER_FILE

    _plan_res, materials = _s3_build_plan(tmp_path, source, budget, state)
    resumed = materials.tree_states["main.py"]
    assert [e.reason for e in resumed.quarantine if e.node_id == stale_leaf_id] == [
        "stale-identity"
    ]
    assert stale_leaf_id not in resumed.by_id()
    assert current_leaf_ids <= set(resumed.by_id())
    assert materials.recovery_conflict_files == 1


def test_s3_isolated_stale_reducer_reaches_keep_and_is_quarantined_stale_identity(
    tmp_path, monkeypatch
):
    """Objective 3 + 10 (reducer half): every child leaf is current; one reducer
    is stamped ``file-reduction-v3``. Because its children are retained it
    reaches ``_keep`` and fails on its own execution identity -> ``stale-identity``
    (never ``input-digest-mismatch``); the pruned final is ``input-digest-mismatch``.

    Then this exact v3->v4 scenario is driven through real ``build_pipeline_plan``
    and ``build_call_manifest``: the stale reducer leaves ``completed_ids`` and
    becomes an unpaid ``file-reduction`` call with its own node ID as owner; the
    pruned final becomes the required ``file-synthesis`` call; the retained child
    leaves generate no new ``unit-documentation`` calls; the manifest stays
    canonical; and no unrelated node is scheduled."""
    source = _s3_source(600)
    budget = 1500
    plan, tree = _s3_plan_tree(source, budget)
    reducers = tree.unit_consolidation_nodes + tree.general_nodes
    assert len(reducers) == 1, "this scenario needs exactly one reducer node"
    target = reducers[0]
    final_id = tree.final_node.node_id
    content_hash = _s3_sha256(source)
    provider_identity = _s3_provider_identity(tmp_path, _s3_config(budget))
    kw = _s3_validate_kwargs(plan, tree, content_hash, provider_identity)

    state = _s3_completed_state(
        plan, tree, content_hash=content_hash, provider_identity=provider_identity,
        prompt_profile_digest=kw["prompt_profile_digest"],
        stale_reducers={target.node_id}, monkeypatch=monkeypatch,
    )

    # --- direct validator: exact reasons -------------------------------------
    retained, quarantine = file_division.validate_recovered_tree(state.nodes, **kw)
    reasons = {e.node_id: e.reason for e in quarantine}
    retained_ids = {n.node_id for n in retained}
    leaf_ids = {c.chunk_id for c in plan.chunks}

    assert reasons[target.node_id] == "stale-identity"
    assert reasons[final_id] == "input-digest-mismatch"
    assert set(reasons) == {target.node_id, final_id}
    assert leaf_ids <= retained_ids and retained_ids == leaf_ids  # only the leaves

    # --- real planning-resume: completed_ids / re-plan / call manifest ------
    _plan_res, materials = _s3_build_plan(tmp_path, source, budget, state)
    resumed = materials.tree_states["main.py"]
    completed_ids = set(resumed.by_id())

    assert target.node_id not in completed_ids          # stale reducer excluded
    assert final_id not in completed_ids
    assert leaf_ids <= completed_ids                    # current children present
    assert completed_ids == leaf_ids

    previously_paid = {n.node_id for n in state.nodes}
    assert previously_paid - completed_ids == {target.node_id, final_id}
    assert materials.reexecuted_nodes == 2              # reducer + pruned final

    manifest = build_call_manifest(
        [], sorted(_plan_res.agent_rels), "single",
        division_plans=materials.division_plans,
        reduction_trees=materials.reduction_trees,
        tree_states=materials.tree_states,
    )
    by_cat: dict[str, list[str]] = {}
    for call in manifest.calls:
        by_cat.setdefault(call.category, []).append(call.owner)

    assert sorted(_plan_res.agent_rels) == ["main.py"]          # nothing unrelated
    assert by_cat.get("unit-documentation", []) == []           # leaves reused
    assert by_cat.get("file-documentation", []) == []
    assert by_cat["file-reduction"] == [target.node_id]         # the exact reducer
    assert by_cat["file-synthesis"] == ["main.py"]              # the pruned final
    assert len(manifest.calls) == 2


def test_s3_combined_whole_tree_invalidation_allocates_quarantine_reasons(
    tmp_path, monkeypatch
):
    """Objective 4: a partial whose leaves are ALL ``leaf-capsule-v10`` and
    reducers ALL ``file-reduction-v3``. Directly-stale leaves -> ``stale-identity``;
    dependency-pruned reducers/final -> ``input-digest-mismatch``. The reason set
    is closed, untruncated, and NOT flattened onto a single value."""
    source = _s3_source(600)
    budget = 1500
    plan, tree = _s3_plan_tree(source, budget)
    content_hash = _s3_sha256(source)
    provider_identity = _s3_provider_identity(tmp_path, _s3_config(budget))
    kw = _s3_validate_kwargs(plan, tree, content_hash, provider_identity)
    reducer_ids = {n.node_id for n in tree.unit_consolidation_nodes + tree.general_nodes}

    state = _s3_completed_state(
        plan, tree, content_hash=content_hash, provider_identity=provider_identity,
        prompt_profile_digest=kw["prompt_profile_digest"],
        stale_leaves=set(range(len(plan.chunks))),
        stale_reducers=reducer_ids, monkeypatch=monkeypatch,
    )
    node_count = len(plan.chunks) + len(reducer_ids) + 1
    assert len(state.nodes) == node_count

    retained, quarantine = file_division.validate_recovered_tree(state.nodes, **kw)
    assert retained == ()
    assert len(quarantine) == node_count
    reasons = {e.node_id: e.reason for e in quarantine}
    leaf_ids = {c.chunk_id for c in plan.chunks}
    for node_id, reason in reasons.items():
        assert reason in {"stale-identity", "input-digest-mismatch"}
        assert reason == ("stale-identity" if node_id in leaf_ids
                          else "input-digest-mismatch"), node_id
    assert set(reasons.values()) == {"stale-identity", "input-digest-mismatch"}
    assert len({e.node_id for e in quarantine}) == node_count       # no truncation
    assert len(quarantine) <= file_division.MAX_QUARANTINE_ENTRIES_PER_FILE


def _s3_chunks_for(units: int, budget: int) -> int | None:
    """``len(plan.chunks)`` for *units* ``value_i = i`` lines at *budget*, or
    ``None`` when the plan is chunk-capped (i.e. would exceed
    ``MAX_CHUNKS_PER_FILE``)."""
    try:
        return len(
            file_division.build_division_plan(
                rel_path="main.py", language="python",
                content=_s3_source(units), source_budget_chars=budget,
            ).chunks
        )
    except file_division.SplitCapacityBlocked:
        return None


def _s3_source_for_exact_chunks(target: int, budget: int = 1000):
    """Deterministically find a source producing *exactly* ``target`` leaf
    chunks in the current environment (not a hardcoded length): the chunk count
    is monotonic non-decreasing in the unit count, so bisect for the largest
    unit count whose plan is not yet chunk-capped, then confirm it lands on
    ``target``. Fails loudly if the packer no longer admits an exact-``target``
    plan for this construction, forcing a deliberate refresh."""
    lo, hi = 100, 60_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        count = _s3_chunks_for(mid, budget)
        if count is not None and count <= target:
            lo = mid
        else:
            hi = mid - 1
    count = _s3_chunks_for(lo, budget)
    assert count == target, (
        f"no exact {target}-chunk plan for this construction (largest uncapped "
        f"= {count} chunks at {lo} units); refresh the fixture construction"
    )
    return _s3_source(lo), lo


@requires_structure_pack
def test_s3_whole_plan_invalidation_at_the_maximum_plan_size_stays_within_the_bound(
    tmp_path, monkeypatch
):
    """Objective 5 / 9.1 item 30: a MAXIMUM-size predecessor partial -- a
    deterministic syntax-mode plan with exactly ``MAX_CHUNKS_PER_FILE`` (256)
    leaf chunks -- whose entire stored node set is invalidated under the
    revision advance.

    Every directly-stale leaf is quarantined ``stale-identity``; every
    dependency-pruned reducer/final is ``input-digest-mismatch``; every stored
    node gets exactly one untruncated quarantine entry; ``node_count <= 2n`` and
    ``node_count <= MAX_QUARANTINE_ENTRIES_PER_FILE == 512``; and
    ``validate_recovered_tree`` raises no ``SplitRecoveryStateError``. A
    tightened bound (below the produced population) DOES raise it -- proving the
    headroom is real, not a constant compared with itself. Syntax mode is
    required to hit exactly 256 deterministically, so this skips on a base
    install like its sibling identity tests."""
    source, _units = _s3_source_for_exact_chunks(file_division.MAX_CHUNKS_PER_FILE)
    budget = 1000
    plan, tree = _s3_plan_tree(source, budget)
    n = len(plan.chunks)
    assert n == file_division.MAX_CHUNKS_PER_FILE == 256
    assert plan.structural_mode == "syntax"

    all_reducer_ids = {
        node.node_id for node in tree.unit_consolidation_nodes + tree.general_nodes
    }
    node_count = n + len(tree.all_nodes)          # leaves + reducers + final
    assert len(tree.all_nodes) == len(all_reducer_ids) + 1
    assert node_count <= 2 * n                    # the plan's own "at most 2n" claim
    assert (
        file_division.MAX_QUARANTINE_ENTRIES_PER_FILE
        == 2 * file_division.MAX_CHUNKS_PER_FILE
        == 512
    )
    assert node_count <= file_division.MAX_QUARANTINE_ENTRIES_PER_FILE

    content_hash = _s3_sha256(source)
    # Every stored node is expected to be invalidated, but not all by the same
    # route: the 256 leaves carry leaf-capsule-v10 and the 8 reducers carry
    # file-reduction-v3 (the relevant predecessor identities), while the single
    # final node is built under its CURRENT identity and is rejected only by
    # dependency closure once its children are pruned (input-digest-mismatch,
    # not stale-identity). The retained/quarantined outcome is therefore
    # independent of the provider identity; one fixed shaped value is used
    # consistently for both state construction and validation (kw below).
    provider_identity = "division-execution:" + "b" * 64
    kw = _s3_validate_kwargs(plan, tree, content_hash, provider_identity)
    state = _s3_completed_state(
        plan, tree, content_hash=content_hash, provider_identity=provider_identity,
        prompt_profile_digest=kw["prompt_profile_digest"],
        stale_leaves=set(range(n)),
        stale_reducers=all_reducer_ids,
        monkeypatch=monkeypatch,
    )
    assert len(state.nodes) == node_count

    retained, quarantine = file_division.validate_recovered_tree(state.nodes, **kw)

    assert retained == ()
    assert len(quarantine) == node_count
    assert len(quarantine) <= file_division.MAX_QUARANTINE_ENTRIES_PER_FILE
    # exactly one entry per stored node, no truncation / duplication
    assert len({e.node_id for e in quarantine}) == node_count
    assert {e.node_id for e in quarantine} == {n.node_id for n in state.nodes}

    leaf_ids = {c.chunk_id for c in plan.chunks}
    reasons = {e.node_id: e.reason for e in quarantine}
    for node_id, reason in reasons.items():
        assert reason == (
            "stale-identity" if node_id in leaf_ids else "input-digest-mismatch"
        ), node_id
    assert set(reasons.values()) == {"stale-identity", "input-digest-mismatch"}

    # Mutation resistance: a bound below the produced population raises through
    # the real validator.
    monkeypatch.setattr(
        file_division, "MAX_QUARANTINE_ENTRIES_PER_FILE", node_count - 1
    )
    with pytest.raises(file_division.SplitRecoveryStateError):
        file_division.validate_recovered_tree(state.nodes, **kw)


def _s3_multi_reducer_tree(source, budget):
    """The plan/tree for *source* plus the three reducer nodes this all-category
    partial-resume scenario depends on, identified structurally (never by a
    hardcoded node id): the two level-1 unit-consolidation reducers and the
    single level-2 reducer whose child is the final node. Fails loudly if the
    packer no longer produces that exact shape, forcing a deliberate refresh."""
    plan, tree = _s3_plan_tree(source, budget)
    reducers = tree.unit_consolidation_nodes + tree.general_nodes
    assert len(plan.chunks) >= 20, "need a wide plan to spread leaves over 2 reducers"
    assert len(reducers) == 3, f"scenario needs exactly 3 reducers, got {len(reducers)}"
    level1 = [r for r in reducers if r.level == 1]
    level2 = [r for r in reducers if r.level == 2]
    assert len(level1) == 2 and len(level2) == 1, "need 2 level-1 + 1 level-2 reducers"
    idx = {c.chunk_id: i for i, c in enumerate(plan.chunks)}
    l1_leaves = [
        {idx[c] for c in r.child_ids if c in idx} for r in level1
    ]
    assert all(l1_leaves), "each level-1 reducer must consolidate leaf chunks"
    # {0, 1} must land wholly inside ONE level-1 reducer so the OTHER level-1
    # reducer has only current children and can reach ``_keep``.
    holder = next(
        (i for i, leaves in enumerate(l1_leaves) if {0, 1} <= leaves), None
    )
    assert holder is not None, "stale leaves 0 and 1 are not under one reducer"
    pruned_reducer = level1[holder]
    clean_reducer = level1[1 - holder]
    clean_leaves = l1_leaves[1 - holder]
    assert clean_leaves.isdisjoint({0, 1}), "clean reducer unexpectedly holds a stale leaf"
    l2 = level2[0]
    assert set(l2.child_ids) == {pruned_reducer.node_id, clean_reducer.node_id}
    assert tree.final_node.child_ids == (l2.node_id,)
    return plan, tree, pruned_reducer, clean_reducer, l2


@requires_structure_pack
def test_s3_partial_resume_across_the_boundary_has_an_exact_per_category_call_delta(
    tmp_path, monkeypatch
):
    """P2-3 (partial resume): the same saved partial, resumed once with every
    stored identity CURRENT and once with a real predecessor slice
    (``leaf-capsule-v10`` leaves + a ``file-reduction-v3`` reducer), yields an
    EXACT current-vs-predecessor call-manifest delta -- not merely a nonzero
    ``split_reexecuted_nodes`` against a zero baseline.

    The partial carries every paid node of a three-reducer tree, so all three
    invalidation categories are exercised:

    * two directly-stale leaves -> ``stale-identity`` -> two ``unit-documentation``
      calls, owned by exactly those chunk ids;
    * their level-1 reducer, dependency-pruned -> ``input-digest-mismatch``;
    * a sibling level-1 reducer stamped ``file-reduction-v3`` whose children are
      all retained -> it reaches ``_keep`` and fails on its own identity ->
      ``stale-identity``;
    * the level-2 reducer and the final node, dependency-pruned ->
      ``input-digest-mismatch`` -> three ``file-reduction`` calls (the two
      level-1 reducers + the level-2 reducer) and one ``file-synthesis`` call.

    The predecessor resume therefore plans STRICTLY more calls; the additional
    call owners are EXACTLY the invalidated / pruned node ids; the per-category
    delta is exact (unit-documentation +2, file-reduction +3, file-synthesis
    +1); every unrelated retained leaf reappears in neither manifest; the
    manifest stays canonical; and the nonzero quarantine count corresponds
    one-for-one to the affected recovered nodes."""
    source = _s3_source(3000)
    budget = 1000
    plan, tree, pruned_reducer, clean_reducer, l2 = _s3_multi_reducer_tree(source, budget)
    assert plan.structural_mode == "syntax"
    final_id = tree.final_node.node_id
    leaf0, leaf1 = plan.chunks[0].chunk_id, plan.chunks[1].chunk_id
    all_leaf_ids = {c.chunk_id for c in plan.chunks}
    retained_leaf_ids = {c.chunk_id for c in plan.chunks[2:]}
    all_node_ids = all_leaf_ids | {
        pruned_reducer.node_id, clean_reducer.node_id, l2.node_id, final_id
    }

    content_hash = _s3_sha256(source)
    provider_identity = _s3_provider_identity(tmp_path, _s3_config(budget))
    kw = _s3_validate_kwargs(plan, tree, content_hash, provider_identity)

    def _resume(*, stale_leaves=frozenset(), stale_reducers=frozenset()):
        state = _s3_completed_state(
            plan, tree, content_hash=content_hash,
            provider_identity=provider_identity,
            prompt_profile_digest=kw["prompt_profile_digest"],
            monkeypatch=monkeypatch,
            stale_leaves=stale_leaves, stale_reducers=stale_reducers,
        )
        plan_res, materials = _s3_build_plan(tmp_path, source, budget, state)
        manifest = build_call_manifest(
            [], sorted(plan_res.agent_rels), "single",
            division_plans=materials.division_plans,
            reduction_trees=materials.reduction_trees,
            tree_states=materials.tree_states,
        )
        by_cat: dict[str, list[str]] = {}
        for call in manifest.calls:
            by_cat.setdefault(call.category, []).append(call.owner)
        return state, plan_res, materials, manifest, by_cat

    # ---- current baseline: every stored identity valid -> nothing re-planned --
    state_cur, res_cur, mat_cur, man_cur, cat_cur = _resume()
    assert sorted(res_cur.agent_rels) == ["main.py"]
    assert mat_cur.reexecuted_nodes == 0
    assert set(mat_cur.tree_states["main.py"].by_id()) == all_node_ids
    assert not mat_cur.tree_states["main.py"].quarantine
    assert list(man_cur.calls) == []                 # zero unpaid calls

    # ---- predecessor slice: 2 stale leaves + 1 stale (but keep-reaching) reducer
    state_pred, res_pred, mat_pred, man_pred, cat_pred = _resume(
        stale_leaves={0, 1}, stale_reducers={clean_reducer.node_id},
    )

    # direct validator: the exact closed reasons for every affected node
    retained, quarantine = file_division.validate_recovered_tree(state_pred.nodes, **kw)
    reasons = {e.node_id: e.reason for e in quarantine}
    assert {n.node_id for n in retained} == retained_leaf_ids
    assert reasons == {
        leaf0: "stale-identity",
        leaf1: "stale-identity",
        pruned_reducer.node_id: "input-digest-mismatch",
        clean_reducer.node_id: "stale-identity",
        l2.node_id: "input-digest-mismatch",
        final_id: "input-digest-mismatch",
    }
    assert set(reasons.values()) == {"stale-identity", "input-digest-mismatch"}

    # planning-resume accounting
    resumed = mat_pred.tree_states["main.py"]
    completed_ids = set(resumed.by_id())
    assert completed_ids == retained_leaf_ids
    previously_paid = {n.node_id for n in state_pred.nodes}
    invalidated = {
        leaf0, leaf1, pruned_reducer.node_id, clean_reducer.node_id,
        l2.node_id, final_id,
    }
    assert previously_paid - completed_ids == invalidated
    assert mat_pred.reexecuted_nodes == len(invalidated) == 6
    assert {e.node_id for e in resumed.quarantine} == invalidated   # 1:1 with nodes
    assert mat_pred.recovery_conflict_files == 1

    # ---- the call-manifest delta is exact, per category and per owner --------
    assert sorted(res_pred.agent_rels) == ["main.py"]               # nothing unrelated
    assert sorted(cat_pred["unit-documentation"]) == sorted([leaf0, leaf1])
    assert sorted(cat_pred["file-reduction"]) == sorted(
        [pruned_reducer.node_id, clean_reducer.node_id, l2.node_id]
    )
    assert cat_pred["file-synthesis"] == ["main.py"]
    assert cat_pred.get("file-documentation", []) == []
    assert len(man_pred.calls) == 6

    # no retained leaf is scheduled
    assert retained_leaf_ids.isdisjoint(
        {owner for owners in cat_pred.values() for owner in owners}
    )

    # strictly more work than the current resume, and the increase is EXACTLY
    # the invalidated/pruned closure -- nothing dropped, nothing extra.
    owners_cur = {(c.category, c.owner) for c in man_cur.calls}
    owners_pred = {(c.category, c.owner) for c in man_pred.calls}
    assert owners_cur == set()
    assert owners_pred - owners_cur == {
        ("unit-documentation", leaf0),
        ("unit-documentation", leaf1),
        ("file-reduction", pruned_reducer.node_id),
        ("file-reduction", clean_reducer.node_id),
        ("file-reduction", l2.node_id),
        ("file-synthesis", "main.py"),
    }
    assert owners_cur - owners_pred == set()
    for cat, added in (
        ("unit-documentation", 2), ("file-reduction", 3), ("file-synthesis", 1),
    ):
        assert len(cat_pred.get(cat, [])) - len(cat_cur.get(cat, [])) == added

    # canonical manifest digest over the planned call ids
    assert man_pred.digest == _s3_sha256("\n".join(c.call_id for c in man_pred.calls))


def test_s3_corrected_leaf_capsule_is_a_resumable_checkpoint_across_the_advance(
    tmp_path, monkeypatch
):
    """Objective 1: a leaf whose first response is an over-cap ``description``
    (real ``fixed_cap_exceeded`` -> the 0.14.9 F-1 cap-repair route) is
    corrected, checkpointed, and -- after an interrupt on a later leaf -- is
    RESTORED unchanged on resume (zero re-execution), its checkpoint carrying
    the corrected shorter description. The corrected node is a genuine
    resumable checkpoint (section 5.6.1 premise), current under v11/v4."""
    over_cap = "d" * 320
    corrected = "A concise corrected fragment description."

    class _CorrectFirstLeafThenStop(SmartFake):
        def __init__(self) -> None:
            super().__init__()
            self.first_leaf_failed = False
            self.leaf_checkpoints = 0
            self.correction_calls = 0

        def complete_json(self, prompt, system=""):
            if "Previous response (verbatim" in prompt:
                self.correction_calls += 1
                return json.dumps({"description": corrected})
            if "This is one bounded fragment of a larger" in prompt:
                if not self.first_leaf_failed:
                    self.first_leaf_failed = True
                    return json.dumps({"description": over_cap})
                self.leaf_checkpoints += 1
                if self.leaf_checkpoints >= 2:
                    raise LLMError("interrupt after the corrected leaf checkpointed")
            return super().complete_json(prompt, system)

    source = _s3_source(600)
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = {**_s3_config(1500), "response_correction_enabled": True}
    recovery_path = tmp_path / "docs" / "crash_recovery.json"

    provider = _CorrectFirstLeafThenStop()
    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: provider)
        stats = run_pipeline(tmp_path, config)
    assert stats["failed"] == 1
    assert provider.correction_calls == 1
    assert recovery_path.exists()

    partial = json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
        "partial_files"
    ]["main.py"]
    checkpointed = {n["node_id"]: n for n in partial["nodes"]}
    corrected_nodes = [
        n for n in checkpointed.values()
        if json.loads(n["result_json"]).get("description") == corrected
    ]
    assert len(corrected_nodes) == 1, "the corrected leaf capsule must be checkpointed"
    corrected_node = corrected_nodes[0]
    # The checkpoint carries the corrected shorter text and NOT the rejected
    # over-cap value -- no metadata was lost persisting the corrected capsule.
    assert json.loads(corrected_node["result_json"])["description"] == corrected
    assert over_cap not in corrected_node["result_json"]
    assert corrected_node["node_type"] == "leaf"

    # Resume under the same current v11/v4 constants: the corrected leaf is
    # restored, never re-executed or quarantined.
    resume_provider = SmartFake()
    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: resume_provider)
        resume_stats = run_pipeline(tmp_path, config)

    assert resume_stats["failed"] == 0
    assert resume_stats["split_restored_complete_chunks"] >= 1
    assert resume_stats["split_quarantined_nodes"] == 0
    # The corrected leaf was not re-paid: the resume runs strictly fewer leaf
    # calls than a fresh run of the same plan would.
    assert resume_provider.doc_calls < len(_s3_plan_tree(source, 1500)[0].chunks) + 1
    assert not recovery_path.exists()
    record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert record.get("description")


def test_s3_resume_across_the_revision_boundary_replans_quarantined_leaves(
    tmp_path, monkeypatch
):
    """Objective 10 (leaf half): a genuine interrupted split whose leaf
    checkpoints were written while ``LEAF_CAPSULE_SCHEMA_REVISION`` is patched
    back to ``leaf-capsule-v10``, then resumed under the real current
    ``leaf-capsule-v11``:
      - the stale leaf checkpoints leave ``completed_ids``;
      - the resumed run PLANS MORE work than an identical-identity resume;
      - ``split_quarantined_nodes`` is nonzero and drives ``split_reexecuted_nodes``;
      - the run still completes and clears recovery.

    Only the leaf revision is patched back here: the interrupt fires after the
    second leaf chunk and before any reducer or final checkpoint, so the
    saved partial contains no reducer checkpoint and the reducer revision is
    irrelevant to what this particular resume validates. Reducer-transition
    resume evidence (a checkpoint stamped ``file-reduction-v3`` reaching
    ``_keep`` and re-planned as an unpaid ``file-reduction`` call) is supplied
    by ``test_s3_isolated_stale_reducer_reaches_keep_and_is_quarantined_stale_identity``
    and the combined whole-tree scenario, not here."""
    source = _s3_source(600)
    budget = 1500
    config = _s3_config(budget)
    recovery_path = tmp_path / "docs" / "crash_recovery.json"

    # A real interrupted run under the OLD leaf revision -> genuine
    # leaf-capsule-v10 leaf checkpoints on disk.
    original_leaf = Orchestrator.process_leaf_chunk
    seen = {"n": 0}

    def _fail_after_two(self, request):
        if seen["n"] >= 2:
            raise LLMError("interrupted for the v10/v3 recovery fixture")
        seen["n"] += 1
        return original_leaf(self, request)

    # Only LEAF_CAPSULE_SCHEMA_REVISION is patched back: the interrupt fires
    # before any reducer checkpoint, so only leaf identities matter, and the
    # reducer revision is not on the call-manifest path this run validates.
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", _S3_OLD_LEAF_REV)
        mp.setattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION", _S3_OLD_LEAF_REV)
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
        mp.setattr(Orchestrator, "process_leaf_chunk", _fail_after_two)
        interrupted = run_pipeline(tmp_path, config)
    assert interrupted["failed"] == 1
    assert recovery_path.exists()
    stale_ids = [
        n["node_id"]
        for n in json.loads(recovery_path.read_text(encoding="utf-8"))["_codedoc"][
            "partial_files"
        ]["main.py"]["nodes"]
    ]
    assert len(stale_ids) == 2

    # Baseline: a resume whose SAME two checkpoints are current re-plans nothing
    # extra (reexecuted == 0), so any increase below is the invalidation cost.
    plan, tree = _s3_plan_tree(source, budget)
    content_hash = _s3_sha256(source)
    provider_identity = _s3_provider_identity(tmp_path, config)
    kw = _s3_validate_kwargs(plan, tree, content_hash, provider_identity)
    current_two = _s3_completed_state(
        plan, tree, content_hash=content_hash, provider_identity=provider_identity,
        prompt_profile_digest=kw["prompt_profile_digest"], monkeypatch=monkeypatch,
    )
    current_partial = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
        rel_path="main.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
        nodes=tuple(n for n in current_two.nodes if n.node_id in set(stale_ids)),
    )
    _pr_cur, mat_cur = _s3_build_plan(tmp_path, source, budget, current_partial)
    baseline_reexecuted = mat_cur.reexecuted_nodes
    assert set(mat_cur.tree_states["main.py"].by_id()) == set(stale_ids)
    assert not mat_cur.tree_states["main.py"].quarantine
    assert baseline_reexecuted == 0

    # Resume the REAL interrupted recovery under current v11/v4.
    resume_provider = SmartFake()
    with monkeypatch.context() as mp:
        mp.setattr("codedoc.pipeline.create_provider", lambda _c: resume_provider)
        resumed = run_pipeline(tmp_path, config)

    assert resumed["failed"] == 0
    assert resumed["split_quarantined_nodes"] >= 2
    assert resumed["split_reexecuted_nodes"] >= 2
    assert resumed["split_restored_complete_chunks"] == 0     # nothing v10 restored
    # the resumed plan pays for the invalidated leaves that a current-identity
    # resume would have restored for free.
    assert resumed["split_reexecuted_nodes"] > baseline_reexecuted
    assert not recovery_path.exists()
    assert json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0].get("description")
