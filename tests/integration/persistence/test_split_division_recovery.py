"""Node-keyed split recovery (schema version 2 — D11/D12/section 12).

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
from codedoc.core.execution import (
    _process_divided_file,
    _process_one_file_with_retries,
)
from codedoc.core.file_division import (
    build_division_plan,
    build_reduction_tree,
    tree_node_state,
)
from codedoc.core.execution_model import CallManifestTracker, build_call_manifest
from codedoc.core.project_view import json_from_view
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import build_recovery_identity
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import (
    ConfigError,
    LiveBackupWriteError,
    LLMError,
    UnrecoverableProviderError,
)
from tests.support.execution_requests import make_execution_request
from tests.support.providers import SmartFake
from tests.support.run_metadata_cases import _view as run_metadata_view

pytestmark = pytest.mark.future_split_recovery

_CONTENT_HASH = "0" * 64
_PLAN_DIGEST = "division-plan:" + "1" * 64
_TREE_DIGEST = "reduction-tree:" + "2" * 64
_EXECUTION_ID = "division-execution:" + "3" * 64
_CHUNK_ID = "chunk_" + "4" * 64


def _leaf_node(rel_path: str):
    return tree_node_state(
        node_id=_CHUNK_ID,
        node_type="leaf",
        rel_path=rel_path,
        content_hash=_CONTENT_HASH,
        division_plan_digest=_PLAN_DIGEST,
        reduction_tree_digest=_TREE_DIGEST,
        execution_identity_digest=_EXECUTION_ID,
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
    writer.record_tree_node("src/large.py", _leaf_node("src/large.py"))

    text = writer.path.read_text(encoding="utf-8")

    assert '"partial_files"' in text
    assert '"files": []' in text
    assert "src/large.py" in text


def test_schema_two_recovery_retains_valid_nodes_beside_malformed_siblings(
    tmp_path,
) -> None:
    writer = SafeWriter(
        tmp_path / "crash_recovery.json",
        "json",
        None,
        {},
        {"version": 1},
    )
    writer.record_tree_node("src/large.py", _leaf_node("src/large.py"))
    payload = json.loads(writer.path.read_text(encoding="utf-8"))
    nodes = payload["_codedoc"]["partial_files"]["src/large.py"]["nodes"]
    valid_node = next(iter(nodes.values()))
    nodes["chunk_" + "5" * 64] = {
        **valid_node,
        "content_hash": "9" * 64,
    }
    nodes["chunk_" + "6" * 64] = {
        "node_type": "leaf",
        "child_ids": 7,
    }
    writer.path.write_text(json.dumps(payload), encoding="utf-8")

    document = read_codedoc_document(writer.path)

    assert len(document.partial_files) == 1
    assert tuple(document.partial_files[0].by_id()) == (_CHUNK_ID,)


def test_completed_record_clears_same_path_split_partial(tmp_path) -> None:
    writer = SafeWriter(tmp_path / "crash_recovery.json", "json", None, {"src/large.py": {}})
    writer.record_tree_node("src/large.py", _leaf_node("src/large.py"))

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
    writer.record_tree_node("src/large.py", _leaf_node("src/large.py"))
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
    writer.record_tree_node("src/large.py", _leaf_node("src/large.py"))

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
    tree = build_reduction_tree(plan, max_content_chars=2000)
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

        def synthesize_divided_file(self, _request, _digest, _manifest):
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
    assert result["_large_file_identity"].startswith("large-file-v2:")
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
    tree = build_reduction_tree(plan, max_content_chars=2000)
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
                raise LLMError(
                    "openai", "Your credit balance is too low to continue"
                )
            self.completed += 1
            return {"description": f"chunk {chunk_request.chunk_id}"}

        def process_reduction_node(self, _request):
            self.reduction_calls += 1
            return {"narrative": "combined"}

        def synthesize_divided_file(self, _request, _digest, _manifest):
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

        def synthesize_divided_file(self, _request, _digest, _manifest):
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
    tree = build_reduction_tree(plan, max_content_chars=2000)
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
    tree = build_reduction_tree(plan, max_content_chars=2000)
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
        node for node in partials["main.py"]["nodes"].values()
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
    assert record["_large_file_identity"].startswith("large-file-v2:")
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
    assert "'main.py'" in message
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
    writer.record_tree_node("main.py", _leaf_node("main.py"))
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    container = payload["_codedoc"]["partial_files"]["main.py"]
    if malformation == "unsupported-schema":
        container["schema_version"] = 3
    elif malformation == "non-object-nodes":
        container["nodes"] = []
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
    writer.record_tree_node("src/main.py", _leaf_node("src/main.py"))
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
