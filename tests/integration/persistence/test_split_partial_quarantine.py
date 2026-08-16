"""Schema-3 quarantine and forced-file carry state (section 9/14, D11/D12).

A node-local rejection during recovered-tree validation must not discard a
valid sibling's checkpoint, and must not silently vanish either: the
rejected node's bounded raw JSON is quarantined until a valid replacement or
a completed record supersedes it. Separately, forcing a split file bypasses
reuse and recovery for scheduling, but the file's prior checkpoint survives
on disk as untouched carry state until the forced run's own completed
record replaces it (section 9's "force bypasses execution, not
preservation").
"""

from __future__ import annotations

import hashlib
import json

import pytest

from codedoc.core.file_division import (
    MAX_QUARANTINE_ENTRIES_PER_FILE,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    SplitTreeState,
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
    leaf_execution_identity,
    leaf_input_digest,
    provider_execution_identity,
    tree_node_state,
    validate_recovered_tree,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.loader import load_config
from codedoc.core.planning import build_pipeline_plan
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError


def _large_source(lines: int = 220) -> str:
    return "\n".join(f"value_{i} = {i}" for i in range(lines)) + "\n"


def _file_map(tmp_path, rel_path: str = "main.py") -> dict:
    return {
        rel_path: {
            "path": tmp_path / rel_path,
            "rel_path": rel_path,
            "language": "python",
            "extension": ".py",
        }
    }


def _split_config(tmp_path, max_chars: int = 2000, **overrides) -> dict:
    return load_config(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": max_chars,
            "propagate_changes": False,
            **overrides,
        },
    )


def _split_fixture(tmp_path, *, corrupt_second: bool):
    """A real division plan/tree plus a hand-built two-leaf recovered
    container: the first leaf checkpointed exactly as a live run would, the
    second either the same (valid) or carrying a corrupted execution
    identity (simulating a stale/tampered checkpoint)."""
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(plan, max_content_chars=2000, language="python")
    assert len(plan.chunks) >= 2
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    nodes = []
    for index, chunk in enumerate(plan.chunks[:2]):
        identity = leaf_execution_identity(
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=chunk,
        )
        if index == 1 and corrupt_second:
            identity = "division-execution:" + "9" * 64
        nodes.append(
            tree_node_state(
                node_id=chunk.chunk_id,
                node_type="leaf",
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                input_digest=leaf_input_digest(
                    rel_path=plan.rel_path,
                    language="python",
                    chunk=chunk,
                    unit_indexes=plan.unit_positions(chunk),
                    unit_count=len(plan.units),
                ),
                execution_identity_digest=identity,
                unit_id=None,
                child_ids=(),
                coverage_leaf_ids=(chunk.chunk_id,),
                result={
                    "description": f"leaf {index}",
                    "chunk_id": chunk.chunk_id,
                    "unit_id": chunk.unit_id,
                },
            )
        )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(nodes),
    )
    return config, plan, tree, provider_identity, content_hash, recovered


def test_node_local_rejection_quarantines_beside_a_retained_valid_sibling(tmp_path) -> None:
    config, plan, _tree, _provider_identity, _content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=True
    )

    file_map = _file_map(tmp_path)
    graph = DependencyGraph()
    graph.add_file("main.py")
    _, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, [], config,
        recovered_partials={"main.py": recovered},
    )

    state = materials.tree_states["main.py"]
    retained_ids = set(state.by_id())
    assert plan.chunks[0].chunk_id in retained_ids
    assert plan.chunks[1].chunk_id not in retained_ids
    assert len(state.quarantine) == 1
    entry = state.quarantine[0]
    assert entry.node_id == plan.chunks[1].chunk_id
    assert entry.reason == "stale-identity"

    # The quarantine entry's raw JSON is the rejected node re-serialized,
    # never the corrected/expected version, and never enters recovered-work
    # counts (only the retained node contributes to completed_ids above).
    raw = json.loads(entry.raw_json)
    assert raw["node_id"] == plan.chunks[1].chunk_id
    assert raw["execution_identity_digest"] == "division-execution:" + "9" * 64


def test_one_files_stale_recovery_state_does_not_abort_an_unrelated_file(tmp_path) -> None:
    """0.14.4: a file's stale-but-planned recovery state is quarantined and
    re-executed in isolation -- it must never abort planning or execution
    for a completely unrelated file in the same run."""
    config, plan, _tree, _provider_identity, _content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=True
    )
    (tmp_path / "other.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    file_map = {
        **_file_map(tmp_path, "main.py"),
        **_file_map(tmp_path, "other.py"),
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    graph.add_file("other.py")

    plan_result, materials = build_pipeline_plan(
        file_map, graph, {"main.py", "other.py"}, "main.py", {}, [], config,
        recovered_partials={"main.py": recovered},
    )

    # main.py: one retained leaf, one quarantined -- exactly as the
    # single-file case above, proving nothing about it changed.
    state = materials.tree_states["main.py"]
    assert len(state.quarantine) == 1
    assert plan.chunks[0].chunk_id in set(state.by_id())

    # other.py: no recovery state at all, planned as an ordinary fresh split
    # file -- never quarantined, never blocked, never even aware main.py's
    # recovery state exists.
    assert "other.py" not in materials.tree_states
    assert "other.py" not in plan_result.division_blocked
    assert "other.py" in plan_result.agent_rels or "other.py" in plan_result.division_plan_rels


def test_quarantine_round_trips_through_the_recovery_file_and_clears_on_replacement(tmp_path) -> None:
    _config, plan, tree, provider_identity, content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=True
    )
    retained_nodes, quarantine_entries = validate_recovered_tree(
        recovered.nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=deterministic_imports_digest(()),
        language="python",
    )
    state = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=retained_nodes,
        quarantine=quarantine_entries,
    )

    writer = SafeWriter(tmp_path / "docs" / "crash_recovery.json", "json", None, {})
    writer.load(preloaded_partials={"main.py": state})
    writer.initialize_empty()

    on_disk = json.loads((tmp_path / "docs" / "crash_recovery.json").read_text(encoding="utf-8"))
    partial = on_disk["_codedoc"]["partial_files"]["main.py"]
    assert len(partial["quarantine"]) == 1
    assert partial["quarantine"][0]["reason"] == "stale-identity"
    assert partial["quarantine"][0]["node_id"] == plan.chunks[1].chunk_id

    # A valid replacement for the same node ID clears its quarantine entry.
    corrected = tree_node_state(
        node_id=plan.chunks[1].chunk_id,
        node_type="leaf",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=leaf_input_digest(
            rel_path=plan.rel_path,
            language="python",
            chunk=plan.chunks[1],
            unit_indexes=plan.unit_positions(plan.chunks[1]),
            unit_count=len(plan.units),
        ),
        execution_identity_digest=leaf_execution_identity(
            rel_path="main.py",
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=plan.chunks[1],
        ),
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(plan.chunks[1].chunk_id,),
        result={
            "description": "leaf 1 (corrected)",
            "chunk_id": plan.chunks[1].chunk_id,
            "unit_id": plan.chunks[1].unit_id,
        },
    )
    writer.record_tree_node(
        "main.py", corrected, reduction_tree_digest=tree.tree_digest
    )

    after = json.loads((tmp_path / "docs" / "crash_recovery.json").read_text(encoding="utf-8"))
    after_partial = after["_codedoc"]["partial_files"]["main.py"]
    assert "quarantine" not in after_partial
    assert len(after_partial["nodes"]) == 2


def test_stale_checkpoint_above_the_old_32_bound_recovers_instead_of_aborting(
    tmp_path,
) -> None:
    """0.14.4: the exact scenario the raised quarantine bound fixes. Advancing
    the leaf/reducer revisions invalidates every node of an existing schema-4
    checkpoint; a file with more than the pre-0.14.4 bound (32) of stale leaf
    nodes must be quarantined and re-executed within the new 512 bound rather
    than raising SplitRecoveryStateError and aborting the whole run."""
    source = "\n".join(f"value_{i} = {i}" for i in range(400)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=100
    )
    # A small division budget forces many leaf chunks; the reduction tree's
    # own ceiling is independent and only needs to be large enough to fit the
    # narrative fan-in -- unrelated to the number of leaves being proven here.
    tree = build_reduction_tree(plan, max_content_chars=2000, language="python")
    assert len(plan.chunks) > 32, "fixture must exceed the pre-0.14.4 bound to prove the fix"

    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    provider_identity = "provider-execution:" + "b" * 64

    # Every leaf carries a deliberately wrong execution identity, simulating
    # every node of an existing checkpoint going stale after a leaf-capsule
    # revision advance (exactly leaf-capsule-v6 -> v7).
    nodes = [
        tree_node_state(
            node_id=chunk.chunk_id,
            node_type="leaf",
            rel_path="main.py",
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=leaf_input_digest(
                rel_path="main.py",
                language="python",
                chunk=chunk,
                unit_indexes=plan.unit_positions(chunk),
                unit_count=len(plan.units),
            ),
            execution_identity_digest="division-execution:" + "9" * 64,
            unit_id=None,
            child_ids=(),
            coverage_leaf_ids=(chunk.chunk_id,),
            result={"description": "stale", "chunk_id": chunk.chunk_id, "unit_id": chunk.unit_id},
        )
        for chunk in plan.chunks
    ]

    retained, quarantine_entries = validate_recovered_tree(
        nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=deterministic_imports_digest(()),
        language="python",
    )

    assert retained == ()
    assert len(quarantine_entries) == len(plan.chunks) > 32
    assert all(entry.reason == "stale-identity" for entry in quarantine_entries)

    # The quarantined container itself is a valid SplitTreeState under the
    # new bound -- this is what would have raised ValueError pre-0.14.4.
    state = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=retained,
        quarantine=quarantine_entries,
    )
    assert len(state.quarantine) == len(quarantine_entries)


def test_forced_split_file_carries_prior_checkpoint_without_reuse_or_counting(tmp_path) -> None:
    config, _plan, _tree, _provider_identity, content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=False
    )

    file_map = _file_map(tmp_path)
    graph = DependencyGraph()
    graph.add_file("main.py")

    plan_result, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, ["main.py"], config,
        recovered_partials={"main.py": recovered},
    )

    # Forced: bypasses reuse/recovery for scheduling (nothing retained), but
    # the structurally valid recovered container is preserved as carry state.
    assert "main.py" not in materials.tree_states
    assert materials.carry_states["main.py"] is recovered
    assert "main.py" in plan_result.unpaid_action_rels

    writer = SafeWriter(tmp_path / "docs" / "crash_recovery.json", "json", None, {})
    writer.load(preloaded_carry_partials=dict(materials.carry_states))
    assert writer.has_partial_state()
    assert writer.get_tree_state("main.py") is None

    writer.initialize_empty()
    on_disk = json.loads((tmp_path / "docs" / "crash_recovery.json").read_text(encoding="utf-8"))
    partial = on_disk["_codedoc"]["partial_files"]["main.py"]
    assert len(partial["nodes"]) == 2

    # The forced file's own completed record clears its carry state, exactly
    # like a retained partial.
    writer.record("main.py", {"description": "forced fresh result"}, content_hash)
    assert not writer.has_partial_state()
    after = json.loads((tmp_path / "docs" / "crash_recovery.json").read_text(encoding="utf-8"))
    assert "partial_files" not in after["_codedoc"]


def test_forced_carry_state_survives_a_failed_run_and_keeps_the_recovery_file_alive(tmp_path) -> None:
    _config, _plan, _tree, _provider_identity, _content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=False
    )

    writer = SafeWriter(tmp_path / "docs" / "crash_recovery.json", "json", None, {})
    writer.load(preloaded_carry_partials={"main.py": recovered})
    writer.initialize_empty()

    # The forced file never completes this run (simulated failure): carry
    # state is untouched, so the recovery file must survive a would-be
    # clean-completion delete check.
    assert writer.has_partial_state()
    on_disk = json.loads(
        (tmp_path / "docs" / "crash_recovery.json").read_text(encoding="utf-8")
    )
    partial = on_disk["_codedoc"]["partial_files"]["main.py"]
    assert len(partial["nodes"]) == 2
    assert "quarantine" not in partial


def test_recovery_file_with_an_over_bound_quarantine_map_is_preserved_and_makes_no_call(
    tmp_path, monkeypatch
) -> None:
    """0.14.4 audit fix: the plan-required recovery-*loading* integration
    regression. The dataclass and validation layers (above) prove the bound
    in isolation; this drives an actual over-bound container -- one whose
    on-disk quarantine array already exceeds MAX_QUARANTINE_ENTRIES_PER_FILE
    -- through the real pipeline. A quarantine map that exceeds the bound
    must still raise and stop the run: the original recovery file must stay
    byte-identical and zero provider calls may occur."""
    _config, plan, tree, provider_identity, content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=True
    )
    retained_nodes, quarantine_entries = validate_recovered_tree(
        recovered.nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=deterministic_imports_digest(()),
        language="python",
    )
    state = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=retained_nodes,
        quarantine=quarantine_entries,
    )

    recovery_path = tmp_path / "codedoc" / "crash_recovery.json"
    writer = SafeWriter(recovery_path, "json", None, {})
    writer.load(preloaded_partials={"main.py": state})
    writer.initialize_empty()

    # Tamper the on-disk container past the bound -- this cannot be produced
    # by constructing SplitTreeState/QuarantineEntry in Python (the
    # dataclass's own __post_init__ already refuses it, per the layer-2 unit
    # test), so it is written directly, simulating a corrupted or
    # hand-edited recovery file.
    raw = json.loads(recovery_path.read_text(encoding="utf-8"))
    raw["_codedoc"]["partial_files"]["main.py"]["quarantine"] = [
        {"node_id": f"chunk_{index:04d}".ljust(64, "0"), "reason": "stale-revision", "raw_json": "{}"}
        for index in range(MAX_QUARANTINE_ENTRIES_PER_FILE + 1)
    ]
    recovery_path.write_text(json.dumps(raw), encoding="utf-8")
    before_bytes = recovery_path.read_bytes()

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail(
            "an over-bound recovery container must never reach provider creation"
        ),
    )

    with pytest.raises(ConfigError, match="malformed split-partial container"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "analysis_mode": "single",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "propagate_changes": False,
            },
        )

    assert recovery_path.read_bytes() == before_bytes
