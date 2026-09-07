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

import codedoc.core.file_division as file_division
import codedoc.core.planning as planning_mod
import codedoc.core.record_meta as record_meta
from codedoc.core.file_division import (
    MAX_QUARANTINE_ENTRIES_PER_FILE,
    MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    QuarantineEntry,
    SplitTreeState,
    build_division_plan,
    build_fact_ledger,
    build_reduction_tree,
    deterministic_imports_digest,
    final_execution_identity,
    final_input_digest,
    final_synthesis_input,
    leaf_execution_identity,
    leaf_input_digest,
    provider_execution_identity,
    reduction_execution_identity,
    reduction_input_digest,
    refine_narrative_inputs,
    tree_node_state,
    validate_recovered_tree,
)
from codedoc.core.execution_model import build_call_manifest
from codedoc.core.result_assembly import flat_combined_result
from codedoc.core.graph import DependencyGraph
from codedoc.core.loader import load_config
from codedoc.core.planning import build_pipeline_plan
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.structure_extra import requires_structure_pack


def _effective_synthesis(budget: int) -> int:
    """The automatic synthesis budget production planning carries for *budget*."""
    return max(budget, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS)


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
    # Build the reconstructed tree with the exact synthesis budget production
    # planning carries (the automatic 12,000 floor), so a same-plan recovered
    # container's reduction_tree_digest matches what build_pipeline_plan
    # computes -- otherwise the cross-plan fresh-preserve predicate would
    # (correctly) treat this same-plan fixture as a transition.
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=_effective_synthesis(2000), language="python"
    )
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


# ---------------------------------------------------------------------------
# 0.14.6: the leaf-capsule-v7 -> v8 revision transition
# ---------------------------------------------------------------------------
# The predecessor state below is never hand-authored. It is produced by the
# production identity functions themselves with the module constant patched
# back to "leaf-capsule-v7", then validated with the patch undone. A fabricated
# "wrong digest" would be insensitive to the constant and would still pass with
# the v8 advance reverted, proving nothing about the invalidation.


def _leaf_nodes_for_every_chunk(
    plan, *, content_hash: str, provider_identity: str
) -> tuple:
    """One checkpointed leaf per planned chunk, exactly as a live run writes
    them under whatever `LEAF_CAPSULE_SCHEMA_REVISION` is in force."""
    return tuple(
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
                "description": f"leaf {index}",
                "chunk_id": chunk.chunk_id,
                "unit_id": chunk.unit_id,
            },
        )
        for index, chunk in enumerate(plan.chunks)
    )


def test_leaf_capsule_v7_partial_is_owned_but_stale_and_re_executes(
    tmp_path, monkeypatch
) -> None:
    """0.14.6: a schema-4 partial written by 0.14.5 stays a valid owned
    container, but every `leaf-capsule-v7` leaf in it is stale.

    Two directions, so this cannot pass vacuously. First the genuine v7
    checkpoints are shown to be *retained* while the constant reads
    `leaf-capsule-v7` -- proving they are real predecessor state, not junk.
    Then, under the real current `leaf-capsule-v8`, the identical nodes are
    quarantined under the closed reason `stale-identity` and re-executed,
    within the existing `MAX_QUARANTINE_ENTRIES_PER_FILE` bound and with no
    schema-version change.
    """
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(plan, max_content_chars=2000, language="python")
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert len(plan.chunks) >= 2

    # The current revision's *value* is owned by
    # `test_only_the_leaf_capsule_revision_advanced_for_0_14_6`. Pinning it
    # here too would make this proof short-circuit on a literal instead of on
    # the staleness behaviour it exists to demonstrate.
    monkeypatch.setattr(
        file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v7"
    )
    predecessor_nodes = _leaf_nodes_for_every_chunk(
        plan, content_hash=content_hash, provider_identity=provider_identity
    )

    # Direction one: genuine, currently-valid v7 state.
    retained_v7, quarantine_v7 = validate_recovered_tree(
        predecessor_nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=config.get("_prompt_profile_digest", ""),
        imports_digest=deterministic_imports_digest(()),
        language="python",
    )
    assert quarantine_v7 == ()
    assert len(retained_v7) == len(plan.chunks)

    # Direction two: the same bytes under the real current revision.
    monkeypatch.undo()

    retained_v8, quarantine_v8 = validate_recovered_tree(
        predecessor_nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=config.get("_prompt_profile_digest", ""),
        imports_digest=deterministic_imports_digest(()),
        language="python",
    )
    assert retained_v8 == ()
    assert len(quarantine_v8) == len(plan.chunks)
    assert all(entry.reason == "stale-identity" for entry in quarantine_v8)
    assert len(quarantine_v8) <= MAX_QUARANTINE_ENTRIES_PER_FILE

    # Owned-but-stale, not rejected: the container itself is still a valid
    # schema-4 state carrying the quarantined nodes, with nothing to reuse.
    state = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=retained_v8,
        quarantine=quarantine_v8,
    )
    assert state.schema_version == SPLIT_PARTIAL_SCHEMA_VERSION == 4
    assert state.nodes == ()
    assert tuple(entry.node_id for entry in state.quarantine) == tuple(
        chunk.chunk_id for chunk in plan.chunks
    )


# ===========================================================================
# 0.14.7 section 6.3 / 5.7: cross-plan fresh-preserve recovery core
# ===========================================================================
# A schema-4 predecessor whose content hash, division-plan digest, OR
# reduction-tree digest no longer matches the current plan is a cross-plan
# transition. It is handled BEFORE validate_recovered_tree(): the predecessor
# container is carried byte-for-byte, zero predecessor nodes are reused, and
# the whole current split is scheduled fresh. Old-plan node IDs are never fed
# to current validation. This is distinct from the same-plan stale-identity
# quarantine path exercised above.

_OLD_REVISIONS = {
    "PACKER_SCHEMA_REVISION": "division-packer-v5",
    "LEAF_CAPSULE_SCHEMA_REVISION": "leaf-capsule-v8",
    "REDUCTION_PACKING_REVISION": "reduction-packing-v4",
    "REDUCER_PROMPT_REVISION": "file-reduction-v2",
}


def _leaf_result(chunk, index):
    return {
        "description": f"leaf {index}",
        "chunk_id": chunk.chunk_id,
        "unit_id": chunk.unit_id,
    }


def _reducer_result(node):
    return {"narrative": f"reduced narrative {node.phase} {node.ordinal}"}


def _leaf_node_for(
    chunk, *, plan, rel_path, content_hash, provider_identity, index
):
    return tree_node_state(
        node_id=chunk.chunk_id,
        node_type="leaf",
        rel_path=rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=leaf_input_digest(
            rel_path=rel_path,
            language="python",
            chunk=chunk,
            unit_indexes=plan.unit_positions(chunk),
            unit_count=len(plan.units),
        ),
        execution_identity_digest=leaf_execution_identity(
            rel_path=rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=chunk,
        ),
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(chunk.chunk_id,),
        result=_leaf_result(chunk, index),
    )


def _leaf_results_for(plan):
    """The ordered leaf-capsule result dicts, keyed by chunk id -- exactly the
    objects ``validate_recovered_tree`` reloads from each retained leaf node."""
    return {
        chunk.chunk_id: _leaf_result(chunk, index)
        for index, chunk in enumerate(plan.chunks)
    }


def _predecessor_split_state(
    monkeypatch,
    *,
    rel_path,
    source,
    source_budget,
    content_hash,
    provider_identity,
    patched_revisions=(),
    raw_synthesis=False,
    extra_quarantine_phases=(),
):
    """A genuine schema-4 ``SplitTreeState`` whose plan digest, tree digest,
    leaf chunk IDs, and leaf execution identities are all produced by the
    production functions, with *patched_revisions* (keys of ``_OLD_REVISIONS``)
    monkeypatched back to their pre-0.14.7 values only for the build and then
    restored. Nothing is hand-authored.

    ``raw_synthesis`` builds the predecessor tree from the raw source budget
    through the deprecated ``max_content_chars`` alias, reproducing the coupled
    pre-0.14.7 synthesis sizing. ``extra_quarantine_phases`` appends one
    bounded ``QuarantineEntry`` per requested reducer phase, so the container's
    unique predecessor ID count (nodes ∪ quarantine) can exceed its leaf count
    for a topology-contraction proof.
    """
    with monkeypatch.context() as mp:
        for name in patched_revisions:
            mp.setattr(file_division, name, _OLD_REVISIONS[name])
        plan = build_division_plan(
            rel_path=rel_path,
            language="python",
            content=source,
            source_budget_chars=source_budget,
        )
        if raw_synthesis:
            tree = build_reduction_tree(
                plan, max_content_chars=source_budget, language="python"
            )
        else:
            tree = build_reduction_tree(
                plan,
                synthesis_manifest_chars=_effective_synthesis(source_budget),
                language="python",
            )
        nodes = tuple(
            _leaf_node_for(
                chunk,
                plan=plan,
                rel_path=rel_path,
                content_hash=content_hash,
                provider_identity=provider_identity,
                index=index,
            )
            for index, chunk in enumerate(plan.chunks)
        )
        used: set[str] = set()
        quarantine = []
        pool = (
            list(tree.unit_consolidation_nodes)
            + list(tree.general_nodes)
            + [tree.final_node]
        )
        for phase in extra_quarantine_phases:
            node = next(
                candidate
                for candidate in pool
                if candidate.phase == phase and candidate.node_id not in used
            )
            used.add(node.node_id)
            quarantine.append(
                QuarantineEntry(
                    node_id=node.node_id,
                    reason="input-digest-mismatch",
                    raw_json="{}",
                )
            )
        state = SplitTreeState(
            schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
            owner="codedoc-ai",
            rel_path=rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            nodes=nodes,
            quarantine=tuple(quarantine),
        )
    return plan, tree, state


def _current_split(rel_path, source, budget):
    plan = build_division_plan(
        rel_path=rel_path,
        language="python",
        content=source,
        source_budget_chars=budget,
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=_effective_synthesis(budget),
        language="python",
    )
    return plan, tree


def _current_planned_ids(plan, tree):
    return {chunk.chunk_id for chunk in plan.chunks} | {
        node.node_id for node in tree.all_nodes
    }


def _no_validation_sentinel(*_args, **_kwargs):
    raise AssertionError(
        "validate_recovered_tree must not run for a cross-plan transition -- "
        "old-plan node IDs would reach current-plan validation."
    )


def _plan_with_recovered(tmp_path, config, recovered, rel_path="main.py"):
    file_map = _file_map(tmp_path, rel_path)
    graph = DependencyGraph()
    graph.add_file(rel_path)
    return build_pipeline_plan(
        file_map, graph, {rel_path}, rel_path, {}, [], config,
        recovered_partials={rel_path: recovered},
    )


def test_cross_plan_content_hash_conflict_carries_before_validation(
    tmp_path, monkeypatch
) -> None:
    """A predecessor whose content hash differs -- plan and tree digests still
    equal -- enters cross-plan carry before validate_recovered_tree runs."""
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    current_plan, current_tree = _current_split("main.py", source, 2000)

    stale_hash = hashlib.sha256(b"a completely different source revision").hexdigest()
    plan_old, tree_old, recovered = _predecessor_split_state(
        monkeypatch,
        rel_path="main.py",
        source=source,
        source_budget=2000,
        content_hash=stale_hash,
        provider_identity=provider_identity,
    )
    # Only the content hash differs.
    assert plan_old.plan_digest == current_plan.plan_digest
    assert tree_old.tree_digest == current_tree.tree_digest
    assert recovered.content_hash != hashlib.sha256(source.encode("utf-8")).hexdigest()

    monkeypatch.setattr(planning_mod, "validate_recovered_tree", _no_validation_sentinel)
    plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    assert materials.carry_states["main.py"] is recovered
    assert "main.py" not in materials.tree_states
    assert materials.recovery_conflict_files == 1
    assert materials.recovery_discarded_predecessor_nodes == len(recovered.nodes)
    current_ids = _current_planned_ids(current_plan, current_tree)
    assert materials.recovery_replacement_nodes_planned == len(current_ids)
    # Plan and tree digests are preserved here, so the two predecessor leaf IDs
    # ARE current unpaid IDs -- the persisted reexecuted count records exactly
    # those, and never exceeds the replacement (unpaid) count (section 6.3).
    predecessor_ids = {node.node_id for node in recovered.nodes}
    assert materials.reexecuted_nodes == len(predecessor_ids & current_ids)
    assert materials.reexecuted_nodes == len(recovered.nodes)
    assert materials.reexecuted_nodes <= materials.recovery_replacement_nodes_planned
    assert "main.py" in plan_result.division_plan_rels
    assert "main.py" not in plan_result.completed_split_reuse_rels


def test_cross_plan_division_plan_digest_conflict_carries_before_validation(
    tmp_path, monkeypatch
) -> None:
    """A predecessor built under ``division-packer-v5`` -- same file content --
    has a different plan digest and enters carry before validation."""
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    current_plan, current_tree = _current_split("main.py", source, 2000)

    plan_old, _tree_old, recovered = _predecessor_split_state(
        monkeypatch,
        rel_path="main.py",
        source=source,
        source_budget=2000,
        content_hash=content_hash,
        provider_identity=provider_identity,
        patched_revisions=("PACKER_SCHEMA_REVISION",),
    )
    assert recovered.content_hash == content_hash
    assert plan_old.plan_digest != current_plan.plan_digest
    assert recovered.division_plan_digest != current_plan.plan_digest

    monkeypatch.setattr(planning_mod, "validate_recovered_tree", _no_validation_sentinel)
    plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    assert materials.carry_states["main.py"] is recovered
    assert "main.py" not in materials.tree_states
    assert materials.recovery_conflict_files == 1
    predecessor_ids = {node.node_id for node in recovered.nodes} | {
        entry.node_id for entry in recovered.quarantine
    }
    assert materials.recovery_discarded_predecessor_nodes == len(predecessor_ids)
    assert materials.recovery_replacement_nodes_planned == len(
        _current_planned_ids(current_plan, current_tree)
    )
    # Cross-plan: no old-plan ID collides with a current-plan ID.
    assert predecessor_ids & _current_planned_ids(current_plan, current_tree) == set()
    assert materials.reexecuted_nodes == 0
    assert "main.py" in plan_result.division_plan_rels


def test_cross_plan_reduction_tree_digest_conflict_carries_before_validation(
    tmp_path, monkeypatch
) -> None:
    """A predecessor with the SAME content hash and division-plan digest but a
    reduction-tree digest built under ``reduction-packing-v4`` still enters
    cross-plan carry before validation (tree identities bind the packing
    revision and the carried manifest budget)."""
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    current_plan, current_tree = _current_split("main.py", source, 2000)

    # The predecessor tree was sized under the coupled pre-0.14.7 raw synthesis
    # value (source budget), so its tree digest differs from the current tree,
    # which carries the automatic 12,000 floor -- tree identities bind the
    # carried manifest budget (section 6.3). Plan digest and content are equal.
    plan_old, tree_old, recovered = _predecessor_split_state(
        monkeypatch,
        rel_path="main.py",
        source=source,
        source_budget=2000,
        content_hash=content_hash,
        provider_identity=provider_identity,
        raw_synthesis=True,
    )
    assert recovered.content_hash == content_hash
    assert plan_old.plan_digest == current_plan.plan_digest
    assert recovered.division_plan_digest == current_plan.plan_digest
    assert tree_old.synthesis_manifest_chars == 2000
    assert current_tree.synthesis_manifest_chars == 12000
    assert tree_old.tree_digest != current_tree.tree_digest
    assert recovered.reduction_tree_digest != current_tree.tree_digest

    monkeypatch.setattr(planning_mod, "validate_recovered_tree", _no_validation_sentinel)
    plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    assert materials.carry_states["main.py"] is recovered
    assert "main.py" not in materials.tree_states
    assert materials.recovery_conflict_files == 1
    assert materials.recovery_discarded_predecessor_nodes == len(recovered.nodes)
    current_ids = _current_planned_ids(current_plan, current_tree)
    assert materials.recovery_replacement_nodes_planned == len(current_ids)
    # Same plan digest -> the predecessor leaf IDs equal current chunk IDs, so
    # the persisted reexecuted count records exactly those and stays within the
    # replacement (unpaid) count. Only the tree digest moved.
    assert materials.reexecuted_nodes == len(
        {node.node_id for node in recovered.nodes} & current_ids
    )
    assert materials.reexecuted_nodes == len(recovered.nodes)
    assert materials.reexecuted_nodes <= materials.recovery_replacement_nodes_planned
    assert "main.py" in plan_result.division_plan_rels


def test_cross_plan_contraction_reports_zero_reexecuted_four_discarded_three_replacement(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3 topology contraction: an old plan with four unique paid-or-
    quarantined node IDs cannot be mapped onto a smaller current plan with
    three nodes. It reports honest counts instead of crashing:
    0 reexecuted / 4 discarded / 3 replacement."""
    predecessor_source = _large_source()  # 220 lines -> 2 leaves + 1 uc + 1 final
    current_source = (
        "def alpha():\n" + "    a = 1\n" * 90 + "\ndef beta():\n" + "    b = 2\n" * 90 + "\n"
    )
    (tmp_path / "main.py").write_text(current_source, encoding="utf-8", newline="")
    config = _split_config(tmp_path, max_chars=1200)
    provider_identity = provider_execution_identity(config)

    current_plan, current_tree = _current_split("main.py", current_source, 1200)
    current_ids = _current_planned_ids(current_plan, current_tree)
    assert len(current_ids) == 3  # 2 leaves + 1 final, no reducer

    _plan_old, _tree_old, recovered = _predecessor_split_state(
        monkeypatch,
        rel_path="main.py",
        source=predecessor_source,
        source_budget=2000,
        content_hash=hashlib.sha256(predecessor_source.encode("utf-8")).hexdigest(),
        provider_identity=provider_identity,
        extra_quarantine_phases=("unit-consolidation", "final"),
    )
    predecessor_ids = {node.node_id for node in recovered.nodes} | {
        entry.node_id for entry in recovered.quarantine
    }
    assert len(predecessor_ids) == 4  # 2 paid leaves + uc + final quarantined

    monkeypatch.setattr(planning_mod, "validate_recovered_tree", _no_validation_sentinel)
    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    assert materials.carry_states["main.py"] is recovered
    assert materials.reexecuted_nodes == 0
    assert materials.recovery_discarded_predecessor_nodes == 4
    assert materials.recovery_replacement_nodes_planned == 3
    # The persisted re-executed count never exceeds current unpaid nodes.
    assert materials.reexecuted_nodes <= materials.recovery_replacement_nodes_planned


def test_cross_plan_expansion_discards_fewer_than_it_replaces(
    tmp_path, monkeypatch
) -> None:
    """The mirror of the contraction case: a small old plan expands into a
    larger current one. Discarded < replacement, still zero reexecuted."""
    predecessor_source = (
        "def alpha():\n" + "    a = 1\n" * 90 + "\ndef beta():\n" + "    b = 2\n" * 90 + "\n"
    )
    current_source = "\n".join(f"value_{i} = {i}" for i in range(400)) + "\n"
    (tmp_path / "main.py").write_text(current_source, encoding="utf-8", newline="")
    config = _split_config(tmp_path, max_chars=1000)
    provider_identity = provider_execution_identity(config)

    current_plan, current_tree = _current_split("main.py", current_source, 1000)
    current_ids = _current_planned_ids(current_plan, current_tree)

    _plan_old, _tree_old, recovered = _predecessor_split_state(
        monkeypatch,
        rel_path="main.py",
        source=predecessor_source,
        source_budget=1200,
        content_hash=hashlib.sha256(predecessor_source.encode("utf-8")).hexdigest(),
        provider_identity=provider_identity,
        extra_quarantine_phases=("final",),
    )
    predecessor_ids = {node.node_id for node in recovered.nodes} | {
        entry.node_id for entry in recovered.quarantine
    }

    monkeypatch.setattr(planning_mod, "validate_recovered_tree", _no_validation_sentinel)
    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    assert materials.carry_states["main.py"] is recovered
    assert materials.recovery_discarded_predecessor_nodes == len(predecessor_ids)
    assert materials.recovery_replacement_nodes_planned == len(current_ids)
    assert (
        materials.recovery_discarded_predecessor_nodes
        < materials.recovery_replacement_nodes_planned
    )
    assert materials.reexecuted_nodes == 0


def test_forced_carry_is_not_a_cross_plan_transition_and_does_not_count(
    tmp_path,
) -> None:
    """Forcing a split file carries its prior checkpoint, but that is force
    preservation, not a cross-plan transition: neither transition counter
    moves, and no recovery conflict is recorded."""
    config, _plan, _tree, _provider_identity, _content_hash, recovered = _split_fixture(
        tmp_path, corrupt_second=False
    )
    file_map = _file_map(tmp_path)
    graph = DependencyGraph()
    graph.add_file("main.py")

    _plan_result, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, ["main.py"], config,
        recovered_partials={"main.py": recovered},
    )

    assert materials.carry_states["main.py"] is recovered
    assert materials.recovery_discarded_predecessor_nodes == 0
    assert materials.recovery_replacement_nodes_planned == 0
    assert materials.recovery_conflict_files == 0
    assert materials.reexecuted_nodes == 0


def test_current_digest_container_with_unplanned_id_still_hard_fails_closed(
    tmp_path,
) -> None:
    """A container that claims the CURRENT content/plan/tree digests but carries
    an unplanned node ID is a hard fail-closed error -- the cross-plan branch
    must not launder malformed current recovery into carry state."""
    config, plan, tree, provider_identity, content_hash, valid = _split_fixture(
        tmp_path, corrupt_second=False
    )
    foreign_leaf = _leaf_node_for(
        plan.chunks[0],
        plan=plan,
        rel_path="main.py",
        content_hash=content_hash,
        provider_identity=provider_identity,
        index=0,
    )
    # A node ID that is not any planned chunk or tree node.
    object.__setattr__(foreign_leaf, "node_id", "node_" + "f" * 64)
    object.__setattr__(foreign_leaf, "coverage_leaf_ids", ("node_" + "f" * 64,))
    malformed = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=(valid.nodes[0], foreign_leaf),
    )

    file_map = _file_map(tmp_path)
    graph = DependencyGraph()
    graph.add_file("main.py")
    with pytest.raises(ConfigError, match="cannot be safely bounded"):
        build_pipeline_plan(
            file_map, graph, {"main.py"}, "main.py", {}, [], config,
            recovered_partials={"main.py": malformed},
        )

    # An unplanned *quarantine* ID under current digests is equally fatal --
    # never laundered into carry state.
    bad_quarantine = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=(valid.nodes[0],),
        quarantine=(
            QuarantineEntry(
                node_id="node_" + "e" * 64,
                reason="stale-identity",
                raw_json="{}",
            ),
        ),
    )
    with pytest.raises(ConfigError, match="cannot be safely bounded"):
        build_pipeline_plan(
            file_map, graph, {"main.py"}, "main.py", {}, [], config,
            recovered_partials={"main.py": bad_quarantine},
        )


def _child_narratives(child_results, child_ids):
    """Extract per-child narrative text exactly as production
    ``validate_recovered_tree`` does: ``narrative`` then ``description``."""
    return tuple(
        child_results[cid].get("narrative", child_results[cid].get("description", ""))
        for cid in child_ids
    )


def _reducer_node_for(
    node, *, plan, rel_path, content_hash, provider_identity, tree_digest,
    child_results,
):
    """A paid reducer checkpoint whose stage-local input digest is recomputed
    from its exact ordered child results -- ``refine_narrative_inputs`` wrapped,
    mirroring ``validate_recovered_tree`` (section 11)."""
    return tree_node_state(
        node_id=node.node_id,
        node_type=node.phase,
        rel_path=rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=reduction_input_digest(
            rel_path=rel_path,
            phase=node.phase,
            level=node.level,
            unit_id=node.unit_id,
            child_count=len(node.child_ids),
            ordered_child_narratives=refine_narrative_inputs(
                _child_narratives(child_results, node.child_ids)
            ),
        ),
        execution_identity_digest=reduction_execution_identity(
            rel_path=rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree_digest,
            provider_identity=provider_identity,
            node=node,
        ),
        unit_id=node.unit_id,
        child_ids=node.child_ids,
        coverage_leaf_ids=node.leaf_ids,
        result=_reducer_result(node),
    )


def _final_node_for(
    tree, *, plan, rel_path, content_hash, provider_identity,
    leaf_results, child_results, final_description="predecessor final synthesis",
):
    """A paid final checkpoint whose expected input digest is derived exactly
    as production ``validate_recovered_tree`` derives it: ordered leaf capsules
    (``plan.chunks`` order) -> ``build_fact_ledger`` -> final-child narratives
    (``final_node.child_ids`` order) -> ``refine_narrative_inputs`` ->
    ``final_synthesis_input(max_chars=tree.synthesis_manifest_chars)`` ->
    ``final_input_digest``. The stored result is the live-cleaner shape from
    ``flat_combined_result``. Nothing is hand-authored."""
    node = tree.final_node
    imports_digest = deterministic_imports_digest(())
    leaf_capsules_ordered = [leaf_results[chunk.chunk_id] for chunk in plan.chunks]
    ledger = build_fact_ledger(
        leaf_capsules_ordered,
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )
    manifest_json = final_synthesis_input(
        rel_path=rel_path,
        language="python",
        imports=(),
        root_narratives=refine_narrative_inputs(
            _child_narratives(child_results, node.child_ids)
        ),
        root_coverage_leaf_ids=node.leaf_ids,
        ledger=ledger,
        max_chars=tree.synthesis_manifest_chars,
    )
    return tree_node_state(
        node_id=node.node_id,
        node_type="final",
        rel_path=rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=final_input_digest(
            imports_digest=imports_digest,
            resolved_shape_digest="no-prompt-profile-v1",
            manifest_json=manifest_json,
        ),
        execution_identity_digest=final_execution_identity(
            rel_path=rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            provider_identity=provider_identity,
            prompt_profile_digest="no-prompt-profile-v1",
            imports_digest=imports_digest,
            node=node,
        ),
        unit_id=None,
        child_ids=node.child_ids,
        coverage_leaf_ids=node.leaf_ids,
        result=flat_combined_result(
            rel_path, "python", [], {"description": final_description}
        ),
    )


def _full_current_valid_container(
    plan, tree, *, rel_path, content_hash, provider_identity,
):
    """A same-plan schema-4 container with every leaf, the reducer(s), and the
    final node checkpointed as current-valid, dependency-consistent state --
    every digest built by the production functions."""
    leaf_results = _leaf_results_for(plan)
    leaves = tuple(
        _leaf_node_for(
            chunk, plan=plan, rel_path=rel_path, content_hash=content_hash,
            provider_identity=provider_identity, index=index,
        )
        for index, chunk in enumerate(plan.chunks)
    )
    child_results = dict(leaf_results)
    reducers = []
    for node in tree.unit_consolidation_nodes + tree.general_nodes:
        reducers.append(
            _reducer_node_for(
                node, plan=plan, rel_path=rel_path, content_hash=content_hash,
                provider_identity=provider_identity, tree_digest=tree.tree_digest,
                child_results=child_results,
            )
        )
        child_results[node.node_id] = _reducer_result(node)
    final_node = _final_node_for(
        tree, plan=plan, rel_path=rel_path, content_hash=content_hash,
        provider_identity=provider_identity,
        leaf_results=leaf_results, child_results=child_results,
    )
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=leaves + tuple(reducers) + (final_node,),
    )


def test_positive_control_full_current_valid_container_is_fully_retained(
    tmp_path,
) -> None:
    """Positive control for section 6.3 dependency pruning: a same-plan
    schema-4 container whose leaves, reducer, and final checkpoints are ALL
    current-valid (every digest from the production functions) must be retained
    in full -- no cross-plan carry, empty quarantine, zero re-executed nodes.

    This proves the corrected ``_final_node_for`` produces a genuinely valid
    final checkpoint, so the dependency-pruning tests below are causal: their
    final is rejected only because a dependency was lost, not because it was
    independently malformed.
    """
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan, tree = _current_split("main.py", source, 2000)
    assert len(tree.unit_consolidation_nodes) == 1

    recovered = _full_current_valid_container(
        plan, tree, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity,
    )
    every_node_id = {node.node_id for node in recovered.nodes}
    assert every_node_id == (
        {chunk.chunk_id for chunk in plan.chunks}
        | {n.node_id for n in tree.all_nodes}
    )

    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    assert "main.py" not in materials.carry_states
    state = materials.tree_states["main.py"]
    assert set(state.by_id()) == every_node_id  # every node retained
    assert state.quarantine == ()
    assert materials.reexecuted_nodes == 0
    assert materials.recovery_discarded_predecessor_nodes == 0
    assert materials.recovery_replacement_nodes_planned == 0
    # Nothing left to pay for: the whole tree is a valid recovered checkpoint.
    assert "main.py" not in _plan_result.unpaid_action_rels
    manifest = build_call_manifest(
        [], _plan_result.agent_rels, "single",
        materials.division_plans, materials.reduction_trees, materials.tree_states,
    )
    assert [call for call in manifest.calls] == []


def test_same_plan_old_reducer_prompt_identity_quarantines_the_reducer_independently(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3: a v6/v5 container whose leaves are current-valid, whose
    final checkpoint is *demonstrably valid* (its input digest built exactly as
    production derives it -- proven by the positive control above), but whose
    reducer was genuinely paid under ``file-reduction-v2``. It does NOT enter
    carry; the reducer alone is quarantined ``stale-identity`` (independently of
    leaf revision); the previously-valid final is then pruned
    ``input-digest-mismatch`` *because its reducer dependency was lost*, not
    because it was independently malformed; and both are re-executed within the
    512-entry bound.
    """
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan, tree = _current_split("main.py", source, 2000)
    assert len(tree.unit_consolidation_nodes) == 1
    uc = tree.unit_consolidation_nodes[0]

    leaf_results = _leaf_results_for(plan)
    leaves = tuple(
        _leaf_node_for(
            chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
            provider_identity=provider_identity, index=index,
        )
        for index, chunk in enumerate(plan.chunks)
    )
    child_results = dict(leaf_results)
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "REDUCER_PROMPT_REVISION", "file-reduction-v2")
        assert file_division.REDUCER_PROMPT_REVISION == "file-reduction-v2"
        stale_reducer = _reducer_node_for(
            uc,
            plan=plan,
            rel_path="main.py",
            content_hash=content_hash,
            provider_identity=provider_identity,
            tree_digest=tree.tree_digest,
            child_results=child_results,
        )
    child_results[uc.node_id] = _reducer_result(uc)
    valid_final = _final_node_for(
        tree, plan=plan, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity,
        leaf_results=leaf_results, child_results=child_results,
    )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=leaves + (stale_reducer, valid_final),
    )

    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    state = materials.tree_states["main.py"]
    assert "main.py" not in materials.carry_states
    # Same-plan quarantine is not a cross-plan transition.
    assert materials.recovery_discarded_predecessor_nodes == 0
    assert materials.recovery_replacement_nodes_planned == 0

    retained_ids = set(state.by_id())
    leaf_ids = {chunk.chunk_id for chunk in plan.chunks}
    # All current leaves retained; only the reducer + final are rejected.
    assert retained_ids == leaf_ids
    reasons = {entry.node_id: entry.reason for entry in state.quarantine}
    assert reasons[uc.node_id] == "stale-identity"
    assert uc.node_id not in retained_ids
    assert tree.final_node.node_id not in retained_ids
    assert reasons[tree.final_node.node_id] == "input-digest-mismatch"
    assert len(state.quarantine) <= MAX_QUARANTINE_ENTRIES_PER_FILE
    # reexecuted == previously paid current node IDs not retained (reducer + final).
    assert materials.reexecuted_nodes == 2

    # The rejected reducer and final are real unpaid work in the current manifest.
    manifest = build_call_manifest(
        [], _plan_result.agent_rels, "single",
        materials.division_plans, materials.reduction_trees, materials.tree_states,
    )
    owners = {call.owner for call in manifest.calls}
    assert uc.node_id in owners
    assert any(
        call.category == "file-synthesis" and call.owner == "main.py"
        for call in manifest.calls
    )


def test_same_plan_old_leaf_prompt_identity_quarantines_leaves_and_prunes_dependents(
    tmp_path, monkeypatch
) -> None:
    """Section 6.3 same-plan path, leaf-revision case, with genuine paid
    dependents. One current-v6/v5 container carries: leaf checkpoints whose
    execution identities were really generated under ``leaf-capsule-v8``; a
    paid unit-consolidation reducer checkpoint; and a paid final checkpoint --
    all with current content / division-plan / reduction-tree digests, every
    identity and input digest built by the production functions.

    After the leaf-revision patch is undone: the container enters same-plan
    validation (never carry); the old-v8 leaves are quarantined
    ``stale-identity``; the checkpointed reducer and final nodes are pruned
    ``input-digest-mismatch`` and are not retained; the quarantine stays within
    ``MAX_QUARANTINE_ENTRIES_PER_FILE``; ``reexecuted_nodes`` equals the exact
    number of previously paid current node IDs that were not retained; every
    rejected node ID is scheduled as unpaid work in the current call manifest;
    and both cross-plan transition counters remain zero.
    """
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan, tree = _current_split("main.py", source, 2000)
    assert len(tree.unit_consolidation_nodes) == 1

    leaf_results = _leaf_results_for(plan)
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v8")
        stale_leaves = tuple(
            _leaf_node_for(
                chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
                provider_identity=provider_identity, index=index,
            )
            for index, chunk in enumerate(plan.chunks)
        )
    # Paid dependents built with production functions under the CURRENT
    # revisions -- their only defect is that every leaf beneath them is stale.
    # The final is demonstrably valid (proven by the positive control above).
    child_results = dict(leaf_results)
    uc = tree.unit_consolidation_nodes[0]
    paid_reducer = _reducer_node_for(
        uc,
        plan=plan,
        rel_path="main.py",
        content_hash=content_hash,
        provider_identity=provider_identity,
        tree_digest=tree.tree_digest,
        child_results=child_results,
    )
    child_results[uc.node_id] = _reducer_result(uc)
    paid_final = _final_node_for(
        tree, plan=plan, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity,
        leaf_results=leaf_results, child_results=child_results,
    )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=stale_leaves + (paid_reducer, paid_final),
    )
    assert {node.node_type for node in recovered.nodes} == {"leaf", "unit-consolidation", "final"}

    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)

    # Same-plan validation, not carry; the two transition counters stay zero.
    assert "main.py" not in materials.carry_states
    assert materials.recovery_discarded_predecessor_nodes == 0
    assert materials.recovery_replacement_nodes_planned == 0

    state = materials.tree_states["main.py"]
    assert set(state.by_id()) == set()  # nothing retained
    reasons = {entry.node_id: entry.reason for entry in state.quarantine}
    leaf_ids = {chunk.chunk_id for chunk in plan.chunks}
    reducer_id = tree.unit_consolidation_nodes[0].node_id
    final_id = tree.final_node.node_id
    assert {reasons[cid] for cid in leaf_ids} == {"stale-identity"}
    assert reasons[reducer_id] == "input-digest-mismatch"
    assert reasons[final_id] == "input-digest-mismatch"
    # Dependents are not retained.
    assert reducer_id not in state.by_id()
    assert final_id not in state.by_id()
    assert len(state.quarantine) <= MAX_QUARANTINE_ENTRIES_PER_FILE

    # reexecuted_nodes == exactly the previously-paid current node IDs not
    # retained (every leaf + reducer + final; nothing was retained).
    previously_paid = leaf_ids | {reducer_id, final_id}
    assert materials.reexecuted_nodes == len(previously_paid)

    # Every rejected node ID is real unpaid work in the current call manifest,
    # not merely absent from retained state.
    manifest = build_call_manifest(
        [], _plan_result.agent_rels, "single",
        materials.division_plans, materials.reduction_trees, materials.tree_states,
    )
    scheduled_owners = {call.owner for call in manifest.calls}
    assert leaf_ids <= scheduled_owners
    assert reducer_id in scheduled_owners
    assert any(
        call.category == "file-synthesis" and call.owner == "main.py"
        for call in manifest.calls
    )


# ===========================================================================
# 0.14.8 P2-2: full leaf/reducer/final recovery-node evidence for the reached
# ledger / final-synthesis / leaf-capsule identity advances.
#
# Every node in these containers is built by the production identity/input
# functions. Each scenario patches exactly the constant that owns the node it
# means to make stale, then validates through the real ``build_pipeline_plan``
# recovery path and asserts the exact retained / quarantined / re-executed set
# and the minimum scheduled calls -- not only a digest inequality.
# ===========================================================================


def _current_leaves_and_reducers(
    plan, tree, *, rel_path, content_hash, provider_identity
):
    """The ordered current-valid leaf nodes, the current-valid reducer nodes,
    and the ``child_results`` map a final node is derived from -- every digest
    from the production functions under the current revisions."""
    leaf_results = _leaf_results_for(plan)
    leaves = tuple(
        _leaf_node_for(
            chunk, plan=plan, rel_path=rel_path, content_hash=content_hash,
            provider_identity=provider_identity, index=index,
        )
        for index, chunk in enumerate(plan.chunks)
    )
    child_results = dict(leaf_results)
    reducers = []
    for node in tree.unit_consolidation_nodes + tree.general_nodes:
        reducers.append(
            _reducer_node_for(
                node, plan=plan, rel_path=rel_path, content_hash=content_hash,
                provider_identity=provider_identity, tree_digest=tree.tree_digest,
                child_results=child_results,
            )
        )
        child_results[node.node_id] = _reducer_result(node)
    return leaf_results, leaves, tuple(reducers), child_results


@pytest.mark.parametrize(
    "constant_name, prior_value",
    [
        ("LEDGER_SCHEMA_REVISION", "fact-ledger-v6"),
        ("FINAL_SYNTHESIS_REVISION", "file-synthesis-v3"),
    ],
)
def test_same_plan_stale_final_bound_revision_quarantines_only_the_final_node(
    tmp_path, monkeypatch, constant_name, prior_value
) -> None:
    """A same-plan schema-4 container whose leaves and reducer are current-valid
    but whose final checkpoint was genuinely paid under a predecessor ledger /
    final-synthesis revision: the container never enters carry; every leaf and
    every reducer is retained; only the final node is quarantined
    ``stale-identity``; final synthesis is scheduled exactly once; and reverting
    the production constant to that predecessor value makes the same final node
    a current, retained checkpoint (mutation sensitivity, finding P2-2 E)."""
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan, tree = _current_split("main.py", source, 2000)
    assert len(tree.unit_consolidation_nodes) == 1

    leaf_results, leaves, reducers, child_results = _current_leaves_and_reducers(
        plan, tree, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity,
    )
    leaf_ids = {chunk.chunk_id for chunk in plan.chunks}
    reducer_ids = {node.node_id for node in tree.all_intermediate_nodes}
    final_id = tree.final_node.node_id

    # Only the final checkpoint is stamped under the predecessor revision.
    with monkeypatch.context() as mp:
        mp.setattr(file_division, constant_name, prior_value)
        stale_final = _final_node_for(
            tree, plan=plan, rel_path="main.py", content_hash=content_hash,
            provider_identity=provider_identity, leaf_results=leaf_results,
            child_results=child_results,
        )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
        rel_path="main.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=leaves + reducers + (stale_final,),
    )

    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)
    state = materials.tree_states["main.py"]
    assert "main.py" not in materials.carry_states
    assert materials.recovery_discarded_predecessor_nodes == 0
    assert materials.recovery_replacement_nodes_planned == 0

    retained_ids = set(state.by_id())
    assert leaf_ids <= retained_ids            # every compatible leaf retained
    assert reducer_ids <= retained_ids         # every compatible reducer retained
    assert final_id not in retained_ids        # only the stale final rejected
    reasons = {entry.node_id: entry.reason for entry in state.quarantine}
    assert list(reasons) == [final_id]
    assert reasons[final_id] == "stale-identity"
    assert materials.reexecuted_nodes == 1     # exactly one final rerun

    manifest = build_call_manifest(
        [], _plan_result.agent_rels, "single",
        materials.division_plans, materials.reduction_trees, materials.tree_states,
    )
    synthesis_calls = [
        call for call in manifest.calls
        if call.category == "file-synthesis" and call.owner == "main.py"
    ]
    assert len(synthesis_calls) == 1
    assert not any(
        call.owner in leaf_ids or call.owner in reducer_ids
        for call in manifest.calls
    )

    # Mutation sensitivity: with the production constant reverted, the very same
    # final node matches the current identity and is retained -- so it is the
    # advance, not the container shape, that quarantined it above.
    with monkeypatch.context() as mp:
        mp.setattr(file_division, constant_name, prior_value)
        reverted = SplitTreeState(
            schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
            rel_path="main.py", content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            nodes=leaves + reducers + (stale_final,),
        )
        _pr2, materials2 = _plan_with_recovered(tmp_path, config, reverted)
    state2 = materials2.tree_states["main.py"]
    assert final_id in set(state2.by_id())
    assert state2.quarantine == ()
    assert materials2.reexecuted_nodes == 0


def test_same_plan_stale_leaves_prune_only_their_dependents_no_reducer_topology(
    tmp_path, monkeypatch
) -> None:
    """Leaf-capsule predecessor, complete tree, standalone-leaf topology: two of
    six current leaves are re-stamped under ``leaf-capsule-v9``. After the patch
    is undone: exactly those two are quarantined ``stale-identity``; the other
    four leaves stay retained; the final node -- whose dependency chain now has
    a hole -- is pruned ``input-digest-mismatch``; and no unrelated compatible
    node is invalidated."""
    parts = []
    for f in range(12):
        parts.append(f"def fn_{f}():")
        parts.extend(f"    a_{f}_{k} = {k}" for k in range(30))
        parts.append("")
    source = "\n".join(parts) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path, max_chars=1200)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=1200,
    )
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=_effective_synthesis(1200), language="python",
    )
    # A no-reducer topology: every leaf feeds the final node directly.
    assert tree.all_intermediate_nodes == ()
    assert len(plan.chunks) >= 4

    leaf_results = _leaf_results_for(plan)
    stale_chunks = plan.chunks[:2]
    stale_ids = {chunk.chunk_id for chunk in stale_chunks}

    with monkeypatch.context() as mp:
        mp.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v9")
        stale_leaves = tuple(
            _leaf_node_for(
                chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
                provider_identity=provider_identity, index=index,
            )
            for index, chunk in enumerate(stale_chunks)
        )
    current_leaves = tuple(
        _leaf_node_for(
            chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
            provider_identity=provider_identity, index=index,
        )
        for index, chunk in enumerate(plan.chunks)
        if chunk.chunk_id not in stale_ids
    )
    child_results = dict(leaf_results)
    final_node = _final_node_for(
        tree, plan=plan, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity, leaf_results=leaf_results,
        child_results=child_results,
    )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
        rel_path="main.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=stale_leaves + current_leaves + (final_node,),
    )

    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)
    state = materials.tree_states["main.py"]
    assert "main.py" not in materials.carry_states

    retained_ids = set(state.by_id())
    good_leaf_ids = {c.chunk_id for c in plan.chunks} - stale_ids
    assert retained_ids == good_leaf_ids       # only the compatible leaves
    reasons = {entry.node_id: entry.reason for entry in state.quarantine}
    assert {reasons[cid] for cid in stale_ids} == {"stale-identity"}
    assert reasons[tree.final_node.node_id] == "input-digest-mismatch"
    assert tree.final_node.node_id not in retained_ids


@requires_structure_pack
def test_same_plan_stale_leaves_prune_only_the_dependent_reducer(
    tmp_path, monkeypatch
) -> None:
    """Leaf-capsule predecessor, complete multi-reducer tree: a syntax-mode
    source of three functions fans out into disjoint unit-consolidation
    reducers. Only the leaves under the first reducer are re-stamped under
    ``leaf-capsule-v9``. After the patch is undone: those leaves are quarantined
    ``stale-identity``; their reducer is pruned ``input-digest-mismatch``; the
    other reducers and every leaf beneath them stay retained; the final node is
    pruned; no unrelated compatible node is invalidated."""
    parts = []
    for f in range(3):
        parts.append(f"def func_{f}():")
        parts.extend(f"    v_{f}_{b} = {b}" for b in range(90))
        parts.append("")
    source = "\n".join(parts) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path, max_chars=1000)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=1000,
    )
    tree = build_reduction_tree(
        plan, synthesis_manifest_chars=_effective_synthesis(1000), language="python",
    )
    assert plan.structural_mode == "syntax"
    assert len(tree.unit_consolidation_nodes) >= 2
    target_reducer = tree.unit_consolidation_nodes[0]
    other_reducers = tree.unit_consolidation_nodes[1:] + tree.general_nodes
    stale_ids = set(target_reducer.child_ids)
    assert stale_ids and stale_ids.issubset({c.chunk_id for c in plan.chunks})

    leaf_results = _leaf_results_for(plan)
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v9")
        stale_leaves = tuple(
            _leaf_node_for(
                chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
                provider_identity=provider_identity, index=index,
            )
            for index, chunk in enumerate(plan.chunks)
            if chunk.chunk_id in stale_ids
        )
    current_leaves = tuple(
        _leaf_node_for(
            chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
            provider_identity=provider_identity, index=index,
        )
        for index, chunk in enumerate(plan.chunks)
        if chunk.chunk_id not in stale_ids
    )
    child_results = dict(leaf_results)
    reducers = []
    for node in tree.unit_consolidation_nodes + tree.general_nodes:
        reducers.append(
            _reducer_node_for(
                node, plan=plan, rel_path="main.py", content_hash=content_hash,
                provider_identity=provider_identity, tree_digest=tree.tree_digest,
                child_results=child_results,
            )
        )
        child_results[node.node_id] = _reducer_result(node)
    final_node = _final_node_for(
        tree, plan=plan, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity, leaf_results=leaf_results,
        child_results=child_results,
    )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
        rel_path="main.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=stale_leaves + current_leaves + tuple(reducers) + (final_node,),
    )

    _plan_result, materials = _plan_with_recovered(tmp_path, config, recovered)
    state = materials.tree_states["main.py"]
    assert "main.py" not in materials.carry_states

    retained_ids = set(state.by_id())
    reasons = {entry.node_id: entry.reason for entry in state.quarantine}
    assert {reasons[cid] for cid in stale_ids} == {"stale-identity"}
    assert reasons[target_reducer.node_id] == "input-digest-mismatch"
    assert target_reducer.node_id not in retained_ids
    for node in other_reducers:
        assert node.node_id in retained_ids
    assert ({c.chunk_id for c in plan.chunks} - stale_ids) <= retained_ids
    assert tree.final_node.node_id not in retained_ids


def _plan_with_record_and_recovered(
    tmp_path, config, record, recovered, rel_path="main.py"
):
    file_map = _file_map(tmp_path, rel_path)
    graph = DependencyGraph()
    graph.add_file(rel_path)
    return build_pipeline_plan(
        file_map, graph, {rel_path}, rel_path, {rel_path: record}, [], config,
        recovered_partials={rel_path: recovered},
    )


def test_combined_predecessor_completed_record_and_stale_tree_reprocess_the_minimum(
    tmp_path, monkeypatch
) -> None:
    """P2-2 D: all four predecessor identities represented across their real
    ownership boundaries -- a completed ``file-doc-v3`` record, ``leaf-capsule-v9``
    leaf nodes, and a final node paid under both ``fact-ledger-v6`` and
    ``file-synthesis-v3``. Planning rejects the completed record; the stored
    tree is validated through its real nodes; the retained / quarantined set
    follows the dependency graph; and a current-valid regenerated container is
    then fully reusable with zero re-executed nodes."""
    source = _large_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = _split_config(tmp_path)
    provider_identity = provider_execution_identity(config)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    plan, tree = _current_split("main.py", source, 2000)
    assert len(tree.unit_consolidation_nodes) == 1

    stale_record = {
        "path": "main.py",
        "hash": content_hash,
        "language": "python",
        "description": "predecessor completed split",
        "_analysis_revision": "file-doc-v3",
        "_analysis_mode": "single",
        "_large_file_identity": record_meta.expected_large_file_identity(
            source_chars=len(source), max_chars=2000, rel_path="main.py",
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            structural_mode=plan.structural_mode,
            imports_digest=deterministic_imports_digest(()),
        ),
    }

    leaf_results = _leaf_results_for(plan)
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v9")
        stale_leaves = tuple(
            _leaf_node_for(
                chunk, plan=plan, rel_path="main.py", content_hash=content_hash,
                provider_identity=provider_identity, index=index,
            )
            for index, chunk in enumerate(plan.chunks)
        )
    child_results = dict(leaf_results)
    uc = tree.unit_consolidation_nodes[0]
    reducer = _reducer_node_for(
        uc, plan=plan, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity, tree_digest=tree.tree_digest,
        child_results=child_results,
    )
    child_results[uc.node_id] = _reducer_result(uc)
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "LEDGER_SCHEMA_REVISION", "fact-ledger-v6")
        mp.setattr(file_division, "FINAL_SYNTHESIS_REVISION", "file-synthesis-v3")
        stale_final = _final_node_for(
            tree, plan=plan, rel_path="main.py", content_hash=content_hash,
            provider_identity=provider_identity, leaf_results=leaf_results,
            child_results=child_results,
        )
    recovered = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
        rel_path="main.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=stale_leaves + (reducer, stale_final),
    )

    plan_result, materials = _plan_with_record_and_recovered(
        tmp_path, config, stale_record, recovered
    )

    # The completed file-doc-v3 record is rejected by planning.
    assert "main.py" not in plan_result.unchanged_rels
    assert "main.py" in plan_result.changed_rels

    # The stored tree was validated through its real nodes (not carried). The
    # dependency graph dominates: every ``leaf-capsule-v9`` leaf is rejected
    # ``stale-identity``, and the reducer and final node above them are pruned
    # ``input-digest-mismatch`` for the lost dependency -- the final node's own
    # ``fact-ledger-v6`` / ``file-synthesis-v3`` staleness never gets its own
    # identity check here, and is proven in isolation by
    # ``test_same_plan_stale_final_bound_revision_quarantines_only_the_final_node``.
    assert "main.py" not in materials.carry_states
    state = materials.tree_states["main.py"]
    leaf_ids = {chunk.chunk_id for chunk in plan.chunks}
    reasons = {entry.node_id: entry.reason for entry in state.quarantine}
    assert {reasons[cid] for cid in leaf_ids} == {"stale-identity"}
    assert reasons[uc.node_id] == "input-digest-mismatch"
    assert reasons[tree.final_node.node_id] == "input-digest-mismatch"
    assert set(state.by_id()) == set()          # nothing retained
    assert len(state.quarantine) <= MAX_QUARANTINE_ENTRIES_PER_FILE

    manifest = build_call_manifest(
        [], plan_result.agent_rels, "single",
        materials.division_plans, materials.reduction_trees, materials.tree_states,
    )
    scheduled = {call.owner for call in manifest.calls}
    assert leaf_ids <= scheduled
    assert uc.node_id in scheduled
    assert any(
        call.category == "file-synthesis" and call.owner == "main.py"
        for call in manifest.calls
    )

    # After regeneration: a fully current container is reusable, zero re-exec.
    current = _full_current_valid_container(
        plan, tree, rel_path="main.py", content_hash=content_hash,
        provider_identity=provider_identity,
    )
    _pr2, materials2 = _plan_with_recovered(tmp_path, config, current)
    state2 = materials2.tree_states["main.py"]
    assert set(state2.by_id()) == {node.node_id for node in current.nodes}
    assert state2.quarantine == ()
    assert materials2.reexecuted_nodes == 0
