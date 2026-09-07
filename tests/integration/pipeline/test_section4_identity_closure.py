"""Section 4 (plan section 5.7 / workstream G / section 9.1 tests 16-17):
behavior-based proof that the four reached cache/recovery identities advanced,
that each advance both invalidates stale state and preserves independently
compatible state, and that every deliberate non-advance is behaviourally
inert.

These tests exercise the real production identity/recovery functions and the
real ``build_pipeline_plan`` reuse predicate. They never assert a bare
constant literal as the whole proof: each mutation flips a constant with
``monkeypatch`` and observes the reuse/recovery decision change.
"""

from __future__ import annotations

import json

import pytest

from codedoc.core import file_division, record_meta
from codedoc.core.db import compute_file_hash
from codedoc.core.file_division import (
    MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS,
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
    final_execution_identity,
    final_input_digest,
    leaf_execution_identity,
    leaf_input_digest,
    reduction_execution_identity,
    reduction_input_digest,
    tree_node_state,
    validate_recovered_tree,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.planning import build_pipeline_plan
from codedoc.core.record_meta import (
    ANALYSIS_REVISION,
    expected_analysis_identity,
    expected_large_file_identity,
    expected_max_context_revision,
    expected_ordinary_path_identity,
)
from codedoc.pipeline import run_pipeline
from tests.support.one_call_cases import _CountingProvider

_SOURCE = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"


# ===========================================================================
# helpers
# ===========================================================================


def _ordinary_file_map(tmp_path, text: str):
    src = tmp_path / "main.py"
    src.write_text(text, encoding="utf-8", newline="")
    return src, {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        },
    }


def _plan_for(file_map, record, config):
    graph = DependencyGraph()
    graph.add_file("main.py")
    result, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config
    )
    return result


def _ordinary_record(src, *, analysis_revision, functions=None, mcr=None):
    rec = {
        "path": "main.py",
        "hash": compute_file_hash(src),
        "description": "cached",
        "language": "python",
        "_analysis_revision": analysis_revision,
        "_analysis_mode": "single",
        "_ordinary_path_identity": expected_ordinary_path_identity("main.py"),
    }
    if mcr is not None:
        rec["_max_context_revision"] = mcr
    if functions is not None:
        rec["functions"] = functions
    return rec


_ORDINARY_CONFIG = {
    "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
    "max_content_chars": 12000, "truncation_head_ratio": 0.70,
}
_TRUNCATE_CONFIG = {
    "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
    "max_content_chars": 1000, "truncation_head_ratio": 0.70,
}
_SPLIT_CONFIG = {
    "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
    "large_file_strategy": "split",
    "max_content_chars": 2000, "truncation_head_ratio": 0.70,
}


def _split_identity(source: str, *, max_chars: int = 2000) -> str:
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=max_chars,
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    return expected_large_file_identity(
        source_chars=len(source), max_chars=max_chars, rel_path="main.py",
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode,
        imports_digest=deterministic_imports_digest(()),
    )


# ===========================================================================
# A. ANALYSIS_REVISION -- ordinary / truncate / split
# ===========================================================================


def test_current_generated_ordinary_record_carries_file_doc_v4():
    assert ANALYSIS_REVISION == "file-doc-v4"
    assert expected_analysis_identity("single")["_analysis_revision"] == "file-doc-v4"
    assert expected_analysis_identity("triple")["_analysis_revision"] == "file-doc-v4"


@pytest.mark.parametrize(
    "config, text, mcr",
    [
        (_ORDINARY_CONFIG, "x = 1\n", None),
        (
            _TRUNCATE_CONFIG,
            "x" * 4000,
            expected_max_context_revision(4000, max_chars=1000, head_ratio=0.70),
        ),
    ],
    ids=["ordinary", "truncate"],
)
def test_stale_file_doc_v3_record_is_reprocessed_once_then_reusable(
    tmp_path, config, text, mcr
):
    src, file_map = _ordinary_file_map(tmp_path, text)
    stale = _plan_for(
        file_map, _ordinary_record(src, analysis_revision="file-doc-v3", mcr=mcr), config
    )
    assert "main.py" in stale.agent_rels
    assert "main.py" not in stale.unchanged_rels
    # The regenerated record carries the current revision and is zero-call
    # reusable -- reprocessing happens exactly once, not every run.
    current = _plan_for(
        file_map,
        _ordinary_record(src, analysis_revision=ANALYSIS_REVISION, mcr=mcr),
        config,
    )
    assert "main.py" in current.unchanged_rels
    assert "main.py" not in current.agent_rels


def test_stale_file_doc_v3_split_completed_record_is_reprocessed_once_then_reusable(
    tmp_path,
):
    src, file_map = _ordinary_file_map(tmp_path, _SOURCE)
    identity = _split_identity(_SOURCE)

    def _rec(rev):
        return {
            "path": "main.py", "hash": compute_file_hash(src), "description": "cached",
            "language": "python", "_analysis_revision": rev, "_analysis_mode": "single",
            "_large_file_identity": identity,
        }

    stale = _plan_for(file_map, _rec("file-doc-v3"), _SPLIT_CONFIG)
    assert "main.py" not in stale.unchanged_rels
    assert "main.py" in stale.changed_rels

    current = _plan_for(file_map, _rec(ANALYSIS_REVISION), _SPLIT_CONFIG)
    assert "main.py" in current.unchanged_rels


def test_reverting_analysis_revision_makes_a_stale_v3_record_reusable(
    tmp_path, monkeypatch
):
    # Mutation check: the file-doc-v4 advance is what rejects a v3 record.
    # ``record_meta`` owns ANALYSIS_REVISION and ``expected_analysis_identity``
    # reads it at call time, so reverting it there makes the identical v3
    # record reuse zero-call.
    src, file_map = _ordinary_file_map(tmp_path, "x = 1\n")
    record = _ordinary_record(src, analysis_revision="file-doc-v3")
    assert "main.py" in _plan_for(file_map, record, _ORDINARY_CONFIG).agent_rels

    monkeypatch.setattr(record_meta, "ANALYSIS_REVISION", "file-doc-v3")
    reverted = _plan_for(file_map, record, _ORDINARY_CONFIG)
    assert "main.py" in reverted.unchanged_rels
    assert "main.py" not in reverted.agent_rels


def _seed_completed_record(tmp_path, *, rel_path, source, functions, analysis_revision):
    """Write a completed ordinary ``codedoc.json`` whose one file record carries
    *analysis_revision*, a matching content hash, the current ordinary-path
    identity, and a stored ``functions`` array."""
    src = tmp_path / rel_path
    src.write_text(source, encoding="utf-8", newline="")
    out = tmp_path / "codedoc"
    out.mkdir(exist_ok=True)
    out.joinpath("codedoc.json").write_text(
        json.dumps(
            {
                "_codedoc": {"entry_file": rel_path, "schema_version": "1.4"},
                "files": [
                    {
                        "path": rel_path,
                        "hash": compute_file_hash(src),
                        "language": "python",
                        "description": "seeded",
                        "role_in_system": "core",
                        "functions": functions,
                        "classes": [],
                        "exports": [],
                        "key_concepts": ["seed"],
                        "usage_example": "import x",
                        "_analysis_revision": analysis_revision,
                        "_analysis_mode": "single",
                        "_ordinary_path_identity": expected_ordinary_path_identity(
                            rel_path
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return src


def _published_record(tmp_path):
    return json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]


_ONE_DECLARATION = "def only_one():\n    return 1\n"
_DUP_FUNCTIONS = [
    {"name": "only_one", "description": "seed dup a"},
    {"name": "only_one", "description": "seed dup b"},
]
_DUP_COMBINED_RESPONSE = json.dumps(
    {
        "description": "A module.",
        "role_in_system": "core",
        # The model attempts the exact-identical duplicate again.
        "functions": [
            {"name": "only_one", "description": "the one function"},
            {"name": "only_one", "description": "the one function"},
        ],
        "classes": [],
        "exports": [],
        "key_concepts": ["k"],
        "usage_example": "import x",
        "dependencies_analysis": {"internal": [], "external": []},
    }
)


def test_stale_v3_record_with_a_duplicate_declaration_regenerates_through_the_authority(
    tmp_path, monkeypatch
):
    """A completed ``file-doc-v3`` record carrying an exact-identical duplicate
    declaration is not reused: the real ordinary pipeline reprocesses the file
    exactly once, the source-backed reconciliation/publication path runs, and
    the regenerated public record carries ``file-doc-v4`` with the one real
    ``only_one`` occurrence exactly once. A second run is then zero-call."""
    _seed_completed_record(
        tmp_path,
        rel_path="main.py",
        source=_ONE_DECLARATION,
        functions=_DUP_FUNCTIONS,
        analysis_revision="file-doc-v3",
    )

    provider = _CountingProvider(raw=_DUP_COMBINED_RESPONSE)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: provider)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "analysis_mode": "single", "propagate_changes": False},
    )

    # Stale record not reused -> the file is executed, exactly once (not looped).
    assert stats["checked"] == 1
    assert provider.calls == 1

    rec = _published_record(tmp_path)
    # The reconciliation/publication path ran: the regenerated record is current
    # and the one real declaration is published exactly once despite the model
    # returning it twice.
    assert rec["_analysis_revision"] == ANALYSIS_REVISION == "file-doc-v4"
    names = [f["name"] for f in rec["functions"]]
    assert names == ["only_one"]
    assert names.count("only_one") == 1

    # A subsequent run over the regenerated current record makes no call:
    # constructing a provider at all would be a reuse failure.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("regenerated current record must be zero-call reusable"),
    )
    again = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "analysis_mode": "single", "propagate_changes": False},
    )
    assert again["checked"] == 0  # skipped as unchanged, not reprocessed
    assert [f["name"] for f in _published_record(tmp_path)["functions"]] == ["only_one"]


_TWO_OVERLOADS = (
    "def dispatch(a):\n    return a\n\n\n"
    "def dispatch(a, b):\n    return a + b\n"
)
_OVERLOAD_COMBINED_RESPONSE = json.dumps(
    {
        "description": "A module.",
        "role_in_system": "core",
        "functions": [
            {"name": "dispatch", "description": "one arg"},
            {"name": "dispatch", "description": "two args"},
        ],
        "classes": [],
        "exports": [],
        "key_concepts": ["k"],
        "usage_example": "import x",
        "dependencies_analysis": {"internal": [], "external": []},
    }
)


def test_genuine_same_name_overloads_both_survive_the_publication_path(
    tmp_path, monkeypatch
):
    """Two distinct same-name declarations with distinct source occurrences are
    both published by the real ``run_pipeline`` reconciliation/publication path.
    Reconciliation is occurrence-based, never global name or dict-equality
    dedup, so a legitimate overload pair is preserved as two entries."""
    _seed_completed_record(
        tmp_path,
        rel_path="svc.py",
        source=_TWO_OVERLOADS,
        functions=[{"name": "dispatch", "description": "stale single entry"}],
        analysis_revision="file-doc-v3",
    )

    provider = _CountingProvider(raw=_OVERLOAD_COMBINED_RESPONSE)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: provider)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "svc.py", "analysis_mode": "single", "propagate_changes": False},
    )

    assert stats["checked"] == 1
    assert provider.calls == 1
    rec = _published_record(tmp_path)
    names = [f["name"] for f in rec["functions"]]
    # Both genuine occurrences remain -- the pair is not collapsed by name.
    assert names == ["dispatch", "dispatch"]
    assert names.count("dispatch") == 2


# ===========================================================================
# B. LEAF_CAPSULE_SCHEMA_REVISION -- recovery quarantine before schema check
# ===========================================================================


def _build_leaf_nodes(plan, *, content_hash, provider_identity):
    # The leaf-capsule revision in force when this runs is bound into every
    # ``leaf_execution_identity`` below; a caller stamps a prior revision by
    # monkeypatching ``file_division.LEAF_CAPSULE_SCHEMA_REVISION`` around the
    # call.
    nodes = []
    for chunk in plan.chunks:
        exec_id = leaf_execution_identity(
            rel_path=plan.rel_path, content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity, chunk=chunk,
        )
        nodes.append(
            tree_node_state(
                node_id=chunk.chunk_id, node_type="leaf", rel_path=plan.rel_path,
                content_hash=content_hash, division_plan_digest=plan.plan_digest,
                input_digest=leaf_input_digest(
                    rel_path=plan.rel_path, language="unknown", chunk=chunk,
                    unit_indexes=plan.unit_positions(chunk), unit_count=len(plan.units),
                ),
                execution_identity_digest=exec_id,
                unit_id=None, child_ids=(), coverage_leaf_ids=(chunk.chunk_id,),
                # a fully valid leaf-capsule result shape: it would pass
                # _node_result_matches_live_schema on its own.
                result={"description": "a bounded fragment",
                        "chunk_id": chunk.chunk_id, "unit_id": chunk.unit_id},
            )
        )
    return nodes


def test_stale_leaf_capsule_node_is_quarantined_on_identity_not_schema(monkeypatch):
    plan = build_division_plan(
        rel_path="v.py", language="unknown",
        content="\n".join(f"a_{i} = {i}" for i in range(90)) + "\n",
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64
    imports_digest = deterministic_imports_digest(())

    # Stamp every leaf under the PRIOR leaf-capsule revision, then restore.
    monkeypatch.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v9")
    stale_nodes = _build_leaf_nodes(
        plan, content_hash=content_hash, provider_identity=provider_identity
    )
    monkeypatch.undo()
    assert file_division.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v10"

    retained, quarantine = validate_recovered_tree(
        stale_nodes, plan=plan, tree=tree, content_hash=content_hash,
        provider_identity=provider_identity, prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=imports_digest, language="unknown",
    )
    assert list(retained) == []
    assert {q.node_id for q in quarantine} == {c.chunk_id for c in plan.chunks}
    # Identity is checked before stored-node schema revalidation: the reason is
    # stale-identity, never live-schema-mismatch, even though the stored result
    # is a well-formed leaf capsule.
    assert {q.reason for q in quarantine} == {"stale-identity"}


def test_current_leaf_capsule_nodes_are_retained():
    plan = build_division_plan(
        rel_path="v.py", language="unknown",
        content="\n".join(f"a_{i} = {i}" for i in range(90)) + "\n",
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64
    nodes = _build_leaf_nodes(
        plan, content_hash=content_hash, provider_identity=provider_identity
    )
    retained, quarantine = validate_recovered_tree(
        nodes, plan=plan, tree=tree, content_hash=content_hash,
        provider_identity=provider_identity, prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=deterministic_imports_digest(()), language="unknown",
    )
    assert quarantine == ()
    assert {n.node_id for n in retained} == {c.chunk_id for c in plan.chunks}


# ===========================================================================
# C/D. LEDGER_SCHEMA_REVISION and FINAL_SYNTHESIS_REVISION
#      final-node identity moves; leaf/reducer identity does not.
# ===========================================================================


def _tree_bits():
    plan = build_division_plan(
        rel_path="main.py", language="python", content=_SOURCE, source_budget_chars=2000
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(2000, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    return plan, tree


_FINAL_KW = dict(
    rel_path="main.py", content_hash="c" * 64,
    provider_identity="provider-execution:" + "d" * 64,
    prompt_profile_digest="no-prompt-profile-v1",
    imports_digest=deterministic_imports_digest(()),
)


@pytest.mark.parametrize(
    "constant_name, prior_value",
    [
        ("LEDGER_SCHEMA_REVISION", "fact-ledger-v6"),
        ("FINAL_SYNTHESIS_REVISION", "file-synthesis-v3"),
    ],
)
def test_reverting_a_final_bound_revision_moves_only_the_final_identity(
    monkeypatch, constant_name, prior_value
):
    """Digest-level binding check: reverting ``LEDGER_SCHEMA_REVISION`` or
    ``FINAL_SYNTHESIS_REVISION`` moves the final-node execution identity, the
    final exact input digest, and the completed split identity, while the leaf
    and reducer identities are unchanged. The full recovery-state behaviour --
    stale final quarantined, leaves and reducers retained, one final rerun --
    is proven in ``tests/integration/persistence/test_split_partial_quarantine.py``
    (``test_same_plan_stale_final_bound_revision_quarantines_only_the_final_node``)."""
    plan, tree = _tree_bits()
    final_node = tree.final_node
    reducer_nodes = tree.all_intermediate_nodes
    leaf_chunk = plan.chunks[0]

    def _final_id():
        return final_execution_identity(node=final_node, division_plan_digest=plan.plan_digest,
                                        reduction_tree_digest=tree.tree_digest, **_FINAL_KW)

    def _final_input():
        return final_input_digest(
            imports_digest=_FINAL_KW["imports_digest"],
            resolved_shape_digest="shape-v1", manifest_json='{"m":1}',
        )

    def _leaf_id():
        return leaf_execution_identity(
            rel_path="main.py", content_hash=_FINAL_KW["content_hash"],
            division_plan_digest=plan.plan_digest,
            provider_identity=_FINAL_KW["provider_identity"], chunk=leaf_chunk,
        )

    def _reducer_id():
        if not reducer_nodes:
            return None
        return reduction_execution_identity(
            rel_path="main.py", content_hash=_FINAL_KW["content_hash"],
            division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
            provider_identity=_FINAL_KW["provider_identity"], node=reducer_nodes[0],
        )

    def _large_id():
        return expected_large_file_identity(
            source_chars=len(_SOURCE), max_chars=2000, rel_path="main.py",
            division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
            structural_mode=plan.structural_mode, imports_digest=_FINAL_KW["imports_digest"],
        )

    cur_final, cur_final_in = _final_id(), _final_input()
    cur_leaf, cur_reducer, cur_large = _leaf_id(), _reducer_id(), _large_id()

    # Revert on BOTH the file_division owner and the record_meta alias so the
    # completed-split identity mutation is real, not a stale imported alias.
    monkeypatch.setattr(file_division, constant_name, prior_value)
    monkeypatch.setattr(record_meta, constant_name, prior_value)

    # Final-node execution identity + final exact input digest + completed
    # split identity all move -> a stored final node / completed record from
    # the prior value is rejected and rerun.
    assert _final_id() != cur_final
    assert _final_input() != cur_final_in
    assert _large_id() != cur_large

    # Leaf and reducer identities do NOT move: independently compatible leaf
    # and reducer work is retained where the dependency graph allows it.
    assert _leaf_id() == cur_leaf
    if cur_reducer is not None:
        assert _reducer_id() == cur_reducer


def test_final_synthesis_revision_also_moves_the_run_call_manifest_digest(monkeypatch):
    # FINAL_SYNTHESIS_REVISION reaches the run call-manifest digest through
    # ``file_synthesis_call_id`` (via ``_synthesis_prompt_revision``). Reverting
    # it changes the synthesis call id, so the whole manifest digest moves --
    # a completed run's identity is not silently reused after the advance.
    from codedoc.core.execution_model import build_call_manifest

    plan, tree = _tree_bits()

    def _digest():
        return build_call_manifest(
            [],
            ["main.py"],
            "single",
            division_plans={"main.py": plan},
            reduction_trees={"main.py": tree},
        ).digest

    current = _digest()
    monkeypatch.setattr(file_division, "FINAL_SYNTHESIS_REVISION", "file-synthesis-v3")
    assert _digest() != current
    monkeypatch.undo()
    assert _digest() == current


# ===========================================================================
# E. leaf-capsule predecessor -- every recovered leaf is rejected
# ===========================================================================


def test_leaf_capsule_predecessor_quarantines_every_recovered_leaf(monkeypatch):
    """The ``leaf-capsule-v9`` -> ``leaf-capsule-v10`` advance alone rejects every
    leaf checkpointed under the prior revision: each is quarantined
    ``stale-identity`` and nothing is retained, so any tree assembled from only
    those leaves has to re-execute in full.

    Reducer / final dependency closure across a complete leaf+reducer+final
    tree -- stale leaves rejected, independent leaves and reducers kept, only
    dependents pruned -- is proven in
    ``tests/integration/persistence/test_split_partial_quarantine.py``
    (``test_same_plan_stale_leaves_prune_only_the_dependent_reducer`` and
    ``test_combined_predecessor_completed_record_and_stale_tree_reprocess_the_minimum``)."""
    plan = build_division_plan(
        rel_path="leaf_only.py", language="unknown",
        content="\n".join(f"a_{i} = {i}" for i in range(90)) + "\n",
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64

    monkeypatch.setattr(
        file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v9"
    )
    stale_nodes = _build_leaf_nodes(
        plan, content_hash=content_hash, provider_identity=provider_identity
    )
    monkeypatch.undo()

    retained, quarantine = validate_recovered_tree(
        stale_nodes, plan=plan, tree=tree, content_hash=content_hash,
        provider_identity=provider_identity, prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=deterministic_imports_digest(()), language="unknown",
    )
    assert list(retained) == []
    assert {q.node_id for q in quarantine} == {c.chunk_id for c in plan.chunks}
    assert {q.reason for q in quarantine} == {"stale-identity"}


# ===========================================================================
# F. non-advance preservation matrix -- behavior-based
# ===========================================================================

# Every non-advanced revision still participates in the completed split
# identity payload, so advancing it *would* invalidate otherwise compatible
# state -- which is exactly why it is deliberately held. Each entry: the
# constant on ``file_division``/``record_meta`` and a value it does not hold.
_NON_ADVANCED = [
    ("STRUCTURE_SCHEMA_REVISION", "source-structure-v9"),
    ("UNIT_SCHEMA_REVISION", "semantic-unit-v9"),
    ("PACKER_SCHEMA_REVISION", "division-packer-v9"),
    ("REDUCTION_PACKING_REVISION", "reduction-packing-v9"),
    ("REDUCER_PROMPT_REVISION", "file-reduction-v9"),
    ("REDUCTION_CAPSULE_SCHEMA_REVISION", "reduction-capsule-v9"),
]


@pytest.mark.parametrize("name, other_value", _NON_ADVANCED)
def test_non_advanced_revision_still_participates_so_holding_it_avoids_invalidation(
    monkeypatch, name, other_value
):
    plan, tree = _tree_bits()
    kw = dict(
        source_chars=len(_SOURCE), max_chars=2000, rel_path="main.py",
        division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode,
        imports_digest=deterministic_imports_digest(()),
    )
    current = expected_large_file_identity(**kw)
    monkeypatch.setattr(record_meta, name, other_value)
    assert expected_large_file_identity(**kw) != current, (
        f"{name} no longer participates in the completed split identity"
    )


def test_non_advanced_revisions_hold_their_established_values():
    assert file_division.STRUCTURE_SCHEMA_REVISION == "source-structure-v2"
    assert file_division.UNIT_SCHEMA_REVISION == "semantic-unit-v3"
    assert file_division.PACKER_SCHEMA_REVISION == "division-packer-v6"
    assert file_division.REDUCTION_PACKING_REVISION == "reduction-packing-v5"
    assert file_division.REDUCER_PROMPT_REVISION == "file-reduction-v3"
    assert file_division.REDUCTION_CAPSULE_SCHEMA_REVISION == "reduction-capsule-v1"
    assert file_division.EXECUTION_IDENTITY_SCHEMA_REVISION == "division-execution-v6"
    assert file_division.SPLIT_PARTIAL_SCHEMA_VERSION == 4
    from codedoc.core.project_view import SCHEMA_VERSION
    assert SCHEMA_VERSION == "1.4"


def test_reducer_prompt_revision_not_advanced_keeps_reducer_nodes_reusable(monkeypatch):
    # REDUCER_PROMPT_REVISION governs reduction_execution_identity /
    # reduction_input_digest. It did not move, so a reducer node from before
    # the terminology work stays a current checkpoint (the internal reduction
    # prompt is deliberately excluded from the terminology rules).
    plan, tree = _tree_bits()
    # _SOURCE at a 2000-char source budget deterministically fans out into
    # several leaves under at least one reducer node.
    assert tree.all_intermediate_nodes, "fixture no longer produces a reducer node"
    node = tree.all_intermediate_nodes[0]
    rid = reduction_input_digest(
        phase=node.phase, level=node.level, unit_id=node.unit_id,
        child_count=len(node.child_ids), ordered_child_narratives=("n",),
        rel_path="main.py",
    )
    current = reduction_input_digest(
        phase=node.phase, level=node.level, unit_id=node.unit_id,
        child_count=len(node.child_ids), ordered_child_narratives=("n",),
        rel_path="main.py",
    )
    assert rid == current  # deterministic
    monkeypatch.setattr(file_division, "REDUCER_PROMPT_REVISION", "file-reduction-v9")
    assert reduction_input_digest(
        phase=node.phase, level=node.level, unit_id=node.unit_id,
        child_count=len(node.child_ids), ordered_child_narratives=("n",),
        rel_path="main.py",
    ) != current  # it still participates -> correctly held, not advanced


# ---------------------------------------------------------------------------
# P2-3: the rest of the deliberate non-advances, each with behavior-based
# preservation evidence (owner, what it governs, and a mutation that moves
# ONLY its governed identity). Strong existing coverage is cited rather than
# duplicated; a focused test is added only where none existed.
# ---------------------------------------------------------------------------


def _leaf_reducer_final_identities(plan, tree):
    """The current node execution identities and stage-local input digests for
    one leaf, one reducer, and the final node -- all from the production
    functions under whatever revisions are in force at call time."""
    leaf = plan.chunks[0]
    reducer = tree.all_intermediate_nodes[0]
    final = tree.final_node
    imports_digest = deterministic_imports_digest(())
    return {
        "leaf_exec": leaf_execution_identity(
            rel_path="main.py", content_hash="a" * 64,
            division_plan_digest=plan.plan_digest,
            provider_identity="provider-execution:" + "b" * 64, chunk=leaf,
        ),
        "reducer_exec": reduction_execution_identity(
            rel_path="main.py", content_hash="a" * 64,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            provider_identity="provider-execution:" + "b" * 64, node=reducer,
        ),
        "final_exec": final_execution_identity(
            rel_path="main.py", content_hash="a" * 64,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            provider_identity="provider-execution:" + "b" * 64,
            prompt_profile_digest="no-prompt-profile-v1",
            imports_digest=imports_digest, node=final,
        ),
        "leaf_input": leaf_input_digest(
            rel_path="main.py", language="python", chunk=leaf,
            unit_indexes=plan.unit_positions(leaf), unit_count=len(plan.units),
        ),
        "reducer_input": reduction_input_digest(
            rel_path="main.py", phase=reducer.phase, level=reducer.level,
            unit_id=reducer.unit_id, child_count=len(reducer.child_ids),
            ordered_child_narratives=("n",),
        ),
        "final_input": final_input_digest(
            imports_digest=imports_digest, resolved_shape_digest="shape-v1",
            manifest_json='{"m":1}',
        ),
        "large_file": expected_large_file_identity(
            source_chars=len(_SOURCE), max_chars=2000, rel_path="main.py",
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            structural_mode=plan.structural_mode, imports_digest=imports_digest,
        ),
    }


def test_execution_identity_schema_revision_governs_only_node_execution_identities(
    monkeypatch,
):
    """``EXECUTION_IDENTITY_SCHEMA_REVISION`` (owner: ``file_division``) is the
    schema revision for every split node's execution-identity payload. Held at
    ``division-execution-v6``. Mutating it moves the leaf, reducer, and final
    execution identities and nothing else: the plan digest, the tree digest,
    the three stage-local input digests, and the completed large-file identity
    are all unchanged, so advancing it would needlessly quarantine every
    otherwise-compatible node."""
    plan, tree = _tree_bits()
    assert tree.all_intermediate_nodes
    before = _leaf_reducer_final_identities(plan, tree)

    monkeypatch.setattr(
        file_division, "EXECUTION_IDENTITY_SCHEMA_REVISION", "division-execution-v9"
    )
    after = _leaf_reducer_final_identities(plan, tree)

    assert after["leaf_exec"] != before["leaf_exec"]
    assert after["reducer_exec"] != before["reducer_exec"]
    assert after["final_exec"] != before["final_exec"]
    # Everything the revision does NOT own is untouched.
    assert after["leaf_input"] == before["leaf_input"]
    assert after["reducer_input"] == before["reducer_input"]
    assert after["final_input"] == before["final_input"]
    assert after["large_file"] == before["large_file"]
    plan2, tree2 = _tree_bits()
    assert plan2.plan_digest == plan.plan_digest
    assert tree2.tree_digest == tree.tree_digest


@pytest.mark.parametrize(
    "revision_name, moved_key",
    [
        ("LEAF_INPUT_DIGEST_REVISION", "leaf_input"),
        ("REDUCTION_INPUT_DIGEST_REVISION", "reducer_input"),
        ("FINAL_INPUT_DIGEST_REVISION", "final_input"),
    ],
)
def test_input_digest_schema_revisions_are_domain_separated(
    monkeypatch, revision_name, moved_key
):
    """The three stage-local input-digest schema revisions (owner:
    ``file_division``; held at ``leaf-input-v1`` / ``reduction-input-v1`` /
    ``final-input-v1``) are domain-separated: mutating one moves only its own
    stage's input digest. The other two input digests, all three node
    execution identities, and the completed large-file identity are unchanged,
    so none of the three is a ceremonial version marker."""
    plan, tree = _tree_bits()
    assert tree.all_intermediate_nodes
    before = _leaf_reducer_final_identities(plan, tree)

    monkeypatch.setattr(file_division, revision_name, "changed-input-v9")
    after = _leaf_reducer_final_identities(plan, tree)

    input_keys = {"leaf_input", "reducer_input", "final_input"}
    assert after[moved_key] != before[moved_key]
    for key in input_keys - {moved_key}:
        assert after[key] == before[key], key
    for key in ("leaf_exec", "reducer_exec", "final_exec", "large_file"):
        assert after[key] == before[key], key


def test_deterministic_imports_binding_is_final_and_completed_identity_only(monkeypatch):
    """``deterministic_imports_digest`` (owner: ``file_division``) feeds the
    final-node execution identity, the final exact input digest, and the
    completed large-file identity, and is deliberately excluded from leaf and
    reducer identities (an import change reruns only final synthesis). Its
    output is deterministic for a given import sequence."""
    assert deterministic_imports_digest(("a", "b")) == deterministic_imports_digest(
        ("a", "b")
    )
    plan, tree = _tree_bits()
    leaf, reducer, final = plan.chunks[0], tree.all_intermediate_nodes[0], tree.final_node
    kw = dict(
        rel_path="main.py", content_hash="a" * 64,
        division_plan_digest=plan.plan_digest,
        provider_identity="provider-execution:" + "b" * 64,
    )

    leaf_id = leaf_execution_identity(chunk=leaf, **kw)
    reducer_id = reduction_execution_identity(
        node=reducer, reduction_tree_digest=tree.tree_digest, **kw
    )
    d0 = deterministic_imports_digest(())
    d1 = deterministic_imports_digest(("os",))
    assert d0 != d1

    def _final(imports_digest):
        return final_execution_identity(
            reduction_tree_digest=tree.tree_digest,
            prompt_profile_digest="no-prompt-profile-v1",
            imports_digest=imports_digest, node=final, **kw,
        )

    def _large(imports_digest):
        return expected_large_file_identity(
            source_chars=len(_SOURCE), max_chars=2000, rel_path="main.py",
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            structural_mode=plan.structural_mode, imports_digest=imports_digest,
        )

    assert _final(d0) != _final(d1)          # final identity is import-bound
    assert _large(d0) != _large(d1)          # completed identity is import-bound
    # Leaf and reducer identities take no imports argument at all -> unchanged.
    assert leaf_execution_identity(chunk=leaf, **kw) == leaf_id
    assert reduction_execution_identity(
        node=reducer, reduction_tree_digest=tree.tree_digest, **kw
    ) == reducer_id


def test_held_wire_and_output_schema_versions_accept_current_state():
    """``SPLIT_PARTIAL_SCHEMA_VERSION`` (owner: ``file_division``, held at 4) is
    the recovery wire-format version; the public document ``SCHEMA_VERSION``
    (owner: ``project_view``, held at "1.4") is the output schema version; the
    ``large-file-v3:`` prefix (owner: ``record_meta``) is the completed
    split-identity namespace. None changed, so current state stays valid.

    Behavioral coverage lives in dedicated suites and is cited here rather than
    duplicated:
      * schema-4 round trip / full retained container --
        tests/integration/persistence/test_split_partial_quarantine.py
        (test_positive_control_full_current_valid_container_is_fully_retained,
         test_quarantine_round_trips_through_the_recovery_file_and_clears_on_replacement);
      * cross-version schema-4 read --
        tests/integration/persistence/test_cross_version_split_state.py;
      * output SCHEMA_VERSION "1.4" round trip --
        tests/contract/output/test_cross_format_roundtrip.py,
        tests/contract/output/test_golden_serialization.py;
      * large-file-v3 prefix stability across a revision mutation --
        tests/unit/core/test_record_metadata.py
        (test_completed_large_file_identity_binds_leaf_capsule_revision_and_bound).
    """
    from codedoc.core.project_view import SCHEMA_VERSION

    assert file_division.SPLIT_PARTIAL_SCHEMA_VERSION == 4
    assert SCHEMA_VERSION == "1.4"

    plan, tree = _tree_bits()
    identity = expected_large_file_identity(
        source_chars=len(_SOURCE), max_chars=2000, rel_path="main.py",
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode,
        imports_digest=deterministic_imports_digest(()),
    )
    assert identity.startswith("large-file-v3:")


def test_ordinary_path_and_truncate_identities_are_held():
    """``expected_ordinary_path_identity`` (``_ordinary_path_identity``, owner:
    ``record_meta``) and ``expected_max_context_revision``
    (``_max_context_revision``, owner: ``record_meta``) are unchanged.

    Behavioral coverage is cited rather than duplicated:
      * a missing ``_ordinary_path_identity`` regenerates exactly once then
        reuses zero-call, and round-trips through Markdown --
        tests/integration/pipeline/test_cache_identity.py
        (test_pre_0_14_4_record_stays_invalid_until_successfully_replaced);
      * a path change moves the identity --
        tests/unit/core/test_record_metadata.py
        (test_ordinary_path_identity_differs_by_path);
      * raising the ceiling / changing the head ratio reprocesses a truncated
        file while a small file stays reusable across both --
        tests/integration/pipeline/test_cache_identity.py
        (test_raising_ceiling_reprocesses_truncated_file,
         test_changing_head_ratio_reprocesses_truncated_file,
         test_small_file_reusable_across_ceiling_and_ratio_changes).
    """
    # Held shapes: a path-scoped identity and a ceiling+ratio-scoped token.
    assert expected_ordinary_path_identity("a.py") != expected_ordinary_path_identity(
        "b.py"
    )
    token = expected_max_context_revision(4000, max_chars=1000, head_ratio=0.70)
    assert token.startswith("truncate-v1:")
    assert token != expected_max_context_revision(4000, max_chars=2000, head_ratio=0.70)
    assert token != expected_max_context_revision(4000, max_chars=1000, head_ratio=0.85)
