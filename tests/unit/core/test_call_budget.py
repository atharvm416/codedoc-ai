"""Tests organized by feature ownership."""

from __future__ import annotations

import hashlib

import pytest

from codedoc.core.execution_model import CallManifest, PlannedCall, build_call_manifest
from codedoc.core.file_division import (
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
    leaf_execution_identity,
    reduction_execution_identity,
    final_execution_identity,
    tree_node_state,
    SplitTreeState,
)
from codedoc.core.planning import PipelinePlan
from codedoc.core.prompt_profiles import ReviewBatch, ReviewComponent


def _plan(agent_rels=(), **overrides) -> PipelinePlan:
    defaults = dict(
        scanned_rels=frozenset(agent_rels),
        documented_rels=frozenset(agent_rels),
        changed_rels=frozenset(agent_rels),
        forced_rels=frozenset(),
        process_rels=frozenset(agent_rels),
        unchanged_rels=frozenset(),
        identical_reuse_rels=frozenset(),
        agent_rels=frozenset(agent_rels),
        entry_rel=None,
        max_files=0,
        max_files_exceeded=False,
    )
    defaults.update(overrides)
    return PipelinePlan(**defaults)


def _batch(ordinal=1, count=1, stream_digest="digest-a", component_ids=("single/combined/*",)):
    components = tuple(
        ReviewComponent(component=cid, block_text="block", fields=(("description", "string"),))
        for cid in component_ids
    )
    return ReviewBatch(
        ordinal=ordinal,
        count=count,
        review_id="rev-test",
        stream_digest=stream_digest,
        unit_total=1,
        components=components,
        units=(),
        text="batch text",
    )


class TestPerModeCounts:
    def test_single_mode_documentation_count_is_one_per_file(self):
        manifest = build_call_manifest([], ["a.py", "b.py"], "single")
        plan = _plan(["a.py", "b.py"]).with_call_manifest(manifest, 0)
        assert plan.documentation_calls_planned == 2
        assert plan.review_calls_planned == 0
        assert plan.total_calls_planned == 2

    def test_triple_mode_documentation_count_is_three_per_file(self):
        manifest = build_call_manifest([], ["a.py", "b.py"], "triple")
        plan = _plan(["a.py", "b.py"]).with_call_manifest(manifest, 0)
        assert plan.documentation_calls_planned == 6
        assert plan.total_calls_planned == 6

    def test_review_batches_counted_separately_from_documentation(self):
        batches = [_batch(1, 2), _batch(2, 2)]
        manifest = build_call_manifest(batches, ["a.py"], "single")
        plan = _plan(["a.py"]).with_call_manifest(manifest, 0)
        assert plan.review_calls_planned == 2
        assert plan.documentation_calls_planned == 1
        assert plan.total_calls_planned == 3


class TestEmptyAndAllReusedRuns:
    def test_empty_manifest_reports_zero_calls(self):
        manifest = build_call_manifest([], [], "single")
        plan = _plan([]).with_call_manifest(manifest, 0)
        assert plan.review_calls_planned == 0
        assert plan.documentation_calls_planned == 0
        assert plan.total_calls_planned == 0
        assert plan.max_planned_calls_exceeded is False

    def test_all_reused_plan_uses_the_same_manifest_helper(self):
        """A plan whose files were all identical-content-reused has an empty
        agent_rels; the manifest helper is used exactly as for any other run
        and reports zero calls, never a special-cased path."""
        reused_plan = _plan([], identical_reuse_rels=frozenset({"a.py", "b.py"}))
        manifest = build_call_manifest([], reused_plan.agent_rels, "single")
        plan = reused_plan.with_call_manifest(manifest, 5)
        assert plan.total_calls_planned == 0
        assert plan.max_planned_calls == 5
        assert plan.max_planned_calls_exceeded is False

    def test_fully_synthesized_split_partial_has_a_valid_empty_manifest(self):
        source = "\n".join(
            f"value_{index} = {index}" for index in range(220)
        ) + "\n"
        division = build_division_plan(
            rel_path="large.py",
            language="python",
            content=source,
            source_budget_chars=1000,
        )
        tree = build_reduction_tree(division, max_content_chars=12000)
        content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        provider_identity = "provider-execution:" + "a" * 64

        _INPUT_DIGEST = "test-input:" + "7" * 64
        nodes = []
        for chunk in division.chunks:
            identity = leaf_execution_identity(
                rel_path=division.rel_path,
                content_hash=content_hash,
                division_plan_digest=division.plan_digest,
                provider_identity=provider_identity,
                chunk=chunk,
            )
            nodes.append(
                tree_node_state(
                    node_id=chunk.chunk_id,
                    node_type="leaf",
                    rel_path=division.rel_path,
                    content_hash=content_hash,
                    division_plan_digest=division.plan_digest,
                    input_digest=_INPUT_DIGEST,
                    execution_identity_digest=identity,
                    unit_id=None,
                    child_ids=(),
                    coverage_leaf_ids=(chunk.chunk_id,),
                    result={"description": "restored"},
                )
            )
        for node in tree.unit_consolidation_nodes + tree.general_nodes:
            identity = reduction_execution_identity(
                rel_path=division.rel_path,
                content_hash=content_hash,
                division_plan_digest=division.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                node=node,
            )
            nodes.append(
                tree_node_state(
                    node_id=node.node_id,
                    node_type=node.phase,
                    rel_path=division.rel_path,
                    content_hash=content_hash,
                    division_plan_digest=division.plan_digest,
                    input_digest=_INPUT_DIGEST,
                    execution_identity_digest=identity,
                    unit_id=node.unit_id,
                    child_ids=node.child_ids,
                    coverage_leaf_ids=node.leaf_ids,
                    result={"narrative": "restored"},
                )
            )
        final_identity = final_execution_identity(
            rel_path=division.rel_path,
            content_hash=content_hash,
            division_plan_digest=division.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            provider_identity=provider_identity,
            prompt_profile_digest="test-profile",
            imports_digest=deterministic_imports_digest(()),
            node=tree.final_node,
        )
        nodes.append(
            tree_node_state(
                node_id=tree.final_node.node_id,
                node_type="final",
                rel_path=division.rel_path,
                content_hash=content_hash,
                division_plan_digest=division.plan_digest,
                input_digest=_INPUT_DIGEST,
                execution_identity_digest=final_identity,
                unit_id=None,
                child_ids=tree.final_node.child_ids,
                coverage_leaf_ids=tree.final_node.leaf_ids,
                result={"description": "restored synthesis"},
            )
        )
        restored = SplitTreeState(
            schema_version=4,
            owner="codedoc-ai",
            rel_path=division.rel_path,
            content_hash=content_hash,
            division_plan_digest=division.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            nodes=tuple(nodes),
        )
        manifest = build_call_manifest(
            [],
            [division.rel_path],
            "single",
            {division.rel_path: division},
            {division.rel_path: tree},
            {division.rel_path: restored},
        )

        plan = _plan(
            [division.rel_path],
            division_plan_rels=frozenset({division.rel_path}),
        ).with_call_manifest(
            manifest,
            0,
            {division.rel_path: division},
            {division.rel_path: tree},
            {division.rel_path: restored},
        )

        assert plan.total_calls_planned == 0
        assert plan.unit_documentation_calls_planned == 0
        assert plan.file_reduction_calls_planned == 0
        assert plan.synthesis_calls_planned == 0


class TestCapEnforcement:
    def test_zero_cap_means_unlimited(self):
        manifest = build_call_manifest([], [f"f{i}.py" for i in range(50)], "triple")
        plan = _plan([f"f{i}.py" for i in range(50)]).with_call_manifest(manifest, 0)
        assert plan.total_calls_planned == 150
        assert plan.max_planned_calls_exceeded is False

    def test_cap_equal_to_total_does_not_exceed(self):
        manifest = build_call_manifest([], ["a.py", "b.py"], "single")
        plan = _plan(["a.py", "b.py"]).with_call_manifest(manifest, 2)
        assert plan.total_calls_planned == 2
        assert plan.max_planned_calls_exceeded is False

    def test_cap_one_below_total_exceeds(self):
        manifest = build_call_manifest([], ["a.py", "b.py"], "single")
        plan = _plan(["a.py", "b.py"]).with_call_manifest(manifest, 1)
        assert plan.total_calls_planned == 2
        assert plan.max_planned_calls_exceeded is True

    def test_max_files_and_max_planned_calls_are_independent(self):
        """A plan may exceed max_files while max_planned_calls is unlimited,
        and vice versa — each cap's outcome depends only on its own field."""
        manifest = build_call_manifest([], ["a.py", "b.py"], "single")
        plan = _plan(
            ["a.py", "b.py"], max_files=1, max_files_exceeded=True
        ).with_call_manifest(manifest, 0)
        assert plan.max_files_exceeded is True
        assert plan.max_planned_calls_exceeded is False


class TestManifestDigestAndCallIdSurface:
    def test_call_manifest_digest_is_attached_to_the_plan(self):
        manifest = build_call_manifest([], ["a.py"], "single")
        plan = _plan(["a.py"]).with_call_manifest(manifest, 0)
        assert plan.call_manifest_digest == manifest.digest
        assert plan.call_manifest_digest != ""

    def test_with_call_manifest_is_idempotent_for_the_same_inputs(self):
        """Calling with_call_manifest() again with an unchanged manifest/cap
        reproduces identical counts — planned counts have no mechanism to grow
        from repeated calls, matching how retries never inflate them."""
        manifest = build_call_manifest([], ["a.py", "b.py"], "triple")
        base = _plan(["a.py", "b.py"])
        first = base.with_call_manifest(manifest, 4)
        second = base.with_call_manifest(manifest, 4)
        assert first.total_calls_planned == second.total_calls_planned == 6
        assert first.max_planned_calls_exceeded == second.max_planned_calls_exceeded is True


class TestStrictManifestValidation:
    def test_unknown_category_is_rejected_at_manifest_build_time(self):
        calls = (PlannedCall(call_id="x", category="bogus", owner="a.py", ordinal=1),)
        bad_manifest = CallManifest(calls=calls, digest="d")
        with pytest.raises(ValueError, match="known categories"):
            _plan(["a.py"]).with_call_manifest(bad_manifest, 0)

    def test_documentation_count_not_a_multiple_of_agent_rels_is_rejected(self):
        calls = (
            PlannedCall(call_id="a", category="file-documentation", owner="a.py", ordinal=1),
            PlannedCall(call_id="b", category="file-documentation", owner="b.py", ordinal=1),
        )
        bad_manifest = CallManifest(calls=calls, digest="d")
        # Two documentation calls for a THREE-file, single-mode (1-per-file) plan:
        # 2 is not a whole multiple of 3.
        with pytest.raises(ValueError, match="owners do not exactly match"):
            _plan(["a.py", "b.py", "c.py"]).with_call_manifest(bad_manifest, 0)

    def test_documentation_calls_with_no_agent_rels_is_rejected(self):
        calls = (
            PlannedCall(call_id="a", category="file-documentation", owner="a.py", ordinal=1),
        )
        bad_manifest = CallManifest(calls=calls, digest="d")
        with pytest.raises(ValueError, match="owners do not exactly match"):
            _plan([]).with_call_manifest(bad_manifest, 0)
