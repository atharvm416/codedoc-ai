"""Section 6 -- bounded provider-free diagnostic substrate.

These tests exercise only the value-safe substrate that later sections
consume for scanner diagnostics, real preflight reporting, CLI presentation,
and billing:

* the two diagnostic-retention constants (section 5.4);
* the measured ``SplitCapacityBlocked`` contract (section 5.8);
* the ``split_plan`` / ``truncate_plan`` / ``split_blocked`` categories and
  their four ``_details_*`` suffixes;
* the canonical full-stream integrity digest;
* bounded top-``K`` retention with a flattened item budget; and
* the bounded, JSON-escaped split-blocked ``ConfigError``.

They are written test-first: every behavioural assertion here fails against
the pre-section-6 implementation for the missing behaviour, not merely a
constant pin.
"""

from __future__ import annotations

import json

import pytest

import codedoc.core.file_division as file_division
from codedoc.core.file_division import (
    BLOCKED_REASON_ORDER,
    MAX_EPHEMERAL_PLAN_DETAIL_ITEMS,
    PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS,
    SplitCapacityBlocked,
    blocked_split_descriptor,
    build_division_plan,
    build_reduction_tree,
    canonical_json,
    canonical_stream_digest,
    split_plan_file_detail,
)


# ---------------------------------------------------------------------------
# Fixtures: real provider-free division plans
# ---------------------------------------------------------------------------


def _one_oversized_span(total_chars: int) -> str:
    """One indivisible assignment statement of exactly *total_chars* code points.

    A single physical line with no nested declaration and no interior newline,
    so local subdivision can only fall back to ``balanced-codepoint`` cuts.
    """
    prefix = 'X = "'
    suffix = '"\n'
    body = "a" * (total_chars - len(prefix) - len(suffix))
    text = prefix + body + suffix
    assert len(text) == total_chars
    return text


def _three_natural_units(first: int, second: int, third: int) -> str:
    """Three ``def`` statements whose bodies are string literals of the given
    code-point lengths, so the first is locally subdivided while the two small
    fitting units retain identity and co-pack."""

    def _decl(name: str, size: int) -> str:
        head = f"def {name}(): return "
        lit = '"' + "a" * (size - len(head) - 3) + '"'
        line = head + lit + "\n"
        assert len(line) == size, (name, len(line), size)
        return line

    return _decl("a", first) + _decl("b", second) + _decl("c", third)


def _plan(content: str, *, budget: int):
    return build_division_plan(
        rel_path="main.py",
        language="python",
        content=content,
        source_budget_chars=budget,
    )


def _tree(plan, *, synthesis: int = None):
    return build_reduction_tree(
        plan,
        synthesis_manifest_chars=synthesis
        or max(plan.source_budget_chars, file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
        imports=(),
    )


# ---------------------------------------------------------------------------
# A. Diagnostic constants + isolation
# ---------------------------------------------------------------------------


def test_diagnostic_constants_have_the_frozen_values():
    assert PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS == 20
    assert MAX_EPHEMERAL_PLAN_DETAIL_ITEMS == 4096
    assert PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS < MAX_EPHEMERAL_PLAN_DETAIL_ITEMS


def test_changing_detail_retention_cannot_change_division_bytes_or_topology(monkeypatch):
    """Section 5.4: the two caps bound diagnostic presentation only -- never
    chunking, reduction topology, call counts, or identity."""
    content = _three_natural_units(1243, 482, 285)
    baseline_plan = _plan(content, budget=1000)
    baseline_tree = _tree(baseline_plan)
    baseline = (
        baseline_plan.plan_digest,
        tuple(c.chunk_id for c in baseline_plan.chunks),
        tuple(c.payload for c in baseline_plan.chunks),
        baseline_tree.tree_digest,
        tuple(n.node_id for n in baseline_tree.all_nodes),
    )

    for detail_cap, record_cap in ((1, 1), (3, 2), (10**6, 10**5)):
        monkeypatch.setattr(file_division, "MAX_EPHEMERAL_PLAN_DETAIL_ITEMS", detail_cap)
        monkeypatch.setattr(
            file_division, "PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS", record_cap
        )
        plan = _plan(content, budget=1000)
        tree = _tree(plan)
        assert (
            plan.plan_digest,
            tuple(c.chunk_id for c in plan.chunks),
            tuple(c.payload for c in plan.chunks),
            tree.tree_digest,
            tuple(n.node_id for n in tree.all_nodes),
        ) == baseline


# ---------------------------------------------------------------------------
# B. Measured SplitCapacityBlocked
# ---------------------------------------------------------------------------


_EXPECTED_PHASE = {
    "atom-cap": "division-structure",
    "symbol-cap": "division-structure",
    "unit-cap": "division-packing",
    "chunk-cap": "division-packing",
    "reduction-envelope-cap": "reduction-envelope",
    "reduction-fan-in-cap": "reduction-fan-in",
    "reduction-depth-cap": "reduction-depth",
    "final-synthesis-envelope-cap": "final-synthesis",
}
_EXPECTED_GUIDANCE = {
    "atom-cap": "simplify-or-exclude",
    "symbol-cap": "simplify-or-exclude",
    "unit-cap": "simplify-or-exclude",
    "chunk-cap": "raise-source-ceiling-or-split-source",
    "reduction-envelope-cap": "report-planning-capacity-defect",
    "reduction-fan-in-cap": "report-planning-capacity-defect",
    "reduction-depth-cap": "report-planning-capacity-defect",
    "final-synthesis-envelope-cap": "inspect-authoritative-metadata-or-exclude",
}


@pytest.mark.parametrize("reason", BLOCKED_REASON_ORDER)
def test_split_capacity_blocked_carries_frozen_phase_guidance_and_measurements(reason):
    exc = SplitCapacityBlocked("pkg/mod.py", reason, observed=257, limit=256)
    assert exc.reason == reason
    assert exc.phase == _EXPECTED_PHASE[reason]
    assert exc.guidance_code == _EXPECTED_GUIDANCE[reason]
    assert exc.observed == 257
    assert exc.limit == 256
    detail = exc.detail
    assert set(detail) == {
        "path",
        "reason",
        "phase",
        "observed",
        "limit",
        "guidance_code",
    }
    assert detail["path"] == "pkg/mod.py"
    assert detail == blocked_split_descriptor(
        path="pkg/mod.py",
        reason=reason,
        phase=_EXPECTED_PHASE[reason],
        observed=257,
        limit=256,
        guidance_code=_EXPECTED_GUIDANCE[reason],
    )
    # value-safe: no source, prompt, id, range, or provider text
    assert canonical_json(detail) == canonical_json(detail)


def test_split_capacity_blocked_requires_real_measurements():
    with pytest.raises(TypeError):
        SplitCapacityBlocked("m.py", "atom-cap")  # missing observed/limit
    with pytest.raises(ValueError):
        SplitCapacityBlocked("m.py", "atom-cap", observed=-1, limit=10)
    with pytest.raises(ValueError):
        SplitCapacityBlocked("m.py", "not-a-reason", observed=1, limit=0)


def test_real_atom_cap_block_reports_actual_count_versus_structural_cap(monkeypatch):
    monkeypatch.setattr(file_division, "MAX_ATOMS_PER_FILE", 3)
    content = _three_natural_units(400, 360, 360) + 'def d(): return "xxxxxxxxxxxxxxx"\n'
    with pytest.raises(SplitCapacityBlocked) as caught:
        _plan(content, budget=1000)
    exc = caught.value
    assert exc.reason == "atom-cap"
    assert exc.limit == 3
    assert exc.observed > 3


def test_real_chunk_cap_block_reports_observed_chunk_count(monkeypatch):
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 2)
    content = _three_natural_units(1600, 1600, 1600)
    with pytest.raises(SplitCapacityBlocked) as caught:
        _plan(content, budget=1000)
    exc = caught.value
    assert exc.reason == "chunk-cap"
    assert exc.limit == 2
    assert exc.observed > 2
    assert exc.phase == "division-packing"
    assert exc.guidance_code == "raise-source-ceiling-or-split-source"


def test_real_fan_in_block_reports_child_capacity_versus_two(monkeypatch):
    monkeypatch.setattr(file_division, "REDUCTION_ENVELOPE_OVERHEAD_CHARS", 40)
    plan = _plan(_three_natural_units(1600, 1600, 1600), budget=1000)
    with pytest.raises(SplitCapacityBlocked) as caught:
        build_reduction_tree(
            plan, synthesis_manifest_chars=350, language="python", imports=()
        )
    exc = caught.value
    assert exc.reason == "reduction-fan-in-cap"
    assert exc.limit == 2
    assert exc.observed < 2
    assert exc.guidance_code == "report-planning-capacity-defect"


# ---------------------------------------------------------------------------
# D. Canonical full-stream integrity digest
# ---------------------------------------------------------------------------


def test_empty_stream_digest_is_the_digest_of_canonical_empty_list():
    import hashlib

    expected = "sha256:" + hashlib.sha256(b"[]").hexdigest()
    assert canonical_stream_digest(()) == expected
    assert canonical_stream_digest([]) == expected
    assert file_division.EMPTY_PLAN_DETAILS_DIGEST == expected


def test_stream_digest_frames_every_descriptor_and_is_duplicate_sensitive():
    import hashlib

    a = {"path": "a.py", "n": 1}
    b = {"path": "b.py", "n": 2}
    manual = hashlib.sha256()
    manual.update(b"[")
    manual.update(canonical_json(a).encode("utf-8"))
    manual.update(b",")
    manual.update(canonical_json(b).encode("utf-8"))
    manual.update(b",")
    manual.update(canonical_json(a).encode("utf-8"))
    manual.update(b"]")
    assert canonical_stream_digest([a, b, a]) == "sha256:" + manual.hexdigest()
    # duplicates are not collapsed
    assert canonical_stream_digest([a, a]) != canonical_stream_digest([a])
    # order matters (no commutative accumulator)
    assert canonical_stream_digest([a, b]) != canonical_stream_digest([b, a])


def test_stream_digest_changes_when_only_an_omitted_descriptor_changes():
    kept = [{"path": f"k{i}.py", "v": i} for i in range(3)]
    omitted_v1 = {"path": "z.py", "v": 1}
    omitted_v2 = {"path": "z.py", "v": 2}
    assert canonical_stream_digest([*kept, omitted_v1]) != canonical_stream_digest(
        [*kept, omitted_v2]
    )


# ---------------------------------------------------------------------------
# F. Split-plan descriptor schema + maintainer examples
# ---------------------------------------------------------------------------

# Defect D: every nested list also carries its four integrity siblings.
_NESTED_META = lambda name: {  # noqa: E731
    name + "_total",
    name + "_retained",
    name + "_omitted",
    name + "_digest",
}
_SPLIT_FILE_FIELDS = {
    "path",
    "source_chars",
    "structural_mode",
    "source_ceiling_chars",
    "synthesis_manifest_ceiling_chars",
    "reduction_levels",
    "reduction_calls",
    "final_calls",
    "initial_calls",
    "units",
    "leaves",
    *_NESTED_META("units"),
    *_NESTED_META("leaves"),
}
_UNIT_FIELDS = {
    "unit_ordinal",
    "natural_source_chars",
    "arithmetic_piece_count",
    "crlf_safe_piece_count",
    "atomicity_extra_piece_count",
    "pieces",
    *_NESTED_META("pieces"),
}
_PIECE_FIELDS = {
    "payload_chars",
    "start_boundary",
    "end_boundary",
    "boundary_constrained_small",
}
_LEAF_FIELDS = {
    "payload_chars",
    "close_reason",
    "start_boundary",
    "end_boundary",
    "constituents",
    *_NESTED_META("constituents"),
}
_CONSTITUENT_FIELDS = {"unit_ordinal", "payload_chars"}

# Actual source text, parser identifiers, ranges, and domain IDs -- none may
# appear in a value-safe descriptor.
_FORBIDDEN_SUBSTRINGS = (
    "return",
    "aaaa",
    "start_byte",
    "end_byte",
    "source_range",
    "atom_id",
    "qualified_name",
    "signature",
)


def _iter_nodes(detail):
    yield detail
    for unit in detail["units"]:
        yield unit
        for piece in unit["pieces"]:
            yield piece
    for leaf in detail["leaves"]:
        yield leaf
        for con in leaf["constituents"]:
            yield con


def _detail_for(content, *, budget):
    plan = _plan(content, budget=budget)
    tree = _tree(plan)
    return plan, tree, split_plan_file_detail(
        "main.py",
        plan,
        tree,
        source_ceiling_chars=budget,
        synthesis_manifest_ceiling_chars=tree.synthesis_manifest_chars,
    )


def test_split_detail_2010_reconstructs_balanced_670_thirds():
    plan, tree, detail = _detail_for(_one_oversized_span(2010), budget=1000)
    assert set(detail) == _SPLIT_FILE_FIELDS
    assert detail["source_chars"] == 2010
    assert detail["initial_calls"] == 5  # 3 leaf + 1 unit-consolidation + 1 final
    assert len(detail["units"]) == 1
    unit = detail["units"][0]
    assert set(unit) == _UNIT_FIELDS
    assert unit["natural_source_chars"] == 2010
    assert unit["arithmetic_piece_count"] == 3  # ceil(2010 / 1000)
    assert unit["crlf_safe_piece_count"] == 3
    assert unit["atomicity_extra_piece_count"] == 0
    payloads = [p["payload_chars"] for p in unit["pieces"]]
    assert payloads == [670, 670, 670]
    assert [p["end_boundary"] for p in unit["pieces"][:-1]] == [
        "balanced-codepoint",
        "balanced-codepoint",
    ]
    assert [len(leaf["constituents"]) for leaf in detail["leaves"]] == [1, 1, 1]
    assert [leaf["payload_chars"] for leaf in detail["leaves"]] == [670, 670, 670]


def test_split_detail_1243_482_285_subdivides_only_the_oversized_unit():
    plan, tree, detail = _detail_for(_three_natural_units(1243, 482, 285), budget=1000)
    units = detail["units"]
    assert len(units) == 3
    assert units[0]["natural_source_chars"] == 1243
    assert units[0]["arithmetic_piece_count"] == 2
    assert [p["payload_chars"] for p in units[0]["pieces"]] == [622, 621]
    # the two fitting units keep identity and carry no continuation pieces
    assert units[1]["natural_source_chars"] == 482
    assert units[2]["natural_source_chars"] == 285
    assert units[1]["pieces"] == []
    assert units[2]["pieces"] == []
    # ... and co-pack into one 767-character leaf with two constituents
    packed = [leaf for leaf in detail["leaves"] if len(leaf["constituents"]) == 2]
    assert len(packed) == 1
    assert packed[0]["payload_chars"] == 767
    assert sorted(c["payload_chars"] for c in packed[0]["constituents"]) == [285, 482]
    ordinals = [c["unit_ordinal"] for c in packed[0]["constituents"]]
    assert ordinals == [1, 2]


def test_split_detail_descriptors_never_leak_source_ids_or_ranges():
    import re

    _, _, detail = _detail_for(_three_natural_units(1243, 482, 285), budget=1000)
    blob = canonical_json(detail)
    for needle in _FORBIDDEN_SUBSTRINGS:
        assert needle not in blob, needle
    # integrity digests (`sha256:` + 64 hex) are value-safe metadata, not a
    # leak; every other 16+ hex run would be a domain identifier.
    stripped = re.sub(r"sha256:[0-9a-f]{64}", "", blob)
    assert not re.search(r"[0-9a-f]{16,}", stripped)
    _allowed_strings = {
        "main.py",
        "syntax",
        "lexical",
        "continuation",
        "oversized-unit-isolation",
        "source-ceiling",
        "metadata-ceiling",
        "source-and-metadata-ceiling",
        "end-of-file",
        "file-start",
        "file-end",
        "semantic-unit",
        "physical-line",
        "balanced-codepoint",
    }
    for node in _iter_nodes(detail):
        for value in node.values():
            if isinstance(value, str):
                assert "\n" not in value
                assert value in _allowed_strings or re.fullmatch(
                    r"sha256:[0-9a-f]{64}", value
                ), value


def test_split_detail_field_allowlists_are_exact():
    _, _, detail = _detail_for(_three_natural_units(1243, 482, 285), budget=1000)
    for unit in detail["units"]:
        assert set(unit) == _UNIT_FIELDS
        for piece in unit["pieces"]:
            assert set(piece) == _PIECE_FIELDS
    for leaf in detail["leaves"]:
        assert set(leaf) == _LEAF_FIELDS
        for con in leaf["constituents"]:
            assert set(con) == _CONSTITUENT_FIELDS


def test_close_reasons_stay_distinct_in_descriptors_and_full_stream_digest():
    """Mutation check 18: a fitting unit flushed before an oversized unit is
    ``oversized-unit-isolation``-closed and stays distinct from every ceiling
    reason and from ``end-of-file`` -- in the ordered leaf descriptor and in
    the ``split_plan`` full-stream digest."""
    content = _three_natural_units(400, 2010, 300)
    plan, tree, detail = _detail_for(content, budget=1000)
    close_reasons = [leaf["close_reason"] for leaf in detail["leaves"]]
    assert "oversized-unit-isolation" in close_reasons
    # the isolation-closed leaf is the fitting 400-char unit, not a piece
    isolation_leaves = [
        leaf for leaf in detail["leaves"] if leaf["close_reason"] == "oversized-unit-isolation"
    ]
    assert len(isolation_leaves) == 1
    assert isolation_leaves[0]["constituents"][0]["payload_chars"] == 400

    from codedoc.core.file_division import iter_split_plan_stream_items

    baseline_items = list(iter_split_plan_stream_items([detail]))
    baseline_digest = canonical_stream_digest(baseline_items)
    # Collapsing that reason to end-of-file (the mutation) changes the stream.
    collapsed = json.loads(json.dumps(detail))
    for leaf in collapsed["leaves"]:
        if leaf["close_reason"] == "oversized-unit-isolation":
            leaf["close_reason"] = "end-of-file"
    collapsed_digest = canonical_stream_digest(
        list(iter_split_plan_stream_items([collapsed]))
    )
    assert collapsed_digest != baseline_digest


def test_boundary_constrained_small_only_near_preferred_cuts():
    # a nested declaration gives an internal syntax candidate; a small piece
    # beside it is boundary-constrained-small, a small piece beside only
    # balanced-codepoint cuts is not.
    _, _, detail = _detail_for(_one_oversized_span(2010), budget=1000)
    # 670/670/670: none are < B/2, so none are constrained-small
    smalls = [
        p
        for unit in detail["units"]
        for p in unit["pieces"]
        if p["boundary_constrained_small"]
    ]
    assert smalls == []


# ---------------------------------------------------------------------------
# integration: aggregates + categories + persistence exclusion
# ---------------------------------------------------------------------------


_SPLIT_AGGREGATE_KEYS = (
    "large_file_strategy_resolved",
    "large_file_source_ceiling_chars",
    "large_files_over_source_ceiling",
    "large_files_routed_split",
    "large_files_routed_truncate",
    "truncate_retained_source_chars",
    "truncate_omitted_source_chars",
    "split_internal_manifest_budget_chars",
    "split_oversized_units",
    "split_continuation_chunks",
    "split_crlf_atomicity_extra_chunks",
    "split_boundary_cuts_syntax",
    "split_boundary_cuts_physical_line",
    "split_boundary_cuts_balanced_codepoint",
    "split_boundary_constrained_small_chunks",
    "split_closures_source_ceiling",
    "split_closures_metadata_ceiling",
    "split_closures_source_and_metadata_ceiling",
    "split_closures_oversized_unit_isolation",
    "split_closures_continuation",
    "split_closures_end_of_file",
    "split_metadata_limited_files",
    "split_metadata_limited_closures",
    "split_chunk_payload_chars_min",
    "split_chunk_payload_chars_max",
    # Section 6.3: route-wide EPHEMERAL recovery-transition counts (0 with no
    # recovery in play) -- published on every resolved-valid route.
    "split_recovery_discarded_predecessor_nodes",
    "split_recovery_replacement_nodes_planned",
)

_CATEGORY_SUFFIXES = (
    "_details",
    "_details_total",
    "_details_retained",
    "_details_omitted",
    "_details_digest",
)


def _run_split(tmp_path, monkeypatch, sources: dict, **overrides):
    from codedoc.pipeline import run_pipeline

    for name, text in sources.items():
        (tmp_path / name).write_text(text, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 1000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
        "dry_run": True,
        **overrides,
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("dry-run diagnostic substrate created a provider"),
    )
    return run_pipeline(tmp_path, config)


def test_split_run_publishes_every_aggregate_and_category(tmp_path, monkeypatch):
    stats = _run_split(
        tmp_path,
        monkeypatch,
        {
            "main.py": "VALUE = 1\n",
            "big.py": _three_natural_units(1243, 482, 285),
        },
    )
    for key in _SPLIT_AGGREGATE_KEYS:
        assert key in stats, key
    assert stats["split_recovery_discarded_predecessor_nodes"] == 0
    assert stats["split_recovery_replacement_nodes_planned"] == 0
    assert stats["large_file_strategy_resolved"] == "split"
    assert stats["large_file_source_ceiling_chars"] == 1000
    assert stats["split_internal_manifest_budget_chars"] == (
        file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
    )
    assert stats["large_files_routed_split"] == 1
    assert stats["large_files_routed_truncate"] == 0
    assert stats["split_oversized_units"] == 1
    assert stats["split_continuation_chunks"] == 2
    assert stats["split_crlf_atomicity_extra_chunks"] == 0
    assert stats["split_chunk_payload_chars_min"] > 0
    assert stats["split_chunk_payload_chars_max"] >= stats["split_chunk_payload_chars_min"]
    for category in ("split_plan", "truncate_plan", "split_blocked"):
        for suffix in _CATEGORY_SUFFIXES:
            assert category + suffix in stats, category + suffix
    assert stats["split_plan_details_total"] >= 1
    assert stats["split_plan_details_digest"].startswith("sha256:")
    assert stats["truncate_plan_details"] == []
    assert stats["truncate_plan_details_total"] == 0
    assert stats["truncate_plan_details_digest"] == (
        file_division.EMPTY_PLAN_DETAILS_DIGEST
    )
    assert "split_blocked_pairs" not in stats


def test_zero_leaf_run_reports_zero_payload_min_max_and_source_ceiling(
    tmp_path, monkeypatch
):
    stats = _run_split(tmp_path, monkeypatch, {"main.py": "VALUE = 1\n"})
    assert stats["large_file_source_ceiling_chars"] == 1000
    assert stats["split_chunk_payload_chars_min"] == 0
    assert stats["split_chunk_payload_chars_max"] == 0
    assert stats["large_files_over_source_ceiling"] == 0


def test_truncate_route_reports_head_tail_omission(tmp_path, monkeypatch):
    from codedoc.agents.base_agent import TRUNCATION_MARKER

    big = "x = 1\n" + "y = 2\n" * 900  # > 1000 chars
    stats = _run_split(
        tmp_path,
        monkeypatch,
        {"main.py": "VALUE = 1\n", "big.py": big},
        large_file_strategy="truncate",
    )
    assert stats["large_file_strategy_resolved"] == "truncate"
    assert stats["large_files_routed_truncate"] == 1
    assert stats["large_files_routed_split"] == 0
    detail = stats["truncate_plan_details"]
    assert len(detail) == 1
    entry = detail[0]
    assert set(entry) == {
        "path",
        "source_chars",
        "resolved_strategy",
        "retained_head_chars",
        "retained_tail_chars",
        "omitted_chars",
        "initial_calls",
    }
    budget = 1000 - len(TRUNCATION_MARKER)
    assert entry["retained_head_chars"] == int(budget * 0.70)
    assert entry["retained_tail_chars"] == budget - int(budget * 0.70)
    assert entry["omitted_chars"] == (
        entry["source_chars"] - entry["retained_head_chars"] - entry["retained_tail_chars"]
    )
    assert stats["truncate_retained_source_chars"] == (
        entry["retained_head_chars"] + entry["retained_tail_chars"]
    )
    assert stats["truncate_omitted_source_chars"] == entry["omitted_chars"]


def test_blocked_category_is_bounded_and_json_escaped(tmp_path, monkeypatch):
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 1)
    stats = _run_split(
        tmp_path,
        monkeypatch,
        {
            "main.py": "VALUE = 1\n",
            "zeta.py": _three_natural_units(1600, 1600, 1600),
            "alpha.py": _three_natural_units(1600, 1600, 1600),
        },
    )
    assert "split_blocked_pairs" not in stats
    details = stats["split_blocked_details"]
    assert stats["split_blocked_details_total"] == 2
    assert [d["path"] for d in details] == ["alpha.py", "zeta.py"]
    for d in details:
        assert set(d) == {
            "path",
            "reason",
            "phase",
            "observed",
            "limit",
            "guidance_code",
        }
        assert d["reason"] == "chunk-cap"
        assert d["observed"] > 1
        assert d["limit"] == 1
    assert stats["split_blocked_details_digest"].startswith("sha256:")


def test_triple_plus_split_helper_emits_no_split_stats(tmp_path, monkeypatch):
    from codedoc.pipeline import _split_division_stats

    assert _split_division_stats({"large_file_strategy": "split", "analysis_mode": "triple"}, None, None) == {}


def test_detail_fields_are_absent_from_persisted_last_run():
    from codedoc.core.project_view import build_project_view
    from tests.support.run_metadata_cases import _split_record, _split_stats

    stats = _split_stats()
    stats.update(
        {
            "split_plan_details": [{"path": "x"}],
            "split_plan_details_total": 1,
            "split_plan_details_retained": 1,
            "split_plan_details_omitted": 0,
            "split_plan_details_digest": "sha256:deadbeef",
            "truncate_plan_details": [],
            "truncate_plan_details_total": 0,
            "truncate_plan_details_retained": 0,
            "truncate_plan_details_omitted": 0,
            "truncate_plan_details_digest": "sha256:deadbeef",
            "split_blocked_details": [],
            "split_blocked_details_total": 0,
            "split_blocked_details_retained": 0,
            "split_blocked_details_omitted": 0,
            "split_blocked_details_digest": "sha256:deadbeef",
            "large_file_strategy_resolved": "split",
            # Section 6.3: the two EPHEMERAL recovery-transition counts.
            "split_recovery_discarded_predecessor_nodes": 5,
            "split_recovery_replacement_nodes_planned": 4,
        }
    )
    last_run = build_project_view([_split_record()], stats)["last_run"]
    for category in ("split_plan", "truncate_plan", "split_blocked"):
        for suffix in _CATEGORY_SUFFIXES:
            assert category + suffix not in last_run
    assert "large_file_strategy_resolved" not in last_run
    assert "split_recovery_discarded_predecessor_nodes" not in last_run
    assert "split_recovery_replacement_nodes_planned" not in last_run


def test_bounded_split_blocked_config_error_is_capped_and_escaped():
    """A hostile normalized relative path -- newline, tab, quote, bidi control
    -- survives as one ``json.dumps(path, ensure_ascii=True)`` field and cannot
    inject a terminal line or control sequence into the bounded ConfigError."""
    from codedoc.pipeline import _blocked_split_category, _blocked_split_files_message

    hostile = 'a\nb\t"c‮evil.py'
    blocked = _blocked_route_plan(
        [
            SplitCapacityBlocked(
                hostile, "chunk-cap", observed=257, limit=256
            ).detail,
            SplitCapacityBlocked(
                "z/plain.py", "atom-cap", observed=9000, limit=4096
            ).detail,
        ]
    )
    message = _blocked_split_files_message(blocked)
    assert "Split never falls back to truncation" in message
    assert "2 file(s) cannot be completely split-planned" in message
    escaped = json.dumps(hostile, ensure_ascii=True)
    assert escaped in message
    # the raw hostile path never lands verbatim (its newline/tab/control chars
    # are all escaped inside the single JSON field)
    assert hostile not in message
    assert "‮" not in message
    assert message.count("\n" + hostile) == 0
    category = _blocked_split_category(blocked)
    assert category["details_digest"] in message
    assert category["details_total"] == 2
    # display ranking is BLOCKED_REASON_ORDER then path: atom-cap outranks
    # chunk-cap regardless of path
    assert [d["path"] for d in category["details"]] == ["z/plain.py", hostile]


def test_split_plan_digest_is_stable_under_permuted_file_discovery(tmp_path, monkeypatch):
    sources = {
        "main.py": "VALUE = 1\n",
        "b_big.py": _three_natural_units(1243, 482, 285),
        "a_big.py": _one_oversized_span(2010),
    }
    first = _run_split(tmp_path, monkeypatch, sources)
    # a fresh tmp dir with the same files written in a different order
    other = tmp_path / "again"
    other.mkdir()
    from codedoc.pipeline import run_pipeline

    for name in reversed(list(sources)):
        (other / name).write_text(sources[name], encoding="utf-8", newline="")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("permutation run created a provider"),
    )
    second = run_pipeline(
        other,
        {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "large_file_strategy": "split",
            "max_content_chars": 1000,
            "parallel_agents": False,
            "propagate_changes": False,
            "output_dir": "docs",
            "dry_run": True,
        },
    )
    assert first["split_plan_details_digest"] == second["split_plan_details_digest"]
    assert first["split_plan_details_total"] == second["split_plan_details_total"]
    assert first["split_plan_details_retained"] == second["split_plan_details_retained"]


def test_split_digest_order_is_canonical_not_display_ranking():
    """Mutation check 24: the ``split_plan`` integrity stream is path-ascending
    and independent of the descending-initial-calls display ranking."""
    from codedoc.core.file_division import (
        build_split_plan_diagnostics,
        iter_split_plan_stream_items,
        split_plan_file_detail,
    )

    small = _plan(_one_oversized_span(2010), budget=1000)  # 3 leaves
    small_tree = _tree(small)
    big = _plan(_three_natural_units(1243, 1243, 1243), budget=1000)  # more leaves
    big_tree = _tree(big)
    # Producer-canonical (path-ascending) input order [a_small, z_big]
    # deliberately disagrees with the descending-initial-calls display ranking
    # [z_big, a_small].
    a_ic = (
        len(small.chunks)
        + len(small_tree.unit_consolidation_nodes)
        + len(small_tree.general_nodes)
        + 1
    )
    z_ic = (
        len(big.chunks)
        + len(big_tree.unit_consolidation_nodes)
        + len(big_tree.general_nodes)
        + 1
    )
    entries = [
        ("a_small.py", small, small_tree, 1000, small_tree.synthesis_manifest_chars, a_ic),
        ("z_big.py", big, big_tree, 1000, big_tree.synthesis_manifest_chars, z_ic),
    ]
    assert len(big.chunks) > len(small.chunks)
    category = build_split_plan_diagnostics(iter(entries))
    # display ranking puts the higher-initial-calls file first ...
    assert category["details"][0]["path"] == "z_big.py"
    assert category["details"][1]["path"] == "a_small.py"
    # ... but the digest is the input-order (producer-canonical) stream, not the
    # display ranking: reconstruct it independently in that order.
    canonical = []
    for rel, plan, tree, sc, mc, _ic in entries:
        canonical.extend(
            iter_split_plan_stream_items(
                [
                    split_plan_file_detail(
                        rel,
                        plan,
                        tree,
                        source_ceiling_chars=sc,
                        synthesis_manifest_ceiling_chars=mc,
                        initial_calls=_ic,
                    )
                ]
            )
        )
    assert category["details_digest"] == canonical_stream_digest(canonical)
    # same input twice -> identical output (deterministic, no hidden ordering)
    again = build_split_plan_diagnostics(iter(entries))
    assert again["details_digest"] == category["details_digest"]
    assert [d["path"] for d in again["details"]] == [
        d["path"] for d in category["details"]
    ]


def test_retained_split_items_never_exceed_the_flattened_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(file_division, "MAX_EPHEMERAL_PLAN_DETAIL_ITEMS", 12)
    sources = {"main.py": "VALUE = 1\n"}
    for i in range(6):
        sources[f"big_{i}.py"] = _three_natural_units(1243, 482, 285)
    stats = _run_split(tmp_path, monkeypatch, sources)

    def _count(details) -> int:
        n = 0
        for f in details:
            n += 1
            for u in f["units"]:
                n += 1
                n += len(u["pieces"])
            for leaf in f["leaves"]:
                n += 1
                n += len(leaf["constituents"])
        return n

    assert _count(stats["split_plan_details"]) <= 12
    assert stats["split_plan_details_retained"] <= 12
    assert stats["split_plan_details_total"] > stats["split_plan_details_retained"]
    assert stats["split_plan_details_omitted"] == (
        stats["split_plan_details_total"] - stats["split_plan_details_retained"]
    )
    # digest still covers the complete stream regardless of retention
    assert stats["split_plan_details_digest"].startswith("sha256:")
    assert stats["split_plan_details_total"] >= 6  # at least one header per file


# ===========================================================================
# Section 6 observability correction -- defects A, B, C, D
# ===========================================================================

import tracemalloc  # noqa: E402

from codedoc.core.file_division import (  # noqa: E402
    EMPTY_PLAN_DETAILS_DIGEST,
    RoutePlanEntry,
    build_flat_plan_diagnostics,
    build_split_plan_diagnostics,
)


class _OneShot:
    """A strictly one-shot iterable: no ``len``, no indexing, no replay."""

    def __init__(self, items):
        self._it = iter(items)
        self._used = False

    def __iter__(self):
        if self._used:
            raise AssertionError("diagnostic helper iterated its input twice")
        self._used = True
        return self._it


def _split_route_entry(rel, content, *, budget=1000, initial_calls=None):
    plan = _plan(content, budget=budget)
    tree = _tree(plan)
    if initial_calls is None:
        initial_calls = (
            len(plan.chunks)
            + len(tree.unit_consolidation_nodes)
            + len(tree.general_nodes)
            + 1
        )
    return (rel, plan, tree, budget, tree.synthesis_manifest_chars, initial_calls)


# ---------------------------------------------------------------------------
# Defect C -- bounded memory / one-shot inputs
# ---------------------------------------------------------------------------


def _maximal_shape_source(units: int = 110, body: int = 900) -> str:
    """Many independent ``def`` declarations, each just over half the 1000-char
    ceiling so almost none co-pack -> a file with ~``units`` semantic units and
    ~``units`` leaf calls (a near-maximal single-file descriptor shape)."""
    lines = []
    for index in range(units):
        head = f"def declaration_{index:04d}():\n    literal = "
        filler = '"' + "a" * (body - len(head) - 3) + '"'
        lines.append(head + filler + "\n")
    return "".join(lines)


def test_split_diagnostics_accepts_a_one_shot_iterable():
    entries = [
        _split_route_entry("a.py", _one_oversized_span(2010)),
        _split_route_entry("b.py", _three_natural_units(1243, 482, 285)),
    ]
    result = build_split_plan_diagnostics(_OneShot(entries))
    assert result["details_total"] > 0
    assert result["details_digest"].startswith("sha256:")
    assert [d["path"] for d in result["details"]]  # non-empty


def test_split_diagnostics_peak_memory_is_bounded_for_maximum_shape_files():
    """Defect C maximum-shape resource test: with a fixed cap, adding more
    near-maximal-shape files (each ~130 units / ~130 leaves) does not grow the
    diagnostic peak allocation -- it stays bounded by K plus one file."""
    cap = 60
    body = _maximal_shape_source()
    big = _plan(body, budget=1000)
    assert len(big.chunks) >= 100  # genuinely large single-file shape

    def _peak(n_files):
        entries = [_split_route_entry(f"m{i:04d}.py", body) for i in range(n_files)]
        tracemalloc.start()
        tracemalloc.reset_peak()
        result = build_split_plan_diagnostics(iter(entries), item_budget=cap)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        flat = 0
        for f in result["details"]:
            flat += 1
            for u in f["units"]:
                flat += 1 + len(u["pieces"])
            for leaf in f["leaves"]:
                flat += 1 + len(leaf["constituents"])
        assert flat <= cap
        assert result["details_total"] > cap
        return peak

    small = _peak(6)
    large = _peak(60)
    assert large < small * 3, (small, large)


def test_flat_diagnostics_accepts_a_one_shot_iterable():
    records = [
        blocked_split_descriptor(
            path=p, reason="chunk-cap", phase="division-packing",
            observed=300, limit=256,
            guidance_code="raise-source-ceiling-or-split-source",
        )
        for p in ("a.py", "b.py", "c.py")
    ]
    result = build_flat_plan_diagnostics(
        _OneShot(records),
        rank_key=lambda r: (BLOCKED_REASON_ORDER.index(r["reason"]), r["path"]),
    )
    assert result["details_total"] == 3
    assert result["details_digest"] == canonical_stream_digest(records)


def test_split_diagnostics_peak_memory_does_not_scale_with_file_count():
    """Defect C: with a fixed cap, diagnostic peak allocation is bounded by
    K + one maximal file, not by the number of oversized files."""
    cap = 40
    body = _three_natural_units(1243, 482, 285)

    def _peak(n_files):
        entries = [_split_route_entry(f"f{i:04d}.py", body) for i in range(n_files)]
        tracemalloc.start()
        tracemalloc.reset_peak()
        build_split_plan_diagnostics(iter(entries), item_budget=cap)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small = _peak(30)
    large = _peak(600)
    # 20x more files must not produce anywhere near 20x the peak diagnostic
    # allocation; allow generous slack for interpreter noise.
    assert large < small * 3, (small, large)


def test_split_diagnostics_per_file_peak_is_independent_of_chunk_count():
    """Defect C1: with a fixed item budget and ONE retained file, the diagnostic
    peak must not scale with that file's chunk count. Auxiliary memory must be
    O(K + nesting_depth) -- NOT O(K + one maximal file). The immutable plan and
    tree are built before tracemalloc starts, so only diagnostic-owned
    allocations are measured.
    """
    import gc

    def _peak(span_chars):
        plan = _plan(_one_oversized_span(span_chars), budget=1000)
        tree = _tree(plan)
        entry = (
            "only.py",
            plan,
            tree,
            1000,
            tree.synthesis_manifest_chars,
            len(plan.chunks)
            + len(tree.unit_consolidation_nodes)
            + len(tree.general_nodes)
            + 1,
        )
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        result = build_split_plan_diagnostics(iter([entry]), item_budget=2)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # retention really is tiny
        d = result["details"][0]
        flat = 1
        for u in d["units"]:
            flat += 1 + len(u["pieces"])
        for leaf in d["leaves"]:
            flat += 1 + len(leaf["constituents"])
        assert flat <= 2
        # nested full-list digests still cover the COMPLETE lists
        assert d["units_total"] == len(plan.atoms)
        assert d["leaves_total"] == len(plan.chunks)
        assert d["units_digest"].startswith("sha256:")
        assert d["leaves_digest"].startswith("sha256:")
        return len(plan.chunks), peak

    small_chunks, small = _peak(12_000)      # ~12 continuation chunks
    large_chunks, large = _peak(240_000)     # ~240 continuation chunks
    assert large_chunks > small_chunks * 15  # the large file really is ~20x
    assert large < small * 3, (small_chunks, small, large_chunks, large)


@pytest.mark.parametrize(
    "body",
    [
        _one_oversized_span(2010),
        _three_natural_units(1243, 482, 285),
        _three_natural_units(700, 1300, 900),
    ],
)
def test_atom_and_unit_counts_are_positionally_one_to_one(body):
    """The streaming path relies on ``len(plan.atoms) == len(plan.units)`` with
    ``plan.units[i]`` derived from ``plan.atoms[i]`` (source order). Lock it."""
    plan = _plan(body, budget=1000)
    assert len(plan.atoms) == len(plan.units)
    for index, unit in enumerate(plan.units):
        assert unit.atom_ids == (plan.atoms[index].atom_id,)


def test_split_diagnostics_streaming_matches_full_detail_digests():
    """The streamed nested full-list digests equal the full-detail helper's,
    proving the streaming path did not silently change the digest contract
    while removing the materialized lists."""
    from codedoc.core.file_division import iter_split_plan_stream_items

    plan = _plan(_three_natural_units(2400, 900, 900), budget=1000)
    tree = _tree(plan)
    ic = (
        len(plan.chunks)
        + len(tree.unit_consolidation_nodes)
        + len(tree.general_nodes)
        + 1
    )
    full = split_plan_file_detail(
        "m.py", plan, tree,
        source_ceiling_chars=1000,
        synthesis_manifest_ceiling_chars=tree.synthesis_manifest_chars,
        initial_calls=ic,
    )
    # retain everything so the streamed shell mirrors `full`
    result = build_split_plan_diagnostics(
        iter([("m.py", plan, tree, 1000, tree.synthesis_manifest_chars, ic)]),
        item_budget=MAX_EPHEMERAL_PLAN_DETAIL_ITEMS,
    )
    d = result["details"][0]
    assert d["units_digest"] == full["units_digest"]
    assert d["leaves_digest"] == full["leaves_digest"]
    assert d["units_total"] == full["units_total"]
    assert d["leaves_total"] == full["leaves_total"]
    for got, want in zip(d["units"], full["units"]):
        assert got["pieces_digest"] == want["pieces_digest"]
        assert got["pieces_total"] == want["pieces_total"]
    for got, want in zip(d["leaves"], full["leaves"]):
        assert got["constituents_digest"] == want["constituents_digest"]
        assert got["constituents_total"] == want["constituents_total"]
    assert result["details_digest"] == canonical_stream_digest(
        list(iter_split_plan_stream_items([full]))
    )


def test_increasing_omitted_records_does_not_grow_retained_output():
    body = _three_natural_units(1243, 482, 285)
    cap = 24
    r_small = build_split_plan_diagnostics(
        iter([_split_route_entry(f"f{i}.py", body) for i in range(4)]),
        item_budget=cap,
    )
    r_large = build_split_plan_diagnostics(
        iter([_split_route_entry(f"f{i:03d}.py", body) for i in range(80)]),
        item_budget=cap,
    )

    def _flat(details):
        n = 0
        for f in details:
            n += 1
            for u in f["units"]:
                n += 1 + len(u["pieces"])
            for leaf in f["leaves"]:
                n += 1 + len(leaf["constituents"])
        return n

    assert _flat(r_small["details"]) <= cap
    assert _flat(r_large["details"]) <= cap
    assert r_large["details_retained"] <= cap
    assert r_large["details_total"] > r_small["details_total"]
    assert r_large["details_omitted"] == (
        r_large["details_total"] - r_large["details_retained"]
    )


def test_route_plan_is_canonically_ordered_regardless_of_discovery(tmp_path):
    """Defect C: canonical input order is established by the provider-free plan
    producer, so the diagnostics layer never sorts an unbounded copy. The
    route-plan view is normalized-path-ascending whatever order discovery /
    graph traversal visited the files in."""
    from codedoc.core.planning import build_pipeline_plan
    from tests.support.pipeline_usage import make_graph

    for name in ("zeta.py", "alpha.py", "mid.py"):
        (tmp_path / name).write_text(
            _one_oversized_span(2010), encoding="utf-8", newline=""
        )
    (tmp_path / "main.py").write_text("import zeta\n", encoding="utf-8", newline="")
    file_map = {
        rel: {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
        for rel in ("main.py", "zeta.py", "alpha.py", "mid.py")
    }
    graph = make_graph("main.py", "zeta.py", "alpha.py", "mid.py")
    _plan, materials = build_pipeline_plan(
        file_map,
        graph,
        set(file_map),
        "main.py",
        {},
        [],
        {
            "large_file_strategy": "split",
            "analysis_mode": "single",
            "max_content_chars": 1000,
            "documentation_scope": "all",
            "propagate_changes": False,
        },
    )
    paths = [entry.rel_path for entry in materials.route_plan]
    assert paths == sorted(paths)
    assert set(paths) == {"alpha.py", "mid.py", "zeta.py"}
    assert all(entry.route == "split" for entry in materials.route_plan)


# ---------------------------------------------------------------------------
# Defect D -- nested list integrity metadata
# ---------------------------------------------------------------------------

_NESTED_SIBLINGS = ("_total", "_retained", "_omitted", "_digest")


def _expected_pieces_digest(unit):
    return canonical_stream_digest(
        [
            {
                "payload_chars": p["payload_chars"],
                "start_boundary": p["start_boundary"],
                "end_boundary": p["end_boundary"],
                "boundary_constrained_small": p["boundary_constrained_small"],
            }
            for p in unit["pieces"]
        ]
    )


def _expected_constituents_digest(leaf):
    return canonical_stream_digest(
        [
            {"unit_ordinal": c["unit_ordinal"], "payload_chars": c["payload_chars"]}
            for c in leaf["constituents"]
        ]
    )


def _expected_units_digest(units):
    # Round 3 P1: the ACTUAL complete unit members -- scalar fields plus the
    # full ordered `pieces` list inline (not a pieces_total / pieces_digest
    # summary).
    return canonical_stream_digest(
        [
            {
                "unit_ordinal": u["unit_ordinal"],
                "natural_source_chars": u["natural_source_chars"],
                "arithmetic_piece_count": u["arithmetic_piece_count"],
                "crlf_safe_piece_count": u["crlf_safe_piece_count"],
                "atomicity_extra_piece_count": u["atomicity_extra_piece_count"],
                "pieces": [
                    {
                        "payload_chars": p["payload_chars"],
                        "start_boundary": p["start_boundary"],
                        "end_boundary": p["end_boundary"],
                        "boundary_constrained_small": p["boundary_constrained_small"],
                    }
                    for p in u["pieces"]
                ],
            }
            for u in units
        ]
    )


def _expected_leaves_digest(leaves):
    return canonical_stream_digest(
        [
            {
                "payload_chars": leaf["payload_chars"],
                "close_reason": leaf["close_reason"],
                "start_boundary": leaf["start_boundary"],
                "end_boundary": leaf["end_boundary"],
                "constituents": [
                    {
                        "unit_ordinal": c["unit_ordinal"],
                        "payload_chars": c["payload_chars"],
                    }
                    for c in leaf["constituents"]
                ],
            }
            for leaf in leaves
        ]
    )


def test_full_detail_nested_lists_carry_reconstructable_integrity_metadata():
    _, _, detail = _detail_for(_three_natural_units(1243, 482, 285), budget=1000)
    for name in ("units", "leaves"):
        for sib in _NESTED_SIBLINGS:
            assert name + sib in detail, name + sib
        assert detail[name + "_total"] == len(detail[name])
        assert detail[name + "_retained"] == len(detail[name])
        assert detail[name + "_omitted"] == 0
    assert detail["units_digest"] == _expected_units_digest(detail["units"])
    assert detail["leaves_digest"] == _expected_leaves_digest(detail["leaves"])
    for unit in detail["units"]:
        for sib in _NESTED_SIBLINGS:
            assert "pieces" + sib in unit
        assert unit["pieces_total"] == len(unit["pieces"])
        assert unit["pieces_omitted"] == 0
        assert unit["pieces_digest"] == _expected_pieces_digest(unit)
    for leaf in detail["leaves"]:
        for sib in _NESTED_SIBLINGS:
            assert "constituents" + sib in leaf
        assert leaf["constituents_total"] == len(leaf["constituents"])
        assert leaf["constituents_digest"] == _expected_constituents_digest(leaf)
    # a unit with no continuation pieces hashes canonical []
    empties = [u for u in detail["units"] if u["pieces_total"] == 0]
    assert empties
    assert empties[0]["pieces_digest"] == EMPTY_PLAN_DETAILS_DIGEST


def test_partial_nested_retention_discloses_full_list_metadata():
    """Defect D: a tiny budget truncates every nesting level; each retained
    parent still discloses its child list's exact total/retained/omitted/digest,
    and nested metadata does not consume the flattened item budget."""
    entry = _split_route_entry("big.py", _three_natural_units(2400, 900, 900))
    # full (unbounded) reference detail for independent digest reconstruction
    _, _, full = _detail_for(_three_natural_units(2400, 900, 900), budget=1000)

    result = build_split_plan_diagnostics(iter([entry]), item_budget=6)
    assert len(result["details"]) == 1
    d = result["details"][0]
    # file-level nested metadata reflects the COMPLETE lists (digests
    # independently reconstructed from the full unbounded detail)
    assert d["units_total"] == len(full["units"])
    assert d["leaves_total"] == len(full["leaves"])
    assert d["units_digest"] == _expected_units_digest(full["units"])
    assert d["leaves_digest"] == _expected_leaves_digest(full["leaves"])
    assert d["units_retained"] == len(d["units"])
    assert d["units_omitted"] == d["units_total"] - d["units_retained"]
    assert d["leaves_retained"] == len(d["leaves"])
    assert d["leaves_omitted"] == d["leaves_total"] - d["leaves_retained"]
    # at least one nesting level is actually truncated by the tiny budget
    assert (d["units_omitted"] > 0) or (d["leaves_omitted"] > 0) or any(
        u["pieces_omitted"] > 0 for u in d["units"]
    )
    # every retained unit still discloses its full pieces list
    for i, unit in enumerate(d["units"]):
        assert unit["pieces_total"] == len(full["units"][i]["pieces"])
        assert unit["pieces_digest"] == _expected_pieces_digest(full["units"][i])
        assert unit["pieces_retained"] == len(unit["pieces"])
        assert unit["pieces_omitted"] == unit["pieces_total"] - unit["pieces_retained"]
    for i, leaf in enumerate(d["leaves"]):
        assert leaf["constituents_total"] == len(full["leaves"][i]["constituents"])
        assert leaf["constituents_digest"] == _expected_constituents_digest(
            full["leaves"][i]
        )
    # details_total counts only flattened descriptors; nested metadata is free
    flat = 1
    for u in d["units"]:
        flat += 1 + len(u["pieces"])
    for leaf in d["leaves"]:
        flat += 1 + len(leaf["constituents"])
    assert result["details_retained"] == flat
    assert flat <= 6


# ---------------------------------------------------------------------------
# Defect A -- reused oversized truncate route stays visible
# ---------------------------------------------------------------------------


def test_reused_oversized_truncate_route_stays_visible_with_zero_initial_calls(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline
    from codedoc.agents.base_agent import TRUNCATION_MARKER
    from tests.support.providers import SmartFake

    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8", newline="")
    (tmp_path / "big.py").write_text(
        "x = 1\n" + "y = 2\n" * 420, encoding="utf-8", newline=""
    )
    cfg = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "truncate",
        "max_content_chars": 1000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    first = run_pipeline(tmp_path, cfg)
    assert first["checked"] >= 1

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("reuse run created a provider"),
    )
    stats = run_pipeline(tmp_path, cfg)

    assert stats["checked"] == 0
    assert stats.get("would_call_llm_for", 0) == 0
    assert stats["large_file_strategy_resolved"] == "truncate"
    assert stats["large_files_over_source_ceiling"] == 1
    assert stats["large_files_routed_truncate"] == 1
    assert stats["large_files_routed_split"] == 0
    details = stats["truncate_plan_details"]
    assert len(details) == 1
    entry = details[0]
    assert entry["path"] == "big.py"
    assert entry["initial_calls"] == 0
    budget = 1000 - len(TRUNCATION_MARKER)
    assert entry["retained_head_chars"] == int(budget * 0.70)
    assert entry["retained_tail_chars"] == budget - int(budget * 0.70)
    assert entry["omitted_chars"] == (
        entry["source_chars"]
        - entry["retained_head_chars"]
        - entry["retained_tail_chars"]
    )
    assert stats["truncate_plan_details_total"] == 1
    assert stats["truncate_plan_details_retained"] == 1
    assert stats["truncate_plan_details_omitted"] == 0
    assert stats["truncate_plan_details_digest"].startswith("sha256:")
    assert stats["truncate_retained_source_chars"] == (
        entry["retained_head_chars"] + entry["retained_tail_chars"]
    )
    assert stats["truncate_omitted_source_chars"] == entry["omitted_chars"]


# ---------------------------------------------------------------------------
# Defect B -- split initial_calls is recovery-aware
# ---------------------------------------------------------------------------


def _split_big_source() -> str:
    return "".join(f"def function_{i}():\n    return {i}\n\n" for i in range(220))


def _split_cfg(**over):
    return {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        **over,
    }


def _route_entry_for(materials, rel):
    matches = [e for e in materials.route_plan if e.rel_path == rel]
    assert len(matches) == 1, (rel, [e.rel_path for e in materials.route_plan])
    return matches[0]


def test_completed_split_reuse_route_reports_zero_initial_calls(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    (tmp_path / "main.py").write_text(
        _split_big_source(), encoding="utf-8", newline=""
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    first = run_pipeline(tmp_path, _split_cfg())
    assert first["checked"] == 1

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("reuse run created a provider"),
    )
    stats = run_pipeline(tmp_path, _split_cfg())
    assert stats["split_completed_files_reused"] == 1
    assert stats["large_files_routed_split"] == 1
    assert stats["large_files_over_source_ceiling"] == 1
    paths = [d["path"] for d in stats["split_plan_details"]]
    assert "main.py" in paths
    detail = next(d for d in stats["split_plan_details"] if d["path"] == "main.py")
    assert detail["initial_calls"] == 0


def test_partial_split_route_initial_calls_equals_unretained_planned_nodes(
    tmp_path, monkeypatch
):
    """Defect B control 3: a compatible partial reports exactly the count of
    current planned node IDs not retained as valid completed nodes."""
    import hashlib

    import codedoc.core.file_division as fd
    from codedoc.core.planning import build_pipeline_plan
    from tests.support.pipeline_usage import make_graph

    source = _split_big_source()
    src = tmp_path / "main.py"
    src.write_text(source, encoding="utf-8", newline="")
    cfg = _split_cfg(max_content_chars=2000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    division = fd.build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = fd.build_reduction_tree(
        division,
        synthesis_manifest_chars=max(2000, fd.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    provider_identity = fd.provider_execution_identity(cfg)

    def _leaf_state(chunk, index):
        return fd.tree_node_state(
            node_id=chunk.chunk_id,
            node_type="leaf",
            rel_path="main.py",
            content_hash=content_hash,
            division_plan_digest=division.plan_digest,
            input_digest=fd.leaf_input_digest(
                rel_path="main.py",
                language="python",
                chunk=chunk,
                unit_indexes=division.unit_positions(chunk),
                unit_count=len(division.units),
            ),
            execution_identity_digest=fd.leaf_execution_identity(
                rel_path="main.py",
                content_hash=content_hash,
                division_plan_digest=division.plan_digest,
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

    graph = make_graph("main.py")
    file_map = {
        "main.py": {
            "path": src,
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }
    planned_ids = {c.chunk_id for c in division.chunks} | {
        n.node_id for n in tree.all_nodes
    }
    retained_leaf_ids = {c.chunk_id for c in division.chunks}
    recovered_partial = fd.SplitTreeState(
        schema_version=fd.SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=division.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(
            _leaf_state(chunk, index) for index, chunk in enumerate(division.chunks)
        ),
    )
    _plan, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, [], cfg,
        recovered_partials={"main.py": recovered_partial},
    )
    entry = _route_entry_for(materials, "main.py")
    assert isinstance(entry, RoutePlanEntry)
    assert entry.route == "split"
    # exactly the current planned node IDs not retained as valid completed leaves
    assert entry.split_payable_calls == len(planned_ids - retained_leaf_ids)
    assert entry.split_payable_calls == len(tree.all_nodes)  # every reducer + final
    assert entry.split_payable_calls > 0
    assert entry.payable is True


# ===========================================================================
# Section 6 Round 3 -- actual-list nested digests, true breadth-first,
# recovery diagnostic controls
# ===========================================================================

_PREFERRED = frozenset({"syntax", "physical-line"})


def _actual_nested_lists_from_plan(plan):
    """Reconstruct the COMPLETE actual unit and leaf member lists straight from
    the immutable plan -- deliberately NOT via any production digest helper
    (`_unit_digest_member`, `_leaf_digest_member`, `_nested_metadata`,
    `split_plan_unit_summaries`, `split_plan_leaf_descriptors`,
    `split_plan_file_detail`)."""
    budget = plan.source_budget_chars
    atoms = plan.atoms
    chunks = plan.chunks

    def _bcs(payload_chars, start_boundary, end_boundary):
        near = start_boundary in _PREFERRED or end_boundary in _PREFERRED
        return near and (2 * payload_chars < budget)

    units = []
    i = 0
    ordinal = 0
    n = len(chunks)
    while i < n:
        chunk = chunks[i]
        if chunk.unit_chunk_count > 1:
            count = chunk.unit_chunk_count
            natural = len(atoms[ordinal].source)
            arithmetic = -(-natural // budget) if natural else 1
            pieces = []
            for p in range(count):
                piece = chunks[i + p]
                pieces.append(
                    {
                        "payload_chars": piece.payload_chars,
                        "start_boundary": piece.start_boundary,
                        "end_boundary": piece.end_boundary,
                        "boundary_constrained_small": _bcs(
                            piece.payload_chars,
                            piece.start_boundary,
                            piece.end_boundary,
                        ),
                    }
                )
            units.append(
                {
                    "unit_ordinal": ordinal,
                    "natural_source_chars": natural,
                    "arithmetic_piece_count": arithmetic,
                    "crlf_safe_piece_count": count,
                    "atomicity_extra_piece_count": count - arithmetic,
                    "pieces": pieces,
                }
            )
            i += count
            ordinal += 1
        else:
            for _su in chunk.semantic_units:
                natural = len(atoms[ordinal].source)
                arithmetic = -(-natural // budget) if natural else 1
                units.append(
                    {
                        "unit_ordinal": ordinal,
                        "natural_source_chars": natural,
                        "arithmetic_piece_count": arithmetic,
                        "crlf_safe_piece_count": 1,
                        "atomicity_extra_piece_count": 1 - arithmetic,
                        "pieces": [],
                    }
                )
                ordinal += 1
            i += 1

    leaves = []
    unit_cursor = 0
    for chunk in chunks:
        if chunk.unit_chunk_count > 1:
            constituents = [
                {"unit_ordinal": unit_cursor, "payload_chars": chunk.payload_chars}
            ]
            if chunk.unit_chunk_index == chunk.unit_chunk_count - 1:
                unit_cursor += 1
        else:
            constituents = [
                {
                    "unit_ordinal": unit_cursor + j,
                    "payload_chars": len(atoms[unit_cursor + j].source),
                }
                for j in range(len(chunk.semantic_units))
            ]
            unit_cursor += len(chunk.semantic_units)
        leaves.append(
            {
                "payload_chars": chunk.payload_chars,
                "close_reason": chunk.close_reason,
                "start_boundary": chunk.start_boundary,
                "end_boundary": chunk.end_boundary,
                "constituents": constituents,
            }
        )
    return units, leaves


# --------------------------------------------------------------------------- P1
# nested digests must hash the ACTUAL complete lists
# ---------------------------------------------------------------------------


def test_nested_digests_hash_the_actual_complete_lists_1243_482_285():
    """P1 failing-first: `units_digest` / `leaves_digest` must equal a
    canonical stream digest over the actual complete unit/leaf members
    (with the `pieces` / `constituents` arrays inline), not a summary schema."""
    plan = _plan(_three_natural_units(1243, 482, 285), budget=1000)
    tree = _tree(plan)
    ic = (
        len(plan.chunks)
        + len(tree.unit_consolidation_nodes)
        + len(tree.general_nodes)
        + 1
    )
    actual_units, actual_leaves = _actual_nested_lists_from_plan(plan)
    want_units = canonical_stream_digest(actual_units)
    want_leaves = canonical_stream_digest(actual_leaves)
    # pinned expected values from the Round-3 prompt
    assert want_units == (
        "sha256:cf6e4b603eb33ad3b0ec911d505e53ead5ac1f3d17ed13224723b116f4975b07"
    )
    assert want_leaves == (
        "sha256:24368e98d9497bcb5412d1e7118c8c05e4a2fbf4fa02e211d29834a3de4cea4e"
    )

    # bounded streaming path
    result = build_split_plan_diagnostics(
        iter([("main.py", plan, tree, 1000, tree.synthesis_manifest_chars, ic)]),
        item_budget=MAX_EPHEMERAL_PLAN_DETAIL_ITEMS,
    )
    published = result["details"][0]
    assert published["units_digest"] == want_units
    assert published["leaves_digest"] == want_leaves

    # full-detail / introspection helper must publish the SAME literal digests
    detail = split_plan_file_detail(
        "main.py",
        plan,
        tree,
        source_ceiling_chars=1000,
        synthesis_manifest_ceiling_chars=tree.synthesis_manifest_chars,
        initial_calls=ic,
    )
    assert detail["units_digest"] == want_units
    assert detail["leaves_digest"] == want_leaves


@pytest.mark.parametrize(
    "body",
    [
        _one_oversized_span(2010),
        _three_natural_units(1243, 482, 285),
        _three_natural_units(700, 1300, 900),
        _one_oversized_span(30_000),
    ],
)
def test_nested_digests_equal_actual_lists_for_many_shapes(body):
    plan = _plan(body, budget=1000)
    tree = _tree(plan)
    ic = (
        len(plan.chunks)
        + len(tree.unit_consolidation_nodes)
        + len(tree.general_nodes)
        + 1
    )
    actual_units, actual_leaves = _actual_nested_lists_from_plan(plan)
    result = build_split_plan_diagnostics(
        iter([("m.py", plan, tree, 1000, tree.synthesis_manifest_chars, ic)]),
        item_budget=MAX_EPHEMERAL_PLAN_DETAIL_ITEMS,
    )
    d = result["details"][0]
    assert d["units_digest"] == canonical_stream_digest(actual_units)
    assert d["leaves_digest"] == canonical_stream_digest(actual_leaves)
    detail = split_plan_file_detail(
        "m.py", plan, tree,
        source_ceiling_chars=1000,
        synthesis_manifest_ceiling_chars=tree.synthesis_manifest_chars,
        initial_calls=ic,
    )
    assert detail["units_digest"] == canonical_stream_digest(actual_units)
    assert detail["leaves_digest"] == canonical_stream_digest(actual_leaves)


def test_incremental_canonical_array_member_is_byte_equivalent_to_canonical_json():
    """The streaming nested-member encoder must emit the exact bytes
    `canonical_json({**scalars, arr_key: [items]})` would emit -- proven by
    reconstructing the whole member from the `(prefix, suffix)` frame and the
    per-item `canonical_json` bytes, then comparing byte-for-byte against
    `canonical_json` on the fully materialized member."""
    from codedoc.core.file_division import _actual_member_frame

    def _encode(scalars, arr_key, items):
        prefix, suffix = _actual_member_frame(scalars, arr_key)
        first = True
        body = b""
        for it in items:
            if not first:
                body += b","
            first = False
            body += canonical_json(it).encode("utf-8")
        return prefix + body + suffix

    def _reference(scalars, arr_key, items):
        return canonical_json({**scalars, arr_key: items}).encode("utf-8")

    cases = [
        ({"unit_ordinal": 0, "natural_source_chars": 1243,
          "arithmetic_piece_count": 2, "crlf_safe_piece_count": 2,
          "atomicity_extra_piece_count": 0}, "pieces",
         [{"payload_chars": 622, "start_boundary": "semantic-unit",
           "end_boundary": "balanced-codepoint", "boundary_constrained_small": False},
          {"payload_chars": 621, "start_boundary": "balanced-codepoint",
           "end_boundary": "semantic-unit", "boundary_constrained_small": False}]),
        ({"unit_ordinal": 1, "natural_source_chars": 482,
          "arithmetic_piece_count": 1, "crlf_safe_piece_count": 1,
          "atomicity_extra_piece_count": 0}, "pieces", []),
        ({"payload_chars": 767, "close_reason": "end-of-file",
          "start_boundary": "semantic-unit", "end_boundary": "file-end"},
         "constituents",
         [{"unit_ordinal": 1, "payload_chars": 482},
          {"unit_ordinal": 2, "payload_chars": 285}]),
        # duplicate members
        ({"payload_chars": 9, "close_reason": "continuation",
          "start_boundary": "syntax", "end_boundary": "syntax"}, "constituents",
         [{"unit_ordinal": 0, "payload_chars": 9},
          {"unit_ordinal": 0, "payload_chars": 9}]),
        # strings that require escaping (quote, newline, tab, backslash, control)
        ({"a": 'x"y' + chr(10) + chr(9) + chr(92) + "z", "b": 1}, "items",
         [{"k": "v" + chr(1) + "w"}, {"k": "plain"}]),
        # non-ascii + bidi control, empty list
        ({"a": "héllo‮"}, "items", []),
    ]
    for scalars, arr_key, items in cases:
        assert _encode(scalars, arr_key, items) == _reference(scalars, arr_key, items), (
            scalars, arr_key, items
        )


# --------------------------------------------------------------------------- P1
# true ranked breadth-first retention across files
# ---------------------------------------------------------------------------


def _rr_shape():
    """Two large first-ranked files (30 units / 30 leaves / 30 constituents =>
    90 nested each) and one short last-ranked file (1 unit / 0 pieces / 1 leaf /
    1 constituent => 3 nested). details_total == 186."""
    big = _maximal_shape_source(30, 900)
    a_plan = _plan(big, budget=1000)
    a_tree = _tree(a_plan)
    b_plan = _plan(big, budget=1000)
    b_tree = _tree(b_plan)
    z_plan = build_division_plan(
        rel_path="z.py", language="python", content="Z = 1\n", source_budget_chars=1000
    )
    z_tree = build_reduction_tree(
        z_plan, synthesis_manifest_chars=12000, language="python"
    )

    def _ic(plan, tree):
        return (
            len(plan.chunks)
            + len(tree.unit_consolidation_nodes)
            + len(tree.general_nodes)
            + 1
        )

    entries = [
        ("a.py", a_plan, a_tree, 1000, a_tree.synthesis_manifest_chars, _ic(a_plan, a_tree)),
        ("b.py", b_plan, b_tree, 1000, b_tree.synthesis_manifest_chars, _ic(b_plan, b_tree)),
        ("z.py", z_plan, z_tree, 1000, z_tree.synthesis_manifest_chars, _ic(z_plan, z_tree)),
    ]
    return entries


def _nested_count(file_detail):
    n = 0
    for u in file_detail["units"]:
        n += 1 + len(u["pieces"])
    for leaf in file_detail["leaves"]:
        n += 1 + len(leaf["constituents"])
    return n


def test_true_breadth_first_reclaims_unused_quota_from_a_short_last_file():
    """P1 failing-first: a short last-ranked file cannot use its per-file quota;
    the unused positions must be returned to the earlier, non-exhausted files."""
    entries = _rr_shape()
    result = build_split_plan_diagnostics(iter(entries), item_budget=40)

    assert result["details_total"] == 186
    # after the fix the whole budget is used
    assert result["details_retained"] == 40
    assert result["details_omitted"] == 186 - 40

    by_path = {d["path"]: d for d in result["details"]}
    assert sorted(by_path) == ["a.py", "b.py", "z.py"]
    assert _nested_count(by_path["a.py"]) == 17
    assert _nested_count(by_path["b.py"]) == 17
    assert _nested_count(by_path["z.py"]) == 3
    # z.py fully retained; a.py / b.py still have omitted descriptors
    assert by_path["z.py"]["units_omitted"] == 0
    assert by_path["z.py"]["leaves_omitted"] == 0
    assert by_path["a.py"]["units_omitted"] + by_path["a.py"]["leaves_omitted"] > 0
    assert by_path["b.py"]["units_omitted"] + by_path["b.py"]["leaves_omitted"] > 0


def test_breadth_first_partial_final_round_favours_the_earlier_ranked_file():
    """The extra item in an odd final partial round goes to the earlier-ranked
    still-active file."""
    from codedoc.core.file_division import _breadth_first_quotas

    # two equally-ranked files, capacities 5 and 5, 7 positions to spend
    assert _breadth_first_quotas([5, 5], 7) == [4, 3]
    # a short middle file is skipped once exhausted; leftovers go round-robin
    assert _breadth_first_quotas([90, 90, 3], 37) == [17, 17, 3]
    assert _breadth_first_quotas([2, 100], 10) == [2, 8]
    assert _breadth_first_quotas([100, 2], 10) == [8, 2]
    assert _breadth_first_quotas([1, 1, 1], 10) == [1, 1, 1]
    assert _breadth_first_quotas([], 10) == []
    assert _breadth_first_quotas([5, 5], 0) == [0, 0]


def test_breadth_first_retention_still_bounded_after_the_nested_digest_change():
    """The fixed-budget 12-vs-240-chunk memory probe stays bounded."""
    import gc
    import tracemalloc as _tm

    def _peak(span_chars):
        plan = _plan(_one_oversized_span(span_chars), budget=1000)
        tree = _tree(plan)
        entry = (
            "only.py", plan, tree, 1000, tree.synthesis_manifest_chars,
            len(plan.chunks)
            + len(tree.unit_consolidation_nodes)
            + len(tree.general_nodes)
            + 1,
        )
        gc.collect()
        _tm.start()
        _tm.reset_peak()
        build_split_plan_diagnostics(iter([entry]), item_budget=2)
        _cur, peak = _tm.get_traced_memory()
        _tm.stop()
        return len(plan.chunks), peak

    small_c, small = _peak(12_000)
    large_c, large = _peak(240_000)
    assert large_c > small_c * 15
    assert large < small * 3, (small_c, small, large_c, large)


# --------------------------------------------------------------------------- P2
# recovery diagnostic controls (newly added proof, not failing-first)
# ---------------------------------------------------------------------------


def _split_cfg_r3(**over):
    return {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        **over,
    }


def _r3_big_source():
    return "".join(f"def function_{i}():\n    return {i}\n\n" for i in range(220))


def _r3_fully_completed_tree_state(plan, tree, *, provider_identity, content_hash):
    """A synthetic but dependency-valid ``SplitTreeState`` covering every leaf,
    reducer, and final node -- ported from the canonical
    ``test_split_division._fully_completed_tree_state`` (the suite architecture
    contract forbids importing another test module, so it is reproduced here).
    Every stage-local input digest is recomputed from the exact fixture result
    content of its own children, mirroring the live executor."""
    import codedoc.core.file_division as fd
    from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST
    from codedoc.core.result_assembly import flat_combined_result

    results_by_id: dict = {}
    nodes = []
    for index, chunk in enumerate(plan.chunks):
        result = {
            "description": f"restored {index}",
            "chunk_id": chunk.chunk_id,
            "unit_id": chunk.unit_id,
        }
        results_by_id[chunk.chunk_id] = result
        nodes.append(
            fd.tree_node_state(
                node_id=chunk.chunk_id,
                node_type="leaf",
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                input_digest=fd.leaf_input_digest(
                    rel_path=plan.rel_path,
                    language="python",
                    chunk=chunk,
                    unit_indexes=plan.unit_positions(chunk),
                    unit_count=len(plan.units),
                ),
                execution_identity_digest=fd.leaf_execution_identity(
                    rel_path=plan.rel_path,
                    content_hash=content_hash,
                    division_plan_digest=plan.plan_digest,
                    provider_identity=provider_identity,
                    chunk=chunk,
                ),
                unit_id=None,
                child_ids=(),
                coverage_leaf_ids=(chunk.chunk_id,),
                result=result,
            )
        )
    for node in tree.unit_consolidation_nodes + tree.general_nodes:
        result = {"narrative": "restored narrative"}
        raw_narratives = tuple(
            results_by_id[child_id].get(
                "narrative", results_by_id[child_id].get("description", "")
            )
            for child_id in node.child_ids
        )
        nodes.append(
            fd.tree_node_state(
                node_id=node.node_id,
                node_type=node.phase,
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                input_digest=fd.reduction_input_digest(
                    rel_path=plan.rel_path,
                    phase=node.phase,
                    level=node.level,
                    unit_id=node.unit_id,
                    child_count=len(node.child_ids),
                    ordered_child_narratives=fd.refine_narrative_inputs(raw_narratives),
                ),
                execution_identity_digest=fd.reduction_execution_identity(
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
        )
        results_by_id[node.node_id] = result
    final = tree.final_node
    imports_digest = fd.deterministic_imports_digest(())
    leaf_capsules_ordered = [results_by_id[chunk.chunk_id] for chunk in plan.chunks]
    ledger = fd.build_fact_ledger(
        leaf_capsules_ordered, language="python", chunks=plan.chunks, symbols=plan.symbols
    )
    final_raw_narratives = tuple(
        results_by_id[child_id].get(
            "narrative", results_by_id[child_id].get("description", "")
        )
        for child_id in final.child_ids
    )
    manifest_json = fd.final_synthesis_input(
        rel_path=plan.rel_path,
        language="python",
        imports=(),
        root_narratives=fd.refine_narrative_inputs(final_raw_narratives),
        root_coverage_leaf_ids=final.leaf_ids,
        ledger=ledger,
        max_chars=plan.source_budget_chars,
    )
    nodes.append(
        fd.tree_node_state(
            node_id=final.node_id,
            node_type="final",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=fd.final_input_digest(
                imports_digest=imports_digest,
                resolved_shape_digest=NO_PROMPT_PROFILE_DIGEST,
                manifest_json=manifest_json,
            ),
            execution_identity_digest=fd.final_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                prompt_profile_digest=NO_PROMPT_PROFILE_DIGEST,
                imports_digest=imports_digest,
                node=final,
            ),
            unit_id=None,
            child_ids=final.child_ids,
            coverage_leaf_ids=final.leaf_ids,
            result=flat_combined_result(
                plan.rel_path, "python", [], {"description": "restored complete file"}
            ),
        )
    )
    return fd.SplitTreeState(
        schema_version=fd.SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(nodes),
    )


def test_fully_recovered_current_tree_reports_zero_payable(tmp_path):
    """Proof control: a genuine current-valid state containing every current
    leaf, reducer, and final node -> route == split, split_payable_calls == 0,
    payable is False, not in unpaid_action_rels, and the retained split
    diagnostic detail (built through the real ``build_split_plan_diagnostics``
    path) reports ``initial_calls == 0``."""
    import hashlib

    import codedoc.core.file_division as fd
    from codedoc.core.planning import build_pipeline_plan
    from tests.support.pipeline_usage import make_graph

    source = _r3_big_source()
    src = tmp_path / "main.py"
    src.write_text(source, encoding="utf-8", newline="")
    cfg = _split_cfg_r3()
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    division = fd.build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = fd.build_reduction_tree(
        division,
        synthesis_manifest_chars=max(2000, fd.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    provider_identity = fd.provider_execution_identity(cfg)

    fully = _r3_fully_completed_tree_state(
        division,
        tree,
        provider_identity=provider_identity,
        content_hash=content_hash,
    )
    graph = make_graph("main.py")
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py", "language": "python", "extension": ".py",
        }
    }
    plan, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, [], cfg,
        recovered_partials={"main.py": fully},
    )
    current_planned_ids = {c.chunk_id for c in division.chunks} | {
        n.node_id for n in tree.all_nodes
    }
    retained_ids = set(materials.tree_states["main.py"].by_id())
    assert retained_ids == current_planned_ids
    entry = next(e for e in materials.route_plan if e.rel_path == "main.py")
    assert entry.route == "split"
    assert entry.split_payable_calls == 0
    assert entry.payable is False
    assert "main.py" not in plan.unpaid_action_rels

    # Diagnostic-snapshot coverage: feed the real ``materials.route_plan``
    # through ``build_split_plan_diagnostics`` exactly as the pipeline does
    # (only ``route == "split"`` entries), then assert the retained split
    # detail for ``main.py`` reports the recovery-aware ``initial_calls`` as
    # exactly zero -- not a fabricated dict, not a private replica.
    split_category = build_split_plan_diagnostics(
        e for e in materials.route_plan if e.route == "split"
    )
    assert split_category["details_retained"] >= 1
    main_detail = next(
        d for d in split_category["details"] if d["path"] == "main.py"
    )
    assert main_detail["initial_calls"] == 0
    # The ``build_split_plan_diagnostics`` snapshot exposes no aggregate
    # initial-call total field (only ``details`` / ``details_total`` /
    # ``details_retained`` / ``details_omitted`` / ``details_digest``); the
    # summed retained-detail initial calls are the aggregate here, and they
    # are zero.
    assert not any("initial" in key for key in split_category)
    assert sum(d["initial_calls"] for d in split_category["details"]) == 0


def _r3_cross_plan_big_source():
    """A split source whose reduction tree has more than the single final node
    and whose topology shifts with the synthesis-manifest budget: a genuine
    predecessor tree built at ``current budget + 1,000`` keeps the stable leaf
    chunk IDs but ends up with a different reduction-tree digest and a partly
    different node-ID set from the current plan."""
    return "".join(
        f"def f{i}():\n    x = {'1 + ' * 8}1\n    return x\n\n" for i in range(200)
    )


def test_cross_plan_carry_pays_for_the_whole_current_replacement(tmp_path):
    """Proof control: a GENUINE, non-empty, fully completed predecessor
    ``SplitTreeState`` whose reduction-tree digest no longer matches the
    current plan -> Section 5 cross-plan fresh-preserve carry. The whole
    current split is payable, and the predecessor node IDs -- which genuinely
    overlap the current planned IDs on the stable leaf chunks -- are NOT
    subtracted from the payable count. Structurally mutation-sensitive: because
    the overlap is non-empty, an implementation that subtracted predecessor IDs
    would report a strictly smaller count and fail the assertions below."""
    import hashlib

    import codedoc.core.file_division as fd
    from codedoc.core.planning import build_pipeline_plan
    from tests.support.pipeline_usage import make_graph

    source = _r3_cross_plan_big_source()
    src = tmp_path / "main.py"
    src.write_text(source, encoding="utf-8", newline="")
    cfg = _split_cfg_r3()
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    division = fd.build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    current_synthesis = max(2000, fd.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS)
    tree = fd.build_reduction_tree(
        division, synthesis_manifest_chars=current_synthesis, language="python"
    )
    # A genuine predecessor reduction tree for the SAME division plan, built at
    # a deterministically different synthesis-manifest budget (current + 1,000).
    predecessor_tree = fd.build_reduction_tree(
        division,
        synthesis_manifest_chars=current_synthesis + 1000,
        language="python",
    )
    provider_identity = fd.provider_execution_identity(cfg)
    # A genuine, fully completed predecessor SplitTreeState from that tree:
    # every leaf / reducer / final node, with real recomputed stage-local
    # digests (Round-3 helper) -- not a fabricated digest of zeros.
    predecessor = _r3_fully_completed_tree_state(
        division,
        predecessor_tree,
        provider_identity=provider_identity,
        content_hash=content_hash,
    )

    current_planned_ids = {c.chunk_id for c in division.chunks} | {
        n.node_id for n in tree.all_nodes
    }
    predecessor_node_ids = {n.node_id for n in predecessor.nodes} | {
        e.node_id for e in predecessor.quarantine
    }
    overlap = predecessor_node_ids & current_planned_ids

    # --- the predecessor is genuine and non-empty, proven BEFORE planning ---
    assert predecessor.nodes  # non-empty node list
    assert len(predecessor.nodes) == len(predecessor_tree.all_nodes) + len(
        division.chunks
    )
    assert predecessor.reduction_tree_digest != tree.tree_digest
    assert predecessor.content_hash == content_hash
    assert predecessor.division_plan_digest == division.plan_digest
    assert predecessor_node_ids  # non-empty predecessor node-ID set
    assert overlap  # >=1 predecessor node ID is also a current planned ID
    # the stable leaf chunk IDs are exactly that overlap source
    assert {c.chunk_id for c in division.chunks} <= overlap
    # each tree genuinely has node IDs the other lacks
    assert current_planned_ids - predecessor_node_ids
    assert predecessor_node_ids - current_planned_ids

    graph = make_graph("main.py")
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py", "language": "python", "extension": ".py",
        }
    }
    plan, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, [], cfg,
        recovered_partials={"main.py": predecessor},
    )

    assert "main.py" in materials.carry_states
    assert materials.carry_states["main.py"] is predecessor
    assert "main.py" not in materials.tree_states
    entry = next(e for e in materials.route_plan if e.rel_path == "main.py")
    assert entry.route == "split"
    assert entry.split_payable_calls == len(current_planned_ids)
    assert entry.split_payable_calls == len(division.chunks) + len(tree.all_nodes)
    assert entry.payable is True
    assert "main.py" in plan.unpaid_action_rels

    # --- explicit non-subtraction ---
    # A predecessor-ID-subtracting implementation would report only the
    # non-overlapping remainder; the real count is the whole current plan.
    buggy_if_subtracted = len(current_planned_ids - predecessor_node_ids)
    assert buggy_if_subtracted < len(current_planned_ids)  # the overlap is real
    assert entry.split_payable_calls == len(current_planned_ids)
    assert entry.split_payable_calls > buggy_if_subtracted

    # --- the ephemeral recovery counters (section 6.3) ---
    # every unique predecessor node ID is discarded; every current planned node
    # ID is a replacement.
    assert materials.recovery_discarded_predecessor_nodes == len(predecessor_node_ids)
    assert materials.recovery_replacement_nodes_planned == len(current_planned_ids)


# ===========================================================================
# Section 6 Round 5 -- a category details_digest must cover the COMPLETE
# canonical descriptor stream, including descriptors omitted from bounded
# presentation (plan section 11, mutation check 24). These kill a prohibited
# implementation that hashes only the retained descriptors.
# ===========================================================================


def test_split_category_details_digest_covers_omitted_descriptors():
    """The ``split_plan`` ``details_digest`` published by the real
    ``build_split_plan_diagnostics`` path must equal a canonical stream digest
    over the *complete* flattened descriptor stream -- every file, unit, piece,
    leaf, and constituent -- independently reconstructed from the original full
    plans/trees, and must NOT equal the digest of only the retained subset."""
    from codedoc.core.file_division import iter_split_plan_stream_items

    # Four real split entries in canonical (path-ascending) order. The content
    # is identical, so the frozen rank tiebreak (normalized path) drops the
    # whole of "d.py" from bounded presentation under a tiny budget.
    entries = [
        _split_route_entry(rel, _three_natural_units(1243, 482, 285))
        for rel in ("a.py", "b.py", "c.py", "d.py")
    ]

    # Independent COMPLETE stream, built straight from the ORIGINAL full
    # plans/trees via the real descriptor/stream helpers -- never from the
    # returned digest or the truncated returned details.
    full_details = [
        split_plan_file_detail(
            rel,
            plan,
            tree,
            source_ceiling_chars=source_ceiling,
            synthesis_manifest_ceiling_chars=manifest_ceiling,
            initial_calls=initial_calls,
        )
        for (rel, plan, tree, source_ceiling, manifest_ceiling, initial_calls) in entries
    ]
    full_stream = list(iter_split_plan_stream_items(full_details))
    expected_complete_digest = canonical_stream_digest(full_stream)

    result = build_split_plan_diagnostics(iter(entries), item_budget=6)

    # c. omission genuinely occurred.
    assert result["details_omitted"] > 0
    assert result["details_total"] > result["details_retained"]
    assert result["details_total"] == len(full_stream)
    retained_stream = list(iter_split_plan_stream_items(result["details"]))
    assert len(retained_stream) < len(full_stream)  # at least one descriptor omitted
    dropped_paths = {item["path"] for item in full_stream} - {
        item["path"] for item in retained_stream
    }
    assert dropped_paths == {"d.py"}  # a whole file's substream is omitted

    # Independent retained-only stream/digest, from only what the category kept.
    retained_only_digest = canonical_stream_digest(retained_stream)

    # a. published digest == the independently reconstructed COMPLETE-stream digest
    assert result["details_digest"] == expected_complete_digest
    # b. published digest != the retained-only digest
    assert result["details_digest"] != retained_only_digest
    # the two reference digests genuinely differ (the omitted file changes it)
    assert expected_complete_digest != retained_only_digest


def test_flat_category_details_digest_covers_omitted_descriptors():
    """The flat-category (``split_blocked``) ``details_digest`` published by the
    real ``build_flat_plan_diagnostics`` path must equal a canonical stream
    digest over the *complete* canonical input stream, and must NOT equal the
    digest of only the retained records."""
    # Six blocked descriptors in canonical integrity order (constant reason ->
    # normalized path ascending); a tiny budget keeps only the first two.
    records = [
        blocked_split_descriptor(
            path=f"{letter}.py",
            reason="chunk-cap",
            phase="division-packing",
            observed=300 + index,
            limit=256,
            guidance_code="raise-source-ceiling-or-split-source",
        )
        for index, letter in enumerate(("a", "b", "c", "d", "e", "f"))
    ]

    # Independent digest of the COMPLETE canonical input stream -- from the
    # original record list, not the returned details.
    expected_complete_digest = canonical_stream_digest(records)

    result = build_flat_plan_diagnostics(
        iter(records),
        rank_key=lambda record: (
            BLOCKED_REASON_ORDER.index(record["reason"]),
            record["path"],
        ),
        item_budget=2,
    )

    # a/b. omission genuinely occurred.
    assert result["details_omitted"] > 0
    assert result["details_total"] > result["details_retained"]
    assert result["details_total"] == len(records)
    assert result["details_retained"] == 2
    # the omitted records (c.py..f.py) carry distinct values that shift the digest
    assert [record["path"] for record in result["details"]] == ["a.py", "b.py"]

    # Independent retained-only digest, from only the retained records.
    retained_only_digest = canonical_stream_digest(result["details"])
    assert expected_complete_digest != retained_only_digest

    # c. published digest == COMPLETE canonical input-stream digest
    assert result["details_digest"] == expected_complete_digest
    # d. published digest != retained-only digest
    assert result["details_digest"] != retained_only_digest


# ===========================================================================
# Sections 1-6 closure repair -- blocked-diagnostic boundedness & remediation
# (plan section 5.8): O(K + nesting_depth) diagnostic memory, the 20-record
# default presentation cap distinct from the 4096 snapshot cap, a
# complete-stream details_digest, and remediation derived only from the closed
# guidance-code vocabulary.
# ===========================================================================


def _blocked_detail(path, reason):
    """One value-safe capacity-block descriptor via the production exception."""
    return SplitCapacityBlocked(path, reason, observed=9_000, limit=1).detail


def _blocked_route_plan(details):
    """The already-canonical (normalized-path-ascending) route-plan view of
    blocked entries only -- exactly what the pipeline hands the ConfigError
    builder after the Sections 1-6 closure repair."""
    return tuple(
        file_division.RoutePlanEntry(
            rel_path=detail["path"],
            source_chars=10_000,
            route="blocked",
            payable=False,
            blocked_detail=detail,
        )
        for detail in sorted(details, key=lambda d: (d["path"], d["reason"]))
    )


def _blocked_message(details):
    """Adapter to ``pipeline._blocked_split_files_message``. The Sections 1-6
    closure repair switched its input from the ``division_blocked`` dict to the
    already-canonical route-plan view (``_blocked_route_plan``); only this
    adapter body changed -- every behavioural assertion below is identical
    before and after."""
    from codedoc.pipeline import _blocked_split_files_message

    return _blocked_split_files_message(_blocked_route_plan(details))


def _blocked_category(details):
    """Adapter to ``pipeline._blocked_split_category`` (see ``_blocked_message``)."""
    from codedoc.pipeline import _blocked_split_category

    return _blocked_split_category(_blocked_route_plan(details))


def test_blocked_config_error_message_defaults_to_twenty_records_not_four_thousand():
    """Defect D: the human ``ConfigError`` uses the 20-record default
    presentation cap (``PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS``), not the 4096
    snapshot cap. It renders at most 20 detail lines, reports the exact
    total/retained/omitted counts, its ``details_digest`` still covers every
    descriptor, and its size cannot grow with the omitted count."""
    n = 5_000
    details = [_blocked_detail(f"pkg/mod_{index:05d}.py", "chunk-cap") for index in range(n)]
    message = _blocked_message(details)

    rendered_lines = [
        line for line in message.split("\n") if line.startswith("  \"pkg/mod_")
    ]
    assert len(rendered_lines) == 20
    assert f"showing 20 of {n}" in message
    assert f"{n - 20} omitted" in message
    assert f"{n} file(s) cannot be completely split-planned" in message
    # the message stays small regardless of N
    assert len(message) < 8_000

    # details_digest covers ALL descriptors, not the shown 20
    category = _blocked_category(details)
    assert category["details_total"] == n
    assert category["details_retained"] == 20
    assert category["details_omitted"] == n - 20
    assert category["details_digest"] == canonical_stream_digest(
        [d for d in sorted(details, key=lambda d: (d["path"], d["reason"]))]
    )
    assert category["details_digest"] in message


def test_blocked_config_error_size_and_digest_track_omitted_descriptors():
    """Defect C+D: changing an omitted (unsampled) descriptor changes the
    complete-stream ``details_digest`` but not the bounded message size."""
    base = [_blocked_detail(f"pkg/f_{index:05d}.py", "chunk-cap") for index in range(4_200)]
    changed = list(base)
    # mutate one descriptor that is far past the retained 20
    changed[4_000] = _blocked_detail("pkg/f_04000.py", "atom-cap")

    msg_base = _blocked_message(base)
    msg_changed = _blocked_message(changed)
    cat_base = _blocked_category(base)
    cat_changed = _blocked_category(changed)

    assert cat_base["details_digest"] != cat_changed["details_digest"]
    assert cat_base["details_total"] == cat_changed["details_total"] == 4_200
    # message length is bounded and barely moves (only header/among-20 changes)
    assert abs(len(msg_base) - len(msg_changed)) < 200
    assert len(msg_base) < 8_000 and len(msg_changed) < 8_000


def test_blocked_diagnostic_memory_does_not_scale_with_omitted_n():
    """Defect C: the blocked ConfigError / category path must not materialize
    and sort an O(N) copy. With authoritative inputs built before tracemalloc
    starts, diagnostic-only peak memory does not scale with the blocked-file
    count (both far above 20 and 4096), and the rendered message stays bounded."""
    import gc

    def _probe(n):
        details = [
            _blocked_detail(f"pkg/x_{index:06d}.py", "chunk-cap") for index in range(n)
        ]
        expected_digest = canonical_stream_digest(
            [d for d in sorted(details, key=lambda d: (d["path"], d["reason"]))]
        )
        route_plan = _blocked_route_plan(details)
        from codedoc.pipeline import _blocked_split_files_message

        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        message = _blocked_split_files_message(route_plan)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert f"{n} file(s) cannot be completely split-planned" in message
        assert expected_digest in message
        return len(message), peak

    small_len, small_peak = _probe(500)
    large_len, large_peak = _probe(40_000)
    # 80x the blocked files -> diagnostic-only peak stays within 3x (no O(N)
    # sorted copy) and the rendered message length does not scale with N.
    assert large_peak < small_peak * 3, (small_peak, large_peak)
    assert large_len < 8_000
    assert abs(large_len - small_len) < 60


@pytest.mark.parametrize(
    "reason, guidance, required, forbidden",
    [
        (
            "atom-cap",
            "simplify-or-exclude",
            ["simplify-or-exclude", "simplify or refactor"],
            ["raise the source ceiling", "planning-capacity defect"],
        ),
        (
            "chunk-cap",
            "raise-source-ceiling-or-split-source",
            ["raise-source-ceiling-or-split-source", "raise the source ceiling"],
            ["planning-capacity defect", "inspect that metadata"],
        ),
        (
            "reduction-envelope-cap",
            "report-planning-capacity-defect",
            [
                "report-planning-capacity-defect",
                "internal planning-capacity defect",
                "Do not raise max_content_chars",
            ],
            ["raise the source ceiling if the provider supports"],
        ),
        (
            "reduction-depth-cap",
            "report-planning-capacity-defect",
            ["report-planning-capacity-defect", "Do not raise max_content_chars"],
            ["raise the source ceiling if the provider supports"],
        ),
        (
            "final-synthesis-envelope-cap",
            "inspect-authoritative-metadata-or-exclude",
            ["inspect-authoritative-metadata-or-exclude", "inspect that metadata"],
            ["raise the source ceiling if the provider supports", "planning-capacity defect"],
        ),
    ],
)
def test_blocked_message_remediation_is_derived_only_from_guidance_code(
    reason, guidance, required, forbidden
):
    """Defect E: the trailing remediation is derived ONLY from the closed
    ``guidance_code`` vocabulary. For a reduction planning-capacity failure the
    message must not advise raising ``max_content_chars``; every other case
    gets its own guidance and no contradictory advice."""
    details = [_blocked_detail(f"pkg/m_{index}.py", reason) for index in range(3)]
    message = _blocked_message(details)
    assert all(d["guidance_code"] == guidance for d in details)
    for phrase in required:
        assert phrase in message, (phrase, message)
    for phrase in forbidden:
        assert phrase not in message, (phrase, message)
    # the stale generic prose is gone
    assert "raising max_content_chars can help chunk or reduction capacity" not in message
    # path/reason presentation is still escaped and bounded
    assert message.count("\n") < 40


def test_blocked_message_remediation_covers_every_present_guidance_code_once():
    """Defect E: a mixed blocked run shows each present guidance code's
    remediation exactly once, in canonical order, and nothing for absent
    codes."""
    from codedoc.core.file_division import BLOCKED_GUIDANCE_VALUES
    from codedoc.pipeline import _GUIDANCE_REMEDIATION

    details = [
        _blocked_detail("pkg/a.py", "atom-cap"),              # simplify-or-exclude
        _blocked_detail("pkg/b.py", "chunk-cap"),             # raise-source-ceiling...
        _blocked_detail("pkg/c.py", "reduction-fan-in-cap"),  # report-planning-capacity...
    ]
    message = _blocked_message(details)
    present = {"simplify-or-exclude", "raise-source-ceiling-or-split-source",
              "report-planning-capacity-defect"}
    # each present code labels its remediation line ("<code>: ...") exactly once
    for code in present:
        assert message.count(code + ":") == 1, (code, message.count(code + ":"))
        assert _GUIDANCE_REMEDIATION[code] in message
    absent = set(BLOCKED_GUIDANCE_VALUES) - present
    for code in absent:
        assert (code + ":") not in message
        assert _GUIDANCE_REMEDIATION[code] not in message
    # canonical order: simplify < raise-source < report-planning
    assert (
        message.index("simplify-or-exclude:")
        < message.index("raise-source-ceiling-or-split-source:")
        < message.index("report-planning-capacity-defect:")
    )
