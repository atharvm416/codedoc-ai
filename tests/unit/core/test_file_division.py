from __future__ import annotations

from dataclasses import replace

import pytest

import codedoc.core.file_division as file_division
from codedoc.core.file_division import (
    BLOCKED_REASON_ORDER,
    DORMANT_SPLIT_PARTIAL_SCHEMA_VERSION,
    LEGACY_SPLIT_PARTIAL_SCHEMA_VERSION,
    MAX_CHUNKS_PER_FILE,
    MAX_KNOWN_SYMBOLS_PER_CHUNK,
    MAX_LEAF_CAPSULE_CANONICAL_CHARS,
    MAX_LEAF_SYMBOL_ITEMS_PER_KIND,
    MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
    MAX_QUARANTINE_ENTRIES_PER_FILE,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    DivisionInternalDefect,
    QuarantineEntry,
    SemanticUnitIdentity,
    SplitCapacityBlocked,
    SplitRecoveryStateError,
    SplitTreeState,
    build_division_plan,
    build_fact_ledger,
    build_reduction_tree,
    dependency_closed_nodes,
    distinct_units,
    final_execution_identity,
    final_node_covers_every_leaf,
    final_synthesis_input,
    is_legacy_split_partial,
    leaf_execution_identity,
    maximally_populated_fact_ledger,
    maximum_distinct_narratives,
    merge_leaf_capsules,
    provider_execution_identity,
    reduction_depth,
    reduction_execution_identity,
    refine_narrative_inputs,
    tree_node_state,
    validate_node_for_tree,
    validate_recovered_tree,
    verify_provider_execution_identity,
    worst_case_final_synthesis_chars,
    worst_case_reduction_manifest_chars,
)
from codedoc.parser.source_structure import MAX_STRUCTURE_SIGNATURE_CHARS, SourceRange
from codedoc.utils.errors import ConfigError
from tests.support.structure_extra import requires_structure_pack


def _large_source(lines: int = 220) -> str:
    return "\n".join(f"value_{index} = '{index}'" for index in range(lines)) + "\n"


# ---------------------------------------------------------------------------
# Division: complete coverage, determinism, whole-unit packing
# ---------------------------------------------------------------------------


def test_split_plan_is_deterministic_path_bound_and_covers_every_byte() -> None:
    source = _large_source()

    first = build_division_plan(
        rel_path="src/large.py", language="python", content=source, source_budget_chars=1000
    )
    second = build_division_plan(
        rel_path="src/large.py", language="python", content=source, source_budget_chars=1000
    )
    relocated = build_division_plan(
        rel_path="src/renamed.py", language="python", content=source, source_budget_chars=1000
    )

    assert first == second
    assert first.plan_digest == second.plan_digest
    assert first.plan_digest != relocated.plan_digest
    assert len(first.chunks) >= 2
    assert b"".join(atom.source.encode("utf-8") for atom in first.atoms) == source.encode("utf-8")
    assert b"".join(chunk.payload.encode("utf-8") for chunk in first.chunks) == source.encode(
        "utf-8"
    )
    assert all(chunk.payload_chars <= first.source_budget_chars for chunk in first.chunks)
    assert len({unit.unit_id for unit in first.units}) == len(first.units)
    assert len({chunk.chunk_id for chunk in first.chunks}) == len(first.chunks)
    assert all(len(unit.unit_id) == len("unit_") + 64 for unit in first.units)
    assert all(len(chunk.chunk_id) == len("chunk_") + 64 for chunk in first.chunks)


def test_crlf_unicode_plan_is_gap_free_and_repeatably_identical() -> None:
    source = "".join(f"cafe_{index} = '{index}'\r\n" for index in range(150))
    plans = [
        build_division_plan(
            rel_path="src/large.py", language="unknown", content=source, source_budget_chars=1200
        )
        for _ in range(3)
    ]
    plan = plans[0]

    assert all(other == plan for other in plans[1:])
    assert len({other.plan_digest for other in plans}) == 1
    assert len(plan.chunks) >= 2

    data = source.encode("utf-8")
    offset = 0
    for chunk in plan.chunks:
        for source_range in chunk.owning_ranges:
            assert source_range.start_byte == offset
            offset = source_range.end_byte
    assert offset == len(data)
    assert b"".join(chunk.payload.encode("utf-8") for chunk in plan.chunks) == data


def test_small_adjacent_units_pack_into_one_chunk() -> None:
    source = "a = 1\nb = 2\nc = 3\n"
    plan = build_division_plan(
        rel_path="small.py", language="unknown", content=source, source_budget_chars=1000
    )
    assert len(plan.chunks) == 1
    assert len(plan.units) == 3
    assert plan.chunks[0].semantic_units == plan.units
    assert plan.chunks[0].payload == source


def test_leaf_prompt_metadata_cap_preserves_maximum_lexical_plan_linearly(
    monkeypatch,
) -> None:
    source = "x;\n" * file_division.MAX_UNITS_PER_FILE
    render_calls = 0
    original_renderer = file_division.render_leaf_prompt_metadata

    def counted_renderer(**kwargs):
        nonlocal render_calls
        render_calls += 1
        return original_renderer(**kwargs)

    monkeypatch.setattr(
        file_division,
        "render_leaf_prompt_metadata",
        counted_renderer,
    )
    plan = build_division_plan(
        rel_path="maximum.txt",
        language="unknown",
        content=source,
        source_budget_chars=12000,
    )

    assert plan.structural_mode == "lexical"
    assert len(plan.units) == file_division.MAX_UNITS_PER_FILE
    assert len(plan.chunks) <= file_division.MAX_CHUNKS_PER_FILE
    assert "".join(chunk.payload for chunk in plan.chunks) == source
    assert render_calls == len(plan.chunks)
    for chunk in plan.chunks:
        metadata = original_renderer(
            group_unit_id=chunk.unit_id,
            semantic_units=chunk.semantic_units,
            unit_indexes=plan.unit_positions(chunk),
            unit_count=len(plan.units),
            owning_ranges=chunk.owning_ranges,
        )
        assert len(metadata) <= file_division.MAX_LEAF_PROMPT_METADATA_CHARS
        assert len(metadata) == file_division.leaf_prompt_metadata_chars(
            group_unit_id=chunk.unit_id,
            semantic_units=chunk.semantic_units,
            unit_indexes=plan.unit_positions(chunk),
            unit_count=len(plan.units),
            owning_ranges=chunk.owning_ranges,
        )


def test_one_maximal_semantic_unit_metadata_fits_the_fixed_bound() -> None:
    plan = build_division_plan(
        rel_path="one.txt",
        language="unknown",
        content="x\n",
        source_budget_chars=1000,
    )
    chunk = plan.chunks[0]
    unit = replace(
        chunk.semantic_units[0],
        kind="k" * 160,
        qualified_name="n" * 240,
        signature="s" * 600,
    )
    metadata = file_division.render_leaf_prompt_metadata(
        group_unit_id=unit.unit_id,
        semantic_units=(unit,),
        unit_indexes=(0,),
        unit_count=1,
        owning_ranges=chunk.owning_ranges,
    )

    assert len(metadata) <= file_division.MAX_LEAF_PROMPT_METADATA_CHARS
    assert len(metadata) == file_division.leaf_prompt_metadata_chars(
        group_unit_id=unit.unit_id,
        semantic_units=(unit,),
        unit_indexes=(0,),
        unit_count=1,
        owning_ranges=chunk.owning_ranges,
    )


def test_only_oversized_semantic_unit_is_continued() -> None:
    normal = _large_source(120)
    normal_plan = build_division_plan(
        rel_path="normal.py", language="unknown", content=normal, source_budget_chars=1000
    )
    normal_data = normal.encode("utf-8")
    for chunk in normal_plan.chunks:
        for source_range in chunk.owning_ranges:
            owned = normal_data[source_range.start_byte : source_range.end_byte].decode("utf-8")
            assert owned.endswith("\n")

    oversized = "x" * 2101
    plan = build_division_plan(
        rel_path="oversized.py", language="unknown", content=oversized, source_budget_chars=1000
    )
    assert [chunk.payload_chars for chunk in plan.chunks] == [1000, 1000, 101]
    assert "".join(chunk.payload for chunk in plan.chunks) == oversized
    assert len(plan.units) == 1
    assert plan.chunks[0].continuation_before is False
    assert plan.chunks[0].continuation_after is True
    assert plan.chunks[1].continuation_before is True
    assert plan.chunks[1].continuation_after is True
    assert plan.chunks[2].continuation_before is True
    assert plan.chunks[2].continuation_after is False
    assert all(c.unit_chunk_count == 3 for c in plan.chunks)
    assert [c.unit_chunk_index for c in plan.chunks] == [0, 1, 2]


@requires_structure_pack
def test_multiline_oversized_unit_packs_complete_lines_near_budget() -> None:
    body = "".join(
        f"    value_{index:03d} = normalize({index})\n" for index in range(180)
    )
    source = "def calculate():\n" + body + "    return value_179\n"
    budget = 500

    plan = build_division_plan(
        rel_path="service.py",
        language="python",
        content=source,
        source_budget_chars=budget,
    )

    assert 1 <= len(plan.units) <= 2
    assert len(plan.chunks) <= (len(source) + budget - 1) // budget + 1
    assert "".join(chunk.payload for chunk in plan.chunks) == source
    assert all(chunk.payload_chars <= budget for chunk in plan.chunks)
    longest_line = max(len(line) for line in source.splitlines(keepends=True))
    assert all(
        chunk.payload_chars > budget - longest_line
        for chunk in plan.chunks[:-1]
    )
    assert all(chunk.payload.endswith("\n") for chunk in plan.chunks)


@requires_structure_pack
def test_decorated_python_definition_keeps_one_unit_across_budget_edge() -> None:
    source = (
        "@trace\r\n"
        "@tag('\u96ea')\r\n"
        "async def caf\u00e9():\r\n"
        "    return '\u96ea'\r\n"
    )

    fitting = build_division_plan(
        rel_path="decorated.py",
        language="python",
        content=source,
        source_budget_chars=len(source),
    )
    continued = build_division_plan(
        rel_path="decorated.py",
        language="python",
        content=source,
        source_budget_chars=len(source) - 1,
    )

    assert len(fitting.units) == 1
    assert len(fitting.chunks) == 1
    assert fitting.chunks[0].payload == source
    assert len(continued.units) == 1
    assert len(continued.chunks) > 1
    assert "".join(chunk.payload for chunk in continued.chunks) == source
    assert all(chunk.semantic_units == continued.units for chunk in continued.chunks)
    assert all(
        chunk.group_unit_id == continued.units[0].unit_id for chunk in continued.chunks
    )
    assert all(
        chunk.unit_chunk_count == len(continued.chunks) for chunk in continued.chunks
    )


@requires_structure_pack
def test_realistic_large_class_with_many_methods_is_plannable() -> None:
    methods = "".join(
        (
            f"    def operation_{index:03d}(self, value: int) -> int:\n"
            f"        normalized = value + {index}\n"
            "        return normalized\n\n"
        )
        for index in range(140)
    )
    source = (
        "class ApplicationService:\n"
        '    """Coordinates a representative application workflow."""\n\n'
        + methods
    )
    budget = 12000

    plan = build_division_plan(
        rel_path="service.py",
        language="python",
        content=source,
        source_budget_chars=budget,
    )

    assert len(source) > budget
    assert 1 <= len(plan.units) <= 2
    assert 2 <= len(plan.chunks) <= (len(source) + budget - 1) // budget + 1
    assert "".join(chunk.payload for chunk in plan.chunks) == source
    assert all(
        len(chunk.known_symbols) <= file_division.MAX_KNOWN_SYMBOLS_PER_CHUNK
        for chunk in plan.chunks
    )


def test_multibyte_utf8_content_is_covered_exactly() -> None:
    source = ("value_%d = 'x雪%d'\n" % (i, i) for i in range(80))
    content = "".join(source)
    plan = build_division_plan(
        rel_path="unicode.py", language="unknown", content=content, source_budget_chars=200
    )
    assert b"".join(chunk.payload.encode("utf-8") for chunk in plan.chunks) == content.encode(
        "utf-8"
    )
    assert len(plan.chunks) >= 2


def test_malformed_or_missing_grammar_uses_lexical_fallback() -> None:
    # No bundled grammar for language "unknown" -> lexical fallback, never a defect.
    plan = build_division_plan(
        rel_path="mystery.txt", language="unknown", content=_large_source(60), source_budget_chars=500
    )
    assert plan.structural_mode == "lexical"
    assert plan.symbols == ()


# ---------------------------------------------------------------------------
# Capacity blocks: every named reason, frozen precedence, no truncate route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "patch_name", "patch_value"),
    [
        ("atom-cap", "MAX_ATOMS_PER_FILE", 0),
        ("unit-cap", "MAX_UNITS_PER_FILE", 0),
        ("chunk-cap", "MAX_CHUNKS_PER_FILE", 0),
    ],
)
def test_structure_count_capacity_blocks_are_named(monkeypatch, reason, patch_name, patch_value) -> None:
    monkeypatch.setattr(file_division, patch_name, patch_value)
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_division_plan(
            rel_path="src/large.py", language="unknown", content="a\nb\n", source_budget_chars=1000
        )
    assert excinfo.value.reason == reason
    assert excinfo.value.rel_path == "src/large.py"


def _synthetic_symbol_structure(monkeypatch, symbol_count: int) -> None:
    """Monkeypatch extract_structure to return atoms carrying *symbol_count*
    distinct symbols, without depending on the optional tree-sitter package."""
    from codedoc.core.file_division import SourceIndex
    from codedoc.parser.source_structure import (
        Atom,
        StructureResult,
        SymbolFact,
        atom_id_for,
        symbol_id_for,
    )

    def _fake_extract(rel_path, language, content, **_kwargs):
        range_ = SourceIndex(content).range(0, len(content.encode("utf-8")))
        atom_id = atom_id_for(rel_path, "function_definition", range_.start_byte, range_.end_byte)
        symbol_ids = tuple(
            symbol_id_for(rel_path, "function_definition", f"alpha_{i}", range_.start_byte, range_.end_byte)
            for i in range(symbol_count)
        )
        atom = Atom(
            atom_id=atom_id,
            rel_path=rel_path,
            language=language,
            kind="function_definition",
            name="alpha",
            range=range_,
            source=content,
            symbol_ids=symbol_ids,
        )
        symbols = tuple(
            SymbolFact(
                symbol_id=symbol_ids[i],
                rel_path=rel_path,
                language=language,
                kind="function_definition",
                qualified_name=f"alpha_{i}",
                signature=f"def alpha_{i}()",
                range=range_,
                atom_id=atom.atom_id,
            )
            for i in range(symbol_count)
        )
        return StructureResult("syntax", (atom,), symbols, ())

    monkeypatch.setattr(file_division, "extract_structure", _fake_extract)


def test_symbol_cap_is_named(monkeypatch) -> None:
    _synthetic_symbol_structure(monkeypatch, symbol_count=1)
    monkeypatch.setattr(file_division, "MAX_SYMBOLS_PER_FILE", 0)
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_division_plan(
            rel_path="src/sym.py", language="python", content="def alpha(): pass\n", source_budget_chars=1000
        )
    assert excinfo.value.reason == "symbol-cap"


@requires_structure_pack
def test_real_declaration_overflow_reports_symbol_cap() -> None:
    source = "class Many:\n" + "".join(
        f"    def function_{index}(self): return {index}\n"
        for index in range(file_division.MAX_SYMBOLS_PER_FILE + 32)
    )

    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_division_plan(
            rel_path="src/many.py",
            language="python",
            content=source,
            source_budget_chars=1000,
        )

    assert excinfo.value.reason == "symbol-cap"


def test_parser_symbol_count_does_not_split_or_block_a_fitting_unit(
    monkeypatch,
) -> None:
    _synthetic_symbol_structure(monkeypatch, symbol_count=40)
    plan = build_division_plan(
        rel_path="src/heavy.py",
        language="python",
        content="def alpha(): pass\n",
        source_budget_chars=1000,
    )

    assert len(plan.chunks) == 1
    assert len(plan.chunks[0].known_symbols) == (
        file_division.MAX_KNOWN_SYMBOLS_PER_CHUNK
    )


def test_dual_cap_violation_reports_only_earlier_reason(monkeypatch) -> None:
    monkeypatch.setattr(file_division, "MAX_ATOMS_PER_FILE", 0)
    monkeypatch.setattr(file_division, "MAX_UNITS_PER_FILE", 0)
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_division_plan(
            rel_path="src/dual.py", language="unknown", content="a\nb\n", source_budget_chars=1000
        )
    assert excinfo.value.reason == "atom-cap"

    monkeypatch.setattr(file_division, "MAX_UNITS_PER_FILE", 0)
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 0)
    monkeypatch.setattr(file_division, "MAX_ATOMS_PER_FILE", 4096)
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_division_plan(
            rel_path="src/dual2.py", language="unknown", content="a\nb\n", source_budget_chars=1000
        )
    assert excinfo.value.reason == "unit-cap"


def test_reduction_envelope_and_fan_in_and_final_caps_are_named() -> None:
    plan = build_division_plan(
        rel_path="src/large.py", language="unknown", content=_large_source(80), source_budget_chars=200
    )
    with pytest.raises(SplitCapacityBlocked) as envelope_exc:
        build_reduction_tree(plan, max_content_chars=file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS)
    assert envelope_exc.value.reason == "reduction-envelope-cap"

    # Room for overhead but not two worst-case children.
    tight = (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(1)
    )
    with pytest.raises(SplitCapacityBlocked) as fan_in_exc:
        build_reduction_tree(plan, max_content_chars=tight)
    assert fan_in_exc.value.reason == "reduction-fan-in-cap"

    # Reducer fan-in is valid, but the real deterministic imports make even a
    # one-root final manifest impossible.
    final_only = 12000
    oversized_imports = tuple(
        f"dependency_{index:03d}_" + ("x" * 80)
        for index in range(180)
    )
    with pytest.raises(SplitCapacityBlocked) as final_exc:
        build_reduction_tree(
            plan,
            max_content_chars=final_only,
            language="python",
            imports=oversized_imports,
        )
    assert final_exc.value.reason == "final-synthesis-envelope-cap"


def test_final_capacity_accounts_for_json_escaped_narratives() -> None:
    from codedoc.agents.response_cleaning import clean_leaf_capsule_report

    plan = build_division_plan(
        rel_path="x.py",
        language="python",
        content="x = 1\n",
        source_budget_chars=100,
    )
    narrative = "\\" * file_division.MAX_REDUCTION_NARRATIVE_CHARS
    capsule = clean_leaf_capsule_report(
        {"description": narrative},
        plan.rel_path,
    ).value
    assert capsule["description"] == narrative
    live_manifest = final_synthesis_input(
        rel_path=plan.rel_path,
        language="python",
        imports=(),
        root_narratives=(capsule["description"],),
        root_coverage_leaf_ids=(plan.chunks[0].chunk_id,),
        ledger=merge_leaf_capsules(
            (capsule,),
            language="python",
        ),
        max_chars=685,
    )

    assert len(live_manifest) > 685
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_reduction_tree(
            plan,
            max_content_chars=685,
            language="python",
        )
    assert excinfo.value.reason == "final-synthesis-envelope-cap"


def test_capsule_canonical_bounds_use_maximum_json_escape_width() -> None:
    from codedoc.agents.response_cleaning import (
        clean_leaf_capsule_report,
        clean_reduction_capsule_report,
    )

    maximum_char = "\x00"
    maximum_leaf = {
        "description": maximum_char * file_division.MAX_LEAF_DESCRIPTION_CHARS,
        "functions": [
            {
                "name": maximum_char * file_division.MAX_LEAF_SYMBOL_NAME_CHARS,
                "description": (
                    maximum_char
                    * file_division.MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
                ),
                "signature": (
                    maximum_char
                    * file_division.MAX_LEAF_SYMBOL_SIGNATURE_CHARS
                ),
            }
            for _ in range(file_division.MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
        ],
        "classes": [
            {
                "name": maximum_char * file_division.MAX_LEAF_SYMBOL_NAME_CHARS,
                "description": (
                    maximum_char
                    * file_division.MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
                ),
                "signature": (
                    maximum_char
                    * file_division.MAX_LEAF_SYMBOL_SIGNATURE_CHARS
                ),
            }
            for _ in range(file_division.MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
        ],
        "exports": [
            maximum_char * file_division.MAX_LEAF_EXPORT_ITEM_CHARS
            for _ in range(file_division.MAX_LEAF_EXPORT_ITEMS)
        ],
    }
    maximum_reduction = {
        "narrative": maximum_char * file_division.MAX_REDUCTION_NARRATIVE_CHARS
    }

    assert (
        len(file_division.canonical_json(maximum_leaf))
        == file_division.MAX_LEAF_CAPSULE_CANONICAL_CHARS
    )
    assert (
        len(file_division.canonical_json(maximum_reduction))
        == file_division.MAX_REDUCTION_CAPSULE_CANONICAL_CHARS
    )
    assert clean_leaf_capsule_report(
        {
            "description": maximum_leaf["description"],
        },
        "x.py",
    ).value["description"] == maximum_leaf["description"]
    assert clean_reduction_capsule_report(
        maximum_reduction,
        "x.py",
    ).value == maximum_reduction


def test_reduction_depth_cap_is_named(monkeypatch) -> None:
    monkeypatch.setattr(file_division, "MAX_REDUCTION_TREE_DEPTH", 1)
    budget = 50
    line = "x = " + ("1" * (budget * 60)) + "\n"
    plan = build_division_plan(
        rel_path="huge.py", language="python", content=line, source_budget_chars=budget
    )
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_reduction_tree(plan, max_content_chars=12000)
    assert excinfo.value.reason == "reduction-depth-cap"


def test_reduction_depth_cap_applies_across_unit_and_general_phases(monkeypatch) -> None:
    monkeypatch.setattr(file_division, "MAX_REDUCTION_TREE_DEPTH", 2)
    source = ("x" * 3999 + "\n") + "".join("y" * 599 + "\n" for _ in range(4))
    plan = build_division_plan(
        rel_path="mixed.txt",
        language="unknown",
        content=source,
        source_budget_chars=1000,
    )

    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_reduction_tree(plan, max_content_chars=1000)

    assert excinfo.value.reason == "reduction-depth-cap"


def test_reduction_depth_cap_wins_over_final_synthesis_envelope_cap(monkeypatch) -> None:
    """A plan/ceiling combination that violates both reduction-depth-cap and
    final-synthesis-envelope-cap simultaneously must report the earlier reason
    in the frozen order (reduction-depth-cap), never the later one — the
    final-envelope check must never preempt tree-building for a file whose own
    structure would also exceed the depth cap."""
    monkeypatch.setattr(file_division, "MAX_REDUCTION_TREE_DEPTH", 1)
    budget = 50
    # One oversized unit split into 11 continuation chunks — needs more than
    # one reduction level to consolidate at fan_in=2.
    line = "x = " + ("1" * (budget * 10)) + "\n"
    plan = build_division_plan(
        rel_path="huge.py", language="python", content=line, source_budget_chars=budget
    )
    assert len(plan.chunks) > 2

    max_content_chars = (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(2)
    )
    # Confirm reducer fan-in is valid. Oversized real imports independently
    # make the final manifest impossible, so only depth vs. final-envelope
    # precedence is under test.
    assert (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(2)
        <= max_content_chars
    )
    oversized_imports = ("i" * (max_content_chars * 2),)

    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_reduction_tree(
            plan,
            max_content_chars=max_content_chars,
            imports=oversized_imports,
        )
    assert excinfo.value.reason == "reduction-depth-cap"


def test_final_synthesis_envelope_cap_fires_for_a_trivial_single_chunk_file() -> None:
    """A file with nothing to reduce (already a single root) is never
    misreported as a depth violation when the final envelope alone is too
    tight — the correct, meaningful reason still fires."""
    plan = build_division_plan(
        rel_path="tiny.py", language="python", content="x = 1\n", source_budget_chars=1000
    )
    assert len(plan.chunks) == 1

    # Passes reduction-envelope-cap and reduction-fan-in-cap (fan_in == 2)
    # but leaves final_fan_in < 1 — the same window used above, just applied
    # to a plan with nothing to reduce.
    max_content_chars = (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(2)
    )
    with pytest.raises(SplitCapacityBlocked) as excinfo:
        build_reduction_tree(
            plan,
            max_content_chars=max_content_chars,
            imports=("i" * (max_content_chars * 2),),
        )
    assert excinfo.value.reason == "final-synthesis-envelope-cap"


def test_blocked_reason_order_matches_plan_contract() -> None:
    assert BLOCKED_REASON_ORDER == (
        "atom-cap",
        "symbol-cap",
        "unit-cap",
        "chunk-cap",
        "reduction-envelope-cap",
        "reduction-fan-in-cap",
        "reduction-depth-cap",
        "final-synthesis-envelope-cap",
    )


def test_division_internal_defect_is_not_a_capacity_block() -> None:
    assert not issubclass(DivisionInternalDefect, SplitCapacityBlocked)
    assert not issubclass(SplitCapacityBlocked, DivisionInternalDefect)


# ---------------------------------------------------------------------------
# Hierarchical reduction tree
# ---------------------------------------------------------------------------


def test_zero_intermediate_levels_when_leaves_fit_final_envelope() -> None:
    plan = build_division_plan(
        rel_path="small.py", language="unknown", content="a = 1\nb = 2\n", source_budget_chars=1000
    )
    tree = build_reduction_tree(plan, max_content_chars=12000)
    assert tree.unit_consolidation_nodes == ()
    assert tree.general_nodes == ()
    assert tree.final_node.child_ids == tuple(chunk.chunk_id for chunk in plan.chunks)


@requires_structure_pack
def test_170_continuation_leaf_fixture_exercises_hierarchy() -> None:
    budget = 50
    prefix = "x = "
    suffix = "\n"
    line = prefix + ("1" * (budget * 170 - len(prefix) - len(suffix))) + suffix
    assert len(line) == budget * 170
    plan = build_division_plan(
        rel_path="huge.py", language="python", content=line, source_budget_chars=budget
    )
    assert len(plan.units) == 1, "one oversized semantic unit, never 170 units"
    assert len(plan.chunks) == 170
    assert plan.structural_mode == "syntax"
    assert plan.plan_digest == (
        "division-plan:152d94f7f14900632e9cd9b150ddb3ba"
        "fc7a807474273eca8ed74d13c2e363f3"
    )

    tree = build_reduction_tree(plan, max_content_chars=12000)
    expected_node_ids = (
        "node_9de96a1d480637ee05da30704bbd1ac3f228bab2ae4ec603b8105ebe8b5a90a2",
        "node_645058283250c3104d0dd57596e60d08743e8dfe9967acce38611e801ff9cd93",
        "node_24ed9d4019bc9c0d571dcbc6e0e10b771aa082504c9d40df44567144cf6a8e8e",
        "node_45d7c8076d20037474584e721fd2a03123239abc6e55fe01acbf4ce04fc025dd",
        "node_c63c6ed58df4e4e9b4545793eea5e3c564f1c5c37ac0dfef3a7d884732cfee52",
        "node_c4ffeb7696b8324b2c89b175d778f991fc15683359fbdca57c358cac00353295",
        "node_c51587769bff4f18c5d301b298488b8287e07c48652f4495c53b1dde2a7da279",
    )
    assert tree.max_fan_in == 39
    assert len(tree.unit_consolidation_nodes) == 6
    assert len(tree.general_nodes) == 0
    assert [node.level for node in tree.unit_consolidation_nodes] == [1, 1, 1, 1, 1, 2]
    assert tuple(node.node_id for node in tree.all_nodes) == expected_node_ids
    assert tree.final_node.child_ids == (expected_node_ids[-2],)
    assert tree.tree_digest == (
        "reduction-tree:0045c7ac40c5bce2b86117bbdc0714b9"
        "883afab125b63d33a40695536801b5ca"
    )
    expected_leaf_ids = tuple(chunk.chunk_id for chunk in plan.chunks)
    assert tree.final_node.leaf_ids == expected_leaf_ids
    assert len(set(tree.final_node.leaf_ids)) == 170
    for node in tree.all_intermediate_nodes:
        assert len(node.child_ids) >= 2, "no unary reducer may exist"


def test_same_unit_continuations_consolidate_before_cross_unit_grouping() -> None:
    # Named function declarations, not bare top-level statements: under
    # syntax-mode parsing, adjacent bare statements with no intervening
    # declaration can merge into one "gap" unit, which would collapse this
    # fixture's intended 3-unit structure. A `def` is reliably its own
    # semantic unit under both syntax and lexical-fallback parsing.
    budget = 50
    big_body = "9" * (budget * 6)
    content = (
        "def a(): return 1\n"
        f'def big(): return "{big_body}"\n'
        "def b(): return 2\n"
    )
    plan = build_division_plan(
        rel_path="mix.py", language="python", content=content, source_budget_chars=budget
    )
    assert len(plan.units) == 3
    tree = build_reduction_tree(plan, max_content_chars=12000)
    # The oversized unit's own chunks are consolidated into exactly one
    # representative before appearing (indirectly) among the final children.
    big_unit_id = next(unit.unit_id for unit in plan.units if unit.atom_ids and len(
        [c for c in plan.chunks if c.unit_id == unit.unit_id]
    ) > 1)
    big_chunk_ids = {c.chunk_id for c in plan.chunks if c.unit_id == big_unit_id}
    assert big_chunk_ids.isdisjoint(set(tree.final_node.child_ids))
    consolidation_root = [
        node for node in tree.unit_consolidation_nodes if node.unit_id == big_unit_id
    ]
    assert consolidation_root, "the oversized unit must have unit-consolidation nodes"


def test_singleton_remainder_is_promoted_without_a_unary_reducer(monkeypatch) -> None:
    budget = 50
    line = "x" * (budget * 7 - 1) + "\n"
    assert len(line) == budget * 7
    plan = build_division_plan(
        rel_path="seven.py", language="python", content=line, source_budget_chars=budget
    )
    assert len(plan.chunks) == 7
    max_content_chars = 2000
    tree = build_reduction_tree(plan, max_content_chars=max_content_chars)
    assert tree.max_fan_in == 6
    level_one = [n for n in tree.unit_consolidation_nodes if n.level == 1]
    assert len(level_one) == 1
    assert len(level_one[0].child_ids) == 6
    level_two = [n for n in tree.unit_consolidation_nodes if n.level == 2]
    assert len(level_two) == 1
    assert plan.chunks[6].chunk_id in level_two[0].child_ids


def test_reducer_fan_in_uses_exact_rendered_manifest_boundary() -> None:
    plan = build_division_plan(
        rel_path="five.py",
        language="unknown",
        content="".join(f"value_{index} = {index}\n" for index in range(20)),
        source_budget_chars=50,
    )
    max_content_chars = (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(99)
    )

    tree = build_reduction_tree(
        plan,
        max_content_chars=max_content_chars,
    )

    assert tree.max_fan_in == 99
    assert (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(99)
        <= max_content_chars
    )
    assert (
        file_division.REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(100)
        > max_content_chars
    )


def test_equal_short_names_in_different_scopes_never_consolidate(monkeypatch) -> None:
    from codedoc.parser.source_structure import (
        Atom,
        StructureResult,
        SymbolFact,
        atom_id_for,
        symbol_id_for,
    )

    def _fake_extract(rel_path, language, content, **_kwargs):
        half = len(content) // 2
        from codedoc.core.file_division import SourceIndex

        index = SourceIndex(content)
        range_a = index.range(0, half)
        range_b = index.range(half, len(content.encode("utf-8")))
        atom_id_a = atom_id_for(rel_path, "method", range_a.start_byte, range_a.end_byte)
        atom_id_b = atom_id_for(rel_path, "method", range_b.start_byte, range_b.end_byte)
        symbol_id_a = symbol_id_for(rel_path, "method", "run", range_a.start_byte, range_a.end_byte)
        symbol_id_b = symbol_id_for(rel_path, "method", "run", range_b.start_byte, range_b.end_byte)
        atom_a = Atom(
            atom_id=atom_id_a, rel_path=rel_path, language=language,
            kind="method", name="run", range=range_a, source=content[:half],
            symbol_ids=(symbol_id_a,),
        )
        atom_b = Atom(
            atom_id=atom_id_b, rel_path=rel_path, language=language,
            kind="method", name="run", range=range_b, source=content[half:],
            symbol_ids=(symbol_id_b,),
        )
        symbol_a = SymbolFact(
            symbol_id=symbol_id_a, rel_path=rel_path, language=language,
            kind="method", qualified_name="run", signature="def run()",
            range=range_a, atom_id=atom_a.atom_id,
        )
        symbol_b = SymbolFact(
            symbol_id=symbol_id_b, rel_path=rel_path, language=language,
            kind="method", qualified_name="run", signature="def run()",
            range=range_b, atom_id=atom_b.atom_id,
        )
        return StructureResult("syntax", (atom_a, atom_b), (symbol_a, symbol_b), ())

    monkeypatch.setattr(file_division, "extract_structure", _fake_extract)
    # A tight budget keeps the two small atoms in separate chunks (each below
    # the budget alone, but too large combined), so this actually exercises
    # two distinct units rather than one packed group.
    plan = build_division_plan(
        rel_path="classes.py", language="python", content="run_a();run_b();", source_budget_chars=10
    )
    units = plan.units
    assert len(units) == 2
    assert units[0].unit_id != units[1].unit_id
    assert units[0].qualified_name == units[1].qualified_name == "run"


def test_worst_case_envelopes_fit_default_max_content_chars() -> None:
    plan = build_division_plan(
        rel_path="big.py", language="unknown", content=_large_source(400), source_budget_chars=800
    )
    tree = build_reduction_tree(plan, max_content_chars=12000)
    assert tree.max_fan_in >= 2
    assert tuple(sorted(tree.final_node.leaf_ids)) == tuple(sorted(c.chunk_id for c in plan.chunks))


def _plan_and_tree_with_exact_chunk_count(
    chunk_count: int, *, max_content_chars: int, budget: int = 40
):
    """One oversized single-unit continuation fixture with exactly
    *chunk_count* leaf chunks, searching for the shortest content that
    produces it (a single long line divides at line/codepoint-safe budget
    boundaries, so length and chunk count are not related by one fixed
    formula near small counts)."""
    for total_len in range(budget * (chunk_count - 1) + 1, budget * chunk_count + 1):
        line = "x = " + ("1" * total_len) + "\n"
        plan = build_division_plan(
            rel_path="d.py", language="python", content=line, source_budget_chars=budget
        )
        if len(plan.chunks) == chunk_count:
            return plan, build_reduction_tree(plan, max_content_chars=max_content_chars)
    raise AssertionError(f"no fixture found for exactly {chunk_count} chunks")


def test_split_complexity_advisory_fires_at_exact_chunk_count_boundary() -> None:
    """D6a: the advisory is a plain strictly-greater-than comparison against
    SPLIT_COMPLEXITY_ADVISORY_CHUNKS, tested at threshold-1/threshold/
    threshold+1 (per the plan's explicit boundary requirement), with
    reduction depth held safely below its own threshold so only the
    chunk-count condition is under test."""
    from codedoc.pipeline import _split_complexity_advisory

    threshold = file_division.SPLIT_COMPLEXITY_ADVISORY_CHUNKS
    for chunk_count, should_fire in (
        (threshold - 1, False),
        (threshold, False),
        (threshold + 1, True),
    ):
        plan, tree = _plan_and_tree_with_exact_chunk_count(
            chunk_count, max_content_chars=12000
        )
        assert len(plan.chunks) == chunk_count
        assert reduction_depth(tree) <= file_division.SPLIT_COMPLEXITY_ADVISORY_REDUCTION_DEPTH
        advisory = _split_complexity_advisory({"d.py": plan}, {"d.py": tree})
        assert (advisory is not None) is should_fire, (chunk_count, advisory)


def test_split_complexity_advisory_fires_at_exact_reduction_depth_boundary(
    monkeypatch,
) -> None:
    """D6a: the advisory is a plain strictly-greater-than comparison against
    SPLIT_COMPLEXITY_ADVISORY_REDUCTION_DEPTH, tested at threshold-1/
    threshold/threshold+1, with chunk count held safely below its own
    threshold so only the depth condition is under test. Chunk counts (6,
    36, 37) are empirically chosen so that, at fan_in == 6 (max_content_chars
    == 2000 under these constants), reduction_depth lands exactly on
    threshold-1/threshold/threshold+1."""
    from codedoc.pipeline import _split_complexity_advisory

    monkeypatch.setattr(
        file_division,
        "SPLIT_COMPLEXITY_ADVISORY_CHUNKS",
        100,
    )
    max_content_chars = 2000
    threshold = file_division.SPLIT_COMPLEXITY_ADVISORY_REDUCTION_DEPTH
    for chunk_count, expected_depth, should_fire in (
        (6, threshold - 1, False),
        (36, threshold, False),
        (37, threshold + 1, True),
    ):
        plan, tree = _plan_and_tree_with_exact_chunk_count(
            chunk_count, max_content_chars=max_content_chars
        )
        assert len(plan.chunks) <= file_division.SPLIT_COMPLEXITY_ADVISORY_CHUNKS
        assert reduction_depth(tree) == expected_depth
        advisory = _split_complexity_advisory({"d.py": plan}, {"d.py": tree})
        assert (advisory is not None) is should_fire, (chunk_count, expected_depth, advisory)


# ---------------------------------------------------------------------------
# Fact ledger and narrative refinement (section 8)
# ---------------------------------------------------------------------------


def test_ledger_deduplicates_same_named_symbols_and_keeps_overloads_distinct() -> None:
    capsules = [
        {"description": "a", "functions": [{"name": "run", "description": "first"}]},
        {"description": "b", "functions": [{"name": "run", "description": "first"}]},
        {"description": "c", "functions": [{"name": "run", "signature": "run(x)", "description": "other"}]},
    ]
    ledger = build_fact_ledger(capsules)
    names = [f["name"] for f in ledger.functions]
    assert names.count("run") == 2  # one plain "run", one distinct-signature overload


def test_effective_language_invalidates_split_plan_and_node_identities() -> None:
    """A language remap must not silently reuse split work.

    The effective language selects the grammar and is rendered into every
    leaf/final prompt, so remapping an extension to a different language must
    change the division digest and everything derived from it. Ordinary
    (non-split) truncate cache identity is deliberately *not* touched here:
    the 0.14.0 plan requires those identities to retain their golden bytes.
    """
    from codedoc.core.record_meta import expected_large_file_identity

    source = "".join(f"value_{index:03d} = {index}\n" for index in range(200))
    budget = 2000
    seen: dict[str, tuple[str, str, str, str]] = {}
    for language in ("alpha", "beta"):
        plan = build_division_plan(
            rel_path="sample.foo",
            language=language,
            content=source,
            source_budget_chars=budget,
        )
        tree = build_reduction_tree(
            plan, max_content_chars=budget, language=language, imports=()
        )
        seen[language] = (
            plan.plan_digest,
            tree.tree_digest,
            expected_large_file_identity(
                source_chars=len(source),
                max_chars=budget,
                rel_path="sample.foo",
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                structural_mode=plan.structural_mode,
                imports_digest=file_division.deterministic_imports_digest(()),
            ),
            leaf_execution_identity(
                rel_path="sample.foo",
                content_hash="a" * 64,
                division_plan_digest=plan.plan_digest,
                provider_identity="provider-execution:" + "b" * 64,
                chunk=plan.chunks[0],
            ),
        )

    for index in range(4):
        assert seen["alpha"][index] != seen["beta"][index]


def test_overloads_survive_the_live_cleaner_into_the_ledger() -> None:
    """Regression: the fixed leaf cleaner must preserve `signature`.

    The test above builds ledger input directly, so it cannot see a cleaner
    that strips the only field distinguishing `run(int)` from `run(str)`.
    This exercises the real execution path — clean_leaf_capsule_report() and
    then build_fact_ledger() — which previously collapsed both overloads into
    a single published fact.
    """
    from codedoc.agents.response_cleaning import clean_leaf_capsule_report

    raw_capsules = (
        {
            "description": "Integer overload.",
            "functions": [
                {"name": "run", "signature": "run(int)", "description": "int form"}
            ],
        },
        {
            "description": "String overload.",
            "functions": [
                {"name": "run", "signature": "run(str)", "description": "str form"}
            ],
        },
    )
    cleaned = []
    for raw in raw_capsules:
        result = clean_leaf_capsule_report(raw, "Service.java")
        assert "unknown_field" not in result.removal_reason_codes
        assert result.value["functions"][0]["signature"]
        cleaned.append(result.value)

    ledger = build_fact_ledger(cleaned, language="java")

    assert [item["signature"] for item in ledger.functions] == [
        "run(int)",
        "run(str)",
    ]


@requires_structure_pack
def test_ledger_stores_parser_signature_not_model_hint_at_the_600_boundary() -> None:
    """Section 20A item 3: parser authority at the real 600-character
    boundary, using a genuinely parsed declaration rather than a
    hand-constructed `SymbolFact`.

    A real function with 40 typed, defaulted parameters produces a raw
    `def ...` line far longer than 600 characters; the real parser bounds
    the captured `SymbolFact.signature` at exactly `MAX_STRUCTURE_SIGNATURE_CHARS`
    (600), matching this release's leaf response ceiling. A distinct
    552-character string simulates the model's own matching-hint signature
    for the same declaration -- shorter, and different text, so a bug that
    let the model's report win would be visible. `build_fact_ledger` must
    store the parser-owned 600-character signature, never the 552-character
    hint, and must do so identically on a repeated call over the same
    inputs (allocation has no hidden non-determinism)."""
    params = ", ".join(f"param_{i:03d}: int = {i}" for i in range(40))
    source = f"def target_function({params}) -> None:\n    pass\n"
    plan = build_division_plan(
        rel_path="sample.py", language="python", content=source, source_budget_chars=100_000,
    )
    assert plan.structural_mode == "syntax"
    assert len(plan.chunks) == 1
    assert len(plan.symbols) == 1
    real_signature = plan.symbols[0].signature
    assert plan.symbols[0].qualified_name == "target_function"
    assert len(real_signature) == MAX_STRUCTURE_SIGNATURE_CHARS == 600

    model_hint = (
        "def target_function(" + ", ".join(f"p{i}: int" for i in range(70)) + ")"
    )[:552]
    assert len(model_hint) == 552
    assert model_hint != real_signature

    capsules = [
        {
            "description": "A function.",
            "functions": [
                {
                    "name": "target_function",
                    "signature": model_hint,
                    "description": "does work",
                }
            ],
        }
    ]

    ledger = build_fact_ledger(
        capsules, language="python", chunks=plan.chunks, symbols=plan.symbols,
    )
    assert len(ledger.functions) == 1
    stored_signature = ledger.functions[0]["signature"]
    assert stored_signature == real_signature
    assert stored_signature != model_hint

    # Repeated allocation over the identical inputs is deterministic.
    ledger_again = build_fact_ledger(
        capsules, language="python", chunks=plan.chunks, symbols=plan.symbols,
    )
    assert ledger_again.functions == ledger.functions


def test_ledger_uses_authoritative_unit_scope_for_same_named_packed_facts() -> None:
    from codedoc.agents.response_cleaning import clean_leaf_capsule_report

    plan = build_division_plan(
        rel_path="scopes.txt",
        language="unknown",
        content="scope_a\nscope_b\n",
        source_budget_chars=100,
    )
    assert len(plan.chunks) == 1
    chunk = plan.chunks[0]
    assert len(chunk.semantic_units) == 2
    first, second = chunk.semantic_units
    chunk = replace(
        chunk,
        semantic_units=(
            replace(first, qualified_name="ClassA.run", signature="run()"),
            replace(second, qualified_name="ClassB.run", signature="run()"),
        ),
    )
    cleaned = clean_leaf_capsule_report(
        {
            "description": "Two scoped methods.",
            "functions": [
                {"name": "run", "signature": "run()", "description": "Runs."},
                {"name": "run", "signature": "run()", "description": "Runs."},
            ],
        },
        "scopes.txt",
    ).value

    assert len(cleaned["functions"]) == 2
    ledger = build_fact_ledger(
        [cleaned],
        language="unknown",
        chunks=(chunk,),
    )

    assert len(ledger.functions) == 2
    assert [
        item["_provenance"][0]["semantic_unit_ids"]
        for item in ledger.functions
    ] == [[first.unit_id], [second.unit_id]]
    assert [
        item["_provenance"][0]["owning_ranges"]
        for item in ledger.functions
    ] == [
        [source_range.to_public() for source_range in chunk.owning_ranges],
        [source_range.to_public() for source_range in chunk.owning_ranges],
    ]


@requires_structure_pack
def test_ledger_uses_nested_symbol_ids_for_same_named_methods() -> None:
    source = (
        "class A:\n"
        "    def run(self):\n"
        "        return 1\n\n"
        "class B:\n"
        "    def run(self):\n"
        "        return 2\n"
    )
    plan = build_division_plan(
        rel_path="scoped.py",
        language="python",
        content=source,
        source_budget_chars=1000,
    )
    methods = tuple(
        symbol
        for symbol in plan.symbols
        if symbol.qualified_name == "run"
    )
    assert plan.structural_mode == "syntax"
    assert len(plan.chunks) == 1
    assert len(methods) == 2

    ledger = build_fact_ledger(
        [
            {
                "functions": [
                    {
                        "name": "run",
                        "signature": "run(self)",
                        "description": "Runs.",
                    },
                    {
                        "name": "run",
                        "signature": "run(self)",
                        "description": "Runs.",
                    },
                ]
            }
        ],
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 2
    assert [
        item["_provenance"][0]["symbol_id"]
        for item in ledger.functions
    ] == [method.symbol_id for method in methods]
    assert [
        item["_provenance"][0]["symbol"]["source_range"]
        for item in ledger.functions
    ] == [method.range.to_public() for method in methods]


def test_ledger_uses_occurrence_scopes_for_ambiguous_lexical_facts() -> None:
    plan = build_division_plan(
        rel_path="scoped.unknown",
        language="unknown",
        content="def run(): pass\ndef run(): pass\n",
        source_budget_chars=1000,
    )
    assert plan.structural_mode == "lexical"
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0].semantic_units) == 2

    ledger = build_fact_ledger(
        [
            {
                "functions": [
                    {
                        "name": "run",
                        "signature": "run()",
                        "description": "Runs.",
                    },
                    {
                        "name": "run",
                        "signature": "run()",
                        "description": "Runs.",
                    },
                ]
            }
        ],
        language="unknown",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 2
    candidate_ids = [
        unit.unit_id for unit in plan.chunks[0].semantic_units
    ]
    assert [
        item["_provenance"][0]["semantic_unit_ids"]
        for item in ledger.functions
    ] == [candidate_ids, candidate_ids]
    assert all(
        "symbol_id" not in item["_provenance"][0]
        for item in ledger.functions
    )


def test_lexical_single_unit_keeps_distinct_same_name_signatures() -> None:
    plan = build_division_plan(
        rel_path="single-unit.unknown",
        language="unknown",
        content="one lexical source line with no structural symbols\n",
        source_budget_chars=1000,
    )
    assert plan.structural_mode == "lexical"
    assert len(plan.chunks) == 1
    assert len(plan.chunks[0].semantic_units) == 1
    assert plan.symbols == ()

    facts = [
        {"name": "run", "signature": "run(value)", "description": "one"},
        {
            "name": "run",
            "signature": "run(value, other)",
            "description": "two",
        },
    ]
    ledger = build_fact_ledger(
        [{"functions": facts}],
        language="unknown",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )
    reversed_ledger = build_fact_ledger(
        [{"functions": list(reversed(facts))}],
        language="unknown",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert reversed_ledger == ledger
    assert len(ledger.functions) == 2
    assert {fact["signature"] for fact in ledger.functions} == {
        "run(value)",
        "run(value, other)",
    }


def test_lexical_single_unit_merges_signed_and_unsigned_continuations() -> None:
    plan = build_division_plan(
        rel_path="single-unit.unknown",
        language="unknown",
        content="one lexical source line with no structural symbols\n",
        source_budget_chars=1000,
    )
    ledger = build_fact_ledger(
        [
            {
                "functions": [
                    {"name": "run", "description": "continuation"},
                    {
                        "name": "run",
                        "signature": "run(value)",
                        "description": "declaration",
                    },
                ]
            }
        ],
        language="unknown",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 1
    assert ledger.functions[0]["signature"] == "run(value)"


@requires_structure_pack
def test_symbol_scoped_ledger_deduplicates_single_unit_continuations() -> None:
    source = (
        "def run():\n"
        f"    value = {'1' * 500}\n"
        "    return value\n"
    )
    plan = build_division_plan(
        rel_path="continued.py",
        language="python",
        content=source,
        source_budget_chars=100,
    )
    assert plan.structural_mode == "syntax"
    assert len(plan.units) == 1
    assert len(plan.chunks) > 1

    ledger = build_fact_ledger(
        [
            {
                "functions": [
                    {
                        "name": "run",
                        "signature": "run()",
                        "description": "Visible portion.",
                    }
                ]
            }
            for _chunk in plan.chunks
        ],
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 1
    origins = ledger.functions[0]["_provenance"]
    assert len(origins) == len(plan.chunks)
    assert {origin["symbol_id"] for origin in origins} == {
        plan.symbols[0].symbol_id
    }


@requires_structure_pack
def test_symbol_scoped_ledger_merges_signed_and_unsigned_continuations() -> None:
    source = (
        "def run(value: int) -> int:\n"
        f"    payload = {'1' * 500}\n"
        "    return value\n"
    )
    plan = build_division_plan(
        rel_path="continued.py",
        language="python",
        content=source,
        source_budget_chars=100,
    )
    assert len(plan.chunks) > 1

    capsules = []
    for index, _chunk in enumerate(plan.chunks):
        fact = {"name": "run", "description": f"part {index}"}
        if index % 2 == 0:
            fact["signature"] = "run(value: int) -> int"
        capsules.append({"functions": [fact]})

    ledger = build_fact_ledger(
        capsules,
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 1
    assert ledger.functions[0]["signature"] == plan.symbols[0].signature
    assert len(ledger.functions[0]["_provenance"]) == len(plan.chunks)


@requires_structure_pack
def test_symbol_scoped_ledger_reserves_signed_overload_before_unsigned_fact() -> None:
    source = (
        "def run(value: int) -> int:\n"
        "    return value\n\n"
        "def run(value: int, other: int) -> int:\n"
        "    return value + other\n"
    )
    plan = build_division_plan(
        rel_path="overloads.py",
        language="python",
        content=source,
        source_budget_chars=1000,
    )
    run_symbols = tuple(
        symbol for symbol in plan.symbols if symbol.qualified_name.endswith("run")
    )
    assert len(run_symbols) == 2
    assert len(plan.chunks) == 1

    # The model reports the later overload first with a signature, then the
    # earlier declaration without one. Allocation must not consume the later
    # parser symbol twice merely because response order differs from source.
    capsule = {
        "functions": [
            {
                "name": "run",
                "signature": run_symbols[1].signature,
                "description": "two arguments",
            },
            {"name": "run", "description": "one argument"},
        ]
    }
    ledger = build_fact_ledger(
        [capsule],
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )
    reversed_ledger = build_fact_ledger(
        [{"functions": list(reversed(capsule["functions"]))}],
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 2
    assert reversed_ledger == ledger
    assert {fact["signature"] for fact in ledger.functions} == {
        symbol.signature for symbol in run_symbols
    }
    assert {
        fact["_provenance"][0]["symbol_id"] for fact in ledger.functions
    } == {symbol.symbol_id for symbol in run_symbols}


@requires_structure_pack
def test_symbol_scoped_ledger_uses_ambiguity_scope_after_overloads_are_exhausted() -> None:
    source = (
        "def run(value: int) -> int:\n"
        "    return value\n\n"
        "def run(value: int, other: int) -> int:\n"
        "    return value + other\n"
    )
    plan = build_division_plan(
        rel_path="overloads.py",
        language="python",
        content=source,
        source_budget_chars=1000,
    )
    run_symbols = tuple(
        symbol for symbol in plan.symbols if symbol.qualified_name.endswith("run")
    )
    assert len(run_symbols) == 2
    assert len(plan.chunks) == 1

    ledger = build_fact_ledger(
        [
            {
                "functions": [
                    {
                        "name": "run",
                        "signature": run_symbols[0].signature,
                        "description": "first declaration",
                    },
                    {
                        "name": "run",
                        "signature": run_symbols[1].signature,
                        "description": "second declaration",
                    },
                    {"name": "run", "description": "ambiguous extra report"},
                ]
            }
        ],
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )

    assert len(ledger.functions) == 3
    attributed = [
        fact
        for fact in ledger.functions
        if "symbol_id" in fact["_provenance"][0]
    ]
    assert {fact["_provenance"][0]["symbol_id"] for fact in attributed} == {
        symbol.symbol_id for symbol in run_symbols
    }
    ambiguous = next(
        fact
        for fact in ledger.functions
        if fact["description"] == "ambiguous extra report"
    )
    ambiguous_origin = ambiguous["_provenance"][0]
    assert "symbol_id" not in ambiguous_origin
    assert set(ambiguous_origin["semantic_unit_ids"]) == {
        unit.unit_id for unit in plan.units
    }


def test_ledger_deduplicates_continuation_reports_with_complete_provenance() -> None:
    plan = build_division_plan(
        rel_path="continued.txt",
        language="unknown",
        content="x" * 25,
        source_budget_chars=10,
    )
    capsules = [
        {"functions": [{"name": "run", "description": "Visible portion."}]}
        for _chunk in plan.chunks
    ]

    ledger = build_fact_ledger(
        capsules,
        language="unknown",
        chunks=plan.chunks,
    )

    assert len(ledger.functions) == 1
    origins = ledger.functions[0]["_provenance"]
    assert [origin["chunk_id"] for origin in origins] == [
        chunk.chunk_id for chunk in plan.chunks
    ]
    assert [origin["source_order"] for origin in origins] == list(
        range(len(plan.chunks))
    )


def test_ledger_signature_normalization_does_not_split_one_overload() -> None:
    """Whitespace/Unicode spelling differences must not create two entries."""
    capsules = [
        {"functions": [{"name": "run", "signature": "run(int,  str)"}]},
        {"functions": [{"name": "run", "signature": "run(int, str)"}]},
    ]
    ledger = build_fact_ledger(capsules, language="java")
    assert len(ledger.functions) == 1


def test_ledger_unicode_equivalent_names_deduplicate() -> None:
    """NFC-equivalent identifiers are one fact, not two (D7/section 8)."""
    capsules = [
        {"functions": [{"name": "café"}]},          # precomposed
        {"functions": [{"name": "café"}]},          # combining acute
    ]
    ledger = build_fact_ledger(capsules, language="python")
    assert len(ledger.functions) == 1


def test_ledger_normalizes_export_dedup_and_preserves_order() -> None:
    capsules = [
        {"exports": ["Alpha", "Beta"]},
        {"exports": ["Beta", "Gamma"]},
    ]
    ledger = build_fact_ledger(capsules)
    assert ledger.exports == ("Alpha", "Beta", "Gamma")


def test_ledger_language_sensitive_case_normalization() -> None:
    capsules = [{"functions": [{"name": "Run"}]}, {"functions": [{"name": "run"}]}]
    case_sensitive = build_fact_ledger(capsules, language="python")
    assert len(case_sensitive.functions) == 2
    case_insensitive = build_fact_ledger(capsules, language="sql")
    assert len(case_insensitive.functions) == 1


def test_merge_leaf_capsules_is_the_whole_file_entry_point() -> None:
    capsules = [{"functions": [{"name": "a"}]}, {"classes": [{"name": "B"}]}]
    assert merge_leaf_capsules(capsules) == build_fact_ledger(capsules)


def test_repeated_boilerplate_narrative_does_not_grow_with_dedup() -> None:
    narratives = ["Comment-only fragment; no executable symbols."] * 50
    refined = refine_narrative_inputs(narratives)
    assert refined == ("Comment-only fragment; no executable symbols.",)


def test_ledger_order_is_deterministic_across_repeated_builds() -> None:
    capsules = [
        {"functions": [{"name": f"fn_{i}"}]} for i in range(20)
    ]
    first = build_fact_ledger(capsules)
    second = build_fact_ledger(list(reversed(capsules))[::-1])
    assert first == second


# ---------------------------------------------------------------------------
# Final synthesis input
# ---------------------------------------------------------------------------


def test_final_synthesis_input_is_bounded_and_grounded() -> None:
    ledger = build_fact_ledger([{"functions": [{"name": "a"}], "exports": ["A"]}])
    manifest = final_synthesis_input(
        rel_path="a.py",
        language="python",
        imports=("os", "sys"),
        root_narratives=("Refined root narrative.",),
        root_coverage_leaf_ids=("chunk_" + "a" * 64,),
        ledger=ledger,
    )
    import json

    data = json.loads(manifest)
    assert data["file_path"] == "a.py"
    assert data["imports"] == ["os", "sys"]
    assert data["root_narratives"] == ["Refined root narrative."]
    assert data["fact_ledger_synopsis"]["functions"] == ["a"]
    assert data["fact_ledger_synopsis"]["exports"] == ["A"]


def test_final_synthesis_input_accepts_multiple_root_narratives() -> None:
    ledger = build_fact_ledger([])
    manifest = final_synthesis_input(
        rel_path="a.py", language="python", imports=(),
        root_narratives=("first root.", "second root."),
        root_coverage_leaf_ids=("chunk_" + "a" * 64, "chunk_" + "b" * 64), ledger=ledger,
    )
    import json

    data = json.loads(manifest)
    assert data["root_narratives"] == ["first root.", "second root."]


def test_final_synthesis_input_ledger_synopsis_is_bounded() -> None:
    ledger = build_fact_ledger(
        [{"functions": [{"name": f"function_with_a_fairly_long_name_{i}"} for i in range(400)]}]
    )
    manifest = final_synthesis_input(
        rel_path="a.py", language="python", imports=(), root_narratives=("x",),
        root_coverage_leaf_ids=("chunk_" + "a" * 64,), ledger=ledger,
    )
    assert len(manifest) < file_division.MAX_LEDGER_SYNOPSIS_CHARS + 2000


# ---------------------------------------------------------------------------
# Provider/model/effective-endpoint execution identity (D12)
# ---------------------------------------------------------------------------


def test_provider_execution_identity_never_persists_auto_or_credentials() -> None:
    config = {"llm_provider": "auto", "model_name": "", "api_key": "sk-secret", "api_base_url": None}
    identity = provider_execution_identity(config)
    assert "auto" not in identity
    assert "sk-secret" not in identity
    assert identity.startswith("provider-execution:")


def test_provider_execution_identity_changes_with_endpoint() -> None:
    base = {"llm_provider": "openai", "model_name": "gpt-4o-mini"}
    default_endpoint = provider_execution_identity({**base, "api_base_url": None})
    custom_endpoint = provider_execution_identity(
        {**base, "api_base_url": "https://Example.com:9000/v1"}
    )
    assert default_endpoint != custom_endpoint
    # Re-deriving from an equivalent URL (case/whitespace only) matches.
    again = provider_execution_identity(
        {**base, "api_base_url": "  HTTPS://EXAMPLE.com:9000/v1  "}
    )
    assert custom_endpoint == again


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:pw@example.com:9000/v1",
        "https://user@example.com:9000/v1",
        "https://example.com:9000/v1?x=1",
        "https://example.com:9000/v1#frag",
    ],
)
def test_provider_execution_identity_rejects_username_password_query_fragment(
    endpoint: str,
) -> None:
    """A username, password, query string, or fragment falls outside the
    four-field canonical identity (scheme/host/port/path) and would otherwise
    silently canonicalize identically to a "clean" URL missing it -- letting an
    endpoint-trust approval cover a differently-behaving endpoint. Rejected
    instead of dropped, for both the configured api_base_url and a runtime
    approval URL, since both share this same identity function."""
    with pytest.raises(ConfigError) as blocked:
        provider_execution_identity(
            {
                "llm_provider": "openai",
                "model_name": "gpt-4o-mini",
                "api_base_url": endpoint,
            }
        )
    message = str(blocked.value)
    assert "username" in message
    assert "password" in message
    assert "query" in message
    assert "fragment" in message
    assert endpoint not in message


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:super-secret@example.com:notaport/v1",
        "https://user:super-secret@example.com:70000/v1",
        "ftp://user:super-secret@example.com/v1",
        "https://[::1/v1?token=super-secret",
        "https://example.com:/v1?token=super-secret",
    ],
)
def test_provider_execution_identity_rejects_malformed_endpoint_without_leaking_it(
    endpoint: str,
) -> None:
    with pytest.raises(ConfigError) as blocked:
        provider_execution_identity(
            {
                "llm_provider": "openai",
                "model_name": "gpt-4o-mini",
                "api_base_url": endpoint,
            }
        )

    message = str(blocked.value)
    assert "valid HTTP or HTTPS URL" in message
    assert "super-secret" not in message
    assert endpoint not in message


@pytest.mark.parametrize(
    ("implicit", "explicit"),
    [
        ("https://example.com/v1", "https://example.com:443/v1"),
        ("http://example.com/v1", "http://example.com:80/v1"),
    ],
)
def test_provider_execution_identity_normalizes_default_endpoint_ports(
    implicit: str, explicit: str
) -> None:
    base = {"llm_provider": "openai", "model_name": "gpt-4o-mini"}
    assert provider_execution_identity(
        {**base, "api_base_url": implicit}
    ) == provider_execution_identity(
        {**base, "api_base_url": explicit}
    )


@pytest.mark.parametrize(
    ("plain", "trailing"),
    [
        ("https://example.com", "https://example.com/"),
        ("https://example.com/v1", "https://example.com/v1/"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1///"),
    ],
)
def test_provider_execution_identity_normalizes_endpoint_trailing_slashes(
    plain: str,
    trailing: str,
) -> None:
    base = {"llm_provider": "openai", "model_name": "gpt-4o-mini"}
    assert provider_execution_identity(
        {**base, "api_base_url": plain}
    ) == provider_execution_identity(
        {**base, "api_base_url": trailing}
    )


def test_provider_execution_verification_rejects_missing_attestation() -> None:
    config = {"llm_provider": "openai", "model_name": "gpt-4o-mini"}
    planned = provider_execution_identity(config)

    with pytest.raises(ConfigError, match="missing a valid concrete execution attestation"):
        verify_provider_execution_identity(object(), config, planned)


def test_leaf_reduction_final_identities_are_distinct_and_stable() -> None:
    plan = build_division_plan(
        rel_path="a.py",
        language="python",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    chunk = plan.chunks[0]
    reduction_node = (
        tree.unit_consolidation_nodes + tree.general_nodes
    )[0]
    leaf = leaf_execution_identity(
        rel_path="a.py", content_hash="a" * 64, division_plan_digest=plan.plan_digest,
        provider_identity="provider-execution:" + "b" * 64,
        chunk=chunk,
    )
    reduction = reduction_execution_identity(
        rel_path="a.py", content_hash="a" * 64, division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        provider_identity="provider-execution:" + "b" * 64,
        node=reduction_node,
    )
    final = final_execution_identity(
        rel_path="a.py", content_hash="a" * 64, division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        provider_identity="provider-execution:" + "b" * 64,
        prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=file_division.deterministic_imports_digest(()),
        node=tree.final_node,
    )
    assert len({leaf, reduction, final}) == 3
    # A final-shape-only change reruns final synthesis but not leaf/reduction.
    final_other_profile = final_execution_identity(
        rel_path="a.py", content_hash="a" * 64, division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        provider_identity="provider-execution:" + "b" * 64,
        prompt_profile_digest="some-other-digest",
        imports_digest=file_division.deterministic_imports_digest(()),
        node=tree.final_node,
    )
    assert final_other_profile != final


def test_final_synthesis_revision_prunes_only_recovered_final(
    monkeypatch,
) -> None:
    plan = build_division_plan(
        rel_path="a.py",
        language="unknown",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64
    profile_digest = "no-prompt-profile-v1"

    old_identities = {
        chunk.chunk_id: leaf_execution_identity(
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=chunk,
        )
        for chunk in plan.chunks
    }
    old_identities.update(
        {
            node.node_id: reduction_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                node=node,
            )
            for node in tree.all_intermediate_nodes
        }
    )
    old_identities[tree.final_node.node_id] = final_execution_identity(
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        provider_identity=provider_identity,
        prompt_profile_digest=profile_digest,
        imports_digest=file_division.deterministic_imports_digest(()),
        node=tree.final_node,
    )

    _INPUT_DIGEST = "test-input:" + "7" * 64

    states = [
        tree_node_state(
            node_id=chunk.chunk_id,
            node_type="leaf",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=_INPUT_DIGEST,
            execution_identity_digest=old_identities[chunk.chunk_id],
            unit_id=None,
            child_ids=(),
            coverage_leaf_ids=(chunk.chunk_id,),
            result={
                "description": "Leaf.",
                "chunk_id": chunk.chunk_id,
                "unit_id": chunk.unit_id,
            },
        )
        for chunk in plan.chunks
    ]
    states.extend(
        tree_node_state(
            node_id=node.node_id,
            node_type=node.phase,
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=_INPUT_DIGEST,
            execution_identity_digest=old_identities[node.node_id],
            unit_id=node.unit_id,
            child_ids=node.child_ids,
            coverage_leaf_ids=node.leaf_ids,
            result={"narrative": "Reduced."},
        )
        for node in tree.all_intermediate_nodes
    )
    final = tree.final_node
    states.append(
        tree_node_state(
            node_id=final.node_id,
            node_type="final",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=_INPUT_DIGEST,
            execution_identity_digest=old_identities[final.node_id],
            unit_id=None,
            child_ids=final.child_ids,
            coverage_leaf_ids=final.leaf_ids,
            result={"description": "Final."},
        )
    )

    monkeypatch.setattr(
        file_division,
        "FINAL_SYNTHESIS_REVISION",
        "file-synthesis-test-new",
    )
    current_identities = {
        chunk.chunk_id: leaf_execution_identity(
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=chunk,
        )
        for chunk in plan.chunks
    }
    current_identities.update(
        {
            node.node_id: reduction_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                node=node,
            )
            for node in tree.all_intermediate_nodes
        }
    )
    current_identities[final.node_id] = final_execution_identity(
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        provider_identity=provider_identity,
        prompt_profile_digest=profile_digest,
        imports_digest=file_division.deterministic_imports_digest(()),
        node=final,
    )

    assert all(
        old_identities[node_id] == current_identities[node_id]
        for node_id in (
            *(chunk.chunk_id for chunk in plan.chunks),
            *(node.node_id for node in tree.all_intermediate_nodes),
        )
    )
    assert old_identities[final.node_id] != current_identities[final.node_id]
    individually_current = tuple(
        state
        for state in states
        if state.execution_identity_digest
        == current_identities[state.node_id]
    )
    retained = dependency_closed_nodes(
        individually_current,
        plan=plan,
        tree=tree,
    )
    assert {state.node_id for state in retained} == {
        *(chunk.chunk_id for chunk in plan.chunks),
        *(node.node_id for node in tree.all_intermediate_nodes),
    }


# ---------------------------------------------------------------------------
# Node-keyed recovery (schema version 4) and legacy (v1/v2) detection — section 12/D11
# ---------------------------------------------------------------------------


def test_validate_node_for_tree_accepts_exact_match_and_rejects_drift() -> None:
    plan = build_division_plan(
        rel_path="a.py", language="unknown", content="a = 1\nb = 2\n", source_budget_chars=1000
    )
    tree = build_reduction_tree(plan, max_content_chars=12000)
    leaf_chunk = plan.chunks[0]
    identity = "division-execution:" + "d" * 64
    node = tree_node_state(
        node_id=leaf_chunk.chunk_id,
        node_type="leaf",
        rel_path="a.py",
        content_hash="a" * 64,
        division_plan_digest=plan.plan_digest,
        input_digest="test-input:" + "7" * 64,
        execution_identity_digest=identity,
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(leaf_chunk.chunk_id,),
        result={
            "description": "ok",
            "chunk_id": leaf_chunk.chunk_id,
            "unit_id": leaf_chunk.unit_id,
        },
    )
    assert validate_node_for_tree(
        node, plan=plan, tree=tree, content_hash="a" * 64, expected_identity=identity
    )
    assert not validate_node_for_tree(
        node, plan=plan, tree=tree, content_hash="different", expected_identity=identity
    )
    assert not validate_node_for_tree(
        node, plan=plan, tree=tree, content_hash="a" * 64, expected_identity="division-execution:" + "0" * 64
    )


def test_validate_node_for_tree_rejects_foreign_node_id() -> None:
    plan = build_division_plan(
        rel_path="a.py", language="unknown", content="a = 1\nb = 2\n", source_budget_chars=1000
    )
    tree = build_reduction_tree(plan, max_content_chars=12000)
    identity = "division-execution:" + "d" * 64
    node = tree_node_state(
        node_id="chunk_" + "f" * 64,
        node_type="leaf",
        rel_path="a.py",
        content_hash="a" * 64,
        division_plan_digest=plan.plan_digest,
        input_digest="test-input:" + "7" * 64,
        execution_identity_digest=identity,
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=("chunk_" + "f" * 64,),
        result={"description": "ok"},
    )
    assert not validate_node_for_tree(
        node, plan=plan, tree=tree, content_hash="a" * 64, expected_identity=identity
    )


def test_validate_node_for_tree_rejects_stage_swap_and_empty_leaf_capsule() -> None:
    plan = build_division_plan(
        rel_path="a.py",
        language="unknown",
        content="a = 1\nb = 2\n",
        source_budget_chars=1000,
    )
    tree = build_reduction_tree(plan, max_content_chars=12000)
    chunk = plan.chunks[0]
    identity = "division-execution:" + "d" * 64
    common = {
        "node_id": chunk.chunk_id,
        "rel_path": plan.rel_path,
        "content_hash": "a" * 64,
        "division_plan_digest": plan.plan_digest,
        "input_digest": "test-input:" + "7" * 64,
        "execution_identity_digest": identity,
        "unit_id": None,
        "child_ids": (),
        "coverage_leaf_ids": (chunk.chunk_id,),
    }
    stage_swapped = tree_node_state(
        **common,
        node_type="final",
        result={
            "description": "looks plausible",
            "chunk_id": chunk.chunk_id,
            "unit_id": chunk.unit_id,
        },
    )
    empty_leaf = tree_node_state(
        **common,
        node_type="leaf",
        result={
            "chunk_id": chunk.chunk_id,
            "unit_id": chunk.unit_id,
        },
    )

    assert not validate_node_for_tree(
        stage_swapped,
        plan=plan,
        tree=tree,
        content_hash="a" * 64,
        expected_identity=identity,
    )
    assert not validate_node_for_tree(
        empty_leaf,
        plan=plan,
        tree=tree,
        content_hash="a" * 64,
        expected_identity=identity,
    )


def test_validate_node_for_tree_rejects_a_coverage_permutation() -> None:
    """Section 11: coverage is compared by ordered-tuple equality, never a
    sorted set, so a node carrying exactly the planned leaf IDs in a
    different order must be rejected rather than validated.  The `0.14.1`
    baseline compared `tuple(sorted(...))` in both symbols below and would
    accept the permuted node, which is what makes this a regression guard."""
    plan = build_division_plan(
        rel_path="a.py",
        language="unknown",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    identity = "division-execution:" + "d" * 64
    input_digest = "test-input:" + "7" * 64
    reducer = next(
        node for node in tree.all_intermediate_nodes if len(node.leaf_ids) >= 2
    )

    def _reducer_with_coverage(coverage):
        return tree_node_state(
            node_id=reducer.node_id,
            node_type=reducer.phase,
            rel_path=plan.rel_path,
            content_hash="a" * 64,
            division_plan_digest=plan.plan_digest,
            input_digest=input_digest,
            execution_identity_digest=identity,
            unit_id=reducer.unit_id,
            child_ids=reducer.child_ids,
            coverage_leaf_ids=coverage,
            result={"narrative": "Reduced."},
        )

    def _permute(leaf_ids):
        # Same multiset, different order: only an ordered comparison separates
        # these two, so a sorted-set comparison would accept both.
        permuted = (leaf_ids[1], leaf_ids[0], *leaf_ids[2:])
        assert permuted != leaf_ids
        assert sorted(permuted) == sorted(leaf_ids)
        return permuted

    exact = _reducer_with_coverage(reducer.leaf_ids)
    permuted = _reducer_with_coverage(_permute(reducer.leaf_ids))

    assert validate_node_for_tree(
        exact,
        plan=plan,
        tree=tree,
        content_hash="a" * 64,
        expected_identity=identity,
        expected_input_digest=input_digest,
    )
    assert not validate_node_for_tree(
        permuted,
        plan=plan,
        tree=tree,
        content_hash="a" * 64,
        expected_identity=identity,
        expected_input_digest=input_digest,
    )

    # The final-coverage predicate carries the same ordered rule.
    final = tree.final_node
    assert final.leaf_ids == tuple(chunk.chunk_id for chunk in plan.chunks)

    def _final_with_coverage(coverage):
        return tree_node_state(
            node_id=final.node_id,
            node_type="final",
            rel_path=plan.rel_path,
            content_hash="a" * 64,
            division_plan_digest=plan.plan_digest,
            input_digest=input_digest,
            execution_identity_digest=identity,
            unit_id=final.unit_id,
            child_ids=final.child_ids,
            coverage_leaf_ids=coverage,
            result={"description": "Final."},
        )

    assert final_node_covers_every_leaf(plan, _final_with_coverage(final.leaf_ids))
    assert not final_node_covers_every_leaf(
        plan, _final_with_coverage(_permute(final.leaf_ids))
    )


def test_recovered_reducer_and_final_nodes_require_dependency_closure() -> None:
    plan = build_division_plan(
        rel_path="a.py",
        language="unknown",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    final = tree.final_node
    orphan = tree_node_state(
        node_id=final.node_id,
        node_type="final",
        rel_path=plan.rel_path,
        content_hash="a" * 64,
        division_plan_digest=plan.plan_digest,
        input_digest="test-input:" + "7" * 64,
        execution_identity_digest="division-execution:" + "d" * 64,
        unit_id=final.unit_id,
        child_ids=final.child_ids,
        coverage_leaf_ids=final.leaf_ids,
        result={"description": "orphaned final"},
    )

    assert dependency_closed_nodes((orphan,), plan=plan, tree=tree) == ()


# ---------------------------------------------------------------------------
# 0.14.4: quarantine bound raised to 2 * MAX_CHUNKS_PER_FILE (512), so a
# revision advance that invalidates every node of an existing schema-4
# checkpoint quarantines the whole plan instead of aborting. Tested in three
# separate layers because they raise different exception types, and without
# constructing an impossible (over-bound) real plan.
# ---------------------------------------------------------------------------


def test_quarantine_bound_equals_2x_max_chunks_per_file() -> None:
    assert MAX_QUARANTINE_ENTRIES_PER_FILE == 2 * MAX_CHUNKS_PER_FILE == 512


def _quarantine_entries(count: int) -> tuple:
    return tuple(
        QuarantineEntry(
            node_id=f"chunk_{index:04d}".ljust(64, "0"),
            reason="stale-revision",
            raw_json="{}",
        )
        for index in range(count)
    )


def _empty_tree_state(*, quarantine: tuple) -> SplitTreeState:
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash="a" * 64,
        division_plan_digest="division-plan:" + "b" * 64,
        reduction_tree_digest="reduction-tree:" + "c" * 64,
        nodes=(),
        quarantine=quarantine,
    )


def test_split_tree_state_accepts_exactly_the_bounded_quarantine_count() -> None:
    """Container layer: exactly MAX_QUARANTINE_ENTRIES_PER_FILE entries is
    accepted."""
    state = _empty_tree_state(quarantine=_quarantine_entries(MAX_QUARANTINE_ENTRIES_PER_FILE))
    assert len(state.quarantine) == MAX_QUARANTINE_ENTRIES_PER_FILE


def test_split_tree_state_rejects_one_over_the_bound_with_plain_value_error() -> None:
    """Container layer: one entry over the bound raises ValueError from the
    dataclass's own __post_init__ bound check -- not SplitRecoveryStateError,
    even though that type is itself a ValueError subclass."""
    with pytest.raises(ValueError) as caught:
        _empty_tree_state(quarantine=_quarantine_entries(MAX_QUARANTINE_ENTRIES_PER_FILE + 1))

    assert type(caught.value) is ValueError
    assert not isinstance(caught.value, SplitRecoveryStateError)
    assert "quarantine map exceeds" in str(caught.value)


def test_validate_recovered_tree_rejects_over_bound_quarantine_with_recovery_state_error(
    monkeypatch,
) -> None:
    """Validation layer: driving validate_recovered_tree with a synthetic
    planned node set past a (monkeypatched small) bound raises
    SplitRecoveryStateError, distinct from the container layer's plain
    ValueError."""
    plan = build_division_plan(
        rel_path="a.py",
        language="unknown",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    assert len(plan.chunks) >= 2
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64

    # Every leaf carries a deliberately wrong execution identity, so every
    # one of them is quarantined as stale.
    nodes = [
        tree_node_state(
            node_id=chunk.chunk_id,
            node_type="leaf",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=file_division.leaf_input_digest(
                rel_path=plan.rel_path,
                language="unknown",
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

    monkeypatch.setattr(file_division, "MAX_QUARANTINE_ENTRIES_PER_FILE", 1)

    with pytest.raises(SplitRecoveryStateError, match="quarantine exceeds its bounded"):
        validate_recovered_tree(
            nodes,
            plan=plan,
            tree=tree,
            content_hash=content_hash,
            provider_identity=provider_identity,
            prompt_profile_digest="no-prompt-profile-v1",
            imports_digest=file_division.deterministic_imports_digest(()),
            language="unknown",
        )


def test_validate_recovered_tree_prunes_a_reducer_whose_child_narrative_changed() -> None:
    """Section 11: a reducer's execution identity never binds its children's
    actual result content (only structure/provenance), so an individually-
    valid, dependency-closed reducer checkpoint built from a leaf's OLD
    narrative must be rejected once that leaf's stored result changes --
    even though the leaf's own identity and input digest (purely structural)
    still validate on their own. This is the regression the recompute-from-
    retained-children input digest exists to catch."""
    plan = build_division_plan(
        rel_path="a.py",
        language="unknown",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64
    profile_digest = "no-prompt-profile-v1"
    reduction_node = (tree.unit_consolidation_nodes + tree.general_nodes)[0]
    imports_digest = file_division.deterministic_imports_digest(())

    results_by_id: dict[str, dict] = {}
    nodes = []
    for chunk in plan.chunks:
        result = {
            "description": "original",
            "chunk_id": chunk.chunk_id,
            "unit_id": chunk.unit_id,
        }
        results_by_id[chunk.chunk_id] = result
        nodes.append(
            tree_node_state(
                node_id=chunk.chunk_id,
                node_type="leaf",
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                input_digest=file_division.leaf_input_digest(
                    rel_path=plan.rel_path,
                    language="unknown",
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
                result=result,
            )
        )

    raw_narratives = tuple(
        results_by_id[child_id]["description"] for child_id in reduction_node.child_ids
    )
    nodes.append(
        tree_node_state(
            node_id=reduction_node.node_id,
            node_type=reduction_node.phase,
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=file_division.reduction_input_digest(
                rel_path=plan.rel_path,
                phase=reduction_node.phase,
                level=reduction_node.level,
                unit_id=reduction_node.unit_id,
                child_count=len(reduction_node.child_ids),
                ordered_child_narratives=file_division.refine_narrative_inputs(
                    raw_narratives
                ),
            ),
            execution_identity_digest=reduction_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                node=reduction_node,
            ),
            unit_id=reduction_node.unit_id,
            child_ids=reduction_node.child_ids,
            coverage_leaf_ids=reduction_node.leaf_ids,
            result={"narrative": "reduced from original"},
        )
    )

    retained, quarantine_entries = validate_recovered_tree(
        nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=profile_digest,
        imports_digest=imports_digest,
        language="unknown",
    )
    assert {state.node_id for state in retained} == {
        chunk.chunk_id for chunk in plan.chunks
    } | {reduction_node.node_id}
    assert quarantine_entries == ()

    mutated_leaf_id = reduction_node.child_ids[0]
    mutated_unit_id = next(
        chunk.unit_id for chunk in plan.chunks if chunk.chunk_id == mutated_leaf_id
    )
    replaced_result = file_division.canonical_json(
        {
            "description": "replaced",
            "chunk_id": mutated_leaf_id,
            "unit_id": mutated_unit_id,
        }
    )
    mutated_nodes = tuple(
        replace(node, result_json=replaced_result)
        if node.node_id == mutated_leaf_id
        else node
        for node in nodes
    )

    retained_after_mutation, quarantine_after_mutation = validate_recovered_tree(
        mutated_nodes,
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=profile_digest,
        imports_digest=imports_digest,
        language="unknown",
    )
    retained_ids_after = {state.node_id for state in retained_after_mutation}
    assert mutated_leaf_id in retained_ids_after
    assert reduction_node.node_id not in retained_ids_after
    assert len(quarantine_after_mutation) == 1
    assert quarantine_after_mutation[0].node_id == reduction_node.node_id
    assert quarantine_after_mutation[0].reason == "input-digest-mismatch"


def test_worst_case_helpers_use_distinct_maximum_outputs_and_full_ledger() -> None:
    narratives = maximum_distinct_narratives(4)
    assert len(set(narratives)) == 4
    assert all(
        len(narrative) == file_division.MAX_REDUCTION_NARRATIVE_CHARS
        for narrative in narratives
    )
    assert all(
        len(file_division.canonical_json(narrative))
        == 2 + (6 * file_division.MAX_REDUCTION_NARRATIVE_CHARS)
        for narrative in narratives
    )
    assert (
        len(file_division.render_reduction_child_manifest(narratives))
        == worst_case_reduction_manifest_chars(len(narratives))
    )

    ledger = maximally_populated_fact_ledger(3, language="python")
    assert len(ledger.functions) == 3 * file_division.MAX_LEAF_SYMBOL_ITEMS_PER_KIND
    assert len(ledger.classes) == 3 * file_division.MAX_LEAF_SYMBOL_ITEMS_PER_KIND
    assert len(ledger.exports) == 3 * file_division.MAX_LEAF_EXPORT_ITEMS


def test_final_synthesis_character_bound_dominates_valid_trimmed_ledgers() -> None:
    rel_path = "large.py"
    language = "python"
    imports = ("import os",)
    narratives = maximum_distinct_narratives(1)
    coverage = ("leaf-0",)
    control_heavy = maximally_populated_fact_ledger(1, language=language)
    ascii_exports = tuple(
        f"export_{ordinal:02d}".ljust(
            file_division.MAX_LEAF_EXPORT_ITEM_CHARS,
            "x",
        )
        for ordinal in range(file_division.MAX_LEAF_EXPORT_ITEMS)
    )
    ascii_heavy = file_division.FactLedger(exports=ascii_exports)
    bound = worst_case_final_synthesis_chars(
        rel_path=rel_path,
        language=language,
        imports=imports,
        root_count=1,
        leaf_count=1,
    )

    for ledger in (control_heavy, ascii_heavy):
        manifest = final_synthesis_input(
            rel_path=rel_path,
            language=language,
            imports=imports,
            root_narratives=narratives,
            root_coverage_leaf_ids=coverage,
            ledger=ledger,
            max_chars=12000,
        )
        assert len(manifest) <= bound

    assert worst_case_final_synthesis_chars(
        rel_path=rel_path,
        language=language,
        imports=imports,
        root_count=1,
        leaf_count=1,
        max_chars=4000,
    ) == 4000
    assert worst_case_final_synthesis_chars(
        rel_path=rel_path,
        language=language,
        imports=imports,
        root_count=1,
        leaf_count=1,
        max_chars=685,
    ) > 685


def test_legacy_v1_split_partial_is_detected_structurally() -> None:
    legacy = {
        "schema_version": 1,
        "owner": "codedoc-ai",
        "rel_path": "a.py",
        "content_hash": "a" * 64,
        "execution_identity_digest": "division-execution:" + "a" * 64,
        "division_plan_digest": "division-plan:" + "b" * 64,
        "stage": "documenting",
        "completed_chunks": [],
    }
    assert is_legacy_split_partial(legacy)
    current = {"schema_version": 2, "owner": "codedoc-ai", "nodes": []}
    assert not is_legacy_split_partial(current)
    assert not is_legacy_split_partial({"unrelated": True})
    assert not is_legacy_split_partial(None)


def test_distinct_units_preserves_first_seen_order() -> None:
    plan = build_division_plan(
        rel_path="a.py", language="unknown", content=_large_source(60), source_budget_chars=1000
    )
    units = distinct_units(plan.chunks)
    assert units == plan.units


# ---------------------------------------------------------------------------
# Section 2A: split-leaf signature bound aligned with the parser ceiling
# ---------------------------------------------------------------------------


def _source_range() -> SourceRange:
    return SourceRange(
        start_byte=0, end_byte=1, start_line=1, start_column=1, end_line=1, end_column=2
    )


def test_semantic_unit_identity_signature_bound_matches_the_parser_ceiling() -> None:
    """`SemanticUnitIdentity.signature` keeps its existing 600-character
    maximum, now sourced from the same shared `MAX_STRUCTURE_SIGNATURE_CHARS`
    constant that also defines `MAX_LEAF_SYMBOL_SIGNATURE_CHARS` -- so the two
    bounds cannot drift apart -- rather than a separate literal `600`."""
    assert MAX_LEAF_SYMBOL_SIGNATURE_CHARS == MAX_STRUCTURE_SIGNATURE_CHARS == 600

    accepted = SemanticUnitIdentity(
        unit_id="unit_" + "a" * 64,
        kind="function",
        qualified_name="q",
        signature="s" * 600,
        atom_ids=("atom_" + "b" * 64,),
        source_range=_source_range(),
    )
    assert len(accepted.signature) == 600

    with pytest.raises(ValueError, match="exceeds 600 characters"):
        SemanticUnitIdentity(
            unit_id="unit_" + "a" * 64,
            kind="function",
            qualified_name="q",
            signature="s" * 601,
            atom_ids=("atom_" + "b" * 64,),
            source_range=_source_range(),
        )


def test_derived_leaf_capsule_maximum_is_exactly_448672() -> None:
    """The capsule bound is derived from shared constants, not hard-coded.

    `0.14.3` raised `MAX_LEAF_SYMBOL_SIGNATURE_CHARS` from 256 to 600,
    raising the worst-case leaf capsule from 150,656 to 200,192 canonical
    characters. `0.14.4` raises `MAX_LEAF_SYMBOL_ITEMS_PER_KIND` from 12 to
    `MAX_KNOWN_SYMBOLS_PER_CHUNK` (32), raising it again to exactly 448,672 --
    a 248,480-character increase from 20 additional items in each of the two
    per-kind arrays (functions, classes): each additional item's canonical
    JSON (128+300+600 raw characters, each escaped 6-fold as `\\u0000`) plus
    its array-separator comma is 6,212 characters, and
    2 * 20 * 6,212 = 248,480."""
    assert MAX_LEAF_CAPSULE_CANONICAL_CHARS == 448672
    assert MAX_LEAF_CAPSULE_CANONICAL_CHARS - 200192 == 248480


def test_leaf_symbol_per_kind_cap_matches_known_symbols_prompt_bound() -> None:
    """A split leaf prompt can list up to `MAX_KNOWN_SYMBOLS_PER_CHUNK` known
    symbol names per kind (parser-derived prompt grounding); the rendered
    response contract's per-kind cap must accept at least that many, or a
    truthful response naming every known symbol would be rejected in full.
    `MAX_LEAF_SYMBOL_ITEMS_PER_KIND` is derived from the shared constant so
    the two bounds cannot silently diverge again, and the rendered prompt
    text (`_FRAGMENT_SHAPE_BLOCK`) states that same number for both
    `functions` and `classes`."""
    from codedoc.agents.file_documentation_agent import _FRAGMENT_SHAPE_BLOCK

    assert MAX_LEAF_SYMBOL_ITEMS_PER_KIND == MAX_KNOWN_SYMBOLS_PER_CHUNK == 32
    assert (
        f"functions <= {MAX_LEAF_SYMBOL_ITEMS_PER_KIND} items"
        in _FRAGMENT_SHAPE_BLOCK
    )
    assert (
        f"classes <= {MAX_LEAF_SYMBOL_ITEMS_PER_KIND} items"
        in _FRAGMENT_SHAPE_BLOCK
    )


def test_split_partial_schema_generations_are_current4_legacy1_dormant2() -> None:
    """`0.14.3` advances the current writable/executable split-partial
    container generation from 3 to 4; released schema 3 becomes an
    unsupported predecessor generation with no dedicated Python constant,
    because the existing current-schema equality check already rejects
    every non-4 value on its own (section 2A/16) -- adding one would have no
    production consumer. Legacy schema 1 and dormant schema 2 are unchanged."""
    assert SPLIT_PARTIAL_SCHEMA_VERSION == 4
    assert LEGACY_SPLIT_PARTIAL_SCHEMA_VERSION == 1
    assert DORMANT_SPLIT_PARTIAL_SCHEMA_VERSION == 2
    assert not any(
        "PREDECESSOR" in name and "SCHEMA" in name for name in dir(file_division)
    )
