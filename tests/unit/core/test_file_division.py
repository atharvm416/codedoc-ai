from __future__ import annotations

import collections
import json
import pathlib
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


def test_leaf_prompt_unit_value_renders_only_the_600_char_hint() -> None:
    """0.14.7 section 5.4: `_leaf_prompt_unit_value()` must never serialize
    more than `MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS` (600) signature
    characters, regardless of how long the full 2,000-character matching
    signature is -- two units sharing an identical leading 600 characters
    but differing only after that point must render byte-identical
    metadata, proving the raised response/parser bound cannot silently grow
    rendered prompt metadata."""
    shared_prefix = "s" * 600
    short_unit = SemanticUnitIdentity(
        unit_id="unit_" + "a" * 64,
        kind="function",
        qualified_name="fn",
        signature=shared_prefix,
        atom_ids=("atom_" + "b" * 64,),
        source_range=_source_range(),
    )
    long_unit = SemanticUnitIdentity(
        unit_id="unit_" + "a" * 64,
        kind="function",
        qualified_name="fn",
        signature=shared_prefix + ("x" * 1400),
        atom_ids=("atom_" + "b" * 64,),
        source_range=_source_range(),
    )
    assert len(long_unit.signature) == 2000
    assert short_unit.signature != long_unit.signature

    short_value = file_division._leaf_prompt_unit_value(short_unit)
    long_value = file_division._leaf_prompt_unit_value(long_unit)
    assert short_value == long_value
    assert len(short_value["signature"]) == 600


@requires_structure_pack
def test_signature_bound_of_600_vs_2000_produces_identical_chunk_topology() -> None:
    """0.14.7 section 4.8 / 5.4 / mutation check 12: the production-path
    counterexample. Thirty real declarations whose full parser-captured
    signatures are ~678 characters (just over the retired 600 bound) versus
    ~1,888 characters (well under the raised 2,000 bound, both truthful and
    untruncated) must plan to the identical source chunk count, identical
    per-chunk unit membership, and identical rendered metadata size --
    proving the raised full signature bound cannot silently change prompt
    packing or paid-call counts. `source_budget_chars` is large enough that
    only the leaf prompt metadata cap, never source length, can force a
    chunk boundary here."""

    def _declarations(param_count: int, count: int = 30) -> str:
        params = ", ".join(f"p{index:03d}: int" for index in range(param_count))
        return "\n".join(
            f"def fn_{index:03d}({params}) -> int:\n    return {index}\n"
            for index in range(count)
        )

    short_source = _declarations(60)
    long_source = _declarations(170)
    short_line = next(
        line for line in short_source.splitlines() if line.startswith("def fn_000")
    )
    long_line = next(
        line for line in long_source.splitlines() if line.startswith("def fn_000")
    )
    assert 600 < len(short_line) < 2000
    assert 600 < len(long_line) < 2000
    assert len(short_line) != len(long_line)

    short_plan = build_division_plan(
        rel_path="short.py", language="python", content=short_source,
        source_budget_chars=200_000,
    )
    long_plan = build_division_plan(
        rel_path="long.py", language="python", content=long_source,
        source_budget_chars=200_000,
    )
    assert short_plan.structural_mode == "syntax"
    assert long_plan.structural_mode == "syntax"
    assert len(short_plan.symbols) == 30
    assert len(long_plan.symbols) == 30
    # The real captured signatures genuinely differ in length -- otherwise
    # this would not be a counterexample at all.
    assert len(short_plan.symbols[0].signature) != len(long_plan.symbols[0].signature)

    assert len(short_plan.chunks) == len(long_plan.chunks)
    assert [len(c.semantic_units) for c in short_plan.chunks] == [
        len(c.semantic_units) for c in long_plan.chunks
    ]
    assert [c.close_reason for c in short_plan.chunks] == [
        c.close_reason for c in long_plan.chunks
    ]
    # Rendered metadata size is not required to be byte-identical between the
    # two files -- each declaration's source_range byte offsets are larger in
    # the long-signature file purely because the file itself is longer, and a
    # larger integer costs a few more JSON digits regardless of signature
    # length. What must hold is that this difference stays tiny (offset-digit
    # noise only): if the 600-character hint clamp were removed, an
    # unclamped ~1,888-character signature would cost roughly 1,288 more
    # characters *per unit* than the ~678-character one, so any real leak
    # would dwarf this bound by orders of magnitude.
    for short_chunk, long_chunk in zip(short_plan.chunks, long_plan.chunks):
        short_metadata_chars = file_division.leaf_prompt_metadata_chars(
            group_unit_id=short_chunk.unit_id,
            semantic_units=short_chunk.semantic_units,
            unit_indexes=short_plan.unit_positions(short_chunk),
            unit_count=len(short_plan.units),
            owning_ranges=short_chunk.owning_ranges,
        )
        long_metadata_chars = file_division.leaf_prompt_metadata_chars(
            group_unit_id=long_chunk.unit_id,
            semantic_units=long_chunk.semantic_units,
            unit_indexes=long_plan.unit_positions(long_chunk),
            unit_count=len(long_plan.units),
            owning_ranges=long_chunk.owning_ranges,
        )
        assert abs(short_metadata_chars - long_metadata_chars) < 200, (
            short_metadata_chars, long_metadata_chars,
        )


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

    # section 5.6's canonical example: one indivisible 2,010-character span at
    # B=1000 must locally balance to 670/670/670, never the old fixed-window
    # 1000/1000/10 tail (section 4.6's root-cause defect).
    oversized = "x" * 2010
    plan = build_division_plan(
        rel_path="oversized.py", language="unknown", content=oversized, source_budget_chars=1000
    )
    assert [chunk.payload_chars for chunk in plan.chunks] == [670, 670, 670]
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
    assert all(c.close_reason == "continuation" for c in plan.chunks)
    assert plan.chunks[0].start_boundary == "file-start"
    assert plan.chunks[0].end_boundary == "balanced-codepoint"
    assert plan.chunks[1].start_boundary == "balanced-codepoint"
    assert plan.chunks[1].end_boundary == "balanced-codepoint"
    assert plan.chunks[2].start_boundary == "balanced-codepoint"
    assert plan.chunks[2].end_boundary == "file-end"


# ---------------------------------------------------------------------------
# Local boundary-aware subdivision algorithm (section 5.6)
# ---------------------------------------------------------------------------


def test_local_subdivision_splitline_scanner_matches_python_oracle() -> None:
    """The one-pass physical-line boundary scanner must reproduce exactly
    what `str.splitlines(keepends=True)` would cut on, for every separator
    Python recognizes (CR, LF, CRLF, vertical tab, form feed, file/group/
    record separators, NEL, Unicode line/paragraph separator) including
    mixed and random fixtures. Reconstructing lines from the scanner's
    boundaries and comparing to the real `splitlines()` output is the
    required oracle comparison."""

    def lines_from_boundaries(text: str, boundaries: tuple[int, ...]) -> list[str]:
        points = [0] + list(boundaries)
        if not boundaries or boundaries[-1] != len(text):
            points.append(len(text))
        return [
            text[points[i] : points[i + 1]]
            for i in range(len(points) - 1)
            if points[i + 1] > points[i]
        ]

    explicit_cases = [
        "",
        "a",
        "\r\n",
        "\r\n\r\n",
        "a\r\nb\r\n",
        "a\rb\nc\r\nd",
        "\r",
        "\n",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
        "no separators at all",
        "trailing\r\n",
        "\r\ntrailing",
        "mix\vand\fmatch\x1cthese\x1dup\x1etogether\x85please\u2028thanks\u2029.",
    ]
    for text in explicit_cases:
        boundaries = file_division._splitline_boundary_offsets(text)
        assert lines_from_boundaries(text, boundaries) == text.splitlines(keepends=True), text

    import random

    generator = random.Random(20260825)
    separators = ["a", "b", "\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
    for _ in range(2000):
        length = generator.randint(0, 40)
        text = "".join(generator.choice(separators) for _ in range(length))
        boundaries = file_division._splitline_boundary_offsets(text)
        assert lines_from_boundaries(text, boundaries) == text.splitlines(keepends=True), text


def test_local_subdivision_preferred_window_ties_and_fractional_edges() -> None:
    """Section 5.6 point 4: the inclusive integer tolerance window is
    `max(1, target // 10)`; a candidate exactly at the window edge qualifies,
    one code point outside does not; equal-distance candidates resolve to the
    earlier source offset; syntax always outranks a physical-line candidate
    at the same distance."""
    # L=1900, B=1000 -> target=950, tolerance=95, window=[855,1045].
    span = "z" * 1900
    # equal-distance tie (20 away each side of 950): earlier wins.
    cuts = file_division._local_subdivision_cuts(span, 1000, frozenset({930, 970}))
    assert cuts[0] == (930, "syntax")

    # L=1500, B=1000 -> target=750, tolerance=75, window=[675,825], and here
    # the window sits entirely inside the feasible/budget-capped admissible
    # range [500,1000], so the window edges themselves are the binding
    # constraint (not the feasibility floor, unlike the L=1900 case above).
    span_slack = "z" * 1500
    cuts_edge = file_division._local_subdivision_cuts(span_slack, 1000, frozenset({675}))
    assert cuts_edge[0] == (675, "syntax")
    cuts_edge_hi = file_division._local_subdivision_cuts(span_slack, 1000, frozenset({825}))
    assert cuts_edge_hi[0] == (825, "syntax")

    # one code point outside either edge must be rejected; falls back to
    # balanced-codepoint at the unconstrained target (750) instead.
    cuts_outside_lo = file_division._local_subdivision_cuts(span_slack, 1000, frozenset({674}))
    assert cuts_outside_lo[0] == (750, "balanced-codepoint")
    cuts_outside_hi = file_division._local_subdivision_cuts(span_slack, 1000, frozenset({826}))
    assert cuts_outside_hi[0] == (750, "balanced-codepoint")

    # syntax beats an equal-distance physical-line candidate: line boundary
    # at 941 (10 below target 950), syntax candidate at 960 (10 above).
    span_tie = "a" * 940 + "\n" + "b" * 959
    assert len(span_tie) == 1900
    cuts_tie = file_division._local_subdivision_cuts(span_tie, 1000, frozenset({960}))
    assert cuts_tie[0] == (960, "syntax")


def test_local_subdivision_no_qualifying_boundary_falls_back_exactly() -> None:
    """With zero slack (L == 2B), only the exact midpoint is admissible, so
    no preferred-tier search can ever qualify; the fallback lands there."""
    cuts = file_division._local_subdivision_cuts("w" * 2000, 1000)
    assert cuts == ((1000, "balanced-codepoint"),)


def test_local_subdivision_b_and_2b_boundary_edges() -> None:
    """B-1/B/B+1 and 2B-1/2B/2B+1 piece-count edges (section 9.1 item 4)."""
    assert file_division._local_subdivision_cuts("q" * 999, 1000) == ()
    assert file_division._local_subdivision_cuts("q" * 1000, 1000) == ()
    cuts_bp1 = file_division._local_subdivision_cuts("q" * 1001, 1000)
    assert len(cuts_bp1) == 1

    import math

    for length, expected_pieces in ((1999, 2), (2000, 2), (2001, 3)):
        span = "q" * length
        cuts = file_division._local_subdivision_cuts(span, 1000)
        assert len(cuts) + 1 == expected_pieces == math.ceil(length / 1000)
        bounds = [0] + [c for c, _ in cuts] + [length]
        sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        assert sum(sizes) == length
        assert all(size <= 1000 for size in sizes)


def test_local_subdivision_defensive_crlf_atomicity_extra_piece() -> None:
    """Section 5.6: a 2,000-character span with a CRLF pair straddling the
    only naive 1,000/1,000 cut needs a minimum safe piece count of 3, not the
    unsafe arithmetic lower bound of 2; no piece may end/start inside the
    pair; a normal (CRLF-free) canonical-load span of the same length must
    keep the arithmetic minimum with zero atomicity delta."""
    straddled = ("a" * 999) + "\r\n" + ("b" * 999)
    assert len(straddled) == 2000
    cuts = file_division._local_subdivision_cuts(straddled, 1000)
    assert len(cuts) == 2  # 3 pieces
    bounds = [0] + [c for c, _ in cuts] + [len(straddled)]
    for bound in bounds:
        assert bound != 1000, "must never cut inside the CRLF pair"
    sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    assert sizes == [667, 667, 666]
    assert sum(sizes) == 2000

    normal = "a" * 2000
    normal_cuts = file_division._local_subdivision_cuts(normal, 1000)
    assert len(normal_cuts) == 1  # exactly ceil(2000/1000) == 2 pieces


def test_local_subdivision_defensive_crlf_stranded_suffix_adversary() -> None:
    """Section 5.6 / mutation check 17: a 2,900-character span at B=1,000
    with a syntax candidate at 900 is in-window for the first cut (target
    967, tolerance 96) but must be rejected, because cutting there leaves a
    2,000-character remainder whose only exact 2-piece split point (at
    absolute offset 1,900) sits inside a CRLF pair -- the suffix-feasibility
    bound, not merely the window, must reject it."""
    span = ("a" * 1899) + "\r\n" + ("b" * 999)
    assert len(span) == 2900
    cuts = file_division._local_subdivision_cuts(span, 1000, frozenset({900}))
    assert 900 not in [offset for offset, _ in cuts]
    bounds = [0] + [c for c, _ in cuts] + [len(span)]
    sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    assert sum(sizes) == 2900
    assert all(size <= 1000 for size in sizes)
    assert 1900 not in bounds


def test_canonical_filesystem_crlf_normalization_is_composed_with_division(
    tmp_path,
) -> None:
    """Section 5.6 / 9.1 items 4-5 / workstream 0: the *normal* filesystem
    pipeline strips CRLF before split planning, so a canonically loaded source
    plans with zero CRLF-atomicity delta. This test composes the real
    production loader (`read_source_snapshot`) with `build_division_plan`
    rather than comparing against an already-normalized literal -- and it does
    not touch production loader code.
    """
    from codedoc.core.db import read_source_snapshot

    # Contrast: fed *unnormalized* into the pure algorithm, a CRLF straddling
    # the sole arithmetic 1,000 cut of a 2,000-char span costs a defensive
    # extra piece (k_safe 3 > ceil(L/B) 2); a CR-free span of the same length
    # takes the arithmetic minimum of 2.
    straddled = ("a" * 999) + "\r\n" + ("a" * 999)
    assert len(file_division._local_subdivision_cuts(straddled, 1000)) + 1 == 3
    assert len(file_division._local_subdivision_cuts("a" * 2000, 1000)) + 1 == 2

    # Raw bytes: a BOM, a CRLF, a lone CR, and one big line whose CRLF-adjacent
    # byte length (2,010) would straddle an arithmetic cut if the raw bytes
    # reached planning unnormalized.
    raw = (
        b"\xef\xbb\xbf"  # UTF-8 BOM
        + b"header = 1\r\n"
        + b"lone\rcr\r\n"
        + (b"w" * 2010)
        + b"\r\n"
        + b"trailer = 2\r\n"
    )
    source_file = tmp_path / "crlf_source.py"
    source_file.write_bytes(raw)

    content_hash, content = read_source_snapshot(source_file)

    # 3: the content hash still binds the *original raw bytes*.
    import hashlib

    assert content_hash == hashlib.sha256(raw).hexdigest()
    # 4: the canonical decoded content has no carriage return at all, and the
    # leading BOM is gone.
    assert "\r" not in content
    assert not content.startswith("﻿")
    assert "header = 1\n" in content
    assert "lone\ncr\n" in content

    # 5-6: division reconstructs the canonical *decoded* bytes exactly.
    plan = build_division_plan(
        rel_path="crlf_source.py",
        language="unknown",
        content=content,
        source_budget_chars=1000,
    )
    decoded_bytes = content.encode("utf-8")
    assert b"".join(c.payload.encode("utf-8") for c in plan.chunks) == decoded_bytes

    # 7: the one oversized unit ("w" * 2010) plans to the arithmetic minimum;
    # there is no defensive CRLF-atomicity extra piece anywhere.
    oversized = [c for c in plan.chunks if c.unit_chunk_count > 1]
    assert oversized, "the 2,010-char line must be one oversized unit"
    unit_chars = sum(c.payload_chars for c in oversized)
    assert len(oversized) == -(-unit_chars // 1000) == 3  # exact ceil, no +1

    # 8: no chunk boundary splits a UTF-8 code point (every owning range
    # endpoint round-trips through decode).
    offset = 0
    for chunk in plan.chunks:
        for source_range in chunk.owning_ranges:
            assert source_range.start_byte == offset
            decoded_bytes[source_range.start_byte : source_range.end_byte].decode("utf-8")
            offset = source_range.end_byte
    assert offset == len(decoded_bytes)


def test_fitting_semantic_spans_are_never_subdivided_or_redistributed() -> None:
    """Section 5.6 / 9.1 item 2: the plan's exact worked example -- natural
    fitting spans of 672, 495, and 843 canonical code points at B=1,000.
    None is individually oversized, so none is subdivided; and no two of them
    can co-pack (672+495=1167 > 1,000 and 495+843=1338 > 1,000), so each keeps
    its own leaf, canonical bytes, byte range, and unit identity unchanged --
    balancing must never redistribute source across a fitting-unit boundary."""
    spans = ("a" * 671 + "\n", "b" * 494 + "\n", "c" * 842 + "\n")
    assert [len(span) for span in spans] == [672, 495, 843]
    content = "".join(spans)
    assert len(content) == 2010

    plan = build_division_plan(
        rel_path="spans.py", language="unknown", content=content, source_budget_chars=1000
    )
    assert len(plan.units) == 3
    assert [c.unit_chunk_count for c in plan.chunks] == [1, 1, 1]
    assert [len(c.semantic_units) for c in plan.chunks] == [1, 1, 1]
    assert [c.payload_chars for c in plan.chunks] == [672, 495, 843]
    assert [c.payload for c in plan.chunks] == list(spans)
    # exact canonical byte ranges and unit identities are retained one-for-one.
    data = content.encode("utf-8")
    offset = 0
    for span, chunk, unit in zip(spans, plan.chunks, plan.units):
        (source_range,) = chunk.owning_ranges
        assert (source_range.start_byte, source_range.end_byte) == (
            offset,
            offset + len(span.encode("utf-8")),
        )
        assert chunk.semantic_units[0].unit_id == unit.unit_id
        assert chunk.semantic_units[0].source_range == unit.source_range
        assert data[source_range.start_byte : source_range.end_byte].decode("utf-8") == span
        offset = source_range.end_byte
    assert offset == len(data)
    assert "".join(c.payload for c in plan.chunks) == content
    # a second identical build is byte-, identity-, and digest-stable.
    again = build_division_plan(
        rel_path="spans.py", language="unknown", content=content, source_budget_chars=1000
    )
    assert again == plan
    assert again.plan_digest == plan.plan_digest


def test_only_the_oversized_span_subdivides_and_fitting_units_still_co_pack() -> None:
    """Section 5.6 / 9.1 item 3: in 1243/482/285 at B=1,000, only the 1,243
    unit is subdivided (near 622/621); the 482 and 285 units retain exact
    canonical bytes/ranges/identity and still co-pack into one 767-char call,
    exactly as the plan's worked example states. Identity retention is proved
    against a same-content/same-path *control* plan whose larger budget leaves
    all three units unsplit -- so the 482/285 evidence does not come only from
    the derived `plan.units` of the plan under test."""
    lines = ["a" * 1242, "b" * 481, "c" * 284]  # +1 each for the trailing "\n"
    content = "\n".join(lines) + "\n"
    plan = build_division_plan(
        rel_path="mixed.py", language="unknown", content=content, source_budget_chars=1000
    )
    control = build_division_plan(
        rel_path="mixed.py", language="unknown", content=content, source_budget_chars=5000
    )

    assert [c.payload_chars for c in plan.chunks] == [622, 621, 767]
    assert [c.unit_chunk_count for c in plan.chunks] == [2, 2, 1]
    assert "".join(c.payload for c in plan.chunks) == content
    # control: 1,243 not subdivided; the whole file is one co-packed leaf.
    assert [c.unit_chunk_count for c in control.chunks] == [1]
    assert len(control.atoms) == len(plan.atoms) == 3

    data = content.encode("utf-8")
    # authoritative extraction (plan.atoms), independent of chunk packing:
    # the 482 and 285 atoms are byte-, range-, and id-identical in both plans,
    # in the same order.
    for index in (1, 2):
        test_atom = plan.atoms[index]
        control_atom = control.atoms[index]
        assert test_atom.atom_id == control_atom.atom_id
        assert test_atom.range == control_atom.range
        assert test_atom.source == control_atom.source
        assert data[
            test_atom.range.start_byte : test_atom.range.end_byte
        ].decode("utf-8") == test_atom.source
    assert [a.atom_id for a in plan.atoms] == [a.atom_id for a in control.atoms]

    # the co-packed 767-char leaf carries units #2 and #3 with identical
    # unit_id, atom IDs, and authoritative range as the control plan's units.
    co_packed = plan.chunks[2]
    assert len(co_packed.semantic_units) == 2
    for offset, unit in enumerate(co_packed.semantic_units):
        control_unit = control.units[offset + 1]
        assert unit.unit_id == control_unit.unit_id
        assert unit.atom_ids == control_unit.atom_ids
        assert unit.source_range == control_unit.source_range
        assert unit.qualified_name == control_unit.qualified_name
        assert unit.kind == control_unit.kind
    assert co_packed.payload_chars == 767  # 482 + 285 co-pack allowed at B=1000


@requires_structure_pack
def test_nested_declaration_becomes_a_real_syntax_boundary_candidate() -> None:
    """Section 5.6 point 4: the syntax tier is populated from real nested
    `SymbolFact` ranges strictly inside an oversized owning span -- not only
    from synthetic offsets injected directly into the pure algorithm. A large
    class with one nested method declaration positioned so its start/end
    falls inside the balance window must be preferred over a physical-line or
    balanced-codepoint cut at the same distance."""
    padding_before = "z" * 480
    padding_after = "z" * 480
    source = (
        "class Big:\n"
        f"    filler_before = '{padding_before}'\n"
        "    def inner(self):\n"
        "        return 1\n"
        f"    filler_after = '{padding_after}'\n"
    )
    plan = build_division_plan(
        rel_path="nested.py", language="python", content=source, source_budget_chars=600
    )
    assert len(plan.units) == 1, "the whole class body is one oversized unit"
    assert any(c.unit_chunk_count > 1 for c in plan.chunks)
    boundary_kinds = {c.start_boundary for c in plan.chunks} | {
        c.end_boundary for c in plan.chunks
    }
    assert "syntax" in boundary_kinds, boundary_kinds
    assert "".join(c.payload for c in plan.chunks) == source


@requires_structure_pack
def test_multiline_oversized_unit_packs_complete_lines_near_budget() -> None:
    """0.14.7 section 5.6: local subdivision balances an oversized syntax
    unit near an even target rather than filling every piece to the budget
    and leaving a small fixed-window tail. With densely available physical-
    line boundaries every ~30 characters, the achieved piece count must still
    equal the arithmetic minimum -- balancing must not manufacture extra
    calls -- and every piece stays a genuine line-bounded fraction of the
    target, never a tiny remainder."""
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
    assert "".join(chunk.payload for chunk in plan.chunks) == source
    assert all(chunk.payload_chars <= budget for chunk in plan.chunks)
    assert all(chunk.payload.endswith("\n") for chunk in plan.chunks)

    oversized_unit_chunks = [c for c in plan.chunks if c.unit_chunk_count > 1]
    assert oversized_unit_chunks, "the function body must be one oversized unit"
    unit_chars = sum(c.payload_chars for c in oversized_unit_chunks)
    minimum_pieces = -(-unit_chars // budget)
    assert len(oversized_unit_chunks) == minimum_pieces
    target = unit_chars / minimum_pieces
    assert all(
        chunk.payload_chars > target / 2 for chunk in oversized_unit_chunks
    ), [c.payload_chars for c in oversized_unit_chunks]
    assert all(
        chunk.close_reason == "continuation" for chunk in oversized_unit_chunks
    )
    assert all(
        chunk.start_boundary in ("physical-line", "file-start", "semantic-unit")
        for chunk in oversized_unit_chunks
    )


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


# ---------------------------------------------------------------------------
# Section 5.6 core: exact topology, close-reason precedence, boundary
# descriptors, half-open ownership, and repeat determinism
# ---------------------------------------------------------------------------


def _one_oversized_atom_structure(content: str, nested_symbols=()):
    """Build a `StructureResult` whose single syntax atom spans the whole
    canonical source, plus any requested strictly-nested symbols. Suitable for
    ``monkeypatch.setattr(file_division, "extract_structure", ...)`` so a test
    can place an exact nested-symbol range without depending on a real grammar.

    ``nested_symbols`` is an iterable of ``(qualified_name, start_byte,
    end_byte)`` tuples.
    """
    from codedoc.core.file_division import SourceIndex
    from codedoc.parser.source_structure import (
        Atom,
        StructureResult,
        SymbolFact,
        atom_id_for,
        symbol_id_for,
    )

    def _factory(rel_path, language, source, **_kwargs):
        index = SourceIndex(source)
        total = len(index.data)
        atom_kind = "class"
        the_atom_id = atom_id_for(rel_path, atom_kind, 0, total)
        symbols = tuple(
            SymbolFact(
                symbol_id=symbol_id_for(rel_path, "method", name, start, end),
                rel_path=rel_path,
                language=language,
                kind="method",
                qualified_name=name,
                signature="def " + name.split(".")[-1] + "()",
                range=index.range(start, end),
                atom_id=the_atom_id,
            )
            for name, start, end in nested_symbols
        )
        the_atom = Atom(
            atom_id=the_atom_id,
            rel_path=rel_path,
            language=language,
            kind=atom_kind,
            name="Whole",
            range=index.range(0, total),
            source=source,
            symbol_ids=tuple(symbol.symbol_id for symbol in symbols),
        )
        return StructureResult("syntax", (the_atom,), symbols, ())

    return _factory


def test_oversized_2010_span_exact_pieces_ranges_and_five_call_topology() -> None:
    """Section 5.6 / 9.1 item 1 / workstream 0: one indivisible 2,010-character
    span at B=1,000 balances to exactly 670/670/670 -- never the retired
    1,000/1,000/10 fixed-window tail -- with exact byte ranges, exact
    `SourceIndex.data` reconstruction, continuation flags/indexes, boundary
    kinds, `close_reason`, digest-stable repeats, and the exact initial
    topology of 3 leaves + 1 unit consolidation + 1 final = 5 provider calls.
    """
    content = "x" * 2010
    plan = build_division_plan(
        rel_path="oversized.py",
        language="unknown",
        content=content,
        source_budget_chars=1000,
    )

    assert len(plan.units) == 1
    assert [c.payload_chars for c in plan.chunks] == [670, 670, 670]
    assert "".join(c.payload for c in plan.chunks) == content
    assert b"".join(
        c.payload.encode("utf-8") for c in plan.chunks
    ) == content.encode("utf-8")
    ranges = [(r.start_byte, r.end_byte) for c in plan.chunks for r in c.owning_ranges]
    assert ranges == [(0, 670), (670, 1340), (1340, 2010)]
    data = content.encode("utf-8")
    for (start, end), chunk in zip(ranges, plan.chunks):
        assert data[start:end].decode("utf-8") == chunk.payload

    assert [c.unit_chunk_index for c in plan.chunks] == [0, 1, 2]
    assert all(c.unit_chunk_count == 3 for c in plan.chunks)
    assert [
        (c.continuation_before, c.continuation_after) for c in plan.chunks
    ] == [(False, True), (True, True), (True, False)]
    assert [c.close_reason for c in plan.chunks] == ["continuation"] * 3
    assert [c.start_boundary for c in plan.chunks] == [
        "file-start",
        "balanced-codepoint",
        "balanced-codepoint",
    ]
    assert [c.end_boundary for c in plan.chunks] == [
        "balanced-codepoint",
        "balanced-codepoint",
        "file-end",
    ]
    # one internal cut kind, shown identically on both adjacent descriptors.
    for left, right in zip(plan.chunks, plan.chunks[1:]):
        assert left.end_boundary == right.start_boundary

    again = build_division_plan(
        rel_path="oversized.py",
        language="unknown",
        content=content,
        source_budget_chars=1000,
    )
    assert again == plan
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in plan.chunks]
    assert again.plan_digest == plan.plan_digest

    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    assert len(tree.unit_consolidation_nodes) == 1
    assert len(tree.general_nodes) == 0
    consolidation = tree.unit_consolidation_nodes[0]
    assert consolidation.phase == "unit-consolidation"
    assert set(consolidation.child_ids) == {c.chunk_id for c in plan.chunks}
    assert tree.final_node.child_ids == (consolidation.node_id,)
    leaf_calls = len(plan.chunks)
    intermediate_calls = len(tree.all_intermediate_nodes)
    assert (leaf_calls, intermediate_calls, 1) == (3, 1, 1)
    assert leaf_calls + intermediate_calls + 1 == 5
    assert build_reduction_tree(
        again, synthesis_manifest_chars=12000
    ).tree_digest == tree.tree_digest


def test_build_reduction_tree_synthesis_budget_keyword_contract() -> None:
    """Section 5.7 / 9.1 item 7: the effective synthesis budget enters
    ``build_reduction_tree`` through the preferred ``synthesis_manifest_chars``
    keyword; the deprecated ``max_content_chars`` alias is retained only for
    direct callers; neither and both are rejected; the chosen value is
    normalized once, stored on ``ReductionTreePlan.synthesis_manifest_chars``,
    and bound into ``tree_digest``."""
    plan = build_division_plan(
        rel_path="keyword.py",
        language="unknown",
        content="x" * 2010,
        source_budget_chars=1000,
    )

    with pytest.raises(ValueError, match="exactly one of synthesis_manifest_chars"):
        build_reduction_tree(plan)
    with pytest.raises(ValueError, match="exactly one of synthesis_manifest_chars"):
        build_reduction_tree(
            plan, synthesis_manifest_chars=12000, max_content_chars=12000
        )

    preferred = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    assert preferred.synthesis_manifest_chars == 12000

    # The deprecated alias still works for a direct caller and, given the same
    # value, produces a byte-identical tree -- the value is normalized once,
    # not once per keyword name.
    aliased = build_reduction_tree(plan, max_content_chars=12000)
    assert aliased.synthesis_manifest_chars == 12000
    assert aliased.tree_digest == preferred.tree_digest

    # The value is bound into the digest: a different budget changes the digest
    # (and, for this single-unit three-chunk plan, only the digest), never the
    # node identities.
    other = build_reduction_tree(plan, synthesis_manifest_chars=13000)
    assert other.synthesis_manifest_chars == 13000
    assert other.tree_digest != preferred.tree_digest
    assert [n.node_id for n in other.all_intermediate_nodes] == [
        n.node_id for n in preferred.all_intermediate_nodes
    ]
    assert other.final_node.node_id == preferred.final_node.node_id


def test_automatic_synthesis_floor_plans_the_maximum_chunk_and_unit_allocations() -> None:
    """Section 5.7 / 9.1 item 7: under the automatic 12,000 synthesis floor,
    both the maximum chunk allocation (one oversized unit divided into exactly
    ``MAX_CHUNKS_PER_FILE`` leaves) and the maximum unit allocation
    (``MAX_UNITS_PER_FILE`` tiny lexical units) build a complete reduction tree
    -- within ``MAX_REDUCTION_TREE_DEPTH``, covering every leaf exactly once,
    with no envelope/fan-in/depth capacity block."""
    max_depth = file_division.MAX_REDUCTION_TREE_DEPTH

    chunk_cap_plan = build_division_plan(
        rel_path="chunkcap.py",
        language="unknown",
        content="x" * (1000 * MAX_CHUNKS_PER_FILE),
        source_budget_chars=1000,
    )
    assert len(chunk_cap_plan.units) == 1
    assert len(chunk_cap_plan.chunks) == MAX_CHUNKS_PER_FILE
    chunk_cap_tree = build_reduction_tree(
        chunk_cap_plan, synthesis_manifest_chars=12000
    )
    assert reduction_depth(chunk_cap_tree) <= max_depth
    assert tuple(sorted(chunk_cap_tree.final_node.leaf_ids)) == tuple(
        sorted(chunk.chunk_id for chunk in chunk_cap_plan.chunks)
    )
    assert len(set(chunk_cap_tree.final_node.leaf_ids)) == MAX_CHUNKS_PER_FILE

    unit_cap_plan = build_division_plan(
        rel_path="unitcap.txt",
        language="unknown",
        content="x;\n" * file_division.MAX_UNITS_PER_FILE,
        source_budget_chars=12000,
    )
    assert unit_cap_plan.structural_mode == "lexical"
    assert len(unit_cap_plan.units) == file_division.MAX_UNITS_PER_FILE
    assert len(unit_cap_plan.chunks) <= MAX_CHUNKS_PER_FILE
    unit_cap_tree = build_reduction_tree(
        unit_cap_plan, synthesis_manifest_chars=12000
    )
    assert reduction_depth(unit_cap_tree) <= max_depth
    assert tuple(sorted(unit_cap_tree.final_node.leaf_ids)) == tuple(
        sorted(chunk.chunk_id for chunk in unit_cap_plan.chunks)
    )
    assert len(set(unit_cap_tree.final_node.leaf_ids)) == len(unit_cap_plan.chunks)


def test_automatic_floor_still_fails_a_genuinely_oversized_authoritative_final_field() -> None:
    """Section 5.7 / 9.1 item 7: the automatic floor removes the *starved source
    budget* block, not genuine bounded failures. A plan built at a healthy
    source budget whose real authoritative imports alone push the final
    manifest past the carried 12,000 ceiling still fails closed with
    ``final-synthesis-envelope-cap``."""
    content = "\n".join(f"CONSTANT_{index} = {index}" for index in range(1500)) + "\n"
    plan = build_division_plan(
        rel_path="authoritative_final.py",
        language="python",
        content=content,
        source_budget_chars=12000,
    )
    assert len(plan.chunks) >= 2

    # Baseline: with no imports the same plan at the same 12,000 budget builds.
    baseline = build_reduction_tree(
        plan, synthesis_manifest_chars=12000, language="python", imports=()
    )
    assert baseline.synthesis_manifest_chars == 12000

    oversized_imports = tuple(
        f"package.subpackage.module_{index:05d}" for index in range(400)
    )
    assert (
        worst_case_final_synthesis_chars(
            rel_path="authoritative_final.py",
            language="python",
            imports=oversized_imports,
            root_count=1,
            leaf_count=len(plan.chunks),
            max_chars=12000,
        )
        > 12000
    )
    with pytest.raises(SplitCapacityBlocked) as blocked:
        build_reduction_tree(
            plan,
            synthesis_manifest_chars=12000,
            language="python",
            imports=oversized_imports,
        )
    assert blocked.value.reason == "final-synthesis-envelope-cap"


@pytest.mark.parametrize("budget", [12000, 20000])
def test_source_budget_at_or_above_the_floor_is_carried_unchanged(budget) -> None:
    """Section 5.7 / 9.1 item 7: for a source budget already at or above the
    12,000 floor, ``max(budget, 12000) == budget``; the preferred keyword
    reproduces the prior raw-budget (deprecated-alias) topology exactly -- same
    effective ceiling, fan-in, node structure, call count, and digest."""
    assert (
        max(budget, file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS) == budget
    )
    plan = build_division_plan(
        rel_path="atfloor.py",
        language="unknown",
        content="x" * 2010,
        source_budget_chars=1000,
    )
    preferred = build_reduction_tree(plan, synthesis_manifest_chars=budget)
    prior = build_reduction_tree(plan, max_content_chars=budget)

    assert preferred.synthesis_manifest_chars == budget
    assert prior.synthesis_manifest_chars == budget
    assert preferred.max_fan_in == prior.max_fan_in
    assert preferred.tree_digest == prior.tree_digest
    assert [n.node_id for n in preferred.all_intermediate_nodes] == [
        n.node_id for n in prior.all_intermediate_nodes
    ]
    assert preferred.final_node.child_ids == prior.final_node.child_ids
    leaf_calls = len(plan.chunks)
    assert leaf_calls + len(preferred.all_intermediate_nodes) + 1 == 5


def test_local_subdivision_repeated_plans_are_fully_identical() -> None:
    """Section 5.6 / 9.1 item 5 / mutation-check baseline: two independent
    builds of a subdivided plan agree on bytes, ranges, chunk/unit IDs,
    boundary + closure metadata, the plan digest, and the reduction call
    count -- determinism is a hard requirement, not incidental."""
    lines = ["a" * 1242, "b" * 481, "c" * 284]
    content = "\n".join(lines) + "\n"
    first = build_division_plan(
        rel_path="mixed.py", language="unknown", content=content, source_budget_chars=1000
    )
    second = build_division_plan(
        rel_path="mixed.py", language="unknown", content=content, source_budget_chars=1000
    )
    assert first == second
    assert first.plan_digest == second.plan_digest
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
    assert [u.unit_id for u in first.units] == [u.unit_id for u in second.units]
    assert [
        (c.payload_chars, c.close_reason, c.start_boundary, c.end_boundary)
        for c in first.chunks
    ] == [
        (c.payload_chars, c.close_reason, c.start_boundary, c.end_boundary)
        for c in second.chunks
    ]
    assert [c.payload_chars for c in first.chunks] == [622, 621, 767]
    tree_a = build_reduction_tree(first, synthesis_manifest_chars=12000)
    tree_b = build_reduction_tree(second, synthesis_manifest_chars=12000)
    assert tree_a.tree_digest == tree_b.tree_digest
    assert len(tree_a.all_nodes) == len(tree_b.all_nodes)


def test_nested_syntax_candidate_bytes_strict_nesting_excludes_owner_edges() -> None:
    """Section 5.6 point 4: the syntax tier is the deduplicated start *and*
    end byte offsets of every symbol whose authoritative range is *strictly*
    nested inside the owning span; a symbol sharing the owner's own outer
    start or end never contributes that shared edge, and a symbol owned by a
    different atom is ignored entirely."""
    from codedoc.parser.source_structure import (
        Atom,
        SourceRange,
        SymbolFact,
        atom_id_for,
        symbol_id_for,
    )

    rel = "x.py"

    def rng(start, end):
        return SourceRange(start, end, 1, 1, 1, 1)

    owner = Atom(
        atom_id=atom_id_for(rel, "class", 0, 1000),
        rel_path=rel,
        language="python",
        kind="class",
        name="Big",
        range=rng(0, 1000),
        source="z" * 1000,
        symbol_ids=(),
    )

    def sym(name, start, end, atom_id):
        return SymbolFact(
            symbol_id=symbol_id_for(rel, "method", name, start, end),
            rel_path=rel,
            language="python",
            kind="method",
            qualified_name=name,
            signature="def m()",
            range=rng(start, end),
            atom_id=atom_id,
        )

    strictly_inside = sym("Big.m", 200, 400, owner.atom_id)
    shares_owner_start = sym("Big.s", 0, 300, owner.atom_id)
    shares_owner_end = sym("Big.e", 700, 1000, owner.atom_id)
    foreign = sym("Other.x", 120, 160, atom_id_for(rel, "class", 4000, 5000))

    candidates = file_division._nested_syntax_candidate_bytes(
        owner, [strictly_inside, shares_owner_start, shares_owner_end, foreign]
    )
    # {200, 400} from the strictly-nested symbol; 300 (inner end of the
    # start-sharing symbol) and 700 (inner start of the end-sharing symbol);
    # never 0 or 1000 (the owner's own outer edges); never a foreign offset.
    assert sorted(candidates) == [200, 300, 400, 700]


@requires_structure_pack
def test_syntax_boundary_wins_when_it_coincides_with_a_physical_line_offset() -> None:
    """Section 5.6 point 4: if one offset belongs to both the syntax and the
    physical-line tier, syntax wins -- the cut is reported as ``syntax``."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        # A newline at code point 669 -> physical-line boundary at 670, which is
        # also the balanced target of the first cut (ceil(2010/3)); place a
        # strictly-nested symbol starting at 670 so the offset is in both tiers.
        content = "a" * 669 + "\n" + "z" * 1340
        assert len(content) == 2010
        monkeypatch.setattr(
            file_division,
            "extract_structure",
            _one_oversized_atom_structure(content, [("Whole.inner", 670, 900)]),
        )
        plan = build_division_plan(
            rel_path="coin.py",
            language="python",
            content=content,
            source_budget_chars=1000,
        )
    finally:
        monkeypatch.undo()
    assert len(plan.units) == 1
    assert plan.chunks[0].payload_chars == 670
    assert plan.chunks[0].end_boundary == "syntax"
    assert plan.chunks[1].start_boundary == "syntax"
    assert "".join(c.payload for c in plan.chunks) == content


def test_pack_chunks_close_reason_precedence_isolation_continuation_source_eof(
    monkeypatch,
) -> None:
    """Section 5.6 closure precedence: a pending fitting unit flushed before an
    individually oversized unit closes ``oversized-unit-isolation``; every
    piece of the oversized unit closes ``continuation`` (including its first
    and last); a later source-only overflow closes ``source-ceiling``; and the
    trailing leaf closes ``end-of-file``. File edges outrank coincident
    semantic-unit edges."""
    content = (
        ("k" * 400 + "\n")
        + ("x" * 2010 + "\n")
        + ("y" * 900 + "\n")
        + ("z" * 900 + "\n")
    )
    plan = build_division_plan(
        rel_path="reasons.py",
        language="unknown",
        content=content,
        source_budget_chars=1000,
    )
    assert [c.close_reason for c in plan.chunks] == [
        "oversized-unit-isolation",
        "continuation",
        "continuation",
        "continuation",
        "source-ceiling",
        "end-of-file",
    ]
    assert [c.unit_chunk_count for c in plan.chunks] == [1, 3, 3, 3, 1, 1]
    assert plan.chunks[0].start_boundary == "file-start"
    assert plan.chunks[0].end_boundary == "semantic-unit"
    # first continuation piece is not at file byte zero and has no internal cut
    # before it -> "semantic-unit"; its last piece is followed by more fitting
    # units, so that edge is the unit's own -> "semantic-unit", not "file-end".
    assert plan.chunks[1].start_boundary == "semantic-unit"
    assert plan.chunks[3].end_boundary == "semantic-unit"
    assert plan.chunks[-1].end_boundary == "file-end"
    for left, right in zip(plan.chunks[1:4], plan.chunks[2:4]):
        assert left.end_boundary == right.start_boundary
        assert left.end_boundary == "balanced-codepoint"
    assert "".join(c.payload for c in plan.chunks) == content


def test_pack_chunks_metadata_only_and_source_and_metadata_ceilings(
    monkeypatch,
) -> None:
    """Section 5.6 closure precedence: when the *next* fitting unit is
    considered, source and rendered-metadata overflow are computed
    independently -- metadata only -> ``metadata-ceiling``; both at once ->
    ``source-and-metadata-ceiling``. An exact ceiling value still fits and the
    closure reason is set by the following unit that overflows."""
    # metadata-only: source is trivially small, but a tight prompt-metadata cap
    # forces every unit onto its own leaf.
    monkeypatch.setattr(file_division, "MAX_LEAF_PROMPT_METADATA_CHARS", 900)
    meta_content = "".join(f"m{index} = {index}\n" for index in range(6))
    meta_plan = build_division_plan(
        rel_path="meta.py",
        language="unknown",
        content=meta_content,
        source_budget_chars=1_000_000,
    )
    assert [c.close_reason for c in meta_plan.chunks] == (
        ["metadata-ceiling"] * (len(meta_plan.chunks) - 1) + ["end-of-file"]
    )
    assert len(meta_plan.chunks) >= 2
    assert "".join(c.payload for c in meta_plan.chunks) == meta_content

    # both ceilings trip on the same next unit.
    monkeypatch.setattr(file_division, "MAX_LEAF_PROMPT_METADATA_CHARS", 700)
    both_content = "".join("p" * 400 + "\n" for _ in range(4))
    both_plan = build_division_plan(
        rel_path="both.py",
        language="unknown",
        content=both_content,
        source_budget_chars=700,
    )
    assert both_plan.chunks[0].close_reason == "source-and-metadata-ceiling"
    assert both_plan.chunks[-1].close_reason == "end-of-file"
    assert "".join(c.payload for c in both_plan.chunks) == both_content


def test_exact_ceiling_value_fits_and_following_overflow_sets_the_reason() -> None:
    """Section 5.6: 'Equality fits.' Two 500-character units co-pack to exactly
    B=1,000; alone that leaf closes ``end-of-file``, but with a following
    oversized-on-source unit the same leaf instead closes ``source-ceiling``.
    """
    fits_only = ("z" * 499 + "\n") + ("w" * 499 + "\n")
    fits_plan = build_division_plan(
        rel_path="fits.py", language="unknown", content=fits_only, source_budget_chars=1000
    )
    assert [c.payload_chars for c in fits_plan.chunks] == [1000]
    assert fits_plan.chunks[0].close_reason == "end-of-file"

    with_follower = fits_only + ("q" * 800 + "\n")
    follow_plan = build_division_plan(
        rel_path="fits2.py",
        language="unknown",
        content=with_follower,
        source_budget_chars=1000,
    )
    assert [c.payload_chars for c in follow_plan.chunks] == [1000, 801]
    assert follow_plan.chunks[0].close_reason == "source-ceiling"
    assert follow_plan.chunks[1].close_reason == "end-of-file"


def test_all_closed_boundary_kinds_are_reachable() -> None:
    """Section 5.6: the outer descriptor kinds ``file-start``/``file-end``/
    ``semantic-unit`` and the internal cut kinds ``syntax``/``physical-line``/
    ``balanced-codepoint`` are each produced by at least one deterministic
    plan, and every kind is one of the closed `CHUNK_BOUNDARY_VALUES`."""
    seen: set[str] = set()

    # balanced-codepoint + file-start + file-end, from a homogeneous span.
    homo = build_division_plan(
        rel_path="a.py", language="unknown", content="x" * 2010, source_budget_chars=1000
    )
    for chunk in homo.chunks:
        seen.add(chunk.start_boundary)
        seen.add(chunk.end_boundary)

    # semantic-unit, from a fitting unit followed by an oversized one.
    iso = build_division_plan(
        rel_path="b.py",
        language="unknown",
        content=("k" * 400 + "\n") + ("x" * 2010 + "\n"),
        source_budget_chars=1000,
    )
    for chunk in iso.chunks:
        seen.add(chunk.start_boundary)
        seen.add(chunk.end_boundary)

    # physical-line, from one oversized syntax unit whose body spans many
    # physical lines (a line boundary lands in the first cut's balance window).
    monkeypatch = pytest.MonkeyPatch()
    try:
        lined_content = ("x" * 49 + "\n") * 41  # 2,050 chars, breaks every 50
        monkeypatch.setattr(
            file_division,
            "extract_structure",
            _one_oversized_atom_structure(lined_content, ()),
        )
        lined = build_division_plan(
            rel_path="c.py",
            language="python",
            content=lined_content,
            source_budget_chars=1000,
        )
    finally:
        monkeypatch.undo()
    assert any(
        "physical-line" in (chunk.start_boundary, chunk.end_boundary)
        for chunk in lined.chunks
    )
    for chunk in lined.chunks:
        seen.add(chunk.start_boundary)
        seen.add(chunk.end_boundary)

    # syntax, via a strictly-nested symbol range placed inside the balance
    # window of the first cut.
    try:
        syn_content = "z" * 2010
        monkeypatch.setattr(
            file_division,
            "extract_structure",
            _one_oversized_atom_structure(syn_content, [("Whole.inner", 670, 900)]),
        )
        syn = build_division_plan(
            rel_path="d.py",
            language="python",
            content=syn_content,
            source_budget_chars=1000,
        )
    finally:
        monkeypatch.undo()
    for chunk in syn.chunks:
        seen.add(chunk.start_boundary)
        seen.add(chunk.end_boundary)

    assert seen == set(file_division.CHUNK_BOUNDARY_VALUES)
    assert {
        "file-start",
        "file-end",
        "semantic-unit",
        "syntax",
        "physical-line",
        "balanced-codepoint",
    } <= seen


# A tightly scoped descriptor mirroring exactly the three canonical ChunkPlan
# fields the boundary-constrained-small rule may consult (section 5.6). The
# later observability workstream owns the real public counter/presenter; this
# prompt only proves the rule is fully derivable from these fields.
_Descriptor = collections.namedtuple(
    "_Descriptor", ("payload_chars", "start_boundary", "end_boundary")
)

_PREFERRED_BOUNDARIES = frozenset({"syntax", "physical-line"})


def _boundary_constrained_indexes(descriptors, budget) -> list[int]:
    """Derive the boundary-constrained-small attribution set from descriptor
    fields alone: a piece adjacent to a preferred (syntax/physical-line) cut
    whose ``2 * payload_chars < budget`` is attributed. Each descriptor is one
    leaf object, so it contributes at most one index -- a middle piece with a
    preferred boundary on *both* sides is inspected from both sides but
    deduplicated to a single entry."""
    attributed: list[int] = []
    for index, descriptor in enumerate(descriptors):
        left_preferred = descriptor.start_boundary in _PREFERRED_BOUNDARIES
        right_preferred = descriptor.end_boundary in _PREFERRED_BOUNDARIES
        counted_from = 0
        for side_is_preferred in (left_preferred, right_preferred):
            if side_is_preferred and 2 * descriptor.payload_chars < budget:
                counted_from += 1  # inspected from this side of a preferred cut
        if counted_from and index not in attributed:
            attributed.append(index)  # one leaf object -> exactly one entry
    return attributed


def test_boundary_constrained_small_attribution_is_derivable_from_descriptor_fields() -> None:
    """Section 5.6 boundary-constrained-small: executable proof (no
    comments-as-evidence) that attribution needs only `payload_chars`,
    `start_boundary`, and `end_boundary`; that a middle piece bounded by a
    preferred cut on *both* sides is returned exactly once, not twice; that a
    small piece adjacent only to balanced-codepoint boundaries is not
    attributable; and that `2 * payload_chars == B` is not small (strict
    ``<``)."""
    B = 1000
    descriptors = [
        # 0: small, left edge is a preferred (syntax) cut -> attributable.
        _Descriptor(300, "syntax", "physical-line"),
        # 1: small MIDDLE piece, preferred cut on BOTH sides -> inspected from
        #    both sides but counted exactly once.
        _Descriptor(200, "physical-line", "syntax"),
        # 2: small, only balanced-codepoint boundaries -> NOT attributable.
        _Descriptor(120, "balanced-codepoint", "balanced-codepoint"),
        # 3: adjacent to a preferred cut but 2*payload == B exactly -> NOT small.
        _Descriptor(B // 2, "syntax", "balanced-codepoint"),
        # 4: adjacent to a preferred cut, one over the strict threshold -> small.
        _Descriptor(B // 2 - 1, "balanced-codepoint", "syntax"),
        # 5: large, preferred both sides -> not small, not attributed.
        _Descriptor(900, "syntax", "syntax"),
        # 6: small but no preferred boundary at all (file edges) -> not attributed.
        _Descriptor(10, "file-start", "file-end"),
    ]
    result = _boundary_constrained_indexes(descriptors, B)
    assert result == [0, 1, 4]  # 3 excluded (== not <), 2/5/6 excluded
    assert len(result) == len(set(result))  # index 1 counted once, not twice

    # the middle piece really is inspected from both sides: flipping its own
    # boundaries to balanced-codepoint removes it, proving both sides mattered.
    flipped = list(descriptors)
    flipped[1] = _Descriptor(200, "balanced-codepoint", "balanced-codepoint")
    assert _boundary_constrained_indexes(flipped, B) == [0, 4]


def test_boundary_constrained_small_rule_on_real_chunkplan_descriptors(monkeypatch) -> None:
    """The same rule applied to real `ChunkPlan` objects, using only their
    canonical `payload_chars`/`start_boundary`/`end_boundary` fields."""
    B = 1000

    # balanced-only subdivision: no preferred boundary, nothing attributed even
    # though every 670-char piece is a real ChunkPlan.
    balanced = build_division_plan(
        rel_path="bal.py", language="unknown", content="w" * 2010, source_budget_chars=B
    )
    assert all(
        c.start_boundary in ("file-start", "balanced-codepoint")
        and c.end_boundary in ("file-end", "balanced-codepoint")
        for c in balanced.chunks
    )
    assert _boundary_constrained_indexes(balanced.chunks, B) == []

    # 2-piece split with an in-window syntax cut at 451: the 451-char left piece
    # is boundary-constrained-small (2*451 < 1000); the 550-char right piece is
    # not (2*550 >= 1000).
    small = "z" * 1001
    monkeypatch.setattr(
        file_division,
        "extract_structure",
        _one_oversized_atom_structure(small, [("Whole.inner", 451, 700)]),
    )
    small_plan = build_division_plan(
        rel_path="small.py", language="python", content=small, source_budget_chars=B
    )
    assert [c.payload_chars for c in small_plan.chunks] == [451, 550]
    assert (small_plan.chunks[0].end_boundary, small_plan.chunks[1].start_boundary) == (
        "syntax",
        "syntax",
    )
    assert _boundary_constrained_indexes(small_plan.chunks, B) == [0]
    monkeypatch.undo()

    # 3-piece split with syntax cuts near both balanced targets: the middle
    # ChunkPlan is bounded by a preferred cut on both sides and is a single
    # object -> the rule counts it at most once.
    triple = "z" * 2010
    monkeypatch.setattr(
        file_division,
        "extract_structure",
        _one_oversized_atom_structure(
            triple, [("Whole.a", 670, 900), ("Whole.b", 1340, 1400)]
        ),
    )
    triple_plan = build_division_plan(
        rel_path="triple.py", language="python", content=triple, source_budget_chars=B
    )
    assert [c.payload_chars for c in triple_plan.chunks] == [670, 670, 670]
    middle = triple_plan.chunks[1]
    assert middle.start_boundary == "syntax" and middle.end_boundary == "syntax"
    assert triple_plan.chunks.count(middle) == 1  # one leaf object
    counted = _boundary_constrained_indexes(triple_plan.chunks, B)
    assert counted.count(1) <= 1  # never double-counted from its two sides
    assert counted == []  # 2*670 >= 1000, so not small here -- but still single


def test_known_symbols_for_half_open_deterministic_capped_and_deduped() -> None:
    """Section 5.6 / workstream 0A: `_known_symbols_for` assigns a symbol whose
    start byte equals a piece's end (a cut) to the *following* piece
    (half-open ``r.start_byte <= sym.start < r.end_byte``); its output is
    first-seen deterministic, holds no duplicate qualified name, and is capped
    at `MAX_KNOWN_SYMBOLS_PER_CHUNK`."""
    from codedoc.parser.source_structure import (
        SourceRange,
        SymbolFact,
        atom_id_for,
        symbol_id_for,
    )

    rel = "x.py"
    atom_id = atom_id_for(rel, "class", 0, 4096)

    def rng(start, end):
        return SourceRange(start, end, 1, 1, 1, 1)

    def sym(name, start, end):
        return SymbolFact(
            symbol_id=symbol_id_for(rel, "function", name, start, end),
            rel_path=rel,
            language="python",
            kind="function",
            qualified_name=name,
            signature="",
            range=rng(start, end),
            atom_id=atom_id,
        )

    before = sym("before", 50, 60)
    at_cut = sym("at_cut", 100, 130)
    left_range, right_range = rng(0, 100), rng(100, 200)
    assert file_division._known_symbols_for((before, at_cut), [left_range]) == ("before",)
    assert file_division._known_symbols_for((before, at_cut), [right_range]) == ("at_cut",)

    duplicates = tuple(sym("dup", 10, 20) for _ in range(3))
    distinct = tuple(sym(f"n{index}", 300 + index, 301 + index) for index in range(50))
    result = file_division._known_symbols_for(
        duplicates + distinct, [rng(0, 4096)]
    )
    assert len(result) == file_division.MAX_KNOWN_SYMBOLS_PER_CHUNK
    assert result.count("dup") == 1
    assert result[0] == "dup"  # first-seen order preserved
    assert len(set(result)) == len(result)


def _sym_factory():
    from codedoc.parser.source_structure import (
        SourceRange,
        SymbolFact,
        atom_id_for,
        symbol_id_for,
    )

    rel = "eq.py"
    atom_id = atom_id_for(rel, "class", 0, 10_000_000)

    def rng(start, end):
        return SourceRange(start, end, 1, 1, 1, 1)

    def sym(name, start, end):
        return SymbolFact(
            symbol_id=symbol_id_for(rel, "function", name, start, end),
            rel_path=rel,
            language="python",
            kind="function",
            qualified_name=name,
            signature="",
            range=rng(start, end),
            atom_id=atom_id,
        )

    return rng, sym


def test_known_symbols_by_chunk_batch_sweep_matches_reference_full_scan() -> None:
    """Workstream 0A: `_known_symbols_by_chunk` is the plan-mandated sorted-start
    linear sweep. Its per-chunk output (names, original tuple order, first-seen
    dedup, 32 cap, half-open ownership) must be byte-identical to a reference
    per-chunk full ordered scan across randomized *valid production partitions*:
    ordered, non-overlapping, gap-free half-open ranges covering the source.
    Also proves: unsorted input symbols, source- and cut-aligned starts,
    duplicate qualified names, empty symbols, chunks with no symbols, no symbol
    assigned to two chunks, deterministic repeats."""
    import random

    MAX = file_division.MAX_KNOWN_SYMBOLS_PER_CHUNK
    rng, sym = _sym_factory()

    def reference_one_chunk(symbols, ranges) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol.qualified_name in seen:
                continue
            if any(
                r.start_byte <= symbol.range.start_byte < r.end_byte for r in ranges
            ):
                seen.add(symbol.qualified_name)
                names.append(symbol.qualified_name)
                if len(names) == MAX:
                    return tuple(names)
        return tuple(names)

    generator = random.Random(20260828)
    for iteration in range(200):
        # a valid partition of [0, source_len): ascending, contiguous, half-open.
        chunk_count = generator.randint(1, 7)
        cuts = sorted(generator.randint(1, 6000) for _ in range(chunk_count - 1))
        bounds = [0, *cuts, cuts[-1] + generator.randint(1, 400) if cuts else 6000]
        # split each chunk span into 1-3 contiguous owning sub-ranges.
        chunk_range_lists = []
        for lo, hi in zip(bounds, bounds[1:]):
            if hi <= lo:
                hi = lo + 1
            inner = sorted({lo, hi, *(generator.randint(lo, hi) for _ in range(2))})
            chunk_range_lists.append(
                [rng(a, b) for a, b in zip(inner, inner[1:]) if b > a]
            )
        source_len = bounds[-1]

        symbols = []
        for _ in range(generator.randint(0, 120)):
            start = generator.randint(0, source_len - 1)
            if generator.random() < 0.4 and cuts:
                start = generator.choice(cuts)  # cut-aligned start
                if start >= source_len:
                    start = source_len - 1
            name = f"q{generator.randint(0, 20)}"  # names repeat -> dedup bites
            symbols.append(sym(name, start, min(start + 3, source_len)))
        generator.shuffle(symbols)  # arbitrary (non-source) tuple order
        symbols = tuple(symbols)

        got = file_division._known_symbols_by_chunk(symbols, chunk_range_lists)
        expected = tuple(
            reference_one_chunk(symbols, ranges) for ranges in chunk_range_lists
        )
        assert got == expected, iteration
        # deterministic repeat
        assert file_division._known_symbols_by_chunk(symbols, chunk_range_lists) == got
        # no symbol name assigned to two different chunks (partition invariant):
        # a symbol occurrence is owned by exactly one chunk, so a *unique* name
        # can appear in at most one chunk's list.
        placements: dict[str, int] = {}
        unique_names = [
            s.qualified_name
            for s in symbols
            if [t.qualified_name for t in symbols].count(s.qualified_name) == 1
        ]
        for chunk_index, names in enumerate(got):
            for name in names:
                if name in unique_names:
                    assert name not in placements
                    placements[name] = chunk_index

    # explicit edge cases
    empty_chunks = file_division._known_symbols_by_chunk((), [[rng(0, 10)], [rng(10, 20)]])
    assert empty_chunks == ((), ())
    only_one = file_division._known_symbols_by_chunk(
        (sym("a", 0, 2), sym("b", 12, 14)), [[rng(0, 10)], [rng(10, 20)], [rng(20, 30)]]
    )
    assert only_one == (("a",), ("b",), ())  # middle populated, last empty


def test_known_symbols_by_chunk_half_open_ownership_at_every_boundary() -> None:
    """Section 5.6 / 9.1 item 5: a symbol whose start byte equals an internal
    cut belongs *only* to the chunk that begins at that cut."""
    rng, sym = _sym_factory()
    chunk_range_lists = [[rng(0, 100)], [rng(100, 250)], [rng(250, 400)]]
    at_cut_1 = sym("cut1", 100, 110)
    at_cut_2 = sym("cut2", 250, 260)
    interior_0 = sym("in0", 40, 50)
    interior_2 = sym("in2", 399, 400)
    # tuple order: interior_2, at_cut_2, interior_0, at_cut_1
    result = file_division._known_symbols_by_chunk(
        (interior_2, at_cut_2, interior_0, at_cut_1), chunk_range_lists
    )
    # cut1 (start 100) -> chunk 1, not chunk 0; cut2 (start 250) -> chunk 2, not
    # chunk 1. The last chunk emits in original tuple order (in2 before cut2),
    # never source order.
    assert result == (("in0",), ("cut1",), ("in2", "cut2"))


def test_known_symbols_by_chunk_cap_dedup_and_same_name_in_different_chunks() -> None:
    """Cap is 32 per chunk; first-seen tuple order wins on dedup; the same
    qualified name declared in two different chunks appears in each."""
    rng, sym = _sym_factory()

    # cap + dedup in one chunk.
    cap_symbols = tuple(
        [sym("dup", 5, 6) for _ in range(3)]
        + [sym(f"n{i}", 10 + i, 11 + i) for i in range(50)]  # 50 distinct > cap
    )
    (capped,) = file_division._known_symbols_by_chunk(cap_symbols, [[rng(0, 1000)]])
    assert len(capped) == file_division.MAX_KNOWN_SYMBOLS_PER_CHUNK
    assert capped.count("dup") == 1
    assert capped[0] == "dup"  # first-seen order preserved
    assert len(set(capped)) == len(capped)

    # same qualified name declared once in each of two chunks -> in both lists.
    chunk_range_lists = [[rng(0, 1000)], [rng(1000, 2000)]]
    symbols = (
        sym("shared", 700, 701),
        sym("only_b", 1200, 1201),
        sym("shared", 1500, 1501),
    )
    result = file_division._known_symbols_by_chunk(symbols, chunk_range_lists)
    assert result == (("shared",), ("only_b", "shared"))


def test_known_symbols_for_single_chunk_wrapper_matches_reference() -> None:
    """The `_known_symbols_for` single-chunk wrapper delegates to the batch
    sweep and stays byte-identical to the reference full ordered scan for
    arbitrary range collections -- unsorted, adjacent, and overlapping."""
    import random

    MAX = file_division.MAX_KNOWN_SYMBOLS_PER_CHUNK
    rng, sym = _sym_factory()

    def reference(symbols, ranges) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol.qualified_name in seen:
                continue
            if any(
                r.start_byte <= symbol.range.start_byte < r.end_byte for r in ranges
            ):
                seen.add(symbol.qualified_name)
                names.append(symbol.qualified_name)
                if len(names) == MAX:
                    return tuple(names)
        return tuple(names)

    generator = random.Random(20260827)
    for _ in range(120):
        count = generator.randint(0, 90)
        symbols = []
        for _ in range(count):
            start = generator.randint(0, 4000)
            symbols.append(
                sym(f"q{generator.randint(0, count + 3)}", start, start + 5)
            )
        generator.shuffle(symbols)
        symbols = tuple(symbols)
        ranges = [
            rng(lo := generator.randint(0, 4000), lo + generator.randint(1, 900))
            for _ in range(generator.randint(1, 4))
        ]
        assert file_division._known_symbols_for(symbols, ranges) == reference(
            symbols, ranges
        )


def test_nested_symbol_abutting_a_cut_is_owned_by_the_following_leaf(monkeypatch) -> None:
    """Section 5.6 / 9.1 item 5: a declaration whose authoritative range starts
    exactly on a continuation cut is owned, exactly once, by the leaf that
    begins at that cut -- never double-counted across the abutting leaves."""
    content = "z" * 2010
    monkeypatch.setattr(
        file_division,
        "extract_structure",
        _one_oversized_atom_structure(content, [("Whole.at_cut", 670, 690)]),
    )
    plan = build_division_plan(
        rel_path="abut.py", language="python", content=content, source_budget_chars=1000
    )
    # the nested symbol's start (670) is an in-window syntax candidate for the
    # first cut, so the cut lands exactly on it.
    assert [(r.start_byte, r.end_byte) for c in plan.chunks for r in c.owning_ranges] == [
        (0, 670),
        (670, 1340),
        (1340, 2010),
    ]
    owners = [tuple(c.known_symbols) for c in plan.chunks]
    assert owners == [(), ("Whole.at_cut",), ()]
    assert sum(chunk.known_symbols.count("Whole.at_cut") for chunk in plan.chunks) == 1


def test_multibyte_oversized_unit_subdivides_with_exact_codepoint_and_byte_reconstruction() -> None:
    """Section 5.6 point 6 / 9.1 item 5: subdividing an oversized unit made of
    BMP, astral, and combining-mark code points reconstructs the canonical
    snapshot exactly in both Python code points and UTF-8 bytes. No
    grapheme-cluster guarantee is claimed or tested."""
    unit = "é雪\U0001f9ea\U0001f600Aéन्ष"  # mixed widths
    content = unit * 400  # 4,000 code points, one oversized lexical line
    assert "\n" not in content
    budget = 1000
    plan = build_division_plan(
        rel_path="uni.py", language="unknown", content=content, source_budget_chars=budget
    )
    assert len(plan.units) == 1
    assert len(plan.chunks) == -(-len(content) // budget)  # exact arithmetic minimum
    assert all(c.payload_chars <= budget for c in plan.chunks)
    assert "".join(c.payload for c in plan.chunks) == content
    assert b"".join(
        c.payload.encode("utf-8") for c in plan.chunks
    ) == content.encode("utf-8")
    data = content.encode("utf-8")
    offset = 0
    for chunk in plan.chunks:
        (source_range,) = chunk.owning_ranges
        assert source_range.start_byte == offset
        assert data[source_range.start_byte : source_range.end_byte].decode("utf-8") == (
            chunk.payload
        )
        offset = source_range.end_byte
    assert offset == len(data)


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
    # This fixture divides evenly (170 * 50 == 8,500 with no remainder), so
    # every chunk's raw bytes/ranges are unchanged from the pre-0.14.7
    # packer; only the plan/tree digests move. `division-packer-v6` binds the
    # `close_reason`/`start_boundary`/`end_boundary` fields -- and, in section
    # 5.4, the `leaf_prompt_signature_hint_chars` fixed-bound entry -- into the
    # canonical plan payload even where a piece's text is identical; and
    # `reduction-packing-v5` (section 6.1/6.3) additionally moves every
    # reduction node ID and the reduction-tree digest below.
    assert plan.plan_digest == (
        "division-plan:a6d935320693098a975533b8e39e9984"
        "edfa7f5ebe347777880d9d4aea6ba914"
    )

    tree = build_reduction_tree(plan, max_content_chars=12000)
    expected_node_ids = (
        "node_afc9ef88c31b8cb011a223e1eb4fe623ee3514ffa13afb9132ac4c0d3fa033a1",
        "node_ecc912ace8e55ae441ea7b51d0546885a2d7c0b714f484f57dca3c479548f341",
        "node_1c2bd19f0683403f3a039843bc1706fba255e230cbc427f7936e15080f3c31e1",
        "node_0c740c1040d7ab717bef3286c78523567bf40225896e1bdd79f1d0eb5f22a686",
        "node_d45caee5fdeec6582a6b0fc4d3f9d5096482871147bb1fbfcf53ba9e80060100",
        "node_80afe210f3088bc196819859f2fb6166dfa474014273d8b51480f9371f857832",
        "node_4f88202e3c0ed14f5cd75d96eba2e5b0f6972402aa0476ec9b9ee5cdccb68628",
    )
    assert tree.max_fan_in == 39
    assert len(tree.unit_consolidation_nodes) == 6
    assert len(tree.general_nodes) == 0
    assert [node.level for node in tree.unit_consolidation_nodes] == [1, 1, 1, 1, 1, 2]
    assert tuple(node.node_id for node in tree.all_nodes) == expected_node_ids
    assert tree.final_node.child_ids == (expected_node_ids[-2],)
    assert tree.tree_digest == (
        "reduction-tree:1d236ab6ad49a2ed655b6bb0658a7d35"
        "0b36ee971fcf1cf8d473f7d2ce0856c6"
    )
    expected_leaf_ids = tuple(chunk.chunk_id for chunk in plan.chunks)
    assert tree.final_node.leaf_ids == expected_leaf_ids
    assert len(set(tree.final_node.leaf_ids)) == 170
    for node in tree.all_intermediate_nodes:
        assert len(node.child_ids) >= 2, "no unary reducer may exist"


def test_reduction_packing_v5_moves_tree_and_transitive_completed_identity(
    monkeypatch,
) -> None:
    """Section 6.1 / 6.3 / 9.23: ``reduction-packing-v5`` owns the reduction
    tree's versioned identity. Reverting only ``REDUCTION_PACKING_REVISION`` to
    ``reduction-packing-v4`` must move the tree digest, every reduction/final
    node ID, the reducer/final execution identities, and -- transitively
    through the supplied reduction-tree digest -- the completed large-file
    identity. Ordinary/truncate identity and the synthesis-budget binding are
    untouched.

    This fails *behaviorally* when the constant is reverted (digest equality),
    not merely because a ``== "reduction-packing-v5"`` pin says otherwise.
    """
    import inspect

    from codedoc.core.record_meta import (
        expected_analysis_identity,
        expected_large_file_identity,
        expected_ordinary_path_identity,
    )

    assert file_division.REDUCTION_PACKING_REVISION == "reduction-packing-v5"

    source = _large_source(120)
    plan = build_division_plan(
        rel_path="pack.py", language="python", content=source, source_budget_chars=200
    )
    synthesis = max(200, file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS)
    tree_v5 = build_reduction_tree(
        plan, synthesis_manifest_chars=synthesis, language="python"
    )
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "REDUCTION_PACKING_REVISION", "reduction-packing-v4")
        tree_v4 = build_reduction_tree(
            plan, synthesis_manifest_chars=synthesis, language="python"
        )

    # The division plan is untouched by the reduction-packing revision.
    assert tree_v5.division_plan_digest == tree_v4.division_plan_digest == plan.plan_digest
    # Tree digest and node identities move.
    assert tree_v5.tree_digest != tree_v4.tree_digest
    v5_node_ids = {node.node_id for node in tree_v5.all_nodes}
    v4_node_ids = {node.node_id for node in tree_v4.all_nodes}
    assert v5_node_ids.isdisjoint(v4_node_ids)
    assert tree_v5.packing_revision == "reduction-packing-v5"
    assert tree_v4.packing_revision == "reduction-packing-v4"
    # The carried synthesis budget is identical -- only the revision moved.
    assert (
        tree_v5.synthesis_manifest_chars
        == tree_v4.synthesis_manifest_chars
        == synthesis
    )

    provider_identity = "provider-execution:" + "b" * 64
    content_hash = "a" * 64
    reducer_v5 = reduction_execution_identity(
        rel_path="pack.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree_v5.tree_digest,
        provider_identity=provider_identity,
        node=tree_v5.all_intermediate_nodes[0],
    )
    reducer_v4 = reduction_execution_identity(
        rel_path="pack.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree_v4.tree_digest,
        provider_identity=provider_identity,
        node=tree_v4.all_intermediate_nodes[0],
    )
    assert reducer_v5 != reducer_v4

    imports_digest = file_division.deterministic_imports_digest(())
    final_v5 = final_execution_identity(
        rel_path="pack.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree_v5.tree_digest,
        provider_identity=provider_identity,
        prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=imports_digest, node=tree_v5.final_node,
    )
    final_v4 = final_execution_identity(
        rel_path="pack.py", content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree_v4.tree_digest,
        provider_identity=provider_identity,
        prompt_profile_digest="no-prompt-profile-v1",
        imports_digest=imports_digest, node=tree_v4.final_node,
    )
    assert final_v5 != final_v4

    # Completed large-file identity moves purely through the supplied tree
    # digest: hold every other argument constant, pass only the two real
    # tree digests. No parallel synthesis-budget argument exists.
    common = dict(
        source_chars=len(source), max_chars=200, rel_path="pack.py",
        division_plan_digest=plan.plan_digest,
        structural_mode=plan.structural_mode, imports_digest=imports_digest,
    )
    completed_v5 = expected_large_file_identity(
        reduction_tree_digest=tree_v5.tree_digest, **common
    )
    completed_v4 = expected_large_file_identity(
        reduction_tree_digest=tree_v4.tree_digest, **common
    )
    assert completed_v5 != completed_v4
    assert completed_v5.startswith("large-file-v3:")
    params = set(inspect.signature(expected_large_file_identity).parameters)
    assert "synthesis_manifest_chars" not in params
    assert "synthesis_budget" not in params
    assert not any("budget" in name for name in params if name != "source_budget")

    # Ordinary / truncate identity is unaffected by the reduction-packing
    # revision in either direction.
    ordinary = expected_ordinary_path_identity("pack.py")
    analysis = expected_analysis_identity("single")
    with monkeypatch.context() as mp:
        mp.setattr(file_division, "REDUCTION_PACKING_REVISION", "reduction-packing-v4")
        assert expected_ordinary_path_identity("pack.py") == ordinary
        assert expected_analysis_identity("single") == analysis


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
def test_ledger_stores_parser_signature_not_model_hint_at_the_2000_boundary() -> None:
    """Section 20A item 3: parser authority at the real 2,000-character
    boundary, using a genuinely parsed declaration rather than a
    hand-constructed `SymbolFact`.

    A real function with 100 typed, defaulted parameters produces a raw
    `def ...` line far longer than 2,000 characters; the real parser bounds
    the captured `SymbolFact.signature` at exactly `MAX_STRUCTURE_SIGNATURE_CHARS`
    (2,000, raised from 600 by 0.14.7), matching this release's leaf response
    ceiling. A distinct 1,952-character string simulates the model's own
    matching-hint signature for the same declaration -- shorter, and
    different text, so a bug that let the model's report win would be
    visible. `build_fact_ledger` must store the parser-owned 2,000-character
    signature, never the 1,952-character hint, and must do so identically on
    a repeated call over the same inputs (allocation has no hidden
    non-determinism)."""
    params = ", ".join(f"param_{i:03d}: int = {i}" for i in range(100))
    source = f"def target_function({params}) -> None:\n    pass\n"
    plan = build_division_plan(
        rel_path="sample.py", language="python", content=source, source_budget_chars=100_000,
    )
    assert plan.structural_mode == "syntax"
    assert len(plan.chunks) == 1
    assert len(plan.symbols) == 1
    real_signature = plan.symbols[0].signature
    assert plan.symbols[0].qualified_name == "target_function"
    assert len(real_signature) == MAX_STRUCTURE_SIGNATURE_CHARS == 2000

    model_hint = (
        "def target_function(" + ", ".join(f"p{i}: int" for i in range(250)) + ")"
    )[:1952]
    assert len(model_hint) == 1952
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


@requires_structure_pack
def test_shortened_leaf_signatures_do_not_collapse_overloads_via_aligned_scope() -> None:
    """0.14.7 section 5.1 clause 3 / section 9.1 item 9: `MAX_LEAF_SYMBOL_SIGNATURE_CHARS`
    caps any *accepted* signature at 2,000 characters, so two real declarations
    that differ only after that point can be reported with byte-identical
    accepted (shortened) signature text -- and the parser's own signature is
    truncated at the identical bound, so even the parser-owned value
    `build_fact_ledger` overwrites onto the response is identical for both.
    The ledger must still keep them distinct: with aligned chunks, parser
    scope is the authority and the reported signature is only a matching
    hint (`build_fact_ledger` docstring). This would fail if a future change
    ever made the reported signature text the identity authority instead."""
    shared_params = ", ".join(f"p{index:03d}: int" for index in range(200))
    source = (
        f"def run({shared_params}, tag: int) -> int:\n"
        "    return tag\n\n"
        f"def run({shared_params}, tag: str) -> str:\n"
        "    return tag\n"
    )
    first_line, second_line = (
        line for line in source.splitlines() if line.startswith("def run")
    )
    assert len(first_line) > MAX_LEAF_SYMBOL_SIGNATURE_CHARS
    assert len(second_line) > MAX_LEAF_SYMBOL_SIGNATURE_CHARS
    shared = MAX_LEAF_SYMBOL_SIGNATURE_CHARS
    assert first_line[:shared] == second_line[:shared]
    assert first_line[shared:] != second_line[shared:]

    plan = build_division_plan(
        rel_path="overloads.py", language="python", content=source,
        source_budget_chars=6000,
    )
    run_symbols = tuple(
        symbol for symbol in plan.symbols if symbol.qualified_name.endswith("run")
    )
    assert len(run_symbols) == 2
    assert len(plan.chunks) == 1
    # The parser's own signature is already truncated to the hard bound, so
    # both real declarations -- sharing an identical first 2,000 characters --
    # collapse to byte-identical accepted signature text.
    assert run_symbols[0].signature == run_symbols[1].signature

    capsule = {
        "functions": [
            {
                "name": "run",
                "signature": run_symbols[0].signature,
                "description": "int form",
            },
            {
                "name": "run",
                "signature": run_symbols[1].signature,
                "description": "str form",
            },
        ]
    }
    ledger = build_fact_ledger(
        [capsule], language="python", chunks=plan.chunks, symbols=plan.symbols,
    )

    assert len(ledger.functions) == 2
    assert ledger.functions[0]["signature"] == ledger.functions[1]["signature"]
    assert {
        fact["_provenance"][0]["symbol_id"] for fact in ledger.functions
    } == {symbol.symbol_id for symbol in run_symbols}


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


def test_reducer_revision_invalidates_independently_of_the_leaf_revision(
    monkeypatch,
) -> None:
    """0.14.7 section 6.3 / workstream E: reducer nodes invalidate on
    `REDUCER_PROMPT_REVISION` in their own right, because
    `reduction_execution_identity` binds it -- independently of
    `LEAF_CAPSULE_SCHEMA_REVISION`, which `leaf_execution_identity` binds
    separately. Every leaf here is checkpointed under the real, current leaf
    revision (unaffected), while the one reducer is checkpointed under the
    superseded `file-reduction-v2` reducer revision -- reconstructed via the
    real `reduction_execution_identity()` function with the constant
    monkeypatched back, never a hand-written placeholder digest. Validating
    under the real current state must retain every leaf untouched and
    quarantine only the reducer, under the closed reason `stale-identity`."""
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
    reduction_input_digest = file_division.reduction_input_digest(
        rel_path=plan.rel_path,
        phase=reduction_node.phase,
        level=reduction_node.level,
        unit_id=reduction_node.unit_id,
        child_count=len(reduction_node.child_ids),
        ordered_child_narratives=file_division.refine_narrative_inputs(raw_narratives),
    )

    monkeypatch.setattr(
        file_division, "REDUCER_PROMPT_REVISION", "file-reduction-v2"
    )
    assert file_division.REDUCER_PROMPT_REVISION == "file-reduction-v2"
    stale_reducer_identity = reduction_execution_identity(
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        provider_identity=provider_identity,
        node=reduction_node,
    )
    monkeypatch.undo()

    # The reducer node is checkpointed under the superseded v2 identity; the
    # leaves above are all checkpointed under the real, current identity.
    nodes.append(
        tree_node_state(
            node_id=reduction_node.node_id,
            node_type=reduction_node.phase,
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=reduction_input_digest,
            execution_identity_digest=stale_reducer_identity,
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
    }
    assert len(quarantine_entries) == 1
    assert quarantine_entries[0].node_id == reduction_node.node_id
    assert quarantine_entries[0].reason == "stale-identity"


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
    """`SemanticUnitIdentity.signature`'s maximum is sourced from the same
    shared `MAX_STRUCTURE_SIGNATURE_CHARS` constant that also defines
    `MAX_LEAF_SYMBOL_SIGNATURE_CHARS` -- so the two bounds cannot drift
    apart -- rather than a separate literal. 0.14.7 raises both from 600 to
    2,000 (section 3.2's AST-walk census: 3 of 794 real declarations
    exceeded 600, the largest at 1,295 normalized / 1,395 raw characters)."""
    assert MAX_LEAF_SYMBOL_SIGNATURE_CHARS == MAX_STRUCTURE_SIGNATURE_CHARS == 2000

    accepted = SemanticUnitIdentity(
        unit_id="unit_" + "a" * 64,
        kind="function",
        qualified_name="q",
        signature="s" * 2000,
        atom_ids=("atom_" + "b" * 64,),
        source_range=_source_range(),
    )
    assert len(accepted.signature) == 2000

    with pytest.raises(ValueError, match="exceeds 2000 characters"):
        SemanticUnitIdentity(
            unit_id="unit_" + "a" * 64,
            kind="function",
            qualified_name="q",
            signature="s" * 2001,
            atom_ids=("atom_" + "b" * 64,),
            source_range=_source_range(),
        )


def test_derived_leaf_capsule_maximum_is_exactly_986272() -> None:
    """The capsule bound is derived from shared constants, not hard-coded.

    `0.14.3` raised `MAX_LEAF_SYMBOL_SIGNATURE_CHARS` from 256 to 600,
    raising the worst-case leaf capsule from 150,656 to 200,192 canonical
    characters. `0.14.4` raised `MAX_LEAF_SYMBOL_ITEMS_PER_KIND` from 12 to
    `MAX_KNOWN_SYMBOLS_PER_CHUNK` (32), raising it again to 448,672. 0.14.7
    raises `MAX_LEAF_SYMBOL_SIGNATURE_CHARS` again, from 600 to 2,000 -- a
    537,600-character increase to exactly 986,272, being 2 kinds x 32 items
    x 1,400 extra raw characters x 6 escaped (each additional raw character
    of an already-present signature field escapes 6-fold as `\\u0000`)."""
    assert MAX_LEAF_CAPSULE_CANONICAL_CHARS == 986272
    assert MAX_LEAF_CAPSULE_CANONICAL_CHARS - 448672 == 537600


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


# ---------------------------------------------------------------------------
# 0.14.7 section 9 / 9.1 item 16 / mutation check 22: same short name, same
# first 2,000 declaration characters, distinct authoritative ranges
# ---------------------------------------------------------------------------


@requires_structure_pack
def test_same_name_long_prefix_declarations_stay_distinct_by_parser_identity() -> None:
    """Two methods both named `handler`, whose declaration lines share their
    first 2,000 characters and differ only afterwards, occupy distinct
    authoritative ranges/scopes. The parser truncates both captured
    signatures to the identical 2,000-character prefix, and the model returns
    one identical shortened hint for both -- yet `build_fact_ledger` keeps
    both facts, each attached to its own parser-owned range, never collapsing
    them through the identical returned text."""
    shared_prefix = "def handler(" + "p" * (2000 - len("def handler("))
    assert len(shared_prefix) == 2000
    line_a = shared_prefix + "p" * 90 + ") -> int:"
    line_b = shared_prefix + "q" * 90 + ") -> int:"
    assert line_a[:2000] == line_b[:2000] and line_a[2000:] != line_b[2000:]
    source = (
        f"class A:\n    {line_a}\n        return 1\n\n"
        f"class B:\n    {line_b}\n        return 2\n"
    )
    plan = build_division_plan(
        rel_path="dup.py", language="python", content=source, source_budget_chars=200_000
    )
    assert plan.structural_mode == "syntax"
    assert len(plan.chunks) == 1

    handler_symbols = [s for s in plan.symbols if s.qualified_name == "handler"]
    assert len(handler_symbols) == 2  # same short name...
    sig_a, sig_b = (s.signature for s in handler_symbols)
    assert sig_a == sig_b == line_a[:2000]  # ...identical 2,000-char parser prefix
    assert len(sig_a) == MAX_LEAF_SYMBOL_SIGNATURE_CHARS == 2000
    range_pairs = {(s.range.start_byte, s.range.end_byte) for s in handler_symbols}
    assert len(range_pairs) == 2  # ...but distinct authoritative ranges/scopes

    identical_hint = shared_prefix[:800]
    capsule = {
        "description": "Two same-named methods sharing a long prefix.",
        "functions": [
            {"name": "handler", "signature": identical_hint, "description": "a"},
            {"name": "handler", "signature": identical_hint, "description": "b"},
        ],
    }
    ledger = build_fact_ledger(
        [capsule], language="python", chunks=plan.chunks, symbols=plan.symbols
    )
    handler_facts = [f for f in ledger.functions if f["name"] == "handler"]
    assert len(handler_facts) == 2, handler_facts  # no collapse through the hint

    stored_ranges = set()
    stored_unit_ids = set()
    for fact in handler_facts:
        # the parser-owned signature is authoritative, not the identical hint.
        assert fact["signature"] == sig_a
        assert fact["signature"] != identical_hint
        provenance = fact["_provenance"][0]
        symbol_range = provenance["symbol"]["source_range"]
        stored_ranges.add((symbol_range["start_byte"], symbol_range["end_byte"]))
        stored_unit_ids.update(provenance["semantic_unit_ids"])
    # each fact maps to its own distinct parser-owned range and semantic unit.
    assert stored_ranges == range_pairs
    assert len(stored_unit_ids) == 2


# ---------------------------------------------------------------------------
# 0.14.7 section 4.8 / 5.10 / 11 / mutation check 12: the exact production-path
# topology fixture is invariant between a 600- and a 2,000-character full
# parser matching signature, because only the 600-character hint is rendered.
# ---------------------------------------------------------------------------

_SECTION_48_BIG_DECL_NAME = "big_declaration"


def _section_48_fixture() -> str:
    """Exactly 12,500 Python code points: 164 tiny definitions, one
    1,690-character single-line declaration, and comment filler."""
    tiny = "".join(f"def t{i:03d}(aaaaaa): pass\n" for i in range(164))
    pad = 1690 - len(f"def {_SECTION_48_BIG_DECL_NAME}(") - len(") -> int:")
    big = f"def {_SECTION_48_BIG_DECL_NAME}(" + "p" * pad + ") -> int:\n    return 0\n"
    body = tiny + big
    comment_line = "# " + "x" * 60 + "\n"
    fill = 12_500 - len(body)
    whole, remainder = divmod(fill, len(comment_line))
    source = body + comment_line * whole + ("#" * remainder if remainder else "")
    assert len(source) == 12_500
    return source


def _section_48_membership(plan) -> list:
    """Per-chunk semantic-unit membership by the fields that must stay stable
    across a full-signature bound change: kind, qualified name, source range
    (never unit_id or signature, which intentionally move with the bound)."""
    return [
        [(u.kind, u.qualified_name, u.source_range) for u in chunk.semantic_units]
        for chunk in plan.chunks
    ]


def _section_48_initial_calls(plan, tree) -> int:
    # one leaf call per chunk + every reducer node + one final synthesis call.
    return len(plan.chunks) + len(tree.all_intermediate_nodes) + 1


@requires_structure_pack
def test_section_48_topology_is_invariant_between_600_and_2000_signatures(
    monkeypatch,
) -> None:
    """Section 4.8 / 5.10 / 11 / mutation check 12: the exact 12,500-character
    / 164-definition / one-1,690-character-declaration fixture is planned twice
    through real syntax extraction -- once with the full parser matching
    signature bound at 600, once at 2,000 -- and BOTH real plans are retained
    and compared directly. Only the independent 600-character hint reaches
    prompt metadata, so the raised full bound changes neither the source
    topology nor a paid call. A tail sensitivity check lifts the hint clamp on
    the same fixture and proves it would otherwise add a chunk and a call."""
    import codedoc.parser.tree_sitter_structure as tree_sitter_structure

    source = _section_48_fixture()

    with monkeypatch.context() as mp:
        mp.setattr(tree_sitter_structure, "MAX_STRUCTURE_SIGNATURE_CHARS", 600)
        plan_600 = build_division_plan(
            rel_path="s48.py", language="python", content=source,
            source_budget_chars=12_000,
        )
        tree_600 = build_reduction_tree(plan_600, synthesis_manifest_chars=12_000)

    with monkeypatch.context() as mp:
        mp.setattr(tree_sitter_structure, "MAX_STRUCTURE_SIGNATURE_CHARS", 2000)
        plan_2000 = build_division_plan(
            rel_path="s48.py", language="python", content=source,
            source_budget_chars=12_000,
        )
        tree_2000 = build_reduction_tree(plan_2000, synthesis_manifest_chars=12_000)

    # --- both plans exist and are identical where identity must not move ---
    assert plan_600.structural_mode == plan_2000.structural_mode == "syntax"
    assert len(plan_600.chunks) == len(plan_2000.chunks) == 2
    assert [c.payload for c in plan_600.chunks] == [c.payload for c in plan_2000.chunks]
    assert [c.payload_chars for c in plan_600.chunks] == [1955, 10545]
    assert [c.payload_chars for c in plan_2000.chunks] == [1955, 10545]
    assert [c.owning_ranges for c in plan_600.chunks] == [
        c.owning_ranges for c in plan_2000.chunks
    ]
    assert (
        [c.close_reason for c in plan_600.chunks]
        == [c.close_reason for c in plan_2000.chunks]
        == ["metadata-ceiling", "end-of-file"]
    )
    assert (
        [(c.start_boundary, c.end_boundary) for c in plan_600.chunks]
        == [(c.start_boundary, c.end_boundary) for c in plan_2000.chunks]
        == [("file-start", "semantic-unit"), ("semantic-unit", "file-end")]
    )
    assert _section_48_membership(plan_600) == _section_48_membership(plan_2000)

    # --- the full internal matching signature genuinely differs ---
    big_600 = next(
        u for u in plan_600.units if u.qualified_name == _SECTION_48_BIG_DECL_NAME
    )
    big_2000 = next(
        u for u in plan_2000.units if u.qualified_name == _SECTION_48_BIG_DECL_NAME
    )
    assert len(big_600.signature) == 600
    assert len(big_2000.signature) == 1690

    # --- but the rendered prompt hint is exactly 600 in both ---
    assert len(file_division._leaf_prompt_unit_value(big_600)["signature"]) == 600
    assert len(file_division._leaf_prompt_unit_value(big_2000)["signature"]) == 600

    # --- reduction topology: 2 leaf + 0 reducer + 1 final = 3 initial calls ---
    for plan, tree in ((plan_600, tree_600), (plan_2000, tree_2000)):
        assert len(tree.unit_consolidation_nodes) == 0
        assert len(tree.general_nodes) == 0
        assert len(tree.final_node.child_ids) == 2
        assert _section_48_initial_calls(plan, tree) == 3

    # --- sensitivity: remove the hint clamp on the SAME fixture ---
    with monkeypatch.context() as mp:
        mp.setattr(tree_sitter_structure, "MAX_STRUCTURE_SIGNATURE_CHARS", 2000)
        mp.setattr(file_division, "MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS", 2000)
        plan_unclamped = build_division_plan(
            rel_path="s48.py", language="python", content=source,
            source_budget_chars=12_000,
        )
        tree_unclamped = build_reduction_tree(
            plan_unclamped, synthesis_manifest_chars=12_000
        )
    assert file_division.MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS == 600  # restored
    assert plan_unclamped.structural_mode == "syntax"
    assert len(plan_unclamped.chunks) == 3
    assert [c.close_reason for c in plan_unclamped.chunks] == [
        "metadata-ceiling",
        "metadata-ceiling",
        "end-of-file",
    ]
    assert len(tree_unclamped.final_node.child_ids) == 3
    assert _section_48_initial_calls(plan_unclamped, tree_unclamped) == 4


def test_section_48_reference_topology_is_stable_and_two_chunks() -> None:
    """A monkeypatch-free anchor: the fixture at the current (2,000) parser
    bound plans to exactly 2 chunks / 3 initial calls, so the invariance test
    above is comparing against a real, stable topology."""
    source = _section_48_fixture()
    first = build_division_plan(
        rel_path="s48.py", language="python", content=source, source_budget_chars=12_000
    )
    second = build_division_plan(
        rel_path="s48.py", language="python", content=source, source_budget_chars=12_000
    )
    assert first.plan_digest == second.plan_digest
    assert len(first.chunks) == 2
    assert first.structural_mode == "syntax"


def test_prompt_signature_hint_policy_is_bound_through_the_real_plan_digest(
    monkeypatch,
) -> None:
    """Section 5.4 / 6.C: `MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS` is bound into
    division-plan / completed identity *through the plan's fixed-bound
    payload* -- not by duplicating a hint bound inside `record_meta`. On a
    deterministic split source whose units carry no signature at all (so 600
    vs 601 changes not one rendered byte), flipping only that policy constant
    leaves the source topology untouched while moving the real division-plan
    digest, and everything computed from it: the reduction-tree digest, the
    corresponding leaf execution identity, and the completed large-file
    identity from the real `expected_large_file_identity()`."""
    from codedoc.core.record_meta import expected_large_file_identity

    source = _large_source(200)
    budget = 2000
    rel_path = "hint.py"
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64
    imports_digest = file_division.deterministic_imports_digest(())

    def _build():
        plan = build_division_plan(
            rel_path=rel_path, language="python", content=source,
            source_budget_chars=budget,
        )
        tree = build_reduction_tree(plan, synthesis_manifest_chars=12_000)
        return plan, tree

    plan_600, tree_600 = _build()
    assert len(plan_600.chunks) >= 2
    assert {len(u.signature) for u in plan_600.units} == {0}  # pure policy mutation

    with monkeypatch.context() as mp:
        mp.setattr(file_division, "MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS", 601)
        assert file_division.MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS == 601
        plan_601, tree_601 = _build()
    assert file_division.MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS == 600  # restored

    # source topology is unchanged by the policy-only change.
    assert len(plan_601.chunks) == len(plan_600.chunks)
    assert [c.payload for c in plan_601.chunks] == [c.payload for c in plan_600.chunks]
    assert [c.owning_ranges for c in plan_601.chunks] == [
        c.owning_ranges for c in plan_600.chunks
    ]
    assert [c.close_reason for c in plan_601.chunks] == [
        c.close_reason for c in plan_600.chunks
    ]
    assert [c.chunk_id for c in plan_601.chunks] == [
        c.chunk_id for c in plan_600.chunks
    ]

    # ...but the real division-plan and reduction-tree digests move.
    assert plan_600.plan_digest != plan_601.plan_digest
    assert tree_600.tree_digest != tree_601.tree_digest

    leaf_600 = leaf_execution_identity(
        rel_path=rel_path, content_hash=content_hash,
        division_plan_digest=plan_600.plan_digest,
        provider_identity=provider_identity, chunk=plan_600.chunks[0],
    )
    leaf_601 = leaf_execution_identity(
        rel_path=rel_path, content_hash=content_hash,
        division_plan_digest=plan_601.plan_digest,
        provider_identity=provider_identity, chunk=plan_601.chunks[0],
    )
    assert leaf_600 != leaf_601

    def _completed(plan, tree) -> str:
        return expected_large_file_identity(
            source_chars=len(source),
            max_chars=budget,
            rel_path=rel_path,
            division_plan_digest=plan.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            structural_mode=plan.structural_mode,
            imports_digest=imports_digest,
        )

    completed_600 = _completed(plan_600, tree_600)
    completed_601 = _completed(plan_601, tree_601)
    assert completed_600 != completed_601
    assert completed_600.startswith("large-file-v3:")
    assert completed_601.startswith("large-file-v3:")


# ---------------------------------------------------------------------------
# 0.14.7 section 12 / 6.2: derived capsule constant and the two independent
# tracemalloc probes
# ---------------------------------------------------------------------------


def test_derived_capsule_constant_change_is_exactly_the_signature_raise() -> None:
    """Section 6.2 / 12: the 448,672 -> 986,272 jump is precisely the
    signature raise -- 2 kinds x 32 items x (2,000 - 600) raw characters x
    6-fold JSON escaping -- recomputed from the real shape, and the capsule
    constant reaches no capacity-block decision."""
    escape = "\x00"  # serializes as \u0000 (6 chars) under json.dumps
    assert len(json.dumps(escape)) == len('"\\u0000"')

    def _capsule(signature_chars: int) -> dict:
        symbol = {
            "name": escape * file_division.MAX_LEAF_SYMBOL_NAME_CHARS,
            "description": escape * file_division.MAX_LEAF_SYMBOL_DESCRIPTION_CHARS,
            "signature": escape * signature_chars,
        }
        return {
            "description": escape * file_division.MAX_LEAF_DESCRIPTION_CHARS,
            "functions": [dict(symbol) for _ in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)],
            "classes": [dict(symbol) for _ in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)],
            "exports": [
                escape * file_division.MAX_LEAF_EXPORT_ITEM_CHARS
                for _ in range(file_division.MAX_LEAF_EXPORT_ITEMS)
            ],
        }

    at_2000 = len(file_division.canonical_json(_capsule(2000)))
    at_600 = len(file_division.canonical_json(_capsule(600)))
    assert at_2000 == MAX_LEAF_CAPSULE_CANONICAL_CHARS == 986_272
    assert at_600 == 448_672
    assert at_2000 - at_600 == 2 * MAX_LEAF_SYMBOL_ITEMS_PER_KIND * (2000 - 600) * 6

    # identity-only: no BlockedReason mentions the capsule, and the constant is
    # not consulted anywhere in file_division outside its own definition line.
    assert not any("capsule" in reason for reason in BLOCKED_REASON_ORDER)
    fd_source = pathlib.Path(file_division.__file__).read_text(encoding="utf-8")
    assert fd_source.count("MAX_LEAF_CAPSULE_CANONICAL_CHARS") == 1  # the definition


def test_tracemalloc_probe_maximum_canonical_leaf_capsule_under_3_mib() -> None:
    """Section 5.10 / 12: constructing and serializing the maximum canonical
    leaf capsule peaks below 3 MiB on the recorded interpreter. Every maximal
    field is a fresh, independent allocation built inside the traced region --
    matching the production `_MAX_LEAF_CAPSULE` comprehension shape, not a
    shallow copy that would share preallocated strings. Measured on its own,
    never combined with the metadata probe."""
    import tracemalloc

    escape = "\x00"  # a lone 1-char string; every `escape * N` below is fresh
    name_chars = file_division.MAX_LEAF_SYMBOL_NAME_CHARS
    symbol_desc_chars = file_division.MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
    signature_chars = MAX_LEAF_SYMBOL_SIGNATURE_CHARS
    description_chars = file_division.MAX_LEAF_DESCRIPTION_CHARS
    export_item_chars = file_division.MAX_LEAF_EXPORT_ITEM_CHARS
    items_per_kind = MAX_LEAF_SYMBOL_ITEMS_PER_KIND
    export_items = file_division.MAX_LEAF_EXPORT_ITEMS

    tracemalloc.start()
    try:
        capsule = {
            "description": escape * description_chars,
            "functions": [
                {
                    "name": escape * name_chars,
                    "description": escape * symbol_desc_chars,
                    "signature": escape * signature_chars,
                }
                for _ in range(items_per_kind)
            ],
            "classes": [
                {
                    "name": escape * name_chars,
                    "description": escape * symbol_desc_chars,
                    "signature": escape * signature_chars,
                }
                for _ in range(items_per_kind)
            ],
            "exports": [escape * export_item_chars for _ in range(export_items)],
        }
        rendered = file_division.canonical_json(capsule)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(rendered) == MAX_LEAF_CAPSULE_CANONICAL_CHARS == 986_272
    # every entry is a distinct object with distinct strings.
    assert len({id(item) for item in capsule["functions"]}) == items_per_kind
    assert len({id(item["signature"]) for item in capsule["functions"]}) == items_per_kind
    assert peak < 3 * 1024 * 1024, peak


def test_tracemalloc_probe_maximum_rendered_leaf_metadata_under_3_mib() -> None:
    """Section 5.10 / 12: rendering the maximum leaf prompt metadata that
    still fits `MAX_LEAF_PROMPT_METADATA_CHARS` peaks below 3 MiB. Fresh
    maximal 2,000-character-signature units are built inside a fresh traced
    session, the real renderer is called, and every serialized signature is
    proved clamped to exactly 600. Measured on its own, never combined with
    the capsule probe."""
    import tracemalloc

    group_unit_id = "unit_" + "0" * 64

    def _make_unit(index: int) -> SemanticUnitIdentity:
        return SemanticUnitIdentity(
            unit_id="unit_" + f"{index:064x}",
            kind="k" * 160,
            qualified_name="q" * 240,
            signature="s" * MAX_LEAF_SYMBOL_SIGNATURE_CHARS,  # 2,000, fresh
            atom_ids=("atom_" + f"{index:064x}",),
            source_range=SourceRange(index * 10, index * 10 + 5, 1, 1, 1, 6),
        )

    # maximum fitting unit count -- computed outside the traced region.
    fitting_count = 0
    while True:
        probe = tuple(_make_unit(i) for i in range(fitting_count + 1))
        chars = file_division.leaf_prompt_metadata_chars(
            group_unit_id=group_unit_id,
            semantic_units=probe,
            unit_indexes=tuple(range(fitting_count + 1)),
            unit_count=fitting_count + 1,
            owning_ranges=tuple(u.source_range for u in probe),
        )
        if chars > file_division.MAX_LEAF_PROMPT_METADATA_CHARS:
            break
        fitting_count += 1
    assert fitting_count >= 1

    tracemalloc.start()
    try:
        units = tuple(_make_unit(i) for i in range(fitting_count))  # fresh, in trace
        rendered = file_division.render_leaf_prompt_metadata(
            group_unit_id=group_unit_id,
            semantic_units=units,
            unit_indexes=tuple(range(fitting_count)),
            unit_count=fitting_count,
            owning_ranges=tuple(u.source_range for u in units),
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(rendered) <= file_division.MAX_LEAF_PROMPT_METADATA_CHARS
    serialized = json.loads(rendered)
    serialized_signatures = [
        entry["signature"] for entry in serialized["semantic_units"]
    ]
    assert len(serialized_signatures) == fitting_count
    assert all(sig == "s" * 600 for sig in serialized_signatures)
    assert all(len(sig) <= 600 for sig in serialized_signatures)
    # the full input identities still carry their 2,000-character signatures.
    assert all(len(u.signature) == MAX_LEAF_SYMBOL_SIGNATURE_CHARS == 2000 for u in units)
    assert peak < 3 * 1024 * 1024, peak


# ---------------------------------------------------------------------------
# mutation check: leaf-capsule identity is bound into leaf_execution_identity
# ---------------------------------------------------------------------------


def test_leaf_capsule_reverting_to_the_prior_value_makes_a_leaf_checkpoint_stale(
    monkeypatch,
) -> None:
    """`leaf_execution_identity` binds `LEAF_CAPSULE_SCHEMA_REVISION`, so
    reverting `leaf-capsule-v10` to `leaf-capsule-v9` changes a leaf's
    execution identity. A single leaf checkpointed under the reconstructed
    prior identity (real function, constant monkeypatched back, then undone)
    is quarantined as `stale-identity` by the current validation while every
    sibling leaf checkpointed under the real current identity is retained.

    Full leaf/reducer/final dependency-closure behaviour for this same advance
    -- stale leaves rejected, independent leaves kept, only dependent reducers
    and the final node pruned -- is proven in
    `tests/integration/persistence/test_split_partial_quarantine.py`."""
    plan = build_division_plan(
        rel_path="leaf_capsule_revert.py",
        language="unknown",
        content=_large_source(90),
        source_budget_chars=200,
    )
    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    content_hash = "a" * 64
    provider_identity = "provider-execution:" + "b" * 64
    profile_digest = "no-prompt-profile-v1"
    imports_digest = file_division.deterministic_imports_digest(())

    def _leaf_id(chunk):
        return leaf_execution_identity(
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            provider_identity=provider_identity,
            chunk=chunk,
        )

    stale_chunk = plan.chunks[0]
    current_id = _leaf_id(stale_chunk)
    monkeypatch.setattr(file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v9")
    assert file_division.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v9"
    stale_id = _leaf_id(stale_chunk)
    monkeypatch.undo()
    # direct behavioral identity evidence: the revision reversal moves the digest.
    assert stale_id != current_id
    assert file_division.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v10"

    nodes = []
    for chunk in plan.chunks:
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
                execution_identity_digest=(
                    stale_id if chunk.chunk_id == stale_chunk.chunk_id else _leaf_id(chunk)
                ),
                unit_id=None,
                child_ids=(),
                coverage_leaf_ids=(chunk.chunk_id,),
                result={"description": "orig", "chunk_id": chunk.chunk_id,
                        "unit_id": chunk.unit_id},
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

    retained_ids = {state.node_id for state in retained}
    # every sibling leaf checkpointed under the real current (leaf-capsule-v10)
    # identity is retained; only the leaf stamped under the prior
    # (leaf-capsule-v9) identity is quarantined, under `stale-identity`.
    assert retained_ids == {
        c.chunk_id for c in plan.chunks if c.chunk_id != stale_chunk.chunk_id
    }
    assert len(quarantine_entries) == 1
    assert quarantine_entries[0].node_id == stale_chunk.chunk_id
    assert quarantine_entries[0].reason == "stale-identity"
