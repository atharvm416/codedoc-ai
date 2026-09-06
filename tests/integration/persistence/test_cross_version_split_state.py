"""Cross-version split-partial generation boundary (sections 5 and 14).

`0.14.3` advances the current node-keyed split-partial container schema from
3 to 4 alongside the `leaf-capsule-v6` signature-bound correction. The
frozen fixtures under `tests/fixtures/split_state/` are inert data proving,
without any build or provider contact, that:

- a released schema-3 container is rejected on its schema version alone,
  before any node is deserialized or quarantined (section 9's early,
  uniform boundary), leaving the recovery artifact byte-identical;
- that rejection is reached identically whether the container holds one
  node or more than `MAX_QUARANTINE_ENTRIES_PER_FILE`, so the bounded
  quarantine path is never touched by a predecessor container;
- the current schema-4 generation still validates, resumes, and
  checkpoints exactly as before, proving the boundary did not disturb
  same-version behavior;
- legacy schema-1 and dormant schema-2 predecessors remain preserved and
  blocked;
- a document whose `partial_files` map mixes a schema-3 and a schema-4
  container is rejected as a whole on the first unsupported entry, with no
  partial acceptance of the schema-4 sibling;
- a malformed current-schema container fails closed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from codedoc.core.document import read_codedoc_document
from codedoc.core.file_division import (
    PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    ReductionNodeState,
    canonical_stream_digest,
)
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import build_recovery_identity
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.fixture_paths import FIXTURES_ROOT
from tests.support.structure_extra import requires_structure_pack

_FIXTURES = FIXTURES_ROOT / "split_state"
_REL_PATH = "src/large.py"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _large_source(lines: int = 220) -> str:
    return "\n".join(f"value_{i} = {i}" for i in range(lines)) + "\n"


def _prepare_recovery(tmp_path, rel_paths: tuple[str, ...] = (_REL_PATH,)):
    """Write real source for *rel_paths* and initialize a genuine, empty,
    identity-stamped recovery file the same way a real run would, so the
    fixture container(s) can then be spliced in as the only difference from
    a real recovery file."""
    for rel_path in rel_paths:
        source_path = tmp_path / rel_path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(_large_source(), encoding="utf-8", newline="")
    output_dir = tmp_path / "docs"
    recovery_path = output_dir / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path,
        "json",
        rel_paths[0],
        {},
        build_recovery_identity(
            project_root=tmp_path,
            json_target=output_dir / "codedoc.json",
            md_target=None,
            entry_file=rel_paths[0],
            documentation_scope="entry",
            analysis_mode="single",
            analysis_revision=ANALYSIS_REVISION,
            large_file_strategy="split",
        ),
    )
    writer.initialize_empty()
    return recovery_path


def _splice_partial_files(recovery_path: Path, partial_files: dict) -> None:
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload["_codedoc"]["partial_files"] = partial_files
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")


def _run_blocked(
    tmp_path, *, entry_rel: str = _REL_PATH, strict_no_node_construction: bool = False
):
    """Run a blocked recovery scenario with sentinels on every early-boundary
    seam (section 20A item 4), not merely a provider-call count.

    Exception text plus zero provider calls is insufficient on its own: a
    defect that let a rejected container reach planning or writer
    construction could still coincidentally raise `ConfigError` and still
    make zero provider calls (e.g. an unrelated downstream failure), passing
    the old assertions for the wrong reason.

    `strict_no_node_construction` (default `False`) selects the
    node-construction sentinel's behavior:

    - `True`: the sentinel raises `AssertionError` immediately if
      `ReductionNodeState` is constructed at all -- the literal reading of
      "install sentinels that fail on node construction/deserialization."
      `AssertionError` is neither `TypeError` nor `ValueError`, so it is not
      caught by `document.py::_partial_files_from_meta`'s own narrow
      `except (TypeError, ValueError)` around that exact call, and it is not
      `ConfigError`, so it is not caught by this function's own
      `pytest.raises(ConfigError)` either -- it propagates all the way to
      pytest as an unambiguous failure. Used by both released-schema-3
      tests (single node and the more-than-32-node fixture) and by the
      mandatory rejected-first mixed-generation test -- every one of these
      three is known, by construction of the fixture it uses, to reach
      zero node constructions, so strict failure is the correct proof
      there, not merely a stricter option.
    - `False` (the parameter default, kept for the remaining callers this
      pass did not touch): the sentinel wraps the real `ReductionNodeState`,
      records the call, and returns a genuine, fully-constructed instance.
      This is a genuine functional requirement -- not merely a default left
      unconverted -- for exactly one caller: the current-first
      mixed-generation test, whose schema-4 sibling sorts before its
      rejected schema-3 sibling in `_partial_files_from_meta`'s
      `sorted(partial_files)` walk, so that sibling's one node is genuinely
      constructed and used immediately afterward before the loop ever
      reaches the entry that rejects the whole document -- raising there
      would turn a `ConfigError` into an unrelated `AttributeError` and
      produce a false failure in this harness itself, not in the code
      under test. The legacy-schema-1, dormant-schema-2, and
      malformed-container tests also currently pass `False` (the default)
      even though, like the schema-3 tests, they always reach zero node
      constructions too -- they were simply outside this pass's scope, not
      cases requiring non-strict behavior."""
    provider_creations: list[bool] = []
    node_constructions: list[str] = []
    plan_calls: list[bool] = []
    writer_constructions: list[bool] = []

    def _run(monkeypatch):
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda _config: provider_creations.append(True),
        )
        monkeypatch.setattr(
            "codedoc.core.planning.build_pipeline_plan",
            lambda *a, **k: plan_calls.append(True),
        )
        monkeypatch.setattr(
            "codedoc.pipeline.build_pipeline_plan",
            lambda *a, **k: plan_calls.append(True),
        )
        monkeypatch.setattr(
            "codedoc.pipeline.SafeWriter",
            lambda *a, **k: writer_constructions.append(True),
        )

        if strict_no_node_construction:
            def _tracking_node_state(*args, **kwargs):
                raise AssertionError(
                    "ReductionNodeState constructed for "
                    f"{kwargs.get('rel_path')!r}: node deserialization must "
                    "never be reached for this blocked scenario"
                )
        else:
            def _tracking_node_state(*args, **kwargs):
                # Recorded by rel_path, not just a bare count: the caller
                # needs to know which container each recorded construction
                # belongs to, not just that construction happened at all.
                # document.py's one call site always uses keyword arguments,
                # so rel_path is always in kwargs.
                node_constructions.append(kwargs["rel_path"])
                return ReductionNodeState(*args, **kwargs)

        monkeypatch.setattr(
            "codedoc.core.document.ReductionNodeState", _tracking_node_state,
        )
        with pytest.raises(ConfigError) as blocked:
            run_pipeline(
                tmp_path,
                {
                    "entry_file": entry_rel,
                    "large_file_strategy": "split",
                    "max_content_chars": 2000,
                    "propagate_changes": False,
                    "output_dir": "docs",
                },
            )
        return blocked

    return _run, provider_creations, node_constructions, plan_calls, writer_constructions


def test_released_schema3_partial_blocks_before_node_deserialization(
    tmp_path, monkeypatch
) -> None:
    """A real `0.14.3` run must reject a released schema-3 container on its
    schema version alone, before its single node is ever deserialized,
    quarantined, or reaches planning, `SafeWriter`, or provider construction
    -- and must leave the recovery artifact byte-identical."""
    fixture = _load_fixture("partial_schema3_0_14_2.json")
    assert fixture["schema_version"] == 3
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(recovery_path, {_REL_PATH: fixture})
    original = recovery_path.read_bytes()

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(tmp_path, strict_no_node_construction=True)
    )
    blocked = run(monkeypatch)

    # Section 20A item 4: every early-boundary seam, not just the provider.
    assert node_constructions == []
    assert plan_calls == []
    assert writer_constructions == []
    assert provider_creations == []
    assert recovery_path.read_bytes() == original
    assert not (tmp_path / "docs" / "codedoc.json").exists()
    message = str(blocked.value)
    assert "unsupported" in message.lower()


def test_released_schema3_many_node_partial_never_reaches_quarantine_bound(
    tmp_path, monkeypatch
) -> None:
    """A released schema-3 container with more than
    `MAX_QUARANTINE_ENTRIES_PER_FILE` nodes is rejected by the same early
    schema check, before any of its nodes are inspected -- it must never
    raise `SplitRecoveryStateError` for exceeding the bounded quarantine
    path, because that path is never reached at all.

    This is an honest adversarial derivative of the real
    `partial_schema3_0_14_2.json` container (section 4A), not independently
    released-produced data: its envelope and first node are byte-identical
    to that real fixture, and only the 36 additional nodes appended after
    it are synthetic clones with unique `node_id`/`coverage_leaf_ids`.
    Asserted directly below, not merely claimed in a comment, so a future
    regeneration that accidentally re-synthesizes the first node too
    (exactly the section 20A item 3 defect this fixed) fails loudly."""
    fixture = _load_fixture("partial_schema3_many_nodes_0_14_2.json")
    assert fixture["schema_version"] == 3
    assert len(fixture["nodes"]) > 32
    real_single_node_fixture = _load_fixture("partial_schema3_0_14_2.json")
    assert fixture["nodes"][0] == real_single_node_fixture["nodes"][0], (
        "the many-node derivative's first node must be byte-for-byte "
        "identical to the real fixture's only node, not a synthetic clone"
    )
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(recovery_path, {_REL_PATH: fixture})
    original = recovery_path.read_bytes()

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(tmp_path, strict_no_node_construction=True)
    )
    blocked = run(monkeypatch)

    # The early per-container schema rejection is the failure reached, not
    # the bounded-quarantine invariant (which requires nodes to have been
    # deserialized at all).
    assert "bounded entry count" not in str(blocked.value)
    # Section 20A item 4: not one node of the 37 was ever constructed.
    assert node_constructions == []
    assert plan_calls == []
    assert writer_constructions == []
    assert provider_creations == []
    assert not (tmp_path / "docs" / "codedoc.json").exists()
    assert recovery_path.read_bytes() == original


@requires_structure_pack
def test_current_schema4_partial_still_resumes_normally(tmp_path) -> None:
    """The current schema-4 generation must still parse AND validate as a
    genuinely retainable, dependency-closed checkpoint under a real division
    plan -- not merely deserialize structurally -- proving the schema-3
    rejection did not disturb same-version behavior.

    The frozen fixture's digests are not placeholders: they were computed
    from the exact same deterministic `build_division_plan` /
    `build_reduction_tree` / `leaf_execution_identity` / `leaf_input_digest`
    calls this test reconstructs below, over the same 220-line source and
    `max_content_chars=2000` budget used throughout this module. Because
    division planning is deterministic (a core design invariant of this
    codebase), reconstructing that plan here byte-identically reproduces
    what generated the fixture, so `validate_recovered_tree` can genuinely
    retain it rather than only proving the JSON parses.

    Because the fixture's leaf digests bind `LEAF_CAPSULE_SCHEMA_REVISION`,
    every advance of that revision regenerates the affected fields from this
    same reconstruction, keeping the fixture a genuine *current*-generation
    checkpoint. `0.14.4` (`v6` -> `v7`) and `0.14.6` (`v7` -> `v8`) each
    touched only `execution_identity_digest` and `input_digest` on the one
    retained node, because only the leaf-capsule revision moved. `0.14.7`
    (`v8` -> `v9`) also advances `PACKER_SCHEMA_REVISION`
    (`division-packer-v5` -> `v6`) and `REDUCTION_PACKING_REVISION`
    (`reduction-packing-v4` -> `v5`) and adds the leaf-prompt
    signature-hint bound into the division-plan digest (section 5.4) --
    all three feed `division_plan_digest`/`reduction_tree_digest`, and the
    packer change also shifts where `pack_chunks` places a continuation
    boundary for this exact 220-line/2000-char-budget source, so the
    regenerated fixture's retained leaf carries a genuinely different
    `node_id` (chunk identity) this time, not merely refreshed digests
    under an unchanged chunk boundary. Every field below was produced by
    calling the four production functions named above -- none is
    hand-computed.

    That reconstruction runs the real parser (`structural_mode == "syntax"`
    for this source when the optional `structure` extra is installed), so
    it depends on that extra exactly as the frozen fixture's own digests do
    (section 20A item 1). A base install has no way to reproduce a
    syntax-mode digest, so this test skips there rather than failing for an
    environment it was never generated in; the schema-3/mixed-generation
    rejection tests above, which never reconstruct a division plan, remain
    unconditional and prove the base-install contract directly.
    """
    from codedoc.core.file_division import (
        build_division_plan,
        build_reduction_tree,
        provider_execution_identity,
        validate_recovered_tree,
    )
    from codedoc.core.loader import load_config

    fixture = _load_fixture("partial_schema4_0_14_3.json")
    assert fixture["schema_version"] == SPLIT_PARTIAL_SCHEMA_VERSION == 4
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(recovery_path, {_REL_PATH: fixture})

    document = read_codedoc_document(recovery_path, include_partial_files=True)

    assert len(document.partial_files) == 1
    state = document.partial_files[0]
    assert state.schema_version == 4
    assert state.rel_path == _REL_PATH
    assert tuple(state.by_id()) == (fixture["nodes"][0]["node_id"],)

    # Genuine semantic proof: reconstruct the identical real plan this
    # fixture was generated from, and confirm its one leaf node is actually
    # RETAINED (not quarantined) by the same topological validation a real
    # resume uses -- i.e. it is a real, currently-valid checkpoint, not just
    # a structurally-parseable document.
    source = _large_source()
    resolved_config = load_config(tmp_path, {"llm_provider": "openai", "model_name": "gpt-test"})
    provider_identity = provider_execution_identity(resolved_config)
    plan = build_division_plan(
        rel_path=_REL_PATH, language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(plan, max_content_chars=2000, language="python")
    assert plan.plan_digest == state.division_plan_digest
    assert tree.tree_digest == state.reduction_tree_digest

    retained, quarantine = validate_recovered_tree(
        state.nodes,
        plan=plan,
        tree=tree,
        content_hash=state.content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=resolved_config.get("_prompt_profile_digest", ""),
        imports_digest="",
        language="python",
    )
    assert quarantine == ()
    assert tuple(n.node_id for n in retained) == tuple(state.by_id())


def test_legacy_schema1_partial_blocks_preserve_first(tmp_path, monkeypatch) -> None:
    """The frozen predecessor ordered-prefix (schema version 1) container is
    migration-readable only: it fails closed with the D11
    preserve-or-move-aside remedy, never resumed or executed."""
    fixture = _load_fixture("partial_schema1_legacy.json")
    assert fixture["schema_version"] == 1
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(recovery_path, {_REL_PATH: fixture})
    original = recovery_path.read_bytes()

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(tmp_path)
    )
    blocked = run(monkeypatch)

    assert "schema version 1" in str(blocked.value)
    assert provider_creations == []
    assert recovery_path.read_bytes() == original


def test_dormant_schema2_partial_blocks_preserve_first(tmp_path, monkeypatch) -> None:
    """The dormant, per-node tree-digest-gated schema-2 generation remains
    recognized only enough to fail closed -- never executed or migrated."""
    fixture = _load_fixture("partial_schema2_dormant.json")
    assert fixture["schema_version"] == 2
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(recovery_path, {_REL_PATH: fixture})
    original = recovery_path.read_bytes()

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(tmp_path)
    )
    blocked = run(monkeypatch)

    assert "unsupported" in str(blocked.value).lower()
    assert provider_creations == []
    assert recovery_path.read_bytes() == original


def test_mixed_generation_document_rejected_as_a_whole(tmp_path, monkeypatch) -> None:
    """Mandatory case (section 4A, section 20A item 4, read literally): a
    recovery document whose `partial_files` map holds both a schema-3 and a
    schema-4 container is rejected as a whole when the unsupported schema-3
    entry is reached, with the rejected entry sorting *first* --
    `"legacy_large.py"` before `"present_large.py"` -- so the schema check
    fires before any node of either container is ever constructed or
    deserialized. The node-construction sentinel is strict here: it raises
    `AssertionError` immediately if `ReductionNodeState` is reached at all,
    which is the literal "install sentinels that fail on node
    construction/deserialization," not an indirect count checked after the
    fact. No recovery state is ever returned by the reader, no writer or
    provider is constructed, no final output is created, and the schema-4
    sibling is never partially accepted, scheduled, or resumed from this
    rejected artifact. Zero planning/writer/provider calls alone is not
    proof of any of that, so each is asserted directly."""
    fixture = _load_fixture("partial_mixed_generations.json")
    mixed = fixture["partial_files"]
    assert set(mixed) == {"src/legacy_large.py", "src/present_large.py"}
    assert mixed["src/legacy_large.py"]["schema_version"] == 3
    assert mixed["src/present_large.py"]["schema_version"] == 4
    # The rejected schema-3 entry must sort first, so _partial_files_from_meta's
    # sorted(partial_files) walk reaches and rejects it before the loop ever
    # gets to the valid schema-4 sibling -- this is what makes zero node
    # construction of any kind the correct, literal expectation below.
    assert sorted(mixed) == ["src/legacy_large.py", "src/present_large.py"]

    recovery_path = _prepare_recovery(
        tmp_path, rel_paths=("src/legacy_large.py", "src/present_large.py")
    )
    _splice_partial_files(recovery_path, mixed)
    original = recovery_path.read_bytes()

    # Direct proof that the reader itself never returns a recovery state for
    # this document -- not merely that some exception eventually surfaces
    # somewhere in the full run_pipeline call chain.
    with pytest.raises(ConfigError, match="unsupported"):
        read_codedoc_document(recovery_path, include_partial_files=True)
    assert recovery_path.read_bytes() == original

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(
            tmp_path, entry_rel="src/legacy_large.py", strict_no_node_construction=True
        )
    )
    blocked = run(monkeypatch)

    assert "unsupported" in str(blocked.value).lower()

    # No planning, writer, or provider construction, and no final output --
    # for either path.
    assert plan_calls == []
    assert writer_constructions == []
    assert provider_creations == []
    assert not (tmp_path / "docs" / "codedoc.json").exists()
    assert recovery_path.read_bytes() == original


def _current_first_variant(mixed: dict) -> dict:
    """Derive the current-first mixed-generation scenario in memory from the
    one section-4A-authorized fixture (`partial_mixed_generations.json`,
    which is rejected-first), rather than persisting a second fixture file
    -- section 4A authorizes exactly eight new fixture paths under
    `tests/fixtures/split_state/`, and a ninth is not one of them.

    Deep-copies *mixed*, then renames/rekeys its schema-4 container (whose
    key currently sorts *after* the schema-3 one) to a key that sorts
    *before* it, updating the container's own `rel_path` and its one node's
    `rel_path` to match -- consistently, so the derived document is
    internally coherent, not merely relabeled at the top level."""
    derived = copy.deepcopy(mixed)
    schema4_key = next(k for k, v in derived.items() if v["schema_version"] == 4)
    schema3_key = next(k for k, v in derived.items() if v["schema_version"] == 3)
    new_key = "src/current_large.py"
    assert new_key < schema3_key, "derived key must sort before the schema-3 entry"
    container = derived.pop(schema4_key)
    container["rel_path"] = new_key
    for node in container["nodes"]:
        node["rel_path"] = new_key
    derived[new_key] = container
    return derived


def test_mixed_generation_current_first_construction_is_never_scheduled(
    tmp_path, monkeypatch
) -> None:
    """Additional case, preserved alongside the mandatory one above, not
    substituted for it: when the valid schema-4 sibling instead sorts
    *first* (`"current_large.py"` before `"legacy_large.py"`), its one node
    is genuinely, correctly constructed before the loop reaches the entry
    that rejects the whole document. That construction is legitimate --
    proven here by name, not merely tolerated -- but the resulting
    `SplitTreeState`, though fully built, must never escape the reader, and
    the document as a whole must never be scheduled, written, or executed.

    Derives its document in memory from the one authorized mixed-generation
    fixture rather than a second on-disk file (see `_current_first_variant`)."""
    fixture = _load_fixture("partial_mixed_generations.json")
    mixed = _current_first_variant(fixture["partial_files"])
    assert set(mixed) == {"src/legacy_large.py", "src/current_large.py"}
    assert mixed["src/legacy_large.py"]["schema_version"] == 3
    assert mixed["src/current_large.py"]["schema_version"] == 4
    assert mixed["src/current_large.py"]["rel_path"] == "src/current_large.py"
    assert mixed["src/current_large.py"]["nodes"][0]["rel_path"] == "src/current_large.py"
    # "current_large.py" (schema 4, valid) sorts before "legacy_large.py"
    # (schema 3, rejected): confirmed by source inspection of
    # document.py::_partial_files_from_meta -- the rejected entry's schema
    # check runs before that entry's own `nodes` key is even read, and the
    # exception it raises propagates out before the function's
    # `return tuple(partials), tuple(legacy_rel_paths)` -- so the
    # already-built schema-4 SplitTreeState in the local `partials` list
    # never escapes the function despite existing momentarily.
    assert sorted(mixed) == ["src/current_large.py", "src/legacy_large.py"]

    recovery_path = _prepare_recovery(
        tmp_path, rel_paths=("src/legacy_large.py", "src/current_large.py")
    )
    _splice_partial_files(recovery_path, mixed)
    original = recovery_path.read_bytes()

    with pytest.raises(ConfigError, match="unsupported"):
        read_codedoc_document(recovery_path, include_partial_files=True)
    assert recovery_path.read_bytes() == original

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(tmp_path, entry_rel="src/legacy_large.py")
    )
    blocked = run(monkeypatch)

    assert "unsupported" in str(blocked.value).lower()

    # Exactly the valid schema-4 sibling's one node was constructed; nothing
    # belonging to the rejected schema-3 container was.
    assert "src/legacy_large.py" not in node_constructions
    assert node_constructions == ["src/current_large.py"]

    # No planning, writer, or provider construction, and no final output --
    # the genuinely-constructed node never gets scheduled, written, or
    # executed, for either path.
    assert plan_calls == []
    assert writer_constructions == []
    assert provider_creations == []
    assert not (tmp_path / "docs" / "codedoc.json").exists()
    assert recovery_path.read_bytes() == original


def test_malformed_current_schema_container_fails_closed(tmp_path, monkeypatch) -> None:
    """A structurally malformed current-schema (4) container -- here, an
    unknown extra field -- fails closed without mutating the recovery
    artifact or reaching the provider, exactly like every other unsupported
    or foreign container shape."""
    fixture = _load_fixture("malformed_container.json")
    assert fixture["schema_version"] == 4
    assert "unexpected_field" in fixture
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(recovery_path, {_REL_PATH: fixture})
    original = recovery_path.read_bytes()

    run, provider_creations, node_constructions, plan_calls, writer_constructions = (
        _run_blocked(tmp_path)
    )
    blocked = run(monkeypatch)

    assert "unknown or missing field" in str(blocked.value)
    assert provider_creations == []
    assert recovery_path.read_bytes() == original

# ===========================================================================
# Section 8: the predecessor (schema version 1) split-partial remedy that a
# resolved-valid split real-run (and, after this section, a split dry-run)
# reaches is bounded and JSON-escaped -- exact total, at most
# PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS retained paths each rendered as one
# ensure_ascii=True JSON string, the omitted count, and a full-stream
# details_digest over every legacy path. Diagnostic memory does not grow with
# the omitted path count and hostile path bytes cannot inject a terminal line.
# ===========================================================================

_S8_LEGACY_CAP = PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS


def _s8_legacy_container(rel_path: str) -> dict:
    return {
        "schema_version": 1,
        "owner": "codedoc-ai",
        "rel_path": rel_path,
        "completed_chunks": [["chunk_" + "a" * 58, '{"description": "legacy"}']],
    }


def test_s8_legacy_schema1_many_path_remedy_is_bounded_and_full_stream_digested(
    tmp_path, monkeypatch
) -> None:
    count = _S8_LEGACY_CAP + 6
    recovery_path = _prepare_recovery(tmp_path)
    containers = {
        f"pkg/legacy_{i:02d}.py": _s8_legacy_container(f"pkg/legacy_{i:02d}.py")
        for i in range(count)
    }
    _splice_partial_files(recovery_path, containers)
    original = recovery_path.read_bytes()

    run, provider_creations, _nodes, plan_calls, writer_constructions = _run_blocked(
        tmp_path
    )
    blocked = run(monkeypatch)
    message = str(blocked.value)

    ordered = sorted(containers)
    # Exact total, exact omitted, bounded retention.
    assert f"{count}" in message
    assert f"{count - _S8_LEGACY_CAP} more" in message
    rendered = [json.dumps(p, ensure_ascii=True) for p in ordered]
    for shown in rendered[:_S8_LEGACY_CAP]:
        assert shown in message
    # The omitted paths are NOT individually rendered.
    for hidden in rendered[_S8_LEGACY_CAP:]:
        assert hidden not in message
    # Full-stream digest over EVERY legacy path, in container sorted() order.
    assert canonical_stream_digest(ordered) in message
    assert canonical_stream_digest(ordered[:_S8_LEGACY_CAP]) not in message
    # Preserve-first wording and both remedies intact.
    assert "schema version 1" in message
    assert "predecessor" in message
    assert "re-run the exact predecessor CodeDoc build" in message
    assert "move" in message and "delete it" in message
    # Nothing mutated / planned / constructed.
    assert provider_creations == []
    assert plan_calls == []
    assert writer_constructions == []
    assert recovery_path.read_bytes() == original
    assert not (tmp_path / "docs" / "codedoc.json").exists()


def test_s8_legacy_schema1_remedy_rendered_message_stays_bounded_as_omitted_count_grows(
    tmp_path, monkeypatch
) -> None:
    """The RENDERED remedy string (not a memory measurement) stays bounded as
    the legacy-container size grows: at most PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS
    JSON-escaped paths, an exact total, an exact omitted count, and one
    full-stream digest -- so between a small and a very large container the
    message differs only by the integer/hex-digest fields, never by carrying
    more path text. The production structural contract behind it (no complete
    legacy-path list / set / joined string retained by
    ``_partial_files_from_meta``) is asserted directly."""
    from codedoc.core.document import _partial_files_from_meta

    def _message_for(n: int) -> tuple[str, object]:
        rp = _prepare_recovery(tmp_path / f"n{n}")
        containers = {
            f"pkg/legacy_{i:04d}.py": _s8_legacy_container(f"pkg/legacy_{i:04d}.py")
            for i in range(n)
        }
        _splice_partial_files(rp, containers)
        _partials, evidence = _partial_files_from_meta({"partial_files": containers})
        run, _pc, _nc, _plc, _wc = _run_blocked(tmp_path / f"n{n}")
        return str(run(monkeypatch).value), evidence

    small_n, large_n = _S8_LEGACY_CAP + 5, _S8_LEGACY_CAP + 380
    small, small_ev = _message_for(small_n)
    large, large_ev = _message_for(large_n)

    # Structural contract: only the bounded retained subset is materialised.
    for n, ev in ((small_n, small_ev), (large_n, large_ev)):
        assert ev.total == n
        assert len(ev.retained) == _S8_LEGACY_CAP <= PLAN_SUMMARY_DEFAULT_DETAIL_RECORDS
        assert ev.omitted == n - _S8_LEGACY_CAP
        assert isinstance(ev.retained, tuple)
        # digest is recomputed over EVERY path (sorted-container order), not the
        # retained subset -- and differs from the retained-only digest.
        every = sorted(f"pkg/legacy_{i:04d}.py" for i in range(n))
        assert ev.details_digest == canonical_stream_digest(every)
        assert ev.details_digest != canonical_stream_digest(list(ev.retained))

    # Rendered message: identical bounded path payload; only counts + digest move.
    assert large.count('"pkg/legacy_') == _S8_LEGACY_CAP
    assert small.count('"pkg/legacy_') == _S8_LEGACY_CAP
    assert f"{small_n}" in small and f"{large_n}" in large
    assert f"{small_ev.omitted} more" in small and f"{large_ev.omitted} more" in large
    assert small_ev.details_digest in small and large_ev.details_digest in large
    # The 400-path message is only longer by the integer/hex-digest fields.
    assert len(large) - len(small) < 40
    # preserve-first prose intact; no unbounded raw join.
    assert "re-run the exact predecessor CodeDoc build" in small
    assert "', '" not in large  # no raw comma-joined single-quoted path list


def test_s8_legacy_schema1_hostile_path_remedy_is_json_escaped_single_field(
    tmp_path, monkeypatch
) -> None:
    hostile = (
        "pkg/ev" + chr(10) + "il" + chr(9) + chr(0x1B) + chr(0x202E) + '"' + chr(0x5C) + ".py"
    )
    recovery_path = _prepare_recovery(tmp_path)
    _splice_partial_files(
        recovery_path,
        {hostile: _s8_legacy_container(hostile), "pkg/plain.py": _s8_legacy_container("pkg/plain.py")},
    )
    original = recovery_path.read_bytes()

    run, provider_creations, _nodes, _plan_calls, _writer = _run_blocked(tmp_path)
    message = str(run(monkeypatch).value)

    # No raw control / bidi / newline / tab from the path reaches the message.
    for raw_ch in (chr(10), chr(9), chr(0x1B), chr(0x202E)):
        assert raw_ch not in message
    # The hostile path appears exactly as one ensure_ascii JSON rendering of its
    # normalized form (normalize_rel_path collapses the lone backslash to '/').
    from codedoc.parser.source_structure import normalize_rel_path

    assert json.dumps(normalize_rel_path(hostile), ensure_ascii=True) in message
    assert json.dumps("pkg/plain.py", ensure_ascii=True) in message
    # Exactly one rendered field per retained path -> no injected extra line.
    assert message.count("\n") == 0
    assert "schema version 1" in message
    assert provider_creations == []
    assert recovery_path.read_bytes() == original
