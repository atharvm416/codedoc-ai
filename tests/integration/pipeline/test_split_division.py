"""End-to-end split-mode pipeline behavior: leaf/reduction/final call
sequencing, retry/correction honesty, rate-limit step-down, terminal-failure
checkpointing, node-keyed recovery resumption, and the D8 whole-run-abort
contract for a genuine ``DivisionInternalDefect``.

Split is valid only in ``analysis_mode: 'single'`` (D2); every scenario here
uses single mode only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from pathlib import Path

import pytest
import codedoc.core.file_division as file_division
from codedoc.core.db import read_source_snapshot
from codedoc.core.result_assembly import flat_combined_result
from codedoc.core.execution import _process_descriptor_batch
from codedoc.core.execution_model import build_call_manifest
from codedoc.core.file_division import (
    BLOCKED_REASON_ORDER,
    MAX_CHUNKS_PER_FILE,
    MAX_LEAF_EXPORT_ITEMS,
    MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
    MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    DivisionInternalDefect,
    SplitCapacityBlocked,
    SplitTreeState,
    build_division_plan,
    build_fact_ledger,
    build_reduction_tree,
    canonical_json,
    deterministic_imports_digest,
    final_execution_identity,
    final_input_digest,
    final_synthesis_input,
    leaf_execution_identity,
    leaf_input_digest,
    provider_execution_identity,
    reduction_depth,
    reduction_execution_identity,
    reduction_input_digest,
    refine_narrative_inputs,
    split_plan_leaf_descriptors,
    split_plan_unit_summaries,
    tree_node_state,
    validate_recovered_tree,
)
from codedoc.core.loader import DEFAULTS, load_config
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    resolve_profile_source,
)
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import RecoveryState, build_recovery_identity
from codedoc.core.safe_writer import SafeWriter
from codedoc.llm.factory import (
    ProviderExecutionDescriptor,
    attest_provider_execution,
)
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import (
    ConfigError,
    ErrorReporter,
    LLMError,
    UnrecoverableProviderError,
)
from tests.support.execution_requests import (
    make_execution_request,
    make_execution_requests,
)
from tests.support.providers import SmartFake
from tests.support.pipeline_scenarios import write_existing_json
from tests.support.profiles import INLINE
from tests.support.provider_failures import provider_failure_error
from tests.support.structure_extra import requires_structure_pack


def _large_python_source(lines: int = 220) -> str:
    # Named function declarations, not bare top-level statements: under
    # syntax-mode parsing, adjacent bare statements with no intervening
    # declaration can merge into one shared "gap" unit spanning the whole
    # fixture, producing wildly different (and here, capacity-blocked or
    # retry-budget-exhausting) chunk counts than under lexical fallback. A
    # `def` is reliably its own semantic unit under both parsing modes.
    return "\n".join(f"def fn_{index}(): return {index}" for index in range(lines)) + "\n"


def _realistic_service_source(methods: int = 140) -> str:
    return (
        "class ApplicationService:\n"
        '    """Coordinates a representative application workflow."""\n\n'
        + "".join(
            (
                f"    def operation_{index:03d}(self, value: int) -> int:\n"
                f"        normalized = value + {index}\n"
                "        return normalized\n\n"
            )
            for index in range(methods)
        )
    )


def test_split_dry_run_reuses_completed_split_output_like_a_second_real_run(
    tmp_path, monkeypatch
) -> None:
    """Section 5.8: for the resolved-valid single+split route, dry-run is a
    read-only preview of the SAME payable work a real run would do at the same
    repository state. A completed split record in the stable output is reused,
    never re-planned as fresh split work, and the dry-run payable manifest is
    field-for-field identical to a second real run's (which also does nothing).
    """
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    cfg = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
    }
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    first = run_pipeline(tmp_path, cfg)
    assert first["checked"] == 1
    assert first["failed"] == 0
    assert first["split_divided_files"] == 1
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("a reuse-only run created a provider"),
    )
    dry = run_pipeline(tmp_path, {**cfg, "dry_run": True})
    second_real = run_pipeline(tmp_path, cfg)

    comparable = (
        "documentation_calls_planned",
        "total_calls_planned",
        "file_documentation_calls_planned",
        "unit_documentation_calls_planned",
        "file_reduction_calls_planned",
        "synthesis_calls_planned",
        "split_divided_files",
        "split_completed_files_reused",
        "call_manifest_digest",
    )
    assert {k: dry[k] for k in comparable} == {k: second_real[k] for k in comparable}
    # Reuse, not fresh split work.
    assert dry["total_calls_planned"] == 0
    assert dry["unit_documentation_calls_planned"] == 0
    assert dry["split_divided_files"] == 0
    assert dry["split_completed_files_reused"] == 1
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()


class _SignedLeafFake(SmartFake):
    """`SmartFake`, but the one leaf fragment response that actually
    documents `fn_0` carries a bounded, deliberately-wrong model-returned
    signature for it -- exercising the section 2A response-cap boundary,
    and section 20A item 2's parser-authority proof, through a real split
    execution with response correction disabled.

    Checking for `"fn_0("` in the prompt (rather than reacting to every
    fragment prompt alike) matters here: `_large_python_source` splits
    into multiple leaf chunks, and `fn_0` is a real declaration living in
    only one of them. A fake that claimed to be describing `fn_0` for
    every chunk would make every *other* chunk's response an unmatched
    claim (parser authority can only attribute a symbol within its own
    chunk's scope), and `build_fact_ledger`'s cross-chunk merge would then
    let one of those unmatched, wrong-scope claims silently overwrite the
    one genuinely-matched, correctly-authoritative entry -- collapsing the
    very distinction this test exists to prove, for a reason that has
    nothing to do with parser authority itself."""

    def __init__(self, signature_chars: int, verdict="SAFE") -> None:
        super().__init__(verdict)
        self._signature = "s" * signature_chars

    def complete_json(self, prompt, system=""):
        if "This is one bounded fragment of a larger" in prompt and "fn_0(" in prompt:
            self.doc_calls += 1
            return json.dumps({
                "description": "A fragment.",
                "functions": [
                    {"name": "fn_0", "description": "does f", "signature": self._signature}
                ],
            })
        return super().complete_json(prompt, system)


def test_bounded_leaf_signature_survives_real_split_execution_and_stays_private(
    tmp_path, monkeypatch
) -> None:
    """Section 2A: with response correction disabled (the default), a real
    split execution accepts a bounded 552-character model-returned leaf
    signature -- the installed 0.14.2 TestPyPI observation -- and never
    publishes it: the private signature is internal ledger/matching
    metadata only and must not reach the final public JSON output.

    Parser-independent: this holds regardless of structural mode (the
    private-field bound and the public-output projection are the same in
    lexical fallback and syntax mode), so unlike its sibling below it
    carries no `@requires_structure_pack` and must run in a base install."""
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    provider = _SignedLeafFake(552)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
            "response_correction_enabled": False,
        },
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    output = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    assert "signature" not in output
    assert "s" * 552 not in output


@requires_structure_pack
def test_bounded_leaf_signature_ledger_uses_parser_authority_not_the_model_hint(
    tmp_path, monkeypatch
) -> None:
    """Section 20A item 2: public-field absence alone (proven unconditionally
    by the sibling test above) does not prove the internal ledger held
    parser authority -- a `run_pipeline` regression that stopped passing
    `division_plan.symbols` into `build_fact_ledger` (so the model's own
    hint became authoritative by default) would still leave "signature" and
    the 552-character string out of public JSON, since neither is ever a
    public field regardless of which value won internally. This
    instruments the exact `build_fact_ledger` seam
    `execution.py::_execute_divided_file` calls during this real
    `run_pipeline` execution -- wrapping, not stubbing, so the genuine
    ledger the pipeline actually produces and uses is captured, not a
    substitute computed separately -- and asserts the captured ledger's
    `fn_0` fact carries the real parser-owned signature (independently
    reconstructed from the same deterministic division plan `run_pipeline`
    itself builds for this source/budget), not the fake provider's
    552-character hint.

    Presence-dependent, unlike its sibling: `SymbolFact` -- the
    parser-owned fact parser authority allocates from -- is a real,
    grammar-derived concept that lexical fallback does not produce, so
    `reference_plan.symbols` is empty in a base install and this specific
    proof has nothing to compare against there (confirmed directly: without
    `@requires_structure_pack` this failed with `StopIteration` in a fresh
    base-env, not a false pass -- the absence-simulation contract the
    sibling test proves is unaffected)."""
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    provider = _SignedLeafFake(552)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    captured_ledgers = []
    real_build_fact_ledger = build_fact_ledger

    def _capturing_build_fact_ledger(*args, **kwargs):
        ledger = real_build_fact_ledger(*args, **kwargs)
        captured_ledgers.append(ledger)
        return ledger

    monkeypatch.setattr(
        "codedoc.core.execution.build_fact_ledger", _capturing_build_fact_ledger
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
            "response_correction_enabled": False,
        },
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0

    # Deterministic division planning (a core design invariant of this
    # codebase) means reconstructing the plan here, over the identical
    # source and budget, byte-identically reproduces what run_pipeline
    # itself built -- including division_plan.symbols, the parser-owned
    # facts execution.py passes into build_fact_ledger.
    reference_plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000,
    )
    assert reference_plan.structural_mode == "syntax"
    real_signature = next(
        symbol.signature
        for symbol in reference_plan.symbols
        if symbol.qualified_name == "fn_0"
    )
    assert real_signature != "s" * 552

    assert captured_ledgers, "build_fact_ledger was never called by run_pipeline"
    matching_facts = [
        fact for ledger in captured_ledgers for fact in ledger.functions
        if fact["name"] == "fn_0"
    ]
    assert matching_facts, "no captured ledger fact named fn_0"
    for fact in matching_facts:
        assert fact["signature"] == real_signature
        assert fact["signature"] != "s" * 552


# ---------------------------------------------------------------------------
# 0.14.7 section 8.D / 9.1 items 5 & 15 / mutation check 22: the mandatory
# B=1000 oversized-declaration matrix across exact lengths 1,001 / 1,500 /
# 2,000 / 2,001, on BOTH the syntax-divided and lexical-fallback paths, with a
# true literal-fragment provider double.
# ---------------------------------------------------------------------------

_OVERSIZED_DECL_NAME = "oversized_declaration"


def _oversized_declaration_source(length: int, filler_defs: int = 24) -> str:
    """One real single-line Python declaration of EXACTLY *length* code points
    -- `def oversized_declaration(<one long parameter identifier>) -> int:` --
    followed by enough tiny `def` declarations to force the split path under
    both structural modes."""
    head, tail = f"def {_OVERSIZED_DECL_NAME}(", ") -> int:"
    pad = length - len(head) - len(tail)
    assert pad >= 1, length
    declaration_line = head + ("p" * pad) + tail
    assert len(declaration_line) == length and "\n" not in declaration_line
    filler = "\n".join(f"def fn_{index}(): return {index}" for index in range(filler_defs))
    return f"{declaration_line}\n    return 0\n\n{filler}\n"


def _visible_fragment_source(prompt: str) -> str | None:
    """Extract the literal fragment payload from either the initial fragment
    prompt (`Visible fragment source:`) or the fixed-capsule correction prompt
    (`Code:`); both end the payload at the start of `_FRAGMENT_SHAPE_BLOCK`."""
    for marker in ("Visible fragment source:\n", "\nCode:\n"):
        if marker in prompt:
            after = prompt.split(marker, 1)[1]
            return after.split(
                "\n\nReturn exactly this fixed internal JSON shape", 1
            )[0]
    return None


class _LiteralFragmentProvider(SmartFake):
    """A true literal-fragment provider (section 8.D).

    It receives NO whole source and NO whole declaration. For every fixed
    split-leaf call (initial or its one correction) it answers *only* from the
    literal fragment text passed to that invocation: any returned ``signature``
    is a contiguous prefix of that fragment's first line, and a fragment with
    no visibly-declared name yields a description only. It cannot inspect
    parser facts, another continuation, or an invisible parameter.

    ``invalid_first`` makes the first response for the oversized declaration's
    header fragment a deliberately over-bound value, to drive the real
    correction component; the correction answer follows the same
    literal-fragment rule.
    """

    _HEAD_RE = re.compile(r"\s*((?:async\s+)?def)\s+([A-Za-z_]\w*)\s*\(")

    def __init__(self, verdict: str = "SAFE", *, invalid_first: bool = False) -> None:
        super().__init__(verdict)
        self._invalid_first = invalid_first
        self.invalid_first_fired = False
        self.correction_calls = 0
        self.leaf_prompts_seen = 0
        self.pairs: list[tuple[str, dict]] = []  # (fragment_payload, answer)

    def _answer_from_fragment(self, payload: str) -> dict:
        first_line = payload.split("\n", 1)[0]
        match = self._HEAD_RE.match(first_line)
        if not match:
            return {"description": "Interior of a larger declaration; no name visible."}
        name = match.group(2)
        signature = first_line[: MAX_LEAF_SYMBOL_SIGNATURE_CHARS]
        assert signature and signature in payload  # verbatim contiguous, by construction
        return {
            "description": "Header of a declaration visible in this fragment.",
            "functions": [
                {"name": name, "description": "does x", "signature": signature}
            ],
        }

    def complete_json(self, prompt, system=""):
        is_initial = "This is one bounded fragment of a larger" in prompt
        is_correction = (
            "did not satisfy the required JSON contract" in prompt
            and "Return exactly this fixed internal JSON shape" in prompt
            and "Refine one combined narrative" not in prompt
        )
        if not (is_initial or is_correction):
            return super().complete_json(prompt, system)

        payload = _visible_fragment_source(prompt)
        assert payload is not None
        self.leaf_prompts_seen += 1
        # the provider observes the contract exactly once, as 8.D requires.
        assert prompt.count("Signature contract for the optional") == 1
        assert "preferably roughly 600-1,000 characters" in prompt

        header_fragment = payload.lstrip().startswith(f"def {_OVERSIZED_DECL_NAME}(")
        if is_correction:
            self.correction_calls += 1
        elif self._invalid_first and header_fragment and not self.invalid_first_fired:
            self.invalid_first_fired = True
            self.doc_calls += 1
            return json.dumps({
                "description": "deliberately over-bound",
                "functions": [
                    {
                        "name": _OVERSIZED_DECL_NAME,
                        "signature": "s" * (MAX_LEAF_SYMBOL_SIGNATURE_CHARS + 1),
                    }
                ],
            })

        answer = self._answer_from_fragment(payload)
        self.pairs.append((payload, answer))
        self.doc_calls += 1
        return json.dumps(answer)


_OVERSIZED_MATRIX_LENGTHS = (1001, 1500, 2000, 2001)
_OVERSIZED_MATRIX_MODES = ("syntax", "lexical")


@pytest.mark.parametrize("length", _OVERSIZED_MATRIX_LENGTHS)
@pytest.mark.parametrize("mode", _OVERSIZED_MATRIX_MODES)
def test_oversized_declaration_matrix_completes_with_literal_fragment_provider(
    tmp_path, monkeypatch, length, mode
) -> None:
    """Section 8.D / 9.1 item 15: an exact-length oversized declaration
    completes on both the syntax and lexical `B=1000` split paths under a true
    literal-fragment provider. Every returned signature is verbatim-visible in
    its own fragment; a fragment with no visible name reconstructs nothing;
    the signature contract appears once per initial leaf prompt; parser-owned
    identity attaches facts; no signature reaches public output; clean
    completion removes `crash_recovery.json`."""
    if mode == "lexical":
        monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    source = _oversized_declaration_source(length)
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    declaration_line = next(
        line for line in source.splitlines() if line.startswith(f"def {_OVERSIZED_DECL_NAME}")
    )
    assert len(declaration_line) == length

    provider = _LiteralFragmentProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    captured_ledgers: list = []
    real_build_fact_ledger = build_fact_ledger

    def _capturing(*args, **kwargs):
        ledger = real_build_fact_ledger(*args, **kwargs)
        captured_ledgers.append(ledger)
        return ledger

    monkeypatch.setattr("codedoc.core.execution.build_fact_ledger", _capturing)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 1000,
            "output_format": "both",
            "parallel_agents": False,
            "propagate_changes": False,
            "response_correction_enabled": False,
        },
    )

    reference_plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=1000
    )
    assert reference_plan.structural_mode == mode
    assert stats["checked"] == 1 and stats["failed"] == 0
    assert provider.leaf_prompts_seen > 0

    # every returned signature is a contiguous substring of its own fragment.
    decl_header_answers = []
    decl_interior_answers = []
    for payload, answer in provider.pairs:
        for entry in answer.get("functions", []):
            assert entry["signature"] in payload
            assert len(entry["signature"]) <= MAX_LEAF_SYMBOL_SIGNATURE_CHARS
        first_line = payload.split("\n", 1)[0]
        if first_line.startswith(f"def {_OVERSIZED_DECL_NAME}("):
            decl_header_answers.append((payload, answer))
        elif _OVERSIZED_DECL_NAME not in first_line and (
            "int" in first_line or ") -> int:" in first_line or first_line.strip() == ""
        ):
            decl_interior_answers.append((payload, answer))

    # the header fragment reports the visible name + a leading source-order
    # signature; it is a genuine prefix of the real declaration line.
    assert decl_header_answers, "the oversized declaration header fragment was never seen"
    for payload, answer in decl_header_answers:
        entry = next(e for e in answer["functions"] if e["name"] == _OVERSIZED_DECL_NAME)
        assert declaration_line.startswith(entry["signature"])
        assert entry["signature"] == payload.split("\n", 1)[0][:MAX_LEAF_SYMBOL_SIGNATURE_CHARS]

    # an interior/continuation fragment invents no name and reports no
    # functions for the oversized declaration.
    for _payload, answer in decl_interior_answers:
        assert not any(
            e["name"] == _OVERSIZED_DECL_NAME for e in answer.get("functions", [])
        )

    # parser-owned identity, not the model hint, controls fact attachment.
    facts = [
        fact
        for ledger in captured_ledgers
        for fact in ledger.functions
        if fact["name"] == _OVERSIZED_DECL_NAME
    ]
    assert len(facts) == 1, facts  # merged to exactly one across continuations
    if mode == "syntax":
        parser_signature = next(
            s.signature
            for s in reference_plan.symbols
            if s.qualified_name == _OVERSIZED_DECL_NAME
        )
        assert facts[0].get("signature") == parser_signature
    else:
        assert reference_plan.symbols == ()

    output_json = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    output_md = (tmp_path / "codedoc" / "codedoc.md").read_text(encoding="utf-8")
    assert '"signature"' not in output_json
    assert "signature" not in output_md.lower()
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()


@pytest.mark.parametrize("length", _OVERSIZED_MATRIX_LENGTHS)
@pytest.mark.parametrize("mode", _OVERSIZED_MATRIX_MODES)
def test_oversized_declaration_matrix_reaches_real_correction(
    tmp_path, monkeypatch, length, mode
) -> None:
    """Section 8.D: every declaration length reaches the real correction
    component on both structural modes. The oversized declaration's header
    fragment gets a deliberately over-bound initial response; the real
    `ResponseCorrectionAgent` receives a prompt carrying the same signature
    contract once and accepts a literal-fragment replacement; the file still
    completes and publishes no signature."""
    if mode == "lexical":
        monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    source = _oversized_declaration_source(length)
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")

    provider = _LiteralFragmentProvider(invalid_first=True)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 1000,
            "output_format": "both",
            "parallel_agents": False,
            "propagate_changes": False,
            "response_correction_enabled": True,
        },
    )

    reference_plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=1000
    )
    assert reference_plan.structural_mode == mode
    assert stats["checked"] == 1 and stats["failed"] == 0
    assert provider.invalid_first_fired is True
    assert provider.correction_calls >= 1

    output_json = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    assert '"signature"' not in output_json
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()


def _reduction_total(tree) -> int:
    return len(tree.unit_consolidation_nodes) + len(tree.general_nodes)


def _provider_identity_for(tmp_path, overrides: dict) -> str:
    """The exact provider-free identity `run_pipeline` will compute for this
    config — resolved the same way planning resolves it, so a hand-built
    recovery checkpoint validates as current."""
    resolved = load_config(tmp_path, overrides)
    return provider_execution_identity(resolved)


def _fully_completed_tree_state(
    plan,
    tree,
    *,
    provider_identity: str,
    content_hash: str,
    prompt_profile_digest: str = NO_PROMPT_PROFILE_DIGEST,
    final_fields: dict | None = None,
    imports: tuple[str, ...] = (),
) -> SplitTreeState:
    """A synthetic but dependency-valid SplitTreeState covering every leaf,
    reduction, and final node — as if the whole tree had already been paid
    for and checkpointed in an earlier run.

    Every stage-local input digest is recomputed from the exact fixture
    result content of its own retained children (mirroring the live
    executor's narrative extraction and final-manifest assembly exactly),
    since ``validate_recovered_tree`` now recomputes and compares this
    digest rather than trusting a stored claim (section 11)."""
    results_by_id: dict[str, dict] = {}
    nodes = []
    for index, chunk in enumerate(plan.chunks):
        result = {
            "description": f"restored {index}",
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
            tree_node_state(
                node_id=node.node_id,
                node_type=node.phase,
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                input_digest=reduction_input_digest(
                    rel_path=plan.rel_path,
                    phase=node.phase,
                    level=node.level,
                    unit_id=node.unit_id,
                    child_count=len(node.child_ids),
                    ordered_child_narratives=refine_narrative_inputs(raw_narratives),
                ),
                execution_identity_digest=reduction_execution_identity(
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
    imports_digest = file_division.deterministic_imports_digest(imports)
    leaf_capsules_ordered = [results_by_id[chunk.chunk_id] for chunk in plan.chunks]
    ledger = build_fact_ledger(
        leaf_capsules_ordered,
        language="python",
        chunks=plan.chunks,
        symbols=plan.symbols,
    )
    final_raw_narratives = tuple(
        results_by_id[child_id].get(
            "narrative", results_by_id[child_id].get("description", "")
        )
        for child_id in final.child_ids
    )
    manifest_json = final_synthesis_input(
        rel_path=plan.rel_path,
        language="python",
        imports=imports,
        root_narratives=refine_narrative_inputs(final_raw_narratives),
        root_coverage_leaf_ids=final.leaf_ids,
        ledger=ledger,
        max_chars=plan.source_budget_chars,
    )
    nodes.append(
        tree_node_state(
            node_id=final.node_id,
            node_type="final",
            rel_path=plan.rel_path,
            content_hash=content_hash,
            division_plan_digest=plan.plan_digest,
            input_digest=final_input_digest(
                imports_digest=imports_digest,
                resolved_shape_digest=prompt_profile_digest,
                manifest_json=manifest_json,
            ),
            execution_identity_digest=final_execution_identity(
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
            result=flat_combined_result(
                plan.rel_path,
                "python",
                list(imports),
                final_fields or {"description": "restored complete file"},
            ),
        )
    )
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(nodes),
    )


def test_equal_length_import_change_preserves_leaves_and_reducers_only() -> None:
    # The frozen source/division payload is unchanged while the exact parser-
    # derived import tuple changes. This isolates the final-only imports input
    # from an edit that also changes a leaf payload (which would correctly
    # invalidate that leaf through leaf_input_digest).
    source = _large_python_source()
    before_plan = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    after_plan = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    before_tree = build_reduction_tree(
        before_plan,
        max_content_chars=2000,
        language="python",
        imports=("alpha",),
    )
    after_tree = build_reduction_tree(
        after_plan,
        max_content_chars=2000,
        language="python",
        imports=("bravo",),
    )
    assert before_plan.plan_digest == after_plan.plan_digest
    assert before_tree.tree_digest == after_tree.tree_digest
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    provider_identity = "provider-execution:" + "b" * 64
    recovered = _fully_completed_tree_state(
        before_plan,
        before_tree,
        provider_identity=provider_identity,
        content_hash=content_hash,
        imports=("alpha",),
    )

    retained, quarantine = validate_recovered_tree(
        recovered.nodes,
        plan=after_plan,
        tree=after_tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=NO_PROMPT_PROFILE_DIGEST,
        imports_digest=deterministic_imports_digest(("bravo",)),
        imports=("bravo",),
        language="python",
    )

    retained_ids = {node.node_id for node in retained}
    assert retained_ids == {
        *(chunk.chunk_id for chunk in after_plan.chunks),
        *(node.node_id for node in after_tree.all_intermediate_nodes),
    }
    assert tuple(entry.node_id for entry in quarantine) == (
        after_tree.final_node.node_id,
    )


def _one_leaf_completed_tree_state(plan, tree, *, provider_identity: str, content_hash: str) -> SplitTreeState:
    leaf_identity = leaf_execution_identity(
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        provider_identity=provider_identity,
        chunk=plan.chunks[0],
    )
    node = tree_node_state(
        node_id=plan.chunks[0].chunk_id,
        node_type="leaf",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        input_digest=leaf_input_digest(
            rel_path=plan.rel_path,
            language="python",
            chunk=plan.chunks[0],
            unit_indexes=plan.unit_positions(plan.chunks[0]),
            unit_count=len(plan.units),
        ),
        execution_identity_digest=leaf_identity,
        unit_id=None,
        child_ids=(),
        coverage_leaf_ids=(plan.chunks[0].chunk_id,),
        result={
            "description": "restored",
            "chunk_id": plan.chunks[0].chunk_id,
            "unit_id": plan.chunks[0].unit_id,
        },
    )
    return SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path=plan.rel_path,
        content_hash=content_hash,
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=(node,),
    )


def test_split_division_manifest_counts_leaves_reduction_and_synthesis() -> None:
    source = "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n"
    plan = build_division_plan(
        rel_path="src/large.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)

    manifest = build_call_manifest(
        [],
        ["src/large.py"],
        "single",
        division_plans={"src/large.py": plan},
        reduction_trees={"src/large.py": tree},
    )

    categories = [call.category for call in manifest.calls]
    assert categories.count("unit-documentation") == len(plan.chunks)
    assert categories.count("file-reduction") == _reduction_total(tree)
    assert categories.count("file-synthesis") == 1
    assert not any(category == "file-documentation" for category in categories)


def test_split_pipeline_documents_all_chunks_then_synthesizes(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    output = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    record = output["files"][0]
    assert stats["failed"] == 0
    assert stats["checked"] == 1
    assert provider.doc_calls == stats["documentation_calls_attempted"]
    assert stats["unit_documentation_calls_planned"] == len(division.chunks)
    assert stats["file_reduction_calls_planned"] == _reduction_total(tree)
    assert stats["synthesis_calls_planned"] == 1
    assert stats["split_chunks"] == len(division.chunks)
    assert "division" not in record
    assert "documentation_units" not in record
    assert record["_large_file_identity"].startswith("large-file-v3:")
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


@requires_structure_pack
def test_realistic_large_class_plans_and_executes_with_proportional_calls(
    tmp_path, monkeypatch
) -> None:
    source = _realistic_service_source()
    (tmp_path / "service.py").write_text(
        source,
        encoding="utf-8",
        newline="",
    )
    config = {
        "entry_file": "service.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 12000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    division = build_division_plan(
        rel_path="service.py",
        language="python",
        content=source,
        source_budget_chars=12000,
    )
    tree = build_reduction_tree(division, max_content_chars=12000)

    assert len(source) > 12000
    assert 2 <= len(division.chunks) <= (len(source) + 11999) // 12000 + 1

    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})
    assert dry_stats["split_blocked_files"] == 0
    assert dry_stats["split_chunks"] == len(division.chunks)
    assert dry_stats["unit_documentation_calls_planned"] == len(
        division.chunks
    )
    assert not (tmp_path / "docs").exists()

    provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: provider,
    )
    stats = run_pipeline(tmp_path, config)

    assert stats["failed"] == 0
    assert stats["checked"] == 1
    assert stats["split_chunks"] == len(division.chunks)
    assert stats["file_reduction_calls_planned"] == _reduction_total(tree)
    assert stats["synthesis_calls_planned"] == 1
    assert provider.doc_calls == stats["documentation_calls_attempted"]


def test_constructed_provider_identity_mismatch_aborts_before_any_call(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "main.py").write_text(
        "def main():\n    return 1\n",
        encoding="utf-8",
        newline="",
    )
    provider = SmartFake()
    provider._codedoc_provider_execution_descriptor = (
        ProviderExecutionDescriptor(
            provider_kind="anthropic",
            model="different-model",
            endpoint_identity="provider-default",
        )
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: provider,
    )

    with pytest.raises(ConfigError, match="does not match the provider-free plan"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "analysis_mode": "single",
                "large_file_strategy": "split",
                "parallel_agents": False,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    assert provider.doc_calls == 0


def test_split_leaf_retry_repeats_only_the_incomplete_leaf(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)

    class FailSecondLeafOnce(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self.leaf_prompts: list[str] = []
            self.failed = False

        def complete_json(self, prompt, system=""):
            if "This is one bounded fragment of a larger" in prompt:
                self.leaf_prompts.append(prompt)
                if len(self.leaf_prompts) == 2 and not self.failed:
                    self.failed = True
                    self.doc_calls += 1
                    raise LLMError("openai", "temporary provider outage")
            return super().complete_json(prompt, system)

    provider = FailSecondLeafOnce()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert provider.leaf_prompts[1] == provider.leaf_prompts[2]
    assert provider.leaf_prompts.count(provider.leaf_prompts[0]) == 1
    assert provider.leaf_prompts.count(provider.leaf_prompts[1]) == 2
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned
    assert stats["failed_calls"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_split_synthesis_retry_does_not_repeat_completed_leaves(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)

    class FailSynthesisOnce(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self.leaf_prompts: list[str] = []
            self.synthesis_prompts: list[str] = []

        def complete_json(self, prompt, system=""):
            if "This is one bounded fragment of a larger" in prompt:
                self.leaf_prompts.append(prompt)
            if "Synthesize one final file-level documentation JSON object" in prompt:
                self.synthesis_prompts.append(prompt)
                if len(self.synthesis_prompts) == 1:
                    self.doc_calls += 1
                    raise LLMError("openai", "temporary provider outage")
            return super().complete_json(prompt, system)

    provider = FailSynthesisOnce()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert len(provider.leaf_prompts) == len(division.chunks)
    assert len(set(provider.leaf_prompts)) == len(division.chunks)
    assert len(provider.synthesis_prompts) == 2
    assert provider.synthesis_prompts[0] == provider.synthesis_prompts[1]
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned
    assert stats["failed_calls"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_terminal_split_failure_leaves_later_planned_calls_unattempted(
    tmp_path, monkeypatch
) -> None:
    """Section 5.5's fourth accounting proof: a permanent early split failure
    (a terminal provider fault on the first leaf) aborts the run before any
    later leaf, reduction, or final call is ever attempted -- those calls
    are correctly accounted as planned but unattempted, not silently dropped
    or miscounted as failed."""
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    assert len(division.chunks) > 1, "test requires more than one leaf"

    class TerminalOnFirstLeaf(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self.leaf_prompts: list[str] = []

        def complete_json(self, prompt, system=""):
            if "This is one bounded fragment of a larger" in prompt:
                self.leaf_prompts.append(prompt)
                if len(self.leaf_prompts) == 1:
                    self.doc_calls += 1
                    raise provider_failure_error(
                        "openai", "provider-quota-exhausted", status=429
                    )
            return super().complete_json(prompt, system)

    provider = TerminalOnFirstLeaf()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    with pytest.raises(UnrecoverableProviderError) as caught:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "max_parallel_files": 1,
                "file_retry_attempts": 0,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    stats = caught.value.stats
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == 1
    assert stats["planned_calls_not_attempted"] == planned - 1
    assert (tmp_path / "docs" / "crash_recovery.json").exists()


def test_split_response_correction_uses_the_originating_leaf_call(
    tmp_path, monkeypatch
) -> None:
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)

    class CorrectFirstLeaf(SmartFake):
        def __init__(self) -> None:
            super().__init__()
            self.failed_initial = False
            self.correction_calls = 0

        def complete_json(self, prompt, system=""):
            if "Previous response (verbatim" in prompt:
                self.correction_calls += 1
                return super().complete_json(prompt, system)
            if (
                "This is one bounded fragment of a larger" in prompt
                and "File: main.py" in prompt
                and not self.failed_initial
            ):
                self.failed_initial = True
                self.doc_calls += 1
                return json.dumps({"functions": ["missing required description"]})
            return super().complete_json(prompt, system)

    provider = CorrectFirstLeaf()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    callback_calls = []
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "parallel_agents": False,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 1,
            "response_correction_enabled": True,
            "propagate_changes": False,
            "output_dir": "docs",
        },
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert provider.correction_calls == 1
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned + 1
    assert stats["failed_calls"] == 0
    assert stats["response_contract_failures"] == 1
    assert stats["response_correction_calls_attempted"] == 1
    assert stats["response_correction_calls_succeeded"] == 1
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert callback_calls == []
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()
    assert json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]["description"] == "A file."


def test_mixed_ordinary_and_divided_files_resume_after_rate_limit_step_down(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "small.py").write_text(
        "def small_value():\n    return 1\n", encoding="utf-8"
    )
    # Smaller than the shared _large_python_source() default: this test
    # exercises genuine two-thread concurrency (large.py and small.py process
    # simultaneously before the rate-limit step-down), so it deliberately
    # minimizes large.py's own checkpoint-write volume — just enough to
    # require 2+ chunks under both parsing modes — rather than adding
    # unrelated I/O contention on top of the concurrency already under test.
    source = _large_python_source(120)
    (tmp_path / "large.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="large.py", language="python", content=source, source_budget_chars=2000
    )

    class RateLimitSecondLargeLeaf(SmartFake):
        provider_name = "openai"

        def __init__(self) -> None:
            super().__init__()
            self._lock = threading.Lock()
            self.large_leaf_prompts: list[str] = []
            self.failed = False

        def complete_json(self, prompt, system=""):
            if (
                "This is one bounded fragment of a larger" in prompt
                and "File: large.py" in prompt
            ):
                with self._lock:
                    self.large_leaf_prompts.append(prompt)
                    should_fail = (
                        len(self.large_leaf_prompts) == 2 and not self.failed
                    )
                    if should_fail:
                        self.failed = True
                        self.doc_calls += 1
                if should_fail:
                    raise provider_failure_error("openai", "provider-rate-limited", status=429, limit_type="tpm")
            return super().complete_json(prompt, system)

    provider = RateLimitSecondLargeLeaf()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda _seconds: None)

    callback_calls = []
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "analysis_mode": "single",
            "parallel_agents": False,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "max_parallel_files": 2,
            "rate_limit_adaptive": True,
            "file_retry_attempts": 0,
            "rate_limit_backoff_s": 0,
            "respect_retry_after": False,
            "propagate_changes": False,
            "output_dir": "docs",
        },
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    payload = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert {record["path"] for record in payload["files"]} == {
        "large.py",
        "small.py",
    }
    assert stats["checked"] == 2
    assert stats["failed"] == 0
    assert stats["split_ordinary_files"] == 1
    assert stats["split_divided_files"] == 1
    assert len(stats["rate_limit_warnings"]) == 1
    assert len(provider.large_leaf_prompts) == len(division.chunks) + 1
    assert provider.large_leaf_prompts.count(provider.large_leaf_prompts[0]) == 1
    assert provider.large_leaf_prompts.count(provider.large_leaf_prompts[1]) == 2
    assert stats["additional_attempts"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert callback_calls == []
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_terminal_split_failure_cancels_pending_file_tasks_and_keeps_checkpoint(
    tmp_path,
) -> None:
    source = _large_python_source(120)
    split_request = make_execution_request(
        tmp_path,
        "a_large.py",
        source,
        analysis_mode="single",
        max_content_chars=2000,
    )
    plan = build_division_plan(
        rel_path=split_request.rel_path,
        language=split_request.language,
        content=source,
        source_budget_chars=2000,
    )
    tree = build_reduction_tree(plan, synthesis_manifest_chars=12000)
    ordinary_requests = make_execution_requests(
        tmp_path,
        [f"ordinary_{index}.py" for index in range(5)],
    )

    class TerminalSplitOrchestrator:
        class _LLM:
            provider_name = "openai"

        llm = _LLM()

        def __init__(self) -> None:
            self.leaf_calls = 0
            self.ordinary_calls = 0
            self.ordinary_delay = threading.Event()

        def process_leaf_chunk(self, chunk_request):
            self.leaf_calls += 1
            if self.leaf_calls == 2:
                raise provider_failure_error(
                    "openai", "provider-quota-exhausted", status=429
                )
            return {"description": f"chunk {chunk_request.chunk_id}"}

        def process_reduction_node(self, _request):
            pytest.fail("terminal split task reached a reduction node")

        def synthesize_divided_file(self, _request, _digest, _manifest, terminology_source=""):
            pytest.fail("terminal split task reached synthesis")

        def process(self, request):
            self.ordinary_calls += 1
            self.ordinary_delay.wait(0.1)
            return {
                "file_path": request.rel_path,
                "language": request.language,
                "description": "ordinary",
            }

    class Queue:
        def __init__(self) -> None:
            self.checked: list[str] = []
            self.failed: list[tuple[str, str]] = []

        def mark_checked(self, rel_path):
            self.checked.append(rel_path)

        def mark_failed(self, rel_path, reason):
            self.failed.append((rel_path, reason))

    orchestrator = TerminalSplitOrchestrator()
    queue = Queue()
    stats = {"checked": 0, "failed": 0}
    writer = SafeWriter(
        tmp_path / "docs" / "crash_recovery.json",
        "json",
        None,
        {
            request.rel_path: {}
            for request in (split_request, *ordinary_requests)
        },
    )

    with pytest.raises(UnrecoverableProviderError):
        _process_descriptor_batch(
            [split_request, *ordinary_requests],
            orchestrator,
            queue,
            stats,
            ErrorReporter(),
            max_workers=1,
            recorder=writer,
            division_plans={split_request.rel_path: plan},
            reduction_trees={split_request.rel_path: tree},
            provider_identity="test-provider",
            split_execution_mode="recovery",
        )

    checkpoint = writer.get_tree_state(split_request.rel_path)
    assert checkpoint is not None
    completed = checkpoint.by_id()
    assert len(completed) == 1
    assert all(node.node_type == "leaf" for node in completed.values())
    assert orchestrator.leaf_calls == 2
    assert orchestrator.ordinary_calls <= 1
    assert writer.get_record(split_request.rel_path) is None
    assert queue.failed == []
    assert stats["failed"] == 0
    persisted = json.loads(writer.path.read_text(encoding="utf-8"))
    persisted_nodes = persisted["_codedoc"]["partial_files"][split_request.rel_path][
        "nodes"
    ]
    assert len(persisted_nodes) == 1
    assert persisted_nodes[0]["node_type"] == "leaf"


def test_fully_synthesized_split_recovery_finalizes_without_a_provider(
    tmp_path, monkeypatch
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_files": 1,
        "propagate_changes": False,
        "output_dir": "docs",
        "prompt_profiles": INLINE,
    }
    provider_identity = _provider_identity_for(tmp_path, config)
    resolved_config = load_config(tmp_path, config)
    profile_resolution = resolve_profile_source(
        resolved_config,
        tmp_path,
        known_extensions=frozenset(resolved_config["extension_language_map"]),
        active_mode="single",
    )
    profile_digest = ResolvedProfile(
        "single", profile_resolution.profile
    ).file_digest("main.py")
    recovered = _fully_completed_tree_state(
        division,
        tree,
        provider_identity=provider_identity,
        content_hash=content_hash,
        prompt_profile_digest=profile_digest,
        final_fields={
            "description": "restored complete file",
            "key_concepts": ["restored"],
        },
    )
    monkeypatch.setattr(
        "codedoc.pipeline.load_recovery_records_if_compatible",
        lambda *_args, **_kwargs: RecoveryState(
            records=(
                (
                    "main.py",
                    canonical_json({"path": "main.py", "hash": "stale"}),
                ),
            ),
            partial_files=(recovered,),
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("restored synthesis created a provider")
        ),
    )

    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})
    # Section 5.8: dry-run is a read-only preview of the SAME payable work as
    # the real run below -- a fully-synthesized recovered tree is reused, so
    # zero documentation-call candidates and zero planned review calls,
    # exactly matching the real-run assertions further down.
    assert dry_stats["max_files_candidate_files"] == 0
    assert dry_stats["prompt_customization_security_review_calls_planned"] == 0
    assert dry_stats["total_calls_planned"] == 0
    assert dry_stats["split_restored_complete_chunks"] == len(division.chunks)
    assert dry_stats["split_restored_final_synthesis_calls"] == 1

    callback_calls = []
    stats = run_pipeline(
        tmp_path,
        config,
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    output = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert stats["checked"] == 1
    assert stats["resumed"] == 1
    assert stats["total_calls_planned"] == 0
    assert stats["attempted_calls"] == 0
    assert stats["prompt_customization_security_review_calls_planned"] == 0
    assert stats["split_restored_complete_chunks"] == len(division.chunks)
    assert stats["split_restored_unit_consolidation_calls"] + stats[
        "split_restored_general_reduction_calls"
    ] == _reduction_total(tree)
    assert stats["split_restored_final_synthesis_calls"] == 1
    assert callback_calls == []
    assert output["files"][0]["description"] == "restored complete file"
    assert (
        output["last_run"]["split_restored_complete_chunks"]
        == len(division.chunks)
    )
    assert output["last_run"]["split_restored_final_synthesis_calls"] == 1
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_truncate_run_blocks_on_split_recovery_without_erasing_checkpoints(
    tmp_path, monkeypatch
) -> None:
    """A truncate run must fail closed on split recovery, not silently drop it.

    A truncate flush cannot read or rewrite ``partial_files``, so re-flushing one
    would erase already-paid node work.  ``large_file_strategy`` is therefore
    part of the recovery identity: the mismatch is refused before SafeWriter
    initialization and provider creation, the recovery file stays byte-for-byte
    intact, and a later split run still resumes the same checkpoint.
    """
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "truncate",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    provider_identity = _provider_identity_for(
        tmp_path, {**config, "large_file_strategy": "split"}
    )
    recovered = _one_leaf_completed_tree_state(
        division, tree, provider_identity=provider_identity, content_hash=content_hash
    )
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
    for node in recovered.nodes:
        writer.record_tree_node("main.py", node, reduction_tree_digest=tree.tree_digest)
    original_recovery = recovery_path.read_bytes()

    # A dry run stays non-mutating and provider-free: it neither counts nor
    # rewrites the split checkpoint. Under section 5.8 a resolved-valid
    # truncate route still publishes the shared route aggregates (all zeroed
    # here) and the empty split/truncate/blocked categories, but no
    # split-execution counters and no `large_file_strategy: "split"` marker.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("dry run created a provider"),
    )
    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})

    assert dry_stats["would_resume"] == 0
    assert dry_stats.get("large_file_strategy") != "split"
    assert dry_stats["large_file_strategy_resolved"] == "truncate"
    for absent in (
        "split_divided_files",
        "split_units",
        "split_chunks",
        "split_ordinary_files",
        "split_unpaid_nodes",
        "split_blocked_files",
        "split_blocked_by_reason",
        "split_synthesis_input_estimate",
    ):
        assert absent not in dry_stats
    assert dry_stats["split_plan_details"] == []
    assert dry_stats["split_plan_details_total"] == 0
    assert dry_stats["split_chunk_payload_chars_min"] == 0
    assert dry_stats["split_chunk_payload_chars_max"] == 0
    assert recovery_path.read_bytes() == original_recovery

    # The real truncate run blocks before SafeWriter or provider creation.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("incompatible recovery created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.load",
        lambda *_args, **_kwargs: pytest.fail(
            "incompatible recovery initialized the writer"
        ),
    )
    with pytest.raises(ConfigError) as blocked:
        run_pipeline(tmp_path, config)

    assert "large_file_strategy" in str(blocked.value)
    assert "'truncate'" in str(blocked.value)
    assert "'split'" in str(blocked.value)
    assert recovery_path.read_bytes() == original_recovery
    assert not (tmp_path / "docs" / "codedoc.json").exists()

    # The same checkpoint is still resumable by a split run.
    monkeypatch.undo()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda provider_config: attest_provider_execution(
            SmartFake(),
            provider_config,
        ),
    )
    stats = run_pipeline(tmp_path, {**config, "large_file_strategy": "split"})

    assert stats["resumed"] == 1
    assert stats["split_restored_complete_chunks"] == 1
    record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert "division" not in record
    assert "documentation_units" not in record
    assert record["_large_file_identity"].startswith("large-file-v3:")
    assert not recovery_path.exists()


def test_legacy_recovery_without_strategy_stays_compatible_with_truncate(
    tmp_path, monkeypatch
) -> None:
    """An identity written before this release has no strategy field.

    Absence normalizes to the default ``truncate``, so ordinary recovery keeps
    resuming and default identity bytes are unchanged.
    """
    (tmp_path / "main.py").write_bytes(b"VALUE = 1\n")
    legacy_identity = build_recovery_identity(
        project_root=tmp_path,
        json_target=tmp_path / "docs" / "codedoc.json",
        md_target=None,
        entry_file="main.py",
        documentation_scope="entry",
        analysis_mode="single",
        analysis_revision=ANALYSIS_REVISION,
    )

    assert "large_file_strategy" not in legacy_identity

    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    writer = SafeWriter(
        recovery_path, "json", "main.py", {}, legacy_identity
    )
    writer.record(
        "main.py",
        {"file_path": "main.py", "language": "python", "description": "Legacy."},
        hashlib.sha256(b"VALUE = 1\n").hexdigest(),
    )

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda provider_config: attest_provider_execution(
            SmartFake(),
            provider_config,
        ),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    assert stats["resumed"] == 1
    # Section 5.8: a strategy-absent (resolved truncate) route publishes the
    # shared route aggregates and empty categories, but never the split marker
    # or any split-execution counter.
    assert stats.get("large_file_strategy") != "split"
    assert stats["large_file_strategy_resolved"] == "truncate"
    assert stats["large_files_over_source_ceiling"] == 0
    assert stats["split_plan_details"] == []
    assert stats["split_blocked_details"] == []
    for absent in (
        "split_divided_files",
        "split_chunks",
        "split_ordinary_files",
        "split_blocked_files",
        "split_synthesis_input_estimate",
    ):
        assert absent not in stats


def test_split_dry_run_resumes_partial_recovery_matching_a_real_run(
    tmp_path, monkeypatch
) -> None:
    """Section 5.8 / 6.3: for the resolved-valid single+split route, a dry-run
    loads and classifies a GENUINE on-disk ``crash_recovery.json`` (written by
    the production ``SafeWriter`` checkpoint path) exactly as a real preflight
    does. Both invocations pass ``include_partial_files=True`` to the loader; the
    dry-run and real preflight snapshots are field-for-field identical
    (recursively, minus the ``dry_run`` marker); the dry-run leaves the recovery
    file byte-for-byte unchanged and creates / probes nothing; and the reporter
    precedes provider construction on the real run.
    """
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    provider_identity = _provider_identity_for(tmp_path, config)
    recovered = _one_leaf_completed_tree_state(
        division, tree, provider_identity=provider_identity, content_hash=content_hash
    )
    # A genuine on-disk checkpoint via the production writer path.
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
    for node in recovered.nodes:
        writer.record_tree_node("main.py", node, reduction_tree_digest=tree.tree_digest)
    original_recovery = recovery_path.read_bytes()

    with monkeypatch.context() as mp_outer:
        # Spy (do NOT replace) the loader: prove both modes ask for the partial
        # files while the GENUINE on-disk file is what actually gets loaded. The
        # spy persists across both capture contexts below.
        import codedoc.pipeline as _pl
        _real_loader = _pl.load_recovery_records_if_compatible
        loader_kwargs: list = []
        mp_outer.setattr(
            "codedoc.pipeline.load_recovery_records_if_compatible",
            lambda *a, **k: (loader_kwargs.append(k), _real_loader(*a, **k))[1],
        )

        with monkeypatch.context() as mpd:
            dry_snap, dry_events, dry_err = _s8_capture_preflight(
                tmp_path, config, mpd, dry=True, expect_error=False
            )
        assert dry_err is None
        # Byte-for-byte preservation + no output/probe/provider/writer on dry.
        assert recovery_path.read_bytes() == original_recovery
        assert not (tmp_path / "docs" / "codedoc.json").exists()
        assert dry_events == ["report"]

        with monkeypatch.context() as mpr:
            real_snap, real_events, real_err = _s8_capture_preflight(
                tmp_path, config, mpr, dry=False, expect_error=False
            )
        assert real_err is None
        assert real_events[0] == "report"
        if "provider" in real_events:
            assert real_events.index("report") < real_events.index("provider")

    # Full recursive snapshot parity, minus the intentional mode marker.
    assert dry_snap is not None and real_snap is not None
    assert dry_snap == real_snap

    # Scenario presence: the one completed leaf really was restored and the
    # compatible same-plan partial resumes with no re-execution or conflict.
    assert dry_snap["split_restored_complete_chunks"] == 1
    assert dry_snap["split_partial_files_resumed"] == 1
    assert dry_snap["split_reexecuted_nodes"] == 0
    assert dry_snap["split_recovery_conflict_files"] == 0
    assert dry_snap["split_recovery_discarded_predecessor_nodes"] == 0
    assert dry_snap["split_recovery_replacement_nodes_planned"] == 0
    assert dry_snap["unit_documentation_calls_planned"] == len(division.chunks) - 1
    assert dry_snap["synthesis_calls_planned"] == 1
    assert loader_kwargs == [
        {"include_partial_files": True},
        {"include_partial_files": True},
    ]


# ===========================================================================
# Section 5.8 correction round: the COMPLETE dry/real payable-work parity
# matrix. For every non-error scenario the dry-run and real-run provider-free
# preflight snapshots -- captured before any provider/review/writer/output
# side effect -- must be field-for-field identical (recursively) except for the
# intentional ``dry_run`` mode marker, with an identical canonical call-manifest
# digest and identical payable-node / restore / reuse / discard / replacement
# counts. For every recovery-error scenario dry and real raise the SAME
# ConfigError type and exact text at the same early boundary, dry-run leaves
# recovery bytes byte-for-byte unchanged, and neither constructs a provider,
# review, writer or output.
# ===========================================================================


def _s8_norm(value):
    from types import MappingProxyType

    if isinstance(value, (dict, MappingProxyType)):
        return {k: _s8_norm(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_s8_norm(v) for v in value]
    return value


class _S8Sentinel(Exception):
    pass


def _s8_capture_preflight(tmp_path, cfg, monkeypatch, *, dry, expect_error):
    """Run one pipeline; return (normalized_snapshot_or_None, events, error).

    dry=True  -> provider/writer/output-probe are hard sentinels.
    dry=False + expect_error -> same hard sentinels (the run must raise before
                                any of them).
    dry=False + not expect_error -> the run is allowed to complete with a
                                    working SmartFake; ``events`` still proves
                                    the report preceded provider construction.
    """
    events: list = []
    snaps: list = []

    def _prov(_c):
        events.append("provider")
        if dry or expect_error:
            raise _S8Sentinel("provider constructed during a provider-free preflight")
        return SmartFake()

    def _writer(*a, **k):
        events.append("writer")
        if dry or expect_error:
            raise _S8Sentinel("SafeWriter constructed during a provider-free preflight")
        return SafeWriter(*a, **k)

    import codedoc.pipeline as _pl
    real_probe = _pl.preflight_output_accessibility

    def _probe(*a, **k):
        events.append("probe")
        if dry:
            raise _S8Sentinel("output probed during dry-run")
        return real_probe(*a, **k)

    monkeypatch.setattr("codedoc.pipeline.create_provider", _prov)
    monkeypatch.setattr("codedoc.pipeline.SafeWriter", _writer)
    monkeypatch.setattr("codedoc.pipeline.preflight_output_accessibility", _probe)

    def _reporter(snap):
        events.append("report")
        snaps.append(_s8_norm(snap))

    err = None
    try:
        run_pipeline(tmp_path, {**cfg, "dry_run": dry}, plan_reporter=_reporter)
    except (ConfigError, _S8Sentinel) as exc:
        err = exc
    snap = snaps[0] if snaps else None
    if snap is not None:
        snap.pop("dry_run", None)
    return snap, events, err


def _s8_large_src(n=260):
    return "\n".join(f"value_{i} = {i}" for i in range(n)) + "\n"


_S8_PARITY_COUNT_KEYS = (
    "call_manifest_digest", "total_calls_planned", "documentation_calls_planned",
    "file_documentation_calls_planned", "unit_documentation_calls_planned",
    "file_reduction_calls_planned", "synthesis_calls_planned",
    "split_divided_files", "split_completed_files_reused",
    "split_partial_files_resumed", "split_restored_complete_chunks",
    "split_restored_unit_consolidation_calls", "split_restored_general_reduction_calls",
    "split_restored_final_synthesis_calls", "split_reexecuted_nodes",
    "split_recovery_discarded_predecessor_nodes",
    "split_recovery_replacement_nodes_planned",
    "split_recovery_conflict_files",
    "split_quarantined_nodes", "split_blocked_files", "would_reuse", "would_resume",
    "max_files_candidate_files",
)


def _s8_prep_fresh(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text(_s8_large_src(), encoding="utf-8", newline="")
    return {"entry_file": "main.py", "large_file_strategy": "split",
            "max_content_chars": 2000, "parallel_agents": False,
            "propagate_changes": False, "output_dir": "docs"}


def _s8_prep_completed(tmp_path, monkeypatch):
    cfg = _s8_prep_fresh(tmp_path, monkeypatch)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    first = run_pipeline(tmp_path, cfg)
    assert first["checked"] == 1 and first["failed"] == 0
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()
    monkeypatch.undo()
    return cfg




def _s8_prep_ordinary_reuse(tmp_path, monkeypatch):
    from codedoc.core.db import compute_file_hash

    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    write_existing_json(
        tmp_path / "docs" / "codedoc.json",
        compute_file_hash(tmp_path / "main.py"),
        "Cached.",
    )
    return {"entry_file": "main.py", "output_dir": "docs", "propagate_changes": False}


def _s8_prep_truncate_reuse(tmp_path, monkeypatch):
    # A genuine completed/current truncate state: a real initial truncate run
    # documents an oversized file; the second capture reuses it with zero
    # payable documentation calls.
    (tmp_path / "main.py").write_text(_s8_large_src(400), encoding="utf-8", newline="")
    cfg = {"entry_file": "main.py", "output_dir": "docs",
           "large_file_strategy": "truncate", "max_content_chars": 2000,
           "parallel_agents": False, "propagate_changes": False}
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    first = run_pipeline(tmp_path, cfg)
    assert first["checked"] == 1 and first["failed"] == 0
    written = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    rec = next(f for f in written["files"] if f["path"] == "main.py")
    # route identity: the completed record is a truncate record, not split.
    assert "_large_file_identity" not in rec
    monkeypatch.undo()
    return cfg


def _s8_prep_ordinary_in_split_run(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("import helper\nVALUE = 1\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text(_s8_large_src(), encoding="utf-8", newline="")
    return {"entry_file": "main.py", "documentation_scope": "all",
            "large_file_strategy": "split", "max_content_chars": 2000,
            "parallel_agents": False, "propagate_changes": False, "output_dir": "docs"}


def _s8_prep_zero_work(tmp_path, monkeypatch):
    return {"entry_file": None, "auto_entry_candidates": [], "propagate_changes": False,
            "output_dir": "docs"}


def _s8_prep_split_blocked(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "zeta.py").write_text(
        "\n".join(f"def fn_{i}(): return {i}" for i in range(400)) + "\n",
        encoding="utf-8", newline="",
    )
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 1)
    return {"entry_file": "main.py", "documentation_scope": "all",
            "large_file_strategy": "split", "max_content_chars": 2000,
            "parallel_agents": False, "propagate_changes": False, "output_dir": "docs"}


_S8_PARITY_SCENARIOS = {
    "fresh_split": (_s8_prep_fresh, False),
    "completed_current_split_record": (_s8_prep_completed, False),
    # NOTE: compatible-partial, same-plan-stale, and cross-plan expansion/
    # contraction parity is proven against a GENUINE on-disk crash_recovery
    # .json in tests/integration/persistence/test_split_division_recovery.py
    # (test_cr2_recovery_transition_dry_real_parity_on_disk) so byte
    # preservation and the exact transition counters are real, not
    # monkeypatched.
    "ordinary_reuse": (_s8_prep_ordinary_reuse, False),
    "truncate_reuse": (_s8_prep_truncate_reuse, False),
    "ordinary_file_inside_split_run": (_s8_prep_ordinary_in_split_run, False),
    "zero_admitted_work": (_s8_prep_zero_work, False),
    "split_blocked_work": (_s8_prep_split_blocked, True),
}


@pytest.mark.parametrize("scenario", sorted(_S8_PARITY_SCENARIOS))
def test_s8_dry_real_payable_parity_matrix(tmp_path, monkeypatch, scenario):
    prep, is_error = _S8_PARITY_SCENARIOS[scenario]

    # dry-run is read-only, so both preflights are captured from the SAME
    # repository state in the SAME directory -- the only legitimate difference
    # is the ``dry_run`` marker (popped inside _s8_capture_preflight).
    with monkeypatch.context() as mpp:
        cfg = prep(tmp_path, mpp)
        with monkeypatch.context() as mpd:
            dry_snap, dry_events, dry_err = _s8_capture_preflight(
                tmp_path, cfg, mpd, dry=True, expect_error=is_error
            )
        with monkeypatch.context() as mpr:
            real_snap, real_events, real_err = _s8_capture_preflight(
                tmp_path, cfg, mpr, dry=False, expect_error=is_error
            )

    # the reporter fired exactly once, first, on both routes
    assert dry_events.count("report") == 1
    assert real_events.count("report") == 1
    assert dry_events[0] == "report" and real_events[0] == "report"
    # dry-run performed no provider / writer / output probe at all
    assert dry_events == ["report"]
    # dry-run always reports and returns -- it never raises in this matrix.
    assert dry_err is None, dry_err
    if is_error:
        # the real run raises the deterministic division-blocked ConfigError
        # AFTER the shared report -- never an _S8Sentinel (which would mean a
        # provider / writer / output probe was constructed first).
        assert isinstance(real_err, ConfigError), real_err
        assert not isinstance(real_err, _S8Sentinel)
    else:
        assert real_err is None, real_err
    if not is_error:
        # real preflight was captured strictly before any provider construction
        if "provider" in real_events:
            assert real_events.index("report") < real_events.index("provider")
        if "writer" in real_events:
            assert real_events.index("report") < real_events.index("writer")
        if "probe" in real_events:
            assert real_events.index("report") < real_events.index("probe")

    # FIELD-FOR-FIELD (recursive) snapshot parity, minus the mode marker.
    assert dry_snap is not None and real_snap is not None
    assert dry_snap == real_snap
    # explicit call-manifest digest + payable/restore/reuse/discard counts
    for k in _S8_PARITY_COUNT_KEYS:
        if k in dry_snap:
            assert dry_snap[k] == real_snap[k], k

    # scenario-presence: prove the run ACTUALLY entered the named scenario,
    # not merely that dry == real on some other (e.g. fresh) shape.
    _sp = dry_snap
    if scenario == "fresh_split":
        assert _sp["split_divided_files"] == 1 and _sp["total_calls_planned"] > 0
        assert _sp["split_completed_files_reused"] == 0
    elif scenario == "completed_current_split_record":
        assert _sp["split_completed_files_reused"] == 1
        assert _sp["total_calls_planned"] == 0 and _sp["split_divided_files"] == 0
    elif scenario == "ordinary_reuse":
        assert _sp["total_calls_planned"] == 0
        assert (_sp["would_reuse"] + _sp["unchanged"]) >= 1
        assert _sp["large_files_routed_split"] == 0
    elif scenario == "truncate_reuse":
        assert _sp["total_calls_planned"] == 0
        assert (_sp["would_reuse"] + _sp["unchanged"]) >= 1
        assert _sp["large_file_strategy_resolved"] == "truncate"
    elif scenario == "ordinary_file_inside_split_run":
        assert _sp["split_divided_files"] == 1
        assert _sp["split_ordinary_files"] >= 1
    elif scenario == "zero_admitted_work":
        assert _sp["total_calls_planned"] == 0
        assert _sp["scanner_size_skip_details_digest"].startswith("sha256:")
        assert _sp["call_manifest_digest"]
    elif scenario == "split_blocked_work":
        assert _sp["split_blocked_details_total"] >= 1
        assert _sp["split_blocked_by_reason"].get("chunk-cap", 0) >= 1

    # (real-run error type + dry/real error assertions are made above, before
    # the snapshot-parity comparison.)


def test_s8_dry_real_recovery_error_parity_malformed_container(tmp_path, monkeypatch):
    """A malformed current-schema recovery container raises the identical
    ConfigError in dry-run and real preflight, at the same early boundary; the
    dry run leaves the recovery file byte-for-byte unchanged and constructs no
    provider / review / writer / output."""
    source = _s8_large_src(220)
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    output_dir = tmp_path / "docs"
    output_dir.mkdir()
    recovery_path = output_dir / "crash_recovery.json"
    identity = build_recovery_identity(
        project_root=tmp_path,
        json_target=output_dir / "codedoc.json",
        md_target=None,
        entry_file="main.py",
        documentation_scope="entry",
        analysis_mode="single",
        analysis_revision=ANALYSIS_REVISION,
        large_file_strategy="split",
    )
    writer = SafeWriter(recovery_path, "json", "main.py", {}, identity)
    writer.initialize_empty()
    payload = json.loads(recovery_path.read_text(encoding="utf-8"))
    payload["_codedoc"]["partial_files"] = {
        "main.py": {"schema_version": SPLIT_PARTIAL_SCHEMA_VERSION, "nodes": "not-a-list"}
    }
    recovery_path.write_text(json.dumps(payload), encoding="utf-8")
    original = recovery_path.read_bytes()

    cfg = {"entry_file": "main.py", "large_file_strategy": "split",
           "max_content_chars": 2000, "analysis_mode": "single",
           "propagate_changes": False, "output_dir": "docs"}

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("malformed-recovery run constructed a provider"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter",
        lambda *a, **k: pytest.fail("malformed-recovery run constructed a writer"),
    )

    dry_reports: list = []
    with pytest.raises(ConfigError) as dry_exc:
        run_pipeline(tmp_path, {**cfg, "dry_run": True},
                     plan_reporter=lambda s: dry_reports.append(s))
    assert recovery_path.read_bytes() == original
    assert dry_reports == []  # blocked at recovery load, before the snapshot

    with pytest.raises(ConfigError) as real_exc:
        run_pipeline(tmp_path, cfg, plan_reporter=lambda s: dry_reports.append(s))

    assert type(dry_exc.value) is type(real_exc.value)
    assert str(dry_exc.value) == str(real_exc.value)
    assert recovery_path.read_bytes() == original


# Cross-plan expansion / contraction dry/real parity, the exact §6.3 transition
# counters, and dry-run recovery-byte preservation are proven against a GENUINE
# on-disk crash_recovery.json in
# tests/integration/persistence/test_split_division_recovery.py::
# test_cr2_recovery_transition_dry_real_parity_on_disk -- not with a
# monkeypatched in-memory RecoveryState, which cannot prove byte preservation.


def test_disconnected_split_call_count_uses_exact_chunk_manifest(
    tmp_path,
) -> None:
    (tmp_path / "main.py").write_bytes(b"ENTRY = True\n")
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "orphan.py").write_bytes(source.encode("utf-8"))

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "documentation_scope": "all",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
        },
    )

    assert stats["disconnected_paid_files"] == 1
    assert stats["disconnected_planned_calls"] == (
        stats["unit_documentation_calls_planned"]
        + stats["file_reduction_calls_planned"]
        + stats["synthesis_calls_planned"]
    )
    assert stats["disconnected_planned_calls"] > stats["initial_calls_per_file"]


def test_capacity_blocked_split_dry_run_reports_bounded_details_and_excludes_max_files(
    tmp_path, monkeypatch, capsys
) -> None:
    """D8 / section 5.8: dry-run exposes every blocked file through the bounded,
    measured ``split_blocked`` category (``split_blocked_pairs`` is retired),
    remains provider- and writer-free, and excludes blocked files from the
    paid-file safety cap. The bounded CLI rendering of the category is a later
    section; here the frozen dry-run exit code and no-mutation contract hold."""
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8", newline="")
    oversized = _large_python_source(120)
    (tmp_path / "zeta.py").write_text(oversized, encoding="utf-8", newline="")
    (tmp_path / "alpha.py").write_text(oversized, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_files": 1,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }

    # Exercise the real division capacity check rather than manufacturing a
    # planning result. Each oversized fixture needs more than one chunk.
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 1)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("capacity-blocked dry run created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "capacity-blocked dry run constructed a writer"
        ),
    )

    stats = run_pipeline(tmp_path, {**config, "dry_run": True})

    assert "split_blocked_pairs" not in stats
    assert stats["split_blocked_files"] == 2
    assert stats["split_blocked_by_reason"] == {"chunk-cap": 2}
    assert stats["split_blocked_details_total"] == 2
    assert stats["split_blocked_details_retained"] == 2
    assert stats["split_blocked_details_omitted"] == 0
    assert stats["split_blocked_details_digest"].startswith("sha256:")
    assert [d["path"] for d in stats["split_blocked_details"]] == ["alpha.py", "zeta.py"]
    for detail in stats["split_blocked_details"]:
        assert detail["reason"] == "chunk-cap"
        assert detail["phase"] == "division-packing"
        assert detail["guidance_code"] == "raise-source-ceiling-or-split-source"
        assert detail["observed"] > detail["limit"] == 1
    assert stats["max_files_candidate_files"] == 1
    assert stats["would_call_llm_for"] == 1
    assert stats["max_files_exceeded"] is False
    assert not (tmp_path / "docs").exists()

    # The public CLI retains the frozen dry-run exit code 0 and no-mutation
    # contract; bounded rendering of the split_blocked category is a later
    # section.
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(
            {
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "max_files": 1,
                "parallel_agents": False,
                "propagate_changes": False,
                "output_dir": "docs",
            }
        ),
        encoding="utf-8",
        newline="",
    )
    from codedoc.cli.cli import run_cli

    capsys.readouterr()
    exit_code = run_cli(
        [
            str(tmp_path),
            "--entry",
            "main.py",
            "--documentation-scope",
            "all",
            "--dry-run",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert not (tmp_path / "docs").exists()

    # The corresponding real run aborts at the already-patched writer/provider
    # boundary with the bounded, JSON-escaped, reason-ordered ConfigError.
    with pytest.raises(ConfigError) as blocked:
        run_pipeline(tmp_path, config)
    message = str(blocked.value)
    assert "Split never falls back to truncation" in message
    assert "2 file(s) cannot be completely split-planned" in message
    assert message.index('"alpha.py": chunk-cap') < message.index('"zeta.py": chunk-cap')
    assert "details_digest sha256:" in message
    assert not (tmp_path / "docs").exists()


@pytest.mark.parametrize("reason", BLOCKED_REASON_ORDER)
def test_every_capacity_block_reason_aborts_before_provider_or_writer(
    tmp_path, monkeypatch, reason
) -> None:
    """Every frozen capacity reason follows the same no-truncate real-run
    boundary; none may be demoted to an ordinary provider action."""
    (tmp_path / "main.py").write_text(
        _large_python_source(120), encoding="utf-8", newline=""
    )
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            SplitCapacityBlocked(kwargs["rel_path"], reason, observed=999, limit=256)
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail(f"{reason} block created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_args, **_kwargs: pytest.fail(f"{reason} block constructed a writer"),
    )

    with pytest.raises(ConfigError, match=reason) as blocked:
        run_pipeline(tmp_path, config)

    assert "Split never falls back to truncation" in str(blocked.value)
    assert not (tmp_path / "docs").exists()


def test_new_capacity_block_cannot_be_bypassed_by_reuse_and_cli_exits_two(
    tmp_path, monkeypatch, capsys
) -> None:
    """D8: feasibility is recalculated before reuse. Even a current completed
    split record cannot hide a newly blocked plan; the real run and CLI both
    preserve prior stable output and fail before writer/provider creation."""
    source = _large_python_source(120)
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )
    first = run_pipeline(tmp_path, config)
    assert first["checked"] == 1

    stable_path = tmp_path / "docs" / "codedoc.json"
    stable_bytes = stable_path.read_bytes()
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    assert not recovery_path.exists()

    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **kwargs: (_ for _ in ()).throw(
            SplitCapacityBlocked(
                kwargs["rel_path"], "chunk-cap", observed=257, limit=256
            )
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("new capacity block created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_args, **_kwargs: pytest.fail(
            "new capacity block constructed a writer"
        ),
    )

    with pytest.raises(ConfigError, match=r'"main\.py": chunk-cap \(phase division-packing'):
        run_pipeline(tmp_path, config)

    assert stable_path.read_bytes() == stable_bytes
    assert not recovery_path.exists()

    # Exercise the public real-run exit contract against the same still-valid
    # completed record. The config file is not a supported source extension.
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(config), encoding="utf-8", newline=""
    )
    from codedoc.cli.cli import run_cli

    capsys.readouterr()
    assert run_cli([str(tmp_path)]) == 2
    stderr = capsys.readouterr().err
    assert '"main.py": chunk-cap (phase division-packing' in stderr
    assert "Split never falls back to truncation" in stderr
    assert stable_path.read_bytes() == stable_bytes
    assert not recovery_path.exists()


def test_division_internal_defect_aborts_the_whole_run_uncaught(
    tmp_path, monkeypatch
) -> None:
    """D8: a genuine DivisionInternalDefect is a programming-invariant failure,
    not a per-file outcome. It propagates uncaught out of planning and aborts
    the whole run — dry or real — before any provider or writer side effect,
    even when another ordinary file in the same run would otherwise succeed."""
    source = _large_python_source()
    (tmp_path / "big.py").write_bytes(source.encode("utf-8"))
    (tmp_path / "main.py").write_text(
        "import big\n\ndef tiny():\n    return 1\n", encoding="utf-8", newline=""
    )
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **_kwargs: (_ for _ in ()).throw(
            DivisionInternalDefect("forced division invariant")
        ),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("division defect created a provider"),
    )

    with pytest.raises(DivisionInternalDefect, match="forced division invariant"):
        run_pipeline(tmp_path, {**config, "dry_run": True})

    assert not (tmp_path / "docs").exists()

    with pytest.raises(DivisionInternalDefect, match="forced division invariant"):
        run_pipeline(tmp_path, config)

    assert not (tmp_path / "docs" / "codedoc.json").exists()
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_division_internal_defect_leaves_prior_state_untouched_and_unblocked(
    tmp_path, monkeypatch
) -> None:
    """A defect discovered on a later run must not corrupt or block runs
    around it: prior stable output from an earlier successful run survives
    byte-for-byte, and a subsequent differently-configured run is not blocked
    by any residue (because the aborted run never wrote anything)."""
    source = _large_python_source()
    (tmp_path / "big.py").write_bytes(source.encode("utf-8"))
    (tmp_path / "main.py").write_text(
        "import big\n\ndef tiny():\n    return 1\n", encoding="utf-8", newline=""
    )
    config = {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )
    first = run_pipeline(tmp_path, config)
    assert first["checked"] == 2
    assert first["failed"] == 0
    stable_bytes = (tmp_path / "docs" / "codedoc.json").read_bytes()
    recovery_path = tmp_path / "docs" / "crash_recovery.json"
    assert not recovery_path.exists()

    monkeypatch.setattr(
        "codedoc.core.planning.build_division_plan",
        lambda **_kwargs: (_ for _ in ()).throw(
            DivisionInternalDefect("forced division invariant")
        ),
    )
    with pytest.raises(DivisionInternalDefect):
        run_pipeline(tmp_path, {**config, "force_files": ["big.py"]})

    assert (tmp_path / "docs" / "codedoc.json").read_bytes() == stable_bytes
    assert not recovery_path.exists()

    monkeypatch.undo()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda provider_config: attest_provider_execution(
            SmartFake(),
            provider_config,
        ),
    )
    later = run_pipeline(tmp_path, {**config, "large_file_strategy": "truncate"})

    assert later["failed"] == 0
    assert not recovery_path.exists()


def test_rejected_recovery_partial_does_not_force_retention(
    tmp_path, monkeypatch
) -> None:
    """Parsed partials are not automatically retainable.

    ``SafeWriter.load()`` can import every structurally readable partial, while
    planning accepts only the subset passing ``validate_node_for_tree()``. The
    distinguishing case is a stale partial for a path that is *not* recorded
    this run — an unchanged/reused file never calls ``record()``, so nothing
    ever pops the checkpoint. Seeding the writer from every parsed partial would
    therefore keep ``crash_recovery.json`` alive permanently; seeding from the
    pipeline-authorized retention set removes it.
    """
    source = _large_python_source()
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    config = {
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )
    first = run_pipeline(tmp_path, config)
    assert first["checked"] == 1

    # Rebuild an in-progress envelope: the completed record plus a stale,
    # execution-incompatible partial for that same now-unchanged path.
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    stale_hash = "9" * 64
    provider_identity = _provider_identity_for(tmp_path, config)
    stale = _one_leaf_completed_tree_state(
        division, tree, provider_identity=provider_identity, content_hash=stale_hash
    )
    stable_record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
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
    writer.load(preloaded={"main.py": stable_record})
    for node in stale.nodes:
        writer.record_tree_node("main.py", node, reduction_tree_digest=tree.tree_digest)
    assert recovery_path.exists()

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("unchanged file must not call a provider"),
    )

    second = run_pipeline(tmp_path, config)

    # The file is unchanged, so record() never runs and never pops the stale
    # checkpoint. It must still not keep the recovery file alive.
    assert second["checked"] == 0
    assert second["skipped"] == 1
    assert second["split_restored_complete_chunks"] == 0
    assert not recovery_path.exists()


# ---------------------------------------------------------------------------
# 0.14.6: the lexical data-module export-contract regression
# ---------------------------------------------------------------------------
# Structural reproduction of the released 0.14.5 failure, provider-free. The
# external project's own source is deliberately not copied here.

_EXPORT_CONTRACT_MARKER = "Module-export contract for the optional"

_DATA_MODULE_ENTRIES = 160
_DATA_MODULE_BUDGET = 2000

#: Real module-level exports of `_data_module_source()`, in source order.
_REAL_DATA_MODULE_EXPORTS = [
    "OPTION_ROWS",
    "DEFAULT_ROW_ID",
    "ROW_COUNT",
    "ROW_GROUPS",
]


def _data_module_source(
    entries: int = _DATA_MODULE_ENTRIES, groups: int = 24
) -> str:
    """A valid, oversized, data-only TypeScript module.

    Formatted one entry per line, exactly like the module that failed. Under
    the lexical fallback an atom is one physical line, so `pack_chunks()`
    packs this into several *independent* chunks -- not a marked continuation
    group -- and every chunk between the first and the last shows only
    exported array interior, with no `export` declaration in sight.
    """
    lines = ["export const OPTION_ROWS = ["]
    for index in range(entries):
        lines.append(f'  {{ id: "row_{index:03d}", label: "Row {index:03d}" }},')
    lines.append("];")
    lines.append("")
    lines.append('export const DEFAULT_ROW_ID = "row_000";')
    lines.append(f"export const ROW_COUNT = {entries};")
    lines.append("")
    lines.append("export const ROW_GROUPS = [")
    for index in range(groups):
        lines.append(f'  {{ key: "group_{index:02d}" }},')
    lines.append("];")
    return "\n".join(lines) + "\n"


class _DataModuleFake(SmartFake):
    """Leaf-aware double for the data-module regression.

    A leaf whose visible source carries real `export` declarations answers
    with exactly those names. The first leaf showing only array interior
    answers with an over-cap export list built from the visible data IDs --
    the observed 0.14.5 misreading -- which the fixed cleaner must reject as
    `fixed_cap_exceeded`; the one correction call then returns a truthful,
    export-free capsule.

    Prompts are recorded, never asserted on in-line: an assertion raised
    inside a provider double would surface as an opaque provider fault and be
    absorbed by the file-retry path instead of failing the test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.leaf_prompts: list[str] = []
        self.correction_prompts: list[str] = []
        self.prompts_without_contract: list[str] = []
        self.over_cap_id_count = 0

    def _record(self, prompt: str, bucket: list[str]) -> None:
        bucket.append(prompt)
        if _EXPORT_CONTRACT_MARKER not in prompt:
            self.prompts_without_contract.append(prompt)

    def complete_json(self, prompt, system=""):
        if "Previous response (verbatim" in prompt:
            self._record(prompt, self.correction_prompts)
            self.doc_calls += 1
            return json.dumps(
                {"description": "Rows of an exported data table."}
            )
        if "This is one bounded fragment of a larger" in prompt:
            self._record(prompt, self.leaf_prompts)
            self.doc_calls += 1
            declared = re.findall(r"^export const (\w+)", prompt, flags=re.M)
            if declared:
                return json.dumps(
                    {
                        "description": "Exported data declarations.",
                        "exports": declared,
                    }
                )
            if not self.over_cap_id_count:
                # The released misreading: array members promoted to exports.
                ids = re.findall(r'id: "(row_\d+)"', prompt)
                self.over_cap_id_count = len(ids)
                return json.dumps(
                    {
                        "description": "Rows of the exported table.",
                        "exports": ids[: MAX_LEAF_EXPORT_ITEMS + 1],
                    }
                )
            return json.dumps({"description": "Rows of the exported table."})
        return super().complete_json(prompt, system)


def test_lexical_data_module_never_publishes_array_interior_as_exports(
    tmp_path, monkeypatch
) -> None:
    """0.14.6 end-to-end: the structural shape that failed on 0.14.5.

    A valid oversized data-only TypeScript module, divided through the
    supported base-install lexical fallback, completes with one rejected leaf
    response and one successful correction, and publishes only the module's
    real language-level exports -- no `id`, label, group key, or other array
    member is ever promoted to a module export.

    The optional `structure` extra is simulated away rather than skipped, so
    this proves the supported base-install path in every environment.
    """
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)

    source = _data_module_source()
    (tmp_path / "data.tsx").write_text(source, encoding="utf-8", newline="")
    division = build_division_plan(
        rel_path="data.tsx",
        language="tsx",
        content=source,
        source_budget_chars=_DATA_MODULE_BUDGET,
    )
    # `run_pipeline` below plans through the automatic synthesis floor; this
    # reconstruction of the expected topology must use the same effective value
    # (no request object is available here), not the raw source budget.
    tree = build_reduction_tree(
        division,
        synthesis_manifest_chars=max(
            _DATA_MODULE_BUDGET, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
        ),
        language="tsx",
    )

    # Fixture integrity: lexical fallback, several chunks, at least two of them
    # showing nothing but exported-array interior, and NOT a continuation
    # group. If a future chunker change turned these into continuation
    # fragments this test would silently stop covering the reported failure,
    # so the shape is asserted, never assumed.
    assert division.structural_mode == "lexical"
    assert len(division.chunks) >= 4
    interior_only = [
        chunk for chunk in division.chunks if "export const " not in chunk.payload
    ]
    assert len(interior_only) >= 2
    assert all(
        chunk.continuation_before is False
        and chunk.continuation_after is False
        and chunk.unit_chunk_count == 1
        for chunk in division.chunks
    )
    # An interior-only fragment must show more data IDs than the export cap
    # allows, or the over-cap misreading it provokes could not be reproduced.
    assert all(
        len(re.findall(r'id: "row_\d+"', chunk.payload)) > MAX_LEAF_EXPORT_ITEMS
        for chunk in interior_only
    )
    assert len(_REAL_DATA_MODULE_EXPORTS) < MAX_LEAF_EXPORT_ITEMS

    provider = _DataModuleFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    callback_calls: list = []
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "data.tsx",
            "analysis_mode": "single",
            "parallel_agents": False,
            "large_file_strategy": "split",
            "max_content_chars": _DATA_MODULE_BUDGET,
            "max_parallel_files": 1,
            "file_retry_attempts": 0,
            "response_correction_enabled": True,
            "propagate_changes": False,
            "output_dir": "docs",
        },
        confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
    )

    # Both routes carried the contract; the over-cap response really was built
    # from visible array interior, not from a name the source never declared.
    assert provider.prompts_without_contract == []
    assert len(provider.leaf_prompts) == len(division.chunks)
    assert len(provider.correction_prompts) == 1
    assert provider.over_cap_id_count > MAX_LEAF_EXPORT_ITEMS

    planned = len(division.chunks) + _reduction_total(tree) + 1
    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert stats["total_calls_planned"] == planned
    assert stats["attempted_logical_calls"] == planned
    assert stats["attempted_calls"] == planned + 1
    assert stats["successful_calls"] == planned + 1
    assert stats["failed_calls"] == 0
    assert stats["planned_calls_not_attempted"] == 0
    assert stats["response_contract_failures"] == 1
    assert stats["response_correction_calls_attempted"] == 1
    assert stats["response_correction_calls_succeeded"] == 1
    assert stats["response_correction_calls_failed"] == 0
    assert stats["additional_attempts"] == 1
    assert callback_calls == []

    document = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    # A clean run publishes no "errors" key at all.
    assert "errors" not in document
    assert len(document["files"]) == 1
    record = document["files"][0]
    assert record["exports"] == _REAL_DATA_MODULE_EXPORTS
    # A data-only module declares no functions or classes, so the published
    # record omits both -- the ledger never invented one from array interior.
    assert record.get("functions", []) == []
    assert record.get("classes", []) == []
    # Nothing from an exported value's interior may reach the published record
    # as an export, whatever a model returned for an interior-only fragment.
    serialized_exports = json.dumps(record["exports"])
    for interior_token in ("row_", "group_", "Row 0", "Group 0"):
        assert interior_token not in serialized_exports

    # A clean completion removes recovery; nothing is left behind to resume.
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


def test_leaf_capsule_v7_tree_cannot_leave_a_reducer_or_final_node_alive(
    tmp_path, monkeypatch
) -> None:
    """0.14.6: no v7-derived reducer or final result survives its stale leaves.

    Reduction and final execution identities deliberately do not bind
    `LEAF_CAPSULE_SCHEMA_REVISION`, so each of those nodes remains
    individually well-formed across the advance. Dependency closure is what
    removes them: `validate_recovered_tree` never validates a reducer until
    every ordered child is already retained, gates the final node on every
    leaf being retained, and then quarantines each checkpointed node left
    outside the closure. Without that sweep a v8 run could publish a final
    synthesis built from exports a v7 leaf mis-derived.

    Proven in two directions from one set of bytes: the complete tree is
    genuinely reusable while the constant reads `leaf-capsule-v7`, and
    entirely non-reusable under the real current `leaf-capsule-v8`.
    """
    source = _large_python_source()
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(plan, max_content_chars=2000, language="python")
    provider_identity = _provider_identity_for(
        tmp_path,
        {
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
        },
    )
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert _reduction_total(tree) >= 1

    # Deliberately not pinning the current revision's value here: this proof
    # must fail on the retained/quarantined outcome if the advance is ever
    # reverted, not on a constant-equality guard that owns nothing.
    monkeypatch.setattr(
        file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v7"
    )
    predecessor = _fully_completed_tree_state(
        plan,
        tree,
        provider_identity=provider_identity,
        content_hash=content_hash,
    )
    node_count = len(plan.chunks) + _reduction_total(tree) + 1
    assert len(predecessor.nodes) == node_count

    validate_kwargs = dict(
        plan=plan,
        tree=tree,
        content_hash=content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=NO_PROMPT_PROFILE_DIGEST,
        imports_digest=deterministic_imports_digest(()),
        language="python",
        # The final node's live-schema re-check needs the resolved final
        # shape, or it would be rejected in *both* directions and this test
        # could not show that closure -- not its own identity -- removed it.
        resolved_shape=ResolvedProfile("single", None).resolve_block(
            "combined", "main.py"
        ),
    )

    retained_v7, quarantine_v7 = validate_recovered_tree(
        predecessor.nodes, **validate_kwargs
    )
    assert quarantine_v7 == ()
    assert len(retained_v7) == node_count

    monkeypatch.undo()

    retained_v8, quarantine_v8 = validate_recovered_tree(
        predecessor.nodes, **validate_kwargs
    )
    assert retained_v8 == ()
    assert len(quarantine_v8) == node_count

    by_id = {entry.node_id: entry.reason for entry in quarantine_v8}
    leaf_ids = {chunk.chunk_id for chunk in plan.chunks}
    for node_id, reason in by_id.items():
        if node_id in leaf_ids:
            # The leaf itself binds the revision: its identity no longer matches.
            assert reason == "stale-identity", node_id
        else:
            # A reducer/final node is pruned by dependency closure, not by its
            # own identity -- which is unchanged across this advance.
            assert reason == "input-digest-mismatch", node_id
    assert len(quarantine_v8) <= file_division.MAX_QUARANTINE_ENTRIES_PER_FILE


def test_leaf_capsule_v7_partial_is_actually_re_executed_by_a_real_run(
    tmp_path, monkeypatch
) -> None:
    """0.14.6: the v7 partial migration, driven end to end.

    Validating a recovered container in isolation proves the *decision*; it
    does not prove the run then pays for the work, publishes the file, and
    clears recovery. Here a complete v7 tree -- every leaf, reducer, and the
    final node, all written by the production identity functions with the
    constant patched back -- is handed to a real `run_pipeline` under the
    current revision. Every node must be quarantined, re-executed against the
    provider, and the run must finish cleanly.

    Contrast with `test_fully_synthesized_split_recovery_finalizes_without_a
    _provider`, whose identical tree is *current* and therefore costs nothing.
    """
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_bytes(source.encode("utf-8"))
    division = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(division, synthesis_manifest_chars=12000)
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    config = {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_parallel_files": 1,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    provider_identity = _provider_identity_for(tmp_path, config)

    with monkeypatch.context() as predecessor_revision:
        predecessor_revision.setattr(
            file_division, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v7"
        )
        recovered = _fully_completed_tree_state(
            division,
            tree,
            provider_identity=provider_identity,
            content_hash=content_hash,
        )

    node_count = len(division.chunks) + _reduction_total(tree) + 1
    assert len(recovered.nodes) == node_count

    monkeypatch.setattr(
        "codedoc.pipeline.load_recovery_records_if_compatible",
        lambda *_args, **_kwargs: RecoveryState(
            records=(
                (
                    "main.py",
                    canonical_json({"path": "main.py", "hash": "stale"}),
                ),
            ),
            partial_files=(recovered,),
        ),
    )
    provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    stats = run_pipeline(tmp_path, config)

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    # Nothing survived the advance: no node was restored, every one was
    # quarantined, and the whole tree was paid for again.
    assert stats["split_restored_complete_chunks"] == 0
    assert stats["split_restored_unit_consolidation_calls"] == 0
    assert stats["split_restored_general_reduction_calls"] == 0
    assert stats["split_restored_final_synthesis_calls"] == 0
    assert stats["split_quarantined_nodes"] == node_count
    assert stats["total_calls_planned"] == node_count
    assert stats["attempted_calls"] == node_count
    assert stats["successful_calls"] == node_count
    assert provider.doc_calls == node_count

    output = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert len(output["files"]) == 1
    # The freshly executed description, not the restored v7 one.
    assert output["files"][0]["description"] == "A file."
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()


# ---------------------------------------------------------------------------
# 0.14.7 section 9.3 / section 7.2 fixture freeze: the single tracked,
# provider-free live-validation source at
# `tests/fixtures/live_validation/oversized_signature.py`.
#
# Section 9.3 mandates this fixture and requires that "its exact SHA-256,
# decoded length, structural mode, chunk sizes/ranges, reducer count, and
# initial documentation-call count are frozen by tests and recorded before
# live use." These regressions are that freeze. They derive every value from
# the real production canonical loader / division planner / reduction-tree
# builder / pipeline planning path over the actual tracked file -- never from
# a source string re-inlined here -- so a byte change to the fixture, or a
# planning-topology drift, fails loudly and forces a deliberate re-freeze
# before the paid attempt.
# ---------------------------------------------------------------------------

_LIVE_VALIDATION_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "live_validation"
    / "oversized_signature.py"
)

#: Frozen at fixture-authoring time from the required interpreter with the
#: pinned `tree-sitter-language-pack`. Section 9.3's "recorded before live
#: use" values; a mismatch here is a re-freeze gate, not a flake.
_FIXTURE_SHA256 = "cfbf8716bcab26e996d6d467997eeb57fe53b172d21a9831d8fd42f623b24da1"
_FIXTURE_RAW_BYTES = 2317
_FIXTURE_CANONICAL_CHARS = 2317
_FIXTURE_STRUCTURAL_MODE = "syntax"
_FIXTURE_ATOM_COUNT = 8
_FIXTURE_SYMBOL_CENSUS = (
    ("class_definition", "ResolvedConfiguration"),
    ("function_definition", "__init__"),
    ("function_definition", "get"),
    ("function_definition", "_coerce_bool"),
    ("function_definition", "_coerce_int"),
    ("function_definition", "merge_resolved_configuration"),
)
_FIXTURE_OVERSIZED_DECLARATION = "merge_resolved_configuration"
_FIXTURE_OVERSIZED_SIGNATURE_CHARS = 1520

_LIVE_B = 1000
_LIVE_SYNTHESIS_MANIFEST_CHARS = 12000

_FIXTURE_CHUNK_PAYLOAD_CHARS = (649, 834, 834)
_FIXTURE_CHUNK_OUTER_BYTE_RANGES = ((0, 649), (649, 1483), (1483, 2317))
_FIXTURE_CHUNK_BOUNDARY_TYPES = (
    ("file-start", "semantic-unit"),
    ("semantic-unit", "balanced-codepoint"),
    ("balanced-codepoint", "file-end"),
)
_FIXTURE_CHUNK_CLOSE_REASONS = (
    "oversized-unit-isolation",
    "continuation",
    "continuation",
)

_FIXTURE_LEAF_CALLS = 3
_FIXTURE_UNIT_CONSOLIDATION_NODES = 1
_FIXTURE_GENERAL_NODES = 0
_FIXTURE_REDUCER_CALLS = 1
_FIXTURE_FINAL_CALLS = 1
_FIXTURE_INITIAL_PROVIDER_CALLS = 5


def _live_validation_canonical() -> tuple[str, str]:
    """The fixture's `(raw content hash, canonical decoded snapshot)` from the
    real production loader -- the same `read_source_snapshot` the pipeline
    consumes."""
    return read_source_snapshot(_LIVE_VALIDATION_FIXTURE)


def _live_validation_plan_and_tree(content: str):
    """The production division plan and reduction tree for the fixture at the
    exact section 9.3 live config (`max_content_chars=1000`, automatic
    `synthesis_manifest_chars=12000`)."""
    plan = build_division_plan(
        rel_path="main.py",
        language="python",
        content=content,
        source_budget_chars=_LIVE_B,
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=_LIVE_SYNTHESIS_MANIFEST_CHARS,
        language="python",
        imports=(),
    )
    return plan, tree


def test_live_validation_fixture_exists_and_bytes_are_frozen() -> None:
    """The mandated section 9.3 fixture is present, syntax-valid, admitted by
    the scanner byte gate, and byte-frozen. Its canonical decoded snapshot is
    the raw UTF-8 (no BOM, no CR), and the separate raw content hash binds the
    original bytes exactly."""
    assert _LIVE_VALIDATION_FIXTURE.is_file(), _LIVE_VALIDATION_FIXTURE

    raw = _LIVE_VALIDATION_FIXTURE.read_bytes()
    assert len(raw) == _FIXTURE_RAW_BYTES
    assert hashlib.sha256(raw).hexdigest() == _FIXTURE_SHA256

    # Valid Python (the live copy is documented as `main.py`).
    compile(raw.decode("utf-8"), "oversized_signature.py", "exec")

    # Below the scanner's byte-size admission cap -- it must reach split
    # planning, never be size-skipped first.
    assert len(raw) <= DEFAULTS["max_file_size_kb"] * 1024

    content_hash, canonical = _live_validation_canonical()
    assert content_hash == hashlib.sha256(raw).hexdigest()
    assert raw[:3] != b"\xef\xbb\xbf" and b"\r" not in raw
    assert canonical == raw.decode("utf-8")
    assert len(canonical) == _FIXTURE_CANONICAL_CHARS


@requires_structure_pack
def test_live_validation_fixture_structure_census_is_frozen() -> None:
    """Structural parsing mode, atom/symbol census, and the one oversized
    declaration are frozen. Exactly one declaration exceeds 600 characters and
    it stays well below the 2,000 hard bound, so it is copied in full rather
    than shortened."""
    _content_hash, canonical = _live_validation_canonical()
    plan, _tree = _live_validation_plan_and_tree(canonical)

    assert plan.structural_mode == _FIXTURE_STRUCTURAL_MODE
    assert plan.source_chars == _FIXTURE_CANONICAL_CHARS
    assert plan.source_bytes == _FIXTURE_RAW_BYTES
    assert len(plan.atoms) == _FIXTURE_ATOM_COUNT

    census = tuple((s.kind, s.qualified_name) for s in plan.symbols)
    assert census == _FIXTURE_SYMBOL_CENSUS

    over_600 = [
        (s.qualified_name, len(s.signature))
        for s in plan.symbols
        if len(s.signature) > 600
    ]
    assert over_600 == [
        (_FIXTURE_OVERSIZED_DECLARATION, _FIXTURE_OVERSIZED_SIGNATURE_CHARS)
    ]
    assert _FIXTURE_OVERSIZED_SIGNATURE_CHARS < MAX_LEAF_SYMBOL_SIGNATURE_CHARS


@requires_structure_pack
def test_live_validation_fixture_division_plan_is_frozen() -> None:
    """Exact chunk payload sizes, exact chunk source ranges, exact ordered
    boundary types and close reasons, at least one balanced-codepoint cut,
    exactly-once contiguous coverage, and byte-exact reconstruction of
    `SourceIndex.data` from the planned chunks."""
    _content_hash, canonical = _live_validation_canonical()
    plan, _tree = _live_validation_plan_and_tree(canonical)

    assert len(plan.chunks) == _FIXTURE_LEAF_CALLS
    assert len(plan.chunks) <= MAX_CHUNKS_PER_FILE

    payloads = tuple(c.payload_chars for c in plan.chunks)
    assert payloads == _FIXTURE_CHUNK_PAYLOAD_CHARS
    assert all(p <= _LIVE_B for p in payloads)

    outer_ranges = tuple(
        (c.owning_ranges[0].start_byte, c.owning_ranges[-1].end_byte)
        for c in plan.chunks
    )
    assert outer_ranges == _FIXTURE_CHUNK_OUTER_BYTE_RANGES
    # Contiguous, exactly-once tiling of the whole canonical snapshot.
    assert outer_ranges[0][0] == 0
    assert outer_ranges[-1][1] == len(canonical.encode("utf-8"))
    for (_a0, a1), (b0, _b1) in zip(outer_ranges, outer_ranges[1:]):
        assert a1 == b0

    boundary_types = tuple(
        (c.start_boundary, c.end_boundary) for c in plan.chunks
    )
    assert boundary_types == _FIXTURE_CHUNK_BOUNDARY_TYPES
    close_reasons = tuple(c.close_reason for c in plan.chunks)
    assert close_reasons == _FIXTURE_CHUNK_CLOSE_REASONS

    interior_cut_kinds = [
        c.end_boundary for c in plan.chunks if c.continuation_after
    ]
    assert interior_cut_kinds == ["balanced-codepoint"]
    assert "balanced-codepoint" in {b for pair in boundary_types for b in pair}

    # Exactly one oversized semantic unit, subdivided locally; the fitting
    # units keep their identity and co-pack into the first leaf call.
    unit_summaries = split_plan_unit_summaries(plan)
    oversized = [u for u in unit_summaries if u["arithmetic_piece_count"] > 1]
    assert len(oversized) == 1
    (oversized_unit,) = oversized
    assert oversized_unit["crlf_safe_piece_count"] == 2
    assert oversized_unit["atomicity_extra_piece_count"] == 0
    assert [p["payload_chars"] for p in oversized_unit["pieces"]] == [834, 834]
    assert [p["end_boundary"] for p in oversized_unit["pieces"]] == [
        "balanced-codepoint",
        "file-end",
    ]
    leaf_descriptors = split_plan_leaf_descriptors(plan)
    assert [d["close_reason"] for d in leaf_descriptors] == list(
        _FIXTURE_CHUNK_CLOSE_REASONS
    )
    assert leaf_descriptors[0]["constituents_total"] == 7

    # Byte-exact reconstruction of the canonical decoded snapshot.
    reconstructed = "".join(c.payload for c in plan.chunks)
    assert reconstructed == canonical
    assert reconstructed.encode("utf-8") == canonical.encode("utf-8")

    # No CRLF-atomicity extra pieces for a normal canonically-loaded file.
    assert all(
        u["atomicity_extra_piece_count"] == 0 for u in unit_summaries
    )


@requires_structure_pack
def test_live_validation_fixture_reduction_topology_is_frozen() -> None:
    """Exact reducer count and topology, and the exact initial provider-call
    count (leaf + reducer + final). The oversized unit's continuations
    consolidate under one unit-consolidation reducer; the automatic synthesis
    manifest budget is carried unchanged at 12,000."""
    _content_hash, canonical = _live_validation_canonical()
    plan, tree = _live_validation_plan_and_tree(canonical)

    assert len(tree.unit_consolidation_nodes) == _FIXTURE_UNIT_CONSOLIDATION_NODES
    assert len(tree.general_nodes) == _FIXTURE_GENERAL_NODES
    assert len(tree.all_intermediate_nodes) == _FIXTURE_REDUCER_CALLS
    assert tree.final_node.phase == "final"
    assert len(tree.final_node.child_ids) == 2
    assert len(tree.final_node.leaf_ids) == _FIXTURE_LEAF_CALLS
    assert reduction_depth(tree) == 1
    assert tree.synthesis_manifest_chars == _LIVE_SYNTHESIS_MANIFEST_CHARS
    assert tree.synthesis_manifest_chars >= MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
    assert tree.division_plan_digest == plan.plan_digest

    initial_provider_calls = (
        len(plan.chunks) + len(tree.all_intermediate_nodes) + _FIXTURE_FINAL_CALLS
    )
    assert initial_provider_calls == _FIXTURE_INITIAL_PROVIDER_CALLS


@requires_structure_pack
def test_live_validation_fixture_planning_is_deterministic() -> None:
    """Repeated planning over the same fixture bytes yields identical
    identities -- plan digest, tree digest, chunk IDs, and chunk sizes."""
    _content_hash, canonical = _live_validation_canonical()
    plan_a, tree_a = _live_validation_plan_and_tree(canonical)
    plan_b, tree_b = _live_validation_plan_and_tree(canonical)

    assert plan_a.plan_digest == plan_b.plan_digest
    assert tree_a.tree_digest == tree_b.tree_digest
    assert [c.chunk_id for c in plan_a.chunks] == [
        c.chunk_id for c in plan_b.chunks
    ]
    assert [c.payload_chars for c in plan_a.chunks] == [
        c.payload_chars for c in plan_b.chunks
    ]
    assert [n.node_id for n in tree_a.all_nodes] == [
        n.node_id for n in tree_b.all_nodes
    ]


@requires_structure_pack
def test_live_validation_fixture_dry_run_topology_matches_frozen_plan(
    tmp_path, monkeypatch
) -> None:
    """A provider-free dry run at the exact section 9.3 live config -- the
    fixture copied verbatim as `main.py`, `single + split`,
    `max_content_chars=1000`, no profile, no correction key -- reports the
    frozen initial topology: 3 leaf + 1 unit-consolidation + 1 final = 5
    initial provider calls, zero review calls, and zero CRLF-atomicity extra
    chunks. It must not construct a provider."""
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("dry-run planning constructed a provider"),
    )
    (tmp_path / "main.py").write_bytes(_LIVE_VALIDATION_FIXTURE.read_bytes())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": _LIVE_B,
            "file_retry_attempts": 0,
            "allow_partial": False,
            "parallel_agents": False,
            "propagate_changes": False,
            "dry_run": True,
        },
    )

    assert stats["split_divided_files"] == 1
    assert stats["split_chunks"] == _FIXTURE_LEAF_CALLS
    assert stats["split_oversized_units"] == 1
    assert stats["split_unit_consolidation_calls_planned"] == _FIXTURE_REDUCER_CALLS
    assert stats["split_general_reduction_calls_planned"] == _FIXTURE_GENERAL_NODES
    assert stats["split_final_synthesis_calls_planned"] == _FIXTURE_FINAL_CALLS
    assert stats["split_internal_manifest_budget_chars"] == _LIVE_SYNTHESIS_MANIFEST_CHARS
    assert stats["split_boundary_cuts_balanced_codepoint"] == 1
    assert stats["split_boundary_cuts_syntax"] == 0
    assert stats["split_boundary_cuts_physical_line"] == 0
    assert stats["split_crlf_atomicity_extra_chunks"] == 0
    assert stats["split_chunk_payload_chars_min"] == _FIXTURE_CHUNK_PAYLOAD_CHARS[0]
    assert stats["split_chunk_payload_chars_max"] == max(_FIXTURE_CHUNK_PAYLOAD_CHARS)

    assert stats["initial_provider_calls_planned"] == _FIXTURE_INITIAL_PROVIDER_CALLS
    assert stats["initial_documentation_calls_planned"] == _FIXTURE_INITIAL_PROVIDER_CALLS
    assert stats["total_calls_planned"] == _FIXTURE_INITIAL_PROVIDER_CALLS
    assert stats["documentation_calls_planned"] == _FIXTURE_INITIAL_PROVIDER_CALLS
    assert stats["prompt_review_calls_planned"] == 0
    assert stats["unit_documentation_calls_planned"] == _FIXTURE_LEAF_CALLS
