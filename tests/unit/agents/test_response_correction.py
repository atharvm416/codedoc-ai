"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import re
import pytest
import codedoc.core.file_division as file_division
from codedoc.agents import response_correction_agent as rca
from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
from codedoc.agents.response_diagnostics import (
    MAX_CORRECTION_RESPONSE_CHARS,
    MAX_PATH_CHARS,
    MAX_REMOVAL_ENTRIES,
    REASON_FIXED_CAP_EXCEEDED,
    REASON_MISSING_REQUIRED,
    REMOVAL_ITEM_LIMIT,
    REMOVAL_RESPONSE_CAP,
    REMOVAL_UNKNOWN_FIELD,
    CorrectionLedger,
    RemovedField,
    ResponseDiagnostic,
    process_fixed_capsule_response,
)
from codedoc.core.execution import _process_one_file
from codedoc.utils.errors import ResponseContractError
from tests.support.response_correction_cases import RoutingProvider
from tests.support.response_correction_cases import _orch
from tests.support.response_correction_cases import _request

def test_single_valid_one_call(tmp_path):
    prov = RoutingProvider()
    res = _process_one_file(_request(tmp_path), _orch(prov, enabled=False))
    assert prov.calls == 1
    assert res["description"] == "A file."

def test_single_invalid_then_corrected_two_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=True)
    res = _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 2
    assert prov.correction_calls == 1
    assert res["description"] == "A file."

def test_single_invalid_then_failed_two_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"}, correction_response={"role_in_system": "r"})
    orch = _orch(prov, enabled=True)
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 2
    assert prov.correction_calls == 1
    assert caught.value.correction_attempted is True

def test_single_disabled_no_correction_call(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=False)
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 1
    assert prov.correction_calls == 0
    assert caught.value.correction_attempted is False

def test_triple_all_valid_three_calls(tmp_path):
    prov = RoutingProvider()
    res = _process_one_file(_request(tmp_path, mode="triple"), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 3
    assert res["state"] == "checked"

def test_triple_one_invalid_corrected_four_calls_sibling_isolation(tmp_path):
    prov = RoutingProvider(fail_agents={"structure"})
    res = _process_one_file(_request(tmp_path, mode="triple"), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 4
    assert prov.correction_calls == 1
    # Successful siblings are each called exactly once, never rerun.
    assert prov.per_agent_initial["dependency"] == 1
    assert prov.per_agent_initial["documentation"] == 1
    assert prov.per_agent_initial["structure"] == 1
    assert res["state"] == "checked"

def test_triple_all_three_invalid_six_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"structure", "dependency", "documentation"})
    res = _process_one_file(_request(tmp_path, mode="triple"), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 6
    assert prov.correction_calls == 3
    assert res["state"] == "checked"

def test_correction_prompt_includes_original_response_and_is_capped(tmp_path):
    huge = "z" * (MAX_CORRECTION_RESPONSE_CHARS + 5000)
    prov = RoutingProvider(fail_agents={"combined"})
    # Force a huge original response by returning it on the initial call.
    orig = json.dumps({"role_in_system": huge})

    class HugeInitial(RoutingProvider):
        def complete_json(self, prompt, system=""):
            if "Previous response (verbatim" not in prompt:
                self.calls += 1
                self.per_agent_initial["combined"] = 1
                return orig
            return super().complete_json(prompt, system)

    prov = HugeInitial(fail_agents={"combined"})
    _process_one_file(_request(tmp_path), _orch(prov, enabled=True))
    # The correction prompt embedded the capped original, never the full text: a
    # long run of the original survives, but the full oversized string does not.
    assert prov.last_correction_prompt is not None
    assert huge not in prov.last_correction_prompt
    assert ("z" * 6000) in prov.last_correction_prompt
    assert prov.last_correction_prompt.count("z") <= MAX_CORRECTION_RESPONSE_CHARS

def test_correction_preserves_usable_facts(tmp_path):
    # Initial response has a valid role but no description; correction fills it.
    prov = RoutingProvider(
        fail_agents={"combined"},
        correction_response={"description": "Filled.", "role_in_system": "kept role"},
    )
    res = _process_one_file(_request(tmp_path), _orch(prov, enabled=True))
    assert res["description"] == "Filled."
    assert res["role_in_system"] == "kept role"


# ---------------------------------------------------------------------------
# 0.14.4: every final response-contract error names its closed reason code,
# and leaks nothing else (source, prompt, raw response, or per-field detail).
# ---------------------------------------------------------------------------

def test_disabled_path_message_carries_reason_code_and_leaks_no_detail(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=False)

    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)

    message = str(caught.value)
    assert caught.value.diagnostic["reason_code"] in message
    assert "disabled" in message
    # No per-field removal detail, no raw provider response text.
    assert "role_in_system" not in message
    assert '"r"' not in message


def test_provider_fault_path_message_carries_original_reason_code_and_leaks_no_detail(
    tmp_path,
):
    prov = RoutingProvider(
        fail_agents={"combined"},
        raise_on_correction=RuntimeError("temporary provider outage"),
    )
    orch = _orch(prov, enabled=True)

    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)

    message = str(caught.value)
    assert caught.value.diagnostic["reason_code"] in message
    assert "correction provider call failed" in message
    assert "role_in_system" not in message
    assert '"r"' not in message


def test_still_invalid_path_message_carries_corrected_reason_code_and_leaks_no_detail(
    tmp_path,
):
    prov = RoutingProvider(
        fail_agents={"combined"}, correction_response={"role_in_system": "still bad"}
    )
    orch = _orch(prov, enabled=True)

    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)

    message = str(caught.value)
    assert caught.value.diagnostic["reason_code"] in message
    assert "still failed the schema contract" in message
    assert "role_in_system" not in message
    assert "still bad" not in message


# ===========================================================================
# 0.14.9 F-1 / G-1..G-5: the shared fixed-capsule cap-repair instruction,
# its exact two-condition gate, the closed field-path resolver, privacy, and
# the field-summary ceiling. These drive the real ``_build_prompt`` renderer.
# ===========================================================================

_LEAF_INPUT = {
    "agent": "leaf",
    "mode": "split-leaf",
    "file_path": "m.py",
    "language": "",
    "content": "def f(): pass\n",
    "imports": [],
    "shape_block": "SHAPE-BLOCK-PLACEHOLDER",
    "original_response": '{"description": "x"}',
}


def _agent() -> ResponseCorrectionAgent:
    return ResponseCorrectionAgent(None, None, CorrectionLedger(True), True)


def _diag(
    *,
    reason_code: str = REASON_FIXED_CAP_EXCEEDED,
    removed: tuple[RemovedField, ...] = (),
    observed: frozenset[str] = frozenset({REMOVAL_RESPONSE_CAP}),
) -> ResponseDiagnostic:
    return ResponseDiagnostic(
        stage="clean",
        reason_code=reason_code,
        agent="leaf",
        file_path="m.py",
        removed=removed,
        observed_removal_reasons=observed,
    )


def _prompt_for(diagnostic: ResponseDiagnostic, **overrides) -> str:
    _system, prompt = _agent()._build_prompt({**_LEAF_INPUT, **overrides}, diagnostic)
    return prompt


def _cap(field: str, detail: str = "length 313 exceeds cap 300") -> RemovedField:
    return RemovedField(field=field, reason_code=REMOVAL_RESPONSE_CAP, detail=detail)


# --- the exact two-condition gate (9.1 items 17-19) -------------------------

def test_gate_holds_only_when_both_conditions_are_met():
    both = _diag(removed=(_cap("description"),), observed=frozenset({REMOVAL_RESPONSE_CAP}))
    assert rca._cap_repair_applies(both) is True

    wrong_reason = _diag(
        reason_code=REASON_MISSING_REQUIRED,
        removed=(_cap("description"),),
        observed=frozenset({REMOVAL_RESPONSE_CAP}),
    )
    assert rca._cap_repair_applies(wrong_reason) is False

    no_response_cap = _diag(
        removed=(RemovedField("functions[0]", REMOVAL_ITEM_LIMIT, "index 40 over cap 40"),),
        observed=frozenset({REMOVAL_ITEM_LIMIT}),
    )
    assert rca._cap_repair_applies(no_response_cap) is False


def test_cap_rule_absent_for_configurable_route_global_cap_response_cap():
    """9.1 item 17 / mutation 22: a real ``single``/``combined``
    ``missing_required`` diagnostic that carries ``_enforce_global_cap``
    ``response_cap`` removals never receives the cap-specific rule, and the
    correction prompt is byte-identical to the pre-patch prompt (the
    ``cap_repair_rule`` slot resolves to the empty string, so appending it is a
    no-op)."""
    from codedoc.agents.response_cleaning import clean_combined_report
    from codedoc.agents.response_diagnostics import process_response

    raw = {
        "role_in_system": "core",
        "dependencies_analysis": {
            "warnings": [f"warn-{i}-" + "w" * 250 for i in range(32)],
            "internal": [f"int-{i}-" + "i" * 250 for i in range(32)],
            "external": [f"ext-{i}-" + "e" * 250 for i in range(32)],
        },
        "key_concepts": [f"kc-{i}-" + "c" * 250 for i in range(16)],
    }
    with pytest.raises(ResponseContractError) as caught:
        process_response(
            json.dumps(raw), mode="single", agent="combined", file_path="m.py",
            clean_reporter=clean_combined_report, resolved_shape=None,
        )
    diagnostic = caught.value.diagnostic
    assert diagnostic.reason_code == "missing_required"
    assert any(r.reason_code == "response_cap" for r in diagnostic.removed)
    # The configurable route never populates observed_removal_reasons at all
    # (it is fixed-capsule-only), so BOTH gate conditions fail here.
    assert diagnostic.observed_removal_reasons == frozenset()

    assert rca._cap_repair_applies(diagnostic) is False
    ci = {**_LEAF_INPUT, "mode": "single", "agent": "combined",
          "original_response": json.dumps(raw)}
    prompt = _agent()._build_prompt(ci, diagnostic)[1]
    assert "Cap repair" not in prompt
    # Byte-identity to pre-patch: reproduce the pre-patch template (its only
    # change is the trailing ``{cap_repair_rule}`` placeholder) and format it
    # exactly as ``_build_prompt`` does with the slot empty.
    pre_patch_template = rca._PROMPT_TEMPLATE.replace("{cap_repair_rule}", "")
    expected = pre_patch_template.format(
        mode="single", agent="combined", language="", file_path="m.py",
        reason=diagnostic.reason_code,
        field_summary=ResponseCorrectionAgent._field_summary(diagnostic),
        imports=[], content=ci["content"], shape_block=ci["shape_block"],
        original_response=json.dumps(raw)[:MAX_CORRECTION_RESPONSE_CHARS],
        terminology_rules=rca.NARRATIVE_TERMINOLOGY_RULES + "\n",
    )
    assert prompt == expected


# --- the instruction: clauses, single render, numeric target ---------------

def test_cap_rule_renders_clauses_1_2_5_and_the_260_target_for_prose():
    prompt = _prompt_for(_diag(removed=(_cap("description"),)))
    assert prompt.count("Cap repair") == 1
    assert "rejected in full" in prompt          # clause 1
    assert "must not be copied" in prompt        # clause 2
    assert "within 260 characters" in prompt     # clause 4 numeric target
    assert "shorter, meaning-preserving value" in prompt  # clause 3
    # clause 5: the general preserve rule survives AND is explicitly overridden
    assert "Preserve every valid fact already present in the previous response" in prompt
    assert "every other valid field and fact" in prompt


def test_cap_rule_renders_once_for_multiple_response_cap_removals():
    prompt = _prompt_for(
        _diag(removed=(_cap("description"), _cap("functions[0].description")))
    )
    assert prompt.count("Cap repair") == 1
    assert "description: within 260 characters" in prompt
    assert "functions[0].description: within 260 characters" in prompt


def test_cap_rule_present_even_when_removed_tuple_has_no_response_cap_entry():
    """9.1 item 18: gate reads ``observed_removal_reasons``; a saturated
    ``removed`` tuple with no surviving ``response_cap`` entry still renders the
    rule, in the category-neutral no-path form (names no field, invents no
    number, neither commands nor prohibits prose shortening)."""
    saturated = tuple(
        RemovedField(f"unknown_{i}", REMOVAL_UNKNOWN_FIELD, "str[1]")
        for i in range(MAX_REMOVAL_ENTRIES)
    )
    prompt = _prompt_for(_diag(removed=saturated, observed=frozenset({REMOVAL_RESPONSE_CAP})))
    _assert_saturated_no_path_fallback(prompt.split("Cap repair", 1)[1], prompt)


def _saturated_leaf_diagnostic(**extra_fields):
    """Run the real fixed-capsule path over a leaf capsule with enough earlier
    unknown-field removals to fill ``diagnostic.removed``, then *extra_fields*
    (an over-cap ``description`` or ``functions`` signature) whose
    ``response_cap`` entry is pushed out of the bounded tuple."""
    from codedoc.agents.response_cleaning import clean_leaf_capsule_report

    raw = {f"unknown_{index}": "x" for index in range(MAX_REMOVAL_ENTRIES + 8)}
    raw.update(extra_fields)
    with pytest.raises(ResponseContractError) as caught:
        process_fixed_capsule_response(
            json.dumps(raw), label="split-leaf", agent="leaf", file_path="m.py",
            clean_reporter=clean_leaf_capsule_report,
            requested_paths=("description", "functions", "classes", "exports"),
            required_paths=("description",),
        )
    return caught.value.diagnostic


def _assert_saturated_no_path_fallback(cap: str, prompt: str) -> None:
    """The category-neutral saturated/no-path fallback (Part 2 final
    correction): clauses 1, 2, 5, a neutral governing-shape-contract deferral,
    no numeric target, and -- critically -- NEITHER a prose-shortening command
    NOR a prose-shortening prohibition. It invents no identifier or remedy."""
    # clauses 1, 2, 5
    assert "rejected in full" in cap
    assert "must not be copied" in cap
    assert "every other valid field and fact" in cap
    assert "Preserve every valid fact already present in the previous response" in prompt
    # neutral deferral, explicitly no field-specific rewrite instruction
    assert "governing shape contract" in cap
    assert "no field-specific rewrite instruction" in cap
    # no numeric target -- the field path was unavailable
    assert "within" not in cap
    assert not re.search(r"\d", cap)
    # category-neutral: neither a positive shortening instruction ...
    assert "shorter, meaning-preserving" not in cap
    assert "Rewrite" not in cap
    assert "rewrite shorter" not in cap.lower()
    # ... nor a negative shortening prohibition
    assert "do not shorten" not in cap.lower()
    assert "must not shorten" not in cap.lower()
    assert "not shorten any prose" not in cap.lower()
    # invents nothing
    assert "must not be used to invent" in cap


def test_saturated_diagnostic_hiding_an_over_cap_signature_defers_conservatively():
    """A *real* leaf capsule whose earlier unknown-field removals saturate
    ``diagnostic.removed`` pushes the over-cap ``functions[i].signature``
    ``response_cap`` entry out of the bounded tuple, while
    ``observed_removal_reasons`` still carries ``response_cap``. The gate fires;
    the rendered rule uses the category-neutral no-path fallback and never emits
    a rewrite-shorter instruction, a numeric target, or an invented identifier
    remedy that could corrupt a source-backed signature."""
    from codedoc.core.file_division import MAX_LEAF_SYMBOL_SIGNATURE_CHARS

    diagnostic = _saturated_leaf_diagnostic(
        description="A valid fragment description, well under the cap.",
        functions=[
            {"name": "alpha", "signature": "s" * (MAX_LEAF_SYMBOL_SIGNATURE_CHARS + 1)}
        ],
    )

    assert diagnostic.reason_code == REASON_FIXED_CAP_EXCEEDED
    assert "response_cap" in diagnostic.observed_removal_reasons
    assert not any(r.reason_code == REMOVAL_RESPONSE_CAP for r in diagnostic.removed)
    assert len(diagnostic.removed) <= MAX_REMOVAL_ENTRIES

    assert rca._cap_repair_applies(diagnostic) is True
    prompt = _prompt_for(diagnostic)
    assert "Cap repair" in prompt
    _assert_saturated_no_path_fallback(prompt.split("Cap repair", 1)[1], prompt)


def test_saturated_diagnostic_hiding_an_over_cap_description_defers_conservatively():
    """The plan's own G-2 example: an over-cap leaf ``description`` can also be
    pushed out of the bounded ``diagnostic.removed`` by enough earlier
    unknown-field removals. The no-path fallback cannot know the hidden field is
    prose, so it must NEITHER command prose shortening NOR prohibit it -- that
    field would still need compliant shorter prose -- and must state no numeric
    target and invent no remedy."""
    from codedoc.core.file_division import MAX_LEAF_DESCRIPTION_CHARS

    diagnostic = _saturated_leaf_diagnostic(
        description="d" * (MAX_LEAF_DESCRIPTION_CHARS + 13)
    )

    assert diagnostic.reason_code == REASON_FIXED_CAP_EXCEEDED
    assert "response_cap" in diagnostic.observed_removal_reasons
    assert not any(r.reason_code == REMOVAL_RESPONSE_CAP for r in diagnostic.removed)
    assert len(diagnostic.removed) <= MAX_REMOVAL_ENTRIES

    assert rca._cap_repair_applies(diagnostic) is True
    prompt = _prompt_for(diagnostic)
    assert "Cap repair" in prompt
    _assert_saturated_no_path_fallback(prompt.split("Cap repair", 1)[1], prompt)


def test_surviving_description_path_still_gets_the_derived_260_target():
    """Non-regression: when the over-cap ``description`` DOES survive
    ``diagnostic.removed`` (few earlier removals), the known-prose branch still
    renders the derived 260-character target and the meaning-preserving
    shorter-value instruction, and the saturated fallback line is absent."""
    from codedoc.core.file_division import MAX_LEAF_DESCRIPTION_CHARS

    diagnostic = _diag(
        removed=(
            RemovedField("extra", REMOVAL_UNKNOWN_FIELD, "str[1]"),
            _cap("description", f"length {MAX_LEAF_DESCRIPTION_CHARS + 13} exceeds cap 300"),
        )
    )
    cap = _prompt_for(diagnostic).split("Cap repair", 1)[1]
    assert "description: within 260 characters" in cap
    assert "shorter, meaning-preserving value" in cap
    assert "could not be identified from this diagnostic" not in cap


def test_cap_rule_target_is_derived_from_the_bound_not_hardcoded_260(monkeypatch):
    """9.1 item 25 / mutation 25: monkeypatch a bound to a non-300 value and the
    rendered target moves proportionally."""
    monkeypatch.setattr(file_division, "MAX_LEAF_SYMBOL_DESCRIPTION_CHARS", 150)
    prompt = _prompt_for(_diag(removed=(_cap("functions[0].description"),)))
    # floor(150 * 260 / 300) == 130
    assert "functions[0].description: within 130 characters" in prompt
    assert "within 260 characters" not in prompt


# --- identifier / unknown fields are excluded from the rewrite target ---
# (9.1 items 21, 24, 26; mutation 27; Part 2 correction finding 2)

@pytest.mark.parametrize(
    ("field", "category"),
    [
        ("functions[0].name", "symbol name"),
        ("classes[2].name", "symbol name"),
        ("functions[1].signature", "signature"),
        ("classes[0].signature", "signature"),
        ("exports[3]", "export"),
        ("weird[0].thing", "unmatched path"),
    ],
)
def test_cap_rule_identifier_categories_defer_with_no_prose_or_target(field, category):
    """Every identifier category -- symbol name, signature, export -- and an
    unmatched path get clauses 1, 2, 5 and a deferral to the governing shape
    contract already supplied in the prompt, with clauses 3 and 4 (rewrite
    shorter / numeric target) absent and no invented shorter identifier."""
    prompt = _prompt_for(_diag(removed=(_cap(field),)))
    cap = prompt.split("Cap repair", 1)[1]

    # clauses 1, 2, 5 retained
    assert "rejected in full" in cap
    assert "must not be copied" in cap
    assert "every other valid field and fact" in cap
    assert "Preserve every valid fact already present in the previous response" in prompt
    # the field is named and deferred, not rewritten
    assert field in cap
    assert "governing shape contract" in cap
    assert "must not invent a shorter identifier" in cap
    # clauses 3 and 4 absent for this category
    assert "shorter, meaning-preserving" not in cap
    assert f"{field}: within" not in cap
    assert " characters" not in cap
    assert not re.search(r"within \d", cap)


def test_cap_rule_name_field_deferral_does_not_force_the_signature_contract():
    """Finding 2: a rejected ``functions[i].name`` must not be told to follow the
    signature/exports rules specifically -- its governing contract is the fixed
    shape block. The generic 'governing shape contract for that field' wording is
    accurate and non-conflicting for name, signature, and export alike."""
    for field in ("functions[0].name", "functions[0].signature", "exports[0]"):
        cap = _prompt_for(_diag(removed=(_cap(field),))).split("Cap repair", 1)[1]
        # one accurate deferral, not a name-field-inaccurate "follow the
        # signature/exports shape rules" directive.
        assert "governing shape contract for that field" in cap
        assert "follow the signature/exports shape rules" not in cap


# --- the closed resolver table (items 24, 25) ------------------------------

def test_resolver_is_complete_over_every_fixed_capsule_field_path():
    prose = {
        "description": file_division.MAX_LEAF_DESCRIPTION_CHARS,
        "narrative": file_division.MAX_REDUCTION_NARRATIVE_CHARS,
        "functions[0].description": file_division.MAX_LEAF_SYMBOL_DESCRIPTION_CHARS,
        "classes[2].description": file_division.MAX_LEAF_SYMBOL_DESCRIPTION_CHARS,
    }
    identifier = {
        "functions[0].name": file_division.MAX_LEAF_SYMBOL_NAME_CHARS,
        "classes[0].name": file_division.MAX_LEAF_SYMBOL_NAME_CHARS,
        "functions[9].signature": file_division.MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
        "classes[0].signature": file_division.MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
        "exports[0]": file_division.MAX_LEAF_EXPORT_ITEM_CHARS,
    }
    for path, bound in prose.items():
        assert rca._resolve_field_cap(path) == (bound, "prose"), path
    for path, bound in identifier.items():
        assert rca._resolve_field_cap(path) == (bound, "identifier"), path
    assert rca._resolve_field_cap("not_a_capsule_field") is None
    assert rca._resolve_field_cap("dependencies_analysis.warnings[0]") is None


def test_prose_targets_for_the_two_live_failure_routes_are_260():
    assert rca._prose_correction_target(file_division.MAX_LEAF_DESCRIPTION_CHARS) == 260
    assert rca._prose_correction_target(file_division.MAX_REDUCTION_NARRATIVE_CHARS) == 260


# --- privacy & the 10,000-char field-summary ceiling (item 22, mutations 29-30) ---

def test_field_summary_is_value_free_and_omits_removed_detail():
    diagnostic = _diag(
        removed=(
            RemovedField("description", REMOVAL_RESPONSE_CAP, "length 999 exceeds cap 300"),
            RemovedField("exports[0]", REMOVAL_RESPONSE_CAP, "length 400 exceeds cap 256"),
        )
    )
    summary = ResponseCorrectionAgent._field_summary(diagnostic)
    assert summary == "description (response_cap); exports[0] (response_cap)"
    assert "length 999 exceeds cap 300" not in summary
    assert "exceeds cap" not in summary


def test_field_summary_ceiling_is_a_live_invariant():
    assert rca._FIELD_SUMMARY_CEILING_CHARS == 10_000
    # A normal max-size diagnostic stays comfortably under the ceiling.
    normal = _diag(
        removed=tuple(
            RemovedField("x" * MAX_PATH_CHARS, REMOVAL_RESPONSE_CAP, "d")
            for _ in range(MAX_REMOVAL_ENTRIES)
        )
    )
    assert len(ResponseCorrectionAgent._field_summary(normal)) <= 10_000
    # An impossible over-cap construction trips the invariant rather than
    # silently truncating (mutation 30).
    absurd = _diag(
        removed=tuple(
            RemovedField("y" * MAX_PATH_CHARS, REMOVAL_RESPONSE_CAP, "d")
            for _ in range(400)
        )
    )
    with pytest.raises(AssertionError):
        ResponseCorrectionAgent._field_summary(absurd)


# --- identity preservation by value (9.1 item 34, mutations 17-19) ---------

def test_only_the_two_required_split_revisions_advanced():
    """9.1 item 34 / mutations 17-19: exactly the two section 5.6.2 revisions
    advanced; every identity in section 5.6.3 is unchanged, asserted by value."""
    from codedoc.agents import file_synthesis_agent
    from codedoc.core import record_meta

    assert file_division.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v11"
    assert file_division.REDUCER_PROMPT_REVISION == "file-reduction-v4"

    assert record_meta.ANALYSIS_REVISION == "file-doc-v4"
    assert file_division.FINAL_SYNTHESIS_REVISION == "file-synthesis-v4"
    assert file_synthesis_agent.SYNTHESIS_PROMPT_REVISION == "file-synthesis-v4"
    assert file_division.LEDGER_SCHEMA_REVISION == "fact-ledger-v7"
    assert file_division.REDUCTION_CAPSULE_SCHEMA_REVISION == "reduction-capsule-v1"
    assert file_division.REDUCTION_PACKING_REVISION == "reduction-packing-v5"
    assert file_division.STRUCTURE_SCHEMA_REVISION == "source-structure-v2"
    assert file_division.UNIT_SCHEMA_REVISION == "semantic-unit-v3"
    assert file_division.PACKER_SCHEMA_REVISION == "division-packer-v6"
    assert file_division.EXECUTION_IDENTITY_SCHEMA_REVISION == "division-execution-v6"
    assert file_division.LEAF_INPUT_DIGEST_REVISION == "leaf-input-v1"
    assert file_division.REDUCTION_INPUT_DIGEST_REVISION == "reduction-input-v1"
    assert file_division.FINAL_INPUT_DIGEST_REVISION == "final-input-v1"
