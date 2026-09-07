"""Section 3 (plan section 5.6 / workstream F): the closed deterministic
initialism rule and its delivery through the one-repair response contract.
"""

from __future__ import annotations

import json

import pytest

from codedoc.agents.narrative_terminology import (
    TerminologyEvidence,
    validate_narrative_terminology,
)
from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import (
    CorrectionLedger,
    process_response,
)
from codedoc.utils.errors import ResponseContractError
from tests.support.execution_requests import make_execution_request

# `DPR` appears as an uppercase source token; the phrase does not.
_SRC_WITH_DPR = "const DPR = 1;\nexport function render() { return DPR; }\n"
_UNSUPPORTED = "the DPR (Daily Progress Report) view"
_SUPPORTED_SRC = _SRC_WITH_DPR + "// DPR means Daily Progress Report here\n"


def _ev(source=_SRC_WITH_DPR, metadata=""):
    return TerminologyEvidence(source_text=source, metadata_text=metadata)


# ---------------------------------------------------------------------------
# closed rule: positive (all four conditions proven)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["description", "role_in_system", "usage_example"],
)
def test_provable_unsupported_expansion_is_removed_from_a_scalar_narrative_field(field):
    cleaned = {"description": "ok", field: _UNSUPPORTED}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert field not in out
    assert [r.field for r in removed] == [field]
    assert removed[0].reason_code == "unsupported_terminology"


def test_provable_unsupported_expansion_removed_from_key_concepts_item():
    cleaned = {"description": "ok", "key_concepts": ["safe concept", _UNSUPPORTED]}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out["key_concepts"] == ["safe concept"]
    assert removed[0].field == "key_concepts[1]"


def test_provable_unsupported_expansion_removed_from_symbol_description_only():
    cleaned = {
        "description": "ok",
        "functions": [
            {"name": "render", "description": _UNSUPPORTED},
            {"name": "other", "description": "fine"},
        ],
    }
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out["functions"][0] == {"name": "render"}
    assert out["functions"][1] == {"name": "other", "description": "fine"}
    assert removed[0].field == "functions[0].description"


# ---------------------------------------------------------------------------
# closed rule: negative / false-positive boundaries
# ---------------------------------------------------------------------------


def test_expansion_verbatim_in_source_is_retained():
    cleaned = {"description": _UNSUPPORTED}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=_SUPPORTED_SRC))
    assert out == cleaned
    assert removed == []


def test_expansion_in_trusted_metadata_is_retained():
    cleaned = {"description": _UNSUPPORTED}
    out, removed = validate_narrative_terminology(
        dict(cleaned), _ev(metadata="glossary: Daily Progress Report -> DPR")
    )
    assert out == cleaned
    assert removed == []


def test_lowercase_or_non_token_acronym_occurrence_does_not_trigger():
    # `dpr` lowercase, and `DPRThing` where DPR is not a standalone token.
    src = "const dpr = 1;\nclass DPRThing {}\n"
    cleaned = {"description": _UNSUPPORTED}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert out == cleaned and removed == []


def test_mismatched_initials_do_not_trigger():
    cleaned = {"description": "the DPR is a Daily Weekly Report"}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out == cleaned and removed == []


def test_ambiguous_lowercase_prose_is_retained():
    cleaned = {"description": "this is the daily progress report screen for DPR"}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out == cleaned and removed == []


def test_unrelated_capitalized_phrase_is_retained():
    cleaned = {"description": "Rendered by the New York Times widget."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out == cleaned and removed == []


def test_common_tech_acronym_expansion_is_rejected_without_verbatim_evidence():
    # P2-1: there is no universal-acronym bypass. A conventionally correct
    # expansion is still unsupported unless the complete phrase is verbatim in
    # trusted evidence.
    src = "fetch('/api');\nconst API = 1;\n"
    cleaned = {"description": "ok", "role_in_system": "Calls the Application Programming Interface."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert "role_in_system" not in out
    assert [r.field for r in removed] == ["role_in_system"]


def test_common_tech_acronym_expansion_is_retained_when_verbatim_in_source():
    src = "// exposes the Application Programming Interface\nconst API = 1;\n"
    cleaned = {"description": "Calls the Application Programming Interface."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert out == cleaned and removed == []


# ---------------------------------------------------------------------------
# P2-2: "verbatim" means exactly verbatim -- no case / whitespace normalization
# ---------------------------------------------------------------------------


def test_case_only_near_match_in_source_is_not_verbatim_support():
    src = _SRC_WITH_DPR + "// tracks the daily progress report\n"
    cleaned = {"description": "The DPR is a Daily Progress Report."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert "description" not in out
    assert [r.field for r in removed] == ["description"]


def test_whitespace_only_near_match_in_source_is_not_verbatim_support():
    src = _SRC_WITH_DPR + "// the Daily   Progress Report module\n"
    cleaned = {"description": "The DPR is a Daily Progress Report."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert "description" not in out
    assert [r.field for r in removed] == ["description"]


def test_exact_verbatim_match_including_spacing_is_support():
    # The candidate phrase, exactly as written in the narrative, is present
    # character-for-character in the source.
    src = _SRC_WITH_DPR + "// the Daily Progress Report module\n"
    cleaned = {"description": "The DPR is a Daily Progress Report."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert out == cleaned and removed == []


@pytest.mark.parametrize(
    "source, metadata",
    [
        (_SRC_WITH_DPR + "// Daily Progress Reporter\n", ""),
        (_SRC_WITH_DPR + "// XDaily Progress Report\n", ""),
        (_SRC_WITH_DPR, "glossary: Daily Progress Reporter"),
        (_SRC_WITH_DPR, "glossary: XDaily Progress Report"),
    ],
)
def test_phrase_inside_a_longer_evidence_token_is_not_verbatim_support(
    source, metadata
):
    cleaned = {"description": "The DPR is a Daily Progress Report."}

    out, removed = validate_narrative_terminology(
        dict(cleaned), _ev(source=source, metadata=metadata)
    )

    assert "description" not in out
    assert [r.field for r in removed] == ["description"]


def test_exact_verbatim_phrase_next_to_punctuation_is_support():
    src = _SRC_WITH_DPR + "// Daily Progress Report.\n"
    cleaned = {"description": "The DPR is a Daily Progress Report."}

    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))

    assert out == cleaned and removed == []


# ---------------------------------------------------------------------------
# P2-3: bounded candidate-subspan enumeration inside a longer Title-Case run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "The Daily Progress Report describes DPR",           # leading extra word
        "Daily Progress Report View for DPR",                # trailing extra word
        "See the Daily Progress Report Summary Page for DPR",  # both sides
    ],
)
def test_qualifying_subphrase_inside_a_longer_title_case_run_is_rejected(description):
    cleaned = {"description": description}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert "description" not in out
    assert [r.field for r in removed] == ["description"]


def test_connector_between_significant_words_is_covered_by_the_enumerator():
    # `MVC` in source; the candidate spans a lowercase connector that does not
    # contribute an initial.
    src = "const MVC = 1;\n"
    cleaned = {"description": "Implements the Model of View Controller pattern."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert "description" not in out
    assert [r.field for r in removed] == ["description"]


def test_connector_subphrase_retained_when_verbatim_in_source():
    src = "const MVC = 1;\n// the Model of View Controller layering\n"
    cleaned = {"description": "Implements the Model of View Controller pattern."}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev(source=src))
    assert out == cleaned and removed == []


def test_longer_title_case_run_with_no_matching_subspan_is_untouched():
    # No contiguous 2-6 significant-word subspan has initials equal to `DPR`.
    cleaned = {"description": "The Global Weekly Metrics Overview relates to DPR"}
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out == cleaned and removed == []


def test_enumeration_is_bounded_for_a_long_capitalized_run():
    run = " ".join(["Word"] * 400)
    cleaned = {"description": f"{run} and DPR"}
    # Must finish (bounded work) and, with no DPR-initial subspan, retain.
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert out == cleaned and removed == []


def test_no_source_evidence_is_a_no_op():
    cleaned = {"description": _UNSUPPORTED}
    out, removed = validate_narrative_terminology(dict(cleaned), TerminologyEvidence())
    assert out == cleaned and removed == []
    out2, removed2 = validate_narrative_terminology(dict(cleaned), None)
    assert out2 == cleaned and removed2 == []


def test_non_narrative_fields_are_never_scanned():
    cleaned = {
        "description": "ok",
        "functions": [{"name": "Daily Progress Report"}],   # a name, not narrative
        "exports": ["Daily Progress Report"],
        "imports": ["Daily Progress Report"],
        "dependencies_analysis": {"external": ["Daily Progress Report"]},
    }
    out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    assert removed == []
    assert out["functions"] == [{"name": "Daily Progress Report"}]
    assert out["exports"] == ["Daily Progress Report"]


# ---------------------------------------------------------------------------
# diagnostics / logs are bounded and value-safe
# ---------------------------------------------------------------------------


def test_removal_detail_is_value_safe():
    cleaned = {"description": "ok", "role_in_system": _UNSUPPORTED}
    _out, removed = validate_narrative_terminology(dict(cleaned), _ev())
    blob = repr([(r.field, r.reason_code, r.detail) for r in removed])
    assert "Daily Progress Report" not in blob
    assert "DPR" not in blob
    assert _SRC_WITH_DPR not in blob


# ---------------------------------------------------------------------------
# delivery through the canonical response contract + correction
# ---------------------------------------------------------------------------


def _run(raw, *, evidence, resolved_shape=None):
    return process_response(
        raw,
        mode="single",
        agent="combined",
        file_path="m.py",
        clean_reporter=__import__(
            "codedoc.agents.response_cleaning", fromlist=["clean_combined_report"]
        ).clean_combined_report,
        resolved_shape=resolved_shape,
        terminology_evidence=evidence,
    )


def test_process_response_rejects_a_required_field_with_unsupported_expansion():
    raw = json.dumps({"description": _UNSUPPORTED, "role_in_system": "r"})
    with pytest.raises(ResponseContractError) as caught:
        _run(raw, evidence=_ev())
    diag = caught.value.diagnostic
    assert diag.reason_code == "missing_required"
    assert any(r.reason_code == "unsupported_terminology" for r in diag.removed)
    # no leak in the bounded diagnostic
    blob = json.dumps(diag.as_summary())
    assert "Daily Progress Report" not in blob and "DPR" not in blob


def test_process_response_keeps_valid_response_when_expansion_is_supported():
    raw = json.dumps({"description": _UNSUPPORTED})
    out = _run(raw, evidence=_ev(source=_SUPPORTED_SRC))
    assert out["description"] == _UNSUPPORTED


class _TermProvider:
    """Initial combined response invents a DPR expansion in the required
    ``description``; the one correction response is clean."""

    provider_name = "fake"

    def __init__(self, corrected_still_bad=False):
        self.corrected_still_bad = corrected_still_bad
        self.correction_calls = 0

    def complete_json(self, prompt, system=""):
        if "Analyse the imports" in prompt:
            return json.dumps({"dependencies_analysis": {"internal": [], "external": []}})
        if "Previous response (verbatim" in prompt:
            self.correction_calls += 1
            if self.corrected_still_bad:
                return json.dumps({"description": "still the DPR (Daily Progress Report)"})
            return json.dumps({"description": "renders the DPR value"})
        return json.dumps({"description": "shows the DPR (Daily Progress Report)"})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _orch(provider, *, enabled=True):
    return Orchestrator(
        provider,
        parallel=False,
        analysis_mode="single",
        correction_ledger=CorrectionLedger(enabled),
        response_correction_enabled=enabled,
    )


def test_invalid_initial_then_valid_correction_publishes_corrected_value(tmp_path):
    provider = _TermProvider()
    orch = _orch(provider, enabled=True)
    request = make_execution_request(tmp_path, "m.tsx", _SRC_WITH_DPR, language="tsx")
    result = orch.process(request)
    assert provider.correction_calls == 1
    assert result["description"] == "renders the DPR value"
    assert "Daily Progress Report" not in json.dumps(result)


def test_correction_disabled_fails_closed_with_zero_correction_calls(tmp_path):
    provider = _TermProvider()
    orch = _orch(provider, enabled=False)
    request = make_execution_request(tmp_path, "m.tsx", _SRC_WITH_DPR, language="tsx")
    result = orch.process(request)
    assert provider.correction_calls == 0
    assert result["documentation"].get("error")
    assert result["functions"] == [] and result["classes"] == []


def test_second_invalid_response_fails_after_exactly_one_correction(tmp_path):
    provider = _TermProvider(corrected_still_bad=True)
    orch = _orch(provider, enabled=True)
    request = make_execution_request(tmp_path, "m.tsx", _SRC_WITH_DPR, language="tsx")
    result = orch.process(request)
    assert provider.correction_calls == 1
    assert result["documentation"].get("error")
    assert result["documentation"].get("response_contract_correction_attempted") is True


def test_dry_run_makes_no_provider_call(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "m.tsx").write_text(_SRC_WITH_DPR, encoding="utf-8")

    def _forbidden(_cfg):
        raise AssertionError("dry-run constructed a provider")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _forbidden)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "m.tsx", "documentation_scope": "entry", "dry_run": True},
    )
    assert stats.get("would_call_llm_for", stats.get("would_process", 0)) >= 0
