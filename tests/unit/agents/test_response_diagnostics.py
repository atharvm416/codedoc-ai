"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import logging
import pytest
from codedoc.agents.base_agent import BaseAgent
from codedoc.agents.response_cleaning import (
    clean_combined_report,
    clean_dependency_report,
    clean_leaf_capsule_report,
    clean_structure_report,
)
from codedoc.agents.response_diagnostics import (
    MAX_DETAIL_CHARS,
    MAX_DIAGNOSTIC_KEYS,
    MAX_PARSE_ERROR_CHARS,
    MAX_PATH_CHARS,
    MAX_REMOVAL_ENTRIES,
    ResponseDiagnostic,
    extract_json_candidate,
    process_fixed_capsule_response,
    process_response,
    required_field_paths,
)
from codedoc.core.file_division import (
    MAX_LEAF_SYMBOL_ITEMS_PER_KIND,
    MAX_LEAF_SYMBOL_NAME_CHARS,
)
from codedoc.utils.errors import AgentError, ResponseContractError

def _reasons(removed):
    return {(r.field, r.reason_code) for r in removed}

def _run_combined(raw):
    return process_response(
        raw, mode="single", agent="combined", file_path="m.py",
        clean_reporter=clean_combined_report, resolved_shape=None,
    )

def test_missing_braces_is_no_json_object():
    assert extract_json_candidate("no json here").kind == "no_json_object"

def test_non_object_top_level_is_top_level_not_object():
    assert extract_json_candidate("[1, 2, 3]").kind == "top_level_not_object"

def test_malformed_braced_candidate_is_json_parse_error():
    outcome = extract_json_candidate('{"a": }')
    assert outcome.kind == "json_parse_error"
    assert isinstance(outcome.parse_position, int)
    assert len(outcome.parse_error) <= MAX_PARSE_ERROR_CHARS

def test_valid_object_extracted_with_surrounding_prose():
    outcome = extract_json_candidate('here: {"a": 1} done')
    assert outcome.kind == "object"
    assert outcome.value == {"a": 1}

def test_malformed_json_stage_and_reason():
    with pytest.raises(ResponseContractError) as caught:
        _run_combined('{"description": }')
    diag = caught.value.diagnostic
    assert diag.stage == "json_parse"
    assert diag.reason_code == "json_parse_error"
    assert isinstance(diag.parse_position, int)

def test_no_json_object_reason():
    with pytest.raises(ResponseContractError) as caught:
        _run_combined("totally not json")
    assert caught.value.diagnostic.reason_code == "no_json_object"

def test_top_level_not_object_reason():
    with pytest.raises(ResponseContractError) as caught:
        _run_combined("[1, 2, 3]")
    assert caught.value.diagnostic.reason_code == "top_level_not_object"

def test_no_usable_fields_reason_for_structure():
    with pytest.raises(ResponseContractError) as caught:
        process_response(
            json.dumps({"functions": "not-a-list"}),
            mode="triple", agent="structure", file_path="m.py",
            clean_reporter=clean_structure_report, resolved_shape=None,
        )
    assert caught.value.diagnostic.reason_code == "no_usable_fields"
    assert caught.value.diagnostic.stage == "profile_filter"

def test_explicit_empty_optional_lists_are_valid_structure_response():
    out = process_response(
        json.dumps({"functions": [], "classes": [], "exports": []}),
        mode="triple", agent="structure", file_path="m.py",
        clean_reporter=clean_structure_report, resolved_shape=None,
    )
    assert out == {}

def test_explicit_zero_dependencies_is_valid_response():
    out = process_response(
        json.dumps({
            "dependencies_analysis": {
                "internal": [],
                "external": [],
                "dependency_refs": [],
                "catalog_updates": [],
                "usage_notes": [],
                "warnings": [],
            }
        }),
        mode="triple", agent="dependency", file_path="m.py",
        clean_reporter=clean_dependency_report, resolved_shape=None,
    )
    assert out == {}

def test_missing_required_reason_for_combined():
    with pytest.raises(ResponseContractError) as caught:
        _run_combined(json.dumps({"role_in_system": "r"}))
    diag = caught.value.diagnostic
    assert diag.reason_code == "missing_required"
    assert diag.stage == "required_fields"
    # description is the only registry-required field for single/combined.
    assert "description" in required_field_paths("single", "combined")

def test_precedence_parse_error_beats_field_reasons():
    # A malformed candidate never reaches cleaning, so json_parse_error wins.
    with pytest.raises(ResponseContractError) as caught:
        _run_combined('{"description": 1,,}')
    assert caught.value.diagnostic.reason_code == "json_parse_error"

def test_unknown_wrong_type_empty_duplicate_invalid_reported():
    result = clean_combined_report(
        {
            "description": "ok",
            "bogus": "x",                        # unknown_field
            "role_in_system": 123,               # wrong_type
            "usage_example": "   ",              # empty_value
            "exports": ["a", "a"],               # duplicate
            "dependencies_analysis": {
                "catalog_updates": [
                    {"name": "n", "type": "bad", "used_for": "u"}  # invalid_value
                ]
            },
        },
        "m.py",
    )
    reasons = _reasons(result.removed)
    assert ("bogus", "unknown_field") in reasons
    assert ("role_in_system", "wrong_type") in reasons
    assert ("usage_example", "empty_value") in reasons
    assert ("exports[1]", "duplicate") in reasons
    assert (
        "dependencies_analysis.catalog_updates[0].type",
        "invalid_value",
    ) in reasons

def test_item_limit_reported_for_overflow():
    functions = [{"name": f"f{i}"} for i in range(20)]
    result = clean_combined_report({"description": "d", "functions": functions}, "m.py")
    assert any(r.reason_code == "item_limit" for r in result.removed)


def _run_leaf_capsule(raw):
    return process_fixed_capsule_response(
        json.dumps(raw),
        label="split-leaf",
        agent="leaf",
        file_path="m.py",
        clean_reporter=clean_leaf_capsule_report,
        requested_paths=("description", "functions", "classes", "exports"),
        required_paths=("description",),
    )


def test_fixed_leaf_capsule_preserves_every_fact_at_declared_bounds():
    functions = [
        {"name": f"f{index}".ljust(MAX_LEAF_SYMBOL_NAME_CHARS, "x")}
        for index in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
    ]

    result = _run_leaf_capsule(
        {"description": "visible facts", "functions": functions}
    )

    assert result["functions"] == functions


@pytest.mark.parametrize(
    "functions",
    [
        [{"name": "x" * (MAX_LEAF_SYMBOL_NAME_CHARS + 1)}],
        [
            {"name": f"function_{index}"}
            for index in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND + 1)
        ],
    ],
)
def test_fixed_leaf_capsule_rejects_lossy_fact_caps(functions):
    with pytest.raises(ResponseContractError) as caught:
        _run_leaf_capsule(
            {"description": "visible facts", "functions": functions}
        )

    diagnostic = caught.value.diagnostic
    assert diagnostic.stage == "clean"
    assert diagnostic.reason_code == "fixed_cap_exceeded"
    assert any(
        removal.reason_code in {"response_cap", "item_limit"}
        for removal in diagnostic.removed
    )


def test_fixed_leaf_losslessness_survives_bounded_diagnostic_overflow():
    raw = {"description": "visible facts"}
    raw.update({f"unknown_{index}": "x" for index in range(80)})
    raw["functions"] = [
        {"name": "x" * (MAX_LEAF_SYMBOL_NAME_CHARS + 1)}
    ]

    with pytest.raises(ResponseContractError) as caught:
        _run_leaf_capsule(raw)

    diagnostic = caught.value.diagnostic
    assert diagnostic.reason_code == "fixed_cap_exceeded"
    assert len(diagnostic.removed) <= MAX_REMOVAL_ENTRIES


def test_many_unknown_fixed_leaf_fields_do_not_fake_a_fact_cap_failure():
    raw = {"description": "visible facts"}
    raw.update({f"unknown_{index}": "x" for index in range(80)})

    assert _run_leaf_capsule(raw) == {"description": "visible facts"}

def test_nested_unknown_symbol_and_object_fields_are_reported():
    result = clean_combined_report(
        {
            "description": "d",
            "functions": [{"name": "f", "unexpected": "x"}],
            "dependencies_analysis": {
                "catalog_updates": [
                    {
                        "name": "pkg",
                        "type": "external",
                        "used_for": "work",
                        "unexpected": "x",
                    }
                ]
            },
        },
        "m.py",
    )
    reasons = _reasons(result.removed)
    assert ("functions[0].unexpected", "unknown_field") in reasons
    assert (
        "dependencies_analysis.catalog_updates[0].unexpected",
        "unknown_field",
    ) in reasons

def test_scalar_truncation_is_not_a_removal():
    result = clean_combined_report({"description": "x" * 5000}, "m.py")
    assert len(result.value["description"]) == 1200
    assert not any(r.field == "description" for r in result.removed)

def test_response_cap_removal_reported():
    # Unique items (distinct prefixes) so nothing is deduped; the per-field-capped
    # payload exceeds the global combined cap and the global trim reports response_cap.
    raw = {
        "description": "keep",
        "dependencies_analysis": {
            "warnings": [f"warn-{i}-" + "w" * 250 for i in range(32)],
            "internal": [f"int-{i}-" + "i" * 250 for i in range(32)],
            "external": [f"ext-{i}-" + "e" * 250 for i in range(32)],
        },
        "key_concepts": [f"kc-{i}-" + "c" * 250 for i in range(16)],
        "functions": [{"name": f"f{i}", "description": "d" * 250} for i in range(12)],
    }
    result = clean_combined_report(raw, "m.py")
    assert any(r.reason_code == "response_cap" for r in result.removed)

def test_profile_filter_removals_are_not_requested():
    # A profile requesting only description drops role_in_system as not_requested.
    from codedoc.core.prompt_profiles import ResolvedProfile, validate_profile

    raw = {"single": {"common": {"requested_shape": {"description": "Only this."}}}}
    profile = validate_profile(
        raw, active_mode="single", known_extensions=frozenset({".py"}),
        source="inline", source_path=None,
    )
    resolved = ResolvedProfile("single", profile)
    block = resolved.resolve_block("combined", "m.py")
    out = process_response(
        json.dumps({"description": "d", "role_in_system": "r"}),
        mode="single", agent="combined", file_path="m.py",
        clean_reporter=clean_combined_report, resolved_shape=block,
    )
    assert out == {"description": "d"}

def test_removal_and_key_collections_obey_caps():
    # Many unknown fields -> removal list capped with an omitted marker.
    raw = {"description": "d"}
    for i in range(MAX_REMOVAL_ENTRIES + 50):
        raw[f"unknown_{i}"] = "x"
    result = clean_combined_report(raw, "m.py")
    assert len(result.removed) <= MAX_REMOVAL_ENTRIES
    assert result.removed[-1].field == "..."

def test_diagnostic_key_and_detail_caps():
    long_field = "a" * (MAX_PATH_CHARS + 50)
    result = clean_combined_report({"description": "d", long_field: "x"}, "m.py")
    for r in result.removed:
        assert len(r.field) <= MAX_PATH_CHARS
        assert len(r.detail) <= MAX_DETAIL_CHARS

def test_diagnostic_carries_no_raw_response_text():
    secret = "SECRET_TOKEN_abcdef123456"
    with pytest.raises(ResponseContractError) as caught:
        _run_combined(json.dumps({"role_in_system": secret}))
    summary = json.dumps(caught.value.diagnostic.as_summary())
    assert secret not in summary

def test_legacy_parse_helper_does_not_attach_raw_response_text():
    secret = "SECRET_TOKEN_abcdef123456"

    class Agent(BaseAgent):
        agent_name = "Agent"

        def run(self, file_path, content, imports, language):
            return {}

    agent = Agent(object())
    with pytest.raises(AgentError) as caught:
        agent._parse_json(f"not-json-{secret}", "m.py")
    assert secret not in str(caught.value)

def test_verbose_diagnostic_includes_bounded_returned_types(caplog):
    with caplog.at_level(logging.DEBUG, logger="codedoc.agents.response_diagnostics"):
        with pytest.raises(ResponseContractError):
            _run_combined(json.dumps({"role_in_system": "r"}))
    assert "returned_types=('role_in_system:str[1]',)" in caplog.text

def test_declared_key_cap_is_the_single_source():
    # A directly constructed diagnostic accepts empty key tuples; the hard cap is
    # applied by process_response's helpers and pinned by this single constant.
    empty = ResponseDiagnostic(stage="x", reason_code="y", agent="a", file_path="f")
    assert empty.expected_keys == ()
    assert MAX_DIAGNOSTIC_KEYS == 32

    raw = {f"unknown_{i}": "x" for i in range(MAX_DIAGNOSTIC_KEYS + 20)}
    with pytest.raises(ResponseContractError) as caught:
        _run_combined(json.dumps(raw))
    assert len(caught.value.diagnostic.returned_keys) <= MAX_DIAGNOSTIC_KEYS
    assert caught.value.diagnostic.returned_keys[-1] == "...(truncated)"
