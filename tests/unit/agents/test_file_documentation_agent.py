"""Tests organized by feature ownership."""

from __future__ import annotations

import dataclasses
import json
import pytest
from codedoc.agents import file_documentation_agent as fda
from codedoc.agents.file_documentation_agent import (
    MAX_COMBINED_RESPONSE_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_EXPORT_ITEM_CHARS,
    MAX_EXPORT_ITEMS,
    MAX_KEY_CONCEPT_ITEMS,
    MAX_SYMBOL_ITEMS_PER_KIND,
    FileDocumentationAgent,
    build_fragment_prompt,
    build_prompt,
    clean_combined_response,
)
from codedoc.core.execution_model import UnitChunkExecutionRequest
from codedoc.core.file_division import (
    MAX_LEAF_EXPORT_ITEM_CHARS,
    MAX_LEAF_EXPORT_ITEMS,
    MAX_LEAF_PROMPT_METADATA_CHARS,
    MAX_LEAF_SYMBOL_ITEMS_PER_KIND,
    MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
    build_division_plan,
    render_leaf_prompt_metadata,
)
from codedoc.utils.errors import AgentError, LLMError, ResponseContractError, find_provider_failure
from tests.support.execution_requests import make_execution_request
from tests.support.provider_failures import provider_failure_error

class _Provider:
    """Fake provider returning a fixed raw JSON string."""

    provider_name = "fake"

    def __init__(self, raw: str):
        self._raw = raw
        self.calls = 0

    def complete_json(self, prompt, system=""):
        self.calls += 1
        return self._raw

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


class _CorrectingProvider:
    """Returns *first_response* on the first call.

    On the second call, either returns *corrected_response* or raises
    *corrected_error* (mutually exclusive) -- simulating a correction call
    that itself fails with a provider fault.
    """

    provider_name = "fake"

    def __init__(
        self,
        *,
        first_response: dict,
        corrected_response: dict | None = None,
        corrected_error: Exception | None = None,
    ) -> None:
        assert (corrected_response is None) != (corrected_error is None), (
            "exactly one of corrected_response/corrected_error must be set"
        )
        self.calls = 0
        self.first_response = first_response
        self.corrected_response = corrected_response
        self.corrected_error = corrected_error

    def complete_json(self, prompt, system=""):
        self.calls += 1
        if self.calls == 1:
            return json.dumps(self.first_response)
        if self.corrected_error is not None:
            raise self.corrected_error
        return json.dumps(self.corrected_response)

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


class _RecordingCorrectingProvider(_CorrectingProvider):
    """`_CorrectingProvider` that also keeps every prompt it was sent.

    The 0.14.6 correction-route assertions must inspect the prompt actually
    built and sent by the real `ResponseCorrectionAgent`, not a prompt the
    test reconstructs itself.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompts: list[str] = []

    def complete_json(self, prompt, system=""):
        self.prompts.append(prompt)
        return super().complete_json(prompt, system)


def _real_correction(provider):
    """A live, enabled ``ResponseCorrectionAgent`` sharing *provider*."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    return ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )

_LANG_FIXTURES = {
    "python": {
        "source": "def main():\n    return 0\n",
        "functions": [{"name": "main", "description": "Returns the exit status."}],
        "classes": [],
        "exports": ["main"],
        "usage": "from file import main",
    },
    "tsx": {
        "source": "export const App = () => <main />;\n",
        "functions": [{"name": "App", "description": "Renders the application."}],
        "classes": [],
        "exports": ["App"],
        "usage": "import { App } from './file'",
    },
    "dart": {
        "source": "void main() {}\n",
        "functions": [{"name": "main", "description": "Starts the application."}],
        "classes": [],
        "exports": ["main"],
        "usage": "import 'file.dart';",
    },
    "java": {
        "source": "class Main {}\n",
        "functions": [],
        "classes": [{"name": "Main", "description": "Application type."}],
        "exports": ["Main"],
        "usage": "new Main()",
    },
}

def test_build_prompt_includes_path_imports_and_truncated_content():
    system, prompt = build_prompt("pkg/mod.py", "CODEBODY", ["os", "sys"], "python")
    assert "JSON" in system
    assert "File: pkg/mod.py" in prompt
    assert "['os', 'sys']" in prompt
    assert "CODEBODY" in prompt
    assert "python" in prompt

def test_agent_truncates_oversized_content_once():
    agent = FileDocumentationAgent(
        _Provider(json.dumps({"description": "ok"})), max_content_chars=1000
    )
    # The agent's defensive _truncate keeps content within the ceiling.
    result = agent.run("m.py", "x" * 5000, [], "python")
    assert result["description"] == "ok"

def test_exactly_one_call_per_run():
    provider = _Provider(json.dumps({"description": "ok"}))
    FileDocumentationAgent(provider).run("m.py", "code", [], "python")
    assert provider.calls == 1

def test_unknown_keys_removed():
    cleaned = clean_combined_response(
        {"description": "d", "bogus": "drop me", "_smuggled": "x"}, "m.py"
    )
    assert cleaned == {"description": "d"}

def test_empty_and_null_values_removed():
    cleaned = clean_combined_response(
        {
            "description": "  keep  ",
            "role_in_system": "   ",
            "functions": [],
            "exports": [],
            "usage_example": None,
            "dependencies_analysis": {},
        },
        "m.py",
    )
    assert cleaned == {"description": "keep"}

def test_booleans_never_accepted_as_strings():
    with pytest.raises(AgentError):
        clean_combined_response(
            {"description": True, "role_in_system": False}, "m.py"
        )

def test_strings_are_trimmed():
    cleaned = clean_combined_response({"description": "  spaced  "}, "m.py")
    assert cleaned["description"] == "spaced"

def test_list_dedup_is_case_sensitive_and_order_preserving():
    cleaned = clean_combined_response(
        {"description": "d", "exports": ["A", "a", "A", "b"]}, "m.py"
    )
    assert cleaned["exports"] == ["A", "a", "b"]

def test_malformed_symbol_items_removed_and_name_required():
    cleaned = clean_combined_response(
        {
            "description": "d",
            "functions": [
                {"name": "ok", "description": "does"},
                {"description": "no name"},
                "not an object",
                {"name": "   "},
            ],
        },
        "m.py",
    )
    assert cleaned["functions"] == [{"name": "ok", "description": "does"}]

def test_symbols_deduped_by_canonical_form():
    cleaned = clean_combined_response(
        {
            "description": "d",
            "classes": [
                {"name": "C", "description": "x"},
                {"name": "C", "description": "x"},
                {"name": "C", "description": "y"},
            ],
        },
        "m.py",
    )
    assert cleaned["classes"] == [
        {"name": "C", "description": "x"},
        {"name": "C", "description": "y"},
    ]

def test_catalog_update_requires_all_fields_and_valid_type():
    cleaned = clean_combined_response(
        {
            "description": "d",
            "dependencies_analysis": {
                "catalog_updates": [
                    {"name": "requests", "type": "external", "used_for": "HTTP"},
                    {"name": "x", "type": "bogus", "used_for": "y"},
                    {"name": "x", "used_for": "missing type"},
                    {"name": "x", "type": "internal"},
                ]
            },
        },
        "m.py",
    )
    assert cleaned["dependencies_analysis"]["catalog_updates"] == [
        {"name": "requests", "type": "external", "used_for": "HTTP"}
    ]

def test_usage_note_requires_import_and_used_for():
    cleaned = clean_combined_response(
        {
            "description": "d",
            "dependencies_analysis": {
                "usage_notes": [
                    {"import": "os", "used_for": "paths"},
                    {"import": "os"},
                    {"used_for": "x"},
                ]
            },
        },
        "m.py",
    )
    assert cleaned["dependencies_analysis"]["usage_notes"] == [
        {"import": "os", "used_for": "paths"}
    ]

def test_dependency_objects_deduped_by_canonical_json():
    cleaned = clean_combined_response(
        {
            "description": "d",
            "dependencies_analysis": {
                "catalog_updates": [
                    {"name": "a", "type": "external", "used_for": "u"},
                    {"used_for": "u", "type": "external", "name": "a"},
                ]
            },
        },
        "m.py",
    )
    assert len(cleaned["dependencies_analysis"]["catalog_updates"]) == 1

def test_symbol_count_capped():
    funcs = [{"name": f"f{i}"} for i in range(50)]
    cleaned = clean_combined_response({"description": "d", "functions": funcs}, "m.py")
    assert len(cleaned["functions"]) == MAX_SYMBOL_ITEMS_PER_KIND

def test_export_count_capped():
    cleaned = clean_combined_response(
        {"description": "d", "exports": [f"e{i}" for i in range(100)]}, "m.py"
    )
    assert len(cleaned["exports"]) == MAX_EXPORT_ITEMS


def test_export_item_length_uses_public_export_bound():
    raw_export = "e" * (MAX_EXPORT_ITEM_CHARS + 50)
    cleaned = clean_combined_response(
        {"description": "d", "exports": [raw_export]}, "m.py"
    )

    assert cleaned["exports"] == [raw_export[:MAX_EXPORT_ITEM_CHARS]]
    assert MAX_EXPORT_ITEM_CHARS > fda.MAX_SYMBOL_NAME_CHARS


def test_key_concepts_capped():
    cleaned = clean_combined_response(
        {"description": "d", "key_concepts": [f"c{i}" for i in range(100)]}, "m.py"
    )
    assert len(cleaned["key_concepts"]) == MAX_KEY_CONCEPT_ITEMS

def test_description_length_capped():
    cleaned = clean_combined_response({"description": "x" * 5000}, "m.py")
    assert len(cleaned["description"]) == MAX_DESCRIPTION_CHARS

@pytest.mark.parametrize(
    ("field", "cap"),
    [
        ("description", fda.MAX_DESCRIPTION_CHARS),
        ("role_in_system", fda.MAX_ROLE_CHARS),
        ("usage_example", fda.MAX_USAGE_EXAMPLE_CHARS),
    ],
)
def test_every_scalar_length_bound(field, cap):
    cleaned = clean_combined_response({field: "x" * (cap + 100)}, "m.py")
    assert len(cleaned[field]) == cap

def test_every_symbol_length_bound():
    cleaned = clean_combined_response(
        {
            "functions": [
                {
                    "name": "n" * (fda.MAX_SYMBOL_NAME_CHARS + 10),
                    "description": "d" * (fda.MAX_SYMBOL_DESCRIPTION_CHARS + 10),
                }
            ]
        },
        "m.py",
    )
    symbol = cleaned["functions"][0]
    assert len(symbol["name"]) == fda.MAX_SYMBOL_NAME_CHARS
    assert len(symbol["description"]) == fda.MAX_SYMBOL_DESCRIPTION_CHARS

@pytest.mark.parametrize(
    "field", ["internal", "external", "dependency_refs", "warnings"]
)
def test_every_dependency_string_list_bound(field):
    values = [f"{i}-" + "x" * fda.MAX_LIST_ITEM_CHARS for i in range(100)]
    cleaned = clean_combined_response(
        {"dependencies_analysis": {field: values}}, "m.py"
    )
    result = cleaned["dependencies_analysis"][field]
    assert len(result) == fda.MAX_DEPENDENCY_ITEMS_PER_LIST
    assert all(len(item) == fda.MAX_LIST_ITEM_CHARS for item in result)

@pytest.mark.parametrize(
    ("field", "max_items", "name_field"),
    [
        ("catalog_updates", fda.MAX_CATALOG_UPDATE_ITEMS, "name"),
        ("usage_notes", fda.MAX_USAGE_NOTE_ITEMS, "import"),
    ],
)
def test_every_dependency_object_bound(field, max_items, name_field):
    values = []
    for index in range(max_items + 5):
        item = {
            name_field: f"{index}-" + "n" * fda.MAX_DEPENDENCY_NAME_CHARS,
            "used_for": "u" * (fda.MAX_DEPENDENCY_PURPOSE_CHARS + 10),
        }
        if field == "catalog_updates":
            item["type"] = "external"
        values.append(item)
    cleaned = clean_combined_response(
        {"dependencies_analysis": {field: values}}, "m.py"
    )
    result = cleaned["dependencies_analysis"][field]
    assert len(result) == max_items
    assert all(len(item[name_field]) == fda.MAX_DEPENDENCY_NAME_CHARS for item in result)
    assert all(
        len(item["used_for"]) == fda.MAX_DEPENDENCY_PURPOSE_CHARS for item in result
    )

def test_key_concept_length_bound():
    cleaned = clean_combined_response(
        {"key_concepts": ["x" * (fda.MAX_KEY_CONCEPT_CHARS + 10)]}, "m.py"
    )
    assert len(cleaned["key_concepts"][0]) == fda.MAX_KEY_CONCEPT_CHARS

def test_global_cap_trims_lowest_priority_first():
    raw = {
        "description": "keep the description",
        "functions": [{"name": f"f{i}", "description": "d" * 200} for i in range(12)],
        "key_concepts": ["c" * 250 for _ in range(16)],
        "dependencies_analysis": {
            "warnings": ["w" * 250 for _ in range(32)],
        },
    }
    cleaned = clean_combined_response(raw, "m.py")
    serialized = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    assert len(serialized) <= MAX_COMBINED_RESPONSE_CHARS
    # description (highest priority scalar) and functions survive; warnings
    # (lowest priority list) are trimmed first.
    assert "description" in cleaned
    deps = cleaned.get("dependencies_analysis", {})
    assert len(deps.get("warnings", [])) <= 32

def test_global_cap_truncates_scalar_prose_when_needed():
    raw = {
        "description": "d" * MAX_DESCRIPTION_CHARS,
        "role_in_system": "r" * 800,
        "usage_example": "u" * 2000,
    }
    cleaned = clean_combined_response(raw, "m.py")
    serialized = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    assert len(serialized) <= MAX_COMBINED_RESPONSE_CHARS

def test_non_object_response_raises():
    with pytest.raises(AgentError):
        clean_combined_response(["not", "an", "object"], "m.py")

def test_no_usable_field_raises():
    with pytest.raises(AgentError):
        clean_combined_response({"bogus": "x", "functions": []}, "m.py")

def test_partly_malformed_with_one_usable_field_succeeds():
    cleaned = clean_combined_response(
        {"description": "good", "functions": ["broken"], "exports": []}, "m.py"
    )
    assert cleaned == {"description": "good"}

def test_invalid_json_raises_via_run():
    agent = FileDocumentationAgent(_Provider("not json at all"))
    with pytest.raises(AgentError):
        agent.run("m.py", "code", [], "python")

@pytest.mark.parametrize("language", list(_LANG_FIXTURES))
def test_combined_response_preserves_required_facts_per_language(language):
    fixture = _LANG_FIXTURES[language]
    raw = json.dumps(
        {
            "description": f"A {language} file.",
            "role_in_system": "entry point",
            "functions": fixture["functions"],
            "classes": fixture["classes"],
            "exports": fixture["exports"],
            "key_concepts": ["startup"],
            "usage_example": fixture["usage"],
        }
    )
    agent = FileDocumentationAgent(_Provider(raw))
    cleaned = agent.run(f"src/file.{language}", fixture["source"], [], language)
    assert cleaned["description"] == f"A {language} file."
    assert cleaned["role_in_system"] == "entry point"
    assert cleaned.get("functions", []) == fixture["functions"]
    assert cleaned.get("classes", []) == fixture["classes"]
    assert cleaned["exports"] == fixture["exports"]
    assert cleaned["key_concepts"] == ["startup"]
    assert cleaned["usage_example"] == fixture["usage"]
    assert "dependencies_analysis" not in cleaned

def test_module_exposes_named_bounds_not_magic_numbers():
    expected = {
        "MAX_COMBINED_RESPONSE_CHARS": 12000,
        "MAX_DESCRIPTION_CHARS": 1200,
        "MAX_ROLE_CHARS": 800,
        "MAX_USAGE_EXAMPLE_CHARS": 2000,
        "MAX_SYMBOL_ITEMS_PER_KIND": 12,
        "MAX_SYMBOL_NAME_CHARS": 128,
        "MAX_SYMBOL_DESCRIPTION_CHARS": 300,
        "MAX_EXPORT_ITEMS": 32,
        "MAX_EXPORT_ITEM_CHARS": 256,
        "MAX_DEPENDENCY_ITEMS_PER_LIST": 32,
        "MAX_LIST_ITEM_CHARS": 256,
        "MAX_CATALOG_UPDATE_ITEMS": 16,
        "MAX_USAGE_NOTE_ITEMS": 16,
        "MAX_DEPENDENCY_NAME_CHARS": 128,
        "MAX_DEPENDENCY_PURPOSE_CHARS": 400,
        "MAX_KEY_CONCEPT_ITEMS": 16,
        "MAX_KEY_CONCEPT_CHARS": 300,
    }
    assert {name: getattr(fda, name) for name in expected} == expected


def _leaf_request(
    tmp_path, rel_path: str = "src/large.py", *, content: str | None = None,
    max_content_chars: int = 200, chunk_index: int = 0,
) -> UnitChunkExecutionRequest:
    """One real ``UnitChunkExecutionRequest`` built from an actual division
    plan chunk, exactly as ``codedoc.core.execution._process_divided_file``
    constructs it — never a hand-forged digest or payload."""
    source = content or (
        "\n".join(f"def fn_{i}():\n    return {i}" for i in range(20)) + "\n"
    )
    file_request = make_execution_request(
        tmp_path, rel_path, source, max_content_chars=max_content_chars
    )
    plan = build_division_plan(
        rel_path=rel_path,
        language="python",
        content=source,
        source_budget_chars=max_content_chars,
    )
    chunk = plan.chunks[chunk_index]
    return UnitChunkExecutionRequest(
        rel_path=rel_path,
        language="python",
        full_content_hash=file_request.content_hash,
        division_plan_digest=plan.plan_digest,
        chunk_id=chunk.chunk_id,
        unit_id=chunk.unit_id,
        semantic_units=chunk.semantic_units,
        unit_indexes=plan.unit_positions(chunk),
        unit_count=len(plan.units),
        unit_chunk_index=chunk.unit_chunk_index,
        unit_chunk_count=chunk.unit_chunk_count,
        global_index=chunk.global_index,
        global_count=chunk.global_count,
        owning_ranges=chunk.owning_ranges,
        continuation_before=chunk.continuation_before,
        continuation_after=chunk.continuation_after,
        known_symbols=chunk.known_symbols,
        payload=chunk.payload,
        context=file_request.context,
    )


def test_fragment_request_carries_no_whole_file_imports_field():
    """D5/section 7: a leaf request never carries separately parsed whole-file
    imports — only the visible fragment payload."""
    field_names = {f.name for f in dataclasses.fields(UnitChunkExecutionRequest)}
    assert "imports" not in field_names
    assert {"semantic_units", "unit_indexes", "owning_ranges"} <= field_names
    assert not {
        "unit_kind",
        "unit_qualified_name",
        "unit_signature",
        "unit_index",
    } & field_names


def test_fragment_prompt_contains_exact_position_metadata_and_fixed_shape(tmp_path):
    request = _leaf_request(tmp_path)

    system, prompt = build_fragment_prompt(request)

    assert "senior software engineer" in system
    assert "This is one bounded fragment of a larger" in prompt
    assert "It is NOT the" in prompt
    assert f"File: {request.rel_path}" in prompt
    assert (
        f"fragment {request.unit_chunk_index + 1} of {request.unit_chunk_count}"
        in prompt
    )
    assert f"overall fragment {request.global_index + 1} of {request.global_count}" in prompt
    assert f"Continues an earlier fragment of the same unit: {request.continuation_before}" in prompt
    assert f"Continues into a later fragment of the same unit: {request.continuation_after}" in prompt
    expected_metadata = render_leaf_prompt_metadata(
        group_unit_id=request.unit_id,
        semantic_units=request.semantic_units,
        unit_indexes=request.unit_indexes,
        unit_count=request.unit_count,
        owning_ranges=request.owning_ranges,
    )
    assert (
        "Fragment metadata (ordered JSON; semantic_unit_indexes are 1-based): "
        f"{expected_metadata}"
    ) in prompt
    assert len(expected_metadata) <= MAX_LEAF_PROMPT_METADATA_CHARS
    assert json.loads(expected_metadata)["leaf_call_group_id"] == request.unit_id
    assert (
        json.loads(expected_metadata)["semantic_unit_indexes"]
        == [index + 1 for index in request.unit_indexes]
    )
    assert [
        prompt.index(f'"unit_id":"{unit.unit_id}"')
        for unit in request.semantic_units
    ] == sorted(
        prompt.index(f'"unit_id":"{unit.unit_id}"')
        for unit in request.semantic_units
    )
    assert request.payload in prompt
    assert '"description": "<required' in prompt
    assert "Synthesize one final file-level" not in prompt


def test_fragment_prompt_advertises_the_shared_2000_character_signature_bound(tmp_path):
    """Section 2A / 0.14.7 section 5.4: the fixed fragment prompt's
    hard-bounds sentence must match `MAX_LEAF_SYMBOL_SIGNATURE_CHARS` (raised
    to 2,000, aliased from the parser's own ceiling) rather than the retired
    256-character literal or the pre-0.14.7 600-character bound, so a model
    can never accurately echo a signature CodeDoc itself supplied and still
    fail the whole leaf response."""
    assert MAX_LEAF_SYMBOL_SIGNATURE_CHARS == 2000
    request = _leaf_request(tmp_path)

    _system, prompt = build_fragment_prompt(request)

    assert "each symbol signature <= 2000 characters" in prompt
    assert "signature <= 256" not in prompt
    assert "signature <= 600 characters" not in prompt


def test_fragment_prompt_reports_true_continuation_flags(tmp_path):
    # A single line far larger than the budget forces continuation chunks.
    line = "x = " + "1" * 600 + "\n"
    request = _leaf_request(
        tmp_path, content=line, max_content_chars=200, chunk_index=1
    )

    assert request.continuation_before is True

    _system, prompt = build_fragment_prompt(request)

    assert "Continues an earlier fragment of the same unit: True" in prompt


def test_fragment_prompt_lists_known_symbols_when_present(tmp_path):
    # known_symbols is populated by the structural parser, which is only
    # available with the optional grammar extra; the fixed prompt template's
    # rendering of a populated value is exercised directly here instead.
    base = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    request = dataclasses.replace(base, known_symbols=("alpha", "alpha.helper"))

    _system, prompt = build_fragment_prompt(request)

    assert "alpha" in prompt
    assert "alpha.helper" in prompt


def test_run_fragment_uses_fixed_capsule_cleaner_and_rejects_final_shape_fields(
    tmp_path,
):
    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _Provider(
        json.dumps(
            {
                "description": "Alpha fragment.",
                "functions": [{"name": "alpha", "description": "Returns one."}],
                "role_in_system": "forged final-shape field",
                "usage_example": "forged final-shape field",
                "division": {"strategy": "forged"},
                "documentation_units": [{"unit_id": "forged"}],
                "arbitrary": "forged",
            }
        )
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)

    result = agent.run_fragment(request)

    assert provider.calls == 1
    assert result["description"] == "Alpha fragment."
    assert result["functions"] == [{"name": "alpha", "description": "Returns one."}]
    assert not {
        "role_in_system",
        "usage_example",
        "division",
        "documentation_units",
        "arbitrary",
    } & set(result)


def test_run_fragment_retry_reissues_a_byte_identical_prompt(tmp_path, monkeypatch):
    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _Provider(json.dumps({"description": "Alpha fragment."}))
    prompts: list[str] = []
    original = provider.complete_json

    def recording(prompt, system=""):
        prompts.append(prompt)
        return original(prompt, system)

    monkeypatch.setattr(provider, "complete_json", recording)
    agent = FileDocumentationAgent(provider, max_content_chars=1000)

    agent.run_fragment(request)
    agent.run_fragment(request, additional_attempt=True)

    assert provider.calls == 2
    assert prompts[0] == prompts[1]


def _over_cap_functions() -> list[dict]:
    """One more function than MAX_LEAF_SYMBOL_ITEMS_PER_KIND allows."""
    return [
        {"name": f"f{index}"} for index in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND + 1)
    ]


def test_run_fragment_at_cap_is_accepted(tmp_path) -> None:
    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    functions = [
        {"name": f"f{index}"} for index in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
    ]
    provider = _Provider(json.dumps({"description": "ok", "functions": functions}))
    agent = FileDocumentationAgent(provider, max_content_chars=1000)

    result = agent.run_fragment(request)

    assert result["functions"] == functions
    assert provider.calls == 1


def test_run_fragment_over_cap_is_rejected_without_correction(tmp_path) -> None:
    """With correction disabled (the FileDocumentationAgent default, no
    ``_correction`` attached), an over-cap leaf capsule is rejected in full
    with the closed REASON_FIXED_CAP_EXCEEDED/REMOVAL_ITEM_LIMIT codes and
    makes exactly one provider call -- no correction call."""
    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _Provider(
        json.dumps({"description": "ok", "functions": _over_cap_functions()})
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    assert agent._correction is None

    with pytest.raises(ResponseContractError) as caught:
        agent.run_fragment(request)

    assert caught.value.diagnostic.reason_code == "fixed_cap_exceeded"
    assert any(
        removal.reason_code == "item_limit"
        for removal in caught.value.diagnostic.removed
    )
    assert provider.calls == 1


def test_run_fragment_correction_valid_response_succeeds(tmp_path) -> None:
    """With correction enabled, a rejected over-cap capsule consumes exactly
    one correction call; a valid corrected response succeeds."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _CorrectingProvider(
        first_response={"description": "ok", "functions": _over_cap_functions()},
        corrected_response={"description": "Corrected leaf.", "functions": [{"name": "alpha"}]},
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )

    result = agent.run_fragment(request)

    assert result == {"description": "Corrected leaf.", "functions": [{"name": "alpha"}]}
    assert provider.calls == 2


def test_run_fragment_correction_still_invalid_fails_the_file(tmp_path) -> None:
    """With correction enabled, a corrected response that is itself still
    over the per-kind cap fails the file (not a second correction call)."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _CorrectingProvider(
        first_response={"description": "ok", "functions": _over_cap_functions()},
        corrected_response={"description": "still bad", "functions": _over_cap_functions()},
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )

    with pytest.raises(ResponseContractError) as caught:
        agent.run_fragment(request)

    assert caught.value.correction_attempted is True
    assert "still failed the schema contract" in str(caught.value)
    assert provider.calls == 2


def test_run_fragment_correction_nonterminal_fault_fails_the_file_without_retry(
    tmp_path,
) -> None:
    """With correction enabled, a correction call that fails with a
    nonterminal provider fault fails the file without a second correction
    call -- it is not converted into a retry."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _CorrectingProvider(
        first_response={"description": "ok", "functions": _over_cap_functions()},
        corrected_error=LLMError("test-provider", "temporary provider outage"),
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )

    with pytest.raises(ResponseContractError) as caught:
        agent.run_fragment(request)

    assert caught.value.correction_attempted is True
    assert "correction provider call failed" in str(caught.value)
    assert provider.calls == 2


def test_run_fragment_correction_terminal_fault_preserves_whole_run_abort(
    tmp_path,
) -> None:
    """With correction enabled, a correction call that fails with a
    terminal-billing provider fault is re-raised unchanged (as the AgentError
    the shared provider-call accounting path wraps it in, with the original
    LLMError as its cause) -- preserving the existing whole-run abort -- and
    is never converted into a per-file response-contract failure. Section 5.3
    forbids retaining raw provider text, so the structured envelope reachable
    via ``find_provider_failure`` is asserted instead of the original
    free-form message."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    request = _leaf_request(tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000)
    provider = _CorrectingProvider(
        first_response={"description": "ok", "functions": _over_cap_functions()},
        corrected_error=provider_failure_error(
            "openai", "provider-quota-exhausted", status=429
        ),
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )

    with pytest.raises(AgentError) as caught:
        agent.run_fragment(request)

    assert not isinstance(caught.value, ResponseContractError)
    failure = find_provider_failure(caught.value)
    assert failure is not None
    assert failure.reason_code == "provider-quota-exhausted"
    assert failure.status == 429
    assert isinstance(caught.value.__cause__, LLMError)
    assert provider.calls == 2


# ---------------------------------------------------------------------------
# 0.14.6: the shared split-leaf module-export contract
# ---------------------------------------------------------------------------
# The 0.14.5 failure was a split leaf that reported exported-value *interior*
# (array members, object keys, IDs) as module `exports`, blowing the fixed cap
# on both the initial leaf response and its one targeted correction. The repair
# is a semantic contract, and it must reach the correction route too -- that
# route receives only the shape block, never `_FRAGMENT_PROMPT_TEMPLATE`'s
# fragment-specific rules.

#: Every clause the shared contract must state, asserted as exact substrings so
#: a reworded-but-weakened contract fails instead of silently passing. Spelled
#: out here rather than imported from production, so these assertions cannot be
#: satisfied by the very string they exist to police.
_EXPORT_CONTRACT_CLAUSES = (
    "is a name this module or package exposes as part of its own "
    "language-level API",
    "Containment is not export",
    "not module exports merely because the value containing them is exported",
    "IS the module's own export declaration",
    "its entries are the exported names themselves and must be reported",
    "Judge an export by declaration visibility alone",
    "never by any continuation flag",
    "is never on its own evidence that a name is exported",
    'omit the optional "exports" key entirely',
)


def _at_cap_exports() -> list[str]:
    """Exactly MAX_LEAF_EXPORT_ITEMS distinct export names."""
    return [f"exportName{index}" for index in range(MAX_LEAF_EXPORT_ITEMS)]


def _over_cap_exports() -> list[str]:
    """One more export name than MAX_LEAF_EXPORT_ITEMS allows."""
    return [f"exportName{index}" for index in range(MAX_LEAF_EXPORT_ITEMS + 1)]


def _assert_states_the_export_contract_once(prompt: str) -> None:
    for clause in _EXPORT_CONTRACT_CLAUSES:
        assert prompt.count(clause) == 1, clause
    # The pre-existing hard bounds must survive the contract edit verbatim.
    assert f"exports <= {MAX_LEAF_EXPORT_ITEMS} items" in prompt
    assert f"each export <= {MAX_LEAF_EXPORT_ITEM_CHARS} characters" in prompt


def _interior_only_data_source(entries: int = 120) -> str:
    """A data-only module whose later lexical chunks show array interior only.

    Used where a fixture response claims "no exports": the claim has to be
    true of the fragment the model was actually shown. A fragment whose
    visible source declares a symbol, answered with "data-only fragment, no
    exports", satisfies the schema while contradicting the factuality contract
    this release exists to defend.
    """
    lines = ["ROWS = ["]
    lines.extend(f'    {{"id": "row_{index:03d}"}},' for index in range(entries))
    lines.append("]")
    return "\n".join(lines) + "\n"


def _corrected_leaf_agent(tmp_path):
    """One real leaf agent over an interior-only data fragment: the first
    response is an over-cap `exports` list, and the correction returns a
    truthful description-only capsule -- truthful because the fragment really
    does show nothing but array interior."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    request = _leaf_request(
        tmp_path,
        content=_interior_only_data_source(),
        max_content_chars=600,
        chunk_index=1,
    )
    assert "ROWS = [" not in request.payload, (
        "chunk 1 must show array interior only, with no visible declaration"
    )
    provider = _RecordingCorrectingProvider(
        first_response={"description": "ok", "exports": _over_cap_exports()},
        corrected_response={
            "description": "Entries of a larger data table; no declaration visible."
        },
    )
    agent = FileDocumentationAgent(provider, max_content_chars=600)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )
    return request, provider, agent


def test_fragment_prompt_states_the_module_export_contract_exactly_once(tmp_path):
    """0.14.6: the initial split-leaf prompt must define what an export is.

    Before this release `_FRAGMENT_SHAPE_BLOCK` described `exports` only as
    `"exports": ["...", ...] (optional)` plus its bounds, and
    `_FRAGMENT_PROMPT_TEMPLATE` defined `functions`/`classes` but never
    `exports` -- so a fragment showing only the interior of a large exported
    array was free to report its members as module exports."""
    request = _leaf_request(tmp_path)

    _system, prompt = build_fragment_prompt(request)

    _assert_states_the_export_contract_once(prompt)


def test_fragment_export_contract_is_language_neutral(tmp_path):
    """The reproduction was TypeScript, but the contract governs every
    supported language: it may name generic data shapes, never one language's
    keywords as the only reading."""
    request = _leaf_request(tmp_path)

    _system, prompt = build_fragment_prompt(request)

    assert fda._FRAGMENT_EXPORT_CONTRACT in prompt
    for language_specific in (
        "TypeScript", "JavaScript", "TSX", "Python", "__init__.py",
        "export const", "module.exports", "__all__",
    ):
        assert language_specific not in fda._FRAGMENT_EXPORT_CONTRACT


def test_initial_and_correction_leaf_prompts_carry_the_identical_shape_block(
    tmp_path,
):
    """No-drift guard: both routes must carry `_FRAGMENT_SHAPE_BLOCK` byte for
    byte. Two texts that each merely satisfy a substring check could still
    drift apart; the same string object in both prompts cannot."""
    request, provider, agent = _corrected_leaf_agent(tmp_path)

    agent.run_fragment(request)

    initial_prompt, correction_prompt = provider.prompts
    assert fda._FRAGMENT_SHAPE_BLOCK in initial_prompt
    assert fda._FRAGMENT_SHAPE_BLOCK in correction_prompt


def test_correction_prompt_states_the_same_module_export_contract_once(tmp_path):
    """0.14.6: the one targeted correction call must carry the same export
    contract as the initial call.

    `run_fragment()` hands `_FRAGMENT_SHAPE_BLOCK` -- and nothing else from
    `_FRAGMENT_PROMPT_TEMPLATE` -- to `_finalize_fixed_response()`, so a rule
    added only to the initial template would leave the correction route
    defective. This drives a real over-cap `exports` rejection through the
    real correction component and inspects the prompt actually sent."""
    request, provider, agent = _corrected_leaf_agent(tmp_path)

    # A corrected capsule carrying only a truthful non-empty description is
    # valid: `exports` stays optional and is never forced to an empty list.
    result = agent.run_fragment(request)

    assert result == {
        "description": "Entries of a larger data table; no declaration visible."
    }
    assert provider.calls == 2
    correction_prompt = provider.prompts[1]
    assert "Previous response (verbatim" in correction_prompt
    _assert_states_the_export_contract_once(correction_prompt)


def test_export_contract_is_self_contained_for_the_correction_route(tmp_path):
    """The correction prompt renders no fragment position, no continuation
    flags, and no known-symbol line -- for a fixed capsule it also passes an
    empty language and an empty import list. A contract clause whose condition
    named one of those lines would be unanchored on exactly the defective
    route, so every clause must resolve against the visible source alone."""
    request, provider, agent = _corrected_leaf_agent(tmp_path)

    agent.run_fragment(request)

    correction_prompt = provider.prompts[1]
    assert "Continues an earlier fragment" not in correction_prompt
    assert "Known symbol name(s)" not in correction_prompt
    assert "Fragment position:" not in correction_prompt
    assert fda._FRAGMENT_EXPORT_CONTRACT in correction_prompt
    # No clause may point at one of the initial prompt's own metadata lines.
    # "continuation flag" appears only as something the model must *ignore*,
    # which is exactly the 0.14.5 repair and stays correct with the flags
    # absent -- so the labels themselves are what must not be referenced.
    for initial_only_line in (
        "Fragment position:",
        "Fragment metadata",
        "Continues an earlier fragment",
        "Continues into a later fragment",
        "Known symbol name",
        "known_symbols",
    ):
        assert initial_only_line not in fda._FRAGMENT_EXPORT_CONTRACT


def test_module_export_contract_does_not_reach_unrelated_prompts():
    """Scope guard: the contract governs the fixed split-leaf capsule only. It
    must not leak into the reduction/final-synthesis capsule (which never
    carries exports at all) or into the whole-file `single` prompt, whose own
    export wording this release deliberately leaves unchanged."""
    from codedoc.agents.file_synthesis_agent import _REDUCTION_SHAPE_BLOCK

    _system, whole_file_prompt = build_prompt(
        "src/example.py", "x = 1\n", ["os"], "python"
    )

    for clause in _EXPORT_CONTRACT_CLAUSES:
        assert clause not in _REDUCTION_SHAPE_BLOCK
        assert clause not in whole_file_prompt
    # The whole-file prompt keeps its own, separate export sentence.
    assert "exports are names this module deliberately exposes" in whole_file_prompt


# ---------------------------------------------------------------------------
# 0.14.6: `exports`-specific fixed-capsule cap evidence
# ---------------------------------------------------------------------------
# The at-cap/over-cap regressions above exercise `functions` only. The released
# failure was an over-cap `exports` list, so the rejection path that actually
# fired had no direct coverage.


def test_run_fragment_at_cap_exports_is_accepted(tmp_path) -> None:
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    exports = _at_cap_exports()
    provider = _Provider(json.dumps({"description": "ok", "exports": exports}))
    agent = FileDocumentationAgent(provider, max_content_chars=1000)

    result = agent.run_fragment(request)

    assert result["exports"] == exports
    assert provider.calls == 1


def test_run_fragment_over_cap_exports_is_rejected_without_correction(
    tmp_path,
) -> None:
    """The exact 0.14.5 rejection: an over-cap `exports` list is rejected in
    full with `fixed_cap_exceeded` and an `item_limit` removal reported on an
    `exports[...]` path -- never truncated to the first 32 names."""
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    provider = _Provider(
        json.dumps({"description": "ok", "exports": _over_cap_exports()})
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    assert agent._correction is None

    with pytest.raises(ResponseContractError) as caught:
        agent.run_fragment(request)

    diagnostic = caught.value.diagnostic
    assert diagnostic.reason_code == "fixed_cap_exceeded"
    assert any(
        removal.field.startswith("exports[")
        and removal.reason_code == "item_limit"
        for removal in diagnostic.removed
    )
    assert provider.calls == 1


def test_run_fragment_over_long_export_item_is_rejected_without_correction(
    tmp_path,
) -> None:
    """The collection-size cap is not the only export bound: one item over
    `MAX_LEAF_EXPORT_ITEM_CHARS` is also a lossy removal, reported as
    `response_cap` on that item's own path."""
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    provider = _Provider(
        json.dumps(
            {
                "description": "ok",
                "exports": ["e" * (MAX_LEAF_EXPORT_ITEM_CHARS + 1)],
            }
        )
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)

    with pytest.raises(ResponseContractError) as caught:
        agent.run_fragment(request)

    diagnostic = caught.value.diagnostic
    assert diagnostic.reason_code == "fixed_cap_exceeded"
    assert any(
        removal.field == "exports[0]" and removal.reason_code == "response_cap"
        for removal in diagnostic.removed
    )
    assert provider.calls == 1


def test_export_contract_never_suppresses_a_manifest_style_export(tmp_path):
    """The exclusion is scoped to containment, not to the shape of a literal.

    Several supported languages declare their exports *as* a list or an object
    -- an exported-names manifest, a brace-enclosed export list, an assignment
    to the module's export table -- and CodeDoc's own public surface is one of
    them (`codedoc/__init__.py` uses `__all__`). An unqualified "array
    elements and object properties are not exports" would tell a model to drop
    every real export in those languages, trading the 0.14.5 over-reporting
    bug for a silent under-reporting one.

    Only the prompt wording can be guarded here: the fixed cleaner accepts
    whatever names a response carries, so no end-to-end run can demonstrate
    that the *prompt* stopped suppressing a manifest-style export."""
    request = _leaf_request(tmp_path)

    _system, prompt = build_fragment_prompt(request)
    contract = fda._FRAGMENT_EXPORT_CONTRACT

    # The exclusion must be conditional, never absolute.
    assert "merely because the value containing them is exported" in contract
    exclusion = contract[contract.index("Containment is not export"):]
    assert "are data, not module exports." not in exclusion

    # ...and the carve-out must be present, in the same block, exactly once.
    assert contract.count("IS the module's own export declaration") == 1
    assert "must be reported" in contract
    assert prompt.count("IS the module's own export declaration") == 1

    # Stated neutrally: the carve-out survives the language-neutrality rule
    # rather than being dropped by it.
    for language_specific in ("__all__", "module.exports", "export {", "export const"):
        assert language_specific not in contract


# ---------------------------------------------------------------------------
# 0.14.7: the shared split-leaf signature contract
# ---------------------------------------------------------------------------
# Section 3/4.1: the fixed fragment prompt required an exact copy of the
# visible declaration AND a signature at or below 600 characters. CodeDoc's
# own source contains declarations of 631 and 634 characters, so for those
# no truthful response existed -- unsatisfiable by construction, and (section
# 4.2) the rule never reached the correction route at all, since
# `run_fragment()` passes only `_FRAGMENT_SHAPE_BLOCK` to
# `_finalize_fixed_response()` and the "copied from the visible declaration"
# sentence lived only in `_FRAGMENT_PROMPT_TEMPLATE`. The repair states a
# recommended target below the hard bound and makes a shortened signature a
# correct, expected response, carried through both routes byte-identically --
# exactly the single-source mechanism 0.14.6 established for `exports`.

#: Every clause the shared signature contract must state, spelled out here
#: rather than imported from production, so these assertions cannot be
#: satisfied by the very string they exist to police.
_SIGNATURE_CONTRACT_CLAUSES = (
    "A signature is source-backed declaration text visible in THIS fragment",
    "Never infer a missing prefix, suffix, name, parameter, type, "
    "delimiter, or arity from another continuation, parser metadata, "
    "language convention, or general knowledge",
    "Shorten only a fully visible declaration that exceeds the hard bound",
    "preferably roughly 600-1,000 characters and never more than the "
    "hard bound",
    "a fully visible declaration between that range and the hard bound is "
    "reported in full",
    "report only the contiguous signature text actually visible here, in "
    "source order and at or below the hard bound",
    "Partial visibility, not the declaration's whole-file length, "
    "controls this rule",
    "a shortened or partial signature cannot by itself distinguish them",
    "the parser-owned source range, scope, and semantic-unit identity "
    "remain authoritative",
    "a partial or shortened signature is only a matching hint, never "
    "the declaration's identity",
    'Omit "signature" only when the language expresses none, or this '
    "fragment exposes no usable declaration text",
    "Length alone is never a reason to omit a fully visible declaration",
)


def _assert_states_the_signature_contract_once(prompt: str) -> None:
    for clause in _SIGNATURE_CONTRACT_CLAUSES:
        assert prompt.count(clause) == 1, clause
    assert (
        f"The hard bound is {MAX_LEAF_SYMBOL_SIGNATURE_CHARS} characters"
        in prompt
    )
    assert (
        f"each symbol signature <= {MAX_LEAF_SYMBOL_SIGNATURE_CHARS} characters"
        in prompt
    )
    # The retired `_FRAGMENT_PROMPT_TEMPLATE` sentence must not survive
    # alongside the shared contract -- two statements of the same rule that
    # can drift apart is exactly the failure mode this release removes, and
    # it is also the false claim section 5.1 clause 1 forbids: the reported
    # signature is not what keeps overloads distinct.
    assert "so that overloads sharing a name stay distinguishable" not in prompt


def _corrected_leaf_signature_agent(tmp_path):
    """One real leaf agent over a small fragment: the first response's
    signature exceeds the hard bound and is rejected; the correction
    response shortens it to a truthful leading portion within the
    600-1,000 preferred range, which is accepted."""
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    provider = _RecordingCorrectingProvider(
        first_response={
            "description": "ok",
            "functions": [
                {
                    "name": "alpha",
                    "signature": "s" * (MAX_LEAF_SYMBOL_SIGNATURE_CHARS + 1),
                }
            ],
        },
        corrected_response={
            "description": "Corrected leaf.",
            "functions": [
                {
                    "name": "alpha",
                    "signature": "s" * 800,
                }
            ],
        },
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )
    return request, provider, agent


def test_fragment_prompt_states_the_signature_contract_exactly_once(tmp_path):
    """0.14.7: the initial split-leaf prompt must make the fixed signature
    contract satisfiable. Before this release the shape block bounded
    `signature` to 600 characters while the fragment rules separately
    demanded an exact copy of the visible declaration -- a declaration
    longer than the hard bound (as CodeDoc's own source contains) had no
    truthful accepted response."""
    request = _leaf_request(tmp_path)

    _system, prompt = build_fragment_prompt(request)

    _assert_states_the_signature_contract_once(prompt)


def test_initial_and_correction_leaf_prompts_carry_the_identical_signature_contract(
    tmp_path,
):
    """No-drift guard, signature-specific: this drives a real over-bound
    signature rejection through the real correction component and inspects
    both prompts actually sent, proving the contract reaches the correction
    route the 0.14.6-era rule never did (section 4.2)."""
    request, provider, agent = _corrected_leaf_signature_agent(tmp_path)

    result = agent.run_fragment(request)

    assert result == {
        "description": "Corrected leaf.",
        "functions": [
            {
                "name": "alpha",
                "signature": "s" * 800,
            }
        ],
    }
    assert provider.calls == 2
    initial_prompt, correction_prompt = provider.prompts
    assert fda._FRAGMENT_SHAPE_BLOCK in initial_prompt
    assert fda._FRAGMENT_SHAPE_BLOCK in correction_prompt
    _assert_states_the_signature_contract_once(initial_prompt)
    _assert_states_the_signature_contract_once(correction_prompt)


def test_correction_prompt_states_the_same_signature_contract_once(tmp_path):
    """0.14.7: the one targeted correction call must carry the same
    signature contract as the initial call -- the exact defect section 4.2
    identifies: the correction route received only the bare 600-character
    bound, with no statement of what a signature is or whether it may be
    shortened."""
    request, provider, agent = _corrected_leaf_signature_agent(tmp_path)

    agent.run_fragment(request)

    assert provider.calls == 2
    correction_prompt = provider.prompts[1]
    assert "Previous response (verbatim" in correction_prompt
    _assert_states_the_signature_contract_once(correction_prompt)


def test_signature_contract_is_self_contained_for_the_correction_route(tmp_path):
    """The correction prompt renders no fragment position, no continuation
    flags, and no known-symbol line, so every signature clause must resolve
    against the visible source alone -- the same self-containment guarantee
    0.14.6 proved for the export contract."""
    request, provider, agent = _corrected_leaf_signature_agent(tmp_path)

    agent.run_fragment(request)

    correction_prompt = provider.prompts[1]
    assert "Continues an earlier fragment" not in correction_prompt
    assert "Known symbol name(s)" not in correction_prompt
    assert "Fragment position:" not in correction_prompt
    for initial_only_line in (
        "Fragment position:",
        "Fragment metadata",
        "Continues an earlier fragment",
        "Continues into a later fragment",
        "Known symbol name",
        "known_symbols",
    ):
        assert initial_only_line not in fda._FRAGMENT_SIGNATURE_CONTRACT


def test_signature_contract_does_not_reach_reduction_or_final_or_whole_file_prompts():
    """Scope guard: the contract governs the fixed split-leaf capsule only.
    It must not leak into the reduction/final-synthesis capsule (which never
    carries a signature at all) or into the whole-file `single` prompt."""
    from codedoc.agents.file_synthesis_agent import _REDUCTION_SHAPE_BLOCK

    _system, whole_file_prompt = build_prompt(
        "src/example.py", "x = 1\n", ["os"], "python"
    )

    for clause in _SIGNATURE_CONTRACT_CLAUSES:
        assert clause not in _REDUCTION_SHAPE_BLOCK
        assert clause not in whole_file_prompt


def test_run_fragment_signature_above_preferred_range_but_below_bound_is_accepted_in_full(
    tmp_path,
) -> None:
    """0.14.7 section 5.1 clause 2: the 600-1,000 preferred shortening range
    only applies to a declaration that exceeds the hard bound; it is not a
    second enforced limit or permission to shorten a fully visible
    declaration that already fits. A signature strictly between that range
    and the hard bound (2,000) must be reported in full, unchanged, with no
    correction call. A test suite that rejected this gap would have quietly
    converted guidance into a second hard bound."""
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    signature = "s" * 1500
    assert 1000 < len(signature) <= MAX_LEAF_SYMBOL_SIGNATURE_CHARS
    provider = _Provider(
        json.dumps(
            {
                "description": "ok",
                "functions": [{"name": "alpha", "signature": signature}],
            }
        )
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)

    result = agent.run_fragment(request)

    assert result["functions"][0]["signature"] == signature
    assert provider.calls == 1


def test_run_fragment_signature_over_hard_bound_is_rejected_without_correction(
    tmp_path,
) -> None:
    """The repair makes a truthful response possible; it must not make an
    over-bound response acceptable. An over-bound signature is still
    rejected losslessly with `fixed_cap_exceeded` and a `response_cap`
    removal on the `functions[0].signature` path -- never silently
    shortened to fit."""
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    provider = _Provider(
        json.dumps(
            {
                "description": "ok",
                "functions": [
                    {
                        "name": "alpha",
                        "signature": "s" * (MAX_LEAF_SYMBOL_SIGNATURE_CHARS + 1),
                    }
                ],
            }
        )
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    assert agent._correction is None

    with pytest.raises(ResponseContractError) as caught:
        agent.run_fragment(request)

    diagnostic = caught.value.diagnostic
    assert diagnostic.reason_code == "fixed_cap_exceeded"
    assert any(
        removal.field == "functions[0].signature"
        and removal.reason_code == "response_cap"
        for removal in diagnostic.removed
    )
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# 0.14.7 section 7 / 8.D / 9.1 items 14 & 16: the exact direct-agent
# fully-visible boundary + correction matrix at 1,001 / 1,500 / 2,000 / 2,001
# ---------------------------------------------------------------------------
# Every declaration below is REAL single-line Python source of an exact length,
# embedded verbatim in the request fragment -- never a bare ``"s" * n`` value
# disconnected from the request source (section 7).


class _RecordingProvider(_Provider):
    """`_Provider` that also keeps every prompt it was sent."""

    def __init__(self, raw: str) -> None:
        super().__init__(raw)
        self.prompts: list[str] = []

    def complete_json(self, prompt, system=""):
        self.prompts.append(prompt)
        return super().complete_json(prompt, system)


def _exact_len_def(total: int, name: str = "alpha") -> str:
    """`def <name>(<one long parameter identifier>) -> int:` of exactly
    *total* code points -- one physical line, real parseable Python."""
    head, tail = f"def {name}(", ") -> int:"
    pad = total - len(head) - len(tail)
    assert pad >= 1, total
    declaration = head + ("p" * pad) + tail
    assert len(declaration) == total and "\n" not in declaration
    return declaration


def _exact_len_class(total: int, name: str = "Big") -> str:
    """`class <name>(<one long base identifier>):` of exactly *total* code
    points -- one physical line, real parseable Python."""
    head, tail = f"class {name}(", "):"
    pad = total - len(head) - len(tail)
    assert pad >= 1, total
    declaration = head + ("B" * pad) + tail
    assert len(declaration) == total and "\n" not in declaration
    return declaration


def _fully_visible_leaf_request(tmp_path, declaration: str):
    """A real leaf request whose single fragment shows *declaration* in full."""
    source = declaration + "\n    x = 1\n"
    request = _leaf_request(tmp_path, content=source, max_content_chars=5000)
    assert request.unit_chunk_count == 1
    assert request.continuation_before is False
    assert request.continuation_after is False
    assert declaration in request.payload
    return request, source


def _leaf_agent_with_real_correction(provider):
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import CorrectionLedger
    from codedoc.core.usage import UsageAccumulator

    agent = FileDocumentationAgent(provider, max_content_chars=5000)
    agent._correction = ResponseCorrectionAgent(
        provider, UsageAccumulator(), CorrectionLedger(True), True,
    )
    return agent


_ACCEPTED_IN_FULL_LENGTHS = (1001, 1500, 2000)


@pytest.mark.parametrize("length", _ACCEPTED_IN_FULL_LENGTHS)
@pytest.mark.parametrize("kind", ("function", "class"))
def test_run_fragment_fully_visible_signature_at_or_below_bound_is_copied_in_full(
    tmp_path, length, kind
) -> None:
    """Section 9.1 item 14: a fully visible declaration of exactly 1,001 /
    1,500 / 2,000 source characters is copied and accepted unchanged, with no
    correction call, on both the function and class signature paths."""
    declaration = (
        _exact_len_def(length) if kind == "function" else _exact_len_class(length)
    )
    request, _source = _fully_visible_leaf_request(tmp_path, declaration)
    field = "functions" if kind == "function" else "classes"
    name = "alpha" if kind == "function" else "Big"
    provider = _RecordingProvider(
        json.dumps({
            "description": "Fully visible declaration.",
            field: [{"name": name, "description": "does x", "signature": declaration}],
        })
    )
    agent = FileDocumentationAgent(provider, max_content_chars=5000)

    result = agent.run_fragment(request)

    assert result[field][0]["signature"] == declaration
    assert len(result[field][0]["signature"]) == length
    assert provider.calls == 1
    _assert_states_the_signature_contract_once(provider.prompts[0])


@pytest.mark.parametrize("length", _ACCEPTED_IN_FULL_LENGTHS)
def test_run_fragment_fully_visible_signature_accepted_through_real_correction(
    tmp_path, length
) -> None:
    """Section 9.1 item 14: the same full value is accepted through the real
    correction component after a deliberately invalid initial response, and
    both prompts carry the signature contract exactly once."""
    declaration = _exact_len_def(length)
    request, _source = _fully_visible_leaf_request(tmp_path, declaration)
    provider = _RecordingCorrectingProvider(
        first_response={
            "description": "ok",
            "functions": [{"name": "alpha", "signature": _exact_len_def(2001)}],
        },
        corrected_response={
            "description": "Corrected.",
            "functions": [{"name": "alpha", "signature": declaration}],
        },
    )
    agent = _leaf_agent_with_real_correction(provider)

    result = agent.run_fragment(request)

    assert result["functions"][0]["signature"] == declaration
    assert provider.calls == 2
    for prompt in provider.prompts:
        _assert_states_the_signature_contract_once(prompt)
        assert fda._FRAGMENT_SHAPE_BLOCK in prompt


@pytest.mark.parametrize("kind", ("function", "class"))
def test_run_fragment_fully_visible_over_bound_declaration_shortening_matrix(
    tmp_path, kind
) -> None:
    """Section 9.1 item 14: a fully visible 2,001-character declaration.

    * returning all 2,001 characters is rejected in full (`fixed_cap_exceeded`,
      a `response_cap` removal on the exact ``<kind>[0].signature`` path,
      value-free diagnostics);
    * an 800-character leading source-backed portion -- proved byte-for-byte
      contiguous from the fragment -- is accepted;
    * both outcomes agree between the initial and real correction paths;
    * a correction that still returns all 2,001 characters fails the file
      without a second correction.
    """
    declaration = (
        _exact_len_def(2001) if kind == "function" else _exact_len_class(2001)
    )
    request, source = _fully_visible_leaf_request(tmp_path, declaration)
    field = "functions" if kind == "function" else "classes"
    name = "alpha" if kind == "function" else "Big"
    shortened = declaration[:800]  # a real leading slice of the visible source
    assert declaration.startswith(shortened) and shortened in source
    over = {"description": "ok", field: [{"name": name, "signature": declaration}]}
    accepted = {
        "description": "Shortened but truthful.",
        field: [{"name": name, "description": "does x", "signature": shortened}],
    }

    # 1) initial path rejects the full 2,001-character value, value-free.
    reject_provider = _Provider(json.dumps(over))
    reject_agent = FileDocumentationAgent(reject_provider, max_content_chars=5000)
    assert reject_agent._correction is None
    with pytest.raises(ResponseContractError) as caught:
        reject_agent.run_fragment(request)
    diag = caught.value.diagnostic
    assert diag.reason_code == "fixed_cap_exceeded"
    assert any(
        r.field == f"{field}[0].signature" and r.reason_code == "response_cap"
        for r in diag.removed
    )
    assert declaration not in json.dumps(diag.as_summary())
    assert reject_provider.calls == 1

    # 2) initial path accepts the 800-character leading portion unchanged.
    accept_provider = _Provider(json.dumps(accepted))
    accept_result = FileDocumentationAgent(
        accept_provider, max_content_chars=5000
    ).run_fragment(request)
    assert accept_result[field][0]["signature"] == shortened
    assert accept_provider.calls == 1

    # 3) real correction accepts the same 800-character shortening.
    fix_provider = _CorrectingProvider(
        first_response=over, corrected_response=accepted
    )
    fix_result = _leaf_agent_with_real_correction(fix_provider).run_fragment(request)
    assert fix_result[field][0]["signature"] == shortened
    assert fix_provider.calls == 2

    # 4) a correction that still returns all 2,001 characters fails the file
    #    without a second correction.
    still_bad = _CorrectingProvider(first_response=over, corrected_response=over)
    with pytest.raises(ResponseContractError) as caught2:
        _leaf_agent_with_real_correction(still_bad).run_fragment(request)
    assert caught2.value.correction_attempted is True
    assert still_bad.calls == 2


# ---------------------------------------------------------------------------
# 0.14.9 F-1: the shared fixed-capsule cap-repair instruction reaches the real
# split-leaf correction prompt (section 5.2 route coverage / 9.1 items 6, 20,
# 21, 26). LEAF_CAPSULE_SCHEMA_REVISION advances to v11 (section 5.6.2 / G-4).
# ---------------------------------------------------------------------------

def test_leaf_capsule_schema_revision_advanced_to_v11():
    """Mutation 18: reverting this alone must fail here, independently of the
    reducer revision."""
    from codedoc.core.file_division import LEAF_CAPSULE_SCHEMA_REVISION

    assert LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v11"


def test_leaf_correction_prompt_carries_cap_repair_rule_with_260_target_for_description(
    tmp_path,
):
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    provider = _RecordingCorrectingProvider(
        first_response={"description": "d" * 313},
        corrected_response={"description": "Corrected, concise fragment."},
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = _real_correction(provider)

    result = agent.run_fragment(request)

    assert result == {"description": "Corrected, concise fragment."}
    assert provider.calls == 2
    initial_prompt, correction_prompt = provider.prompts
    assert "Cap repair" not in initial_prompt
    assert fda._FRAGMENT_SHAPE_BLOCK in correction_prompt
    cap = correction_prompt.split("Cap repair", 1)
    assert len(cap) == 2, "the correction prompt must carry the cap-repair rule"
    rule = cap[1]
    assert "rejected in full" in rule                       # clause 1
    assert "must not be copied" in rule                     # clause 2
    assert "shorter, meaning-preserving value" in rule      # clause 3
    assert "description: within 260 characters" in rule     # clause 4 target
    assert "every other valid field and fact" in rule       # clause 5
    # The general preserve-valid-facts rule still stands (subordinate, not gone).
    assert (
        "Preserve every valid fact already present in the previous response"
        in correction_prompt
    )


def test_leaf_correction_prompt_states_260_target_for_symbol_description(tmp_path):
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )
    provider = _RecordingCorrectingProvider(
        first_response={
            "description": "ok",
            "functions": [{"name": "alpha", "description": "x" * 313}],
        },
        corrected_response={
            "description": "ok",
            "functions": [{"name": "alpha", "description": "short"}],
        },
    )
    agent = FileDocumentationAgent(provider, max_content_chars=1000)
    agent._correction = _real_correction(provider)

    agent.run_fragment(request)

    rule = provider.prompts[1].split("Cap repair", 1)[1]
    assert "functions[0].description: within 260 characters" in rule


def test_leaf_correction_over_bound_signature_defers_with_no_rewrite_target(tmp_path):
    """9.1 items 21, 26: a rejected source-backed ``signature`` gets clauses 1,
    2, 5 and shape-contract deferral -- never a rewrite-shorter numeric
    target."""
    request, provider, agent = _corrected_leaf_signature_agent(tmp_path)

    agent.run_fragment(request)

    correction_prompt = provider.prompts[1]
    cap = correction_prompt.split("Cap repair", 1)[1]
    assert "functions[0].signature" in cap
    assert "must not invent a shorter identifier" in cap
    assert "within" not in cap and "260" not in cap
    assert "shorter, meaning-preserving value" not in cap
    # The signature shape contract is still stated exactly once (deferral, not
    # duplication).
    _assert_states_the_signature_contract_once(correction_prompt)


def test_leaf_cap_repair_strict_acceptance_300_ok_301_fails_with_one_call(tmp_path):
    request = _leaf_request(
        tmp_path, content="def alpha():\n    return 1\n", max_content_chars=1000
    )

    ok = _CorrectingProvider(
        first_response={"description": "d" * 313},
        corrected_response={"description": "d" * 300},
    )
    ok_agent = FileDocumentationAgent(ok, max_content_chars=1000)
    ok_agent._correction = _real_correction(ok)
    assert ok_agent.run_fragment(request)["description"] == "d" * 300
    assert ok.calls == 2

    bad = _CorrectingProvider(
        first_response={"description": "d" * 313},
        corrected_response={"description": "d" * 301},
    )
    bad_agent = FileDocumentationAgent(bad, max_content_chars=1000)
    bad_agent._correction = _real_correction(bad)
    with pytest.raises(ResponseContractError) as caught:
        bad_agent.run_fragment(request)
    assert caught.value.correction_attempted is True
    assert bad.calls == 2  # exactly one correction call, no third attempt
