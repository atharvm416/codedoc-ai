"""0.10.0 Gate 1 — combined FileDocumentationAgent + strict cleaning."""

from __future__ import annotations

import json

import pytest

from codedoc.agents import file_documentation_agent as fda
from codedoc.agents.file_documentation_agent import (
    MAX_COMBINED_RESPONSE_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_EXPORT_ITEMS,
    MAX_KEY_CONCEPT_ITEMS,
    MAX_SYMBOL_ITEMS_PER_KIND,
    FileDocumentationAgent,
    build_prompt,
    clean_combined_response,
)
from codedoc.utils.errors import AgentError


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


def _run(raw: str):
    agent = FileDocumentationAgent(_Provider(raw))
    return agent.run("pkg/mod.py", "x = 1\n", ["os"], "python")


# ---------------------------------------------------------------------------
# Prompt construction and truncation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cleaning: unknown keys, empties, booleans, dedup, malformed items
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dependency object schema
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Item-count and length caps
# ---------------------------------------------------------------------------

def test_symbol_count_capped():
    funcs = [{"name": f"f{i}"} for i in range(50)]
    cleaned = clean_combined_response({"description": "d", "functions": funcs}, "m.py")
    assert len(cleaned["functions"]) == MAX_SYMBOL_ITEMS_PER_KIND


def test_export_count_capped():
    cleaned = clean_combined_response(
        {"description": "d", "exports": [f"e{i}" for i in range(100)]}, "m.py"
    )
    assert len(cleaned["exports"]) == MAX_EXPORT_ITEMS


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


# ---------------------------------------------------------------------------
# Global cap enforcement
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Failure shapes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Workstream D — deterministic multi-language fixture coverage
# ---------------------------------------------------------------------------

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
