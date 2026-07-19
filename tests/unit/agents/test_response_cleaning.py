"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811

from codedoc.agents.response_cleaning import (
    clean_dependency_response,
    clean_structure_response,
)

class TestBaseAgent:
    def test_parse_json_strips_fences(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        raw = '```json\n{"key": "value"}\n```'
        result = agent._parse_json(raw, "test.py")
        assert result == {"key": "value"}

def test_triple_structure_cleaner_drops_unknown_keys_booleans_and_empties():
    raw = {
        "description": "  trimmed  ",
        "role_in_system": True,           # boolean rejected
        "functions": [
            {"name": "f", "description": "ok"},
            {"name": "f", "description": "ok"},  # duplicate dropped
            {"bogus": "x"},                       # nameless dropped
        ],
        "classes": [],                    # empty omitted
        "exports": ["E", "E"],            # de-duplicated
        "unknown_key": "ignored",         # unknown dropped
    }
    cleaned = clean_structure_response(raw, "m.py")
    assert cleaned["description"] == "trimmed"
    assert "role_in_system" not in cleaned
    assert cleaned["functions"] == [{"name": "f", "description": "ok"}]
    assert "classes" not in cleaned
    assert cleaned["exports"] == ["E"]
    assert "unknown_key" not in cleaned

def test_triple_dependency_cleaner_preserves_shape_and_drops_empties():
    raw = {
        "dependencies_analysis": {
            "internal": ["./a", "./a"],
            "external": ["react"],
            "warnings": [],
        }
    }
    cleaned = clean_dependency_response(raw, "m.py")
    da = cleaned["dependencies_analysis"]
    assert da["internal"] == ["./a"]
    assert da["external"] == ["react"]
    assert "warnings" not in da
