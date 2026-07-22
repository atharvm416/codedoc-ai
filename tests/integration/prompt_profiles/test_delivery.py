"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.prompt_profile_runs import project  # noqa: F401, F811


import json
from tests.support.providers import SmartFake
from tests.support.prompt_profile_runs import _run
from tests.support.prompt_profile_runs import _output
from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
)
from tests.support.prompt_delivery_cases import _profile
from tests.support.prompt_delivery_cases import _request

def test_no_profile_makes_no_review_and_no_digest(monkeypatch, project):
    fake = SmartFake()
    stats = _run(monkeypatch, project, {"entry_file": "main.py"}, fake)
    assert fake.review_calls == 0 and stats["checked"] == 1
    assert stats["prompt_customization_security_review"] == "not-required"
    assert stats["documentation_calls_attempted"] == stats["attempted_calls"] == 1
    assert stats["prompt_customization_security_review_calls_attempted"] == 0
    rec = json.loads(_output(project).read_text())["files"][0]
    assert "_prompt_profile_digest" not in rec

class CombinedProvider:
    """Volunteers every known top-level and nested field."""

    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return json.dumps({
            "description": "kept",
            "role_in_system": "drop-role",
            "functions": [{"name": "f", "description": "drop"}],
            "classes": [{"name": "C", "description": "drop"}],
            "exports": ["E"],
            "key_concepts": ["kc"],
            "usage_example": "import x",
            "dependencies_analysis": {
                "internal": ["a.py"],
                "external": ["requests"],
                "warnings": ["w"],
            },
        })

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

class TripleProvider:
    """Returns superset structure/dependency/documentation responses."""

    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        if "Generate documentation" in prompt:
            return json.dumps({
                "description": "doc", "role_in_system": "r",
                "key_concepts": ["kc"], "usage_example": "u",
            })
        if "dependencies_analysis" in prompt and "Analyse the imports" in prompt:
            return json.dumps({"dependencies_analysis": {
                "internal": ["a.py"], "external": ["requests"], "warnings": ["w"]}})
        return json.dumps({
            "description": "s", "role_in_system": "r",
            "functions": [{"name": "f", "description": "d"}],
            "classes": [{"name": "C", "description": "d"}],
            "exports": ["E"],
        })

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

def test_single_filters_omitted_top_level_and_nested_fields_and_stamps_digest():
    resolved = _profile({"single": {"fields": [
        {"key": "description", "type": "string", "instruction": "Custom"},
        {"key": "dependencies_analysis.internal", "type": "string_list", "instruction": "i"},
    ]}})
    result = Orchestrator(CombinedProvider()).process(_request(resolved, "x = 1"))
    assert result["description"] == "kept"
    # Omitted known top-level fields removed by the shared filter before merge.
    assert result["role_in_system"] == ""
    assert result["functions"] == []
    assert result["classes"] == []
    assert result["exports"] == []
    assert result["key_concepts"] == []
    # dependencies_analysis keeps only the requested nested member.
    assert result["dependencies_analysis"] == {"internal": ["a.py"]}
    assert result["_prompt_profile_digest"] == resolved.file_digest("a.py")

def test_no_profile_delivery_is_identity_and_unstamped():
    result = Orchestrator(CombinedProvider()).process(_request(None, "x = 1"))
    assert result["role_in_system"] == "drop-role"
    assert result["functions"] == [{"name": "f", "description": "drop"}]
    assert "_prompt_profile_digest" not in result

def test_default_equivalent_profile_is_inactive_and_unstamped():
    from codedoc.core.prompt_profiles import default_prompt_profiles

    resolved = _profile(default_prompt_profiles("single", schema_version=1))
    result = Orchestrator(CombinedProvider()).process(_request(resolved, "x = 1"))
    # developer-standard-equivalent -> no field dropped, no digest stamped.
    assert result["role_in_system"] == "drop-role"
    assert "_prompt_profile_digest" not in result
    assert resolved.file_digest("a.py") == NO_PROMPT_PROFILE_DIGEST

def test_per_extension_override_selected_by_file_basename():
    resolved = _profile({"single": {
        "fields": [{"key": "description", "type": "string", "instruction": "base"}],
        "per_extension": {".py": {"fields": [
            {"key": "description", "type": "string", "instruction": "py"},
            {"key": "exports", "type": "string_list", "instruction": "py exports"}]}},
    }})
    py = resolved.resolve_block("combined", "mod.py")
    java = resolved.resolve_block("combined", "Mod.java")
    assert py.requested_field_paths == ("description", "exports")
    assert java.requested_field_paths == ("description",)
    # Distinct rendered blocks => distinct per-extension digests.
    assert resolved.file_digest("mod.py") != resolved.file_digest("Mod.java")

def test_triple_filters_each_subagent_before_merge():
    resolved = _profile({"triple": {
        "structure": {"fields": [
            {"key": "description", "type": "string", "instruction": "d"},
            {"key": "functions", "type": "symbol_list", "instruction": "fn"}]},
        "dependency": {"fields": [
            {"key": "dependencies_analysis.internal", "type": "string_list", "instruction": "i"}]},
        "documentation": {"fields": [
            {"key": "description", "type": "string", "instruction": "d"}]},
    }}, mode="triple")
    result = Orchestrator(
        TripleProvider(), parallel=False, analysis_mode="triple",
    ).process(_request(resolved, "x = 1", mode="triple"))

    # structure kept functions, dropped classes/exports/role.
    assert result["functions"] == [{"name": "f", "description": "d"}]
    assert result["classes"] == []
    assert result["exports"] == []
    # dependency kept only internal.
    assert result["dependencies_analysis"] == {"internal": ["a.py"]}
    # documentation kept only description (key_concepts/usage_example dropped).
    assert result["key_concepts"] == []
    assert result["usage_example"] == ""
    assert result["_prompt_profile_digest"] == resolved.file_digest("a.py")
