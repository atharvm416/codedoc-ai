"""0.11.0 — agent delivery, the shared post-clean filter, and per-language overrides.

Covers Tests #12, #13, #14, #15 of the plan.
"""

import json

import pytest

from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    validate_profile,
)

LANGS = frozenset({"python", "java"})


def _to_envelope(raw):
    """Wrap a legacy flat profile dict into the 0.11.3 ``common`` envelope."""
    if not isinstance(raw, dict):
        return raw
    out = {k: v for k, v in raw.items() if k in ("schema_version", "$comment")}
    if isinstance(raw.get("single"), dict) and "common" not in raw["single"]:
        sec = raw["single"]
        common = {k: sec[k] for k in ("fields", "requested_shape") if k in sec}
        new_sec = {"common": common}
        if "per_language" in sec:
            new_sec["per_language"] = sec["per_language"]
        out["single"] = new_sec
    elif "single" in raw:
        out["single"] = raw["single"]
    if isinstance(raw.get("triple"), dict) and "common" not in raw["triple"]:
        sec = raw["triple"]
        agents = ("structure", "dependency", "documentation")
        common, per_lang = {}, {}
        for agent in agents:
            block = sec.get(agent, {})
            common[agent] = {k: block[k] for k in ("fields", "requested_shape") if k in block}
            for lang, ov in (block.get("per_language") or {}).items():
                per_lang.setdefault(lang, {})[agent] = ov
        new_sec = {"common": common}
        if per_lang:
            new_sec["per_language"] = per_lang
        out["triple"] = new_sec
    elif "triple" in raw:
        out["triple"] = raw["triple"]
    return out


def _profile(raw, mode="single"):
    return ResolvedProfile(
        mode,
        validate_profile(
            _to_envelope(raw), active_mode=mode, known_languages=LANGS,
            source="inline", source_path=None,
        ),
    )


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


def _descriptor(path="a.py", language="python"):
    return {"rel_path": path, "language": language, "extension": ".py"}


# ---------------------------------------------------------------------------
# Single-mode delivery + filter (Test #12)
# ---------------------------------------------------------------------------

def test_single_filters_omitted_top_level_and_nested_fields_and_stamps_digest():
    resolved = _profile({"single": {"fields": [
        {"key": "description", "type": "string", "instruction": "Custom"},
        {"key": "dependencies_analysis.internal", "type": "string_list", "instruction": "i"},
    ]}})
    result = Orchestrator(CombinedProvider(), resolved_profile=resolved).process(
        _descriptor(), "x = 1", []
    )
    assert result["description"] == "kept"
    # Omitted known top-level fields removed by the shared filter before merge.
    assert result["role_in_system"] == ""
    assert result["functions"] == []
    assert result["classes"] == []
    assert result["exports"] == []
    assert result["key_concepts"] == []
    # dependencies_analysis keeps only the requested nested member.
    assert result["dependencies_analysis"] == {"internal": ["a.py"]}
    assert result["_prompt_profile_digest"] == resolved.file_digest("python")


def test_no_profile_delivery_is_identity_and_unstamped():
    result = Orchestrator(CombinedProvider(), resolved_profile=None).process(
        _descriptor(), "x = 1", []
    )
    assert result["role_in_system"] == "drop-role"
    assert result["functions"] == [{"name": "f", "description": "drop"}]
    assert "_prompt_profile_digest" not in result


def test_default_equivalent_profile_is_inactive_and_unstamped():
    from codedoc.core.prompt_profiles import default_prompt_profiles

    resolved = _profile(default_prompt_profiles("single", schema_version=1))
    result = Orchestrator(CombinedProvider(), resolved_profile=resolved).process(
        _descriptor(), "x = 1", []
    )
    # developer-standard-equivalent -> no field dropped, no digest stamped.
    assert result["role_in_system"] == "drop-role"
    assert "_prompt_profile_digest" not in result
    assert resolved.file_digest("python") == NO_PROMPT_PROFILE_DIGEST


# ---------------------------------------------------------------------------
# Per-language overrides (Test #14)
# ---------------------------------------------------------------------------

def test_per_language_override_selected_by_file_language():
    resolved = _profile({"single": {
        "fields": [{"key": "description", "type": "string", "instruction": "base"}],
        "per_language": {"python": {"fields": [
            {"key": "description", "type": "string", "instruction": "py"},
            {"key": "exports", "type": "string_list", "instruction": "py exports"}]}},
    }})
    py = resolved.resolve_block("combined", "python")
    java = resolved.resolve_block("combined", "java")
    assert py.requested_field_paths == ("description", "exports")
    assert java.requested_field_paths == ("description",)
    # Distinct rendered blocks => distinct per-language digests.
    assert resolved.file_digest("python") != resolved.file_digest("java")


# ---------------------------------------------------------------------------
# Triple-mode delivery + filter-before-merge (Test #13)
# ---------------------------------------------------------------------------

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
        resolved_profile=resolved,
    ).process(_descriptor(), "x = 1", [])

    # structure kept functions, dropped classes/exports/role.
    assert result["functions"] == [{"name": "f", "description": "d"}]
    assert result["classes"] == []
    assert result["exports"] == []
    # dependency kept only internal.
    assert result["dependencies_analysis"] == {"internal": ["a.py"]}
    # documentation kept only description (key_concepts/usage_example dropped).
    assert result["key_concepts"] == []
    assert result["usage_example"] == ""
    assert result["_prompt_profile_digest"] == resolved.file_digest("python")


# ---------------------------------------------------------------------------
# Provider parity (Test #15) — fake clients, no network.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("installer", "class_name", "response_kw"),
    [
        ("_install_openai", "OpenAIProvider", "content"),
        ("_install_anthropic", "AnthropicProvider", "text"),
        ("_install_gemini", "GeminiProvider", "text"),
    ],
)
def test_provider_adapters_receive_identical_documentation_blocks(
    monkeypatch, installer, class_name, response_kw
):
    from codedoc.llm import api_provider
    from tests import test_095_provider_contracts as contracts

    rec = {}
    response = json.dumps({"description": "adapter result"})
    getattr(contracts, installer)(monkeypatch, rec, **{response_kw: response})
    provider = getattr(api_provider, class_name)(api_key="k")
    resolved = _profile({"single": {"fields": [
        {"key": "description", "type": "string", "instruction": "Custom desc"}]}})
    shape = resolved.resolve_block("combined", "python")
    result = Orchestrator(provider, resolved_profile=resolved).process(
        _descriptor(), "x=1", []
    )
    assert result["description"] == "adapter result"
    if class_name == "OpenAIProvider":
        sent = rec["create_kwargs"]["messages"][-1]["content"]
    elif class_name == "AnthropicProvider":
        sent = rec["create_kwargs"]["messages"][0]["content"]
    else:
        sent = rec["generate"]["contents"]
    assert shape.text in sent
