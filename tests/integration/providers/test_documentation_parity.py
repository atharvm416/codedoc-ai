"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import re
import pytest
from tests.support.providers import (
    _install_anthropic,
    _install_gemini,
    _install_openai,
)
from tests.support.one_call_cases import _COMBINED_JSON
from codedoc.utils.errors import LLMError
from tests.support.provider_contract_cases import _make
from codedoc.llm.api_provider import AnthropicProvider, GeminiProvider, OpenAIProvider

_PROVIDER_MATRIX = [
    ("openai", "gpt-test", _install_openai, "content"),
    ("anthropic", "claude-test", _install_anthropic, "text"),
    ("gemini", "gemini-test", _install_gemini, "text"),
]

def _base_config(provider_name, model):
    return {
        "entry_file": "main.py",
        "llm_provider": provider_name,
        "model_name": model,
        "api_key": "test-key",
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
    }

@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_provider_executes_one_call_contract(tmp_path, monkeypatch, provider_name, model, installer, kw):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, **{kw: _COMBINED_JSON})
    stats = run_pipeline(project, _base_config(provider_name, model))

    assert stats["checked"] == 1
    # Exactly one provider request for the single file.
    calls = rec.get("create_calls") or rec.get("generate_calls")
    assert len(calls) == 1
    output = json.loads((project / "codedoc" / "codedoc.json").read_text(encoding="utf-8"))
    assert output["files"][0]["description"] == "A documented module."

@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_provider_malformed_response_fails_file(tmp_path, monkeypatch, provider_name, model, installer, kw):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, **{kw: "this is not json"})
    stats = run_pipeline(project, _base_config(provider_name, model))
    assert stats["checked"] == 0
    assert stats["failed"] == 1

@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_provider_error_propagates_to_failure(tmp_path, monkeypatch, provider_name, model, installer, kw):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, error=RuntimeError("transient boom"))
    stats = run_pipeline(project, _base_config(provider_name, model))
    assert stats["checked"] == 0
    assert stats["failed"] == 1

_ROUTING_RESPONSE = json.dumps(
    {
        "description": "Provider-neutral documentation.",
        "role_in_system": "test",
        "key_concepts": [],
        "usage_example": "",
        "dependencies_analysis": {"internal": [], "external": []},
        "functions": [],
        "classes": [],
        "exports": [],
    }
)

def _sdk_prompts(provider_name, rec):
    if provider_name == "gemini":
        return [call["contents"] for call in rec.get("generate_calls", [])]
    return [call["messages"][-1]["content"] for call in rec.get("create_calls", [])]

@pytest.mark.parametrize(
    "installer, cls",
    [
        (_install_openai, "OpenAIProvider"),
        (_install_anthropic, "AnthropicProvider"),
        (_install_gemini, "GeminiProvider"),
    ],
)
def test_all_providers_return_equivalent_raw_json(monkeypatch, installer, cls):
    rec = {}
    installer(monkeypatch, rec, **({"content": '{"k": 1}'} if cls == "OpenAIProvider" else {"text": '{"k": 1}'}))
    provider = _make(cls)(api_key="k")
    assert provider.complete_json("p", "s") == '{"k": 1}'

@pytest.mark.parametrize(
    "installer, cls",
    [
        (_install_openai, "OpenAIProvider"),
        (_install_anthropic, "AnthropicProvider"),
        (_install_gemini, "GeminiProvider"),
    ],
)
def test_sdk_error_becomes_llm_error(monkeypatch, installer, cls):
    rec = {}
    installer(monkeypatch, rec, error=RuntimeError("boom"))
    provider = _make(cls)(api_key="k")
    with pytest.raises(LLMError):
        provider.complete_json("p", "s")

@pytest.mark.parametrize(
    ("provider_name", "model", "installer"),
    [
        ("openai", "gpt-test", _install_openai),
        ("anthropic", "claude-test", _install_anthropic),
        ("gemini", "gemini-test", _install_gemini),
    ],
)
def test_scope_routing_uses_each_real_adapter_with_cache_reuse(
    tmp_path, monkeypatch, provider_name, model, installer
):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("import helper\n", encoding="utf-8")
    (project / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (project / "orphan.py").write_text("y = 2\n", encoding="utf-8")

    rec = {}
    response_kwarg = {"text": _ROUTING_RESPONSE} if provider_name != "openai" else {
        "content": _ROUTING_RESPONSE
    }
    installer(monkeypatch, rec, **response_kwarg)
    base_config = {
        "entry_file": "main.py",
        "llm_provider": provider_name,
        "model_name": model,
        "api_key": "test-key",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
    }

    entry_stats = run_pipeline(
        project, {**base_config, "documentation_scope": "entry"}
    )
    entry_prompts = _sdk_prompts(provider_name, rec)
    entry_routes = sorted(
        match.group(1).strip()
        for prompt in entry_prompts
        if (match := re.search(r"^File: (.+)$", prompt, re.MULTILINE))
    )
    # 0.10.0: default single mode → one combined call per file.
    assert entry_stats["checked"] == 2
    assert len(entry_prompts) == 2
    assert entry_routes == ["helper.py", "main.py"]

    rec.get("create_calls", []).clear()
    rec.get("generate_calls", []).clear()
    all_stats = run_pipeline(
        project, {**base_config, "documentation_scope": "all"}
    )
    all_prompts = _sdk_prompts(provider_name, rec)
    all_routes = sorted(
        match.group(1).strip()
        for prompt in all_prompts
        if (match := re.search(r"^File: (.+)$", prompt, re.MULTILINE))
    )
    assert all_stats["checked"] == 1
    assert all_stats["skipped"] == 2
    assert all_stats["disconnected_paid_files"] == 1
    assert all_stats["disconnected_planned_calls"] == 1
    assert len(all_prompts) == 1
    assert all_routes == ["orphan.py"]
    output = json.loads(
        (project / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert [file["path"] for file in output["files"]] == [
        "helper.py",
        "orphan.py",
        "main.py",
    ]

def test_provider_choice_does_not_change_pipeline_results_or_cache_policy(
    tmp_path,
    monkeypatch,
):
    from codedoc.pipeline import run_pipeline

    class EquivalentProvider:
        def __init__(self, name):
            self._name = name
            self.calls = 0

        @property
        def provider_name(self):
            return self._name

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            # 0.10.0: one combined response per file (default single mode).
            self.calls += 1
            return json.dumps(
                {
                    "description": "Provider-neutral documentation.",
                    "role_in_system": "Entry point.",
                    "functions": [{"name": "main", "description": "Run"}],
                    "exports": ["main"],
                    "key_concepts": ["startup"],
                    "dependencies_analysis": {
                        "external": ["requests"],
                        "dependency_refs": ["requests"],
                        "catalog_updates": [
                            {
                                "name": "requests",
                                "type": "external",
                                "used_for": "HTTP calls",
                            }
                        ],
                        "usage_notes": [
                            {"import": "requests", "used_for": "HTTP calls"}
                        ],
                    },
                }
            )

    payloads = []
    for provider_name in ("OpenAI(test)", "Anthropic(test)", "Gemini(test)"):
        project = tmp_path / provider_name.split("(", 1)[0].lower()
        project.mkdir()
        (project / "main.py").write_text(
            "import requests\n\ndef main():\n    return requests.get('https://example.test')\n",
            encoding="utf-8",
        )
        provider = EquivalentProvider(provider_name)
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda _config, current=provider: current,
        )
        stats = run_pipeline(
            project,
            {
                "entry_file": "main.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "max_parallel_files": 1,
                "propagate_changes": False,
            },
        )
        assert stats["checked"] == 1
        assert provider.calls == 1
        payloads.append(
            json.loads((project / "docs" / "codedoc.json").read_text(encoding="utf-8"))
        )

        provider_creation_calls = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda _config: provider_creation_calls.append(True),
        )
        cached_stats = run_pipeline(
            project,
            {
                "entry_file": "main.py",
                "output_dir": "docs",
                "llm_provider": "openai",
                "parallel_agents": False,
                "max_parallel_files": 1,
                "propagate_changes": False,
            },
        )
        assert cached_stats["checked"] == 0
        assert cached_stats["skipped"] == 1
        assert provider_creation_calls == []

    assert payloads[0] == payloads[1] == payloads[2]

def test_native_json_entrypoints_present_on_real_adapters():
    # The correction path reuses the same complete_json entrypoint the initial
    # call uses; each real adapter keeps its own native JSON implementation.
    assert "complete_json" in OpenAIProvider.__dict__
    assert "complete_json" not in AnthropicProvider.__dict__  # inherits base fallback
    assert "complete_json" in GeminiProvider.__dict__
