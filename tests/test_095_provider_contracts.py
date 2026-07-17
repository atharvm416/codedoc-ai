"""0.9.5 — three-provider contract matrix with injected fake SDK clients.

Verifies that OpenAI, Anthropic, and Gemini satisfy the same contract without
any network access or credentials:

- OpenAI sends native JSON response-format arguments;
- Anthropic receives the JSON-only system instruction and preserves its output
  token allowance;
- Gemini sends ``response_mime_type="application/json"`` and its system
  instruction;
- all three extract response text and return the raw JSON to the caller;
- all three convert SDK errors to ``LLMError``;
- rate-limit-like SDK errors remain classifiable for shared retry handling.

Fake SDK modules are injected into ``sys.modules`` so the real ``openai`` /
``anthropic`` / ``google-genai`` packages are never required.
"""
from __future__ import annotations

import json
import re
import urllib.error
from pathlib import Path

import pytest

from codedoc.core.execution import _is_rate_limit_error
from codedoc.utils.errors import LLMError
from tests.support.providers import _install_anthropic, _install_gemini, _install_openai

_JSON_HINT = "Respond ONLY with valid JSON"


def _make(provider_cls_name):
    from codedoc.llm import api_provider

    return getattr(api_provider, provider_cls_name)


# ---------------------------------------------------------------------------
# Native JSON-mode contract
# ---------------------------------------------------------------------------

def test_openai_sends_native_json_response_format(monkeypatch):
    rec = {}
    _install_openai(monkeypatch, rec)
    provider = _make("OpenAIProvider")(api_key="k", model="gpt-4o-mini")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert rec["create_kwargs"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_base_url_stays_on_openai_adapter(monkeypatch):
    rec = {}
    _install_openai(monkeypatch, rec)
    provider = _make("OpenAIProvider")(
        api_key="k",
        model="custom-model",
        base_url="https://gateway.example/v1",
    )
    assert provider.provider_name == "OpenAI(custom-model)"
    assert rec["init"]["base_url"] == "https://gateway.example/v1"


def test_anthropic_receives_json_instruction_and_keeps_token_allowance(monkeypatch):
    rec = {}
    _install_anthropic(monkeypatch, rec)
    provider = _make("AnthropicProvider")(api_key="k")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    # JSON-only instruction is carried in the top-level system parameter.
    assert _JSON_HINT in rec["create_kwargs"]["system"]
    # The output token allowance is preserved (not reduced by JSON mode).
    assert rec["create_kwargs"]["max_tokens"] == 4096


def test_gemini_sends_json_mime_type_and_system_instruction(monkeypatch):
    rec = {}
    _install_gemini(monkeypatch, rec)
    provider = _make("GeminiProvider")(api_key="k")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    config = rec["generate"]["config"]
    assert config.response_mime_type == "application/json"
    assert _JSON_HINT in config.system_instruction


# ---------------------------------------------------------------------------
# Equivalent text extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# SDK errors → LLMError
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Rate-limit-like errors remain classifiable for shared retry handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "installer, cls",
    [
        (_install_openai, "OpenAIProvider"),
        (_install_anthropic, "AnthropicProvider"),
        (_install_gemini, "GeminiProvider"),
    ],
)
def test_rate_limit_error_is_classifiable_after_wrapping(monkeypatch, installer, cls):
    rec = {}
    installer(monkeypatch, rec, error=RuntimeError("429 rate limit exceeded"))
    provider = _make(cls)(api_key="k")
    with pytest.raises(LLMError) as excinfo:
        provider.complete_json("p", "s")
    # The shared classifier must still see the rate-limit signal through the
    # LLMError wrapper (message text and/or cause chain).
    assert _is_rate_limit_error(excinfo.value)


def test_provider_adapters_do_not_own_pipeline_policy():
    from codedoc.llm import api_provider

    source = Path(api_provider.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "codedoc.core.discovery",
        "codedoc.core.planning",
        "codedoc.core.safe_writer",
        "codedoc.core.output",
        "codedoc.pipeline",
    )
    assert not any(name in source for name in forbidden_imports)


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


def test_dormant_local_provider_liveness_uses_standard_library(monkeypatch):
    from codedoc.llm.local_provider import LocalProvider

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    provider = object.__new__(LocalProvider)
    provider._base_url = "http://localhost:11434/v1"
    monkeypatch.setattr(
        "codedoc.llm.local_provider.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(),
    )
    assert provider.is_available() is True

    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(
        "codedoc.llm.local_provider.urllib.request.urlopen",
        unavailable,
    )
    assert provider.is_available() is False
