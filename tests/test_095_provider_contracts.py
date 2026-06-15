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

import sys
import types
import urllib.error
from pathlib import Path

import pytest

from codedoc.core.execution import _is_rate_limit_error
from codedoc.utils.errors import LLMError

_JSON_HINT = "Respond ONLY with valid JSON"


# ---------------------------------------------------------------------------
# Shared fake response shapes
# ---------------------------------------------------------------------------

class _OpenAIMessage:
    def __init__(self, content):
        self.content = content


class _OpenAIChoice:
    def __init__(self, content):
        self.message = _OpenAIMessage(content)


class _OpenAIResponse:
    def __init__(self, content):
        self.choices = [_OpenAIChoice(content)]


class _AnthropicBlock:
    def __init__(self, text):
        self.text = text


class _AnthropicResponse:
    def __init__(self, text):
        self.content = [_AnthropicBlock(text)]


class _GeminiResponse:
    def __init__(self, text):
        self.text = text


# ---------------------------------------------------------------------------
# Fake SDK installers
# ---------------------------------------------------------------------------

def _install_openai(monkeypatch, rec, *, content='{"ok": true}', error=None):
    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            rec["create_kwargs"] = kwargs
            if error is not None:
                raise error
            return _OpenAIResponse(content)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            rec["init"] = {"api_key": api_key, "base_url": base_url}
            self.chat = _Chat()

    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)


def _install_anthropic(monkeypatch, rec, *, text='{"ok": true}', error=None):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            rec["create_kwargs"] = kwargs
            if error is not None:
                raise error
            return _AnthropicResponse(text)

    class Anthropic:
        def __init__(self, api_key=None):
            rec["init"] = {"api_key": api_key}
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)


def _install_gemini(monkeypatch, rec, *, text='{"ok": true}', error=None):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            rec.setdefault("configs", []).append(kwargs)

    gtypes.GenerateContentConfig = GenerateContentConfig

    class _Models:
        def generate_content(self, model, contents, config):
            rec["generate"] = {"model": model, "contents": contents, "config": config}
            if error is not None:
                raise error
            return _GeminiResponse(text)

    class Client:
        def __init__(self, api_key=None):
            rec["init"] = {"api_key": api_key}
            self.models = _Models()

    genai.Client = Client
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)


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


def test_provider_choice_does_not_change_pipeline_results_or_cache_policy(
    tmp_path,
    monkeypatch,
):
    import json

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
            self.calls += 1
            if "key_concepts" in prompt:
                return json.dumps(
                    {
                        "description": "Provider-neutral documentation.",
                        "role_in_system": "Entry point.",
                        "key_concepts": ["startup"],
                    }
                )
            if "dependencies_analysis" in prompt:
                return json.dumps(
                    {
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
                        }
                    }
                )
            return json.dumps(
                {
                    "description": "Structured entry point.",
                    "role_in_system": "Entry point.",
                    "functions": [{"name": "main", "description": "Run"}],
                    "exports": ["main"],
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
        assert provider.calls == 3
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
