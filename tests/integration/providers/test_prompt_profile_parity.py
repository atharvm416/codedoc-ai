"""Tests organized by feature ownership."""

import json
import sys
import types
import pytest
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import PromptCustomizationValidationError
from tests.support.providers import (
    _AnthropicResponse,
    _GeminiResponse,
    _OpenAIResponse,
)
from codedoc.agents.orchestrator import Orchestrator
from tests.support.prompt_delivery_cases import _profile
from tests.support.prompt_delivery_cases import _request

_JS_OVERRIDE = "Explain the JavaScript module for a reviewer, with concrete detail."

_PROFILE = {
    "single": {
        "common": {"requested_shape": {"description": "Common file summary."}},
        "per_extension": {".js": {"requested_shape": {"description": _JS_OVERRIDE}}},
    }
}

def _respond(prompt: str, verdict: str, rec: dict) -> str:
    if "standards/safety review" in prompt:
        rec["order"].append("review")
        rec["review_prompts"].append(prompt)
        review_id = next(
            line.split(": ", 1)[1] for line in prompt.splitlines()
            if line.startswith("review_id: ")
        )
        ordinal, count = next(
            line.split(": ", 1)[1].split("/", 1) for line in prompt.splitlines()
            if line.startswith("batch: ")
        )
        return json.dumps({
            "review_id": review_id,
            "batch_index": int(ordinal),
            "batch_count": int(count),
            "verdict": verdict,
            "reasons": ["blocking reason"] if verdict == "TOO_RISKY" else [],
            "warnings": ["please confirm"] if verdict == "RISKY" else [],
        })
    rec["order"].append("doc")
    rec["doc_prompts"].append(prompt)
    return json.dumps({"description": "documented"})

def _install_openai(monkeypatch, rec, verdict):
    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            prompt = kwargs["messages"][-1]["content"]
            return _OpenAIResponse(_respond(prompt, verdict, rec))

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = _Chat()

    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)

def _install_anthropic(monkeypatch, rec, verdict):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"]
            return _AnthropicResponse(_respond(prompt, verdict, rec))

    class Anthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)

def _install_gemini(monkeypatch, rec, verdict):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    gtypes.GenerateContentConfig = GenerateContentConfig

    class _Models:
        def generate_content(self, model, contents, config):
            return _GeminiResponse(_respond(contents, verdict, rec))

    class Client:
        def __init__(self, api_key=None):
            self.models = _Models()

    genai.Client = Client
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)

_PROVIDERS = [
    ("openai", _install_openai),
    ("anthropic", _install_anthropic),
    ("gemini", _install_gemini),
]

def _run(tmp_path, monkeypatch, provider, installer, verdict, confirm=None):
    rec = {"order": [], "review_prompts": [], "doc_prompts": []}
    installer(monkeypatch, rec, verdict)
    (tmp_path / "mod.js").write_text("const y = 2;\n", encoding="utf-8")
    config = {
        "llm_provider": provider,
        "api_key": "k",
        "model_name": "test-model",
        "entry_file": "mod.js",
        "documentation_scope": "all",
        "prompt_profiles": _PROFILE,
    }
    result = run_pipeline(tmp_path, config, confirm_risky=confirm)
    return rec, result

def test_safe_parity_across_providers(tmp_path, monkeypatch):
    shape_texts = []
    for provider, installer in _PROVIDERS:
        sub = tmp_path / provider
        sub.mkdir()
        rec, result = _run(sub, monkeypatch, provider, installer, "SAFE")
        # Review happened before any documentation call.
        assert rec["order"], f"{provider} made no provider calls"
        first_doc = rec["order"].index("doc")
        assert "review" not in rec["order"][first_doc:]
        # Exactly one review batch and one documentation call.
        assert len(rec["review_prompts"]) == 1
        assert len(rec["doc_prompts"]) == 1
        # The .js override text reached the documentation prompt.
        doc_prompt = rec["doc_prompts"][0]
        assert _JS_OVERRIDE in doc_prompt
        shape_texts.append(doc_prompt)
        assert result["checked"] == 1
    # Identical effective requested-shape text on every provider.
    assert len(set(shape_texts)) == 1

def test_too_risky_blocks_documentation_on_every_provider(tmp_path, monkeypatch):
    for provider, installer in _PROVIDERS:
        sub = tmp_path / provider
        sub.mkdir()
        with pytest.raises(PromptCustomizationValidationError, match="TOO RISKY"):
            _run(sub, monkeypatch, provider, installer, "TOO_RISKY")

def test_too_risky_makes_zero_documentation_calls(tmp_path, monkeypatch):
    for provider, installer in _PROVIDERS:
        sub = tmp_path / provider
        sub.mkdir()
        rec = {"order": [], "review_prompts": [], "doc_prompts": []}
        installer(monkeypatch, rec, "TOO_RISKY")
        (sub / "mod.js").write_text("const y = 2;\n", encoding="utf-8")
        config = {
            "llm_provider": provider, "api_key": "k", "model_name": "test-model",
            "entry_file": "mod.js", "documentation_scope": "all",
            "prompt_profiles": _PROFILE,
        }
        with pytest.raises(PromptCustomizationValidationError):
            run_pipeline(sub, config)
        assert rec["doc_prompts"] == []
        assert len(rec["review_prompts"]) == 1

def test_risky_requires_same_confirmation_contract_on_every_provider(tmp_path, monkeypatch):
    for provider, installer in _PROVIDERS:
        # Without confirmation -> blocked, zero documentation calls.
        blocked = tmp_path / f"{provider}-blocked"
        blocked.mkdir()
        rec = {"order": [], "review_prompts": [], "doc_prompts": []}
        installer(monkeypatch, rec, "RISKY")
        (blocked / "mod.js").write_text("const y = 2;\n", encoding="utf-8")
        config = {
            "llm_provider": provider, "api_key": "k", "model_name": "test-model",
            "entry_file": "mod.js", "documentation_scope": "all",
            "prompt_profiles": _PROFILE,
        }
        with pytest.raises(PromptCustomizationValidationError):
            run_pipeline(blocked, config)
        assert rec["doc_prompts"] == []

        # With explicit confirmation -> proceeds to documentation.
        ok = tmp_path / f"{provider}-ok"
        ok.mkdir()
        rec2, result = _run(ok, monkeypatch, provider, installer, "RISKY",
                            confirm=lambda _warnings: True)
        assert len(rec2["doc_prompts"]) == 1
        assert result["checked"] == 1

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
    from tests.support import providers as contracts

    rec = {}
    response = json.dumps({"description": "adapter result"})
    getattr(contracts, installer)(monkeypatch, rec, **{response_kw: response})
    provider = getattr(api_provider, class_name)(api_key="k")
    resolved = _profile({"single": {"fields": [
        {"key": "description", "type": "string", "instruction": "Custom desc"}]}})
    shape = resolved.resolve_block("combined", "a.py")
    result = Orchestrator(provider).process(_request(resolved, "x=1"))
    assert result["description"] == "adapter result"
    if class_name == "OpenAIProvider":
        sent = rec["create_kwargs"]["messages"][-1]["content"]
    elif class_name == "AnthropicProvider":
        sent = rec["create_kwargs"]["messages"][0]["content"]
    else:
        sent = rec["generate"]["contents"]
    assert shape.text in sent
