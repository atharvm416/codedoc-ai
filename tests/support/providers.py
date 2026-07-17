"""Shared fake response shapes, fake SDK installers, and the ``SmartFake``
combined provider double, relocated from ``tests/test_095_provider_contracts.py``
and ``tests/test_110_prompt_profile_cli.py`` for reuse across collected test
modules without importing one collected test module from another.

Symbols keep their exact pre-relocation names and bodies (including the leading
underscore on the installers): pytest derives parametrized node IDs from these
names, so renaming them would silently change collected test IDs.
"""
from __future__ import annotations

import json
import sys
import types


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
            rec.setdefault("create_calls", []).append(kwargs)
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
            rec.setdefault("create_calls", []).append(kwargs)
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
            rec.setdefault("generate_calls", []).append(rec["generate"])
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


# ---------------------------------------------------------------------------
# Combined provider double
# ---------------------------------------------------------------------------

class SmartFake:
    """Serves both the review verdict and documentation responses."""

    provider_name = "fake"

    def __init__(self, verdict="SAFE"):
        self.verdict = verdict
        self.review_calls = 0
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            review_id = next(
                line.split(": ", 1)[1]
                for line in prompt.splitlines()
                if line.startswith("review_id: ")
            )
            ordinal, count = next(
                line.split(": ", 1)[1].split("/", 1)
                for line in prompt.splitlines()
                if line.startswith("batch: ")
            )
            return json.dumps({
                "review_id": review_id,
                "batch_index": int(ordinal),
                "batch_count": int(count),
                "verdict": self.verdict,
                "reasons": ["bad"] if self.verdict == "TOO_RISKY" else [],
                "warnings": ["warn"] if self.verdict == "RISKY" else [],
            })
        self.doc_calls += 1
        if "Analyse the imports" in prompt:
            # A dependency agent can validly report that this file has no
            # dependencies; use the exact dependency response shape rather than
            # feeding it the combined-agent fixture below.
            return json.dumps({
                "dependencies_analysis": {
                    "internal": [],
                    "external": [],
                    "dependency_refs": [],
                    "catalog_updates": [],
                    "usage_notes": [],
                    "warnings": [],
                }
            })
        return json.dumps({
            "description": "A file.", "role_in_system": "core",
            "functions": [{"name": "f", "description": "does f"}],
            "key_concepts": ["kc"], "usage_example": "import x",
        })

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)
