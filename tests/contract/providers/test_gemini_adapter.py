"""Tests organized by feature ownership."""

from __future__ import annotations

import sys
import types
from tests.support.providers import _install_gemini
from tests.support.provider_contract_cases import _JSON_HINT
from tests.support.provider_contract_cases import _make

class TestAPIProviders:
    def test_gemini_provider_uses_generate_content(self, monkeypatch):
        from codedoc.llm.api_provider import GeminiProvider

        captured = {}

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text='{"ok": true}')

        class FakeClient:
            def __init__(self, api_key):
                captured["api_key"] = api_key
                self.models = FakeModels()

        fake_types = types.SimpleNamespace(
            GenerateContentConfig=FakeGenerateContentConfig
        )
        fake_genai = types.SimpleNamespace(Client=FakeClient, types=fake_types)
        fake_google = types.SimpleNamespace(genai=fake_genai)

        monkeypatch.setitem(sys.modules, "google", fake_google)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

        provider = GeminiProvider(api_key="gemini-key", model="gemini-test")
        result = provider.complete_json("Return JSON")

        assert result == '{"ok": true}'
        assert captured["api_key"] == "gemini-key"
        assert captured["model"] == "gemini-test"
        assert captured["config"].kwargs["temperature"] == 0.0
        assert captured["config"].kwargs["response_mime_type"] == "application/json"

def test_gemini_sends_json_mime_type_and_system_instruction(monkeypatch):
    rec = {}
    _install_gemini(monkeypatch, rec)
    provider = _make("GeminiProvider")(api_key="k")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    config = rec["generate"]["config"]
    assert config.response_mime_type == "application/json"
    assert _JSON_HINT in config.system_instruction
