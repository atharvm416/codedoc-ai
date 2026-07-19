"""Tests organized by feature ownership."""

from __future__ import annotations

import sys
import types
from tests.support.providers import _install_openai
from tests.support.provider_contract_cases import _make

class TestAPIProviders:
    def test_openai_provider_uses_chat_completions(self, monkeypatch):
        from codedoc.llm.api_provider import OpenAIProvider

        captured = {}

        class FakeMessage:
            content = '{"ok": true}'

        class FakeChoice:
            message = FakeMessage()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(choices=[FakeChoice()])

        class FakeOpenAI:
            def __init__(self, api_key, base_url=None):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions()
                )

        monkeypatch.setitem(
            sys.modules,
            "openai",
            types.SimpleNamespace(OpenAI=FakeOpenAI),
        )

        provider = OpenAIProvider(api_key="openai-key", model="gpt-test")
        result = provider.complete_json("Return JSON")

        assert result == '{"ok": true}'
        assert captured["api_key"] == "openai-key"
        assert captured["model"] == "gpt-test"
        assert captured["temperature"] == 0.0

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
