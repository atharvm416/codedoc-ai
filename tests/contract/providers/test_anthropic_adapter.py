"""Tests organized by feature ownership."""

from __future__ import annotations

import sys
import types
from tests.support.providers import _install_anthropic
from tests.support.provider_contract_cases import _JSON_HINT
from tests.support.provider_contract_cases import _make

class TestAPIProviders:
    def test_anthropic_provider_uses_messages_api(self, monkeypatch):
        from codedoc.llm.api_provider import AnthropicProvider

        captured = {}

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(text='{"ok": true}')]
                )

        class FakeAnthropicClient:
            def __init__(self, api_key, timeout=None, max_retries=None):
                captured["api_key"] = api_key
                captured["timeout"] = timeout
                captured["max_retries"] = max_retries
                self.messages = FakeMessages()

        monkeypatch.setitem(
            sys.modules,
            "anthropic",
            types.SimpleNamespace(Anthropic=FakeAnthropicClient),
        )

        provider = AnthropicProvider(api_key="anthropic-key", model="claude-test")
        result = provider.complete_json("Return JSON")

        assert result == '{"ok": true}'
        assert captured["api_key"] == "anthropic-key"
        assert captured["model"] == "claude-test"
        assert captured["temperature"] == 0.0

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
