"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.configuration_cases import _load

def test_C12_provider_prefixes_auto_detect_anthropic(tmp_path):
    """C12a: A model starting with 'claude' is auto-detected as anthropic."""
    from codedoc.llm.factory import _resolve_api_provider
    from codedoc.core.loader import DEFAULTS

    prefixes = DEFAULTS["provider_prefixes"]
    assert _resolve_api_provider("auto", "claude-haiku-4-5", prefixes) == "anthropic"

def test_C12_provider_prefixes_auto_detect_gemini(tmp_path):
    """C12b: A model starting with 'gemini' is auto-detected as gemini."""
    from codedoc.llm.factory import _resolve_api_provider
    from codedoc.core.loader import DEFAULTS

    prefixes = DEFAULTS["provider_prefixes"]
    assert _resolve_api_provider("auto", "gemini-2.5-flash", prefixes) == "gemini"

def test_C12_provider_prefixes_custom_prefix(tmp_path):
    """C12c: A custom prefix in provider_prefixes is used for auto-detection."""
    from codedoc.llm.factory import _resolve_api_provider

    custom_prefixes = {
        "anthropic": ["claude"],
        "gemini":    ["gemini"],
        "openai":    ["gpt-", "o1", "o3", "text-", "mycompany-"],
    }
    assert _resolve_api_provider("auto", "mycompany-model-v2", custom_prefixes) == "openai"

def test_C12_remove_prefix_changes_detection(tmp_path):
    """C12d: Removing a prefix via provider_prefixes_remove changes auto-detection."""
    cfg = _load(tmp_path, entry_file="main.py",
                provider_prefixes_remove={"openai": ["o1"]})

    from codedoc.llm.factory import _resolve_api_provider

    prefixes = cfg["provider_prefixes"]
    # "o1-mini" no longer has the "o1" prefix → should fall through to openai (default)
    # but not because of the "o1" prefix — only if "gpt-" etc. match
    assert "o1" not in prefixes["openai"], "o1 prefix must be removed"
    # "o1-mini" still falls through to openai as the default
    assert _resolve_api_provider("auto", "o1-mini", prefixes) == "openai"

def test_C13_api_key_lookup_consistent_with_detection(tmp_path, monkeypatch):
    """C13: _provider_api_key uses the same prefixes as _resolve_api_provider."""
    from codedoc.llm.factory import _provider_api_key, _resolve_api_provider

    custom_prefixes = {
        "anthropic": ["myclaude"],  # custom prefix
        "gemini":    ["gemini"],
        "openai":    ["gpt-"],
    }

    # Simulate env: anthropic key is set
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")

    provider = _resolve_api_provider("auto", "myclaude-v3", custom_prefixes)
    assert provider == "anthropic"

    key = _provider_api_key("auto", "myclaude-v3", custom_prefixes)
    assert key == "sk-test-anthropic", (
        "_provider_api_key must use the same prefixes and return the anthropic key"
    )

class TestProviderFactory:
    def test_selects_openai_provider_explicitly(self, monkeypatch):
        from codedoc.llm.factory import create_provider

        captured = {}

        class FakeOpenAIProvider:
            def __init__(self, api_key, model, base_url=None):
                captured.update(
                    {"api_key": api_key, "model": model, "base_url": base_url}
                )

        monkeypatch.setattr("codedoc.llm.api_provider.OpenAIProvider", FakeOpenAIProvider)

        provider = create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "openai",
                "model_name": "gpt-4o-mini",
                "api_key": "openai-key",
            }
        )

        assert isinstance(provider, FakeOpenAIProvider)
        assert captured["api_key"] == "openai-key"
        assert captured["model"] == "gpt-4o-mini"

    def test_selects_anthropic_provider_explicitly(self, monkeypatch):
        from codedoc.llm.factory import create_provider

        captured = {}

        class FakeAnthropicProvider:
            def __init__(self, api_key, model):
                captured.update({"api_key": api_key, "model": model})

        monkeypatch.setattr(
            "codedoc.llm.api_provider.AnthropicProvider",
            FakeAnthropicProvider,
        )

        provider = create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "anthropic",
                "model_name": "claude-haiku-4-5-20251001",
                "api_key": "anthropic-key",
            }
        )

        assert isinstance(provider, FakeAnthropicProvider)
        assert captured["api_key"] == "anthropic-key"
        assert captured["model"] == "claude-haiku-4-5-20251001"

    def test_selects_gemini_provider_explicitly(self, monkeypatch):
        from codedoc.llm.factory import create_provider

        captured = {}

        class FakeGeminiProvider:
            def __init__(self, api_key, model):
                captured.update({"api_key": api_key, "model": model})

        monkeypatch.setattr("codedoc.llm.api_provider.GeminiProvider", FakeGeminiProvider)

        provider = create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "gemini",
                "model_name": "gemini-2.5-flash",
                "api_key": "gemini-key",
            }
        )

        assert isinstance(provider, FakeGeminiProvider)
        assert captured["api_key"] == "gemini-key"
        assert captured["model"] == "gemini-2.5-flash"

    def test_auto_selects_gemini_from_model_name(self, monkeypatch):
        from codedoc.llm.factory import create_provider

        class FakeGeminiProvider:
            def __init__(self, api_key, model):
                self.api_key = api_key
                self.model = model

        monkeypatch.setattr("codedoc.llm.api_provider.GeminiProvider", FakeGeminiProvider)

        provider = create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "auto",
                "model_name": "gemini-2.5-flash",
                "api_key": "gemini-key",
            }
        )

        assert isinstance(provider, FakeGeminiProvider)
        assert provider.model == "gemini-2.5-flash"

    def test_openai_compatible_endpoint_reaches_openai_provider_with_base_url(
        self, monkeypatch
    ):
        """An OpenAI-compatible local/self-hosted endpoint stays reachable only
        through OpenAIProvider plus api_base_url — there is no separate
        local-provider choice."""
        from codedoc.llm.factory import create_provider

        captured = {}

        class FakeOpenAIProvider:
            def __init__(self, api_key, model, base_url=None):
                captured.update(
                    {"api_key": api_key, "model": model, "base_url": base_url}
                )

        monkeypatch.setattr("codedoc.llm.api_provider.OpenAIProvider", FakeOpenAIProvider)

        provider = create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "openai",
                "model_name": "qwen2.5-coder:7b",
                "api_key": "ollama",
                "api_base_url": "http://localhost:11434/v1",
            }
        )

        assert isinstance(provider, FakeOpenAIProvider)
        assert captured["base_url"] == "http://localhost:11434/v1"
        assert captured["model"] == "qwen2.5-coder:7b"
