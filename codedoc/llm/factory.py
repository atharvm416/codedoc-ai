"""
LLM provider factory.

Reads the loaded config dict and returns the correct LLMProvider.
Adding a new provider only requires adding a branch here.
"""

from __future__ import annotations

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import ConfigError, LLMError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Known API model prefixes → provider class hint
_ANTHROPIC_PREFIXES = ("claude",)
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "text-")


def create_provider(config: dict) -> LLMProvider:
    """
    Instantiate and return an LLMProvider based on config.

    config keys used:
        llm_mode:    "local" | "api"
        model_name:  model identifier string
        api_key:     API key (for api mode)
        api_base_url: custom endpoint (for local mode or compatible APIs)
    """
    mode = config.get("llm_mode", "api")
    model = config.get("model_name", "")
    api_key = config.get("api_key") or ""
    base_url = config.get("api_base_url") or None

    if mode == "local":
        return _make_local(model, base_url)

    if mode == "api":
        return _make_api(model, api_key, base_url)

    raise ConfigError(f"Unknown llm_mode '{mode}'. Must be 'local' or 'api'.")


def _make_local(model: str, base_url: str | None) -> LLMProvider:
    from codedoc.llm.local_provider import LocalProvider, DEFAULT_OLLAMA_URL

    url = base_url or DEFAULT_OLLAMA_URL
    provider = LocalProvider(model=model or "qwen2.5-coder:7b", base_url=url)

    if not provider.is_available():
        raise LLMError(
            "Local",
            f"Local LLM server is not reachable at {url}. "
            "Make sure Ollama or LM Studio is running.",
        )
    return provider


def _make_api(model: str, api_key: str, base_url: str | None) -> LLMProvider:
    if not api_key:
        raise ConfigError(
            "API mode requires an API key. "
            "Set LLM_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) in your .env file."
        )

    model_lower = model.lower()

    # Anthropic Claude
    if any(model_lower.startswith(p) for p in _ANTHROPIC_PREFIXES):
        from codedoc.llm.api_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key, model=model)

    # OpenAI or compatible (default)
    from codedoc.llm.api_provider import OpenAIProvider
    return OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini", base_url=base_url)