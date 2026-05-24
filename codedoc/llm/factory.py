"""
LLM provider factory.

Reads the loaded config dict and returns the correct LLMProvider.
Adding a new provider only requires adding a branch here.

Active providers
----------------
  openai    — OpenAI and OpenAI-compatible endpoints (default)
  anthropic — Anthropic Claude
  gemini    — Google Gemini

Reserved (not exposed in this release)
---------------------------------------
  local     — Ollama / LM Studio via OpenAI-compatible endpoint.
              The implementation lives in local_provider.py and can be
              re-enabled by restoring the commented branch below.
"""

from __future__ import annotations

import os

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import ConfigError, LLMError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Known API model prefixes → provider class hint
_ANTHROPIC_PREFIXES = ("claude",)
_GEMINI_PREFIXES = ("gemini",)
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "text-")


def create_provider(config: dict) -> LLMProvider:
    """
    Instantiate and return an LLMProvider based on config.

    config keys used:
        llm_mode:     must be "api"
        llm_provider: "auto" | "openai" | "anthropic" | "gemini"
        model_name:   model identifier string
        api_key:      API key (resolved from env if not supplied)
        api_base_url: custom OpenAI-compatible endpoint (optional)
    """
    mode = config.get("llm_mode", "api")
    provider = config.get("llm_provider", "auto")
    model = config.get("model_name", "")
    api_key = config.get("api_key") or _provider_api_key(provider, model)
    base_url = config.get("api_base_url") or None

    # ------------------------------------------------------------------
    # Local mode is reserved — not active in this release.
    # To re-enable, uncomment the block below and restore the import at
    # the bottom of this function.
    #
    # if mode == "local":
    #     return _make_local(model, base_url)
    # ------------------------------------------------------------------

    if mode == "api":
        return _make_api(provider, model, api_key, base_url)

    raise ConfigError(
        f"Unsupported llm_mode '{mode}'. The only supported mode is 'api'."
    )


def _make_api(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
) -> LLMProvider:
    if not api_key:
        raise ConfigError(
            "API mode requires an API key. "
            "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY "
            "(or GOOGLE_API_KEY) in your .env file, or pass LLM_API_KEY "
            "as a generic fallback."
        )

    model_lower = model.lower()
    selected = _resolve_api_provider(provider, model_lower)

    if selected == "anthropic":
        from codedoc.llm.api_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-haiku-4-5-20251001",
        )

    if selected == "gemini":
        from codedoc.llm.api_provider import GeminiProvider
        return GeminiProvider(
            api_key=api_key,
            model=model or "gemini-2.5-flash",
        )

    # OpenAI or compatible endpoint (default)
    from codedoc.llm.api_provider import OpenAIProvider
    return OpenAIProvider(
        api_key=api_key,
        model=model or "gpt-4o-mini",
        base_url=base_url,
    )


def _resolve_api_provider(provider: str, model_lower: str) -> str:
    """Resolve a provider name, applying auto-detection from the model name."""
    if provider in ("openai", "anthropic", "gemini"):
        return provider
    if provider != "auto":
        raise ConfigError(
            "llm_provider must be one of: 'auto', 'openai', 'anthropic', or 'gemini'."
        )
    if any(model_lower.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic"
    if any(model_lower.startswith(p) for p in _GEMINI_PREFIXES):
        return "gemini"
    return "openai"


def _provider_api_key(provider: str, model: str) -> str:
    """Resolve the API key from environment variables for a given provider."""
    selected = _resolve_api_provider(provider, model.lower())
    if selected == "anthropic":
        return (
            os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("LLM_API_KEY", "")
        )
    if selected == "gemini":
        return (
            os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("LLM_API_KEY", "")
        )
    # OpenAI / compatible
    return (
        os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("LLM_API_KEY", "")
    )


# ---------------------------------------------------------------------------
# Reserved — local provider helper (not called in this release).
#
# def _make_local(model: str, base_url: str | None) -> LLMProvider:
#     from codedoc.llm.local_provider import LocalProvider, DEFAULT_OLLAMA_URL
#     url = base_url or DEFAULT_OLLAMA_URL
#     provider = LocalProvider(model=model or "qwen2.5-coder:7b", base_url=url)
#     if not provider.is_available():
#         raise LLMError(
#             "Local",
#             f"Local LLM server is not reachable at {url}. "
#             "Make sure Ollama or LM Studio is running.",
#         )
#     return provider
# ---------------------------------------------------------------------------
