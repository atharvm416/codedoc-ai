"""
LLM provider factory.

Reads the loaded config dict and returns the correct LLMProvider.
Adding a new provider only requires adding a branch here.

0.8.1 changes
-------------
- ``_ANTHROPIC_PREFIXES``, ``_GEMINI_PREFIXES``, and ``_OPENAI_PREFIXES`` are
  kept as module-level constants for backward compatibility and as fallbacks,
  but the authoritative values now live in
  ``DEFAULTS["provider_prefixes"]`` in ``loader.py``.
- ``create_provider()`` passes the resolved ``config["provider_prefixes"]``
  dict through to ``_resolve_api_provider()`` and ``_provider_api_key()``
  so that provider auto-detection and API-key lookup use the same source of
  truth.

Active providers
----------------
  openai    — OpenAI and OpenAI-compatible endpoints (default)
  anthropic — Anthropic Claude
  gemini    — Google Gemini

Reserved (not exposed in this release)
---------------------------------------
  ``codedoc.llm.local_provider`` remains importable for compatibility, but the
  factory and CLI do not expose a local-provider choice.
"""

from __future__ import annotations

import os

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import ConfigError, ProviderInitError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level prefix tuples kept as fallback defaults for callers that do not
# pass provider_prefixes from config (e.g. direct tests of the factory).
_ANTHROPIC_PREFIXES = ("claude",)
_GEMINI_PREFIXES = ("gemini",)
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "text-")

# Build the fallback dict once for use in _resolve_api_provider.
_FALLBACK_PREFIXES: dict[str, list[str]] = {
    "anthropic": list(_ANTHROPIC_PREFIXES),
    "gemini":    list(_GEMINI_PREFIXES),
    "openai":    list(_OPENAI_PREFIXES),
}


def create_provider(config: dict) -> LLMProvider:
    """
    Instantiate and return an LLMProvider based on config.

    config keys used:
        llm_mode:         must be "api"
        llm_provider:     "auto" | "openai" | "anthropic" | "gemini"
        model_name:       model identifier string
        api_key:          API key (resolved from env if not supplied)
        api_base_url:     custom OpenAI-compatible endpoint (optional)
        provider_prefixes: dict[str, list[str]] for model-name auto-detection
    """
    mode = config.get("llm_mode", "api")
    provider = config.get("llm_provider", "auto")
    model = config.get("model_name", "")
    provider_prefixes: dict[str, list[str]] = config.get("provider_prefixes") or {}
    api_key = config.get("api_key") or _provider_api_key(provider, model, provider_prefixes)
    base_url = config.get("api_base_url") or None

    if mode == "api":
        # 0.9.2: provider-initialization error boundary.  Construction, import,
        # and auth-configuration failures from provider SDKs are classified as
        # ProviderInitError (a ConfigError subclass → CLI exit code 2).
        try:
            return _make_api(provider, model, api_key, base_url, provider_prefixes)
        except ConfigError:
            raise
        except Exception as exc:
            raise ProviderInitError(
                f"LLM provider initialization failed: {exc}"
            ) from exc

    raise ConfigError(
        f"Unsupported llm_mode '{mode}'. The only supported mode is 'api'."
    )


def _make_api(
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None,
    provider_prefixes: dict[str, list[str]] | None = None,
) -> LLMProvider:
    if not api_key:
        raise ConfigError(
            "API mode requires an API key. "
            "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY "
            "(or GOOGLE_API_KEY) in your .env file, or pass LLM_API_KEY "
            "as a generic fallback."
        )

    model_lower = model.lower()
    selected = _resolve_api_provider(provider, model_lower, provider_prefixes)

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


def _resolve_api_provider(
    provider: str,
    model_lower: str,
    provider_prefixes: dict[str, list[str]] | None = None,
) -> str:
    """Resolve a provider name, applying auto-detection from the model name.

    When *provider* is ``"auto"``, the model name is checked against the
    resolved prefixes from ``config["provider_prefixes"]`` (passed as
    *provider_prefixes*).  Falls back to the module-level ``_FALLBACK_PREFIXES``
    when *provider_prefixes* is not supplied so that direct test callers work
    without a full config dict.
    """
    if provider in ("openai", "anthropic", "gemini"):
        return provider
    if provider != "auto":
        raise ConfigError(
            "llm_provider must be one of: 'auto', 'openai', 'anthropic', or 'gemini'."
        )

    # Use config prefixes when provided, else module-level fallbacks.
    prefixes = provider_prefixes if provider_prefixes else _FALLBACK_PREFIXES

    # Check anthropic first, then gemini; anything else defaults to openai.
    for provider_name in ("anthropic", "gemini"):
        if any(model_lower.startswith(p) for p in prefixes.get(provider_name, [])):
            return provider_name
    return "openai"


def _provider_api_key(
    provider: str,
    model: str,
    provider_prefixes: dict[str, list[str]] | None = None,
) -> str:
    """Resolve the API key from environment variables for a given provider.

    Uses the same *provider_prefixes* passed to :func:`_resolve_api_provider`
    so that model auto-detection and API-key lookup are always consistent.
    """
    selected = _resolve_api_provider(provider, model.lower(), provider_prefixes)
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
