"""Create the configured LLM provider.

Resolved ``provider_prefixes`` drive provider detection and API-key lookup;
module-level prefix constants remain compatibility fallbacks.

Active providers
----------------
  openai    — OpenAI and OpenAI-compatible endpoints (default)
  anthropic — Anthropic Claude
  gemini    — Google Gemini

An OpenAI-compatible local or self-hosted endpoint (e.g. Ollama, LM Studio) is
reached through ``OpenAIProvider`` plus ``api_base_url`` — there is no separate
local-provider choice.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import (
    CodeDocError,
    ConfigError,
    ProviderInitError,
    bounded_exception_summary,
)
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

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
}

_ENDPOINT_DEFAULT_SENTINEL = "provider-default"
_EXECUTION_DESCRIPTOR_ATTRIBUTE = "_codedoc_provider_execution_descriptor"


@dataclass(frozen=True, slots=True)
class ProviderExecutionDescriptor:
    """Non-secret identity inputs for one concrete provider construction."""

    provider_kind: str
    model: str
    endpoint_identity: str


def effective_endpoint_identity(api_base_url: object) -> str:
    """Return a non-secret identity for one effective OpenAI endpoint.

    Shared by the configured ``api_base_url`` and by a runtime endpoint-trust
    approval URL (see ``codedoc.core.loader``'s authorization gate) — both are
    canonicalized and rejected under exactly the same syntax rules, so an
    approval can never cover a differently-shaped URL than the one actually
    used.
    """
    if not isinstance(api_base_url, str) or not api_base_url.strip():
        return _ENDPOINT_DEFAULT_SENTINEL
    try:
        parts = urlsplit(api_base_url.strip())
        scheme = (parts.scheme or "").lower()
        host = parts.hostname
        port = parts.port
        username = parts.username
        password = parts.password
        query = parts.query
        fragment = parts.fragment
    except ValueError:
        raise ConfigError(
            "api_base_url must be a valid HTTP or HTTPS URL with a valid host and port."
        ) from None
    authority = parts.netloc.rsplit("@", 1)[-1]
    if (
        scheme not in ("http", "https")
        or not host
        or any(character.isspace() for character in host)
        or authority.endswith(":")
    ):
        raise ConfigError(
            "api_base_url must be a valid HTTP or HTTPS URL with a valid host and port."
        )
    if username is not None or password is not None or query or fragment:
        # These components fall outside the four-field canonical identity
        # (scheme/host/port/path) below, so a URL carrying one would silently
        # canonicalize identically to a "clean" URL missing it -- letting an
        # approval cover a differently-behaving endpoint. Reject instead of
        # silently dropping them.
        raise ConfigError(
            "api_base_url must not include a username, password, query string, "
            "or fragment."
        )
    if port is None:
        if scheme == "https":
            port = 443
        elif scheme == "http":
            port = 80
    normalized = {
        "scheme": scheme,
        "host": host.lower(),
        "port": port,
        # OpenAI's client treats a trailing slash as the same base endpoint:
        # it ensures exactly that separator before resolving resource paths.
        # Bind recovery to effective routing, not superficial URL spelling.
        "path": (parts.path or "").rstrip("/"),
    }
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Attribute name ``ResolvedConfig`` (``codedoc.core.loader``) uses to carry an
# ``EndpointTrustAttestation``.  Read here via ``getattr()`` rather than an
# ``isinstance`` check against ``ResolvedConfig``, so this module never needs to
# import ``codedoc.core.loader`` (which imports this module for
# ``effective_endpoint_identity`` and this attestation type) -- a plain
# ``dict`` simply has no such attribute and is therefore always unattested.
ENDPOINT_TRUST_ATTRIBUTE = "endpoint_trust"


@dataclass(frozen=True, slots=True)
class EndpointTrustAttestation:
    """Records that a custom ``api_base_url`` was authorized, and how.

    Constructed only by :func:`codedoc.core.loader.load_config` once its
    runtime endpoint-authorization gate succeeds, and carried as a
    non-serialized ``endpoint_trust`` attribute on the ``ResolvedConfig`` it
    returns -- never a dict key, so it cannot be forged through configuration
    and never appears in iteration, ``dict()`` conversion, serialization,
    persistence, cache identity, recovery identity, or provider identity.
    ``digest`` is the canonical :func:`effective_endpoint_identity` of the
    authorized endpoint; ``mechanism`` names which runtime approval source
    supplied it (``"--trust-api-base-url"`` or ``"CODEDOC_TRUST_API_BASE_URL"``).
    """

    digest: str
    mechanism: str


def describe_provider_selection(config: dict) -> tuple[str, str]:
    """Return the provider/model pair that :func:`create_provider` will use."""
    model = str(config.get("model_name", "") or "")
    selected = _resolve_api_provider(
        str(config.get("llm_provider", "auto")),
        model.lower(),
        config.get("provider_prefixes") or {},
    )
    return selected, model or _DEFAULT_MODELS[selected]


def describe_provider_execution(config: dict) -> ProviderExecutionDescriptor:
    """Describe exactly what :func:`create_provider` must construct."""
    selected, model = describe_provider_selection(config)
    endpoint = (
        effective_endpoint_identity(config.get("api_base_url"))
        if selected == "openai"
        else _ENDPOINT_DEFAULT_SENTINEL
    )
    return ProviderExecutionDescriptor(selected, model, endpoint)


def constructed_provider_execution(
    provider: LLMProvider,
) -> ProviderExecutionDescriptor | None:
    """Return the factory's construction attestation, if present."""
    descriptor = getattr(provider, _EXECUTION_DESCRIPTOR_ATTRIBUTE, None)
    return descriptor if isinstance(descriptor, ProviderExecutionDescriptor) else None


def attest_provider_execution(
    provider: object,
    config: dict,
) -> object:
    """Attach an explicit non-secret execution descriptor to an injected provider.

    The production factory attests its own concrete branch automatically.
    Alternate provider factories and network-free test doubles must call this
    helper explicitly; verification never infers an attestation from config.
    """
    setattr(
        provider,
        _EXECUTION_DESCRIPTOR_ATTRIBUTE,
        describe_provider_execution(config),
    )
    return provider


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
    base_url = config.get("api_base_url") or None

    # Endpoint-authorization verification -- resolved before any credential is
    # read or selected.  A non-empty api_base_url requires a present
    # EndpointTrustAttestation whose digest matches this exact endpoint;
    # config is a plain dict for every legacy/test caller, which simply has no
    # ``endpoint_trust`` attribute and is therefore always rejected here when it
    # carries a custom endpoint -- create_provider(custom_plain_dict) cannot
    # bypass authorization.  An empty api_base_url is the default provider
    # endpoint and requires no attestation.
    if base_url:
        expected_digest = effective_endpoint_identity(base_url)
        attestation = getattr(config, ENDPOINT_TRUST_ATTRIBUTE, None)
        if (
            not isinstance(attestation, EndpointTrustAttestation)
            or attestation.digest != expected_digest
        ):
            raise ProviderInitError(
                "api_base_url is set to a custom endpoint that was not authorized "
                "at runtime. Approve the exact endpoint using --trust-api-base-url "
                "or CODEDOC_TRUST_API_BASE_URL, then load configuration through "
                "load_config() so the authorization can be verified."
            )

    api_key = config.get("api_key") or _provider_api_key(provider, model, provider_prefixes)

    if mode == "api":
        # Provider-initialization error boundary.  Construction, import,
        # and auth-configuration failures from provider SDKs are classified as
        # ProviderInitError (a ConfigError subclass → CLI exit code 2).
        try:
            expected = describe_provider_execution(config)
            result = _make_api(
                provider, model, api_key, base_url, provider_prefixes
            )
            actual = constructed_provider_execution(result)
            if actual != expected:
                raise ProviderInitError(
                    "LLM provider construction identity did not match the "
                    "resolved provider/model/effective-endpoint plan."
                )
            return result
        except ConfigError:
            raise
        except Exception as exc:
            # A wrapped CodeDocError (e.g. an LLMError raised by a provider's
            # own __init__ for a missing SDK package) is already bounded by
            # construction and renders unchanged (tier 1). Anything else is
            # reduced (tier 2); "unknown-error" is replaced with the more
            # specific "provider-initialization-failed" for this construction
            # context, since every fault reaching here occurred while building
            # the provider, not while it was already in use.
            if isinstance(exc, CodeDocError):
                detail = str(exc)
            else:
                detail = bounded_exception_summary(exc)
                if detail == "unknown-error":
                    detail = "provider-initialization-failed"
            raise ProviderInitError(
                f"LLM provider initialization failed: {detail}"
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
            "(or GOOGLE_API_KEY) as an environment variable, or pass "
            "LLM_API_KEY as a generic fallback."
        )

    model_lower = model.lower()
    selected = _resolve_api_provider(provider, model_lower, provider_prefixes)
    effective_model = model or _DEFAULT_MODELS[selected]

    if selected == "anthropic":
        from codedoc.llm.api_provider import AnthropicProvider

        result = AnthropicProvider(api_key=api_key, model=effective_model)
        endpoint_identity = _ENDPOINT_DEFAULT_SENTINEL
    elif selected == "gemini":
        from codedoc.llm.api_provider import GeminiProvider

        result = GeminiProvider(api_key=api_key, model=effective_model)
        endpoint_identity = _ENDPOINT_DEFAULT_SENTINEL
    else:
        # OpenAI or compatible endpoint (default)
        from codedoc.llm.api_provider import OpenAIProvider

        result = OpenAIProvider(
            api_key=api_key,
            model=effective_model,
            base_url=base_url,
        )
        endpoint_identity = effective_endpoint_identity(base_url)

    setattr(
        result,
        _EXECUTION_DESCRIPTOR_ATTRIBUTE,
        ProviderExecutionDescriptor(
            provider_kind=selected,
            model=effective_model,
            endpoint_identity=endpoint_identity,
        ),
    )
    return result


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
