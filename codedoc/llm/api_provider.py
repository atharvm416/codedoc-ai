"""
API-based LLM providers.

Supports:
  - OpenAI  (and any OpenAI-compatible endpoint: LiteLLM, Together, Groq, etc.)
  - Anthropic Claude
  - Google Gemini

All three providers follow the same contract:
  - Client is instantiated once in __init__ and reused across all calls.
  - complete()      → free-text response at a given temperature.
  - complete_json() → JSON-only response using each provider's native JSON mode
                      where available, with a text-instruction fallback.
  - provider_name   → human-readable string used in logs.
"""

from __future__ import annotations

import math

import httpx
import requests

from codedoc.llm.base import LLMProvider
from codedoc.llm.model_capabilities import anthropic_capability, openai_omits_temperature
from codedoc.utils.errors import LLMError, ProviderFailureEnvelope, detect_limit_type
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Instruction appended to every JSON request (used by all providers).
_JSON_INSTRUCTION = (
    "Respond ONLY with valid JSON. No markdown fences, no explanation, no preamble."
)

# Safe default for direct library construction (e.g. a test that never
# passes timeout explicitly), matching codedoc.core.loader's own default for
# provider_request_timeout_s. A real run always passes the resolved config
# value explicitly through the factory (section 5.5).
_DEFAULT_PROVIDER_REQUEST_TIMEOUT_S = 120.0

# FIX-4 (section 5.5): CodeDoc's own default endpoint for each provider,
# always passed explicitly to its SDK client so transport can never silently
# fall back to that SDK's own environment-variable override (OPENAI_BASE_URL,
# ANTHROPIC_BASE_URL, GOOGLE_GEMINI_BASE_URL, or a Vertex switch). These are
# the exact strings each installed pinned SDK already falls back to
# internally when nothing else is configured, so passing them explicitly in
# a clean environment produces byte-identical transport to today's implicit
# default.
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
_GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/"


class _CodeDocDefaultTemperature(float):
    """Distinguish an omitted public argument from an explicit ``0.1``."""


_UNSUPPLIED_TEMPERATURE = _CodeDocDefaultTemperature(0.1)


# ---------------------------------------------------------------------------
# Provider-failure envelope construction (section 5.3 / 11.1)
#
# Each classifier below is invoked only from its adapter's own immediate
# exception boundary. It resolves every SDK exception class defensively via
# ``getattr(sdk, name, None)`` and skips the corresponding isinstance check
# when the lookup is ``None`` -- required so the adapter tolerates the
# minimal fake SDK modules used throughout the test suite (which define only
# a client class, no exception hierarchy). It reads only allowlisted
# primitive fields from ``vars(exc)``; the sole ordinary attribute access is
# ``response.headers`` on the object obtained from ``vars(exc)["response"]``.
# No property, descriptor, or callable on the exception itself is invoked.
# ---------------------------------------------------------------------------


def _retry_after_seconds(response: object) -> float | None:
    """Parse a plain-seconds ``Retry-After`` header off *response*.

    Only ordinary attribute access permitted by section 5.3.  No HTTP-date
    parsing -- section 5.7's accepted OpenAI/Anthropic-only plain-seconds
    limitation.
    """
    try:
        headers = getattr(response, "headers", None)
    except Exception:  # noqa: BLE001 - foreign response containers are untrusted
        return None
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except Exception:  # noqa: BLE001 - malformed/foreign header containers are ignored
        return None
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _normalize_status(value: object) -> int | None:
    """Coerce a raw SDK ``status_code``/``code`` field to a plain int, or
    ``None`` when it is not one.

    A malformed value (string, float, bool, list, ...) must never reach
    :class:`ProviderFailureEnvelope` construction: its own validation
    rejects anything but a non-boolean int, and that rejection is a raw
    ``ValueError`` escaping the adapter boundary, not the bounded
    ``LLMError`` section 5.3 requires. Normalizing here means every
    downstream numeric comparison and every construction site already sees
    a safe value.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _canonical_reason(envelope: ProviderFailureEnvelope) -> str:
    return (
        envelope.reason_code
        if envelope.status is None
        else f"{envelope.reason_code} ({envelope.status})"
    )


def _promote_via_configured_signals(
    exc: BaseException,
    envelope: ProviderFailureEnvelope,
    signals: tuple[str, ...],
) -> ProviderFailureEnvelope:
    """Promote an otherwise-unmapped failure to rate-limited via the
    caller's configured rate-limit text signals (section 5.3's final rule).

    Only ever promotes ``provider-request-failed`` -- a failure the closed
    structural table above could not classify -- and never overrides a
    structured reason.  *signals* are pre-lowercased by
    :func:`codedoc.llm.rate_limit_profile.get_rate_limit_profile`; the
    message text is used only transiently to decide the promotion and the
    structured ``limit_type``, never stored on the envelope as raw text.
    """
    if envelope.reason_code != "provider-request-failed" or not signals:
        return envelope
    message = str(exc).lower()
    if not any(signal in message for signal in signals):
        return envelope
    return ProviderFailureEnvelope(
        provider_kind=envelope.provider_kind,
        reason_code="provider-rate-limited",
        status=envelope.status,
        retry_after_s=envelope.retry_after_s,
        limit_type=detect_limit_type(str(exc)),
    )


def _openai_failure_envelope(exc: BaseException, sdk: object) -> ProviderFailureEnvelope:
    data = vars(exc)
    status = _normalize_status(data.get("status_code"))
    code = data.get("code")
    response = data.get("response")

    validation_cls = getattr(sdk, "APIResponseValidationError", None)
    if validation_cls is not None and isinstance(exc, validation_cls):
        return ProviderFailureEnvelope(
            provider_kind="openai", reason_code="provider-response-malformed"
        )

    timeout_cls = getattr(sdk, "APITimeoutError", None)
    if (timeout_cls is not None and isinstance(exc, timeout_cls)) or isinstance(
        exc, httpx.TimeoutException
    ):
        return ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-timeout")

    connection_cls = getattr(sdk, "APIConnectionError", None)
    if (
        (connection_cls is not None and isinstance(exc, connection_cls))
        or isinstance(exc, httpx.TransportError)
        or isinstance(exc, requests.exceptions.RequestException)
    ):
        return ProviderFailureEnvelope(
            provider_kind="openai", reason_code="provider-connection-failed"
        )

    retry_after_s = _retry_after_seconds(response)

    if status == 429 and code == "insufficient_quota":
        return ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-quota-exhausted",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status in (400, 413, 422):
        return ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-input-rejected",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status in (401, 403):
        return ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-authentication-rejected",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status == 404:
        return ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-model-unavailable",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status == 429:
        return ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-rate-limited",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status == 529:
        return ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-rate-limited",
            status=status,
            retry_after_s=retry_after_s,
            limit_type="overloaded",
        )
    return ProviderFailureEnvelope(
        provider_kind="openai",
        reason_code="provider-request-failed",
        status=status,
        retry_after_s=retry_after_s,
    )


_ANTHROPIC_TYPE_REASON_CODES: dict[str, str] = {
    "billing_error": "provider-quota-exhausted",
    "overloaded_error": "provider-rate-limited",
    "timeout_error": "provider-timeout",
    "authentication_error": "provider-authentication-rejected",
    "permission_error": "provider-authentication-rejected",
    "not_found_error": "provider-model-unavailable",
    "rate_limit_error": "provider-rate-limited",
    "invalid_request_error": "provider-input-rejected",
    "api_error": "provider-request-failed",
}


def _anthropic_failure_envelope(exc: BaseException, sdk: object) -> ProviderFailureEnvelope:
    data = vars(exc)
    status = _normalize_status(data.get("status_code"))
    error_type = data.get("type")
    response = data.get("response")

    validation_cls = getattr(sdk, "APIResponseValidationError", None)
    if validation_cls is not None and isinstance(exc, validation_cls):
        return ProviderFailureEnvelope(
            provider_kind="anthropic", reason_code="provider-response-malformed"
        )

    timeout_cls = getattr(sdk, "APITimeoutError", None)
    if (timeout_cls is not None and isinstance(exc, timeout_cls)) or isinstance(
        exc, httpx.TimeoutException
    ):
        return ProviderFailureEnvelope(provider_kind="anthropic", reason_code="provider-timeout")

    connection_cls = getattr(sdk, "APIConnectionError", None)
    if (
        (connection_cls is not None and isinstance(exc, connection_cls))
        or isinstance(exc, httpx.TransportError)
        or isinstance(exc, requests.exceptions.RequestException)
    ):
        return ProviderFailureEnvelope(
            provider_kind="anthropic", reason_code="provider-connection-failed"
        )

    retry_after_s = _retry_after_seconds(response)

    reason_code = (
        _ANTHROPIC_TYPE_REASON_CODES.get(error_type)
        if isinstance(error_type, str)
        else None
    )
    if reason_code is not None:
        limit_type = "overloaded" if error_type == "overloaded_error" else None
        return ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code=reason_code,
            status=status,
            retry_after_s=retry_after_s,
            limit_type=limit_type,
        )

    # Never rely on RequestTooLargeError/OverloadedError being top-level
    # (section 11.1): the anthropic SDK does not export those specific
    # status-bound subclasses, so HTTP-equivalent fallback is by numeric
    # status_code alone, not isinstance.
    if status in (400, 413, 422):
        return ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code="provider-input-rejected",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status == 529:
        return ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code="provider-rate-limited",
            status=status,
            retry_after_s=retry_after_s,
            limit_type="overloaded",
        )
    if status in (401, 403):
        return ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code="provider-authentication-rejected",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status == 404:
        return ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code="provider-model-unavailable",
            status=status,
            retry_after_s=retry_after_s,
        )
    if status == 429:
        return ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code="provider-rate-limited",
            status=status,
            retry_after_s=retry_after_s,
        )
    return ProviderFailureEnvelope(
        provider_kind="anthropic",
        reason_code="provider-request-failed",
        status=status,
        retry_after_s=retry_after_s,
    )


_GEMINI_STATUS_REASON_CODES: dict[str, str] = {
    "RESOURCE_EXHAUSTED": "provider-rate-limited",
    "UNAVAILABLE": "provider-rate-limited",
    "UNAUTHENTICATED": "provider-authentication-rejected",
    "PERMISSION_DENIED": "provider-authentication-rejected",
    "NOT_FOUND": "provider-model-unavailable",
    "DEADLINE_EXCEEDED": "provider-timeout",
    "INVALID_ARGUMENT": "provider-input-rejected",
}

_GEMINI_CODE_REASON_CODES: dict[int, str] = {
    429: "provider-rate-limited",
    503: "provider-rate-limited",
    401: "provider-authentication-rejected",
    403: "provider-authentication-rejected",
    404: "provider-model-unavailable",
    504: "provider-timeout",
    400: "provider-input-rejected",
}


def _gemini_failure_envelope(exc: BaseException, errors_module: object) -> ProviderFailureEnvelope:
    data = vars(exc)
    status_name = data.get("status")
    code = data.get("code")

    unknown_response_cls = (
        getattr(errors_module, "UnknownApiResponseError", None)
        if errors_module is not None
        else None
    )
    if unknown_response_cls is not None and isinstance(exc, unknown_response_cls):
        return ProviderFailureEnvelope(
            provider_kind="gemini", reason_code="provider-response-malformed"
        )

    if isinstance(exc, httpx.TimeoutException) or isinstance(
        exc, requests.exceptions.Timeout
    ):
        return ProviderFailureEnvelope(provider_kind="gemini", reason_code="provider-timeout")
    if isinstance(exc, httpx.TransportError) or isinstance(
        exc, requests.exceptions.RequestException
    ):
        return ProviderFailureEnvelope(
            provider_kind="gemini", reason_code="provider-connection-failed"
        )

    reason_code = (
        _GEMINI_STATUS_REASON_CODES.get(status_name) if isinstance(status_name, str) else None
    )
    if reason_code is None and isinstance(code, int) and not isinstance(code, bool):
        reason_code = _GEMINI_CODE_REASON_CODES.get(code)
    if reason_code is not None:
        limit_type = "overloaded" if status_name == "UNAVAILABLE" or code == 503 else None
        return ProviderFailureEnvelope(
            provider_kind="gemini",
            reason_code=reason_code,
            status=code if isinstance(code, int) and not isinstance(code, bool) else None,
            retry_after_s=None,
            limit_type=limit_type,
        )

    return ProviderFailureEnvelope(
        provider_kind="gemini",
        reason_code="provider-request-failed",
        status=code if isinstance(code, int) and not isinstance(code, bool) else None,
        retry_after_s=None,
    )


# ---------------------------------------------------------------------------
# Post-response signal detection and safe text extraction (section 5.2 / 5.3)
#
# Every function below runs *after* the adapter's own SDK call already
# succeeded -- called from outside the ``try``/``except`` blocks above, so a
# raised ``LLMError`` here is never re-caught and remapped by this module's
# own exception classifiers (never relabeled ``provider-request-failed``).
# A missing or ``None`` finish/stop/feedback field is "no signal", never an
# error -- this also keeps every pre-existing SDK-shaped test double that
# does not set these newer fields working unchanged, since a field this code
# never finds is indistinguishable from one explicitly absent on a real
# response.
# ---------------------------------------------------------------------------

# The nine Gemini policy-refusal finish reasons (the installed google-genai
# SDK's ``FinishReason``); each maps to the existing ``provider-input-rejected``.
_GEMINI_POLICY_FINISH_REASONS: frozenset[str] = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)


# Bounded, human-readable message for each *post-response signal* failure
# (truncation, refusal, or an unexplained empty response) -- never the closed
# reason-code string itself. ``LLMError``'s own message (a CodeDocError) is
# rendered unchanged by ``str(exc)`` at every reporting boundary, including
# ``ErrorReporter.summary()``, which is written into ``codedoc.json`` and
# Markdown; the new ``provider-output-truncated`` code must never appear
# verbatim in that generated JSON, Markdown, or recovery output, so it
# classifies through ``ProviderFailureEnvelope.reason_code`` only. This
# scoping applies only to the three signal failures raised here -- the
# existing SDK-exception path (the ``except Exception`` blocks above) still
# renders its other reason codes through ``_canonical_reason`` unchanged. No
# message here may contain a response body, refusal text, ``stop_details``,
# a prompt, source, or a credential.
_SIGNAL_FAILURE_MESSAGES: dict[str, str] = {
    "provider-output-truncated": "response was truncated before completion",
    "provider-input-rejected": "provider declined or blocked the request",
    "provider-response-malformed": "response contained no usable text",
}


def _raise_signal_failure(provider_display: str, provider_kind: str, reason_code: str) -> None:
    """Raise a typed, envelope-carrying ``LLMError`` for a post-response
    signal (truncation, refusal, or an unexplained empty response). Never
    invoked from inside this module's SDK-exception ``except`` blocks."""
    envelope = ProviderFailureEnvelope(provider_kind=provider_kind, reason_code=reason_code)
    message = _SIGNAL_FAILURE_MESSAGES[reason_code]
    raise LLMError(provider_display, message, provider_failure=envelope)


def _extract_anthropic_text(response: object, *, provider_display: str = "Anthropic") -> str:
    """Select and concatenate ``type == "text"`` content blocks, in response
    order, using the empty string as separator -- never a block whose type is
    anything else (including ``thinking``, whose content is never exposed).
    A single text block returns bytes identical to the historical adapter.
    """
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason in ("max_tokens", "model_context_window_exceeded"):
        _raise_signal_failure(provider_display, "anthropic", "provider-output-truncated")
    if stop_reason == "refusal":
        _raise_signal_failure(provider_display, "anthropic", "provider-input-rejected")
    blocks = getattr(response, "content", None) or []
    text = "".join(
        block.text for block in blocks if getattr(block, "type", None) == "text"
    )
    if text:
        return text
    _raise_signal_failure(provider_display, "anthropic", "provider-response-malformed")


def _extract_openai_text(response: object, *, provider_display: str = "OpenAI") -> str:
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
    message = getattr(choice, "message", None) if choice is not None else None
    refusal = getattr(message, "refusal", None) if message is not None else None
    if finish_reason == "length":
        _raise_signal_failure(provider_display, "openai", "provider-output-truncated")
    if finish_reason == "content_filter" or (isinstance(refusal, str) and refusal):
        _raise_signal_failure(provider_display, "openai", "provider-input-rejected")
    content = getattr(message, "content", None) if message is not None else None
    if content:
        return content
    _raise_signal_failure(provider_display, "openai", "provider-response-malformed")


def _extract_gemini_text(response: object, *, provider_display: str = "Gemini") -> str:
    """Gemini's own ``.text`` property already skips ``thought`` parts (the
    installed google-genai SDK), so no Anthropic-style type-based reselection
    is needed here -- only the finish/prompt-feedback signal check is new."""
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = (
        getattr(prompt_feedback, "block_reason", None) if prompt_feedback is not None else None
    )
    if block_reason is not None and block_reason != "BLOCKED_REASON_UNSPECIFIED":
        # A blocked prompt has no candidate and therefore no finish reason.
        _raise_signal_failure(provider_display, "gemini", "provider-input-rejected")
    candidates = getattr(response, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    finish_reason = getattr(candidate, "finish_reason", None) if candidate is not None else None
    if finish_reason == "MAX_TOKENS":
        _raise_signal_failure(provider_display, "gemini", "provider-output-truncated")
    if finish_reason in _GEMINI_POLICY_FINISH_REASONS:
        _raise_signal_failure(provider_display, "gemini", "provider-input-rejected")
    text = getattr(response, "text", None)
    if text:
        return text
    _raise_signal_failure(provider_display, "gemini", "provider-response-malformed")


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    OpenAI chat completions provider.

    Also works with any OpenAI-compatible API endpoint — LiteLLM proxy,
    Together.ai, Groq, a custom gateway, etc. — by passing ``base_url``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        *,
        timeout: float = _DEFAULT_PROVIDER_REQUEST_TIMEOUT_S,
        max_retries: int = 0,
        rate_limit_signals: tuple[str, ...] = (),
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise LLMError(
                "OpenAI", "openai package not installed. Run: pip install openai"
            ) from exc

        self._model = model
        self._sdk = _openai
        # Keeps the configured value (None for the default endpoint) so
        # FIX-1's exact-model rule and provider_name stay unchanged; the SDK
        # itself always receives an explicit, non-None base_url below.
        self._base_url = base_url
        self._rate_limit_signals = rate_limit_signals
        self._client = _openai.OpenAI(
            api_key=api_key,
            base_url=base_url or _OPENAI_DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.info("OpenAI provider ready — model: %s", model)

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = _UNSUPPLIED_TEMPERATURE,
    ) -> str:
        """Use the exact-model policy for an unsupplied temperature.

        An explicitly supplied value is forwarded unchanged. A subclass or
        wrapper that declares its own ``temperature=0.1`` and forwards it
        supplies that value, so it cannot inherit this method's omission.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(model=self._model, messages=messages)
        if temperature is _UNSUPPLIED_TEMPERATURE:
            if self._base_url or not openai_omits_temperature(self._model):
                kwargs["temperature"] = 0.1
        else:
            kwargs["temperature"] = temperature
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            envelope = _openai_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "OpenAI", _canonical_reason(envelope), provider_failure=envelope
            ) from exc
        return _extract_openai_text(response)

    def complete_json(self, prompt: str, system: str = "") -> str:
        """
        Uses OpenAI's native ``response_format={"type": "json_object"}`` mode
        for guaranteed JSON output rather than relying on text instructions alone.

        Exact IDs in :mod:`codedoc.llm.model_capabilities` omit the CodeDoc
        default ``temperature`` only on the default OpenAI endpoint. Unlisted
        IDs and any custom ``base_url`` keep the legacy request.
        """
        json_system = (
            (system + "\n\n" if system else "") + _JSON_INSTRUCTION
        )
        messages: list[dict] = [
            {"role": "system", "content": json_system},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        on_default_endpoint = not self._base_url
        if not (on_default_endpoint and openai_omits_temperature(self._model)):
            kwargs["temperature"] = 0.0
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            envelope = _openai_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "OpenAI", _canonical_reason(envelope), provider_failure=envelope
            ) from exc
        return _extract_openai_text(response)

    @property
    def provider_name(self) -> str:
        return f"OpenAI({self._model})"


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Uses the ``anthropic`` SDK's Messages API.  The system prompt is passed
    via the top-level ``system`` parameter (not inside the messages list),
    which is the idiomatic Anthropic pattern.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        *,
        timeout: float = _DEFAULT_PROVIDER_REQUEST_TIMEOUT_S,
        max_retries: int = 0,
        rate_limit_signals: tuple[str, ...] = (),
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise LLMError(
                "Anthropic",
                "anthropic package not installed. Run: pip install anthropic",
            ) from exc

        self._model = model
        self._sdk = _anthropic
        self._rate_limit_signals = rate_limit_signals
        self._client = _anthropic.Anthropic(
            api_key=api_key,
            base_url=_ANTHROPIC_DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.info("Anthropic provider ready — model: %s", model)

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = _UNSUPPLIED_TEMPERATURE,
    ) -> str:
        """Use the exact-model policy for an unsupplied temperature.

        Explicit values are forwarded unchanged, even for an omitting model.
        A subclass or wrapper that declares its own ``temperature=0.1`` and
        forwards it supplies that value, so it cannot inherit this method's
        omission. The public path retains ``max_tokens=4096``.
        """
        kwargs: dict = dict(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        if temperature is _UNSUPPLIED_TEMPERATURE:
            capability = anthropic_capability(self._model)
            if capability is None or capability.supports_temperature:
                kwargs["temperature"] = 0.1
        else:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            envelope = _anthropic_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Anthropic", _canonical_reason(envelope), provider_failure=envelope
            ) from exc
        return _extract_anthropic_text(response)

    def complete_json(self, prompt: str, system: str = "") -> str:
        """Explicit Anthropic JSON path (section 5.1): keeps the same JSON
        system instruction ``LLMProvider.complete_json`` built, but selects
        only the request parameters this exact model supports and its own
        conservative output cap, instead of delegating to :meth:`complete`
        with a fixed ``temperature=0.0``/``max_tokens=4096``.

        ``claude-haiku-4-5-20251001`` (the default) keeps the byte-identical
        historical request. An unknown model keeps the same legacy request and
        surfaces a real incompatibility as an ordinary provider error.
        """
        json_system = (
            (system + "\n\n" if system else "") + _JSON_INSTRUCTION
        )
        capability = anthropic_capability(self._model)
        max_tokens = capability.max_tokens if capability is not None else 4096
        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=json_system,
        )
        if capability is None or capability.supports_temperature:
            kwargs["temperature"] = 0.0
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            envelope = _anthropic_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Anthropic", _canonical_reason(envelope), provider_failure=envelope
            ) from exc
        return _extract_anthropic_text(response)

    @property
    def provider_name(self) -> str:
        return f"Anthropic({self._model})"


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    Google Gemini provider using the official ``google-genai`` SDK.

    System instructions are passed via ``GenerateContentConfig.system_instruction``
    (the idiomatic Gemini pattern) rather than being prepended to the user prompt.
    JSON mode uses ``response_mime_type="application/json"`` for native enforcement.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        timeout: float = _DEFAULT_PROVIDER_REQUEST_TIMEOUT_S,
        max_retries: int = 1,
        rate_limit_signals: tuple[str, ...] = (),
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "Gemini",
                "google-genai package not installed. Run: pip install google-genai",
            ) from exc

        self._model = model
        self._rate_limit_signals = rate_limit_signals
        http_options = types.HttpOptions(
            base_url=_GEMINI_DEFAULT_BASE_URL,
            timeout=int(timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=max_retries),
        )
        self._client = genai.Client(
            api_key=api_key, http_options=http_options, vertexai=False
        )
        self._types = types
        self._sdk = genai
        self._errors = getattr(genai, "errors", None)
        logger.info("Gemini provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        try:
            config = self._types.GenerateContentConfig(
                temperature=temperature,
                **({"system_instruction": system} if system else {}),
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            envelope = _gemini_failure_envelope(exc, self._errors)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Gemini", _canonical_reason(envelope), provider_failure=envelope
            ) from exc
        return _extract_gemini_text(response)

    def complete_json(self, prompt: str, system: str = "") -> str:
        """
        Uses Gemini's native ``response_mime_type="application/json"`` for
        guaranteed JSON output, with the JSON instruction passed as the
        system_instruction rather than concatenated into the user prompt.
        """
        json_system = (
            (system + "\n\n" if system else "") + _JSON_INSTRUCTION
        )
        try:
            config = self._types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=json_system,
                response_mime_type="application/json",
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            envelope = _gemini_failure_envelope(exc, self._errors)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Gemini", _canonical_reason(envelope), provider_failure=envelope
            ) from exc
        return _extract_gemini_text(response)

    @property
    def provider_name(self) -> str:
        return f"Gemini({self._model})"
