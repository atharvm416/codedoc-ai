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
        self._rate_limit_signals = rate_limit_signals
        self._client = _openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        logger.info("OpenAI provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            envelope = _openai_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "OpenAI", _canonical_reason(envelope), provider_failure=envelope
            ) from exc

    def complete_json(self, prompt: str, system: str = "") -> str:
        """
        Uses OpenAI's native ``response_format={"type": "json_object"}`` mode
        for guaranteed JSON output rather than relying on text instructions alone.
        """
        json_system = (
            (system + "\n\n" if system else "") + _JSON_INSTRUCTION
        )
        messages: list[dict] = [
            {"role": "system", "content": json_system},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            envelope = _openai_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "OpenAI", _canonical_reason(envelope), provider_failure=envelope
            ) from exc

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
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        logger.info("Anthropic provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        try:
            kwargs: dict = dict(
                model=self._model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            return response.content[0].text or ""
        except Exception as exc:
            envelope = _anthropic_failure_envelope(exc, self._sdk)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Anthropic", _canonical_reason(envelope), provider_failure=envelope
            ) from exc

    # Anthropic does not expose a dedicated JSON-mode parameter in the
    # standard Messages API, so we rely on the base class text-instruction
    # approach (complete_json from LLMProvider) combined with the JSON
    # extraction logic in BaseAgent._parse_json.

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
            timeout=int(timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=max_retries),
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)
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
            return response.text or ""
        except Exception as exc:
            envelope = _gemini_failure_envelope(exc, self._errors)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Gemini", _canonical_reason(envelope), provider_failure=envelope
            ) from exc

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
            return response.text or ""
        except Exception as exc:
            envelope = _gemini_failure_envelope(exc, self._errors)
            envelope = _promote_via_configured_signals(
                exc, envelope, getattr(self, "_rate_limit_signals", ())
            )
            raise LLMError(
                "Gemini", _canonical_reason(envelope), provider_failure=envelope
            ) from exc

    @property
    def provider_name(self) -> str:
        return f"Gemini({self._model})"
