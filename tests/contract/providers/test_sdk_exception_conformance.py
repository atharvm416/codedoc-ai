"""Conformance tests: the closed provider-failure classification (section
5.3 / 11.1) against the *real* installed SDK exception shapes.

Unlike production adapter code -- which tolerates a minimal fake SDK module
by skipping any isinstance check whose class resolves to ``None`` via
``getattr`` -- this module asserts against the genuine installed
``openai``/``anthropic``/``google-genai`` exception hierarchies and fails
loudly if an expected class is missing, since that would mean the installed
SDK version no longer matches what this release's mapping was built against.
"""

from __future__ import annotations

import httpx
import pytest
import requests

import anthropic
import openai
from google.genai import errors as genai_errors

from codedoc.llm.api_provider import (
    _anthropic_failure_envelope,
    _gemini_failure_envelope,
    _openai_failure_envelope,
    _promote_via_configured_signals,
)
from codedoc.utils.errors import ProviderFailureEnvelope, detect_limit_type

_REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def _openai_response(
    status_code: int, body: dict | None = None, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST, json=body or {}, headers=headers)


def _anthropic_response(
    status_code: int, body: dict | None = None, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST, json=body or {}, headers=headers)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def test_openai_insufficient_quota_maps_to_quota_exhausted():
    response = _openai_response(429, {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}})
    exc = openai.RateLimitError("quota", response=response, body={"code": "insufficient_quota"})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.provider_kind == "openai"
    assert envelope.reason_code == "provider-quota-exhausted"
    assert envelope.status == 429


def test_openai_plain_rate_limit_without_insufficient_quota_code():
    response = _openai_response(429, {})
    exc = openai.RateLimitError("rate limited", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.status == 429
    assert envelope.limit_type is None


def test_openai_400_maps_to_input_rejected():
    response = _openai_response(400)
    exc = openai.BadRequestError("bad request", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-input-rejected"
    assert envelope.status == 400


def test_openai_413_maps_to_input_rejected():
    response = _openai_response(413)
    exc = openai.APIStatusError("too large", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-input-rejected"
    assert envelope.status == 413


def test_openai_422_maps_to_input_rejected():
    response = _openai_response(422)
    exc = openai.UnprocessableEntityError("unprocessable", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-input-rejected"
    assert envelope.status == 422


def test_openai_401_maps_to_authentication_rejected():
    response = _openai_response(401)
    exc = openai.AuthenticationError("bad key", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-authentication-rejected"
    assert envelope.status == 401


def test_openai_403_maps_to_authentication_rejected():
    response = _openai_response(403)
    exc = openai.PermissionDeniedError("forbidden", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-authentication-rejected"
    assert envelope.status == 403


def test_openai_404_maps_to_model_unavailable():
    response = _openai_response(404)
    exc = openai.NotFoundError("no such model", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-model-unavailable"
    assert envelope.status == 404


def test_openai_529_maps_to_rate_limited_overloaded():
    response = _openai_response(529)
    exc = openai.APIStatusError("overloaded", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.status == 529
    assert envelope.limit_type == "overloaded"


def test_openai_response_validation_error_maps_to_response_malformed_and_omits_status():
    response = _openai_response(200, {"unexpected": "shape"})
    exc = openai.APIResponseValidationError(response=response, body={"unexpected": "shape"})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-response-malformed"
    assert envelope.status is None


def test_openai_timeout_maps_to_provider_timeout():
    exc = openai.APITimeoutError(request=_REQUEST)
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-timeout"
    assert envelope.status is None


def test_openai_connection_error_maps_to_connection_failed():
    exc = openai.APIConnectionError(request=_REQUEST)
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-connection-failed"
    assert envelope.status is None


def test_openai_raw_httpx_timeout_maps_to_provider_timeout():
    exc = httpx.ConnectTimeout("timed out", request=_REQUEST)
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-timeout"


def test_openai_raw_requests_connection_error_maps_to_connection_failed():
    exc = requests.exceptions.ConnectionError("refused")
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-connection-failed"


def test_openai_unmapped_status_falls_back_to_request_failed():
    response = _openai_response(418)
    exc = openai.APIStatusError("teapot", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.reason_code == "provider-request-failed"
    assert envelope.status == 418


def test_openai_retry_after_header_is_read_as_plain_seconds():
    response = httpx.Response(
        429, request=_REQUEST, json={}, headers={"retry-after": "12"}
    )
    exc = openai.RateLimitError("slow down", response=response, body={})
    envelope = _openai_failure_envelope(exc, openai)
    assert envelope.retry_after_s == 12.0


def test_openai_tolerates_minimal_fake_sdk_module():
    """Section 11.1: the adapter must not raise AttributeError against a
    fake SDK module defining only a client class."""
    import types

    fake_sdk = types.SimpleNamespace()
    exc = RuntimeError("some unrelated local failure")
    envelope = _openai_failure_envelope(exc, fake_sdk)
    assert envelope.reason_code == "provider-request-failed"
    assert envelope.status is None


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _anthropic_status_error(status_code: int, error_type: str | None, *, cls=None):
    body = {"error": {"type": error_type}} if error_type else {}
    response = _anthropic_response(status_code, body)
    exc_cls = cls or anthropic.APIStatusError
    return exc_cls(f"failure {error_type}", response=response, body=body)


def test_anthropic_billing_error_type_maps_to_quota_exhausted():
    exc = _anthropic_status_error(400, "billing_error")
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-quota-exhausted"


def test_anthropic_overloaded_error_type_maps_to_rate_limited_overloaded():
    exc = _anthropic_status_error(529, "overloaded_error")
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.limit_type == "overloaded"


def test_anthropic_timeout_error_type_maps_to_provider_timeout():
    exc = _anthropic_status_error(500, "timeout_error")
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-timeout"


def test_anthropic_authentication_error_type_maps_to_authentication_rejected():
    exc = _anthropic_status_error(401, "authentication_error", cls=anthropic.AuthenticationError)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-authentication-rejected"


def test_anthropic_permission_error_type_maps_to_authentication_rejected():
    exc = _anthropic_status_error(403, "permission_error", cls=anthropic.PermissionDeniedError)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-authentication-rejected"


def test_anthropic_not_found_error_type_maps_to_model_unavailable():
    exc = _anthropic_status_error(404, "not_found_error", cls=anthropic.NotFoundError)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-model-unavailable"


def test_anthropic_rate_limit_error_type_maps_to_rate_limited():
    exc = _anthropic_status_error(429, "rate_limit_error", cls=anthropic.RateLimitError)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.limit_type is None


def test_anthropic_invalid_request_error_type_maps_to_input_rejected():
    exc = _anthropic_status_error(400, "invalid_request_error", cls=anthropic.BadRequestError)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-input-rejected"


def test_anthropic_api_error_type_maps_to_request_failed():
    exc = _anthropic_status_error(500, "api_error", cls=anthropic.InternalServerError)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-request-failed"


def test_anthropic_413_maps_to_input_rejected_by_status():
    # RequestTooLargeError may or may not be exported at anthropic's top
    # level depending on the installed SDK resolution (section 11.1) -- that
    # export is a third-party packaging detail, not a CodeDoc contract. Bind
    # the most specific installed class and fall back, so this body is
    # correct whether or not the SDK exports the convenience subclass; the
    # adapter itself classifies purely from status_code, never isinstance.
    cls = getattr(anthropic, "RequestTooLargeError", None) or anthropic.APIStatusError
    response = _anthropic_response(413, {})
    exc = cls("too large", response=response, body={})
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-input-rejected"
    assert envelope.status == 413


def test_anthropic_529_maps_to_rate_limited_overloaded_by_status():
    cls = getattr(anthropic, "OverloadedError", None) or anthropic.APIStatusError
    response = _anthropic_response(529, {})
    exc = cls("overloaded", response=response, body={})
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.limit_type == "overloaded"


def test_anthropic_response_validation_error_maps_to_response_malformed_and_omits_status():
    response = _anthropic_response(200, {})
    exc = anthropic.APIResponseValidationError(response=response, body={})
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-response-malformed"
    assert envelope.status is None


def test_anthropic_timeout_maps_to_provider_timeout():
    exc = anthropic.APITimeoutError(request=_REQUEST)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-timeout"


def test_anthropic_connection_error_maps_to_connection_failed():
    exc = anthropic.APIConnectionError(request=_REQUEST)
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-connection-failed"


def test_anthropic_retry_after_header_is_read_as_plain_seconds():
    response = _anthropic_response(
        429, {"error": {"type": "rate_limit_error"}}, headers={"retry-after": "3.5"}
    )
    exc = anthropic.RateLimitError(
        "slow down", response=response, body={"error": {"type": "rate_limit_error"}}
    )
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.retry_after_s == 3.5


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_resource_exhausted_maps_to_rate_limited():
    exc = genai_errors.ClientError(429, {"status": "RESOURCE_EXHAUSTED", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.status == 429
    assert envelope.limit_type is None
    assert envelope.retry_after_s is None


def test_gemini_unavailable_maps_to_rate_limited_overloaded():
    exc = genai_errors.ServerError(503, {"status": "UNAVAILABLE", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.limit_type == "overloaded"


def test_gemini_unauthenticated_maps_to_authentication_rejected():
    exc = genai_errors.ClientError(401, {"status": "UNAUTHENTICATED", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-authentication-rejected"


def test_gemini_permission_denied_maps_to_authentication_rejected():
    exc = genai_errors.ClientError(403, {"status": "PERMISSION_DENIED", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-authentication-rejected"


def test_gemini_not_found_maps_to_model_unavailable():
    exc = genai_errors.ClientError(404, {"status": "NOT_FOUND", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-model-unavailable"


def test_gemini_deadline_exceeded_maps_to_timeout():
    exc = genai_errors.ServerError(504, {"status": "DEADLINE_EXCEEDED", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-timeout"


def test_gemini_invalid_argument_maps_to_input_rejected():
    exc = genai_errors.ClientError(400, {"status": "INVALID_ARGUMENT", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-input-rejected"


def test_gemini_unknown_status_falls_back_to_numeric_code():
    exc = genai_errors.ClientError(429, {"message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.status == 429


def test_gemini_unmapped_status_and_code_falls_back_to_request_failed():
    exc = genai_errors.APIError(599, {"status": "WEIRD", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-request-failed"
    assert envelope.status == 599


def test_gemini_unknown_api_response_error_maps_to_response_malformed():
    exc = genai_errors.UnknownApiResponseError("could not parse response as JSON")
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-response-malformed"
    assert envelope.status is None


def test_gemini_retry_after_s_is_always_none():
    exc = genai_errors.ClientError(429, {"status": "RESOURCE_EXHAUSTED", "message": "x"}, None)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.retry_after_s is None


def test_gemini_raw_httpx_timeout_maps_to_provider_timeout():
    exc = httpx.ConnectTimeout("timed out", request=_REQUEST)
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-timeout"


def test_gemini_raw_requests_connection_error_maps_to_connection_failed():
    exc = requests.exceptions.ConnectionError("refused")
    envelope = _gemini_failure_envelope(exc, genai_errors)
    assert envelope.reason_code == "provider-connection-failed"


def test_gemini_tolerates_absent_errors_module():
    """Section 11.1: ``self._errors`` is None when a fake ``genai`` module
    has no ``errors`` attribute; the adapter must not raise."""
    exc = RuntimeError("some unrelated local failure")
    envelope = _gemini_failure_envelope(exc, None)
    assert envelope.reason_code == "provider-request-failed"


# ---------------------------------------------------------------------------
# Envelope never carries raw provider text (section 5.2 / 5.3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# detect_limit_type -- relocated pure classifier (section 11.5 "relocated")
# ---------------------------------------------------------------------------


def test_detect_limit_type_tpm():
    assert detect_limit_type("rate limit reached for tokens per min (TPM)") == "tpm"


def test_detect_limit_type_rpm():
    assert detect_limit_type("rate limit reached for requests per min (RPM)") == "rpm"


def test_detect_limit_type_quota():
    assert detect_limit_type("daily quota exceeded") == "quota"


def test_detect_limit_type_overloaded():
    assert detect_limit_type("529 overloaded") == "overloaded"


def test_detect_limit_type_unknown_returns_none():
    assert detect_limit_type("something unrelated") is None


# ---------------------------------------------------------------------------
# Malformed adapter metadata never escapes as a raw error (section 12.1 C1)
# ---------------------------------------------------------------------------


class _FakeProviderException(Exception):
    """A minimal stand-in exposing arbitrary attributes through ``vars()``,
    for shapes no real SDK exception constructor can produce directly (a
    non-int ``status_code``, a non-string Anthropic ``type``) but that
    adapter metadata handling must still normalize defensively rather than
    assume the SDK's own types are well-formed."""


@pytest.mark.parametrize(
    "envelope_fn,sdk,provider_kind",
    [
        (_openai_failure_envelope, openai, "openai"),
        (_anthropic_failure_envelope, anthropic, "anthropic"),
    ],
)
@pytest.mark.parametrize("malformed_status", ["429", 4.5, True, [429]])
def test_malformed_status_code_never_escapes_as_raw_error(
    envelope_fn, sdk, provider_kind, malformed_status
):
    exc = _FakeProviderException("malformed")
    exc.status_code = malformed_status
    envelope = envelope_fn(exc, sdk)
    assert envelope.provider_kind == provider_kind
    assert envelope.reason_code == "provider-request-failed"
    assert envelope.status is None


@pytest.mark.parametrize("malformed_type", [["billing_error"], {"error": "billing_error"}])
def test_anthropic_non_string_type_never_escapes_as_raw_type_error(malformed_type):
    exc = _FakeProviderException("malformed")
    exc.status_code = 500
    exc.type = malformed_type
    envelope = _anthropic_failure_envelope(exc, anthropic)
    assert envelope.reason_code == "provider-request-failed"


@pytest.mark.parametrize(
    "envelope_fn,sdk",
    [(_openai_failure_envelope, openai), (_anthropic_failure_envelope, anthropic)],
)
@pytest.mark.parametrize(
    "malformed_retry_after",
    ["", "not-seconds", "nan", "inf", "-inf", True, 7, 7.5, [], {}],
)
def test_malformed_retry_after_never_escapes_as_raw_error(
    envelope_fn, sdk, malformed_retry_after
):
    exc = _FakeProviderException("malformed")
    exc.status_code = 500
    exc.response = _FakeProviderException()
    exc.response.headers = {"retry-after": malformed_retry_after}
    envelope = envelope_fn(exc, sdk)
    assert envelope.retry_after_s is None


def test_openai_unmapped_status_preserves_retry_after_through_promotion():
    """Section 12.1 C1's compact proof: an unmapped 418 status with a valid
    plain-seconds retry-after must survive both the adapter's own default
    envelope construction and the subsequent custom-signal promotion --
    never dropped at either step."""
    response = _openai_response(418, {}, headers={"retry-after": "7.5"})
    exc = openai.APIStatusError("teapot custom_overload_signal", response=response, body={})
    baseline = _openai_failure_envelope(exc, openai)
    assert baseline.reason_code == "provider-request-failed"
    assert baseline.retry_after_s == 7.5
    promoted = _promote_via_configured_signals(exc, baseline, ("custom_overload_signal",))
    assert promoted.reason_code == "provider-rate-limited"
    assert promoted.retry_after_s == 7.5


def test_anthropic_unmapped_status_preserves_retry_after_through_promotion():
    response = _anthropic_response(418, {}, headers={"retry-after": "7.5"})
    exc = anthropic.APIStatusError(
        "teapot custom_overload_signal", response=response, body={}
    )
    baseline = _anthropic_failure_envelope(exc, anthropic)
    assert baseline.reason_code == "provider-request-failed"
    assert baseline.retry_after_s == 7.5
    promoted = _promote_via_configured_signals(exc, baseline, ("custom_overload_signal",))
    assert promoted.reason_code == "provider-rate-limited"
    assert promoted.retry_after_s == 7.5


# ---------------------------------------------------------------------------
# Adapter-boundary configured rate-limit signal promotion (section 5.3's
# final rule / section 5.7 limitation 4)
# ---------------------------------------------------------------------------


def test_promotion_only_applies_to_request_failed():
    exc = RuntimeError("429 rate_limit_exceeded tokens per min")
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-request-failed"
    )
    promoted = _promote_via_configured_signals(exc, envelope, ("429", "rate_limit_exceeded"))
    assert promoted.reason_code == "provider-rate-limited"
    assert promoted.limit_type == "tpm"


def test_promotion_never_overrides_a_structured_reason():
    exc = RuntimeError("429 rate_limit_exceeded")
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-authentication-rejected", status=401
    )
    promoted = _promote_via_configured_signals(exc, envelope, ("429", "rate_limit_exceeded"))
    assert promoted is envelope


def test_promotion_does_nothing_without_signal_match():
    exc = RuntimeError("some other unrelated local failure")
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-request-failed"
    )
    promoted = _promote_via_configured_signals(exc, envelope, ("429", "rate_limit_exceeded"))
    assert promoted is envelope


def test_promotion_does_nothing_with_no_configured_signals():
    exc = RuntimeError("429 rate_limit_exceeded")
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-request-failed"
    )
    promoted = _promote_via_configured_signals(exc, envelope, ())
    assert promoted is envelope


def test_promotion_preserves_status_and_retry_after_from_the_original_envelope():
    exc = RuntimeError("429 rate_limit_exceeded tokens per min")
    envelope = ProviderFailureEnvelope(
        provider_kind="openai",
        reason_code="provider-request-failed",
        status=429,
        retry_after_s=2.0,
    )
    promoted = _promote_via_configured_signals(exc, envelope, ("rate_limit_exceeded",))
    assert promoted.status == 429
    assert promoted.retry_after_s == 2.0


def test_openai_adapter_promotes_unmapped_failure_via_configured_signals():
    """End-to-end: an adapter instance with configured signals promotes an
    otherwise-unmapped 418 status through its real exception boundary."""
    response = _openai_response(418, {})
    exc = openai.APIStatusError("teapot rate_limit_exceeded", response=response, body={})
    baseline = _openai_failure_envelope(exc, openai)
    assert baseline.reason_code == "provider-request-failed"

    promoted = _promote_via_configured_signals(exc, baseline, ("rate_limit_exceeded",))
    assert promoted.reason_code == "provider-rate-limited"
    assert promoted.status == 418


def test_provider_instance_with_no_configured_signals_never_promotes(monkeypatch):
    """Section 12.1 C2: a directly-constructed adapter that never received
    ``rate_limit_signals`` (e.g. a test that never went through the
    factory's wiring) defaults to an empty tuple -- never absent, since the
    constructor is now the sole owner of this state -- and must not crash
    and must never promote."""
    from codedoc.llm.api_provider import OpenAIProvider

    rec = {}
    _install_openai_error(
        monkeypatch, rec, openai.APIStatusError("teapot rate_limit_exceeded", response=_openai_response(418, {}), body={})
    )
    provider = OpenAIProvider(api_key="k", model="gpt-test")
    assert provider._rate_limit_signals == ()
    try:
        provider.complete("hi")
    except Exception as exc:  # noqa: BLE001 -- asserting on the raised LLMError
        assert exc.provider_failure.reason_code == "provider-request-failed"
    else:
        raise AssertionError("expected LLMError")


def test_provider_instance_with_rate_limit_signals_attribute_promotes(monkeypatch):
    from codedoc.llm.api_provider import OpenAIProvider

    rec = {}
    _install_openai_error(
        monkeypatch, rec, openai.APIStatusError("teapot rate_limit_exceeded", response=_openai_response(418, {}), body={})
    )
    provider = OpenAIProvider(api_key="k", model="gpt-test")
    provider._rate_limit_signals = ("rate_limit_exceeded",)
    try:
        provider.complete("hi")
    except Exception as exc:  # noqa: BLE001
        assert exc.provider_failure.reason_code == "provider-rate-limited"
    else:
        raise AssertionError("expected LLMError")


def _install_openai_error(monkeypatch, rec, error):
    import sys
    import types

    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kwargs):
            raise error

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class OpenAI:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            self.chat = _Chat()

    mod.OpenAI = OpenAI
    for name in dir(openai):
        if name.endswith("Error") or name in ("APIStatusError",):
            setattr(mod, name, getattr(openai, name))
    monkeypatch.setitem(sys.modules, "openai", mod)


def test_envelopes_never_carry_raw_provider_message_text():
    from codedoc.utils.errors import provider_failure_as_mapping

    response = _openai_response(429, {"error": {"code": "insufficient_quota"}})
    exc = openai.RateLimitError(
        "Your credit balance is too low to continue", response=response, body={"code": "insufficient_quota"}
    )
    envelope = _openai_failure_envelope(exc, openai)
    rendered = repr(envelope) + str(provider_failure_as_mapping(envelope))
    assert "credit balance" not in rendered
