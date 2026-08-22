"""Unit tests for the structured provider-failure envelope (section 5.2).

Covers the frozen/slotted dataclass's own field validation, the single
chain-traversal helper (``find_provider_failure``), the mapping
serializer/deserializer pair, and ``LLMError``'s new ``provider_failure``
attribute -- independent of any adapter or classification wiring, which is
covered elsewhere as those workstreams land.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from codedoc.utils.errors import (
    BOUNDED_EXCEPTION_REASON_CODES,
    PROVIDER_FAILURE_REASON_CODES,
    PROVIDER_KINDS,
    PROVIDER_LIMIT_TYPES,
    LLMError,
    ProviderFailureEnvelope,
    find_provider_failure,
    provider_failure_as_mapping,
    provider_failure_from_mapping,
)

# ---------------------------------------------------------------------------
# Closed domains
# ---------------------------------------------------------------------------


def test_provider_kinds_domain():
    assert PROVIDER_KINDS == {"openai", "anthropic", "gemini"}


def test_provider_failure_reason_codes_domain():
    assert PROVIDER_FAILURE_REASON_CODES == {
        "provider-request-failed",
        "provider-authentication-rejected",
        "provider-rate-limited",
        "provider-quota-exhausted",
        "provider-model-unavailable",
        "provider-timeout",
        "provider-connection-failed",
        "provider-response-malformed",
        "provider-input-rejected",
    }


def test_provider_limit_types_domain():
    assert PROVIDER_LIMIT_TYPES == {"tpm", "rpm", "quota", "overloaded"}


def test_bounded_exception_reason_codes_gained_provider_input_rejected():
    assert "provider-input-rejected" in BOUNDED_EXCEPTION_REASON_CODES


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_construct_with_only_required_fields_defaults_optional_to_none():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-rate-limited"
    )
    assert envelope.provider_kind == "openai"
    assert envelope.reason_code == "provider-rate-limited"
    assert envelope.status is None
    assert envelope.retry_after_s is None
    assert envelope.limit_type is None


def test_construct_with_all_fields():
    envelope = ProviderFailureEnvelope(
        provider_kind="anthropic",
        reason_code="provider-rate-limited",
        status=529,
        retry_after_s=7.5,
        limit_type="overloaded",
    )
    assert envelope.status == 529
    assert envelope.retry_after_s == 7.5
    assert envelope.limit_type == "overloaded"


def test_retry_after_s_int_is_coerced_to_float():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-timeout", retry_after_s=5
    )
    assert envelope.retry_after_s == 5.0
    assert isinstance(envelope.retry_after_s, float)


def test_is_frozen():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-timeout"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        envelope.status = 500  # type: ignore[misc]


def test_is_slotted_rejects_arbitrary_attribute():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-timeout"
    )
    # A slotted+frozen dataclass rejects an unknown attribute name before it
    # ever reaches the frozen check; CPython raises AttributeError for a
    # plain slotted class but TypeError for this frozen/slots combination.
    with pytest.raises((AttributeError, TypeError)):
        envelope.unexpected = "nope"  # type: ignore[attr-defined]
    assert not hasattr(envelope, "__dict__")


def test_unsupported_provider_kind_rejected():
    with pytest.raises(ValueError):
        ProviderFailureEnvelope(provider_kind="openai-compatible", reason_code="provider-timeout")


def test_unsupported_reason_code_rejected():
    with pytest.raises(ValueError):
        ProviderFailureEnvelope(provider_kind="openai", reason_code="not-a-real-code")


def test_bounded_summary_reason_code_not_accepted_as_envelope_reason_code():
    with pytest.raises(ValueError):
        ProviderFailureEnvelope(provider_kind="openai", reason_code="unknown-error")


@pytest.mark.parametrize("bad_status", [True, False, "429", 429.0])
def test_non_boolean_int_status_required(bad_status):
    with pytest.raises(ValueError):
        ProviderFailureEnvelope(
            provider_kind="openai", reason_code="provider-timeout", status=bad_status
        )


@pytest.mark.parametrize(
    "bad_retry_after_s",
    [True, False, "7.5", -0.001, math.inf, -math.inf, math.nan],
)
def test_invalid_retry_after_s_rejected(bad_retry_after_s):
    with pytest.raises(ValueError):
        ProviderFailureEnvelope(
            provider_kind="openai",
            reason_code="provider-timeout",
            retry_after_s=bad_retry_after_s,
        )


def test_retry_after_s_zero_is_accepted():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-timeout", retry_after_s=0.0
    )
    assert envelope.retry_after_s == 0.0


def test_unsupported_limit_type_rejected():
    with pytest.raises(ValueError):
        ProviderFailureEnvelope(
            provider_kind="anthropic",
            reason_code="provider-rate-limited",
            limit_type="minute",
        )


# ---------------------------------------------------------------------------
# find_provider_failure -- the single chain traversal
# ---------------------------------------------------------------------------


def test_find_provider_failure_on_the_exception_itself():
    envelope = ProviderFailureEnvelope(provider_kind="gemini", reason_code="provider-timeout")
    exc = LLMError("gemini", "provider-timeout", provider_failure=envelope)
    assert find_provider_failure(exc) is envelope


def test_find_provider_failure_returns_none_for_none():
    assert find_provider_failure(None) is None


def test_find_provider_failure_returns_none_when_absent_from_chain():
    exc = RuntimeError("plain failure")
    assert find_provider_failure(exc) is None


def test_find_provider_failure_walks_explicit_cause_chain():
    envelope = ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-rate-limited")
    inner = LLMError("openai", "provider-rate-limited", provider_failure=envelope)
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner
    assert find_provider_failure(outer) is envelope


def test_find_provider_failure_walks_implicit_context_chain():
    envelope = ProviderFailureEnvelope(
        provider_kind="anthropic", reason_code="provider-quota-exhausted"
    )
    inner = LLMError("anthropic", "provider-quota-exhausted", provider_failure=envelope)
    outer = RuntimeError("wrapped")
    outer.__context__ = inner
    assert find_provider_failure(outer) is envelope


def test_find_provider_failure_prefers_cause_over_context():
    cause_envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-timeout"
    )
    context_envelope = ProviderFailureEnvelope(
        provider_kind="gemini", reason_code="provider-connection-failed"
    )
    cause = LLMError("openai", "provider-timeout", provider_failure=cause_envelope)
    context = LLMError("gemini", "provider-connection-failed", provider_failure=context_envelope)
    outer = RuntimeError("wrapped")
    outer.__cause__ = cause
    outer.__context__ = context
    assert find_provider_failure(outer) is cause_envelope


def test_find_provider_failure_is_cycle_safe():
    exc = RuntimeError("self-referential")
    exc.__context__ = exc
    assert find_provider_failure(exc) is None


def test_find_provider_failure_ignores_non_envelope_attribute():
    exc = RuntimeError("odd")
    exc.provider_failure = "not-an-envelope"  # type: ignore[attr-defined]
    assert find_provider_failure(exc) is None


# ---------------------------------------------------------------------------
# Mapping serializer / defensive deserializer
# ---------------------------------------------------------------------------


def test_as_mapping_has_exactly_five_keys():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai",
        reason_code="provider-rate-limited",
        status=429,
        retry_after_s=1.5,
        limit_type="tpm",
    )
    mapping = provider_failure_as_mapping(envelope)
    assert mapping == {
        "provider_kind": "openai",
        "reason_code": "provider-rate-limited",
        "status": 429,
        "retry_after_s": 1.5,
        "limit_type": "tpm",
    }


def test_as_mapping_uses_json_null_shape_for_absent_optional_fields():
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-request-failed"
    )
    mapping = provider_failure_as_mapping(envelope)
    assert mapping["status"] is None
    assert mapping["retry_after_s"] is None
    assert mapping["limit_type"] is None


def test_from_mapping_round_trips_as_mapping():
    original = ProviderFailureEnvelope(
        provider_kind="anthropic",
        reason_code="provider-rate-limited",
        status=529,
        retry_after_s=3.0,
        limit_type="overloaded",
    )
    rebuilt = provider_failure_from_mapping(provider_failure_as_mapping(original))
    assert rebuilt == original


def test_from_mapping_rejects_non_dict():
    assert provider_failure_from_mapping("not-a-mapping") is None
    assert provider_failure_from_mapping(None) is None
    assert provider_failure_from_mapping(["provider_kind"]) is None


def test_from_mapping_rejects_missing_key():
    mapping = provider_failure_as_mapping(
        ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-timeout")
    )
    del mapping["limit_type"]
    assert provider_failure_from_mapping(mapping) is None


def test_from_mapping_rejects_extra_key():
    mapping = provider_failure_as_mapping(
        ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-timeout")
    )
    mapping["unexpected"] = "field"
    assert provider_failure_from_mapping(mapping) is None


def test_from_mapping_rejects_dataclass_validation_failure_instead_of_raising():
    mapping = provider_failure_as_mapping(
        ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-timeout")
    )
    mapping["provider_kind"] = "not-a-real-provider"
    assert provider_failure_from_mapping(mapping) is None


def test_from_mapping_rejects_foreign_value_types():
    mapping = provider_failure_as_mapping(
        ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-timeout")
    )
    mapping["status"] = "429"
    assert provider_failure_from_mapping(mapping) is None


# ---------------------------------------------------------------------------
# LLMError.provider_failure
# ---------------------------------------------------------------------------


def test_llm_error_provider_failure_defaults_to_none():
    exc = LLMError("openai", "boom")
    assert exc.provider_failure is None


def test_llm_error_provider_failure_is_keyword_only():
    envelope = ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-timeout")
    exc = LLMError("openai", "boom", provider_failure=envelope)
    assert exc.provider_failure is envelope
