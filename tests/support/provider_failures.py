"""Test-support factory for envelope-bearing provider failures (section 11.2).

Builds a :class:`~codedoc.utils.errors.ProviderFailureEnvelope` from closed
arguments and raises it through exactly the production ``LLMError`` attribute
a real adapter uses, so a network-free double renders identically to a real
adapter (section 5.3's canonical reason rule).

Imported by the thirteen collected provider-shaped-double test modules named
in section 11.2, satisfying ``tests/meta/test_suite_architecture.py``'s
two-importer rule. Deliberately not named ``test_*.py`` and defines no
``Test*`` class.
"""

from __future__ import annotations

from codedoc.utils.errors import LLMError, ProviderFailureEnvelope


def provider_failure_error(
    provider_kind: str,
    reason_code: str,
    *,
    status: int | None = None,
    retry_after_s: float | None = None,
    limit_type: str | None = None,
) -> LLMError:
    """Build an envelope-bearing :class:`LLMError` for a network-free double."""
    envelope = ProviderFailureEnvelope(
        provider_kind=provider_kind,
        reason_code=reason_code,
        status=status,
        retry_after_s=retry_after_s,
        limit_type=limit_type,
    )
    reason = reason_code if status is None else f"{reason_code} ({status})"
    return LLMError(provider_kind, reason, provider_failure=envelope)
