"""0.9.7 — Workstream A: deterministic error-classification tests.

These exercise the two new conservative, network-free, chain-walking
classifiers next to ``_is_rate_limit_error`` and the
``UnrecoverableProviderError`` categories.  No network or credentials are used:
every input is a hand-built exception.

Rules under test (from PLAN_0.9.7.md, Workstream A):
- terminal-billing matches only *specific* phrases, never a bare ``quota`` /
  ``429`` / ``402``;
- global vs. input permanent classification;
- bare numeric HTTP codes (401/402/403/404/413) are NEVER an abort verdict on
  their own — only with a corroborating phrase;
- bare rate-limit signals stay retryable (handled by ``_is_rate_limit_error``);
- generic 5xx / timeout / connection / JSON-parse stay retryable;
- chain-wrapped messages are still classified;
- a message that is both terminal-billing and rate-limit is terminal (billing
  precedence).
"""

from __future__ import annotations

import pytest

from codedoc.core.execution import (
    _build_terminal_abort,
    _classify_failure,
    _classify_permanent_error,
    _is_rate_limit_error,
    _is_terminal_billing_error,
)
from codedoc.utils.errors import AgentError, LLMError, UnrecoverableProviderError


# ---------------------------------------------------------------------------
# Terminal-billing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        # OpenAI insufficient_quota
        "Error code: 429 - insufficient_quota: You exceeded your current quota, "
        "please check your plan and billing details.",
        "insufficient_quota",
        "You exceeded your current quota",
        # Anthropic credit exhaustion
        "Your credit balance is too low to access the Anthropic API.",
        # Generic billing / payment / spend-limit phrasings
        "402 payment required",
        "Access to billing is required to continue.",
        "You have hit your hard limit for this month.",
        "monthly spending limit reached",
    ],
)
def test_terminal_billing_phrases_classify_as_terminal(message):
    exc = LLMError("openai", message)
    assert _is_terminal_billing_error(exc) is True
    assert _classify_failure(exc, None) == "terminal_billing"


def test_terminal_billing_takes_precedence_over_rate_limit():
    # Co-occurs with a rate-limit signal (429 + quota) but carries a terminal
    # billing phrase, so billing precedence wins.
    exc = LLMError("openai", "429 insufficient_quota: exceeded your current quota")
    assert _is_terminal_billing_error(exc) is True
    assert _is_rate_limit_error(exc) is True  # also looks like a rate limit
    assert _classify_failure(exc, None) == "terminal_billing"


def test_context_hard_limit_is_input_specific_not_terminal_billing():
    exc = LLMError(
        "openai",
        "This request exceeds the hard limit for maximum context length",
    )
    assert _is_terminal_billing_error(exc) is False
    assert _classify_failure(exc, None) == "input"


def test_bare_quota_signals_are_not_terminal():
    for message in ("quota", "resource_exhausted", "429", "tpm", "overloaded",
                    "529", "503", "daily quota exceeded"):
        exc = LLMError("gemini", message)
        assert _is_terminal_billing_error(exc) is False, message
        # Still a rate limit, so it remains retryable (bounded by Workstream C).
        assert _is_rate_limit_error(exc) is True, message
        assert _classify_failure(exc, None) == "rate_limit", message


# ---------------------------------------------------------------------------
# Global-permanent (affects every file the same way)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "Incorrect API key provided: sk-***.",
        "invalid_api_key",
        "Invalid x-api-key",
        "authentication_error: invalid credentials",
        "401 Unauthorized",
        "permission_error: you do not have access",
        "403 Forbidden",
        "access denied",
        "The model `gpt-9` does not exist or you do not have access to it.",
        "model_not_found",
        "models/gemini-x is not found for API version v1beta",
        "unknown model requested",
    ],
)
def test_global_permanent_messages_classify_as_global(message):
    exc = LLMError("openai", message)
    assert _classify_permanent_error(exc) == "global"
    assert _is_terminal_billing_error(exc) is False
    assert _classify_failure(exc, None) == "global"


# ---------------------------------------------------------------------------
# Input-permanent (affects only this file's input)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "context_length_exceeded: maximum context length is 8192 tokens",
        "This model's maximum context length is 128000 tokens.",
        "prompt is too long: 250000 tokens > 200000 maximum",
        "Request too large for model.",
        "request_too_large",
        "413 payload too large",
    ],
)
def test_input_permanent_messages_classify_as_input(message):
    exc = LLMError("anthropic", message)
    assert _classify_permanent_error(exc) == "input"
    assert _classify_failure(exc, None) == "input"


def test_input_preferred_over_global_when_both_match():
    # A single message carrying both a global and an input signal must resolve to
    # the narrower, run-continuing "input" verdict.
    exc = LLMError("openai", "forbidden: request too large")
    assert _classify_permanent_error(exc) == "input"


# ---------------------------------------------------------------------------
# Bare numeric HTTP codes must NOT be an abort verdict on their own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["401", "402", "403", "404", "413"])
def test_bare_numeric_code_alone_is_not_an_abort(code):
    # An otherwise-unrelated message (e.g. a request id / token count) that merely
    # contains the digits must not trigger terminal/global/input.
    exc = LLMError("openai", f"transient blip (request id req_{code}xyz, {code} tokens)")
    assert _is_terminal_billing_error(exc) is False
    assert _classify_permanent_error(exc) is None
    assert _classify_failure(exc, None) == "transient"


def test_numeric_code_with_corroborating_phrase_classifies():
    not_found = LLMError("openai", "404 model not found")
    assert _classify_permanent_error(not_found) == "global"

    payment = LLMError("openai", "402 payment required")
    assert _is_terminal_billing_error(payment) is True


# ---------------------------------------------------------------------------
# Generic transient errors stay retryable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "500 internal server error",
        "request timed out",
        "Connection reset by peer",
        "Expecting value: line 1 column 1 (char 0)",
        "temporary provider outage",
    ],
)
def test_generic_errors_remain_retryable(message):
    exc = LLMError("openai", message)
    assert _is_terminal_billing_error(exc) is False
    assert _classify_permanent_error(exc) is None
    assert _classify_failure(exc, None) == "transient"


@pytest.mark.parametrize(
    "message",
    [
        "authentication service temporarily unavailable",
        "billing page request timed out",
        "connection took too long",
        "resource does not exist",
        "not_found_error for an unrelated API resource",
    ],
)
def test_ambiguous_phrases_do_not_trigger_permanent_abort(message):
    exc = LLMError("openai", message)
    assert _is_terminal_billing_error(exc) is False
    assert _classify_permanent_error(exc) is None
    assert _classify_failure(exc, None) == "transient"


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError("permission denied while reading local.py"),
        OSError("access denied to local source file"),
        RuntimeError("local parser reports: string too long"),
    ],
)
def test_local_failures_with_provider_like_phrases_are_not_permanent(exc):
    """Phrase matching must not turn local read/parse errors into provider aborts."""
    assert _classify_permanent_error(exc) in {"global", "input"}
    assert _classify_failure(exc, None) == "transient"


# ---------------------------------------------------------------------------
# Chain walking
# ---------------------------------------------------------------------------

def test_deeply_chained_terminal_phrase_is_detected():
    provider = LLMError("anthropic", "credit balance is too low")
    agent = AgentError("DocumentationAgent", "unknown", "provider failed")
    agent.__cause__ = provider
    exc = AgentError("Orchestrator", "x.py", "documentation failed")
    exc.__cause__ = agent
    assert _is_terminal_billing_error(exc) is True
    assert _classify_failure(exc, None) == "terminal_billing"


def test_deeply_chained_global_phrase_is_detected():
    provider = LLMError("openai", "Incorrect API key provided")
    exc = AgentError("Orchestrator", "x.py", "provider failed")
    exc.__cause__ = provider
    assert _classify_permanent_error(exc) == "global"


def test_phrase_in_own_message_without_cause_is_detected():
    # Route (a): the provider phrase is folded into an AgentError's own message,
    # raised WITHOUT ``from exc`` so there is no __cause__ to walk.
    exc = AgentError(
        "Orchestrator",
        "a.py",
        "StructureAgent: LLMError [openai]: insufficient_quota",
    )
    assert exc.__cause__ is None
    assert _is_terminal_billing_error(exc) is True


# ---------------------------------------------------------------------------
# UnrecoverableProviderError categories
# ---------------------------------------------------------------------------

def test_unrecoverable_provider_error_carries_category():
    terminal = UnrecoverableProviderError("openai", "credit exhausted", "terminal")
    assert terminal.category == "terminal"
    assert terminal.provider == "openai"
    assert isinstance(terminal, LLMError)

    exhausted = UnrecoverableProviderError(
        "gemini", "persistent rate limit", "rate_limit_exhausted"
    )
    assert exhausted.category == "rate_limit_exhausted"
    assert isinstance(exhausted, LLMError)


def test_unrecoverable_provider_error_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unsupported unrecoverable-provider category"):
        UnrecoverableProviderError("openai", "stopped", "unknown")


@pytest.mark.parametrize(
    "message, expected, absent",
    [
        ("Incorrect API key provided", "credentials", "model"),
        ("unknown model requested", "model", "credentials"),
        ("403 Forbidden", "forbidden", "credentials"),
    ],
)
def test_terminal_abort_names_only_the_matched_global_cause(message, expected, absent):
    abort = _build_terminal_abort(LLMError("openai", message), "openai", "global")
    assert expected in abort.reason
    assert absent not in abort.reason
