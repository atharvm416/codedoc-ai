"""Tests organized by feature ownership."""

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
import codedoc.core.execution as execution
from codedoc.core.error_classifier import (
    _build_rate_limit_exhausted_abort,
    _find_insufficient_source_error,
    _raise_rate_limit_exhausted,
)
from codedoc.utils.errors import InsufficientSourceError

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
    assert "resumes compatible completed ordinary files" in abort.reason
    assert "completed fresh-split files are deliberately" in abort.reason
    assert "resumes the unfinished files" not in abort.reason


def test_rate_limit_exhausted_abort_explains_fresh_split_rerun_boundary():
    abort = _build_rate_limit_exhausted_abort("openai")

    assert "resumes compatible completed ordinary files" in abort.reason
    assert "completed fresh-split files are deliberately" in abort.reason
    assert "resumes the unfinished files" not in abort.reason


def test_rate_limit_exhausted_warning_explains_fresh_split_rerun_boundary(capsys):
    class Recorder:
        def record(self, *_args, **_kwargs):
            return None

    with pytest.raises(UnrecoverableProviderError):
        _raise_rate_limit_exhausted("openai", Recorder())

    warning = capsys.readouterr().out
    assert "resumes compatible completed ordinary files" in warning
    assert "completed fresh-split files are deliberately" in warning
    assert "re-run the same command to resume" not in warning

def test_provider_construction_errors_are_classified_as_setup_errors(monkeypatch):
    from codedoc.llm.factory import create_provider
    from codedoc.utils.errors import LLMError, ProviderInitError

    monkeypatch.setattr(
        "codedoc.llm.factory._make_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(LLMError("fake", "bad auth")),
    )
    with pytest.raises(ProviderInitError):
        create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "openai",
                "model_name": "gpt-test",
                "api_key": "test",
            }
        )

class TestErrorClassifierImports:
    def test_B1_symbols_from_error_classifier(self):
        from codedoc.core.error_classifier import (
            _RATE_LIMIT_SIGNALS,
            _build_terminal_abort,
            _classify_failure,
            _parse_retry_after,
        )
        assert callable(_classify_failure)
        assert callable(_build_terminal_abort)
        assert callable(_parse_retry_after)
        assert isinstance(_RATE_LIMIT_SIGNALS, tuple)

    def test_B2_symbols_from_execution_compat(self):
        from codedoc.core.execution import (
            _classify_failure,
            _parse_retry_after,
        )
        assert callable(_classify_failure)
        assert callable(_parse_retry_after)

    def test_B3_billing_error_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "exceeded your current quota and billing is required")
        result = _classify_failure(exc, None)
        assert result == "terminal_billing"

    def test_B3_credential_error_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "invalid api key provided")
        result = _classify_failure(exc, None)
        assert result == "global"

    def test_B3_rate_limit_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "429 rate limit exceeded too many requests")
        result = _classify_failure(exc, None)
        assert result == "rate_limit"

    def test_B3_input_too_large_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "context length exceeded the maximum context allowed")
        result = _classify_failure(exc, None)
        assert result == "input"

    def test_B3_transient_classification(self):
        from codedoc.core.error_classifier import _classify_failure

        exc = RuntimeError("connection reset by peer")
        result = _classify_failure(exc, None)
        assert result == "transient"

    def test_B4_parse_retry_after_try_again(self):
        from codedoc.core.error_classifier import _parse_retry_after
        exc = RuntimeError("try again in 1.5s please")
        assert _parse_retry_after(exc) == pytest.approx(1.5)

    def test_B4_parse_retry_after_retry_after(self):
        from codedoc.core.error_classifier import _parse_retry_after
        exc = RuntimeError("retry.after: 2.0 seconds")
        assert _parse_retry_after(exc) == pytest.approx(2.0)

    def test_B4_parse_retry_after_none(self):
        from codedoc.core.error_classifier import _parse_retry_after
        exc = RuntimeError("no delay info here")
        assert _parse_retry_after(exc) is None

    def test_B5_same_parse_retry_after_both_modules(self):
        from codedoc.core import error_classifier

        exc = RuntimeError("try again in 1.5s")
        assert error_classifier._parse_retry_after(exc) == execution._parse_retry_after(exc)

        exc2 = RuntimeError("retry.after: 2.0")
        assert error_classifier._parse_retry_after(exc2) == execution._parse_retry_after(exc2)

def test_find_insufficient_source_error_walks_bare_cause_and_context():
    typed = InsufficientSourceError("m.py", "empty_or_whitespace_only")
    assert _find_insufficient_source_error(typed) is typed

    cause_wrapper = RuntimeError("outer")
    cause_wrapper.__cause__ = typed
    assert _find_insufficient_source_error(cause_wrapper) is typed

    context_wrapper = RuntimeError("outer")
    context_wrapper.__context__ = typed
    assert _find_insufficient_source_error(context_wrapper) is typed
    assert _find_insufficient_source_error(ValueError("unrelated")) is None

def test_insufficient_source_precedes_rate_limit_path_signals():
    exc = InsufficientSourceError("tpm/quota.py", "empty_or_whitespace_only")
    assert _classify_failure(exc, None) == "insufficient_source"
