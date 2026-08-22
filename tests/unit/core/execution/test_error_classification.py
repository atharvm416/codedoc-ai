"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
from codedoc.core.error_classifier import (
    _REASON_CODE_VERDICTS,
    _build_rate_limit_exhausted_abort,
    _classify_failure,
    _find_insufficient_source_error,
    _raise_rate_limit_exhausted,
)
from codedoc.utils.errors import (
    PROVIDER_FAILURE_REASON_CODES,
    AgentError,
    InsufficientSourceError,
    LLMError,
    ProviderFailureEnvelope,
    UnrecoverableProviderError,
    provider_failure_as_mapping,
)

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

def test_rate_limit_exhausted_abort_explains_reuse_and_recovery_boundary():
    abort = _build_rate_limit_exhausted_abort("openai")

    assert "reuses compatible completed ordinary and split records" in abort.reason
    assert "resumes compatible current schema-4 split node checkpoints" in abort.reason
    assert "deliberately re-documented" not in abort.reason
    assert "resumes the unfinished files" not in abort.reason


def test_rate_limit_exhausted_warning_explains_reuse_and_recovery_boundary(capsys):
    class Recorder:
        def record(self, *_args, **_kwargs):
            return None

    with pytest.raises(UnrecoverableProviderError):
        _raise_rate_limit_exhausted("openai", Recorder())

    warning = capsys.readouterr().out
    assert "reuses compatible completed ordinary and split records" in warning
    assert "resumes compatible current schema-4 split node checkpoints" in warning
    assert "deliberately re-documented" not in warning
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

def test_B3_transient_classification():
    exc = RuntimeError("connection reset by peer")
    result = _classify_failure(exc, None)
    assert result == "transient"

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


# ---------------------------------------------------------------------------
# The closed reason-code verdict table (section 5.3)
# ---------------------------------------------------------------------------


def test_reason_code_verdict_table_key_set_equals_the_nine_closed_codes():
    """Section 5.3: the table is closed -- its key set must equal the nine
    reason codes exactly, with no default and no lookup fallback. Compared
    here as an ordinary test, so the contract survives an optimized import
    that strips the module-level assertion guarding the same invariant."""
    assert set(_REASON_CODE_VERDICTS) == PROVIDER_FAILURE_REASON_CODES


@pytest.mark.parametrize(
    ("reason_code", "expected_verdict"),
    [
        ("provider-quota-exhausted", "terminal_billing"),
        ("provider-authentication-rejected", "global"),
        ("provider-model-unavailable", "global"),
        ("provider-rate-limited", "rate_limit"),
        ("provider-input-rejected", "input"),
        ("provider-timeout", "transient"),
        ("provider-connection-failed", "transient"),
        ("provider-response-malformed", "transient"),
        ("provider-request-failed", "transient"),
    ],
)
def test_every_reason_code_resolves_to_its_closed_table_verdict(
    reason_code, expected_verdict
):
    """Section 5.3: exercise all nine mappings through ``_classify_failure``.

    The expected verdicts are written out independently of
    ``_REASON_CODE_VERDICTS`` rather than read back from it, so this proves
    the released routing rather than restating the table. Each verdict has a
    distinct user-visible consequence, and two are load-bearing enough that a
    silent flip would be expensive: ``global`` and ``terminal_billing`` abort
    the run through ``_build_terminal_abort``, so demoting either to
    ``transient`` would turn a clean stop into a full retry ladder that
    spends one paid provider call per remaining file.
    """
    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code=reason_code
    )
    exc = LLMError("openai", reason_code, provider_failure=envelope)
    assert _classify_failure(exc, None) == expected_verdict


# ---------------------------------------------------------------------------
# New regressions (sections 11.3 and 11.5): prove the closed envelope table
# actually replaced free-text substring matching, rather than assuming it.
# ---------------------------------------------------------------------------


def test_build_terminal_abort_requires_an_envelope_and_never_echoes_raw_reason():
    """Section 11.3: an envelope-less argument to _build_terminal_abort must
    raise ValueError rather than silently returning an abort -- and the
    exception text must never echo the raw provider reason."""
    from codedoc.core.error_classifier import _build_terminal_abort

    bare = LLMError("openai", "insufficient_quota")
    with pytest.raises(ValueError) as caught:
        _build_terminal_abort(bare, "openai", "terminal_billing")
    assert "insufficient_quota" not in str(caught.value)


@pytest.mark.parametrize(
    "file_path, baseline_verdict",
    [
        ("src/auth/forbidden.py", "global"),
        ("src/quota/service.py", "rate_limit"),
        ("api/tpm_meter.py", "rate_limit"),
    ],
)
def test_purely_local_parse_json_failure_is_never_misclassified_by_its_own_path(
    tmp_path, file_path, baseline_verdict
):
    """Section 11.5's closure proof: a purely local ``BaseAgent._parse_json``
    failure -- which involves no provider SDK exception at all -- must
    classify as transient and never abort the run, even though its own file
    path happens to contain provider-signal-shaped substrings ("forbidden",
    "quota", "tpm") that the deleted substring classifier would have matched
    (baseline verdict noted per case: global, rate_limit, rate_limit)."""
    from codedoc.agents.file_documentation_agent import FileDocumentationAgent
    from codedoc.utils.errors import AgentError

    class _RawTextProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            return "not json at all"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    agent = FileDocumentationAgent(_RawTextProvider(), max_content_chars=1000)
    with pytest.raises(AgentError) as caught:
        agent._parse_json("not json at all", file_path)

    assert _classify_failure(caught.value, None) == "transient"
    assert baseline_verdict in ("global", "rate_limit")  # documents the fixed baseline bug


def test_terminal_sibling_precedence_wins_even_when_same_sibling_is_contract_final():
    """Section 5.4: the first terminal-billing/global envelope wins even
    when that same sibling also carries ``response_contract_final`` -- a
    genuine run-level abort is never masked by a contract marker on the
    very call that produced it."""
    from codedoc.core.execution import _raise_result_errors

    class _FakeOrchestrator:
        pass

    _FakeOrchestrator.__name__ = "Orchestrator"

    envelope = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-authentication-rejected", status=401
    )
    result = {
        "structure": {
            "error": "structure agent failed",
            "agent": "StructureAgent",
            "response_contract_final": True,
            "response_contract_correction_attempted": True,
            "provider_failure": provider_failure_as_mapping(envelope),
        },
    }
    with pytest.raises(AgentError) as caught:
        _raise_result_errors(result, _FakeOrchestrator(), "x.py")

    assert _classify_failure(caught.value, None) == "global"
    from codedoc.utils.errors import ResponseContractError

    assert not isinstance(caught.value, ResponseContractError)


def test_first_sibling_provider_failure_never_inherits_a_later_siblings_envelope():
    """Section 12.1 C4: the first fixed-order sibling with an error must
    drive the overall classification even when it carries no envelope --
    never skip past it to a later sibling's envelope, which would
    misclassify an unrelated local failure as rate-limited."""
    from codedoc.core.execution import _raise_result_errors

    class _FakeOrchestrator:
        pass

    _FakeOrchestrator.__name__ = "Orchestrator"

    rate_limited = ProviderFailureEnvelope(
        provider_kind="openai", reason_code="provider-rate-limited", status=429
    )
    result = {
        "structure": {
            "error": "local structure agent failure",
            "agent": "StructureAgent",
            # No "provider_failure" key at all: envelope-less, and first in
            # fixed order.
        },
        "dependencies_analysis": {
            "error": "dependency agent failed",
            "agent": "DependencyAgent",
            "provider_failure": provider_failure_as_mapping(rate_limited),
        },
    }
    with pytest.raises(AgentError) as caught:
        _raise_result_errors(result, _FakeOrchestrator(), "x.py")

    assert caught.value.provider_failure is None
    assert _classify_failure(caught.value, None) == "transient"


def test_bare_llm_error_without_provider_failure_classifies_as_transient_and_warns_never():
    """Section 11.5: a double raising a bare LLMError with rate-limit-shaped
    text but no ``provider_failure`` attribute must classify as transient
    (never rate_limit) -- so it can never produce a rate_limit_warnings entry,
    since that entry is only ever appended on a "rate_limit" verdict."""
    exc = LLMError("openai", "429 rate_limit_exceeded tokens per min")
    assert exc.provider_failure is None
    assert _classify_failure(exc, None) == "transient"
