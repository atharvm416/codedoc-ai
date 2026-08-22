"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
from codedoc.core.execution import _process_one_file, _process_one_file_with_retries
from codedoc.utils.errors import ResponseContractError, UnrecoverableProviderError
from tests.support.provider_failures import provider_failure_error
from tests.support.response_correction_cases import RoutingProvider
from tests.support.response_correction_cases import _orch
from tests.support.response_correction_cases import _request

def test_correction_call_billing_fault_is_terminal(tmp_path):
    # A confirmed-unrecoverable fault on the correction call is a run-level
    # abort routed through the existing terminal path (which lives in the
    # retry wrapper, not _process_one_file). Simulated as an authentication
    # rejection rather than magic "insufficient_quota" text: bounded_exception_
    # summary classifies by exception TYPE only, and openai/anthropic surface
    # both transient rate limits and quota/billing exhaustion as the same
    # RateLimitError type, so a raw RuntimeError with billing-flavored text can
    # no longer drive terminal classification post-bounding -- a credential
    # rejection (a distinct, unambiguously-global SDK type) exercises the
    # identical terminal-abort path this test verifies.
    prov = RoutingProvider(
        fail_agents={"combined"},
        raise_on_correction=provider_failure_error(
            "openai", "provider-authentication-rejected", status=401
        ),
    )
    with pytest.raises(UnrecoverableProviderError):
        _process_one_file_with_retries(
            _request(tmp_path), _orch(prov, enabled=True), retry_attempts=1,
            split_execution_mode="recovery",
        )
    assert prov.correction_calls == 1

def test_correction_call_rate_limit_is_final_not_retried(tmp_path):
    prov = RoutingProvider(
        fail_agents={"combined"},
        raise_on_correction=RuntimeError("rate limit exceeded (429)"),
    )
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), _orch(prov, enabled=True))
    assert prov.correction_calls == 1
    assert caught.value.correction_attempted is True

def test_terminal_sibling_error_is_not_masked_by_contract_failure(tmp_path):
    prov = RoutingProvider(
        fail_agents={"structure"},
        raise_agents={
            "dependency": provider_failure_error(
                "openai", "provider-authentication-rejected", status=401
            )
        },
    )
    with pytest.raises(UnrecoverableProviderError):
        _process_one_file_with_retries(
            _request(tmp_path, mode="triple"),
            _orch(prov, mode="triple", enabled=False, parallel=False),
            retry_attempts=1,
            split_execution_mode="recovery",
        )
    assert prov.per_agent_initial["structure"] == 1
    assert prov.per_agent_initial["dependency"] == 1

def test_pipeline_correction_stats_reconcile_to_attempted_calls(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "response_correction_enabled": True,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )

    assert stats["attempted_calls"] == 2
    assert stats["documentation_calls_attempted"] == 2
    assert stats["response_contract_failures"] == 1
    assert stats["response_correction_calls_attempted"] == 1
    assert stats["response_correction_calls_succeeded"] == 1
    assert stats["response_correction_calls_failed"] == 0
    assert (
        stats["response_correction_calls_succeeded"]
        + stats["response_correction_calls_failed"]
        == stats["response_correction_calls_attempted"]
    )
    # A correction attaches to the originating planned logical call: it never
    # consumes another manifest entry and is never counted as an initially
    # planned call — it appears only as an additional attempt.
    assert stats["total_calls_planned"] == 1
    assert stats["attempted_logical_calls"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert stats["additional_attempts"] == 1
