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


# ===========================================================================
# Section 10 / 8.G: response_correction_enabled now defaults to True. These
# tests exercise the CONFIGURATION-RESOLVED default path -- they never set the
# key -- and prove the new behaviour end to end.
# ===========================================================================

def _default_config(**overrides):
    cfg = {
        "entry_file": "main.py",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
        "file_retry_attempts": 0,
    }
    cfg.update(overrides)
    return cfg


def test_default_run_repairs_one_rejected_response_and_publishes(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})  # 1st invalid, correction valid
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)

    stats = run_pipeline(tmp_path, _default_config())

    assert stats["checked"] == 1 and stats["failed"] == 0
    assert prov.correction_calls == 1  # exactly one correction call
    assert stats["response_correction_enabled"] is True
    assert stats["response_contract_failures"] == 1
    assert stats["response_correction_calls_attempted"] == 1
    assert stats["response_correction_calls_succeeded"] == 1
    assert stats["response_correction_calls_failed"] == 0
    assert stats["additional_attempts"] == 1
    # The correction never consumes a new manifest entry.
    assert stats["total_calls_planned"] == 1
    assert stats["attempted_logical_calls"] == 1

    import json

    record = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert record["path"] == "main.py"
    assert record["description"] == "A file."  # the corrected content


def test_default_run_makes_no_second_repair_when_the_correction_is_invalid(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(
        fail_agents={"combined"},
        correction_response={"role_in_system": "r"},  # corrected reply still invalid
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)

    stats = run_pipeline(tmp_path, _default_config())

    assert stats["checked"] == 0 and stats["failed"] == 1
    assert prov.correction_calls == 1  # the one-call guarantee: no second repair
    assert stats["response_correction_calls_attempted"] == 1
    assert stats["response_correction_calls_succeeded"] == 0
    assert stats["response_correction_calls_failed"] == 1
    assert (
        stats["response_correction_calls_succeeded"]
        + stats["response_correction_calls_failed"]
        == stats["response_correction_calls_attempted"]
    )


def test_explicit_false_still_fully_disables_correction(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)

    stats = run_pipeline(
        tmp_path, _default_config(response_correction_enabled=False)
    )

    assert stats["checked"] == 0 and stats["failed"] == 1
    assert prov.correction_calls == 0  # zero additional calls
    assert stats["response_correction_enabled"] is False
    assert stats["response_contract_failures"] == 1
    assert stats["response_correction_calls_attempted"] == 0
    assert stats["response_correction_calls_succeeded"] == 0
    assert stats["response_correction_calls_failed"] == 0
    assert stats["additional_attempts"] == 0


def test_default_dry_run_reports_the_correction_billing_without_a_provider(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("dry-run must not construct a provider"),
    )

    stats = run_pipeline(tmp_path, _default_config(dry_run=True))

    assert stats["response_correction_enabled"] is True
    assert stats["initial_documentation_calls_planned"] == 1
    assert stats["prompt_review_calls_planned"] == 0
    assert stats["initial_provider_calls_planned"] == 1
    assert stats["correction_calls_possible_max"] == 1  # up to one per doc response
    assert stats["provider_calls_max_before_retries"] == 2  # 1 initial + 1 correction
    assert stats["retries_included_in_ceiling"] is False
    assert stats["max_planned_calls_applies_to"] == "initial_provider_calls_planned"
    # The two legacy documentation-only keys remain.
    assert stats["estimated_calls"] == 1
    assert stats["estimated_calls_max_with_correction"] == 2


def test_max_planned_calls_at_cap_is_accepted_with_a_higher_correction_ceiling(
    tmp_path, monkeypatch
):
    """max_planned_calls caps only the initial manifest (section 5.5 lines
    762-773). A run exactly at the cap is accepted even though its
    correction-inclusive ceiling is higher; retries stay additional."""
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("dry-run must not construct a provider"),
    )

    stats = run_pipeline(
        tmp_path, _default_config(dry_run=True, max_planned_calls=1)
    )

    assert stats["total_calls_planned"] == 1
    assert stats["max_planned_calls"] == 1
    assert stats["max_planned_calls_exceeded"] is False  # at the cap -> accepted
    assert stats["correction_calls_possible_max"] == 1
    assert stats["provider_calls_max_before_retries"] == 2  # > the cap, and disclosed
    assert stats["retries_included_in_ceiling"] is False


def test_max_planned_calls_at_cap_accepts_a_real_run_that_may_correct(
    tmp_path, monkeypatch
):
    """The real run is not blocked either: the cap authorized its one initial
    call, and the correction attaches to that same planned logical call."""
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)

    stats = run_pipeline(tmp_path, _default_config(max_planned_calls=1))

    assert stats["checked"] == 1 and stats["failed"] == 0
    assert stats["total_calls_planned"] == 1
    assert prov.correction_calls == 1
    assert stats["additional_attempts"] == 1  # the correction, outside the cap
