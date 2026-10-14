"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import sys
import types

import pytest
from codedoc.agents import response_correction_agent as rca
from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import CorrectionLedger
from codedoc.core.execution import _process_one_file, _process_one_file_with_retries
from codedoc.utils.errors import (
    LLMError,
    ResponseContractError,
    UnrecoverableProviderError,
    find_provider_failure,
)
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
# FIX-3 regression: the closed reason code must classify the failure but
# never appear verbatim in persisted output. A provider-free pipeline run
# through the REAL OpenAIProvider (backed by a fake openai SDK module, no
# network) exercises the actual production code path end to end, not the
# RoutingProvider fake used elsewhere in this file.
# ===========================================================================


def _install_truncating_openai(monkeypatch, *, bad_marker: str, fake_key: str):
    """A fake ``openai`` SDK module: the request whose prompt contains
    *bad_marker* comes back with ``finish_reason="length"``; every other
    request succeeds normally. Records every call for retry/correction
    counting."""
    calls: list[dict] = []
    mod = types.ModuleType("openai")

    class _Message:
        def __init__(self, content):
            self.content = content
            self.refusal = None

    class _Choice:
        def __init__(self, content, finish_reason):
            self.finish_reason = finish_reason
            self.message = _Message(content)

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][-1]["content"]
            if bad_marker in prompt:
                return types.SimpleNamespace(
                    choices=[_Choice("SENTINEL-PARTIAL-RESPONSE-BODY", "length")]
                )
            return types.SimpleNamespace(
                choices=[_Choice(json.dumps({"description": "A file."}), "stop")]
            )

    class OpenAI:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            assert api_key == fake_key
            self.chat = types.SimpleNamespace(completions=_Completions())

    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return calls


def test_truncated_output_classifies_correctly_and_never_leaks_the_reason_code(
    tmp_path, monkeypatch
):
    from codedoc.llm.api_provider import OpenAIProvider
    from codedoc.pipeline import run_pipeline

    fake_key = "SENTINEL-FAKE-API-KEY-XYZ"
    calls = _install_truncating_openai(monkeypatch, bad_marker="BAD_MARKER", fake_key=fake_key)

    (tmp_path / "bad.py").write_text("BAD_MARKER = 1\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("GOOD_VALUE = 1\n", encoding="utf-8")

    def _create_provider(_config):
        return OpenAIProvider(api_key=fake_key, model="gpt-4o-mini")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _create_provider)

    stats = run_pipeline(
        tmp_path,
        {
            "llm_provider": "openai",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 2,  # must NOT be consumed by the bad file
            "response_correction_enabled": True,  # must NOT be invoked either
            "propagate_changes": False,
            "output_dir": "docs",
            "output_format": "both",
        },
    )

    # -- classification and behavior -----------------------------------
    assert stats["checked"] == 1 and stats["failed"] == 1
    assert stats["response_correction_calls_attempted"] == 0
    # Exactly one call per file: no retry on the failed one, and the good
    # file's success is unaffected (other files continue).
    assert len(calls) == 2

    # -- persisted output must not contain the exact reason code --------
    docs_dir = tmp_path / "docs"
    json_text = (docs_dir / "codedoc.json").read_text(encoding="utf-8")
    md_text = (docs_dir / "codedoc.md").read_text(encoding="utf-8")
    recovery_path = docs_dir / "crash_recovery.json"
    recovery_text = recovery_path.read_text(encoding="utf-8") if recovery_path.exists() else ""

    for text in (json_text, md_text, recovery_text):
        assert "provider-output-truncated" not in text
        assert "SENTINEL-PARTIAL-RESPONSE-BODY" not in text
        assert fake_key not in text

    # -- the public summary is present and useful, but bounded ----------
    payload = json.loads(json_text)
    assert "errors" in payload and payload["errors"]
    assert "provider-output-truncated" not in payload["errors"]
    assert fake_key not in payload["errors"]
    assert "SENTINEL-PARTIAL-RESPONSE-BODY" not in payload["errors"]
    # Still names something a maintainer can act on.
    assert "truncat" in payload["errors"].lower() or "AgentError" in payload["errors"]

    # good.py's own record is unaffected.
    good_record = next(f for f in payload["files"] if f["path"] == "good.py")
    assert good_record["description"] == "A file."


def test_truncated_output_envelope_still_classifies_as_output_truncated(monkeypatch):
    """Direct adapter-level check that the envelope's reason_code is
    unchanged even though the exception message is now bounded and
    human-readable rather than the exact code string."""
    from codedoc.llm.api_provider import OpenAIProvider

    fake_key = "k"
    _install_truncating_openai(monkeypatch, bad_marker="ALWAYS-BAD", fake_key=fake_key)
    provider = OpenAIProvider(api_key=fake_key, model="gpt-4o-mini")

    with pytest.raises(LLMError) as caught:
        provider.complete_json("ALWAYS-BAD prompt")

    failure = find_provider_failure(caught.value)
    assert failure is not None
    assert failure.reason_code == "provider-output-truncated"
    assert "provider-output-truncated" not in str(caught.value)
    assert "SENTINEL-PARTIAL-RESPONSE-BODY" not in str(caught.value)


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


# ===========================================================================
# HARD-1: the field-summary ceiling guard, exercised through the real
# ordinary/triple route (_safe_run -> _agent_error_result -> _raise_result_
# errors), not the direct unit-level ``_field_summary`` call. The ceiling is
# monkeypatched low because it is structurally unreachable through the
# normal bounded diagnostic collection path (section 5.3's own invariant).
# ===========================================================================


def test_ordinary_route_ceiling_breach_makes_zero_correction_calls(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rca, "_FIELD_SUMMARY_CEILING_CHARS", 0)
    prov = RoutingProvider(fail_agents={"combined"})
    ledger = CorrectionLedger(True)
    orch = Orchestrator(
        prov, parallel=False, analysis_mode="single",
        correction_ledger=ledger, response_correction_enabled=True,
    )
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 1
    assert prov.correction_calls == 0
    assert caught.value.correction_attempted is False
    assert ledger.snapshot()["response_correction_calls_attempted"] == 0
    assert ledger.snapshot()["response_correction_calls_succeeded"] == 0
    assert ledger.snapshot()["response_correction_calls_failed"] == 0


def test_ordinary_route_ceiling_breach_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(rca, "_FIELD_SUMMARY_CEILING_CHARS", 0)
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=True)
    with pytest.raises(ResponseContractError):
        _process_one_file_with_retries(
            _request(tmp_path), orch, retry_attempts=3,
            split_execution_mode="recovery",
        )
    # Exactly the one initial call: the response-contract-final marker keeps
    # this off the file-retry ladder despite retry_attempts=3.
    assert prov.calls == 1
    assert prov.correction_calls == 0


def test_triple_route_ceiling_breach_makes_zero_correction_calls(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rca, "_FIELD_SUMMARY_CEILING_CHARS", 0)
    prov = RoutingProvider(fail_agents={"structure"})
    ledger = CorrectionLedger(True)
    orch = Orchestrator(
        prov, parallel=False, analysis_mode="triple",
        correction_ledger=ledger, response_correction_enabled=True,
    )
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path, mode="triple"), orch)
    assert prov.correction_calls == 0
    assert caught.value.correction_attempted is False
    assert ledger.snapshot()["response_correction_calls_attempted"] == 0
    # Successful siblings still each ran exactly once; only the rejected one
    # is final, and it consumed no correction call.
    assert prov.per_agent_initial["dependency"] == 1
    assert prov.per_agent_initial["documentation"] == 1
    assert prov.per_agent_initial["structure"] == 1


def _valid_combined_response(description: str) -> str:
    return json.dumps(
        {
            "description": description,
            "role_in_system": "core",
            "functions": [],
            "classes": [],
            "exports": [],
            "dependencies_analysis": {
                "internal": [], "external": [], "dependency_refs": [],
                "catalog_updates": [], "usage_notes": [], "warnings": [],
            },
            "key_concepts": [],
            "usage_example": "",
        }
    )


class _CeilingBreachOnMarkerProvider:
    """A plain LLMProvider-shaped fake (no SDK involved): every file except
    the one whose source contains *bad_marker* gets a valid combined
    response; the marked file gets a response missing the required
    "description" field, triggering a response-contract failure whose
    correction the (monkeypatched-low) HARD-1 ceiling then refuses."""

    provider_name = "fake"

    def __init__(self, bad_marker: str):
        self._bad_marker = bad_marker
        self.calls = 0

    def complete_json(self, prompt, system=""):
        self.calls += 1
        if self._bad_marker in prompt:
            return json.dumps({"role_in_system": "r"})
        return _valid_combined_response("Documented.")

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def test_ordinary_route_ceiling_breach_preserves_a_pre_existing_successful_record(
    tmp_path, monkeypatch
):
    """Plan requirement: the ceiling-breach route must preserve pre-existing
    output state, not merely leave the failing file's own output absent. Runs
    the real pipeline twice: the first run successfully documents two files;
    the second run adds a third, ceiling-breaching file while the first two
    are unchanged. Their records must survive byte-identical (reused, never
    reprocessed) alongside the new file's recorded failure."""
    from codedoc.pipeline import run_pipeline

    (tmp_path / "good.py").write_text("GOOD_VALUE = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("OTHER_VALUE = 1\n", encoding="utf-8")

    provider_run1 = _CeilingBreachOnMarkerProvider(bad_marker="__never_matches__")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: provider_run1)
    config = {
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    stats_before = run_pipeline(tmp_path, dict(config))
    assert stats_before["checked"] == 2 and stats_before["failed"] == 0

    before_payload = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    before_records = {f["path"]: f for f in before_payload["files"]}
    assert set(before_records) == {"good.py", "other.py"}

    # Second run: add a ceiling-breaching third file. good.py/other.py are
    # byte-unchanged, so ordinary same-path reuse must apply to them without
    # any new provider call.
    (tmp_path / "bad.py").write_text("BAD_MARKER = 1\n", encoding="utf-8")
    monkeypatch.setattr(rca, "_FIELD_SUMMARY_CEILING_CHARS", 0)
    provider_run2 = _CeilingBreachOnMarkerProvider(bad_marker="BAD_MARKER")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: provider_run2)

    stats_after = run_pipeline(tmp_path, dict(config))

    assert stats_after["checked"] == 0  # both prior files reused, not "checked" again
    assert stats_after["failed"] == 1
    assert stats_after["skipped"] == 2  # good.py/other.py reused unchanged
    assert stats_after["response_correction_calls_attempted"] == 0
    # Neither preserved file's content triggered a provider call this run.
    assert provider_run2.calls == 1  # only bad.py

    after_payload = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )
    after_records = {f["path"]: f for f in after_payload["files"]}
    assert after_records["good.py"] == before_records["good.py"]
    assert after_records["other.py"] == before_records["other.py"]
    assert "bad.py" not in after_records


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
