"""0.12.1 — targeted response correction behavior and call budgets.

Proves the single-mode 1/2/2 and triple-mode 3/4/6 call counts, sibling
isolation, the disabled/exhausted no-retry guarantees, correction-call fault
handling, and that no raw response or internal marker reaches a completed record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import (
    MAX_CORRECTION_RESPONSE_CHARS,
    CorrectionLedger,
)
from codedoc.core.execution import _process_one_file, _process_one_file_with_retries
from codedoc.utils.errors import ResponseContractError, UnrecoverableProviderError

_VALID = {
    "combined": {"description": "A file.", "role_in_system": "r",
                 "functions": [{"name": "f", "description": "does f"}]},
    "structure": {"description": "A file.", "functions": [{"name": "f", "description": "d"}]},
    "dependency": {"dependencies_analysis": {"internal": ["./util"]}},
    "documentation": {"description": "A file.", "key_concepts": ["kc"]},
}
# Invalid initial responses (missing required description / no usable fields).
_INVALID = {
    "combined": {"role_in_system": "r"},
    "structure": {"exports": "not-a-list"},
    "dependency": {"dependencies_analysis": {}},
    "documentation": {"role_in_system": "r"},
}


def _agent_of(prompt: str) -> str:
    m = re.search(r"The previous (single|triple)/(\w+) response", prompt)
    if m:
        return m.group(2)
    if "single JSON object" in prompt:
        return "combined"
    if "Generate documentation" in prompt:
        return "documentation"
    if "Analyse the imports" in prompt:
        return "dependency"
    return "structure"


class RoutingProvider:
    """Routes scripted responses per agent; can fail an agent's initial call."""

    provider_name = "fake"

    def __init__(
        self,
        fail_agents=(),
        correction_response=None,
        raise_on_correction=None,
        raise_agents=None,
    ):
        self.fail_agents = set(fail_agents)
        self.correction_response = correction_response  # override corrected reply
        self.raise_on_correction = raise_on_correction  # exception for correction calls
        self.raise_agents = dict(raise_agents or {})
        self.calls = 0
        self.per_agent_initial = {}
        self.correction_calls = 0
        self.last_correction_prompt = None

    def complete_json(self, prompt, system=""):
        self.calls += 1
        agent = _agent_of(prompt)
        is_correction = "Previous response (verbatim" in prompt
        if is_correction:
            self.correction_calls += 1
            self.last_correction_prompt = prompt
            if self.raise_on_correction is not None:
                raise self.raise_on_correction
            if self.correction_response is not None:
                return json.dumps(self.correction_response)
            return json.dumps(_VALID[agent])
        self.per_agent_initial[agent] = self.per_agent_initial.get(agent, 0) + 1
        if agent in self.raise_agents:
            raise self.raise_agents[agent]
        if agent in self.fail_agents:
            return json.dumps(_INVALID[agent])
        return json.dumps(_VALID[agent])

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _descriptor(tmp_path: Path) -> dict:
    p = tmp_path / "m.py"
    p.write_text("x = 1\n", encoding="utf-8")
    return {"path": p, "rel_path": "m.py", "language": "python", "extension": ".py"}


def _orch(provider, *, mode="single", enabled, parallel=False):
    return Orchestrator(
        provider, parallel=parallel, analysis_mode=mode,
        correction_ledger=CorrectionLedger(enabled),
        response_correction_enabled=enabled,
    )


# ---------------------------------------------------------------------------
# Single-mode call counts (1 / 2 / 2)
# ---------------------------------------------------------------------------

def test_single_valid_one_call(tmp_path):
    prov = RoutingProvider()
    res = _process_one_file(_descriptor(tmp_path), _orch(prov, enabled=False))
    assert prov.calls == 1
    assert res["description"] == "A file."


def test_single_invalid_then_corrected_two_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=True)
    res = _process_one_file(_descriptor(tmp_path), orch)
    assert prov.calls == 2
    assert prov.correction_calls == 1
    assert res["description"] == "A file."


def test_single_invalid_then_failed_two_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"}, correction_response={"role_in_system": "r"})
    orch = _orch(prov, enabled=True)
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_descriptor(tmp_path), orch)
    assert prov.calls == 2
    assert prov.correction_calls == 1
    assert caught.value.correction_attempted is True


def test_single_disabled_no_correction_call(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=False)
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_descriptor(tmp_path), orch)
    assert prov.calls == 1
    assert prov.correction_calls == 0
    assert caught.value.correction_attempted is False


# ---------------------------------------------------------------------------
# Triple-mode isolation and call counts (3 / 4 / 6)
# ---------------------------------------------------------------------------

def test_triple_all_valid_three_calls(tmp_path):
    prov = RoutingProvider()
    res = _process_one_file(_descriptor(tmp_path), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 3
    assert res["state"] == "checked"


def test_triple_one_invalid_corrected_four_calls_sibling_isolation(tmp_path):
    prov = RoutingProvider(fail_agents={"structure"})
    res = _process_one_file(_descriptor(tmp_path), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 4
    assert prov.correction_calls == 1
    # Successful siblings are each called exactly once, never rerun.
    assert prov.per_agent_initial["dependency"] == 1
    assert prov.per_agent_initial["documentation"] == 1
    assert prov.per_agent_initial["structure"] == 1
    assert res["state"] == "checked"


def test_triple_all_three_invalid_six_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"structure", "dependency", "documentation"})
    res = _process_one_file(_descriptor(tmp_path), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 6
    assert prov.correction_calls == 3
    assert res["state"] == "checked"


# ---------------------------------------------------------------------------
# Correction input and preservation
# ---------------------------------------------------------------------------

def test_correction_prompt_includes_original_response_and_is_capped(tmp_path):
    huge = "z" * (MAX_CORRECTION_RESPONSE_CHARS + 5000)
    prov = RoutingProvider(fail_agents={"combined"})
    # Force a huge original response by returning it on the initial call.
    orig = json.dumps({"role_in_system": huge})

    class HugeInitial(RoutingProvider):
        def complete_json(self, prompt, system=""):
            if "Previous response (verbatim" not in prompt:
                self.calls += 1
                self.per_agent_initial["combined"] = 1
                return orig
            return super().complete_json(prompt, system)

    prov = HugeInitial(fail_agents={"combined"})
    _process_one_file(_descriptor(tmp_path), _orch(prov, enabled=True))
    # The correction prompt embedded the capped original, never the full text: a
    # long run of the original survives, but the full oversized string does not.
    assert prov.last_correction_prompt is not None
    assert huge not in prov.last_correction_prompt
    assert ("z" * 6000) in prov.last_correction_prompt
    assert prov.last_correction_prompt.count("z") <= MAX_CORRECTION_RESPONSE_CHARS


def test_correction_preserves_usable_facts(tmp_path):
    # Initial response has a valid role but no description; correction fills it.
    prov = RoutingProvider(
        fail_agents={"combined"},
        correction_response={"description": "Filled.", "role_in_system": "kept role"},
    )
    res = _process_one_file(_descriptor(tmp_path), _orch(prov, enabled=True))
    assert res["description"] == "Filled."
    assert res["role_in_system"] == "kept role"


# ---------------------------------------------------------------------------
# Correction-call provider faults
# ---------------------------------------------------------------------------

def test_correction_call_billing_fault_is_terminal(tmp_path):
    # A billing fault on the correction call is a run-level abort routed through the
    # existing terminal path (which lives in the retry wrapper, not _process_one_file).
    prov = RoutingProvider(
        fail_agents={"combined"},
        raise_on_correction=RuntimeError("insufficient_quota: credit balance is too low"),
    )
    with pytest.raises(UnrecoverableProviderError):
        _process_one_file_with_retries(
            _descriptor(tmp_path), _orch(prov, enabled=True), retry_attempts=1
        )
    assert prov.correction_calls == 1


def test_correction_call_rate_limit_is_final_not_retried(tmp_path):
    prov = RoutingProvider(
        fail_agents={"combined"},
        raise_on_correction=RuntimeError("rate limit exceeded (429)"),
    )
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_descriptor(tmp_path), _orch(prov, enabled=True))
    assert prov.correction_calls == 1
    assert caught.value.correction_attempted is True


def test_terminal_sibling_error_is_not_masked_by_contract_failure(tmp_path):
    prov = RoutingProvider(
        fail_agents={"structure"},
        raise_agents={"dependency": RuntimeError("invalid api key")},
    )
    with pytest.raises(UnrecoverableProviderError):
        _process_one_file_with_retries(
            _descriptor(tmp_path),
            _orch(prov, mode="triple", enabled=False, parallel=False),
            retry_attempts=1,
        )
    assert prov.per_agent_initial["structure"] == 1
    assert prov.per_agent_initial["dependency"] == 1


# ---------------------------------------------------------------------------
# Pipeline statistics, dry-run cost ceiling, parallel routing, and cache reuse
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize(
    ("enabled", "possible_extra", "maximum"),
    [(False, 0, 1), (True, 1, 2)],
)
def test_dry_run_reports_baseline_and_correction_ceiling(
    tmp_path, monkeypatch, enabled, possible_extra, maximum
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("dry-run must not construct a provider"),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "dry_run": True,
            "response_correction_enabled": enabled,
            "propagate_changes": False,
        },
    )

    assert stats["estimated_calls"] == 1
    assert stats["response_correction_enabled"] is enabled
    assert stats["response_correction_calls_possible_max"] == possible_extra
    assert stats["estimated_calls_max_with_correction"] == maximum


def test_parallel_contract_failures_do_not_enter_sequential_retry(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "parallel_agents": True,
            "max_parallel_files": 2,
            "file_retry_attempts": 2,
            "response_correction_enabled": False,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["failed"] == 2
    assert stats["attempted_calls"] == 2
    assert stats["response_contract_failures"] == 2


def test_corrected_successful_record_is_reused_from_cache(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    first = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "response_correction_enabled": True,
            "propagate_changes": False,
        },
    )
    assert first["checked"] == 1
    assert first["response_correction_calls_succeeded"] == 1

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("unchanged corrected record must be reusable"),
    )
    second = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "response_correction_enabled": True,
            "propagate_changes": False,
        },
    )
    assert second["checked"] == 0
    assert second["documentation_calls_attempted"] == 0


# ---------------------------------------------------------------------------
# Pipeline: no marker / raw response in completed output
# ---------------------------------------------------------------------------

def test_no_marker_or_raw_response_in_completed_output(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    pipeline.run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "response_correction_enabled": True,
         "parallel_agents": False, "max_parallel_files": 1, "propagate_changes": False},
    )
    text = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    assert "response_contract_final" not in text
    assert "response_contract_diagnostic" not in text
