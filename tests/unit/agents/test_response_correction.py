"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from codedoc.agents.response_diagnostics import (
    MAX_CORRECTION_RESPONSE_CHARS,
)
from codedoc.core.execution import _process_one_file
from codedoc.utils.errors import ResponseContractError
from tests.support.response_correction_cases import RoutingProvider
from tests.support.response_correction_cases import _orch
from tests.support.response_correction_cases import _request

def test_single_valid_one_call(tmp_path):
    prov = RoutingProvider()
    res = _process_one_file(_request(tmp_path), _orch(prov, enabled=False))
    assert prov.calls == 1
    assert res["description"] == "A file."

def test_single_invalid_then_corrected_two_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=True)
    res = _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 2
    assert prov.correction_calls == 1
    assert res["description"] == "A file."

def test_single_invalid_then_failed_two_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"}, correction_response={"role_in_system": "r"})
    orch = _orch(prov, enabled=True)
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 2
    assert prov.correction_calls == 1
    assert caught.value.correction_attempted is True

def test_single_disabled_no_correction_call(tmp_path):
    prov = RoutingProvider(fail_agents={"combined"})
    orch = _orch(prov, enabled=False)
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 1
    assert prov.correction_calls == 0
    assert caught.value.correction_attempted is False

def test_triple_all_valid_three_calls(tmp_path):
    prov = RoutingProvider()
    res = _process_one_file(_request(tmp_path, mode="triple"), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 3
    assert res["state"] == "checked"

def test_triple_one_invalid_corrected_four_calls_sibling_isolation(tmp_path):
    prov = RoutingProvider(fail_agents={"structure"})
    res = _process_one_file(_request(tmp_path, mode="triple"), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 4
    assert prov.correction_calls == 1
    # Successful siblings are each called exactly once, never rerun.
    assert prov.per_agent_initial["dependency"] == 1
    assert prov.per_agent_initial["documentation"] == 1
    assert prov.per_agent_initial["structure"] == 1
    assert res["state"] == "checked"

def test_triple_all_three_invalid_six_calls(tmp_path):
    prov = RoutingProvider(fail_agents={"structure", "dependency", "documentation"})
    res = _process_one_file(_request(tmp_path, mode="triple"), _orch(prov, mode="triple", enabled=True))
    assert prov.calls == 6
    assert prov.correction_calls == 3
    assert res["state"] == "checked"

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
    _process_one_file(_request(tmp_path), _orch(prov, enabled=True))
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
    res = _process_one_file(_request(tmp_path), _orch(prov, enabled=True))
    assert res["description"] == "Filled."
    assert res["role_in_system"] == "kept role"
