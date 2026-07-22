"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import CorrectionLedger
from codedoc.core.execution import _process_one_file
from tests.support.execution_requests import make_execution_request

_INVALID_COMBINED = json.dumps({"role_in_system": "r"})           # missing description

_VALID_COMBINED = json.dumps({"description": "A file.", "role_in_system": "r"})

class _FakeBase:
    """A network-free provider fake that returns invalid-then-valid combined JSON."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt, system=""):
        self.calls += 1
        return _INVALID_COMBINED if self.calls == 1 else _VALID_COMBINED

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

class FakeOpenAI(_FakeBase):
    provider_name = "openai"

class FakeAnthropic(_FakeBase):
    provider_name = "anthropic"

class FakeGemini(_FakeBase):
    provider_name = "gemini"

_FAKES = [FakeOpenAI, FakeAnthropic, FakeGemini]

def _request(tmp_path):
    return make_execution_request(tmp_path, "m.py", "x = 1\n")

@pytest.mark.parametrize("fake_cls", _FAKES)
def test_correction_eligibility_and_counts_are_identical(tmp_path, fake_cls):
    prov = fake_cls()
    orch = Orchestrator(
        prov, analysis_mode="single",
        correction_ledger=CorrectionLedger(True), response_correction_enabled=True,
    )
    res = _process_one_file(_request(tmp_path), orch)
    assert prov.calls == 2
    assert res["description"] == "A file."

@pytest.mark.parametrize("fake_cls", _FAKES)
def test_disabled_reason_code_is_identical(tmp_path, fake_cls):
    from codedoc.utils.errors import ResponseContractError

    prov = fake_cls()
    orch = Orchestrator(
        prov, analysis_mode="single",
        correction_ledger=CorrectionLedger(False), response_correction_enabled=False,
    )
    with pytest.raises(ResponseContractError) as caught:
        _process_one_file(_request(tmp_path), orch)
    # At the execution boundary the carried diagnostic is the bounded summary dict.
    assert caught.value.diagnostic["reason_code"] == "missing_required"
    assert prov.calls == 1
