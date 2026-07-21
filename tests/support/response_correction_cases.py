"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import json
import re
from pathlib import Path
from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import (
    CorrectionLedger,
)
from codedoc.core.execution_model import FileExecutionRequest
from tests.support.execution_requests import make_execution_request

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

_INVALID = {
    "combined": {"role_in_system": "r"},
    "structure": {"exports": "not-a-list"},
    "dependency": {"dependencies_analysis": {}},
    "documentation": {"role_in_system": "r"},
}

_VALID = {
    "combined": {"description": "A file.", "role_in_system": "r",
                 "functions": [{"name": "f", "description": "does f"}]},
    "structure": {"description": "A file.", "functions": [{"name": "f", "description": "d"}]},
    "dependency": {"dependencies_analysis": {"internal": ["./util"]}},
    "documentation": {"description": "A file.", "key_concepts": ["kc"]},
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

def _request(tmp_path: Path, *, mode: str = "single") -> FileExecutionRequest:
    return make_execution_request(tmp_path, "m.py", "x = 1\n", analysis_mode=mode)

def _orch(provider, *, mode="single", enabled, parallel=False):
    return Orchestrator(
        provider, parallel=parallel, analysis_mode=mode,
        correction_ledger=CorrectionLedger(enabled),
        response_correction_enabled=enabled,
    )
