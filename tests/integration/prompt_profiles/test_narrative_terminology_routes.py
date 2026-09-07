"""Section 3 (plan section 5.6 / workstream F): the conservative terminology
rule reaches the *real* rejection -> repair path for every applicable route.

Each test drives the actual agent / finalization / correction machinery with a
deterministic fake provider -- no direct ``_build_prompt`` call and no constant
search. It proves: the initial response carries a provably unsupported
expansion; canonical validation rejects it; the shared
``ResponseCorrectionAgent.repair`` runs exactly once; that real correction
prompt carries ``NARRATIVE_TERMINOLOGY_RULES``; the corrected response is
re-validated against the same ``TerminologyEvidence``; a clean corrected value
is published; the unsupported phrase never enters the record; and no raw
source/response text reaches diagnostics.
"""

from __future__ import annotations

import json

import pytest

from codedoc.agents.narrative_terminology import NARRATIVE_TERMINOLOGY_RULES
from codedoc.agents.orchestrator import Orchestrator
from codedoc.agents.response_diagnostics import CorrectionLedger
from codedoc.pipeline import run_pipeline
from tests.support.execution_requests import make_execution_request

_BAD = "shows the DPR (Daily Progress Report) panel"
_GOOD = "shows the DPR panel"
_PHRASE = "Daily Progress Report"

# `DPR` is an uppercase source token on every line, so it is in the evidence for
# the ordinary routes (truncated source) and for every split chunk payload.
_SRC = "const DPR = 1;\n" + "".join(
    f"export function unit_{i:02d}() {{ return DPR + {i}; }}\n" for i in range(6)
)
_SPLIT_SRC = "const DPR = 1;\n" + "".join(
    f"export function unit_{i:03d}() {{ return DPR + {i}; }}\n" for i in range(90)
)


def _is_correction(prompt: str) -> bool:
    return "Previous response (verbatim" in prompt


# ===========================================================================
# triple structure / triple documentation
# ===========================================================================


class _TripleProvider:
    def __init__(self, bad_agent: str, corrected_ok: bool = True):
        self.bad_agent = bad_agent
        self.corrected_ok = corrected_ok
        self.provider_name = "fake"
        self.correction_calls = 0
        self.correction_prompts: list[str] = []

    def complete_json(self, prompt, system=""):
        if _is_correction(prompt):
            self.correction_calls += 1
            self.correction_prompts.append(prompt)
            body = _BAD if not self.corrected_ok else _GOOD
            return json.dumps({"description": body})
        if "Analyse the imports" in prompt:
            return json.dumps({"dependencies_analysis": {"internal": [], "external": []}})
        if "Generate documentation" in prompt:  # documentation agent
            desc = _BAD if self.bad_agent == "documentation" else "clean doc description"
            return json.dumps({"description": desc})
        # structure agent -- description only, so a rejection empties the
        # response and the real repair path is entered.
        desc = _BAD if self.bad_agent == "structure" else "clean structure description"
        return json.dumps({"description": desc})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _triple_orch(provider, *, enabled=True):
    return Orchestrator(
        provider,
        parallel=False,
        analysis_mode="triple",
        correction_ledger=CorrectionLedger(enabled),
        response_correction_enabled=enabled,
    )


@pytest.mark.parametrize("bad_agent", ["structure", "documentation"])
def test_triple_route_real_rejection_and_one_correction_publishes_clean(
    tmp_path, bad_agent
):
    provider = _TripleProvider(bad_agent=bad_agent, corrected_ok=True)
    orch = _triple_orch(provider)
    request = make_execution_request(
        tmp_path, "svc.ts", _SRC, language="typescript", analysis_mode="triple"
    )
    result = orch.process(request)

    assert provider.correction_calls == 1
    assert provider.correction_prompts
    assert NARRATIVE_TERMINOLOGY_RULES in provider.correction_prompts[0]

    blob = json.dumps(result)
    assert _PHRASE not in blob
    if bad_agent == "documentation":
        assert result["description"] == _GOOD
    else:
        assert result["structure"]["description"] == _GOOD
    # bounded diagnostic (if surfaced) carries no phrase or source text
    diag = result.get("documentation", {}).get("response_contract_diagnostic")
    if diag:
        db = json.dumps(diag)
        assert _PHRASE not in db and "const DPR = 1" not in db


@pytest.mark.parametrize("bad_agent", ["structure", "documentation"])
def test_triple_route_second_bad_response_fails_after_exactly_one_correction(
    tmp_path, bad_agent
):
    provider = _TripleProvider(bad_agent=bad_agent, corrected_ok=False)
    orch = _triple_orch(provider)
    request = make_execution_request(
        tmp_path, "svc.ts", _SRC, language="typescript", analysis_mode="triple"
    )
    result = orch.process(request)
    assert provider.correction_calls == 1
    assert _PHRASE not in json.dumps(result)
    if bad_agent == "structure":
        assert result["structure"].get("error") or result["structure"] == {}
    else:
        assert result["documentation"].get("error")


# ===========================================================================
# split leaf / split final synthesis (real run_pipeline)
# ===========================================================================


class _SplitProvider:
    """`bad_stage` is "leaf" or "synthesis"; only the first bad-stage response
    is unsupported, so exactly one correction call occurs for that route."""

    def __init__(self, bad_stage: str, corrected_ok: bool = True):
        self.bad_stage = bad_stage
        self.corrected_ok = corrected_ok
        self.provider_name = "fake"
        self.correction_calls = 0
        self.correction_prompts: list[str] = []
        self._first_bad_emitted = False

    def complete_json(self, prompt, system=""):
        if _is_correction(prompt):
            self.correction_calls += 1
            self.correction_prompts.append(prompt)
            body = _BAD if not self.corrected_ok else _GOOD
            return json.dumps({"description": body})
        if "Analyse the imports" in prompt:
            return json.dumps({"dependencies_analysis": {"internal": [], "external": []}})
        if "This is one bounded fragment of a larger" in prompt:
            desc = "a bounded fragment"
            if self.bad_stage == "leaf" and not self._first_bad_emitted:
                self._first_bad_emitted = True
                desc = _BAD
            return json.dumps({"description": desc})
        if "Refine one combined narrative from" in prompt:
            return json.dumps({"narrative": "a refined narrative"})
        # final synthesis
        desc = "a synthesized file description"
        if self.bad_stage == "synthesis" and not self._first_bad_emitted:
            self._first_bad_emitted = True
            desc = _BAD
        return json.dumps(
            {
                "description": desc,
                "role_in_system": "core",
                "key_concepts": ["split"],
                "usage_example": "import x",
            }
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _run_split(tmp_path, monkeypatch, provider):
    (tmp_path / "big.ts").write_text(_SPLIT_SRC, encoding="utf-8", newline="")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _cfg: provider
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "big.ts",
            "documentation_scope": "entry",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 1500,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    payload = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    files = payload.get("files") or []
    return stats, (files[0] if files else None)


@pytest.mark.parametrize("bad_stage", ["leaf", "synthesis"])
def test_split_route_real_rejection_and_one_correction_publishes_clean(
    tmp_path, monkeypatch, bad_stage
):
    provider = _SplitProvider(bad_stage=bad_stage, corrected_ok=True)
    stats, record = _run_split(tmp_path, monkeypatch, provider)

    assert stats["checked"] == 1 and stats["failed"] == 0
    assert provider.correction_calls == 1
    correction_prompt = provider.correction_prompts[0]
    assert NARRATIVE_TERMINOLOGY_RULES in correction_prompt
    if bad_stage == "leaf":
        assert "split-leaf" in correction_prompt
    assert _PHRASE not in json.dumps(record)


@pytest.mark.parametrize("bad_stage", ["leaf", "synthesis"])
def test_split_route_second_bad_response_fails_after_exactly_one_correction(
    tmp_path, monkeypatch, bad_stage
):
    provider = _SplitProvider(bad_stage=bad_stage, corrected_ok=False)
    stats, record = _run_split(tmp_path, monkeypatch, provider)
    # The one bad node is repaired once; the still-bad corrected response is
    # re-validated against the same evidence and the node fails terminally.
    assert provider.correction_calls == 1
    assert stats["failed"] == 1 and stats["checked"] == 0
    assert record is None or _PHRASE not in json.dumps(record)


def test_split_final_synthesis_manifest_is_not_trusted_evidence(
    tmp_path, monkeypatch
):
    # The unsupported phrase reaches the synthesis manifest only through a
    # reduction narrative -- never through source. The synthesis validator uses
    # the reconstructed planned source via ``terminology_source`` only, so the
    # synthesis response that repeats the phrase is still rejected and repaired.
    class _ManifestOnly:
        provider_name = "fake"

        def __init__(self):
            self.correction_calls = 0
            self._synth_bad_emitted = False

        def complete_json(self, prompt, system=""):
            if _is_correction(prompt):
                self.correction_calls += 1
                return json.dumps({"description": "a clean synthesized description"})
            if "Analyse the imports" in prompt:
                return json.dumps(
                    {"dependencies_analysis": {"internal": [], "external": []}}
                )
            if "This is one bounded fragment of a larger" in prompt:
                return json.dumps({"description": "a bounded fragment"})
            if "Refine one combined narrative from" in prompt:
                # not a terminology route; the phrase flows into the manifest
                return json.dumps({"narrative": f"narrative mentioning {_PHRASE}"})
            desc = "a synthesized description"
            if not self._synth_bad_emitted:
                self._synth_bad_emitted = True
                desc = f"final: the {_PHRASE} for DPR"
            return json.dumps({"description": desc})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    provider = _ManifestOnly()
    stats, record = _run_split(tmp_path, monkeypatch, provider)
    assert provider.correction_calls == 1
    assert stats["checked"] == 1
    assert record is not None and _PHRASE not in json.dumps(record)
