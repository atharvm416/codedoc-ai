"""Section 3 (plan section 5.6 / workstream F): the conservative narrative
terminology rules reach every applicable initial prompt, the shared correction
prompt for those routes, and cannot be removed by a custom prompt profile.
"""

from __future__ import annotations

import pytest

from codedoc.agents.narrative_terminology import NARRATIVE_TERMINOLOGY_RULES
from codedoc.core.file_division import build_division_plan
from codedoc.core.execution_model import UnitChunkExecutionRequest
from tests.support.execution_requests import make_execution_request

# Each clause the shared rules block must carry, checked line by line so a
# reworded prompt cannot quietly drop one.
_RULE_MARKERS = (
    "Do not expand an acronym unless its expansion appears verbatim",
    "keep the acronym as written",
    'reserve "entry point" for visible startup or bootstrap',
    "Do not turn a filename into a function, class, component",
)


def _assert_rules_present(text: str) -> None:
    assert NARRATIVE_TERMINOLOGY_RULES in text
    for marker in _RULE_MARKERS:
        assert marker in text


# ---------------------------------------------------------------------------
# initial prompts
# ---------------------------------------------------------------------------


def test_combined_initial_prompt_carries_the_terminology_rules():
    from codedoc.agents.file_documentation_agent import build_prompt

    _system, prompt = build_prompt("src/App.tsx", "const DPR = 1;\n", ["react"], "tsx")
    _assert_rules_present(prompt)


def test_structure_initial_prompt_carries_the_terminology_rules():
    from codedoc.agents.structure_agent import build_prompt

    _system, prompt = build_prompt("src/App.tsx", "const DPR = 1;\n", ["react"], "tsx")
    _assert_rules_present(prompt)


def test_documentation_initial_prompt_carries_the_terminology_rules():
    from codedoc.agents.documentation_agent import build_prompt

    _system, prompt = build_prompt(
        "src/App.tsx", "const DPR = 1;\n", "tsx", {}, {}
    )
    _assert_rules_present(prompt)


def test_final_synthesis_initial_prompt_carries_the_terminology_rules():
    from codedoc.agents.file_synthesis_agent import build_prompt

    _system, prompt = build_prompt("src/App.tsx", "manifest text")
    _assert_rules_present(prompt)


def _leaf_request(tmp_path) -> UnitChunkExecutionRequest:
    source = "\n".join(f"def fn_{i}():\n    return {i}" for i in range(20)) + "\n"
    file_request = make_execution_request(
        tmp_path, "src/large.py", source, max_content_chars=200
    )
    plan = build_division_plan(
        rel_path="src/large.py",
        language="python",
        content=source,
        source_budget_chars=200,
    )
    chunk = plan.chunks[0]
    return UnitChunkExecutionRequest(
        rel_path="src/large.py",
        language="python",
        full_content_hash=file_request.content_hash,
        division_plan_digest=plan.plan_digest,
        chunk_id=chunk.chunk_id,
        unit_id=chunk.unit_id,
        semantic_units=chunk.semantic_units,
        unit_indexes=plan.unit_positions(chunk),
        unit_count=len(plan.units),
        unit_chunk_index=chunk.unit_chunk_index,
        unit_chunk_count=chunk.unit_chunk_count,
        global_index=chunk.global_index,
        global_count=chunk.global_count,
        owning_ranges=chunk.owning_ranges,
        continuation_before=chunk.continuation_before,
        continuation_after=chunk.continuation_after,
        known_symbols=chunk.known_symbols,
        payload=chunk.payload,
        context=file_request.context,
    )


def test_split_leaf_initial_prompt_carries_the_terminology_rules(tmp_path):
    from codedoc.agents.file_documentation_agent import build_fragment_prompt

    _system, prompt = build_fragment_prompt(_leaf_request(tmp_path))
    _assert_rules_present(prompt)


def test_internal_reduction_prompt_is_not_a_terminology_target():
    # Plan explicitly excludes the internal reduction prompt; it must stay as-is.
    from codedoc.agents.file_synthesis_agent import _REDUCTION_PROMPT_TEMPLATE

    assert NARRATIVE_TERMINOLOGY_RULES not in _REDUCTION_PROMPT_TEMPLATE


def test_dependency_prompt_is_not_a_terminology_target():
    from codedoc.agents.dependency_agent import _PROMPT_TEMPLATE

    assert NARRATIVE_TERMINOLOGY_RULES not in _PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# correction prompts (the real construction path, not a constant search)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, agent, expected",
    [
        ("single", "combined", True),          # single + split final synthesis
        ("triple", "structure", True),
        ("triple", "documentation", True),
        ("split-leaf", "leaf", True),
        ("triple", "dependency", False),
        ("split-reduction", "reduction", False),
    ],
)
def test_correction_prompt_route_gating(mode, agent, expected):
    from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
    from codedoc.agents.response_diagnostics import ResponseDiagnostic

    corrector = ResponseCorrectionAgent(llm=None, usage=None, ledger=_Ledger(), enabled=True)
    diag = ResponseDiagnostic(
        stage="required_fields", reason_code="missing_required",
        agent=agent, file_path="m.py",
    )
    correction_input = {
        "mode": mode, "agent": agent, "language": "tsx", "file_path": "m.py",
        "content": "const DPR = 1;\n", "imports": [], "shape_block": "SHAPE",
        "original_response": "{}",
    }
    _system, prompt = corrector._build_prompt(correction_input, diag)
    if expected:
        _assert_rules_present(prompt)
    else:
        assert NARRATIVE_TERMINOLOGY_RULES not in prompt


class _Ledger:
    def record_contract_failure(self):
        pass


# ---------------------------------------------------------------------------
# custom profiles cannot remove or override the rules
# ---------------------------------------------------------------------------


def test_custom_profile_cannot_remove_terminology_rules_from_combined_prompt():
    from codedoc.core.prompt_profiles import ResolvedShapeBlock
    from codedoc.agents.file_documentation_agent import build_prompt

    shape = ResolvedShapeBlock(
        text='ONLY return: { "description": "<one sentence>" }',
        digest="deadbeef",
        active=True,
        requested_field_paths=("description",),
    )
    _system, prompt = build_prompt(
        "src/App.tsx", "const DPR = 1;\n", ["react"], "tsx", shape
    )
    _assert_rules_present(prompt)


# ---------------------------------------------------------------------------
# entry-point wording / filename-not-a-declaration are prompt-level only
# ---------------------------------------------------------------------------


def test_every_applicable_prompt_reserves_entry_point_for_startup_not_a_root_component():
    # Requirement 15: the rule that a root component is not the project entry
    # point solely from its filename or component role reaches every applicable
    # initial prompt.
    from codedoc.agents.file_documentation_agent import build_prompt as combined
    from codedoc.agents.structure_agent import build_prompt as structure
    from codedoc.agents.documentation_agent import build_prompt as documentation
    from codedoc.agents.file_synthesis_agent import build_prompt as synthesis

    src = "export const App = () => null;\n"
    prompts = [
        combined("src/App.tsx", src, ["react"], "tsx")[1],
        structure("src/App.tsx", src, ["react"], "tsx")[1],
        documentation("src/App.tsx", src, "tsx", {}, {})[1],
        synthesis("src/App.tsx", "manifest")[1],
    ]
    for prompt in prompts:
        assert 'reserve "entry point" for visible startup or bootstrap' in prompt
        assert (
            "Do not call a file \"the entry point\" only because it defines or "
            "exports a root component" in prompt
        )


def test_terminology_rules_do_not_add_a_filename_or_prose_heuristic():
    # The rules are prompt guidance; there is no deterministic validator that
    # rejects prose merely for saying "entry point" or naming a file, so
    # legitimate startup prose is never removed.
    from codedoc.agents.narrative_terminology import (
        TerminologyEvidence,
        validate_narrative_terminology,
    )

    cleaned = {
        "description": "App.tsx is the entry point and defines the App component.",
        "role_in_system": "App is the root component of the UI.",
        "usage_example": "Run App.tsx to bootstrap the interface.",
    }
    evidence = TerminologyEvidence(source_text="export const App = () => null;\n")
    out, removed = validate_narrative_terminology(cleaned, evidence)
    assert out == cleaned
    assert removed == []
