"""Tests organized by feature ownership."""

from __future__ import annotations

from codedoc.agents import (
    documentation_agent,
    file_documentation_agent,
    structure_agent,
)
from codedoc.agents import (
    dependency_agent,
)
from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES

def test_both_prompt_families_share_symbol_and_usage_definitions():
    fda_system, fda_prompt = file_documentation_agent.build_prompt(
        "a.py", "code", ["os"], "python"
    )
    struct_system, struct_prompt = structure_agent.build_prompt(
        "a.py", "code", ["os"], "python"
    )
    doc_system, doc_prompt = documentation_agent.build_prompt(
        "a.py", "code", "python", {}, {}
    )
    # Local-symbol + re-export definitions appear in both structure prompts.
    for prompt in (fda_prompt, struct_prompt):
        assert "DEFINED IN this file" in prompt
        assert "re-exports" in prompt
    # Usage-example factuality wording appears in both prompts that emit one.
    for prompt in (fda_prompt, doc_prompt):
        assert "Include usage_example only when" in prompt
        assert "path/to/file" in prompt
    # Both families warn the model about the truncation marker.
    for prompt in (fda_prompt, struct_prompt, doc_prompt):
        assert "[truncated]" in prompt

_CLAUSES = tuple(EXACT_JSON_RESPONSE_RULES.split("\n"))

def test_all_four_prompts_contain_every_strengthened_clause():
    prompts = {
        "combined": file_documentation_agent.build_prompt("m.py", "x=1", ["os"], "python")[1],
        "structure": structure_agent.build_prompt("m.py", "x=1", ["os"], "python")[1],
        "dependency": dependency_agent.build_prompt("m.py", "x=1", ["os"], "python")[1],
        "documentation": documentation_agent.build_prompt("m.py", "x=1", "python", {}, {})[1],
    }
    for name, prompt in prompts.items():
        for clause in _CLAUSES:
            assert clause in prompt, f"{name} missing: {clause}"
