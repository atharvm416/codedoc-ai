from __future__ import annotations

import json

import pytest

from codedoc.agents.file_synthesis_agent import (
    FileSynthesisAgent,
    build_prompt,
    build_reduction_prompt,
)
from codedoc.core.execution_model import FileReductionExecutionRequest
from codedoc.core.file_division import (
    MAX_REDUCTION_NARRATIVE_CHARS,
    REDUCTION_ENVELOPE_OVERHEAD_CHARS,
    DivisionInternalDefect,
    build_fact_ledger,
    final_synthesis_input,
    worst_case_reduction_manifest_chars,
)
from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedProfile,
    resolved_synthesis_shape,
)
from tests.support.execution_requests import make_execution_request


class _Provider:
    provider_name = "synthesis-test"

    def __init__(self, response: dict | None = None) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self.response = response or {"description": "Synthesized file."}

    def complete_json(self, prompt, system=""):
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps(self.response)

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _shape(mode: str = "single"):
    bundle = ResolvedProfile(mode, None).resolve_bundle(
        FileScope(basename="large.py")
    )
    return bundle, resolved_synthesis_shape(bundle)


def _manifest(rel_path: str = "src/large.py") -> str:
    ledger = build_fact_ledger(
        [{"description": "leaf", "functions": [{"name": "run", "description": "Runs."}]}],
        language="python",
    )
    return final_synthesis_input(
        rel_path=rel_path,
        language="python",
        imports=["os"],
        root_narratives=("Reviewed narrative.",),
        root_coverage_leaf_ids=("chunk_" + "a" * 64,),
        ledger=ledger,
    )


def test_synthesis_prompt_contains_only_manifest_contract_and_resolved_shape() -> None:
    bundle, shape = _shape()
    manifest = _manifest()

    system, prompt = build_prompt("src/large.py", manifest, shape)

    assert "senior software engineer" in system
    assert "src/large.py" in prompt
    assert manifest in prompt
    assert shape.text in prompt
    assert "Do not invent source ranges" in prompt
    assert "RAW_SOURCE_SECRET" not in prompt
    assert bundle.selections["combined"].block is shape


def test_synthesis_uses_existing_cleaner_and_rejects_model_owned_facts(
    tmp_path,
) -> None:
    request = make_execution_request(
        tmp_path,
        "src/large.py",
        "RAW_SOURCE_SECRET = True\n",
        max_content_chars=1000,
    )
    shape = resolved_synthesis_shape(request.context.resolved_shape_bundle)
    provider = _Provider(
        {
            "description": "Synthesized file.",
            "functions": [{"name": "run", "description": "Runs."}],
            "dependencies_analysis": {"external": ["requests"]},
            "file_path": "forged.py",
            "language": "forged",
            "hash": "forged",
            "state": "forged",
            "documentation_units": [{"unit_id": "forged"}],
            "division": {"strategy": "forged"},
            "relationships": [{"from": "a", "to": "b"}],
            "planner": {"task": "forged"},
            "arbitrary": "forged",
        }
    )
    agent = FileSynthesisAgent(provider, max_content_chars=1000)
    manifest = _manifest(request.rel_path)

    result = agent.run(
        request.rel_path,
        manifest,
        shape,
        call_context=request.context,
    )

    assert provider.calls == 1
    assert "RAW_SOURCE_SECRET" not in provider.prompts[0]
    assert result["description"] == "Synthesized file."
    assert result["functions"] == [{"name": "run", "description": "Runs."}]
    assert result["dependencies_analysis"]["external"] == ["requests"]
    assert not {
        "file_path",
        "language",
        "hash",
        "state",
        "documentation_units",
        "division",
        "relationships",
        "planner",
        "arbitrary",
    } & set(result)


def test_synthesis_requires_resolved_shape_before_provider_call(tmp_path) -> None:
    request = make_execution_request(tmp_path, "large.py", max_content_chars=1000)
    provider = _Provider()
    agent = FileSynthesisAgent(provider, max_content_chars=1000)

    with pytest.raises(ValueError, match="resolved shape"):
        agent.run(
            request.rel_path,
            '{"units":[]}',
            None,
            call_context=request.context,
        )

    assert provider.calls == 0


def test_synthesis_manifest_ceiling_is_enforced_before_provider_call(
    tmp_path,
) -> None:
    request = make_execution_request(tmp_path, "large.py", max_content_chars=1000)
    provider = _Provider()
    agent = FileSynthesisAgent(provider, max_content_chars=1000)
    shape = resolved_synthesis_shape(request.context.resolved_shape_bundle)

    with pytest.raises(ValueError, match="exceeds max_content_chars"):
        agent.run(
            request.rel_path,
            "x" * 1001,
            shape,
            call_context=request.context,
        )

    assert provider.calls == 0


def test_triple_synthesis_shape_is_canonical_combined_union() -> None:
    _bundle, shape = _shape("triple")

    assert shape.requested_field_paths == (
        "description",
        "role_in_system",
        "functions",
        "classes",
        "exports",
        "dependencies_analysis.internal",
        "dependencies_analysis.external",
        "dependencies_analysis.dependency_refs",
        "dependencies_analysis.catalog_updates",
        "dependencies_analysis.usage_notes",
        "dependencies_analysis.warnings",
        "key_concepts",
        "usage_example",
    )


def _reduction_request(
    rel_path: str, context, *, phase: str = "general", level: int = 1
) -> FileReductionExecutionRequest:
    return FileReductionExecutionRequest(
        rel_path=rel_path,
        division_plan_digest="division-plan:" + "1" * 64,
        reduction_tree_digest="reduction-tree:" + "2" * 64,
        node_id="node_" + "3" * 64,
        phase=phase,
        unit_id=None,
        level=level,
        ordinal=1,
        child_ids=("chunk_" + "a" * 64, "chunk_" + "b" * 64),
        child_capsules=({"description": "a"}, {"description": "b"}),
        reducer_revision="reducer-v1",
        context=context,
    )


def test_reduction_prompt_contains_only_child_narratives_and_fixed_shape(
    tmp_path,
) -> None:
    request = make_execution_request(tmp_path, "src/large.py", max_content_chars=1000)
    reduction_request = _reduction_request(request.rel_path, request.context)

    system, prompt = build_reduction_prompt(
        reduction_request, ("First narrative.", "Second narrative.")
    )

    assert "technical writer refining" in system
    assert "src/large.py" in prompt
    assert "First narrative." in prompt
    assert "Second narrative." in prompt
    assert "Reduction phase: general" in prompt
    assert "Tree level: 1" in prompt
    assert '"narrative": "<required, non-empty refined narrative>"' in prompt
    assert "Do not list functions, classes, or exports" in prompt
    assert "Synthesize one final file-level" not in prompt


def test_reduction_prompt_never_includes_whole_file_imports(tmp_path) -> None:
    request = make_execution_request(
        tmp_path, "src/large.py", "import os\nimport sys\n", max_content_chars=1000
    )
    reduction_request = _reduction_request(request.rel_path, request.context)

    _system, prompt = build_reduction_prompt(reduction_request, ("Narrative only.",))

    assert "import os" not in prompt
    assert "import sys" not in prompt


def test_reduction_uses_fixed_capsule_cleaner_and_rejects_non_narrative_fields(
    tmp_path,
) -> None:
    request = make_execution_request(tmp_path, "src/large.py", max_content_chars=1000)
    reduction_request = _reduction_request(request.rel_path, request.context)
    provider = _Provider(
        {
            "narrative": "Combined narrative.",
            "functions": [{"name": "run", "description": "forged"}],
            "description": "forged",
            "division": {"strategy": "forged"},
            "arbitrary": "forged",
        }
    )
    agent = FileSynthesisAgent(provider, max_content_chars=1000)

    result = agent.run_reduction(
        reduction_request, ("First narrative.", "Second narrative.")
    )

    assert provider.calls == 1
    assert result == {"narrative": "Combined narrative."}


def test_reduction_retry_reissues_a_byte_identical_prompt(tmp_path) -> None:
    request = make_execution_request(tmp_path, "src/large.py", max_content_chars=1000)
    reduction_request = _reduction_request(request.rel_path, request.context)
    provider = _Provider({"narrative": "Combined narrative."})
    agent = FileSynthesisAgent(provider, max_content_chars=1000)

    agent.run_reduction(reduction_request, ("First narrative.", "Second narrative."))
    agent.run_reduction(
        reduction_request,
        ("First narrative.", "Second narrative."),
        additional_attempt=True,
    )

    assert provider.calls == 2
    assert provider.prompts[0] == provider.prompts[1]


def test_reduction_runtime_rejects_manifest_above_planned_ceiling(
    tmp_path,
) -> None:
    ceiling = (
        REDUCTION_ENVELOPE_OVERHEAD_CHARS
        + worst_case_reduction_manifest_chars(2)
        - 1
    )
    request = make_execution_request(
        tmp_path,
        "src/large.py",
        max_content_chars=ceiling,
    )
    reduction_request = _reduction_request(request.rel_path, request.context)
    provider = _Provider({"narrative": "Combined narrative."})
    agent = FileSynthesisAgent(provider, max_content_chars=ceiling)
    narratives = (
        "a" * MAX_REDUCTION_NARRATIVE_CHARS,
        "b" * MAX_REDUCTION_NARRATIVE_CHARS,
    )

    with pytest.raises(
        DivisionInternalDefect,
        match="planned reduction child manifest exceeds",
    ):
        agent.run_reduction(reduction_request, narratives)

    assert provider.calls == 0
