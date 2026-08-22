"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811

import json
import pytest
from codedoc.agents.orchestrator import Orchestrator, assemble_final_result
from codedoc.agents.response_cleaning import MAX_SYMBOL_ITEMS_PER_KIND
from codedoc.core.execution import _agent_errors
from codedoc.core.file_division import FactLedger
from tests.support.execution_requests import make_execution_request
from tests.support.one_call_cases import _COMBINED
from tests.support.one_call_cases import _COMBINED_JSON as ONE_CALL_COMBINED_JSON
from tests.support.one_call_cases import _CountingProvider
from tests.support.fixture_paths import PROJECT_FIXTURES

def test_deterministic_fields_cannot_be_replaced_by_model_output(tmp_path):
    raw = json.dumps({
        **_COMBINED,
        "file_path": "EVIL.py", "language": "evil", "imports": ["malware"],
        "extension": ".evil",
    })
    request = make_execution_request(tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",))
    result = Orchestrator(_CountingProvider(raw), analysis_mode="single").process(request)
    assert result["file_path"] == "pkg/mod.py"
    assert result["language"] == "python"
    assert result["extension"] == ".py"
    assert result["imports"] == ["os"]

def test_combined_failure_produces_one_agent_error_via_documentation(tmp_path):
    request = make_execution_request(tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",))
    result = Orchestrator(_CountingProvider("not json"), analysis_mode="single").process(
        request
    )
    assert result["structure"] == {}
    assert result["dependencies_analysis"] == {}
    assert result["documentation"]["agent"] == "FileDocumentationAgent"
    assert result["documentation"]["error"]
    # Identity preserved on the failed flat record.
    assert result["file_path"] == "pkg/mod.py"
    # _agent_errors detects exactly one failure (through the documentation key).
    assert len(_agent_errors(result)) == 1
    # A failed result is not marked as a successful cache record.
    assert "_analysis_revision" not in result

@pytest.mark.parametrize("mode,expected_calls", [("single", 1), ("triple", 3)])
def test_response_contract_failure_does_not_repeat_full_call_set(
    tmp_path, monkeypatch, mode, expected_calls
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    class FirstCallMalformed(_CountingProvider):
        def complete_json(self, prompt, system=""):
            self.calls += 1
            return "not json" if self.calls == 1 else ONE_CALL_COMBINED_JSON

    provider = FirstCallMalformed()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": mode,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 0
    assert stats["failed"] == 1
    assert stats["planned_calls"] == (1 if mode == "single" else 3)
    assert stats["attempted_calls"] == expected_calls
    assert provider.calls == expected_calls

class TestOrchestratorIntegration:
    def test_all_fixtures(self, tmp_path):
        """Run orchestrator across all fixture codebases."""
        from codedoc.agents.orchestrator import Orchestrator
        import json as _json

        fixtures = [
            (PROJECT_FIXTURES / "python_app" / "main.py", "python"),
            (PROJECT_FIXTURES / "react_app" / "App.tsx", "tsx"),
            (PROJECT_FIXTURES / "java_app" / "Main.java", "java"),
            (PROJECT_FIXTURES / "flutter_app" / "main.dart", "dart"),
        ]

        class CombinedMock:
            def complete(self, prompt, system="", temperature=0.1):
                if "dependencies_analysis" in prompt:
                    return _json.dumps({"dependencies_analysis": {"internal": [], "external": [], "usage_notes": [], "warnings": []}})
                if "key_concepts" in prompt:
                    return _json.dumps({"description": "d", "role_in_system": "r", "key_concepts": [], "usage_example": ""})
                return _json.dumps({"description": "d", "role_in_system": "r", "functions": [], "classes": [], "exports": []})

            def complete_json(self, prompt, system=""):
                return self.complete(prompt, system)

            @property
            def provider_name(self):
                return "Combined"

        orch = Orchestrator(CombinedMock(), parallel=False)

        for path, language in fixtures:
            request = make_execution_request(
                tmp_path, path.name, path.read_text(), language=language, write=False
            )
            result = orch.process(request)
            assert result["state"] == "checked", f"Failed for {path.name}"
            assert result["file_path"] == path.name

class TestOrchestrator:
    def test_process_returns_merged_result(self, mock_llm, tmp_path):
        from codedoc.agents.orchestrator import Orchestrator

        request = make_execution_request(
            tmp_path,
            "App.tsx",
            "import React from 'react';\nconst App = () => <div/>;\nexport default App;\n",
            language="tsx",
            imports=("react",),
        )

        orch = Orchestrator(mock_llm, parallel=False)
        result = orch.process(request)

        assert result["file_path"] == "App.tsx"
        assert result["state"] == "checked"
        assert "imports" in result
        assert "description" in result

    def test_parallel_mode(self, mock_llm, tmp_path):
        from codedoc.agents.orchestrator import Orchestrator

        request = make_execution_request(
            tmp_path, "main.py", "import os\ndef main(): pass\n", imports=("os",)
        )
        orch = Orchestrator(mock_llm, parallel=True)
        result = orch.process(request)
        assert result["state"] == "checked"

_COMBINED_JSON = json.dumps({
    "description": "A documented module.",
    "role_in_system": "entry point",
    "functions": [{"name": "main", "description": "runs"}],
    "classes": [{"name": "C", "description": "a class"}],
    "exports": ["main"],
    "dependencies_analysis": {"external": ["requests"], "dependency_refs": ["requests"]},
    "key_concepts": ["startup"],
    "usage_example": "import mod",
})

class _FakeProvider:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return _COMBINED_JSON

    def complete(self, prompt, system="", temperature=0.1):
        return _COMBINED_JSON

def _process(tmp_path, content, *, max_chars=1000, head_ratio=0.70):
    orch = Orchestrator(
        _FakeProvider(), analysis_mode="single",
        max_content_chars=max_chars, truncation_head_ratio=head_ratio,
    )
    request = make_execution_request(
        tmp_path, "pkg/mod.py", content, imports=("os",),
        max_content_chars=max_chars, truncation_head_ratio=head_ratio,
    )
    return orch.process(request)

def test_orchestrator_stamps_revision_for_oversized_file(tmp_path):
    result = _process(tmp_path, "x" * 5000)
    assert result["state"] == "checked"
    assert result["_max_context_revision"] == "truncate-v1:max=1000:head=0.7000"

def test_orchestrator_omits_revision_for_small_file(tmp_path):
    result = _process(tmp_path, "x = 1\n")
    assert result["state"] == "checked"
    assert "_max_context_revision" not in result


def test_orchestrator_owns_the_single_file_synthesis_call(tmp_path):
    provider = _CountingProvider()
    orchestrator = Orchestrator(provider, analysis_mode="single")
    request = make_execution_request(tmp_path, "pkg/large.py")

    result = orchestrator.synthesize_divided_file(
        request,
        "division-plan:" + "0" * 64,
        '{"division":{"complete_source_coverage":true},"units":[]}',
    )

    assert provider.calls == 1
    assert result["file_path"] == "pkg/large.py"
    assert result["language"] == "python"
    assert result["description"] == "A documented module."


def test_split_final_assembly_uses_the_ordinary_bounded_public_schema(tmp_path):
    request = make_execution_request(tmp_path, "pkg/large.py")
    functions = tuple(
        {
            "name": f"function_{index}",
            "description": f"Function {index}.",
            "signature": f"function_{index}()",
            "_provenance": [
                {
                    "chunk_id": "chunk_" + f"{index:064x}",
                    "source_order": index,
                }
            ],
        }
        for index in range(MAX_SYMBOL_ITEMS_PER_KIND + 5)
    )
    ledger = FactLedger(
        functions=functions,
        classes=(
            {
                "name": "Service",
                "signature": "class Service",
                "_provenance": [{"chunk_id": "chunk_" + "f" * 64}],
            },
        ),
        exports=("Service",),
    )

    result = assemble_final_result(
        request,
        {"description": "A large module."},
        ledger,
        ("description", "functions", "classes", "exports"),
        "large-file-v2:" + "a" * 64,
    )

    assert len(result["functions"]) == MAX_SYMBOL_ITEMS_PER_KIND
    assert all(
        set(item) <= {"name", "description"}
        for item in result["functions"] + result["classes"]
    )
    assert result["structure"]["functions"] == result["functions"]
    assert result["structure"]["classes"] == result["classes"]
    assert result["exports"] == ["Service"]


def test_split_final_assembly_discards_unrequested_structural_fields(tmp_path):
    request = make_execution_request(tmp_path, "pkg/large.py")
    result = assemble_final_result(
        request,
        {
            "description": "A large module.",
            "functions": [{"name": "forged"}],
            "classes": [{"name": "Forged"}],
            "exports": ["forged"],
        },
        FactLedger(),
        ("description",),
        "large-file-v2:" + "a" * 64,
    )

    assert result["functions"] == []
    assert result["classes"] == []
    assert result["exports"] == []
    assert result["structure"]["functions"] == []
    assert result["structure"]["classes"] == []
    assert result["structure"]["exports"] == []
