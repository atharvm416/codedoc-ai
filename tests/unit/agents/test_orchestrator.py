"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811

import json
import pytest
from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.execution import _agent_errors
from tests.support.one_call_cases import _COMBINED
from tests.support.one_call_cases import _COMBINED_JSON as ONE_CALL_COMBINED_JSON
from tests.support.one_call_cases import _CountingProvider
from tests.support.one_call_cases import _descriptor
from tests.support.fixture_paths import PROJECT_FIXTURES

def test_deterministic_fields_cannot_be_replaced_by_model_output():
    raw = json.dumps({
        **_COMBINED,
        "file_path": "EVIL.py", "language": "evil", "imports": ["malware"],
        "extension": ".evil",
    })
    result = Orchestrator(_CountingProvider(raw), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
    )
    assert result["file_path"] == "pkg/mod.py"
    assert result["language"] == "python"
    assert result["extension"] == ".py"
    assert result["imports"] == ["os"]

def test_combined_failure_produces_one_agent_error_via_documentation():
    result = Orchestrator(_CountingProvider("not json"), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
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
    def test_all_fixtures(self):
        """Run orchestrator across all fixture codebases."""
        from codedoc.agents.orchestrator import Orchestrator
        import json as _json

        fixtures = [
            (PROJECT_FIXTURES / "python_app" / "main.py", "python", ".py"),
            (PROJECT_FIXTURES / "react_app" / "App.tsx", "tsx", ".tsx"),
            (PROJECT_FIXTURES / "java_app" / "Main.java", "java", ".java"),
            (PROJECT_FIXTURES / "flutter_app" / "main.dart", "dart", ".dart"),
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

        for path, language, ext in fixtures:
            descriptor = {
                "path": path,
                "rel_path": path.name,
                "language": language,
                "extension": ext,
            }
            content = path.read_text()
            result = orch.process(descriptor, content, [])
            assert result["state"] == "checked", f"Failed for {path.name}"
            assert result["file_path"] == path.name

class TestOrchestrator:
    def test_process_returns_merged_result(self, mock_llm):
        from codedoc.agents.orchestrator import Orchestrator
        from pathlib import Path
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".tsx", delete=False, mode="w") as f:
            f.write("import React from 'react';\nconst App = () => <div/>;\nexport default App;\n")
            tmp = f.name

        descriptor = {
            "path": Path(tmp),
            "rel_path": "App.tsx",
            "language": "tsx",
            "extension": ".tsx",
        }

        orch = Orchestrator(mock_llm, parallel=False)
        result = orch.process(descriptor, open(tmp).read(), ["react"])

        assert result["file_path"] == "App.tsx"
        assert result["state"] == "checked"
        assert "imports" in result
        assert "description" in result
        os.unlink(tmp)

    def test_parallel_mode(self, mock_llm):
        from codedoc.agents.orchestrator import Orchestrator
        from pathlib import Path
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write("import os\ndef main(): pass\n")
            tmp = f.name

        descriptor = {
            "path": Path(tmp),
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
        orch = Orchestrator(mock_llm, parallel=True)
        result = orch.process(descriptor, open(tmp).read(), ["os"])
        assert result["state"] == "checked"
        os.unlink(tmp)

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

def _process(content, *, max_chars=1000, head_ratio=0.70):
    orch = Orchestrator(
        _FakeProvider(), analysis_mode="single",
        max_content_chars=max_chars, truncation_head_ratio=head_ratio,
    )
    return orch.process(
        {"rel_path": "pkg/mod.py", "language": "python", "extension": ".py"},
        content, ["os"],
    )

def test_orchestrator_stamps_revision_for_oversized_file():
    result = _process("x" * 5000)
    assert result["state"] == "checked"
    assert result["_max_context_revision"] == "truncate-v1:max=1000:head=0.7000"

def test_orchestrator_omits_revision_for_small_file():
    result = _process("x = 1\n")
    assert result["state"] == "checked"
    assert "_max_context_revision" not in result
