"""Agent integration tests — offline using MockLLMProvider."""

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class AlwaysReturnJSON:
    """Minimal mock that always returns a given JSON string."""
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload)

    def complete(self, prompt, system="", temperature=0.1):
        return self._payload

    def complete_json(self, prompt, system=""):
        return self._payload

    @property
    def provider_name(self):
        return "AlwaysReturn"


def structure_mock():
    return AlwaysReturnJSON({
        "description": "Test file.",
        "role_in_system": "Core module.",
        "functions": [{"name": "main", "description": "Entry point."}],
        "classes": [],
        "exports": ["main"],
    })


def dependency_mock():
    return AlwaysReturnJSON({
        "dependencies_analysis": {
            "internal": ["./utils"],
            "external": ["os"],
            "usage_notes": [],
            "warnings": [],
        }
    })


def doc_mock():
    return AlwaysReturnJSON({
        "description": "Main entry point of the application.",
        "role_in_system": "Bootstraps the app.",
        "key_concepts": ["entry point"],
        "usage_example": "python main.py",
    })


class TestStructureAgentIntegration:
    def test_with_python_fixture(self):
        from codedoc.agents.structure_agent import StructureAgent
        content = (FIXTURES / "python_app" / "main.py").read_text()
        agent = StructureAgent(structure_mock())
        result = agent.run("main.py", content, ["os", ".utils"], "python")
        assert result["description"]
        assert isinstance(result["functions"], list)

    def test_with_react_fixture(self):
        from codedoc.agents.structure_agent import StructureAgent
        content = (FIXTURES / "react_app" / "App.tsx").read_text()
        agent = StructureAgent(structure_mock())
        result = agent.run("App.tsx", content, ["./router"], "tsx")
        assert "description" in result


class TestDependencyAgentIntegration:
    def test_with_java_fixture(self):
        from codedoc.agents.dependency_agent import DependencyAgent
        content = (FIXTURES / "java_app" / "Main.java").read_text()
        agent = DependencyAgent(dependency_mock())
        result = agent.run("Main.java", content, ["java.util.List"], "java")
        assert "dependencies_analysis" in result

    def test_with_flutter_fixture(self):
        from codedoc.agents.dependency_agent import DependencyAgent
        content = (FIXTURES / "flutter_app" / "main.dart").read_text()
        agent = DependencyAgent(dependency_mock())
        result = agent.run("main.dart", content, ["package:flutter/material.dart"], "dart")
        assert "dependencies_analysis" in result


class TestDocumentationAgentIntegration:
    def test_run_with_context(self):
        from codedoc.agents.documentation_agent import DocumentationAgent
        content = (FIXTURES / "python_app" / "main.py").read_text()
        agent = DocumentationAgent(doc_mock())
        result = agent.run_with_context(
            "main.py", content, ["os"], "python",
            structure={"description": "x", "functions": [], "classes": []},
            dependencies={"dependencies_analysis": {"internal": [], "external": ["os"]}},
        )
        assert "description" in result
        assert "key_concepts" in result

    def test_safe_context_runner_is_defined_on_agent_class(self):
        from codedoc.agents.documentation_agent import DocumentationAgent

        assert "_safe_run_with_context" in DocumentationAgent.__dict__


class TestOrchestratorIntegration:
    def test_all_fixtures(self):
        """Run orchestrator across all fixture codebases."""
        from codedoc.agents.orchestrator import Orchestrator
        import json as _json

        fixtures = [
            (FIXTURES / "python_app" / "main.py", "python", ".py"),
            (FIXTURES / "react_app" / "App.tsx", "tsx", ".tsx"),
            (FIXTURES / "java_app" / "Main.java", "java", ".java"),
            (FIXTURES / "flutter_app" / "main.dart", "dart", ".dart"),
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
