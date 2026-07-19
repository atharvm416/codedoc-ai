"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811


from tests.support.fixture_paths import PROJECT_FIXTURES
from tests.support.agent_fakes import AlwaysReturnJSON

def dependency_mock():
    return AlwaysReturnJSON({
        "dependencies_analysis": {
            "internal": ["./utils"],
            "external": ["os"],
            "usage_notes": [],
            "warnings": [],
        }
    })

class TestDependencyAgentIntegration:
    def test_with_java_fixture(self):
        from codedoc.agents.dependency_agent import DependencyAgent
        content = (PROJECT_FIXTURES / "java_app" / "Main.java").read_text()
        agent = DependencyAgent(dependency_mock())
        result = agent.run("Main.java", content, ["java.util.List"], "java")
        assert "dependencies_analysis" in result

    def test_with_flutter_fixture(self):
        from codedoc.agents.dependency_agent import DependencyAgent
        content = (PROJECT_FIXTURES / "flutter_app" / "main.dart").read_text()
        agent = DependencyAgent(dependency_mock())
        result = agent.run("main.dart", content, ["package:flutter/material.dart"], "dart")
        assert "dependencies_analysis" in result

class TestDependencyAgent:
    def test_returns_dependencies_analysis(self, mock_llm):
        from codedoc.agents.dependency_agent import DependencyAgent
        agent = DependencyAgent(mock_llm)
        result = agent.run("src/App.tsx", "import utils from './utils';", ["./utils"], "tsx")
        assert "dependencies_analysis" in result
        da = result["dependencies_analysis"]
        assert "internal" in da
        assert "external" in da
        # 0.10.1 (Workstream G4): an empty warnings list is omitted by the shared
        # strict cleaner; the populated usage_notes survive.
        assert "warnings" not in da
        assert "usage_notes" in da
