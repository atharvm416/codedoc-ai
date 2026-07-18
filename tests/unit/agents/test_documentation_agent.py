"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811


from tests.support.fixture_paths import PROJECT_FIXTURES
from tests.support.agent_fakes import AlwaysReturnJSON

def doc_mock():
    return AlwaysReturnJSON({
        "description": "Main entry point of the application.",
        "role_in_system": "Bootstraps the app.",
        "key_concepts": ["entry point"],
        "usage_example": "python main.py",
    })

class TestDocumentationAgentIntegration:
    def test_run_with_context(self):
        from codedoc.agents.documentation_agent import DocumentationAgent
        content = (PROJECT_FIXTURES / "python_app" / "main.py").read_text()
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

class TestDocumentationAgent:
    def test_returns_expected_keys(self, mock_llm):
        from codedoc.agents.documentation_agent import DocumentationAgent
        agent = DocumentationAgent(mock_llm)
        result = agent.run_with_context(
            "src/App.tsx", "const App = () => {};", [], "tsx",
            structure={"description": "x"}, dependencies={}
        )
        assert "description" in result
        assert "key_concepts" in result
