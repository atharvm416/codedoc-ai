"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811


from tests.support.fixture_paths import PROJECT_FIXTURES
from tests.support.agent_fakes import AlwaysReturnJSON

def structure_mock():
    return AlwaysReturnJSON({
        "description": "Test file.",
        "role_in_system": "Core module.",
        "functions": [{"name": "main", "description": "Entry point."}],
        "classes": [],
        "exports": ["main"],
    })

class TestStructureAgentIntegration:
    def test_with_python_fixture(self):
        from codedoc.agents.structure_agent import StructureAgent
        content = (PROJECT_FIXTURES / "python_app" / "main.py").read_text()
        agent = StructureAgent(structure_mock())
        result = agent.run("main.py", content, ["os", ".utils"], "python")
        assert result["description"]
        assert isinstance(result["functions"], list)

    def test_with_react_fixture(self):
        from codedoc.agents.structure_agent import StructureAgent
        content = (PROJECT_FIXTURES / "react_app" / "App.tsx").read_text()
        agent = StructureAgent(structure_mock())
        result = agent.run("App.tsx", content, ["./router"], "tsx")
        assert "description" in result

class TestStructureAgent:
    def test_returns_expected_keys(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        result = agent.run("src/App.tsx", "const App = () => {};", [], "tsx")
        # 0.10.1 (Workstream G4): triple-mode responses now pass through the same
        # strict cleaners as single mode, so empty keys are omitted rather than
        # returned as empty lists.
        assert "description" in result
        assert "functions" in result
        assert "exports" in result
        assert "classes" not in result  # mock returns an empty classes list

    def test_functions_is_list(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        result = agent.run("src/App.tsx", "const fn = () => {};", [], "tsx")
        assert isinstance(result["functions"], list)
