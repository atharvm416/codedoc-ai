"""
LLM mock tests — run entirely offline, no API key needed.

A MockLLMProvider returns deterministic JSON so every agent
and the orchestrator can be tested without a real model.
"""

import json
import pytest
from codedoc.llm.base import LLMProvider


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Returns canned JSON responses for any prompt."""

    STRUCTURE_RESPONSE = {
        "description": "Mock description of the file.",
        "role_in_system": "Mock role.",
        "functions": [{"name": "mockFn", "description": "Does something."}],
        "classes": [],
        "exports": ["mockFn"],
    }

    DEPENDENCY_RESPONSE = {
        "dependencies_analysis": {
            "internal": ["./utils"],
            "external": ["react"],
            "usage_notes": [{"import": "./utils", "used_for": "utility helpers"}],
            "warnings": [],
        }
    }

    DOC_RESPONSE = {
        "description": "Documented: this file handles the main logic.",
        "role_in_system": "Central controller.",
        "key_concepts": ["hooks", "state management"],
        "usage_example": "import App from './App'",
    }

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        # Return different responses based on which agent is calling
        if "key_concepts" in prompt or "usage_example" in prompt:
            return json.dumps(self.DOC_RESPONSE)
        if "dependencies_analysis" in prompt or "dependency" in prompt.lower():
            return json.dumps(self.DEPENDENCY_RESPONSE)
        return json.dumps(self.STRUCTURE_RESPONSE)

    def complete_json(self, prompt: str, system: str = "") -> str:
        return self.complete(prompt, system)

    @property
    def provider_name(self) -> str:
        return "Mock"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    return MockLLMProvider()


class TestStructureAgent:
    def test_returns_expected_keys(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        result = agent.run("src/App.tsx", "const App = () => {};", [], "tsx")
        assert "description" in result
        assert "functions" in result
        assert "classes" in result
        assert "exports" in result

    def test_functions_is_list(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        result = agent.run("src/App.tsx", "const fn = () => {};", [], "tsx")
        assert isinstance(result["functions"], list)


class TestDependencyAgent:
    def test_returns_dependencies_analysis(self, mock_llm):
        from codedoc.agents.dependency_agent import DependencyAgent
        agent = DependencyAgent(mock_llm)
        result = agent.run("src/App.tsx", "import utils from './utils';", ["./utils"], "tsx")
        assert "dependencies_analysis" in result
        da = result["dependencies_analysis"]
        assert "internal" in da
        assert "external" in da
        assert "warnings" in da


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


class TestOrchestrator:
    def test_process_returns_merged_result(self, mock_llm):
        from codedoc.agents.orchestrator import Orchestrator
        from pathlib import Path
        import tempfile, os

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
        import tempfile, os

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


class TestBaseAgent:
    def test_truncates_long_content(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        long_content = "x = 1\n" * 5000
        truncated = agent._truncate(long_content)
        assert len(truncated) <= 12_100  # slight buffer for the suffix

    def test_parse_json_strips_fences(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        raw = '```json\n{"key": "value"}\n```'
        result = agent._parse_json(raw, "test.py")
        assert result == {"key": "value"}
