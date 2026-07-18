"""Shared test support extracted from mapped source modules."""

import json
import pytest
from codedoc.llm.base import LLMProvider

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

@pytest.fixture
def mock_llm():
    return MockLLMProvider()
