"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import json

_COMBINED = {
    "description": "A documented module.",
    "role_in_system": "entry point",
    "functions": [{"name": "main", "description": "runs"}],
    "classes": [{"name": "C", "description": "a class"}],
    "exports": ["main"],
    "dependencies_analysis": {"external": ["requests"], "dependency_refs": ["requests"]},
    "key_concepts": ["startup"],
    "usage_example": "import mod",
}

_COMBINED_JSON = json.dumps(_COMBINED)

class _CountingProvider:
    provider_name = "fake"

    def __init__(self, raw=_COMBINED_JSON):
        self._raw = raw
        self.calls = 0

    def complete_json(self, prompt, system=""):
        self.calls += 1
        return self._raw

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

def _descriptor():
    return {"rel_path": "pkg/mod.py", "language": "python", "extension": ".py"}
