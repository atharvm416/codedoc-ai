"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import json
from pathlib import Path

class FakeProvider:
    provider_name = "fake"

    def complete_json(self, prompt: str, system: str = "") -> str:
        if "key_concepts" in prompt:
            return json.dumps(
                {
                    "description": "Test.",
                    "role_in_system": "test",
                    "key_concepts": [],
                    "usage_example": "",
                }
            )
        if "dependencies_analysis" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        return json.dumps(
            {
                "description": "Test.",
                "role_in_system": "test",
                "functions": [],
                "classes": [],
                "exports": [],
            }
        )

def make_graph(*paths: str, edges: tuple[tuple[str, str], ...] = ()):
    from codedoc.core.graph import DependencyGraph

    graph = DependencyGraph()
    for path in paths:
        graph.add_file(path)
    for source, dependency in edges:
        graph.add_dependency(source, dependency)
    return graph

def write_py(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
