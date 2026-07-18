"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
from codedoc.core.project_view import build_project_view

def _build_view() -> dict:
    return build_project_view(
        _records(), _stats(), entry_file="src/main.py", graph_edges=_edges()
    )

def _edges() -> list[dict]:
    return [{"from": "src/main.py", "to": "src/helper.py", "type": "internal_import"}]

def _records() -> list[dict]:
    return [
        {
            "hash": "h1",
            "file_path": "src/main.py",
            "language": "python",
            "documentation": {
                "file_path": "src/main.py",
                "language": "python",
                "description": "Entry point — orchestrates startup. Café ☕ Unicode ✓",
                "role_in_system": "bootstrap",
                "imports": ["os", "requests", "src.helper"],
                "functions": [{"name": "main", "description": "Run the app"}],
                "classes": [{"name": "App", "description": "Main app -- with dashes"}],
                "exports": ["main"],
                "key_concepts": ["startup", "config"],
                "usage_example": "python -m src.main --flag\n# comment with --> tricky chars",
                "dependencies_analysis": {
                    "external": ["requests", "os"],
                    "usage_notes": [
                        {"import": "requests", "used_for": "HTTP calls to the API service"}
                    ],
                    "catalog_updates": [
                        {"name": "requests", "type": "external", "used_for": "HTTP client library"}
                    ],
                    "dependency_refs": ["requests"],
                },
            },
        },
        {
            "hash": "h2",
            "file_path": "src/helper.py",
            "language": "python",
            "documentation": {
                "file_path": "src/helper.py",
                "language": "python",
                "description": "Helper utilities.",
                "role_in_system": "utility",
                "imports": ["json", "sys"],
                "functions": [{"name": "load", "description": ""}],
                "classes": [],
                "exports": [],
                "key_concepts": ["io"],
                "usage_example": "",
                "dependencies_analysis": {"external": ["json", "sys"]},
            },
        },
        {
            "hash": "h3",
            "file_path": "README.md",
            "language": "markdown",
            "documentation": {
                "file_path": "README.md",
                "language": "markdown",
                "description": "Docs.",
                "role_in_system": "documentation",
                "dependencies_analysis": {},
            },
        },
    ]

def _stats() -> dict:
    return {"checked": 2, "failed": 0, "skipped": 1, "reused": 0}
