"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _fake_provider(description: str = "Documented."):
    """Minimal fake LLM provider for pipeline tests."""
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": description,
                    "role_in_system": "test",
                    "key_concepts": ["entry point"],
                    "usage_example": "python main.py",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({
                    "dependencies_analysis": {
                        "internal": [], "external": ["requests"],
                        "dependency_refs": ["requests"],
                        "catalog_updates": [
                            {"name": "requests", "type": "external", "used_for": "HTTP"}
                        ],
                        "usage_notes": [
                            {"import": "requests", "used_for": "HTTP requests"}
                        ],
                        "warnings": [],
                    }
                })
            return _json.dumps({
                "description": description,
                "role_in_system": "test",
                "functions": [{"name": "main", "description": "runs app"}],
                "classes": [],
                "exports": ["main"],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()

def _make_view(
    entry_file: str | None = "main.py",
    files: list[dict] | None = None,
    graph_edges: list[dict] | None = None,
    dep_catalog: list[dict] | None = None,
) -> dict:
    """Build a minimal but non-trivial public view dict for testing."""
    from codedoc.core.project_view import SCHEMA_VERSION

    _files = files or [
        {
            "hash": "abc123",
            "path": "main.py",
            "language": "python",
            "description": "Entry point.",
            "role_in_system": "Starts the app.",
            "imports": ["utils"],
            "functions": [{"name": "main", "description": "Runs the app."}],
            "classes": [],
            "exports": ["main"],
            "key_concepts": ["entry point"],
            "usage_example": "python main.py",
            "_deps": {"external": ["requests"], "internal": ["utils"]},
            "links": {
                "internal_dependencies": ["utils.py"],
                "external_dependencies": ["requests"],
            },
        },
        {
            "hash": "def456",
            "path": "utils.py",
            "language": "python",
            "description": "Utility helpers.",
            "imports": ["requests"],
            "functions": [{"name": "fetch", "description": "HTTP fetch."}],
            "key_concepts": ["http", "helpers"],
            "_deps": {"external": ["requests"]},
            "links": {
                "imported_by": ["main.py"],
                "external_dependencies": ["requests"],
            },
        },
    ]

    _edges = graph_edges or [
        {"from": "main.py", "to": "utils.py", "type": "internal_import"},
    ]

    _catalog = dep_catalog or [
        {
            "name": "requests",
            "type": "external",
            "used_for": "HTTP requests",
            "files": ["main.py", "utils.py"],
            "file_count": 2,
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-06-01T12:00:00+00:00",
        "project": {
            "entry_file": entry_file,
            "file_count": len(_files),
            "languages": ["python"],
            "folders": ["."],
        },
        "run": {
            "files_checked": len(_files),
            "files_failed": 0,
            "files_skipped": 0,
            "files_reused": 0,
            "files_documented": len(_files),
        },
        "tree": {
            "main.py": {"type": "file", "path": "main.py"},
            "utils.py": {"type": "file", "path": "utils.py"},
        },
        "folders": [
            {
                "path": ".",
                "summary": "Root-level python files (2 file(s)).",
                "file_count": len(_files),
                "languages": ["python"],
                "files": [f["path"] for f in _files],
                "key_concepts": ["entry point"],
            }
        ],
        "dependency_catalog": _catalog,
        "dependency_graph": _edges,
        "files": _files,
    }
