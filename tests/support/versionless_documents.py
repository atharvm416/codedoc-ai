"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
from codedoc.core.project_view import build_project_view
from codedoc.core.prompt_profiles import validate_profile

def _assert_versionless(value: object) -> None:
    if isinstance(value, dict):
        assert "schema_version" not in value
        for item in value.values():
            _assert_versionless(item)
    elif isinstance(value, list):
        for item in value:
            _assert_versionless(item)

def _validate(raw: dict):
    return validate_profile(
        raw,
        active_mode="single",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )

def _view() -> dict:
    record = {
        "hash": "h1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {
            "description": "Entry point.",
            "language": "python",
        },
    }
    return build_project_view(
        [record], {"checked": 1}, entry_file="main.py", graph_edges=[]
    )
