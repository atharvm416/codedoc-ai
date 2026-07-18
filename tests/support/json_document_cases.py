"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
from codedoc.core.project_view import build_project_view

def _record(path: str = "main.py") -> dict:
    return {
        "hash": f"h-{path}",
        "file_path": path,
        "language": "python",
        "documentation": {
            "description": f"Documentation for {path}.",
            "dependencies_analysis": {},
        },
    }

def _view() -> dict:
    return build_project_view(
        [_record("main.py"), _record("helper.py")],
        {
            "checked": 2,
            "failed": 0,
            "skipped": 0,
            "reused": 0,
            "files_scanned": 2,
            "files_selected": 2,
            "entry_source": "explicit",
            "documentation_scope": "entry",
            "analysis_mode": "single",
        },
        entry_file="main.py",
    )
