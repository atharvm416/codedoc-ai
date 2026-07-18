"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
from codedoc.core.project_view import build_project_view

def _partition_sum(last_run: dict) -> int:
    return (
        last_run["files_reused_unchanged"]
        + last_run["files_reused_identical_content"]
        + last_run["files_documented_by_llm"]
        + last_run["files_failed"]
        + last_run["files_unattempted"]
        + last_run.get("files_skipped_insufficient_source", 0)
    )

def _records() -> list[dict]:
    return [
        {
            "hash": "h-main",
            "file_path": "main.py",
            "language": "python",
            "_analysis_revision": "file-doc-v3",
            "_analysis_mode": "single",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_prompt_profile_digest": "no-profile",
            "documentation": {
                "description": "Entry point.",
                "dependencies_analysis": {"external": ["requests"]},
            },
        },
        {
            "hash": "h-utils",
            "file_path": "utils.py",
            "language": "python",
            "documentation": {"description": "Utilities."},
        },
    ]

def _stats() -> dict:
    return {
        "checked": 1,
        "failed": 1,
        "skipped": 2,
        "reused": 1,
        "resumed": 1,
        "analysis_mode": "single",
        "entry_source": "explicit",
        "documentation_scope": "entry",
        "files_scanned": 7,
        "files_selected": 6,
        "unattempted_files": 1,
    }

def _view() -> dict:
    return build_project_view(_records(), _stats(), entry_file="main.py")
