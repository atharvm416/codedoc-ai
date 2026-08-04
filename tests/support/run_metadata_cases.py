"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

from codedoc.core.record_meta import FRESH_SPLIT_REUSE_CONTRACT
from codedoc.core.project_view import build_project_view

PREDECESSOR_LARGE_FILE_IDENTITY = "large-file-v2:test"

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
    """Mixed ordinary records including an explicit 0.14.1 predecessor split."""
    return [
        {
            "hash": "h-main",
            "file_path": "main.py",
            "language": "python",
            "_analysis_revision": "file-doc-v3",
            "_analysis_mode": "single",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_prompt_profile_digest": "no-profile",
            "_large_file_identity": PREDECESSOR_LARGE_FILE_IDENTITY,
            "_split_reuse_contract": FRESH_SPLIT_REUSE_CONTRACT,
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


def _split_record() -> dict:
    """An explicit 0.14.1 predecessor split record: only the ordinary supported
    file-level shape plus private identity — no division/documentation_units
    or any other internal split content (D9/D14)."""
    return {
        "hash": "h-large",
        "file_path": "large.py",
        "language": "python",
        "_large_file_identity": PREDECESSOR_LARGE_FILE_IDENTITY,
        "_split_reuse_contract": FRESH_SPLIT_REUSE_CONTRACT,
        "documentation": {
            "description": "Large module.",
            "functions": [{"name": "alpha"}],
        },
    }


def _split_stats() -> dict:
    return {
        "checked": 1,
        "files_scanned": 1,
        "files_selected": 1,
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "split_ordinary_files": 0,
        "split_syntax_files": 1,
        "split_lexical_files": 0,
        "split_blocked_files": 0,
        "split_blocked_by_reason": {},
        "split_divided_files": 1,
        "split_units": 1,
        "split_chunks": 2,
        "split_continuation_groups": 0,
        "split_unit_consolidation_levels": 0,
        "split_unit_consolidation_calls_planned": 0,
        "split_general_reduction_levels": 0,
        "split_general_reduction_calls_planned": 0,
        "split_final_synthesis_calls_planned": 1,
        "split_restored_complete_chunks": 0,
        "split_restored_unit_consolidation_calls": 0,
        "split_restored_general_reduction_calls": 0,
        "split_restored_final_synthesis_calls": 0,
        "split_completed_files_reused": 0,
        "split_partial_files_resumed": 0,
        "split_unpaid_nodes": 3,
        "split_reexecuted_nodes": 1,
        "split_quarantined_nodes": 2,
        "split_recovery_conflict_files": 1,
        "file_documentation_calls_planned": 0,
        "unit_documentation_calls_planned": 2,
        "file_reduction_calls_planned": 0,
        "synthesis_calls_planned": 1,
    }
