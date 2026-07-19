"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _records():
    return [
        {
            "hash": "h-main", "file_path": "main.py", "language": "python",
            "documentation": {
                "file_path": "main.py", "language": "python", "description": "Entry.",
                "dependencies_analysis": {"external": ["os", "requests"]},
            },
        },
        {
            "hash": "h-utils", "file_path": "utils.py", "language": "python",
            "documentation": {"file_path": "utils.py", "language": "python",
                              "description": "Helpers."},
        },
    ]
