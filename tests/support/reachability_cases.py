"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _record(path: str) -> dict:
    return {
        "hash": path,
        "file_path": path,
        "language": "python",
        "documentation": {
            "file_path": path,
            "language": "python",
            "description": path,
        },
    }
