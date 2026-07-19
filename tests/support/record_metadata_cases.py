"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import pytest
from codedoc.core import record_meta
from codedoc.core.project_view import (
    build_project_view,
)

def _record(documentation: dict, top_level: dict | None = None) -> dict:
    rec = {
        "hash": "h",
        "file_path": "main.py",
        "language": "python",
        "documentation": documentation,
    }
    if top_level:
        rec.update(top_level)
    return rec

def _view_with_secret(private_key):
    record = _record({"language": "python", "description": "d", "_secret": "TOPSECRET"})
    return build_project_view([record], {"checked": 1})

@pytest.fixture
def private_key(monkeypatch):
    """Register a synthetic private key for the duration of a test."""
    monkeypatch.setattr(record_meta, "PRIVATE_RECORD_KEYS", frozenset({"_secret"}))
    return "_secret"
