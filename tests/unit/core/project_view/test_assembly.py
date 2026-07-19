"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from codedoc.core.project_view import (
    build_project_view,
    json_from_view,
    markdown_from_view,
)
from tests.support.deterministic_records import _records
from tests.support.reachability_cases import _record

def test_build_project_view_does_not_mutate_records():
    records = _records()
    snapshot = json.dumps(records, sort_keys=True)
    build_project_view(records, {"checked": 2}, entry_file="main.py")
    assert json.dumps(records, sort_keys=True) == snapshot

def test_json_from_view_does_not_mutate_view():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    view["generated_at"] = "legacy"
    snapshot = json.dumps(view, sort_keys=True)
    json_from_view(view)
    assert json.dumps(view, sort_keys=True) == snapshot

def test_markdown_from_view_does_not_mutate_view():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    snapshot = json.dumps(view, sort_keys=True)
    markdown_from_view(view)
    assert json.dumps(view, sort_keys=True) == snapshot

def test_direct_view_call_defaults_all_records_to_reachable():
    view = build_project_view([_record("main.py")], {"checked": 1})
    assert view["files"][0]["reachable_from_entry"] is True
