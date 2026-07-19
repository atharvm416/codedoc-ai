"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from codedoc.core.markdown_view import (
    markdown_from_view,
    read_embedded_view,
)
from codedoc.core.project_view import build_project_view, json_from_view
from tests.support.fixture_paths import GOLDEN_DOCUMENT_FIXTURES
from tests.support.project_view_cases import _build_view
from tests.support.deterministic_records import _records as deterministic_records

def _read_golden(name: str) -> str:
    return (GOLDEN_DOCUMENT_FIXTURES / name).read_text(encoding="utf-8")

def _without_reachability(view: dict) -> dict:
    return {
        **view,
        "files": [
            {key: value for key, value in file.items() if key != "reachable_from_entry"}
            for file in view.get("files", [])
        ],
    }

def _without_visible_reachability(markdown: str) -> str:
    return markdown.replace("**Reachable from entry:** Yes  \n\n", "").replace(
        "**Reachable from entry:** No  \n\n", ""
    )

def test_view_assembly_byte_identical():
    view = _without_reachability(_build_view())
    assert json.dumps(view, indent=2, ensure_ascii=False) == _read_golden(
        "project_view.json"
    )

def test_json_serialization_byte_identical():
    view = _without_reachability(_build_view())
    assert json_from_view(view, "No errors.") == _read_golden("completed_output.json")

def test_markdown_serialization_byte_identical():
    view = _without_reachability(_build_view())
    rendered = markdown_from_view(view, "Sample error\nsecond line")
    assert _without_visible_reachability(rendered) == _read_golden(
        "completed_output.md"
    )

def test_empty_view_json_byte_identical():
    empty = build_project_view([], {"checked": 0}, entry_file=None, graph_edges=[])
    assert json_from_view(empty) == _read_golden("empty_output.json")

def test_empty_view_markdown_byte_identical():
    empty = build_project_view([], {"checked": 0}, entry_file=None, graph_edges=[])
    assert markdown_from_view(empty) == _read_golden("empty_output.md")

def test_completed_view_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    assert "generated_at" not in view

def test_completed_json_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    text = json_from_view(view)
    assert "generated_at" not in text

def test_completed_markdown_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    md = markdown_from_view(view)
    assert "generated_at" not in md

def test_embedded_view_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert "generated_at" not in embedded

def test_json_omits_generated_at_even_from_legacy_view():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    view["generated_at"] = "2026-01-01T00:00:00+00:00"  # simulate a legacy caller view
    text = json_from_view(view)
    payload = json.loads(text)
    assert "generated_at" not in payload
    assert "generated_at" not in payload.get("_codedoc", {})

def test_two_runs_produce_byte_identical_json():
    a = json_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    b = json_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    assert a == b

def test_two_runs_produce_byte_identical_markdown():
    a = markdown_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    b = markdown_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    assert a == b
