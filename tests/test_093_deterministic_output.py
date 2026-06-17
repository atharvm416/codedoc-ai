"""0.9.3 — deterministic, timestamp-free completed output.

Completed JSON and Markdown must contain no run-varying timestamp and must be
byte-identical across runs with identical inputs.  Live backups keep
``created_at`` / ``updated_at`` diagnostics.  Old outputs containing the
removed ``generated_at`` field remain readable.
"""
from __future__ import annotations

import json
from pathlib import Path

from codedoc.core.project_view import (
    build_project_view,
    json_from_view,
    markdown_from_view,
    read_embedded_view,
)

FIXTURES = Path(__file__).parent / "fixtures"


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


# ---------------------------------------------------------------------------
# No run-varying timestamp anywhere in completed output
# ---------------------------------------------------------------------------

def test_completed_view_has_no_generated_at():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    assert "generated_at" not in view


def test_completed_json_has_no_generated_at():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    text = json_from_view(view)
    assert "generated_at" not in text


def test_completed_markdown_has_no_generated_at():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    md = markdown_from_view(view)
    assert "generated_at" not in md


def test_embedded_view_has_no_generated_at():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert "generated_at" not in embedded


def test_json_omits_generated_at_even_from_legacy_view():
    view = build_project_view(_records(), {"checked": 2}, entry_file="main.py")
    view["generated_at"] = "2026-01-01T00:00:00+00:00"  # simulate a legacy caller view
    text = json_from_view(view)
    payload = json.loads(text)
    assert "generated_at" not in payload
    assert "generated_at" not in payload.get("_codedoc", {})


# ---------------------------------------------------------------------------
# Byte-identical across identical runs
# ---------------------------------------------------------------------------

def test_two_runs_produce_byte_identical_json():
    a = json_from_view(build_project_view(_records(), {"checked": 2}, entry_file="main.py"))
    b = json_from_view(build_project_view(_records(), {"checked": 2}, entry_file="main.py"))
    assert a == b


def test_two_runs_produce_byte_identical_markdown():
    a = markdown_from_view(build_project_view(_records(), {"checked": 2}, entry_file="main.py"))
    b = markdown_from_view(build_project_view(_records(), {"checked": 2}, entry_file="main.py"))
    assert a == b


# ---------------------------------------------------------------------------
# Serializers do not mutate their inputs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Live backup keeps created_at / updated_at, not generated_at
# ---------------------------------------------------------------------------

def test_live_backup_uses_created_and_updated_at(tmp_path):
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.set_queue_order([])
    sw.initialize_empty()

    meta = json.loads(backup.read_text(encoding="utf-8"))["_codedoc"]
    assert "created_at" in meta
    assert "updated_at" in meta
    assert "generated_at" not in meta


def test_live_backup_load_preserves_legacy_creation_time(tmp_path):
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    legacy_created = "2026-01-01T00:00:00+00:00"
    backup.write_text(
        json.dumps({
            "_crash_safety": "INCOMPLETE",
            "_codedoc": {
                "entry_file": "main.py",
                "schema_version": "1.4",
                "generated_at": legacy_created,
                "status": "in_progress",
            },
            "files": [],
        }),
        encoding="utf-8",
    )
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.load()
    sw.initialize_empty()
    meta = json.loads(backup.read_text(encoding="utf-8"))["_codedoc"]
    assert meta["created_at"] == legacy_created
    assert "generated_at" not in meta


# ---------------------------------------------------------------------------
# Backward compatibility: old outputs with generated_at still read
# ---------------------------------------------------------------------------

def test_old_json_with_generated_at_still_reads():
    from codedoc.core.document import read_codedoc_document

    doc = read_codedoc_document(FIXTURES / "legacy_codedoc_13.json")
    # generated_at lived in _codedoc, which the reader strips from the view.
    assert "generated_at" not in doc.view
    assert doc.entry_file == "main.py"
