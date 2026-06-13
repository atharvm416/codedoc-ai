"""0.9.3 — private record metadata registry and carry-through plumbing.

The production registry is empty; these tests monkeypatch a synthetic
private key (``_secret``) to exercise the carry behaviour through cleaning,
JSON, embedded Markdown, live backup, and resume reconstruction — and verify
that arbitrary underscore-prefixed model output is NOT preserved.
"""
from __future__ import annotations

import json

import pytest

from codedoc.core import record_meta
from codedoc.core.project_view import (
    build_project_view,
    clean_file_record,
    json_from_view,
    markdown_from_view,
    read_embedded_view,
)


@pytest.fixture
def private_key(monkeypatch):
    """Register a synthetic private key for the duration of a test."""
    monkeypatch.setattr(record_meta, "PRIVATE_RECORD_KEYS", frozenset({"_secret"}))
    return "_secret"


# ---------------------------------------------------------------------------
# carry_private_keys unit behaviour
# ---------------------------------------------------------------------------

def test_carry_copies_only_registered_keys(private_key):
    source = {"_secret": "keep", "_other": "drop", "plain": 1}
    target: dict = {}
    record_meta.carry_private_keys(source, target)
    assert target == {"_secret": "keep"}


def test_carry_does_not_mutate_source(private_key):
    source = {"_secret": "keep"}
    snapshot = dict(source)
    record_meta.carry_private_keys(source, {})
    assert source == snapshot


def test_carry_skips_absent_keys(private_key):
    target = {"existing": 1}
    record_meta.carry_private_keys({"plain": 2}, target)
    assert target == {"existing": 1}


@pytest.mark.parametrize("value", [None, "", False, 0, [], {}])
def test_carry_preserves_falsey_values(private_key, value):
    target: dict = {}
    record_meta.carry_private_keys({"_secret": value}, target)
    assert "_secret" in target
    assert target["_secret"] == value


def test_carry_is_idempotent(private_key):
    target: dict = {}
    record_meta.carry_private_keys({"_secret": "v"}, target)
    record_meta.carry_private_keys({"_secret": "v"}, target)
    assert target == {"_secret": "v"}


def test_empty_registry_carries_nothing():
    # Production default: registry is empty.
    target: dict = {}
    record_meta.carry_private_keys({"_secret": "v", "_anything": 1}, target)
    assert target == {}


# ---------------------------------------------------------------------------
# Precedence through _clean_file / clean_file_record
# ---------------------------------------------------------------------------

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


def test_nested_orchestrator_result_source(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d", "_secret": "from-nested"})
    )
    assert cleaned["_secret"] == "from-nested"


def test_top_level_persisted_source(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d"}, {"_secret": "from-top"})
    )
    assert cleaned["_secret"] == "from-top"


def test_top_level_wins_when_both_present(private_key):
    cleaned = clean_file_record(
        _record(
            {"language": "python", "description": "d", "_secret": "from-nested"},
            {"_secret": "from-top"},
        )
    )
    assert cleaned["_secret"] == "from-top"


def test_falsey_registered_value_survives_pruning(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d", "_secret": ""})
    )
    assert "_secret" in cleaned and cleaned["_secret"] == ""


def test_unknown_underscore_key_is_rejected(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d", "_other": "leak"})
    )
    assert "_other" not in cleaned


def test_clean_file_does_not_mutate_source(private_key):
    documentation = {"language": "python", "description": "d", "_secret": "v"}
    record = _record(documentation)
    snapshot = json.dumps(record, sort_keys=True)
    clean_file_record(record)
    assert json.dumps(record, sort_keys=True) == snapshot


# ---------------------------------------------------------------------------
# Persistence paths: JSON, embedded Markdown, visible Markdown exclusion
# ---------------------------------------------------------------------------

def _view_with_secret(private_key):
    record = _record({"language": "python", "description": "d", "_secret": "TOPSECRET"})
    return build_project_view([record], {"checked": 1})


def test_completed_json_preserves_private_key(private_key):
    view = _view_with_secret(private_key)
    payload = json.loads(json_from_view(view))
    assert payload["files"][0]["_secret"] == "TOPSECRET"


def test_embedded_markdown_preserves_private_key(private_key):
    view = _view_with_secret(private_key)
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert embedded is not None
    assert embedded["files"][0]["_secret"] == "TOPSECRET"


def test_visible_markdown_never_renders_private_key(private_key):
    view = _view_with_secret(private_key)
    md = markdown_from_view(view)
    # Strip the hidden base64 block; the value must not appear in visible prose.
    visible = md.split("-->", 2)[-1]
    assert "TOPSECRET" not in visible


# ---------------------------------------------------------------------------
# Resume reconstruction and live backup
# ---------------------------------------------------------------------------

def test_resume_reconstruction_preserves_private_key(private_key):
    from codedoc.pipeline import _public_record_to_doc

    public_record = {"path": "main.py", "language": "python", "_secret": "v"}
    doc = _public_record_to_doc(public_record)
    assert doc["_secret"] == "v"


def test_live_backup_preserves_private_key(tmp_path, private_key):
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {"main.py": {"path": tmp_path / "main.py"}})
    sw.set_queue_order(["main.py"])
    sw.record("main.py", {"language": "python", "description": "d", "_secret": "v"}, "h")

    data = json.loads(backup.read_text(encoding="utf-8"))
    assert data["files"][0]["_secret"] == "v"
