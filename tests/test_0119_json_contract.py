"""0.11.9 completed JSON contract cleanup coverage."""

from __future__ import annotations

import json

import pytest

from codedoc.core.document import read_codedoc_document
from codedoc.core.markdown_view import markdown_from_view, markdown_to_view
from codedoc.core.output import _is_codedoc_owned
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.utils.errors import ConfigError


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


def test_completed_json_omits_removed_top_level_blocks(tmp_path):
    path = tmp_path / "codedoc.json"
    payload = json.loads(json_from_view(_view()))
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert "_codedoc" not in payload
    assert "project" not in payload
    assert "run" not in payload
    assert payload["last_run"]["entry_file"] == "main.py"

    doc = read_codedoc_document(path)
    assert doc.entry_file == "main.py"
    assert doc.metadata == {"entry_file": "main.py"}
    assert len(doc.files) == 2


def test_markdown_embeds_new_shape_without_legacy_wrappers(tmp_path):
    md = markdown_from_view(_view())
    path = tmp_path / "codedoc.md"
    path.write_text(md, encoding="utf-8")

    embedded = markdown_to_view(md)
    assert "last_run" in embedded
    assert "project" not in embedded
    assert "run" not in embedded

    doc = read_codedoc_document(path)
    assert doc.entry_file == "main.py"
    assert doc.view["last_run"]["entry_file"] == "main.py"


def test_legacy_completed_json_remains_owned_and_readable(tmp_path):
    data = json.loads(json_from_view(_view()))
    data["_codedoc"] = {"entry_file": "main.py"}
    data["project"] = {
        "entry_file": "main.py",
        "file_count": 2,
        "languages": ["python"],
        "folders": ["."],
    }
    data["run"] = {
        "files_checked": 2,
        "files_failed": 0,
        "files_skipped": 0,
        "files_reused": 0,
        "files_documented": 2,
    }
    data["last_run"].pop("entry_file")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert _is_codedoc_owned(path)
    doc = read_codedoc_document(path)
    assert doc.entry_file == "main.py"
    assert doc.view["run"]["files_documented"] == 2


def test_foreign_json_without_codedoc_shape_still_fails_closed(tmp_path):
    path = tmp_path / "codedoc.json"
    path.write_text(
        json.dumps({"files": [{"path": "main.py"}], "note": "not codedoc"}),
        encoding="utf-8",
    )

    assert not _is_codedoc_owned(path)
    with pytest.raises(ConfigError):
        read_codedoc_document(path)


def test_foreign_json_with_legacy_schema_but_no_codedoc_shape_fails_closed(tmp_path):
    path = tmp_path / "foreign.json"
    path.write_text(
        json.dumps({
            "schema_version": "1.4",
            "files": [{"path": "main.py"}],
            "note": "not codedoc",
        }),
        encoding="utf-8",
    )

    assert not _is_codedoc_owned(path)
    with pytest.raises(ConfigError):
        read_codedoc_document(path)
