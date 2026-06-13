"""0.9.3 — centralized read-only CodeDoc document reader.

Exercises every accepted document shape, the fail-closed cases, UTF-8
handling, defensive copies, and the intentional ownership tightening.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoc.core.document import (
    CodedocDocument,
    read_codedoc_document,
    records_by_path,
)
from codedoc.core.project_view import build_project_view, markdown_from_view
from codedoc.utils.errors import ConfigError

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completed_json(tmp_path, schema="1.4", entry="main.py", files=None, name="codedoc.json"):
    payload = {
        "_codedoc": {"entry_file": entry, "schema_version": schema},
        "schema_version": schema,
        "project": {"entry_file": entry, "file_count": 0, "languages": [], "folders": []},
        "files": files if files is not None else [],
    }
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _in_progress_json(tmp_path, schema="1.4", include_schema=True, name="codedoc.json"):
    meta = {"entry_file": "main.py", "status": "in_progress", "live_backup": True}
    if include_schema:
        meta["schema_version"] = schema
    payload = {"_crash_safety": "INCOMPLETE", "_codedoc": meta, "files": []}
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _embedded_md(tmp_path, name="codedoc.md"):
    view = build_project_view(
        [{
            "hash": "h", "file_path": "main.py", "language": "python",
            "documentation": {"file_path": "main.py", "language": "python",
                              "description": "d", "imports": ["utils"]},
        }],
        {"checked": 1},
        entry_file="main.py",
    )
    p = tmp_path / name
    p.write_text(markdown_from_view(view), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Accepted JSON shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schema", ["1.3", "1.4"])
def test_completed_json_accepted(tmp_path, schema):
    doc = read_codedoc_document(_completed_json(tmp_path, schema=schema))
    assert doc.format == "json"
    assert doc.schema_version == schema
    assert doc.in_progress is False


def test_in_progress_json_with_schema_accepted(tmp_path):
    doc = read_codedoc_document(_in_progress_json(tmp_path))
    assert doc.in_progress is True


def test_in_progress_json_missing_schema_accepted(tmp_path):
    doc = read_codedoc_document(_in_progress_json(tmp_path, include_schema=False))
    assert doc.in_progress is True
    assert doc.schema_version is None


def test_legacy_13_json_fixture(tmp_path):
    doc = read_codedoc_document(FIXTURES / "legacy_codedoc_13.json")
    assert doc.schema_version == "1.3"
    assert doc.entry_file == "main.py"
    assert set(records_by_path(doc)) == {"main.py", "utils.py"}


def test_null_entry_file_accepted(tmp_path):
    doc = read_codedoc_document(_completed_json(tmp_path, entry=None))
    assert doc.entry_file is None


def test_file_hashes_extracted_from_json(tmp_path):
    files = [{"path": "main.py", "hash": "abc"}]
    doc = read_codedoc_document(_completed_json(tmp_path, files=files))
    assert doc.file_hashes == {"main.py": "abc"}


# ---------------------------------------------------------------------------
# Accepted Markdown shapes
# ---------------------------------------------------------------------------

def test_current_embedded_markdown_preferred(tmp_path):
    doc = read_codedoc_document(_embedded_md(tmp_path))
    assert doc.format == "md"
    assert doc.entry_file == "main.py"
    assert records_by_path(doc)["main.py"]["imports"] == ["utils"]


def test_legacy_visible_markdown_fixture(tmp_path):
    doc = read_codedoc_document(FIXTURES / "legacy_codedoc_14.md")
    assert doc.schema_version == "1.4"
    assert doc.entry_file == "main.py"
    # Lightweight file_hashes are authoritative.
    assert doc.file_hashes["main.py"] == "hash-main-14"


def test_markdown_lightweight_hashes_are_authoritative(tmp_path):
    # Embedded view has its own (empty) hashes; lightweight comment wins.
    md_path = _embedded_md(tmp_path)
    content = md_path.read_text(encoding="utf-8")
    assert "hash-from-meta" not in content
    # Inject a lightweight file_hashes entry by rewriting the meta comment.
    import re
    new = re.sub(
        r'("file_hashes":\s*)\{[^}]*\}',
        r'\1{"main.py": "hash-from-meta"}',
        content,
        count=1,
    )
    md_path.write_text(new, encoding="utf-8")
    doc = read_codedoc_document(md_path)
    assert doc.file_hashes["main.py"] == "hash-from-meta"


# ---------------------------------------------------------------------------
# UTF-8 handling
# ---------------------------------------------------------------------------

def test_utf8_bom_is_accepted(tmp_path):
    p = _completed_json(tmp_path)
    raw = p.read_bytes()
    p.write_bytes(b"\xef\xbb\xbf" + raw)
    doc = read_codedoc_document(p)
    assert doc.format == "json"


def test_invalid_utf8_is_rejected(tmp_path):
    p = tmp_path / "codedoc.json"
    p.write_bytes(b"\xff\xfe\x00bad bytes")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


# ---------------------------------------------------------------------------
# Embedded block: corrupt-present vs absent
# ---------------------------------------------------------------------------

def test_corrupt_embedded_block_is_malformed(tmp_path):
    p = tmp_path / "codedoc.md"
    p.write_text(
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4"} -->\n'
        "<!-- codedoc-ai-view-base64\n!!!notbase64!!!\n-->\n# docs\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_absent_embedded_block_falls_back_to_visible(tmp_path):
    doc = read_codedoc_document(FIXTURES / "legacy_codedoc_14.md")
    # No embedded block present, but the visible parser still yields files.
    assert "main.py" in records_by_path(doc)


# ---------------------------------------------------------------------------
# Structural rejection
# ---------------------------------------------------------------------------

def test_duplicate_paths_rejected(tmp_path):
    files = [{"path": "main.py", "hash": "a"}, {"path": "main.py", "hash": "b"}]
    with pytest.raises(ConfigError):
        read_codedoc_document(_completed_json(tmp_path, files=files))


def test_non_list_files_rejected(tmp_path):
    payload = {"_codedoc": {"entry_file": None, "schema_version": "1.4"}, "files": {}}
    p = tmp_path / "codedoc.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_non_dict_file_record_rejected(tmp_path):
    with pytest.raises(ConfigError):
        read_codedoc_document(_completed_json(tmp_path, files=["not-a-dict"]))


# ---------------------------------------------------------------------------
# Fail-closed cases
# ---------------------------------------------------------------------------

def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_codedoc_document(tmp_path / "nope.json")


def test_foreign_json_rejected(tmp_path):
    p = tmp_path / "codedoc.json"
    p.write_text('{"not": "codedoc"}', encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_malformed_json_rejected(tmp_path):
    p = tmp_path / "codedoc.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_completed_json_missing_schema_rejected(tmp_path):
    payload = {"_codedoc": {"entry_file": "main.py"}, "files": []}
    p = tmp_path / "codedoc.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


@pytest.mark.parametrize("schema", ["1.5", "2.0", "3.1"])
def test_unknown_newer_schema_rejected(tmp_path, schema):
    with pytest.raises(ConfigError):
        read_codedoc_document(_completed_json(tmp_path, schema=schema))


@pytest.mark.parametrize("schema", ["1.3.0", "1.4.1"])
def test_schema_with_extra_components_rejected(tmp_path, schema):
    with pytest.raises(ConfigError):
        read_codedoc_document(_completed_json(tmp_path, schema=schema))


def test_malformed_schema_rejected(tmp_path):
    with pytest.raises(ConfigError):
        read_codedoc_document(_completed_json(tmp_path, schema="1.x"))


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "codedoc.txt"
    p.write_text("whatever", encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_marker_only_malformed_markdown_is_foreign(tmp_path):
    # Intentional tightening: a codedoc-ai marker with no valid metadata and no
    # embedded view must NOT be treated as owned.
    p = tmp_path / "codedoc.md"
    p.write_text("<!-- codedoc-ai: not-json -->\n# docs\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_malformed_metadata_is_rejected_even_with_valid_embedded_view(tmp_path):
    valid = _embedded_md(tmp_path).read_text(encoding="utf-8")
    _, embedded_and_visible = valid.split("\n", 1)
    p = tmp_path / "codedoc.md"
    p.write_text(
        "<!-- codedoc-ai: not-json -->\n" + embedded_and_visible,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_markdown_missing_schema_rejected(tmp_path):
    p = tmp_path / "codedoc.md"
    p.write_text(
        '<!-- codedoc-ai: {"entry_file": "main.py", "file_hashes": {}} -->\n'
        "# docs\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_markdown_file_hashes_must_be_string_map(tmp_path):
    p = tmp_path / "codedoc.md"
    p.write_text(
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"file_hashes": []} -->\n# docs\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_embedded_project_must_be_object(tmp_path):
    import base64

    embedded = {
        "schema_version": "1.4",
        "project": [],
        "files": [],
    }
    encoded = base64.b64encode(json.dumps(embedded).encode("utf-8")).decode("ascii")
    p = tmp_path / "codedoc.md"
    p.write_text(
        '<!-- codedoc-ai: {"entry_file": null, "schema_version": "1.4", '
        '"file_hashes": {}} -->\n'
        f"<!-- codedoc-ai-view-base64\n{encoded}\n-->\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


def test_conflicting_json_schema_versions_rejected(tmp_path):
    payload = {
        "_codedoc": {"entry_file": None, "schema_version": "1.4"},
        "schema_version": "1.3",
        "files": [],
    }
    p = tmp_path / "codedoc.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError):
        read_codedoc_document(p)


# ---------------------------------------------------------------------------
# legacy_role
# ---------------------------------------------------------------------------

def test_stale_build_role_allows_missing_schema(tmp_path):
    payload = {"_codedoc": {"entry_file": "main.py"},
               "files": [{"path": "main.py", "hash": "h"}]}
    p = tmp_path / ".codedoc_build.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    # Without the role, missing schema on a completed build fails closed.
    with pytest.raises(ConfigError):
        read_codedoc_document(p)
    # With the explicit stale-build role it is accepted (parsing only).
    doc = read_codedoc_document(p, legacy_role="stale_build")
    assert "main.py" in records_by_path(doc)


def test_unknown_legacy_role_rejected(tmp_path):
    with pytest.raises(ValueError):
        read_codedoc_document(_completed_json(tmp_path), legacy_role="bogus")


# ---------------------------------------------------------------------------
# Defensive copies + acyclic imports
# ---------------------------------------------------------------------------

def test_returns_defensive_copies(tmp_path):
    files = [{"path": "main.py", "hash": "h", "imports": ["a"]}]
    p = _completed_json(tmp_path, files=files)
    first = records_by_path(read_codedoc_document(p))
    first["main.py"]["imports"].append("mutated")
    second = records_by_path(read_codedoc_document(p))
    assert second["main.py"]["imports"] == ["a"]


def test_read_codedoc_meta_delegates_without_cycle(tmp_path):
    from codedoc.core.project_view import read_codedoc_meta

    p = _completed_json(tmp_path, entry="main.py")
    meta = read_codedoc_meta(p)
    assert meta["entry_file"] == "main.py"
    assert meta["schema_version"] == "1.4"


def test_read_codedoc_meta_preserves_legacy_metadata_fields(tmp_path):
    from codedoc.core.project_view import read_codedoc_meta

    payload = {
        "_codedoc": {
            "entry_file": "main.py",
            "schema_version": "1.4",
            "status": "in_progress",
            "live_backup": True,
            "generated_at": "2026-01-01T00:00:00+00:00",
        },
        "_crash_safety": "INCOMPLETE",
        "files": [],
    }
    p = tmp_path / "codedoc.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    meta = read_codedoc_meta(p)
    assert meta == payload["_codedoc"]


def test_document_dataclass_shape(tmp_path):
    doc = read_codedoc_document(_completed_json(tmp_path))
    assert isinstance(doc, CodedocDocument)
    assert isinstance(doc.metadata, dict)
    assert isinstance(doc.files, tuple)
    assert isinstance(doc.view, dict)
