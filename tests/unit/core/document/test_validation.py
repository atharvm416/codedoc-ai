"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from codedoc.core.document import (
    CodedocDocument,
    read_codedoc_document,
    records_by_path,
)
from codedoc.core.project_view import build_project_view, markdown_from_view
from codedoc.utils.errors import ConfigError
from tests.support.fixture_paths import LEGACY_DOCUMENT_FIXTURES
import base64
from codedoc.core.project_view import (
    json_from_view,
)
from tests.support.versionless_documents import _view as versionless_view
from codedoc.core.resume import (
    _load_existing_file_docs,
)
import copy
from codedoc.core.document import (
    _LAST_RUN_INTEGER_FIELDS,
    _LAST_RUN_OPTIONAL_INTEGER_FIELDS,
)
from codedoc.core.output import _is_codedoc_owned
from tests.support.run_metadata_cases import _view as run_metadata_view
from tests.support.run_metadata_cases import _partition_sum
from tests.support.json_document_cases import _view as json_contract_view

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


def test_partial_recovery_map_rejects_embedded_path_mismatch(tmp_path):
    payload = {
        "_crash_safety": "INCOMPLETE",
        "_codedoc": {
            "entry_file": "main.py",
            "status": "in_progress",
            "live_backup": True,
            "schema_version": "1.4",
            "partial_files": {
                "main.py": {
                    "schema_version": 1,
                    "owner": "codedoc-ai",
                    "rel_path": "other.py",
                    "content_hash": "a" * 64,
                    "execution_identity_digest": "division-execution:" + "b" * 64,
                    "division_plan_digest": "division-plan:" + "c" * 64,
                    "stage": "documenting",
                    "completed_chunks": [],
                    "synthesis_json": None,
                }
            },
        },
        "files": [],
    }
    path = tmp_path / "codedoc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    document = read_codedoc_document(path)

    assert document.partial_files == ()

def test_legacy_13_json_fixture(tmp_path):
    doc = read_codedoc_document(LEGACY_DOCUMENT_FIXTURES / "codedoc_13.json")
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

def test_current_embedded_markdown_preferred(tmp_path):
    doc = read_codedoc_document(_embedded_md(tmp_path))
    assert doc.format == "md"
    assert doc.entry_file == "main.py"
    assert records_by_path(doc)["main.py"]["imports"] == ["utils"]

def test_legacy_visible_markdown_fixture(tmp_path):
    doc = read_codedoc_document(LEGACY_DOCUMENT_FIXTURES / "codedoc_14.md")
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
    doc = read_codedoc_document(LEGACY_DOCUMENT_FIXTURES / "codedoc_14.md")
    # No embedded block present, but the visible parser still yields files.
    assert "main.py" in records_by_path(doc)

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


def test_conflicting_markdown_embedded_schema_is_rejected(tmp_path):
    embedded = {
        "schema_version": "1.3",
        "last_run": {},
        "files": [],
    }
    encoded = base64.b64encode(json.dumps(embedded).encode("utf-8")).decode("ascii")
    path = tmp_path / "codedoc.md"
    path.write_text(
        '<!-- codedoc-ai: {"entry_file": null, "schema_version": "1.4", '
        '"file_hashes": {}} -->\n'
        f"<!-- codedoc-ai-view-base64\n{encoded}\n-->\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="conflicting schema versions"):
        read_codedoc_document(path)


def test_unsupported_markdown_embedded_schema_is_rejected(tmp_path):
    embedded = {
        "schema_version": "2.0",
        "last_run": {},
        "files": [],
    }
    encoded = base64.b64encode(json.dumps(embedded).encode("utf-8")).decode("ascii")
    path = tmp_path / "codedoc.md"
    path.write_text(
        '<!-- codedoc-ai: {"entry_file": null, "file_hashes": {}} -->\n'
        f"<!-- codedoc-ai-view-base64\n{encoded}\n-->\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unsupported CodeDoc schema version"):
        read_codedoc_document(path)


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

def test_A9_read_codedoc_meta_md_null_entry_does_not_raise(tmp_path):
    """A9: read_codedoc_meta() on a Markdown file with entry_file=null must not raise."""
    from codedoc.core.project_view import read_codedoc_meta

    md_path = tmp_path / "codedoc.md"
    meta_payload = {
        "entry_file": None,
        "schema_version": "1.4",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "file_hashes": {},
    }
    md_path.write_text(
        f"<!-- codedoc-ai: {json.dumps(meta_payload)} -->\n# codedoc project documentation\n",
        encoding="utf-8",
    )

    meta = read_codedoc_meta(md_path)
    assert isinstance(meta, dict), "Must return a dict"
    assert meta.get("entry_file") is None
    assert meta.get("schema_version") == "1.4"

def test_A9_read_codedoc_meta_json_null_entry_does_not_raise(tmp_path):
    """A9b: read_codedoc_meta() on a JSON file with entry_file=null must not raise."""
    from codedoc.core.project_view import read_codedoc_meta

    json_path = tmp_path / "codedoc.json"
    payload = {
        "_codedoc": {
            "entry_file": None,
            "schema_version": "1.4",
            "generated_at": "2026-01-01T00:00:00+00:00",
        },
        "files": [],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    meta = read_codedoc_meta(json_path)
    assert isinstance(meta, dict)
    assert meta.get("entry_file") is None

def test_A9_file_with_null_entry_not_treated_as_foreign(tmp_path):
    """A9c: _check_file_ownership must accept a CodeDoc MD file with entry_file=null."""
    from codedoc.core.output import _check_file_ownership

    md_path = tmp_path / "codedoc.md"
    meta_payload = {
        "entry_file": None,
        "schema_version": "1.4",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "file_hashes": {},
    }
    md_path.write_text(
        f"<!-- codedoc-ai: {json.dumps(meta_payload)} -->\n# codedoc\n",
        encoding="utf-8",
    )

    # Must not raise ConfigError
    _check_file_ownership(md_path)

def test_A16_json_null_entry_recognised_not_foreign(tmp_path):
    """A16: A JSON file with _codedoc.entry_file=null is valid CodeDoc output."""
    from codedoc.core.project_view import read_codedoc_meta

    json_path = tmp_path / "codedoc.json"
    json_path.write_text(json.dumps({
        "_codedoc": {
            "entry_file": None,
            "schema_version": "1.4",
            "generated_at": "2026-01-01T00:00:00+00:00",
        },
        "schema_version": "1.4",
        "files": [],
    }), encoding="utf-8")

    meta = read_codedoc_meta(json_path)
    assert isinstance(meta, dict)
    assert meta["entry_file"] is None
    assert meta["schema_version"] == "1.4"

def test_A17_md_null_entry_recognised_not_foreign(tmp_path):
    """A17: A Markdown file with entry_file=null in the comment is valid CodeDoc output."""
    from codedoc.core.project_view import read_codedoc_meta

    md_path = tmp_path / "codedoc.md"
    md_path.write_text(
        '<!-- codedoc-ai: {"entry_file": null, "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n",
        encoding="utf-8",
    )

    meta = read_codedoc_meta(md_path)
    assert isinstance(meta, dict)
    assert meta["entry_file"] is None

def test_old_json_with_generated_at_still_reads():
    from codedoc.core.document import read_codedoc_document

    doc = read_codedoc_document(LEGACY_DOCUMENT_FIXTURES / "codedoc_13.json")
    # generated_at lived in _codedoc, which the reader strips from the view.
    assert "generated_at" not in doc.view
    assert doc.entry_file == "main.py"

def test_supported_versioned_legacy_documents_remain_readable():
    assert read_codedoc_document(LEGACY_DOCUMENT_FIXTURES / "codedoc_13.json").schema_version == "1.3"
    assert read_codedoc_document(LEGACY_DOCUMENT_FIXTURES / "codedoc_14.md").schema_version == "1.4"

def test_versionless_foreign_or_contradictory_json_is_rejected(tmp_path):
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"_codedoc": {}, "files": []}), encoding="utf-8")
    with pytest.raises(ConfigError, match="canonical"):
        read_codedoc_document(foreign)

    data = json.loads(json_from_view(versionless_view()))
    data["_codedoc"] = {"entry_file": "main.py"}
    data["last_run"]["entry_file"] = "other.py"
    foreign.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="contradictory"):
        read_codedoc_document(foreign)

def test_loader_accepts_versionless_and_legacy_fallback_documents(tmp_path):
    legacy_json = LEGACY_DOCUMENT_FIXTURES / "codedoc_13.json"
    legacy_md = LEGACY_DOCUMENT_FIXTURES / "codedoc_14.md"
    assert _load_existing_file_docs(tmp_path / "missing.json", legacy_md, "json")
    assert _load_existing_file_docs(legacy_json, tmp_path / "missing.md", "md")

_REQUIRED_LAST_RUN_INTEGER_FIELDS = {
    "files_scanned",
    "files_selected",
    "files_documented_by_llm",
    "files_failed",
    "files_unattempted",
    "files_reused_unchanged",
    "files_reused_identical_content",
    "files_resumed_from_recovery",
}

def _legacy_data() -> dict:
    data = json.loads(json_from_view(run_metadata_view()))
    data["_codedoc"] = {"entry_file": "main.py"}
    data["project"] = {
        "entry_file": "main.py",
        "file_count": 2,
        "languages": ["python"],
        "folders": ["."],
    }
    data["run"] = {
        "files_checked": 1,
        "files_failed": 1,
        "files_skipped": 2,
        "files_reused": 1,
        "files_documented": 2,
    }
    data["last_run"].pop("entry_file", None)
    return data

def test_json_reader_accepts_valid_last_run(tmp_path):
    path = tmp_path / "codedoc.json"
    path.write_text(json_from_view(run_metadata_view()), encoding="utf-8")

    doc = read_codedoc_document(path)

    assert doc.view["last_run"]["files_selected"] == 6
    assert doc.entry_file == "main.py"
    assert "run" not in doc.view
    assert "project" not in doc.view
    assert "_codedoc" not in doc.view

def test_skip_counter_is_additive_optional_on_read(tmp_path):
    assert _LAST_RUN_INTEGER_FIELDS == _REQUIRED_LAST_RUN_INTEGER_FIELDS
    assert "files_skipped_insufficient_source" not in _LAST_RUN_INTEGER_FIELDS
    assert _LAST_RUN_OPTIONAL_INTEGER_FIELDS == {
        "files_skipped_insufficient_source"
    }

    data = json.loads(json_from_view(run_metadata_view()))
    data["last_run"].pop("files_skipped_insufficient_source")
    path = tmp_path / "without-additive-counter.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    document = read_codedoc_document(path)
    assert "files_skipped_insufficient_source" not in document.view["last_run"]
    assert _partition_sum(document.view["last_run"]) == document.view["last_run"][
        "files_selected"
    ]

@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_json_reader_rejects_malformed_optional_skip_counter(tmp_path, value):
    data = json.loads(json_from_view(run_metadata_view()))
    data["last_run"]["files_skipped_insufficient_source"] = value
    path = tmp_path / "malformed-optional-counter.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError):
        read_codedoc_document(path)

@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("negative", lambda lr: lr.update({"files_failed": -1})),
        ("bool", lambda lr: lr.update({"files_failed": True})),
        ("missing", lambda lr: lr.pop("files_selected")),
        ("entry_source", lambda lr: lr.update({"entry_source": "mystery"})),
        ("scope", lambda lr: lr.update({"documentation_scope": "wide"})),
        ("mode", lambda lr: lr.update({"analysis_mode": "quad"})),
        ("partition", lambda lr: lr.update({"files_selected": 99})),
    ],
)
def test_json_reader_rejects_malformed_last_run(tmp_path, label, mutate):
    data = json.loads(json_from_view(run_metadata_view()))
    mutate(data["last_run"])
    path = tmp_path / f"{label}.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError):
        read_codedoc_document(path)

def test_unknown_envelope_keys_are_tolerated(tmp_path):
    data = _legacy_data()
    data["_codedoc"]["future_key"] = {"ok": True}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert read_codedoc_document(path).metadata["future_key"] == {"ok": True}

def test_envelope_entry_file_cross_check_fails_closed(tmp_path):
    data = _legacy_data()
    broken = copy.deepcopy(data)
    broken["_codedoc"]["entry_file"] = "other.py"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ConfigError):
        read_codedoc_document(path)

def test_legacy_json_without_last_run_is_accepted_and_owned(tmp_path):
    # A 0.11.6 document: has `run`, has no `last_run`.  It must still be owned and
    # readable so a newer build can safely overwrite it.
    data = _legacy_data()
    data.pop("last_run")
    path = tmp_path / "codedoc.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert _is_codedoc_owned(path)
    doc = read_codedoc_document(path)
    assert doc.view["last_run"]["entry_file"] == "main.py"
    assert "run" not in doc.view
    # The canonical counter maps from the legacy run's provider-work counter,
    # not its output file-count field.
    assert doc.view["last_run"]["files_documented_by_llm"] == 1

def test_current_json_without_envelope_is_owned(tmp_path):
    data = json.loads(json_from_view(run_metadata_view()))
    path = tmp_path / "current.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert _is_codedoc_owned(path)
    assert read_codedoc_document(path).entry_file == "main.py"

def test_empty_legacy_envelope_fails_closed(tmp_path):
    data = json.loads(json_from_view(run_metadata_view()))
    data["_codedoc"] = {}
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert not _is_codedoc_owned(path)
    with pytest.raises(ConfigError):
        read_codedoc_document(path)

def test_legacy_completed_json_remains_owned_and_readable(tmp_path):
    data = json.loads(json_from_view(json_contract_view()))
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
    assert "run" not in doc.view
    assert doc.view["last_run"]["files_documented_by_llm"] == 2

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
