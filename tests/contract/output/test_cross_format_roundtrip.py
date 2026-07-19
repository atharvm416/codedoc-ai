"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from tests.support.markdown_cases import _make_view
from codedoc.core.document import read_codedoc_document
from codedoc.core.markdown_view import (
    json_from_markdown,
    markdown_from_view,
    read_embedded_view,
)
from codedoc.core.project_view import json_from_view
from tests.support.versionless_documents import _assert_versionless
from tests.support.versionless_documents import _view as versionless_view
from codedoc.core.document import records_by_path
from codedoc.core.record_meta import CACHE_IDENTITY_KEYS
from codedoc.pipeline import run_pipeline
from tests.support.profiles import INLINE
from tests.support.providers import SmartFake
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _forbid_provider
from codedoc.core.markdown_view import markdown_to_view
from tests.support.run_metadata_cases import _view as run_metadata_view
from codedoc.core.markdown_view import markdown_from_json

def test_public_output_converts_json_and_markdown_without_llm():
    import json

    from codedoc.core.project_view import json_from_markdown, markdown_from_json

    view = {
        "schema_version": "1.3",
        "generated_at": "2026-05-02T00:00:00+00:00",
        "project": {
            "entry_file": "main.py",
            "file_count": 1,
            "languages": ["python"],
            "folders": ["."],
        },
        "run": {
            "files_checked": 1,
            "files_failed": 0,
            "files_skipped": 0,
            "files_reused": 0,
            "files_documented": 1,
        },
        "tree": {"main.py": {"type": "file", "path": "main.py"}},
        "folders": [
            {
                "path": ".",
                "summary": "Root-level python files (1 file(s)). Common concepts: entry point.",
                "file_count": 1,
                "languages": ["python"],
                "files": ["main.py"],
                "key_concepts": ["entry point"],
            }
        ],
        "dependency_graph": [],
        "files": [
            {
                "id": "abc123",
                "path": "main.py",
                "format": "py",
                "language": "python",
                "description": "Main entry point.",
                "role_in_system": "Starts the app.",
                "imports": ["utils"],
                "functions": [{"name": "main", "description": "Runs the app."}],
                "key_concepts": ["entry point"],
                "usage_example": "python main.py",
                "links": {
                    "internal_dependencies": [],
                    "imported_by": [],
                    "external_dependencies": ["click"],
                },
            }
        ],
    }

    markdown = markdown_from_json(view)
    converted = json.loads(json_from_markdown(markdown))

    assert "## Project Overview" in markdown
    assert converted["last_run"]["entry_file"] == "main.py"
    assert "project" not in converted
    assert "run" not in converted
    assert converted["files"][0]["path"] == "main.py"
    assert converted["files"][0]["description"] == "Main entry point."
    assert converted["files"][0]["functions"] == [
        {"name": "main", "description": "Runs the app."}
    ]
    assert converted["files"][0]["links"]["external_dependencies"] == ["click"]

def test_markdown_to_json_does_not_create_empty_default_sections():
    import json

    from codedoc.core.project_view import json_from_markdown

    markdown = """# codedoc project documentation

## Project Overview

- Entry file: `main.py`
- Files documented: 1
- Languages: python
- Folders: `.`

## Run Summary

- Files checked: 1
- Files failed: 0
- Files skipped: 0
- Files reused from cache: 0

## Project Tree

```text
main.py
```

## Files

### main.py

**ID:** `abc123`  
**Format:** py  
**Language:** python  

**Description:** Main entry point.

"""

    converted = json.loads(json_from_markdown(markdown))

    assert "dependency_graph" not in converted
    assert "dependency_catalog" not in converted
    assert "links" not in converted["files"][0]

def test_A5_json_from_markdown_uses_embedded_view():
    """A5: json_from_markdown() returns data from the embedded view, not the visible parser."""
    from codedoc.core.project_view import markdown_from_view, json_from_markdown

    view = _make_view()
    md = markdown_from_view(view)
    regenerated = json.loads(json_from_markdown(md))

    # The embedded view is used, so dependency_catalog must be intact.
    assert "dependency_catalog" in regenerated, "dependency_catalog must be present"
    assert len(regenerated["dependency_catalog"]) > 0

    # dependency_graph must be intact.
    assert "dependency_graph" in regenerated
    edges = regenerated["dependency_graph"]
    assert any(e["from"] == "main.py" and e["to"] == "utils.py" for e in edges)

def test_A5_json_from_markdown_file_hashes_preserved():
    """A5b: file hashes survive the Markdown → JSON round-trip."""
    from codedoc.core.project_view import markdown_from_view, json_from_markdown

    view = _make_view()
    md = markdown_from_view(view)
    regenerated = json.loads(json_from_markdown(md))

    file_map = {f["path"]: f for f in regenerated["files"]}
    assert file_map["main.py"]["hash"] == "abc123"
    assert file_map["utils.py"]["hash"] == "def456"

def test_A11_direct_json_equals_regen_json():
    """A11: json_from_view and json_from_markdown(markdown_from_view) are equivalent."""
    from codedoc.core.project_view import (
        markdown_from_view,
        json_from_view,
        json_from_markdown,
    )

    view = _make_view()
    direct_json = json.loads(json_from_view(view))
    md = markdown_from_view(view)
    regen_json = json.loads(json_from_markdown(md))

    # Strip the _codedoc wrapper (added by json_from_view) before comparing the
    # inner view fields — the wrapper's generated_at is fine to differ.
    for data in (direct_json, regen_json):
        data.pop("_codedoc", None)

    # Files must match exactly (path, hash, language, description, etc.)
    direct_by_path = {f["path"]: f for f in direct_json.get("files", [])}
    regen_by_path = {f["path"]: f for f in regen_json.get("files", [])}
    assert set(direct_by_path) == set(regen_by_path), "Same file paths in both outputs"
    for path, direct_file in direct_by_path.items():
        regen_file = regen_by_path[path]
        assert direct_file == regen_file, (
            f"File record for '{path}' differs:\n  direct={direct_file}\n  regen={regen_file}"
        )

    # dependency_graph must match exactly
    assert direct_json.get("dependency_graph") == regen_json.get("dependency_graph")

    # dependency_catalog must match exactly
    assert direct_json.get("dependency_catalog") == regen_json.get("dependency_catalog")

    assert "schema_version" not in direct_json
    assert "schema_version" not in regen_json

def test_A12_rich_record_survives_round_trip():
    """A12: All rich fields survive the Markdown → embedded view → JSON round-trip."""
    from codedoc.core.project_view import markdown_from_view, json_from_markdown

    rich_files = [
        {
            "hash": "h1hash",
            "path": "src/app.py",
            "language": "python",
            "description": "App with -- dashes and --> arrows.",
            "role_in_system": "Core app logic.",
            "imports": ["os", "sys"],
            "functions": [
                {"name": "run", "description": "Entry point."},
                {"name": "setup", "description": "Setup."},
            ],
            "classes": [{"name": "App", "description": "Main app class."}],
            "exports": ["run", "App"],
            "key_concepts": ["lifecycle", "configuration"],
            "usage_example": "app = App()\napp.run()  # --> starts the app",
            "_deps": {
                "external": ["flask", "sqlalchemy"],
                "internal": ["src/db.py"],
                "dependency_refs": ["flask", "sqlalchemy"],
            },
            "links": {
                "internal_dependencies": ["src/db.py"],
                "imported_by": [],
                "external_dependencies": ["flask", "sqlalchemy"],
            },
        },
        {
            "hash": "h2hash",
            "path": "src/db.py",
            "language": "python",
            "description": "Database layer.",
            "key_concepts": ["database", "orm"],
            "_deps": {"external": ["sqlalchemy"]},
            "links": {
                "imported_by": ["src/app.py"],
                "external_dependencies": ["sqlalchemy"],
            },
        },
    ]

    view = _make_view(
        entry_file="src/app.py",
        files=rich_files,
        graph_edges=[{"from": "src/app.py", "to": "src/db.py", "type": "internal_import"}],
        dep_catalog=[
            {"name": "flask", "type": "external", "used_for": "web framework",
             "files": ["src/app.py"], "file_count": 1},
            {"name": "sqlalchemy", "type": "external", "used_for": "ORM",
             "files": ["src/app.py", "src/db.py"], "file_count": 2},
        ],
    )

    md = markdown_from_view(view)
    regen = json.loads(json_from_markdown(md))

    file_map = {f["path"]: f for f in regen["files"]}
    app = file_map["src/app.py"]

    # Hash
    assert app["hash"] == "h1hash"
    # Description with special chars intact
    assert "-->" in app["description"]
    # role_in_system
    assert app["role_in_system"] == "Core app logic."
    # imports
    assert "os" in app["imports"] and "sys" in app["imports"]
    # functions
    func_names = [fn["name"] for fn in app.get("functions", [])]
    assert "run" in func_names and "setup" in func_names
    # classes
    class_names = [c["name"] for c in app.get("classes", [])]
    assert "App" in class_names
    # exports
    assert "run" in app.get("exports", [])
    # key_concepts
    assert "lifecycle" in app.get("key_concepts", [])
    # usage_example with '-->'
    assert "-->" in app.get("usage_example", "")
    # _deps
    assert "_deps" in app
    # links
    links = app.get("links", {})
    assert "src/db.py" in links.get("internal_dependencies", [])

    # dependency_catalog
    catalog = {c["name"]: c for c in regen.get("dependency_catalog", [])}
    assert "flask" in catalog
    assert "sqlalchemy" in catalog
    assert catalog["sqlalchemy"]["file_count"] == 2

    # dependency_graph
    edges = regen.get("dependency_graph", [])
    assert any(e["from"] == "src/app.py" and e["to"] == "src/db.py" for e in edges)

    # folder summaries survived — _make_view() builds a single "." folder
    folders = {f["path"]: f for f in regen.get("folders", [])}
    assert len(folders) > 0, "At least one folder entry must survive the round-trip"
    # The manually constructed view uses "." as the folder path
    folder_entry = next(iter(folders.values()))
    assert folder_entry.get("file_count", 0) >= 1

def test_final_json_markdown_and_round_trip_are_recursively_versionless(tmp_path):
    view = versionless_view()
    json_data = json.loads(json_from_view(view))
    markdown = markdown_from_view(view)
    embedded = read_embedded_view(markdown)
    regenerated = json.loads(json_from_markdown(markdown))

    _assert_versionless(view)
    _assert_versionless(json_data)
    _assert_versionless(embedded)
    _assert_versionless(regenerated)
    assert "schema_version" not in markdown
    assert {k: v for k, v in json_data.items() if k != "_codedoc"} == {
        k: v for k, v in regenerated.items() if k != "_codedoc"
    }

    json_path = tmp_path / "codedoc.json"
    md_path = tmp_path / "codedoc.md"
    json_path.write_text(json_from_view(view), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    assert read_codedoc_document(json_path).schema_version is None
    assert read_codedoc_document(md_path).schema_version is None

def test_profile_filtered_fields_and_private_metadata_survive_conversion(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    first_fake = SmartFake("SAFE")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: first_fake)
    config = {**_config("json"), "prompt_profiles": INLINE}
    run_pipeline(tmp_path, config)

    original = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.json")
    )["main.py"]
    assert "functions" not in original
    assert "role_in_system" not in original
    assert "_prompt_profile_digest" in original

    _forbid_provider(monkeypatch)
    stats = run_pipeline(tmp_path, {**config, "output_format": "md"})
    converted = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]

    assert stats["checked"] == 0
    assert "functions" not in converted
    assert "role_in_system" not in converted
    for key in CACHE_IDENTITY_KEYS:
        assert converted.get(key) == original.get(key)

def test_last_run_round_trips_through_markdown_and_json():
    view = run_metadata_view()

    # Markdown -> view is lossless through the embedded base64 block.
    back_md = markdown_to_view(markdown_from_view(view))
    assert back_md["last_run"] == view["last_run"]

    # JSON serialise/parse preserves last_run byte-for-byte in the payload.
    assert json.loads(json_from_view(view))["last_run"] == view["last_run"]

def _profile_shaped_view():
    # A record as it would be persisted after a profile omitted role_in_system
    # and reordered prompt fields. The public vocabulary is unchanged, so the
    # deterministic embedded-view round-trip must hold.
    return {
        "schema_version": "1.4",
        "project": {"entry_file": "a.py", "file_count": 1,
                    "languages": ["python"], "folders": ["."]},
        "run": {"files_checked": 1, "files_failed": 0, "files_skipped": 0,
                "files_reused": 0, "files_documented": 1},
        "files": [{
            "path": "a.py", "language": "python", "hash": "abc",
            "description": "kept", "exports": ["E"], "key_concepts": ["k"],
            "_prompt_profile_digest": "pp-v1:abc",
            "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
        }],
    }

def test_json_markdown_round_trip_preserves_profile_shaped_record():
    view = _profile_shaped_view()
    md = markdown_from_json(json.dumps(view))
    back = json.loads(json_from_markdown(md))
    rec = next(f for f in back["files"] if f["path"] == "a.py")
    assert rec["description"] == "kept"
    assert "role_in_system" not in rec
    # Private cache keys survive the embedded-view round-trip.
    assert rec["_prompt_profile_digest"] == "pp-v1:abc"
    assert rec["_analysis_mode"] == "single"
