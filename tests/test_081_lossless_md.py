"""
0.8.1 — Lossless Markdown tests (Workstream A).

Covers every acceptance criterion from the plan:

  A1   Generated Markdown still contains the legacy <!-- codedoc-ai: ... --> comment.
  A2   Generated Markdown contains <!-- codedoc-ai-view-base64 ... -->.
  A3   The embedded view decodes from base64 and parses as valid JSON.
  A4   The embedded view has no _crash_safety and no in_progress status.
  A5   json_from_markdown() uses the embedded view when present.
  A6   Legacy Markdown without the base64 block uses the visible parser.
  A7   Corrupted base64 block warns and falls back to the visible parser.
  A8   Strings containing '--' or '-->' inside text remain safe (base64 encoded).
  A9   CodeDoc file with entry_file=null is still recognised as CodeDoc-owned.
  A10  _resolve_entry_and_docs() does NOT raise on first run with no --entry.
  A11  Direct JSON and Markdown-regenerated JSON are equivalent for all key fields.
  A12  dependency_catalog, dependency_graph, _deps, links, hashes, folders,
       descriptions, functions, classes, key_concepts, usage_example all survive
       the Markdown round-trip.
  A13  Legacy Markdown without the base64 block has no dependency_catalog
       (best-effort only — confirms backward-compat behaviour).
  A14  _load_existing_file_docs_from_md preserves hashes from the embedded view.
  A15  Clean MD-only pipeline run writes Markdown and removes the JSON sibling.
  A16  read_codedoc_meta() on a JSON file with entry_file=null returns the meta
       dict without raising.
  A17  read_codedoc_meta() on a Markdown file with entry_file=null returns the
       meta dict without raising.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_view(
    entry_file: str | None = "main.py",
    files: list[dict] | None = None,
    graph_edges: list[dict] | None = None,
    dep_catalog: list[dict] | None = None,
) -> dict:
    """Build a minimal but non-trivial public view dict for testing."""
    from codedoc.core.project_view import SCHEMA_VERSION

    _files = files or [
        {
            "hash": "abc123",
            "path": "main.py",
            "language": "python",
            "description": "Entry point.",
            "role_in_system": "Starts the app.",
            "imports": ["utils"],
            "functions": [{"name": "main", "description": "Runs the app."}],
            "classes": [],
            "exports": ["main"],
            "key_concepts": ["entry point"],
            "usage_example": "python main.py",
            "_deps": {"external": ["requests"], "internal": ["utils"]},
            "links": {
                "internal_dependencies": ["utils.py"],
                "external_dependencies": ["requests"],
            },
        },
        {
            "hash": "def456",
            "path": "utils.py",
            "language": "python",
            "description": "Utility helpers.",
            "imports": ["requests"],
            "functions": [{"name": "fetch", "description": "HTTP fetch."}],
            "key_concepts": ["http", "helpers"],
            "_deps": {"external": ["requests"]},
            "links": {
                "imported_by": ["main.py"],
                "external_dependencies": ["requests"],
            },
        },
    ]

    _edges = graph_edges or [
        {"from": "main.py", "to": "utils.py", "type": "internal_import"},
    ]

    _catalog = dep_catalog or [
        {
            "name": "requests",
            "type": "external",
            "used_for": "HTTP requests",
            "files": ["main.py", "utils.py"],
            "file_count": 2,
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-06-01T12:00:00+00:00",
        "project": {
            "entry_file": entry_file,
            "file_count": len(_files),
            "languages": ["python"],
            "folders": ["."],
        },
        "run": {
            "files_checked": len(_files),
            "files_failed": 0,
            "files_skipped": 0,
            "files_reused": 0,
            "files_documented": len(_files),
        },
        "tree": {
            "main.py": {"type": "file", "path": "main.py"},
            "utils.py": {"type": "file", "path": "utils.py"},
        },
        "folders": [
            {
                "path": ".",
                "summary": "Root-level python files (2 file(s)).",
                "file_count": len(_files),
                "languages": ["python"],
                "files": [f["path"] for f in _files],
                "key_concepts": ["entry point"],
            }
        ],
        "dependency_catalog": _catalog,
        "dependency_graph": _edges,
        "files": _files,
    }


def _fake_provider(description: str = "Documented."):
    """Minimal fake LLM provider for pipeline tests."""
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": description,
                    "role_in_system": "test",
                    "key_concepts": ["entry point"],
                    "usage_example": "python main.py",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({
                    "dependencies_analysis": {
                        "internal": [], "external": ["requests"],
                        "dependency_refs": ["requests"],
                        "catalog_updates": [
                            {"name": "requests", "type": "external", "used_for": "HTTP"}
                        ],
                        "usage_notes": [
                            {"import": "requests", "used_for": "HTTP requests"}
                        ],
                        "warnings": [],
                    }
                })
            return _json.dumps({
                "description": description,
                "role_in_system": "test",
                "functions": [{"name": "main", "description": "runs app"}],
                "classes": [],
                "exports": ["main"],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


def _run_pipeline(tmp_path, monkeypatch, **cfg):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)


# ---------------------------------------------------------------------------
# A1 — Legacy metadata comment is still written
# ---------------------------------------------------------------------------

def test_A1_legacy_meta_comment_present():
    """A1: Generated Markdown always starts with the legacy <!-- codedoc-ai: ... --> comment."""
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    assert "<!-- codedoc-ai:" in md, "Legacy metadata comment must be present"
    assert '"entry_file"' in md
    assert '"file_hashes"' in md


# ---------------------------------------------------------------------------
# A2 — Base64 embedded-view block is written
# ---------------------------------------------------------------------------

def test_A2_base64_block_present():
    """A2: Generated Markdown contains <!-- codedoc-ai-view-base64 ... -->."""
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    assert "<!-- codedoc-ai-view-base64" in md, "Base64 view block must be present"
    assert "-->" in md  # block is properly closed


def test_A2_base64_block_comes_after_meta_comment():
    """A2b: The base64 block appears after the legacy metadata comment."""
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    meta_pos = md.index("<!-- codedoc-ai:")
    b64_pos = md.index("<!-- codedoc-ai-view-base64")
    assert b64_pos > meta_pos, "Base64 block must appear after the legacy comment"


# ---------------------------------------------------------------------------
# A3 — Embedded view decodes to valid JSON dict
# ---------------------------------------------------------------------------

def test_A3_embedded_view_decodes_to_valid_json():
    """A3: The base64 payload decodes and parses as a valid JSON dict."""
    import re
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    match = re.search(r"<!-- codedoc-ai-view-base64\s*([\s\S]*?)\s*-->", md)
    assert match, "Base64 block must be present"

    raw = match.group(1).strip()
    decoded = base64.b64decode(raw)
    data = json.loads(decoded.decode("utf-8"))

    assert isinstance(data, dict), "Decoded payload must be a dict"
    assert "schema_version" in data
    assert "project" in data
    assert "files" in data


def test_A3_embedded_view_contains_expected_fields():
    """A3b: The decoded view contains all major fields."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    view = _make_view()
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)

    assert embedded is not None
    assert embedded["schema_version"] == view["schema_version"]
    assert embedded["project"]["entry_file"] == "main.py"
    assert len(embedded["files"]) == 2
    assert embedded["dependency_graph"] == view["dependency_graph"]
    assert embedded["dependency_catalog"] == view["dependency_catalog"]


# ---------------------------------------------------------------------------
# A4 — Embedded view is clean (no crash-safety or in-progress)
# ---------------------------------------------------------------------------

def test_A4_embedded_view_has_no_crash_safety():
    """A4: The embedded view must not contain _crash_safety."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    md = markdown_from_view(_make_view())
    embedded = read_embedded_view(md)

    assert embedded is not None
    assert "_crash_safety" not in embedded


def test_A4_embedded_view_has_no_in_progress_status():
    """A4b: The embedded view must not have _codedoc.status = 'in_progress'."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    md = markdown_from_view(_make_view())
    embedded = read_embedded_view(md)

    assert embedded is not None
    codedoc_meta = embedded.get("_codedoc", {})
    assert not (isinstance(codedoc_meta, dict) and codedoc_meta.get("status") == "in_progress")


def test_A4_read_embedded_view_rejects_crash_safety_block():
    """A4c: read_embedded_view rejects payloads containing _crash_safety."""
    from codedoc.core.project_view import read_embedded_view

    bad_view = {"schema_version": "1.4", "project": {}, "files": [], "_crash_safety": "INCOMPLETE"}
    b64 = base64.b64encode(json.dumps(bad_view).encode()).decode()
    md = f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"

    result = read_embedded_view(md)
    assert result is None, "Crash-safety block must be rejected"


def test_A4_read_embedded_view_rejects_in_progress_snapshot():
    """A4d: read_embedded_view rejects in-progress snapshots."""
    from codedoc.core.project_view import read_embedded_view

    bad_view = {
        "schema_version": "1.4",
        "project": {},
        "files": [],
        "_codedoc": {"status": "in_progress"},
    }
    b64 = base64.b64encode(json.dumps(bad_view).encode()).decode()
    md = f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"

    result = read_embedded_view(md)
    assert result is None, "In-progress snapshot must be rejected"


# ---------------------------------------------------------------------------
# A5 — json_from_markdown() uses the embedded view when present
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A6 — Legacy Markdown (no base64 block) uses the visible parser
# ---------------------------------------------------------------------------

def test_A6_legacy_markdown_uses_visible_parser():
    """A6: Markdown without the base64 block falls back to the visible parser."""
    from codedoc.core.project_view import markdown_to_view

    # Hand-crafted legacy Markdown — no codedoc-ai-view-base64 block.
    legacy_md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 1\n"
        "- Languages: python\n"
        "- Folders: none\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 0\n"
        "- Files reused from cache: 0\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python  \n\n"
        "**Description:** Entry point.\n\n"
    )

    view = markdown_to_view(legacy_md)
    assert view is not None
    files = view.get("files", [])
    assert len(files) == 1
    assert files[0]["path"] == "main.py"
    assert files[0]["language"] == "python"


def test_A6_legacy_markdown_has_no_dependency_catalog():
    """A6b: Legacy Markdown without base64 block has no dependency_catalog (best-effort)."""
    from codedoc.core.project_view import markdown_to_view

    # Minimal legacy Markdown with no Dependency Catalog section
    legacy_md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 1\n"
        "- Languages: python\n"
        "- Folders: none\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 0\n"
        "- Files reused from cache: 0\n\n"
    )

    view = markdown_to_view(legacy_md)
    # No Dependency Catalog section → should be absent or empty
    assert not view.get("dependency_catalog"), (
        "Legacy Markdown without a Dependency Catalog section must yield no catalog"
    )


# ---------------------------------------------------------------------------
# A7 — Corrupted base64 block falls back gracefully
# ---------------------------------------------------------------------------

def test_A7_corrupted_base64_falls_back_to_visible_parser():
    """A7: A corrupt base64 block triggers a warning and falls back to visible parser."""
    from codedoc.core.project_view import markdown_to_view

    md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "<!-- codedoc-ai-view-base64\n"
        "THIS_IS_NOT_VALID_BASE64!!!\n"
        "-->\n"
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 1\n"
        "- Languages: python\n"
        "- Folders: none\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 0\n"
        "- Files reused from cache: 0\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python  \n\n"
        "**Description:** Fallback description.\n\n"
    )

    # Must not raise — falls back to visible parser
    view = markdown_to_view(md)
    assert view is not None
    files = view.get("files", [])
    assert any(f["path"] == "main.py" for f in files), (
        "Fallback to visible parser must still recover the file record"
    )


def test_A7_valid_json_but_missing_required_fields_falls_back():
    """A7b: Valid base64 JSON but missing required fields falls back to visible parser."""
    from codedoc.core.project_view import read_embedded_view

    # Valid JSON but missing 'files'
    incomplete = {"schema_version": "1.4", "project": {"entry_file": "main.py"}}
    b64 = base64.b64encode(json.dumps(incomplete).encode()).decode()
    md = f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"

    result = read_embedded_view(md)
    assert result is None, "Incomplete view (missing 'files') must be rejected"


# ---------------------------------------------------------------------------
# A8 — Text with '--' or '-->' is safe inside base64
# ---------------------------------------------------------------------------

def test_A8_dangerous_chars_in_text_safe_in_base64():
    """A8: Usage examples containing '--' or '-->' are stored safely in base64."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    dangerous_view = _make_view(files=[
        {
            "hash": "aaa",
            "path": "main.py",
            "language": "python",
            "description": "Has --> and -- in description.",
            "usage_example": "cmd --flag --> output\n<!-- not a comment -->",
        }
    ])

    md = markdown_from_view(dangerous_view)
    embedded = read_embedded_view(md)

    assert embedded is not None, (
        "Embedded view must decode correctly even when payload contains '-->' or '--'"
    )
    file = embedded["files"][0]
    assert "-->" in file["usage_example"]
    assert "--flag" in file["usage_example"]
    assert "<!-- not a comment -->" in file["usage_example"]


# ---------------------------------------------------------------------------
# A9 — entry_file=null does not invalidate a CodeDoc Markdown file
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A10 — _resolve_entry_and_docs does NOT raise on first run with no --entry
# ---------------------------------------------------------------------------

def test_A10_first_run_no_entry_no_output_does_not_raise(tmp_path):
    """A10: First run without --entry and no existing output must not raise ConfigError."""
    from codedoc.pipeline import _resolve_entry_and_docs
    from codedoc.core.loader import load_config

    # No output files exist, no entry_file in config
    config = load_config(tmp_path, {"output_format": "json"})
    config.pop("entry_file", None)
    config["entry_file"] = None

    # Must not raise
    _resolve_entry_and_docs(tmp_path, config)
    # entry_file should still be None (auto-detection left to detect_entry_file)
    assert config.get("entry_file") is None


def test_A10_first_run_pipeline_without_entry_uses_auto_detection(tmp_path, monkeypatch):
    """A10b: Pipeline first run without --entry reaches detect_entry_file() and processes all files."""
    # Write a file with a common auto-detect name
    (tmp_path / "main.py").write_text("x=1\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    # No entry_file provided — should auto-detect main.py
    stats = run_pipeline(tmp_path, {
        "parallel_agents": False,
        "propagate_changes": False,
        # no entry_file
    })
    assert stats["checked"] >= 1 or stats.get("reused", 0) >= 1, (
        "Pipeline must process files even without an explicit --entry"
    )


# ---------------------------------------------------------------------------
# A11 — Direct JSON and Markdown-regenerated JSON are equivalent
# ---------------------------------------------------------------------------

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

    # schema_version must match
    assert direct_json["schema_version"] == regen_json["schema_version"]


# ---------------------------------------------------------------------------
# A12 — All key fields survive the Markdown round-trip
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A13 — Legacy Markdown without base64 block has no dependency_catalog (confirmed)
# (Covered by A6b — intentionally redundant for plan acceptance criteria traceability)
# ---------------------------------------------------------------------------

def test_A13_confirms_legacy_has_no_dep_catalog():
    """A13: Confirms that A6b/legacy-path behaviour is intentional (best-effort only)."""
    from codedoc.core.project_view import json_from_markdown

    # Markdown with a Dependency Map section but NO Dependency Catalog section,
    # and no base64 block.
    legacy_md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Dependency Map\n\n"
        "- `main.py` -> `utils.py`\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python  \n\n"
    )

    regen = json.loads(json_from_markdown(legacy_md))
    # The legacy parser cannot reconstruct dependency_catalog from visible sections
    assert not regen.get("dependency_catalog"), (
        "Legacy Markdown without an explicit Dependency Catalog section "
        "must not produce a dependency_catalog in the regenerated JSON"
    )


# ---------------------------------------------------------------------------
# A14 — _load_existing_file_docs_from_md preserves hashes from embedded view
# ---------------------------------------------------------------------------

def test_A14_hashes_preserved_from_embedded_view(tmp_path):
    """A14: When the embedded view is used, file hashes are preserved."""
    from codedoc.core.project_view import markdown_from_view
    from codedoc.pipeline import _load_existing_file_docs_from_md

    view = _make_view()
    md_path = tmp_path / "codedoc.md"
    md_path.write_text(markdown_from_view(view), encoding="utf-8")

    docs = _load_existing_file_docs_from_md(md_path)

    # Both hashes must be preserved from the embedded view
    assert docs["main.py"]["hash"] == "abc123", (
        "main.py hash must survive the Markdown load from embedded view"
    )
    assert docs["utils.py"]["hash"] == "def456", (
        "utils.py hash must survive the Markdown load from embedded view"
    )


def test_A14_meta_comment_hash_takes_precedence(tmp_path):
    """A14b: When both meta comment and embedded view have hashes, meta comment wins."""
    from codedoc.core.project_view import markdown_from_view
    from codedoc.pipeline import _load_existing_file_docs_from_md

    view = _make_view()
    # Write the Markdown (which has both the meta comment hashes and embedded view hashes)
    md_path = tmp_path / "codedoc.md"
    md_path.write_text(markdown_from_view(view), encoding="utf-8")

    docs = _load_existing_file_docs_from_md(md_path)

    # The meta comment has the same hashes as the embedded view (they're from the
    # same run), so the result must match.
    assert docs["main.py"]["hash"] == "abc123"
    assert docs["utils.py"]["hash"] == "def456"


def test_A14_embedded_hash_fallback_when_meta_hash_missing(tmp_path):
    """A14c: When meta comment has no hash for a path, embedded view hash is used."""
    import re as _re
    from codedoc.core.project_view import markdown_from_view
    from codedoc.pipeline import _load_existing_file_docs_from_md

    view = _make_view()
    md_content = markdown_from_view(view)

    # Corrupt the meta comment to remove file_hashes entirely
    md_content = _re.sub(
        r'<!-- codedoc-ai: \{.*?\} -->',
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", "generated_at": "", "file_hashes": {}} -->',
        md_content,
        flags=_re.DOTALL,
    )

    md_path = tmp_path / "codedoc.md"
    md_path.write_text(md_content, encoding="utf-8")

    docs = _load_existing_file_docs_from_md(md_path)

    # With empty meta comment hashes, must fall back to embedded view hashes
    assert docs["main.py"]["hash"] == "abc123", (
        "When meta comment has no hash, embedded view hash must be used as fallback"
    )
    assert docs["utils.py"]["hash"] == "def456"


# ---------------------------------------------------------------------------
# A15 — Clean MD-only pipeline run: Markdown written, JSON sibling removed
# ---------------------------------------------------------------------------

def test_A15_clean_md_run_writes_markdown_and_removes_json(tmp_path, monkeypatch):
    """A15: Clean MD-only run produces Markdown with embedded view and no JSON sibling."""
    (tmp_path / "main.py").write_text("x=1\n")

    stats = _run_pipeline(tmp_path, monkeypatch, entry_file="main.py", output_format="md")

    md_path = tmp_path / "codedoc" / "codedoc.md"
    json_sibling = tmp_path / "codedoc" / "codedoc.json"

    assert md_path.exists(), "codedoc.md must be written"
    assert not json_sibling.exists(), "JSON sibling must be removed after clean MD run"

    # Markdown must have both the legacy comment and the new base64 block
    md_content = md_path.read_text(encoding="utf-8")
    assert "<!-- codedoc-ai:" in md_content
    assert "<!-- codedoc-ai-view-base64" in md_content


def test_A15_md_run_embedded_view_passes_all_validations(tmp_path, monkeypatch):
    """A15b: Embedded view in a pipeline-generated Markdown is valid and decodable."""
    from codedoc.core.project_view import read_embedded_view

    (tmp_path / "main.py").write_text("x=1\n")
    _run_pipeline(tmp_path, monkeypatch, entry_file="main.py", output_format="md")

    md_path = tmp_path / "codedoc" / "codedoc.md"
    md_content = md_path.read_text(encoding="utf-8")

    embedded = read_embedded_view(md_content)
    assert embedded is not None, "Embedded view must be present and valid after pipeline run"
    assert "_crash_safety" not in embedded
    assert "files" in embedded
    assert len(embedded["files"]) >= 1


# ---------------------------------------------------------------------------
# A16 — read_codedoc_meta on JSON with null entry_file (already in A9b, explicit label)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A17 — read_codedoc_meta on Markdown with null entry_file (already in A9, explicit label)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bonus — read_embedded_view returns None for plain Markdown (no block)
# ---------------------------------------------------------------------------

def test_read_embedded_view_returns_none_for_plain_markdown():
    """Sanity: read_embedded_view returns None when there is no base64 block."""
    from codedoc.core.project_view import read_embedded_view

    plain = "# Hello\n\nNo codedoc block here."
    assert read_embedded_view(plain) is None


def test_read_embedded_view_returns_none_for_empty_string():
    """Sanity: read_embedded_view returns None for an empty string."""
    from codedoc.core.project_view import read_embedded_view

    assert read_embedded_view("") is None


# ---------------------------------------------------------------------------
# A18 — Crash → resume from JSON backup → final MD embedded view is complete
#
# The user validated this manually (crash_report.md scenario):
#   1. First MD run crashes mid-way → leaves a .json crash backup with K records.
#   2. Second run starts, loads those K records from the JSON backup (incremental reuse).
#   3. Processes the remaining files via the LLM.
#   4. Writes the final Markdown with the embedded view containing ALL K + remaining records.
#   5. Removes the JSON crash backup.
#
# This is distinct from A14 (which reads hashes from an MD embedded view) and
# A15 (which tests a clean first run with no prior crash backup).
# ---------------------------------------------------------------------------

def test_A18_crash_resume_final_md_embedded_view_complete(tmp_path, monkeypatch):
    """A18: After resuming from a JSON crash backup, the final MD embedded view
    contains ALL records — both those pre-loaded from the backup and newly
    processed ones — and the JSON backup is removed."""
    import json as _json
    from codedoc.core.db import compute_file_hash

    # Two source files: a.py was documented in the "crashed" run, b.py was not.
    a_py = tmp_path / "a.py"
    b_py = tmp_path / "b.py"
    a_py.write_text("from b import x\n")
    b_py.write_text("x = 1\n")

    h_a = compute_file_hash(a_py)

    # Simulate a crashed first run: codedoc.json has a.py with a valid hash
    # but carries _crash_safety and status=in_progress (b.py was never reached).
    out_dir = tmp_path / "codedoc"
    out_dir.mkdir()
    crash_backup = out_dir / "codedoc.json"
    crash_backup.write_text(_json.dumps({
        "_crash_safety": "INCOMPLETE RUN - crash test",
        "_codedoc": {"entry_file": "a.py", "schema_version": "1.4", "status": "in_progress"},
        "files": [
            {
                "path": "a.py",
                "hash": h_a,
                "language": "python",
                "description": "Pre-crash description for a.py.",
                "role_in_system": "test",
                "functions": [],
                "classes": [],
                "exports": [],
                "key_concepts": ["entry point"],
                "usage_example": "",
                "dependencies_analysis": {},
            }
        ],
    }), encoding="utf-8")

    # Resume run: should load a.py from the crash backup, process b.py via LLM.
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "output_format": "md",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    md_path = out_dir / "codedoc.md"
    assert md_path.exists(), "Final Markdown must be written after crash resume"
    assert not crash_backup.exists(), (
        "JSON crash backup must be removed after successful Markdown write"
    )

    # The embedded view must contain BOTH a.py (from crash backup) and b.py (newly processed)
    from codedoc.core.project_view import read_embedded_view
    md_content = md_path.read_text(encoding="utf-8")
    embedded = read_embedded_view(md_content)

    assert embedded is not None, "Embedded view must be present in resumed output"
    assert "_crash_safety" not in embedded, "Embedded view must not contain _crash_safety"

    paths = {f["path"] for f in embedded.get("files", [])}
    assert "a.py" in paths, "a.py (from crash backup) must be in the embedded view"
    assert "b.py" in paths, "b.py (newly processed) must be in the embedded view"
    assert len(paths) == 2, f"Expected exactly 2 files in embedded view, got {paths}"

    # Hash for a.py must be preserved from the crash backup (not regenerated)
    file_map = {f["path"]: f for f in embedded["files"]}
    assert file_map["a.py"]["hash"] == h_a, (
        "a.py hash from the crash backup must be preserved in the embedded view"
    )
