"""0.9.4 — serializer extraction (Workstream B) characterization.

These tests prove that moving the Markdown serializer/parser out of
``codedoc.core.project_view`` into ``codedoc.core.markdown_view`` is a pure
refactor:

- serialization of a fixed, already-built view is **byte-identical** to the
  golden output captured from the pre-extraction (0.9.3) code;
- the same names remain importable from ``project_view`` (one-release compat
  shim) and from the new defining module;
- legacy visible-Markdown parsing and the lossless embedded round trip are
  unchanged;
- private record keys survive into the embedded view and never leak into the
  visible prose;
- there is no circular import between the two modules.

The golden files live in ``tests/fixtures/golden_094_*`` and must be
regenerated only with an explicit, reviewed reason — a diff against them means
the serializer behavior changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoc.core import markdown_view, project_view
from codedoc.core.markdown_view import (
    markdown_from_view,
    markdown_to_view,
    read_embedded_view,
)
from codedoc.core.project_view import build_project_view, json_from_view

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixed input view — identical to the golden-snapshot generator.  Do not edit
# without regenerating the golden files (and a reviewed reason for doing so).
# ---------------------------------------------------------------------------

def _records() -> list[dict]:
    return [
        {
            "hash": "h1",
            "file_path": "src/main.py",
            "language": "python",
            "documentation": {
                "file_path": "src/main.py",
                "language": "python",
                "description": "Entry point — orchestrates startup. Café ☕ Unicode ✓",
                "role_in_system": "bootstrap",
                "imports": ["os", "requests", "src.helper"],
                "functions": [{"name": "main", "description": "Run the app"}],
                "classes": [{"name": "App", "description": "Main app -- with dashes"}],
                "exports": ["main"],
                "key_concepts": ["startup", "config"],
                "usage_example": "python -m src.main --flag\n# comment with --> tricky chars",
                "dependencies_analysis": {
                    "external": ["requests", "os"],
                    "usage_notes": [
                        {"import": "requests", "used_for": "HTTP calls to the API service"}
                    ],
                    "catalog_updates": [
                        {"name": "requests", "type": "external", "used_for": "HTTP client library"}
                    ],
                    "dependency_refs": ["requests"],
                },
            },
        },
        {
            "hash": "h2",
            "file_path": "src/helper.py",
            "language": "python",
            "documentation": {
                "file_path": "src/helper.py",
                "language": "python",
                "description": "Helper utilities.",
                "role_in_system": "utility",
                "imports": ["json", "sys"],
                "functions": [{"name": "load", "description": ""}],
                "classes": [],
                "exports": [],
                "key_concepts": ["io"],
                "usage_example": "",
                "dependencies_analysis": {"external": ["json", "sys"]},
            },
        },
        {
            "hash": "h3",
            "file_path": "README.md",
            "language": "markdown",
            "documentation": {
                "file_path": "README.md",
                "language": "markdown",
                "description": "Docs.",
                "role_in_system": "documentation",
                "dependencies_analysis": {},
            },
        },
    ]


def _stats() -> dict:
    return {"checked": 2, "failed": 0, "skipped": 1, "reused": 0}


def _edges() -> list[dict]:
    return [{"from": "src/main.py", "to": "src/helper.py", "type": "internal_import"}]


def _build_view() -> dict:
    return build_project_view(
        _records(), _stats(), entry_file="src/main.py", graph_edges=_edges()
    )


def _read_golden(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Byte-identical golden output (the governing property of this release)
# ---------------------------------------------------------------------------

def test_view_assembly_byte_identical():
    view = _build_view()
    assert json.dumps(view, indent=2, ensure_ascii=False) == _read_golden(
        "golden_094_view.json"
    )


def test_json_serialization_byte_identical():
    view = _build_view()
    assert json_from_view(view, "No errors.") == _read_golden("golden_094_output.json")


def test_markdown_serialization_byte_identical():
    view = _build_view()
    assert markdown_from_view(view, "Sample error\nsecond line") == _read_golden(
        "golden_094_output.md"
    )


def test_empty_view_json_byte_identical():
    empty = build_project_view([], {"checked": 0}, entry_file=None, graph_edges=[])
    assert json_from_view(empty) == _read_golden("golden_094_empty.json")


def test_empty_view_markdown_byte_identical():
    empty = build_project_view([], {"checked": 0}, entry_file=None, graph_edges=[])
    assert markdown_from_view(empty) == _read_golden("golden_094_empty.md")


def test_dependency_catalog_preserved_verbatim():
    """The catalog is preserved as-is in 0.9.4 (not corrected), so the
    byte-identical guarantee is meaningful."""
    view = _build_view()
    catalog = view.get("dependency_catalog", [])
    assert [(d["name"], d["type"], d["file_count"]) for d in catalog] == [
        ("requests", "external", 1)
    ]


# ---------------------------------------------------------------------------
# Lossless embedded round trip + legacy visible-Markdown parsing
# ---------------------------------------------------------------------------

def test_embedded_view_round_trip_lossless():
    view = _build_view()
    md = markdown_from_view(view)
    assert markdown_to_view(md) == view


def test_legacy_visible_markdown_parses():
    """Markdown without the base64 block falls back to the visible parser."""
    legacy = (
        "<!-- codedoc-ai: {\"entry_file\": \"main.py\", \"schema_version\": \"1.4\", "
        "\"file_hashes\": {}} -->\n"
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
        "**Description:** The entry point.\n\n"
    )
    assert read_embedded_view(legacy) is None  # no embedded block
    view = markdown_to_view(legacy)
    assert view["project"]["entry_file"] == "main.py"
    assert view["files"][0]["path"] == "main.py"
    assert view["files"][0]["description"] == "The entry point."


# ---------------------------------------------------------------------------
# Private record keys: survive into embedded view, absent from visible prose
# ---------------------------------------------------------------------------

def test_private_keys_survive_embedded_absent_from_visible(monkeypatch):
    monkeypatch.setattr(
        "codedoc.core.record_meta.PRIVATE_RECORD_KEYS", frozenset({"_secret_marker"})
    )
    records = [
        {
            "hash": "h",
            "file_path": "a.py",
            "language": "python",
            "_secret_marker": "KEEPME",
            "documentation": {"description": "x", "dependencies_analysis": {}},
        }
    ]
    view = build_project_view(records, {"checked": 1}, entry_file="a.py")
    assert view["files"][0]["_secret_marker"] == "KEEPME"

    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert embedded["files"][0]["_secret_marker"] == "KEEPME"
    # The visible Markdown (with the hidden base64 block removed) must not leak it.
    import re

    visible = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    assert "_secret_marker" not in visible
    assert "KEEPME" not in visible


# ---------------------------------------------------------------------------
# Compatibility shim + defining module
# ---------------------------------------------------------------------------

MOVED_NAMES = [
    "markdown_from_view",
    "markdown_to_view",
    "json_from_markdown",
    "markdown_from_json",
    "read_embedded_view",
    "read_embedded_view_result",
    "EmbeddedViewResult",
    "_build_full_view_comment",
    "_public_view_for_embedding",
    "_build_meta_comment",
]


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_moved_names_importable_from_both_modules(name):
    from_mv = getattr(markdown_view, name)
    from_pv = getattr(project_view, name)  # via project_view.__getattr__ shim
    assert from_mv is from_pv


def test_project_view_retains_assembly_api():
    for name in ("build_project_view", "json_from_view", "clean_file_record", "read_codedoc_meta"):
        assert hasattr(project_view, name)


def test_project_view_getattr_rejects_unknown():
    with pytest.raises(AttributeError):
        project_view.this_name_does_not_exist  # noqa: B018


def test_no_circular_import_between_serializer_modules():
    import importlib

    # Importing either first must succeed (markdown_view -> project_view is
    # one-way; project_view -> markdown_view is lazy via __getattr__).
    for first, second in (
        ("codedoc.core.project_view", "codedoc.core.markdown_view"),
        ("codedoc.core.markdown_view", "codedoc.core.project_view"),
    ):
        importlib.import_module(first)
        importlib.import_module(second)
    import codedoc.core.project_view as pv

    src = Path(pv.__file__).read_text(encoding="utf-8")
    # The only reference to markdown_view in project_view is the lazy __getattr__.
    assert src.count("import markdown_view") == 1


def test_project_view_shim_exposes_only_scheduled_names():
    """Unrelated markdown-view internals must not leak through the shim."""
    assert hasattr(markdown_view, "_append_file_markdown")
    with pytest.raises(AttributeError):
        project_view._append_file_markdown
