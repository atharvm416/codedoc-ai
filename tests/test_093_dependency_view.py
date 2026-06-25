"""0.9.3 — dependency classification integrated into the project view.

Covers SDK/external separation in file links, graph edges as the ONLY source
of internal links, deferred internal catalog-hint validation, (type, name)
catalog grouping, and SDK link preservation through JSON/Markdown.
"""
from __future__ import annotations

import json

from codedoc.core.project_view import (
    build_project_view,
    json_from_view,
    markdown_from_view,
    markdown_to_view,
)


def _record(path, language, *, external=None, imports=None, catalog_updates=None,
            dependency_refs=None, usage_notes=None):
    # 0.10.1 (Workstream F): public external/sdk links are projected from the
    # parser ``imports``, not from the model ``dependencies_analysis.external``.
    # These tests express the parser-found dependency names; by default the
    # ``external`` argument *is* the parser import list so each case states the
    # deterministic input once.  ``dependencies_analysis.external`` is still set
    # so the (now bounded) model-enrichment path is exercised, but it can no
    # longer create a public link on its own.
    deps = {}
    if external is not None:
        deps["external"] = external
    if catalog_updates is not None:
        deps["catalog_updates"] = catalog_updates
    if dependency_refs is not None:
        deps["dependency_refs"] = dependency_refs
    if usage_notes is not None:
        deps["usage_notes"] = usage_notes
    parser_imports = imports if imports is not None else (external or [])
    return {
        "hash": f"h-{path}",
        "file_path": path,
        "language": language,
        "documentation": {
            "file_path": path,
            "language": language,
            "description": "d",
            "imports": parser_imports,
            "dependencies_analysis": deps,
        },
    }


# ---------------------------------------------------------------------------
# SDK / external separation
# ---------------------------------------------------------------------------

def test_sdk_and_external_links_are_separated():
    view = build_project_view(
        [_record("main.py", "python", external=["os", "requests", "sys"])],
        {"checked": 1},
    )
    links = view["files"][0]["links"]
    assert links["external_dependencies"] == ["requests"]
    assert links["sdk_dependencies"] == ["os", "sys"]


def test_dart_sdk_separated_from_packages():
    view = build_project_view(
        [_record("lib/main.dart", "dart",
                 external=["dart:async", "package:flutter/material.dart"])],
        {"checked": 1},
    )
    links = view["files"][0]["links"]
    assert links["external_dependencies"] == ["flutter"]
    assert links["sdk_dependencies"] == ["dart:async"]


def test_external_links_are_sorted_and_deduplicated():
    view = build_project_view(
        [_record("m.py", "python", external=["requests", "flask", "requests", "flask"])],
        {"checked": 1},
    )
    assert view["files"][0]["links"]["external_dependencies"] == ["flask", "requests"]


# ---------------------------------------------------------------------------
# Internal links come only from graph edges
# ---------------------------------------------------------------------------

def test_internal_links_come_only_from_graph_edges():
    edges = [{"from": "main.py", "to": "utils.py", "type": "internal_import"}]
    view = build_project_view(
        [_record("main.py", "python"), _record("utils.py", "python")],
        {"checked": 2},
        graph_edges=edges,
    )
    by_path = {f["path"]: f for f in view["files"]}
    assert by_path["main.py"]["links"]["internal_dependencies"] == ["utils.py"]
    assert by_path["utils.py"]["links"]["imported_by"] == ["main.py"]


def test_unresolved_agent_text_cannot_create_internal_link():
    # A path-like parser import must never become an internal link — internal
    # comes only from graph edges (none supplied here); it stays external.
    view = build_project_view(
        [_record("main.py", "python", external=["pkg/submodule"])],
        {"checked": 1},
    )
    links = view["files"][0]["links"]
    assert "internal_dependencies" not in links
    assert links["external_dependencies"] == ["pkg/submodule"]


def test_model_only_external_name_never_becomes_a_public_link():
    # 0.10.1 (Workstream F): a model-supplied dependency name that is NOT in the
    # parser imports can never appear in public links.  Parser imports are the
    # sole authority for dependency identity.
    record = _record("main.py", "python", imports=["os"])
    record["documentation"]["dependencies_analysis"]["external"] = ["requests", "evil_pkg"]
    view = build_project_view([record], {"checked": 1})
    links = view["files"][0]["links"]
    assert links["sdk_dependencies"] == ["os"]
    assert "external_dependencies" not in links  # model-only names dropped


def test_project_import_resolved_by_graph_edge_is_not_external():
    # A project import (``codedoc.*``) that resolves to a graph edge must be an
    # internal link, never an external dependency.
    edges = [{"from": "codedoc/cli.py", "to": "codedoc/core/x.py", "type": "internal_import"}]
    view = build_project_view(
        [
            _record("codedoc/cli.py", "python", imports=["codedoc.core.x", "os"]),
            _record("codedoc/core/x.py", "python"),
        ],
        {"checked": 2},
        graph_edges=edges,
    )
    links = {f["path"]: f.get("links", {}) for f in view["files"]}["codedoc/cli.py"]
    assert links["internal_dependencies"] == ["codedoc/core/x.py"]
    assert links["sdk_dependencies"] == ["os"]
    assert "codedoc" not in links.get("external_dependencies", [])


# ---------------------------------------------------------------------------
# Catalog: deferred internal hint validation + (type, name) grouping
# ---------------------------------------------------------------------------

def test_internal_catalog_hint_accepted_only_for_exact_resolved_path():
    edges = [{"from": "main.py", "to": "utils.py", "type": "internal_import"}]
    view = build_project_view(
        [
            _record("main.py", "python", catalog_updates=[
                {"name": "utils.py", "type": "internal", "used_for": "helpers"},
            ]),
            _record("utils.py", "python"),
        ],
        {"checked": 2},
        graph_edges=edges,
    )
    catalog = {(c["type"], c["name"]): c for c in view.get("dependency_catalog", [])}
    assert ("internal", "utils.py") in catalog
    assert catalog[("internal", "utils.py")]["used_for"] == "helpers"


def test_unresolved_internal_hint_is_discarded_not_reclassified():
    # An internal hint with no resolved graph link and no finalized external
    # link for the same name carries no authoritative evidence — it is discarded
    # entirely, never reclassified into a fabricated external entry.
    view = build_project_view(
        [_record("main.py", "python", catalog_updates=[
            {"name": "requests", "type": "internal", "used_for": "http"},
        ])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_catalog_groups_by_type_and_canonical_name():
    # Entries are evidence-backed: each file declares the finalized external
    # link, and the catalog_update only supplies used_for text.
    view = build_project_view(
        [
            _record("a.py", "python", external=["requests"], catalog_updates=[
                {"name": "requests", "type": "external", "used_for": "http"}]),
            _record("b.py", "python", external=["requests"], catalog_updates=[
                {"name": "requests.adapters", "type": "external", "used_for": "http"}]),
        ],
        {"checked": 2},
    )
    catalog = [c for c in view["dependency_catalog"] if c["name"] == "requests"]
    assert len(catalog) == 1
    assert catalog[0]["file_count"] == 2
    assert sorted(catalog[0]["files"]) == ["a.py", "b.py"]


def test_external_and_sdk_same_name_are_distinct_catalog_entries():
    # `typing` is python-sdk; in an unknown language the same name is external.
    # Each entry must be backed by that file's finalized link, not the
    # catalog_update text alone.
    view = build_project_view(
        [
            _record("a.py", "python", external=["typing"], catalog_updates=[
                {"name": "typing", "used_for": "hints"}]),
            _record("b.rb", "ruby", external=["typing"], catalog_updates=[
                {"name": "typing", "used_for": "thing"}]),
        ],
        {"checked": 2},
    )
    keys = {(c["type"], c["name"]) for c in view["dependency_catalog"]}
    assert ("sdk", "typing") in keys
    assert ("external", "typing") in keys


def test_bare_external_link_without_usage_does_not_fill_catalog():
    view = build_project_view(
        [_record("m.py", "python", external=["requests"])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_catalog_update_without_used_for_does_not_fill_catalog():
    view = build_project_view(
        [_record("m.py", "python", catalog_updates=[
            {"name": "requests", "type": "external", "used_for": ""},
        ])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


# ---------------------------------------------------------------------------
# SDK preservation through JSON / Markdown
# ---------------------------------------------------------------------------

def test_json_preserves_sdk_dependencies():
    view = build_project_view(
        [_record("m.py", "python", external=["os", "requests"])],
        {"checked": 1},
    )
    payload = json.loads(json_from_view(view))
    assert payload["files"][0]["links"]["sdk_dependencies"] == ["os"]


def test_markdown_round_trip_preserves_sdk_dependencies():
    view = build_project_view(
        [_record("m.py", "python", external=["os", "requests"])],
        {"checked": 1},
    )
    md = markdown_from_view(view)
    assert "SDK / Standard Library" in md
    round_tripped = markdown_to_view(md)
    assert round_tripped["files"][0]["links"]["sdk_dependencies"] == ["os"]


def test_visible_markdown_renders_sdk_label_only_when_non_empty():
    view = build_project_view(
        [_record("m.py", "python", external=["requests"])],
        {"checked": 1},
    )
    md = markdown_from_view(view)
    visible = md.split("-->", 2)[-1]
    assert "SDK / Standard Library" not in visible
