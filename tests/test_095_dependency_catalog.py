"""0.9.5 — evidence-based dependency-catalog admission.

The dependency catalog may only contain entries whose ``(type, canonical_name)``
key is authorized by a file's finalized links (graph-resolved internal links or
deterministically classified external/SDK links).  Model-supplied
``catalog_updates``, ``dependency_refs``, and ``usage_notes`` may enrich a proven
dependency with ``used_for`` text but can never create or retype one.  Every
emitted entry carries non-empty ``used_for`` text and at least one backing file.

These tests reproduce the §3 fixture: unresolved internal symbol/function hints,
an exact resolved internal path, a real third-party dependency, a standard
library dependency, and the project's own package root incorrectly reported as
external.
"""
from __future__ import annotations

from codedoc.core.project_view import build_project_view


def _record(path, language, *, external=None, catalog_updates=None,
            dependency_refs=None, usage_notes=None):
    deps = {}
    if external is not None:
        deps["external"] = external
    if catalog_updates is not None:
        deps["catalog_updates"] = catalog_updates
    if dependency_refs is not None:
        deps["dependency_refs"] = dependency_refs
    if usage_notes is not None:
        deps["usage_notes"] = usage_notes
    return {
        "hash": f"h-{path}",
        "file_path": path,
        "language": language,
        "documentation": {
            "file_path": path,
            "language": language,
            "description": "d",
            "dependencies_analysis": deps,
        },
    }


def _catalog(view):
    return {(c["type"], c["name"]): c for c in view.get("dependency_catalog", [])}


# ---------------------------------------------------------------------------
# Unresolved internal symbol / function hints are discarded (A1)
# ---------------------------------------------------------------------------

def test_unresolved_internal_symbol_hint_is_discarded():
    # `BaseAgent` is an internal symbol the agent guessed at; no graph edge
    # resolves it and no finalized link backs it, so it produces no entry.
    view = build_project_view(
        [_record("a.py", "python", catalog_updates=[
            {"name": "BaseAgent", "type": "internal", "used_for": "base class"},
        ])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_unresolved_internal_function_hint_is_discarded():
    view = build_project_view(
        [_record("a.py", "python", catalog_updates=[
            {"name": "get_logger", "type": "internal", "used_for": "logging"},
        ])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_unresolved_internal_hint_is_not_reclassified_external():
    # The discarded internal hint must NOT reappear as a fabricated external.
    view = build_project_view(
        [_record("a.py", "python", catalog_updates=[
            {"name": "BaseAgent", "type": "internal", "used_for": "base class"},
        ])],
        {"checked": 1},
    )
    assert ("external", "BaseAgent") not in _catalog(view)


# ---------------------------------------------------------------------------
# Exact resolved internal path is admitted (A1)
# ---------------------------------------------------------------------------

def test_exact_resolved_internal_path_is_admitted():
    edges = [{"from": "src/main.py", "to": "src/base.py", "type": "internal_import"}]
    view = build_project_view(
        [
            _record("src/main.py", "python", catalog_updates=[
                {"name": "src/base.py", "type": "internal", "used_for": "base agent"}]),
            _record("src/base.py", "python"),
        ],
        {"checked": 2},
        graph_edges=edges,
    )
    entry = _catalog(view).get(("internal", "src/base.py"))
    assert entry is not None
    assert entry["used_for"] == "base agent"
    assert entry["files"] == ["src/main.py"]


def test_exact_internal_path_wins_over_conflicting_external_hint():
    edges = [{"from": "src/main.py", "to": "src/base.py", "type": "internal_import"}]
    view = build_project_view(
        [
            _record("src/main.py", "python", catalog_updates=[
                {"name": "src/base.py", "type": "external", "used_for": "base agent"}]),
            _record("src/base.py", "python"),
        ],
        {"checked": 2},
        graph_edges=edges,
    )
    catalog = _catalog(view)
    assert ("internal", "src/base.py") in catalog
    assert ("external", "src/base.py") not in catalog


def test_internal_entry_requires_used_for():
    # A resolved internal link with no used_for text yields no catalog entry.
    edges = [{"from": "src/main.py", "to": "src/base.py", "type": "internal_import"}]
    view = build_project_view(
        [_record("src/main.py", "python"), _record("src/base.py", "python")],
        {"checked": 2},
        graph_edges=edges,
    )
    assert ("internal", "src/base.py") not in _catalog(view)


# ---------------------------------------------------------------------------
# Real third-party + standard-library dependencies (A2, A4)
# ---------------------------------------------------------------------------

def test_real_third_party_with_link_and_usage_is_kept():
    view = build_project_view(
        [_record("a.py", "python", external=["requests"], usage_notes=[
            {"import": "requests", "used_for": "HTTP calls"}])],
        {"checked": 1},
    )
    entry = _catalog(view).get(("external", "requests"))
    assert entry is not None
    assert entry["used_for"] == "HTTP calls"


def test_sdk_dependency_with_usage_is_kept_as_sdk():
    view = build_project_view(
        [_record("a.py", "python", external=["os"], usage_notes=[
            {"import": "os", "used_for": "path handling"}])],
        {"checked": 1},
    )
    catalog = _catalog(view)
    assert ("sdk", "os") in catalog
    assert ("external", "os") not in catalog


def test_sdk_dependency_without_usage_is_dropped():
    # A finalized SDK link with no used_for text carries nothing useful (A4).
    view = build_project_view(
        [_record("a.py", "python", external=["os"])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


# ---------------------------------------------------------------------------
# Model-only entries are never created (A2)
# ---------------------------------------------------------------------------

def test_catalog_update_without_finalized_link_is_discarded():
    view = build_project_view(
        [_record("a.py", "python", catalog_updates=[
            {"name": "fastapi", "type": "external", "used_for": "web framework"}])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_dependency_refs_alone_cannot_create_entry():
    view = build_project_view(
        [_record("a.py", "python", dependency_refs=["numpy"])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_usage_notes_alone_cannot_create_entry():
    view = build_project_view(
        [_record("a.py", "python", usage_notes=[
            {"import": "numpy", "used_for": "math"}])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view


def test_deterministic_classification_wins_over_model_type():
    # `os` is standard library; an agent hint claiming it is external must not
    # override the deterministic SDK classification of the finalized link.
    view = build_project_view(
        [_record("a.py", "python", external=["os"], catalog_updates=[
            {"name": "os", "type": "external", "used_for": "filesystem"}])],
        {"checked": 1},
    )
    catalog = _catalog(view)
    assert ("sdk", "os") in catalog
    assert ("external", "os") not in catalog


# ---------------------------------------------------------------------------
# Project's own package root is not a false external (A3)
# ---------------------------------------------------------------------------

def test_project_root_reported_as_external_is_suppressed():
    # `src` is the project's own package root (every file lives under src/).
    # An agent that lists `src` as an external dependency must not create a
    # bogus third-party catalog entry for the project's own package.
    view = build_project_view(
        [
            _record("src/main.py", "python", external=["src"], usage_notes=[
                {"import": "src", "used_for": "own package"}]),
            _record("src/base.py", "python"),
        ],
        {"checked": 2},
        graph_edges=[
            {"from": "src/main.py", "to": "src/base.py", "type": "internal_import"}
        ],
    )
    assert ("external", "src") not in _catalog(view)


def test_root_level_module_reported_as_external_is_suppressed():
    view = build_project_view(
        [
            _record("main.py", "python", external=["utils"], usage_notes=[
                {"import": "utils", "used_for": "project helpers"}]),
            _record("utils.py", "python"),
        ],
        {"checked": 2},
        graph_edges=[
            {"from": "main.py", "to": "utils.py", "type": "internal_import"}
        ],
    )
    assert ("external", "utils") not in _catalog(view)


def test_src_layout_package_reported_as_external_is_suppressed():
    view = build_project_view(
        [
            _record("src/acme/main.py", "python", external=["acme"], usage_notes=[
                {"import": "acme", "used_for": "own package"}]),
            _record("src/acme/base.py", "python"),
        ],
        {"checked": 2},
        graph_edges=[
            {
                "from": "src/acme/main.py",
                "to": "src/acme/base.py",
                "type": "internal_import",
            }
        ],
    )
    assert ("external", "acme") not in _catalog(view)


def test_src_layout_module_reported_as_external_is_suppressed():
    view = build_project_view(
        [
            _record("src/main.py", "python", external=["acme"], usage_notes=[
                {"import": "acme", "used_for": "own module"}]),
            _record("src/acme.py", "python"),
        ],
        {"checked": 2},
        graph_edges=[
            {"from": "src/main.py", "to": "src/acme.py", "type": "internal_import"}
        ],
    )
    assert ("external", "acme") not in _catalog(view)


def test_project_root_without_internal_resolution_is_not_suppressed():
    view = build_project_view(
        [
            _record("src/main.py", "python", external=["src"], usage_notes=[
                {"import": "src", "used_for": "third-party package"}]),
            _record("src/base.py", "python"),
        ],
        {"checked": 2},
    )
    assert ("external", "src") in _catalog(view)


def test_non_python_dependency_matching_project_root_is_not_suppressed():
    view = build_project_view(
        [
            _record("src/main.rb", "ruby", external=["src"], usage_notes=[
                {"import": "src", "used_for": "Ruby package"}]),
            _record("src/base.rb", "ruby"),
        ],
        {"checked": 2},
        graph_edges=[
            {"from": "src/main.rb", "to": "src/base.rb", "type": "internal_import"}
        ],
    )
    assert ("external", "src") in _catalog(view)


def test_real_external_not_suppressed_alongside_project_root():
    # Suppressing the project root must not suppress a genuine third-party dep.
    view = build_project_view(
        [
            _record("src/main.py", "python", external=["src", "requests"],
                    usage_notes=[
                        {"import": "src", "used_for": "own package"},
                        {"import": "requests", "used_for": "HTTP"}]),
            _record("src/base.py", "python"),
        ],
        {"checked": 2},
        graph_edges=[
            {"from": "src/main.py", "to": "src/base.py", "type": "internal_import"}
        ],
    )
    catalog = _catalog(view)
    assert ("external", "src") not in catalog
    assert ("external", "requests") in catalog
