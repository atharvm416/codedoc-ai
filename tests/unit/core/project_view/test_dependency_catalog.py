"""Tests organized by feature ownership."""

from __future__ import annotations

from codedoc.core.project_view import build_project_view
from tests.support.project_view_cases import _build_view
from tests.support.dependency_view_cases import _record as dependency_view_record

def _record(path, language, *, external=None, imports=None, catalog_updates=None,
            dependency_refs=None, usage_notes=None):
    # 0.10.1 (Workstream F): finalized external/SDK links are projected from the
    # parser ``imports``.  Each case states the parser-found dependency names via
    # ``external``; by default those names are the parser imports too, so the
    # deterministic input is declared once.  ``dependencies_analysis.external``
    # remains set to exercise the bounded model-enrichment path, which can no
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

def _catalog(view):
    return {(c["type"], c["name"]): c for c in view.get("dependency_catalog", [])}

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

def test_dependency_catalog_preserved_verbatim():
    """The catalog is preserved as-is in 0.9.4 (not corrected), so the
    byte-identical guarantee is meaningful."""
    view = _build_view()
    catalog = view.get("dependency_catalog", [])
    assert [(d["name"], d["type"], d["file_count"]) for d in catalog] == [
        ("requests", "external", 1)
    ]

def test_catalog_groups_by_type_and_canonical_name():
    # Entries are evidence-backed: each file declares the finalized external
    # link, and the catalog_update only supplies used_for text.
    view = build_project_view(
        [
            dependency_view_record("a.py", "python", external=["requests"], catalog_updates=[
                {"name": "requests", "type": "external", "used_for": "http"}]),
            dependency_view_record("b.py", "python", external=["requests"], catalog_updates=[
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
            dependency_view_record("a.py", "python", external=["typing"], catalog_updates=[
                {"name": "typing", "used_for": "hints"}]),
            dependency_view_record("b.rb", "ruby", external=["typing"], catalog_updates=[
                {"name": "typing", "used_for": "thing"}]),
        ],
        {"checked": 2},
    )
    keys = {(c["type"], c["name"]) for c in view["dependency_catalog"]}
    assert ("sdk", "typing") in keys
    assert ("external", "typing") in keys

def test_bare_external_link_without_usage_does_not_fill_catalog():
    view = build_project_view(
        [dependency_view_record("m.py", "python", external=["requests"])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view

def test_catalog_update_without_used_for_does_not_fill_catalog():
    view = build_project_view(
        [dependency_view_record("m.py", "python", catalog_updates=[
            {"name": "requests", "type": "external", "used_for": ""},
        ])],
        {"checked": 1},
    )
    assert "dependency_catalog" not in view
