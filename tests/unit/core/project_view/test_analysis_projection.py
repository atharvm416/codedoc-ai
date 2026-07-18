"""Tests organized by feature ownership."""

from __future__ import annotations

from codedoc.core.project_view import build_project_view

def _py_record(path, imports, model_external=None, *, hash_="h"):
    deps = {}
    if model_external is not None:
        deps["external"] = model_external
    return {
        "hash": hash_,
        "file_path": path,
        "language": "python",
        "documentation": {
            "file_path": path,
            "language": "python",
            "description": "d",
            "imports": imports,
            "dependencies_analysis": deps,
        },
    }

def _links(view, path):
    return {f["path"]: f.get("links", {}) for f in view["files"]}[path]

def test_python_links_identical_regardless_of_model_dependencies():
    # single and triple emit different model dependency text, but for Python the
    # public links project from the parser imports only — so they are identical.
    single = build_project_view(
        [_py_record("a.py", ["os", "requests"], model_external=["os", "requests"])],
        {"checked": 1},
    )
    triple = build_project_view(
        [_py_record("a.py", ["os", "requests"], model_external=["totally", "different"])],
        {"checked": 1},
    )
    assert _links(single, "a.py") == _links(triple, "a.py")
    assert _links(single, "a.py")["external_dependencies"] == ["requests"]
    assert _links(single, "a.py")["sdk_dependencies"] == ["os"]

def test_python_model_only_name_never_appears_in_links():
    record = _py_record("a.py", ["os"], model_external=["requests", "evil"])
    links = _links(build_project_view([record], {"checked": 1}), "a.py")
    assert links.get("external_dependencies", []) == []
    assert links["sdk_dependencies"] == ["os"]

def test_project_import_resolved_by_graph_edge_is_not_external():
    edges = [{"from": "pkg/a.py", "to": "pkg/b.py", "type": "internal_import"}]
    view = build_project_view(
        [
            _py_record("pkg/a.py", ["pkg.b", "os"]),
            _py_record("pkg/b.py", []),
        ],
        {"checked": 2},
        graph_edges=edges,
    )
    links = _links(view, "pkg/a.py")
    assert links["internal_dependencies"] == ["pkg/b.py"]
    assert links["sdk_dependencies"] == ["os"]
    assert "pkg" not in links.get("external_dependencies", [])

def test_stdlib_imports_keep_sdk_classification():
    links = _links(
        build_project_view([_py_record("a.py", ["os", "sys", "json"])], {"checked": 1}),
        "a.py",
    )
    assert links["sdk_dependencies"] == ["json", "os", "sys"]
    assert "external_dependencies" not in links

def test_catalog_for_unmatched_dependency_name_is_discarded():
    # A usage note for a name that is not in the parser imports cannot create a
    # catalog entry.
    record = _py_record("a.py", ["os"])
    record["documentation"]["dependencies_analysis"]["usage_notes"] = [
        {"import": "requests", "used_for": "http"}
    ]
    view = build_project_view([record], {"checked": 1})
    assert "dependency_catalog" not in view

def test_non_python_uses_model_external_when_parser_omits_packages():
    # The JS/TS parser omits bare npm specifiers; the deterministic projection
    # falls back to the model's reported external for non-Python languages.
    record = {
        "hash": "h",
        "file_path": "app.tsx",
        "language": "tsx",
        "documentation": {
            "file_path": "app.tsx",
            "language": "tsx",
            "imports": ["./helper"],
            "dependencies_analysis": {"external": ["react"]},
        },
    }
    links = _links(build_project_view([record], {"checked": 1}), "app.tsx")
    assert links["external_dependencies"] == ["react"]
