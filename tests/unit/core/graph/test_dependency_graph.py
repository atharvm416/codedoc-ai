"""Tests organized by feature ownership."""

from __future__ import annotations

from codedoc.core.graph import DependencyGraph, resolve_import
from pathlib import Path

class TestBuildGraphUnresolved:
    """_build_graph returns unresolved_imports_by_path as expected."""

    def test_unresolved_captured(self, tmp_path):
        """Unresolved imports are separated from graph-resolved ones."""
        from codedoc.core.discovery import _build_graph
        from codedoc.utils.errors import ErrorReporter

        # Write a Python file that imports from within the project and from stdlib
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("import os\nfrom b import something\n", encoding="utf-8")
        b.write_text("import json\n", encoding="utf-8")

        all_files = [
            {"rel_path": "a.py", "path": a, "language": "python", "extension": ".py"},
            {"rel_path": "b.py", "path": b, "language": "python", "extension": ".py"},
        ]
        reporter = ErrorReporter()
        graph, file_map, unresolved = _build_graph(all_files, tmp_path, reporter)

        # b.py should be a graph dependency of a.py
        assert "b.py" in graph.dependencies_of("a.py")
        # "os" does not resolve to a project file → should be in unresolved for a.py
        assert "os" in unresolved.get("a.py", [])
        # "from b import something" resolves → NOT in unresolved for a.py
        # (the resolved import is "b.py", not the string "b")
        assert "b.py" not in unresolved.get("a.py", [])
        # "json" doesn't resolve → should be in unresolved for b.py
        assert "json" in unresolved.get("b.py", [])

class TestDependencyGraph:
    def test_add_and_query(self):
        g = DependencyGraph()
        g.add_file("a.py")
        g.add_file("b.py")
        g.add_dependency("a.py", "b.py")
        assert "b.py" in g.dependencies_of("a.py")
        assert "a.py" in g.dependents_of("b.py")

    def test_topological_order_dependency_first(self):
        g = DependencyGraph()
        g.add_dependency("app.py", "utils.py")
        g.add_dependency("app.py", "models.py")
        order = g.topological_order()
        assert order.index("utils.py") < order.index("app.py")
        assert order.index("models.py") < order.index("app.py")

    def test_all_files(self):
        g = DependencyGraph()
        g.add_dependency("a.py", "b.py")
        g.add_file("c.py")
        assert {"a.py", "b.py", "c.py"} == g.all_files()

    def test_cycle_handled_gracefully(self):
        g = DependencyGraph()
        g.add_dependency("a.py", "b.py")
        g.add_dependency("b.py", "a.py")
        order = g.topological_order()
        assert set(order) == {"a.py", "b.py"}

    def test_empty_graph(self):
        g = DependencyGraph()
        assert g.topological_order() == []

    def test_no_deps(self):
        g = DependencyGraph()
        g.add_file("standalone.py")
        assert g.topological_order() == ["standalone.py"]

def test_unresolved_import_creates_no_internal_edge():
    """A now-unresolved import stays in the parser list but adds no graph edge."""
    all_files = {"main.py", "abc.py"}
    graph = DependencyGraph()
    for rel in all_files:
        graph.add_file(rel)

    imports = ["collections.abc"]  # parser-provided, deterministic
    for imp in imports:
        target = resolve_import(imp, "main.py", all_files, Path("."))
        if target:
            graph.add_dependency("main.py", target)

    # The import is preserved in the per-file list (caller's responsibility),
    # but the graph gained no edge from it.
    assert imports == ["collections.abc"]
    assert graph.dependencies_of("main.py") == set()

def test_graph_reflects_only_real_edges():
    all_files = {"main.py", "helper.py", "abc.py"}
    graph = DependencyGraph()
    for rel in all_files:
        graph.add_file(rel)

    for imp in ("helper", "collections.abc", "Helper"):
        target = resolve_import(imp, "main.py", all_files, Path("."))
        if target:
            graph.add_dependency("main.py", target)

    assert graph.dependencies_of("main.py") == {"helper.py"}
    assert graph.reachable_dependencies("main.py") == {"main.py", "helper.py"}
