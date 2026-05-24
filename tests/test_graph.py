"""Tests for DependencyGraph."""

import pytest
from codedoc.core.graph import DependencyGraph, resolve_import
from pathlib import Path


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


class TestResolveImport:
    def test_resolves_relative_ts(self):
        all_files = {"src/App.tsx", "src/router.tsx"}
        result = resolve_import("./router", "src/App.tsx", all_files, Path("."))
        assert result == "src/router.tsx"

    def test_returns_none_for_unknown(self):
        all_files = {"src/App.tsx"}
        result = resolve_import("./missing", "src/App.tsx", all_files, Path("."))
        assert result is None

    def test_does_not_resolve_external_imports_by_basename(self):
        all_files = {"web/index.html", "lib/main.dart"}

        assert resolve_import("dart:io", "lib/main.dart", all_files, Path(".")) is None
        assert (
            resolve_import("androidx.annotation.Keep", "android/Main.java", all_files, Path("."))
            is None
        )
        assert (
            resolve_import("flutter_bootstrap.js", "web/index.html", all_files, Path("."))
            is None
        )

    def test_resolves_file_import_with_extension(self):
        all_files = {
            "lib/core/widgets/mini_player.dart",
            "lib/core/widgets/full_player_screen.dart",
        }
        result = resolve_import(
            "full_player_screen.dart",
            "lib/core/widgets/mini_player.dart",
            all_files,
            Path("."),
        )
        assert result == "lib/core/widgets/full_player_screen.dart"
