"""Tests for DependencyGraph."""

from codedoc.core.graph import DependencyGraph, resolve_import, _KNOWN_EXTENSIONS, _candidate_variants
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

    # --- G4: Dotted-module hotfix regression tests ---

    def test_resolves_dotted_python_module(self):
        """Dotted Python module resolves via candidate path, not as an extension."""
        all_files = {"codedoc/__main__.py", "codedoc/cli/cli.py"}
        result = resolve_import(
            "codedoc.cli.cli",
            "codedoc/__main__.py",
            all_files,
            Path("."),
        )
        assert result == "codedoc/cli/cli.py"

    def test_resolves_multi_level_dotted_module(self):
        """Multi-level dotted module resolves correctly. Use .models suffix — not a known extension."""
        all_files = {"src/main.py", "src/a/b/models.py"}
        result = resolve_import("a.b.models", "src/main.py", all_files, Path("."))
        assert result == "src/a/b/models.py"

    def test_real_extension_still_resolves(self):
        """Known extension imports still resolve correctly after the guard."""
        all_files = {"src/app.ts", "src/utils.ts"}
        result = resolve_import("utils.ts", "src/app.ts", all_files, Path("."))
        assert result == "src/utils.ts"

    def test_fake_suffix_not_treated_as_extension(self):
        """.name is not a known extension, so the dotted branch fires."""
        all_files = {"src/main.py", "src/my/module/name.py"}
        result = resolve_import("my.module.name", "src/main.py", all_files, Path("."))
        assert result == "src/my/module/name.py"

    def test_dart_package_import_unaffected(self):
        """package: imports use lib/<path-after-first-slash> resolution, unchanged by G5."""
        # package:myapp/screens/home.dart -> candidate lib/screens/home.dart
        all_files = {"lib/main.dart", "lib/screens/home.dart"}
        result = resolve_import(
            "package:myapp/screens/home.dart",
            "lib/main.dart",
            all_files,
            Path("."),
        )
        assert result == "lib/screens/home.dart"

    def test_case_mismatch_is_rejected_on_every_platform(self):
        """0.9.6: resolution is exact-case only and never probes the filesystem,
        so a case-mismatched import is unresolved regardless of host OS."""
        all_files = {"main.py", "foo.py"}
        assert resolve_import("FOO", "main.py", all_files, Path(".")) is None

    def test_exact_case_still_resolves(self):
        all_files = {"main.py", "foo.py"}
        assert resolve_import("foo", "main.py", all_files, Path(".")) == "foo.py"

    def test_resolution_ignores_the_root_argument(self, tmp_path):
        """``root`` is compatibility-only; the same inputs resolve identically
        whatever path is passed."""
        all_files = {"main.py", "foo.py"}
        assert resolve_import("foo", "main.py", all_files, Path(".")) == "foo.py"
        assert resolve_import("foo", "main.py", all_files, tmp_path) == "foo.py"
        assert resolve_import("FOO", "main.py", all_files, tmp_path) is None


class TestCandidateVariants:
    """G5: Extension list consistency tests."""

    def test_go_extension_gets_candidate_variants(self):
        """utils.go must appear in candidates — previously missing because .go was not in the list."""
        variants = _candidate_variants("utils")
        assert "utils.go" in variants

    def test_kotlin_extension_gets_candidate_variants(self):
        variants = _candidate_variants("models")
        assert "models.kt" in variants

    def test_rust_extension_gets_candidate_variants(self):
        variants = _candidate_variants("utils")
        assert "utils.rs" in variants

    def test_swift_extension_gets_candidate_variants(self):
        variants = _candidate_variants("helpers")
        assert "helpers.swift" in variants

    def test_ruby_extension_gets_candidate_variants(self):
        variants = _candidate_variants("app")
        assert "app.rb" in variants

    def test_candidate_variants_uses_known_extensions(self):
        """Guard: variant count must equal len(_KNOWN_EXTENSIONS) + 1 (base) + len(_KNOWN_EXTENSIONS) (index variants) + 1 (__init__.py)."""
        variants = _candidate_variants("foo")
        # base + N extensions + N index/<ext> + __init__.py
        expected_count = 1 + len(_KNOWN_EXTENSIONS) + len(_KNOWN_EXTENSIONS) + 1
        assert len(variants) == expected_count
