"""Import-resolution precision tests (0.9.6).

Resolution is now purely lexical and exact-case: no bare final-segment
candidate and no filesystem-dependent case folding.  The same inputs must
resolve identically on every operating system.
"""

from __future__ import annotations

from pathlib import Path

from codedoc.core.graph import DependencyGraph, resolve_import


def test_stdlib_dotted_import_does_not_resolve_to_unrelated_root_file():
    # `abc.py` exists at the root, but `collections.abc` must NOT resolve to it
    # via its bare last segment.
    all_files = {"main.py", "abc.py"}
    assert resolve_import("collections.abc", "main.py", all_files, Path(".")) is None


def test_generic_dotted_import_does_not_resolve_via_last_segment():
    all_files = {"main.java", "Bar.java"}
    assert resolve_import("com.example.Bar", "main.java", all_files, Path(".")) is None


def test_real_relative_imports_still_resolve():
    all_files = {"src/app.ts", "src/util.ts", "core/x.ts"}
    assert resolve_import("./util", "src/app.ts", all_files, Path(".")) == "src/util.ts"
    assert resolve_import("../core/x", "src/app.ts", all_files, Path(".")) == "core/x.ts"


def test_python_dotted_relative_still_resolves():
    all_files = {"pkg/main.py", "pkg/utils.py"}
    assert resolve_import(".utils", "pkg/main.py", all_files, Path(".")) == "pkg/utils.py"


def test_dart_package_import_still_resolves():
    all_files = {"lib/main.dart", "lib/screens/home.dart"}
    result = resolve_import(
        "package:myapp/screens/home.dart", "lib/main.dart", all_files, Path(".")
    )
    assert result == "lib/screens/home.dart"


def test_directory_anchored_dotted_import_still_resolves():
    # A genuine project module reachable through its directory-anchored form.
    all_files = {"codedoc/__main__.py", "codedoc/cli/cli.py"}
    result = resolve_import(
        "codedoc.cli.cli", "codedoc/__main__.py", all_files, Path(".")
    )
    assert result == "codedoc/cli/cli.py"


def test_exact_case_resolves_case_mismatch_unresolved_every_os():
    all_files = {"main.py", "logger.py"}
    assert resolve_import("logger", "main.py", all_files, Path(".")) == "logger.py"
    assert resolve_import("Logger", "main.py", all_files, Path(".")) is None


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
