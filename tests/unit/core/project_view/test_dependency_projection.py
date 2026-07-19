"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
import json
import re
from codedoc.core.project_view import (
    build_project_view,
)
from tests.support.dependency_view_cases import _record

def test_public_output_normalizes_external_package_names(tmp_path):
    import json

    from codedoc.core.output import write_project_outputs

    json_path, _ = write_project_outputs(
        [
            {
                "id": "main-hash",
                "hash": "main-hash",
                "file_path": "lib/main.dart",
                "format": "dart",
                "language": "dart",
                "last_processed": "2026-05-13T00:00:00+00:00",
                "documentation": {
                    "file_path": "lib/main.dart",
                    "language": "dart",
                    "description": "Starts the app.",
                    # 0.10.1: public links project from the parser imports.
                    "imports": [
                        "flutter/material.dart",
                        "provider/provider.dart",
                        "dart:async",
                    ],
                    "dependencies_analysis": {
                        "external": [
                            "flutter/material.dart",
                            "provider/provider.dart",
                            "dart:async",
                        ],
                        "dependency_refs": [
                            "flutter/material.dart",
                            "provider/provider.dart",
                        ],
                    },
                    "state": "checked",
                },
            }
        ],
        {"checked": 1, "failed": 0, "skipped": 0},
        tmp_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    # 0.9.3: SDK/standard-library names are separated from third-party packages.
    # `dart:async` is a Dart SDK library, so it moves to `sdk_dependencies`.
    links = payload["files"][0]["links"]
    assert links["external_dependencies"] == ["flutter", "provider"]
    assert links["sdk_dependencies"] == ["dart:async"]

class TestProjectDependencyLinks:
    """Unit tests for _project_dependency_links with unresolved imports."""

    def _call(self, file_dict, internal_paths=None, unresolved_imports=None):
        from codedoc.core.project_view import (
            _project_dependency_links,
            _project_import_roots,
        )
        internal_paths = internal_paths or []
        project_roots = _project_import_roots([file_dict])
        return _project_dependency_links(
            file_dict,
            internal_paths,
            project_roots,
            unresolved_imports=unresolved_imports,
        )

    # --- Dart ---

    def test_C1_dart_sdk_import(self):
        """dart:io classifies as SDK."""
        file = {"path": "lib/main.dart", "language": "dart", "imports": []}
        external, sdk = self._call(file, unresolved_imports=["dart:io", "dart:async"])
        assert "dart:io" in sdk
        assert "dart:async" in sdk
        assert external == []

    def test_C2_dart_package_import(self):
        """package:record/record.dart → external 'record'."""
        file = {"path": "lib/player.dart", "language": "dart", "imports": []}
        external, sdk = self._call(
            file,
            unresolved_imports=["package:record/record.dart", "package:flutter/material.dart"],
        )
        assert "record" in external
        assert "flutter" in external
        assert sdk == []

    def test_C3_dart_relative_import_not_external(self):
        """Relative Dart imports in unresolved list do not create external entries."""
        file = {"path": "lib/player.dart", "language": "dart", "imports": []}
        external, sdk = self._call(
            file,
            unresolved_imports=[
                "../models/song_model.dart",
                "full_player_screen.dart",
                "dart:io",
            ],
        )
        # Only dart:io should appear; relative/bare imports must be filtered out
        assert external == []
        assert sdk == ["dart:io"]

    def test_C3_dart_relative_internal_appears_in_graph_not_external(self):
        """Dart internal imports that resolve don't appear as external."""
        # Simulate: the internal import resolved to a graph edge, so it's NOT
        # in unresolved_imports. Only package/sdk imports are unresolved.
        file = {"path": "lib/player.dart", "language": "dart", "imports": []}
        external, sdk = self._call(
            file,
            internal_paths=["lib/models/song_model.dart"],
            unresolved_imports=["package:record/record.dart", "dart:async"],
        )
        assert "record" in external
        assert "dart:async" in sdk
        # No internal paths leak as external
        assert "song_model.dart" not in external
        assert "models/song_model.dart" not in external

    # --- Java ---

    def test_C4_java_uses_unresolved_not_model(self):
        """Java generic-parser uses unresolved imports, not model _deps.external."""
        file = {
            "path": "src/Main.java",
            "language": "java",
            "imports": [],
            "_deps": {"external": ["wrong_package"]},
        }
        external, sdk = self._call(
            file,
            unresolved_imports=["org.springframework.boot.SpringApplication", "com.google.gson.Gson"],
        )
        # Should use unresolved imports, not model _deps.external
        assert "wrong_package" not in external
        # The generic classifier returns full canonical for Java
        assert "org.springframework.boot.SpringApplication" in external or len(external) > 0

    def test_C4_java_fallback_when_no_unresolved(self):
        """Java falls back to model _deps.external when unresolved_imports is None."""
        file = {
            "path": "src/Main.java",
            "language": "java",
            "imports": [],
            "_deps": {"external": ["spring-boot"]},
        }
        external, sdk = self._call(file, unresolved_imports=None)
        assert "spring-boot" in external

    # --- Go ---

    def test_C5_go_uses_unresolved_not_model(self):
        """Go generic-parser uses unresolved imports, not model _deps.external."""
        file = {
            "path": "main.go",
            "language": "go",
            "imports": [],
            "_deps": {"external": ["wrong_lib"]},
        }
        external, sdk = self._call(
            file,
            unresolved_imports=["github.com/gin-gonic/gin", "fmt"],
        )
        assert "wrong_lib" not in external
        assert "github.com/gin-gonic/gin" in external

    # --- React/TSX ---

    def test_C6_react_uses_model_deps(self):
        """React/TSX always uses model _deps.external for npm packages."""
        file = {
            "path": "src/App.tsx",
            "language": "tsx",
            "imports": ["./components/Header", "./utils/api"],
            "_deps": {"external": ["react", "@mui/material"]},
        }
        # Even with unresolved_imports supplied, React uses model _deps
        external, sdk = self._call(
            file,
            unresolved_imports=["./components/Header"],
        )
        assert "react" in external
        assert "@mui/material" in external

    def test_C6_jsx_uses_model_deps(self):
        """JSX also uses model _deps.external."""
        file = {
            "path": "src/App.jsx",
            "language": "jsx",
            "imports": [],
            "_deps": {"external": ["react", "axios"]},
        }
        external, sdk = self._call(file, unresolved_imports=["./utils"])
        assert "react" in external
        assert "axios" in external

    # --- Python (unchanged from 0.10.1) ---

    def test_C7_python_uses_unresolved_imports(self):
        """Python uses unresolved imports (graph-filtered), dropping internal ones."""
        file = {
            "path": "codedoc/core/loader.py",
            "language": "python",
            "imports": ["os", "codedoc.core.graph", "pathlib", "codedoc.utils.errors"],
        }
        # Simulate: codedoc.* resolved to graph edges, stdlib/external didn't
        unresolved = ["os", "pathlib"]
        external, sdk = self._call(
            file,
            internal_paths=["codedoc/core/graph.py", "codedoc/utils/errors.py"],
            unresolved_imports=unresolved,
        )
        # os and pathlib are stdlib
        assert "os" in sdk
        assert "pathlib" in sdk
        # codedoc.* should not appear (they were resolved to internal graph edges)
        assert "codedoc" not in external

    def test_C7_python_fallback_without_unresolved(self):
        """Python falls back to file['imports'] when unresolved_imports is None."""
        file = {
            "path": "my_module.py",
            "language": "python",
            "imports": ["requests", "os"],
        }
        external, sdk = self._call(file, unresolved_imports=None)
        assert "requests" in external
        assert "os" in sdk

class TestBuildProjectViewUnresolvedThreading:
    """Integration: unresolved_imports_by_path threads through build_project_view."""

    def test_unresolved_used_for_dart(self):
        """build_project_view passes per-file unresolved imports to projection."""
        from codedoc.core.project_view import build_project_view

        records = [{
            "hash": "abc",
            "file_path": "lib/player.dart",
            "language": "dart",
            "documentation": {
                "file_path": "lib/player.dart",
                "language": "dart",
                "imports": ["package:record/record.dart", "dart:io"],
                "description": "Player",
                "role_in_system": "",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {},
                "key_concepts": [], "usage_example": "",
                "state": "checked",
            },
        }]
        stats = {"checked": 1, "failed": 0, "skipped": 0, "reused": 0}

        # With unresolved_imports: both package:record and dart:io are unresolved
        view = build_project_view(
            records, stats,
            unresolved_imports_by_path={
                "lib/player.dart": ["package:record/record.dart", "dart:io"],
            },
        )
        player = view["files"][0]
        links = player.get("links", {})
        assert "record" in links.get("external_dependencies", [])
        assert "dart:io" in links.get("sdk_dependencies", [])

    def test_no_unresolved_fallback(self):
        """build_project_view without unresolved_imports falls back gracefully."""
        from codedoc.core.project_view import build_project_view

        records = [{
            "hash": "abc",
            "file_path": "src/App.tsx",
            "language": "tsx",
            "documentation": {
                "file_path": "src/App.tsx",
                "language": "tsx",
                "imports": [],
                "description": "App",
                "role_in_system": "",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {"external": ["react"]},
                "key_concepts": [], "usage_example": "",
                "state": "checked",
            },
        }]
        stats = {"checked": 1, "failed": 0, "skipped": 0, "reused": 0}

        # Without unresolved_imports_by_path, React falls back to model _deps
        view = build_project_view(records, stats)
        app = view["files"][0]
        links = app.get("links", {})
        assert "react" in links.get("external_dependencies", [])

_DOWNSTREAM_CASES = [
    {
        "language": "python",
        "entry": "main.py",
        "helper": "helper.py",
        "entry_source": (
            "import requests\nfrom helper import Helper\n\n"
            "def main():\n    return Helper()\n"
        ),
        "helper_source": "class Helper:\n    pass\n",
        "functions": [{"name": "main", "description": "Builds a Helper."}],
        "classes": [],
        "exports": ["main"],
        "external": "requests",
        "catalog_name": "requests",
        "usage": "from main import main",
    },
    {
        "language": "tsx",
        "entry": "main.tsx",
        "helper": "helper.tsx",
        "entry_source": (
            "import React from 'react';\nimport { helper } from './helper';\n"
            "export const App = () => <main>{helper()}</main>;\n"
        ),
        "helper_source": "export const helper = () => 'ready';\n",
        "functions": [{"name": "App", "description": "Renders the application."}],
        "classes": [],
        "exports": ["App"],
        "external": "react",
        "catalog_name": "react",
        "usage": "import { App } from './main'",
    },
    {
        "language": "dart",
        "entry": "lib/main.dart",
        "helper": "lib/helper.dart",
        "entry_source": (
            "import 'package:http/http.dart';\nimport 'helper.dart';\n"
            "void main() { helper(); }\n"
        ),
        "helper_source": "void helper() {}\n",
        "functions": [{"name": "main", "description": "Starts the application."}],
        "classes": [],
        "exports": ["main"],
        "external": "package:http/http.dart",
        "catalog_name": "http",
        "usage": "import 'main.dart';",
    },
    {
        "language": "java",
        "entry": "src/Main.java",
        "helper": "src/Helper.java",
        "entry_source": (
            "import src.Helper;\nimport org.slf4j.Logger;\n"
            "public class Main { public static void main(String[] args) { new Helper(); } }\n"
        ),
        "helper_source": "package src; public class Helper {}\n",
        "functions": [{"name": "main", "description": "Starts the application."}],
        "classes": [{"name": "Main", "description": "Application entry type."}],
        "exports": ["Main"],
        "external": "org.slf4j.Logger",
        "catalog_name": "org.slf4j.Logger",
        "usage": "new Main()",
    },
]

@pytest.mark.parametrize("case", _DOWNSTREAM_CASES, ids=lambda case: case["language"])
def test_combined_response_maps_losslessly_to_public_catalog_and_graph(
    tmp_path, monkeypatch, case
):
    from codedoc.pipeline import run_pipeline

    entry = tmp_path / case["entry"]
    helper = tmp_path / case["helper"]
    entry.parent.mkdir(parents=True, exist_ok=True)
    helper.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(case["entry_source"], encoding="utf-8")
    helper.write_text(case["helper_source"], encoding="utf-8")

    entry_response = {
        "description": f"Documented {case['language']} entry.",
        "role_in_system": "Application entry point.",
        "functions": case["functions"],
        "classes": case["classes"],
        "exports": case["exports"],
        "dependencies_analysis": {
            "external": [case["external"]],
            "dependency_refs": [case["external"]],
            "catalog_updates": [
                {
                    "name": case["external"],
                    "type": "external",
                    "used_for": "Supports the application entry point.",
                }
            ],
            "usage_notes": [
                {
                    "import": case["external"],
                    "used_for": "Used directly by the entry file.",
                }
            ],
        },
        "key_concepts": ["startup"],
        "usage_example": case["usage"],
    }

    class FixtureProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            match = re.search(r"^File: (.+)$", prompt, re.MULTILINE)
            assert match is not None
            response = entry_response if match.group(1) == case["entry"] else {
                "description": "Project helper.",
                "role_in_system": "Supports the entry point.",
            }
            return json.dumps(response)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: FixtureProvider())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": case["entry"],
            "analysis_mode": "single",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 2

    payload = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    files = {item["path"]: item for item in payload["files"]}
    documented = files[case["entry"]]
    assert documented["description"] == entry_response["description"]
    assert documented.get("functions", []) == case["functions"]
    assert documented.get("classes", []) == case["classes"]
    assert documented["exports"] == case["exports"]
    assert documented["key_concepts"] == ["startup"]
    assert documented["usage_example"] == case["usage"]
    assert documented["links"]["internal_dependencies"] == [case["helper"]]
    assert payload["dependency_graph"] == [
        {"from": case["entry"], "to": case["helper"], "type": "internal_import"}
    ]
    catalog = {item["name"]: item for item in payload["dependency_catalog"]}
    assert case["catalog_name"] in catalog
    assert catalog[case["catalog_name"]]["used_for"] == (
        "Supports the application entry point."
    )

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
