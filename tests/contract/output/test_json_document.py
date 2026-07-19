"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.record_metadata_cases import private_key  # noqa: F401, F811

import json
from tests.support.response_correction_cases import RoutingProvider
from codedoc.core.project_view import (
    build_project_view,
    json_from_view,
)
from tests.support.record_metadata_cases import _view_with_secret
from tests.support.dependency_view_cases import _record as dependency_view_record
from tests.support.recovery_cache_cases import _Fake
from codedoc.core.output import write_project_outputs
from tests.support.reachability_cases import _record as reachability_record
from codedoc.core.document import read_codedoc_document
from tests.support.json_document_cases import _view

def test_public_output_contains_tree_folders_and_dependency_graph(tmp_path):
    import json


    output_dir = tmp_path / "docs_output"
    records = [
        {
            "id": "main-hash",
            "hash": "main-hash",
            "file_path": "src/main.tsx",
            "format": "tsx",
            "language": "tsx",
            "last_processed": "2026-05-02T00:00:00+00:00",
            "author": "Should Not Leak",
            "documentation": {
                "file_path": "src/main.tsx",
                "language": "tsx",
                "imports": ["react", "./router"],
                "description": "Starts the frontend app.",
                "role_in_system": "Application entry.",
                "functions": [],
                "classes": [],
                "exports": ["App"],
                "dependencies_analysis": {
                    "external": ["react"],
                    "dependency_refs": ["react"],
                    "catalog_updates": [
                        {
                            "name": "react",
                            "type": "external",
                            "used_for": "Rendering UI components.",
                        }
                    ],
                    "usage_notes": [
                        {"import": "react", "used_for": "Creates this component tree."}
                    ],
                },
                "key_concepts": ["rendering"],
                "state": "checked",
            },
        },
        {
            "id": "router-hash",
            "hash": "router-hash",
            "file_path": "src/router.tsx",
            "format": "tsx",
            "language": "tsx",
            "last_processed": "2026-05-02T00:00:00+00:00",
            "documentation": {
                "file_path": "src/router.tsx",
                "language": "tsx",
                "imports": ["react", "react-router-dom"],
                "description": "Defines routes.",
                "role_in_system": "Routes application screens.",
                "functions": [],
                "classes": [],
                "exports": ["Router"],
                "dependencies_analysis": {
                    "external": ["react", "react-router-dom"],
                    "dependency_refs": ["react", "react-router-dom"],
                    "catalog_updates": [
                        {
                            "name": "react-router-dom",
                            "type": "external",
                            "used_for": "Routing screens.",
                        }
                    ],
                    "usage_notes": [
                        {"import": "react", "used_for": "Supports route component rendering."},
                        {"import": "react-router-dom", "used_for": "Defines app routes."},
                    ],
                },
                "key_concepts": ["routing"],
                "state": "checked",
            },
        },
    ]

    json_path, md_path = write_project_outputs(
        records,
        {"checked": 2, "failed": 0, "skipped": 0, "reused": 0},
        output_dir,
        output_format="both",
        entry_file="src/main.tsx",
        graph_edges=[
            {
                "from": "src/main.tsx",
                "to": "src/router.tsx",
                "type": "internal_import",
            }
        ],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["last_run"]["entry_file"] == "src/main.tsx"
    assert "project" not in payload
    assert "run" not in payload
    assert "_codedoc" not in payload
    assert payload["tree"]["src"]["main.tsx"]["type"] == "file"
    assert payload["folders"][0]["path"] == "src"
    assert payload["dependency_catalog"][0]["name"] == "react"
    assert payload["dependency_catalog"][0]["file_count"] == 2
    assert payload["dependency_catalog"][0]["used_for"] == "Rendering UI components."
    assert payload["dependency_graph"] == [
        {
            "from": "src/main.tsx",
            "to": "src/router.tsx",
            "type": "internal_import",
        }
    ]
    assert payload["files"][0]["links"]["internal_dependencies"] == ["src/router.tsx"]
    assert payload["files"][1]["links"]["imported_by"] == ["src/main.tsx"]
    assert "author" not in json_path.read_text(encoding="utf-8")
    assert "result" not in json_path.read_text(encoding="utf-8")

    markdown = md_path.read_text(encoding="utf-8")
    assert "## Project Tree" in markdown
    assert "## Dependency Catalog" in markdown
    assert "### react" in markdown
    assert "src/" in markdown
    assert "`src/main.tsx` -> `src/router.tsx`" in markdown

def test_4_files_array_follows_queue_order(tmp_path):
    """Test 4: files in live backup follow set_queue_order, not completion order."""
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    backup = out / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.set_queue_order(["a.py", "b.py", "c.py"])
    sw.initialize_empty()

    # Record out of topological order
    sw.record("c.py", {"language": "python"}, file_hash="HC")
    sw.record("a.py", {"language": "python"}, file_hash="HA")
    sw.record("b.py", {"language": "python"}, file_hash="HB")

    data = json.loads(backup.read_text(encoding="utf-8"))
    paths = [f["path"] for f in data["files"]]
    assert paths == ["a.py", "b.py", "c.py"], f"Expected queue order, got {paths}"

def test_17_recovered_warnings_not_in_final_json(tmp_path):
    """Test 17: warning-level ErrorReporter entries do not appear in final JSON."""
    from codedoc.utils.errors import ErrorReporter

    output_dir = tmp_path / "codedoc"
    output_dir.mkdir()

    reporter = ErrorReporter()
    reporter.record(
        RuntimeError("429 rate_limit_exceeded"),
        context="rate limit step-down",
        level="warning",
    )

    # summary() must return "" for warning-only
    assert reporter.summary() == "", "summary() must be empty for warning-only"
    assert not reporter.has_errors(), "has_errors() must be False for warning-only"
    assert reporter.has_issues(), "has_issues() must be True"
    assert reporter.issue_count() == 1

    # write_project_outputs passes summary() — empty string means no errors field
    records = [{
        "hash": "H1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {"file_path": "main.py", "language": "python",
                          "description": "test", "role_in_system": "r",
                          "functions": [], "classes": [], "exports": [],
                          "dependencies_analysis": {}},
    }]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        error_summary=reporter.summary(),  # "" — empty
        output_format="json",
        entry_file="main.py",
    )

    result = json.loads((output_dir / "codedoc.json").read_text(encoding="utf-8"))
    assert "errors" not in result, (
        f"'errors' field must NOT appear in clean JSON for warning-only runs, got keys: {list(result)}"
    )

def test_17_hard_errors_still_appear_in_final_json(tmp_path):
    """Test 17b: error-level entries DO appear in final JSON."""
    from codedoc.utils.errors import ErrorReporter

    output_dir = tmp_path / "codedoc"
    output_dir.mkdir()

    reporter = ErrorReporter()
    reporter.record(RuntimeError("parse failed"), context="test", level="error")

    assert reporter.has_errors()
    assert reporter.summary() != ""

    records = [{
        "hash": "H1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {"file_path": "main.py", "language": "python",
                          "description": "test", "role_in_system": "r",
                          "functions": [], "classes": [], "exports": [],
                          "dependencies_analysis": {}},
    }]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 1, "skipped": 0},
        output_dir,
        error_summary=reporter.summary(),
        output_format="json",
        entry_file="main.py",
    )

    result = json.loads((output_dir / "codedoc.json").read_text(encoding="utf-8"))
    assert "errors" in result, "Hard errors must appear in final JSON"

def test_no_marker_or_raw_response_in_completed_output(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    pipeline.run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "response_correction_enabled": True,
         "parallel_agents": False, "max_parallel_files": 1, "propagate_changes": False},
    )
    text = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    assert "response_contract_final" not in text
    assert "response_contract_diagnostic" not in text

def test_completed_json_preserves_private_key(private_key):
    view = _view_with_secret(private_key)
    payload = json.loads(json_from_view(view))
    assert payload["files"][0]["_secret"] == "TOPSECRET"

def test_json_preserves_sdk_dependencies():
    view = build_project_view(
        [dependency_view_record("m.py", "python", external=["os", "requests"])],
        {"checked": 1},
    )
    payload = json.loads(json_from_view(view))
    assert payload["files"][0]["links"]["sdk_dependencies"] == ["os"]

def test_correction_stats_absent_from_completed_output(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: _Fake())
    stats = pipeline.run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "output_format": "both",
         "response_correction_enabled": True, "propagate_changes": False},
    )
    # Run stats carry the internal correction counters (like documentation_calls_*).
    assert "response_correction_calls_attempted" in stats
    assert "documentation_calls_attempted" in stats
    # Neither the counters nor any correction internal leak into completed output.
    for name in ("codedoc.json", "codedoc.md"):
        text = (tmp_path / "codedoc" / name).read_text(encoding="utf-8")
        assert "response_correction_calls_attempted" not in text
        assert "response_contract_final" not in text
        assert "documentation_calls_attempted" not in text

def test_reachability_is_present_for_true_and_false_records():
    view = build_project_view(
        [reachability_record("main.py"), reachability_record("orphan.py")],
        {"checked": 2},
        entry_file="main.py",
        reachable_rels={"main.py"},
    )
    by_path = {file["path"]: file for file in view["files"]}
    assert by_path["main.py"]["reachable_from_entry"] is True
    assert by_path["orphan.py"]["reachable_from_entry"] is False
    assert '"reachable_from_entry": false' in json_from_view(view)

def test_completed_json_omits_removed_top_level_blocks(tmp_path):
    path = tmp_path / "codedoc.json"
    payload = json.loads(json_from_view(_view()))
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert "_codedoc" not in payload
    assert "project" not in payload
    assert "run" not in payload
    assert payload["last_run"]["entry_file"] == "main.py"

    doc = read_codedoc_document(path)
    assert doc.entry_file == "main.py"
    assert doc.metadata == {"entry_file": "main.py"}
    assert len(doc.files) == 2
