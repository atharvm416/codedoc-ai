"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from codedoc.core.discovery import _select_files
from codedoc.core.graph import DependencyGraph
from codedoc.core.loader import load_config
from tests.support.selection_projects import _project
from tests.support.selection_projects import _graph_and_map
from codedoc.core.output import write_project_outputs
from tests.support.reachability_cases import _record

def test_select_files_warns_and_excludes_unreachable_files(tmp_path, caplog):
    """A1 visibility: files not reachable from --entry are excluded, and the
    exclusion is logged loudly (never silent).  Pure, no LLM."""
    import logging

    from codedoc.core.graph import DependencyGraph
    from codedoc.pipeline import _select_files

    # entry.py imports helper.py; orphan.py is imported by nobody.
    (tmp_path / "entry.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "orphan.py").write_text("y = 2\n", encoding="utf-8")

    graph = DependencyGraph()
    for rel in ("entry.py", "helper.py", "orphan.py"):
        graph.add_file(rel)
    graph.add_dependency("entry.py", "helper.py")

    file_map = {
        rel: {"path": tmp_path / rel, "rel_path": rel, "language": "python", "extension": ".py"}
        for rel in ("entry.py", "helper.py", "orphan.py")
    }

    config = load_config(tmp_path, {"entry_file": "entry.py"})

    with caplog.at_level(logging.WARNING):
        reachable, selected, entry_rel = _select_files(tmp_path, config, graph, file_map)

    assert entry_rel == "entry.py"
    assert selected == {"entry.py", "helper.py"}
    assert reachable == selected
    assert "orphan.py" not in selected
    # The omission must be visible.
    warning_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "orphan.py" in warning_text
    assert "disconnected" in warning_text.lower()

def test_select_files_no_exclusion_when_all_reachable(tmp_path, caplog):
    """No spurious warning when every scanned file is reachable from the entry."""
    import logging

    from codedoc.core.graph import DependencyGraph
    from codedoc.pipeline import _select_files

    (tmp_path / "entry.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")

    graph = DependencyGraph()
    for rel in ("entry.py", "helper.py"):
        graph.add_file(rel)
    graph.add_dependency("entry.py", "helper.py")

    file_map = {
        rel: {"path": tmp_path / rel, "rel_path": rel, "language": "python", "extension": ".py"}
        for rel in ("entry.py", "helper.py")
    }

    config = load_config(tmp_path, {"entry_file": "entry.py"})

    with caplog.at_level(logging.WARNING):
        reachable, selected, entry_rel = _select_files(tmp_path, config, graph, file_map)

    assert selected == {"entry.py", "helper.py"}
    assert reachable == selected
    warning_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "not reachable" not in warning_text.lower()

def test_select_files_keeps_reachability_separate_from_documentation_scope(tmp_path):
    _project(tmp_path)
    graph, file_map = _graph_and_map(tmp_path)

    reachable, documented, entry = _select_files(
        tmp_path,
        {"entry_file": "main.py", "documentation_scope": "all"},
        graph,
        file_map,
    )

    assert entry == "main.py"
    assert reachable == {"main.py", "helper.py"}
    assert documented == set(file_map)

def test_no_entry_documents_all_files_as_reachable(tmp_path):
    """Auto-detection failure documents every scanned file; both sets are full."""
    graph = DependencyGraph()
    file_map = {}
    for rel in ("alpha.py", "beta.py"):
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
        graph.add_file(rel)
        file_map[rel] = {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
    reachable, documented, entry = _select_files(tmp_path, {}, graph, file_map)
    assert entry is None
    assert reachable == documented == set(file_map)

def test_write_project_outputs_propagates_reachable_rels(tmp_path):
    json_path, _ = write_project_outputs(
        [_record("main.py"), _record("orphan.py")],
        {"checked": 2},
        tmp_path,
        output_format="json",
        entry_file="main.py",
        reachable_rels={"main.py"},
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_path = {file["path"]: file for file in data["files"]}
    assert by_path["main.py"]["reachable_from_entry"] is True
    assert by_path["orphan.py"]["reachable_from_entry"] is False

def test_write_project_outputs_defaults_to_reachable_for_direct_callers(tmp_path):
    json_path, _ = write_project_outputs(
        [_record("main.py")],
        {"checked": 1},
        tmp_path,
        output_format="json",
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["files"][0]["reachable_from_entry"] is True

def test_write_project_outputs_builds_project_view_once(tmp_path, monkeypatch):
    import codedoc.core.output as output_module

    calls = []
    real_builder = output_module.build_project_view

    def counting_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(output_module, "build_project_view", counting_builder)
    write_project_outputs(
        [_record("main.py")],
        {"checked": 1},
        tmp_path,
        output_format="both",
        reachable_rels={"main.py"},
    )
    assert len(calls) == 1
