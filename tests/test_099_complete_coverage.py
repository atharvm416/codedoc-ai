from __future__ import annotations

import json
import re
from dataclasses import asdict, fields

import pytest

from codedoc.cli.cli import build_parser
from codedoc.core.discovery import _select_files
from codedoc.core.graph import DependencyGraph
from codedoc.core.loader import load_config
from codedoc.core.planning import PipelinePlan, build_pipeline_plan
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError


def _project(tmp_path):
    (tmp_path / "main.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "orphan.py").write_text("y = 2\n", encoding="utf-8")


def _graph_and_map(tmp_path):
    graph = DependencyGraph()
    file_map = {}
    for rel in ("main.py", "helper.py", "orphan.py"):
        graph.add_file(rel)
        file_map[rel] = {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
    graph.add_dependency("main.py", "helper.py")
    return graph, file_map


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


def test_select_files_defaults_to_entry_and_rejects_invalid_direct_value(tmp_path):
    _project(tmp_path)
    graph, file_map = _graph_and_map(tmp_path)
    reachable, documented, _ = _select_files(
        tmp_path, {"entry_file": "main.py"}, graph, file_map
    )
    assert documented == reachable == {"main.py", "helper.py"}

    with pytest.raises(ConfigError, match="documentation_scope"):
        _select_files(
            tmp_path,
            {"entry_file": "main.py", "documentation_scope": "wide"},
            graph,
            file_map,
        )


def test_loader_and_cli_validate_documentation_scope(tmp_path):
    assert load_config(tmp_path, {})["documentation_scope"] == "entry"
    assert load_config(tmp_path, {"documentation_scope": "all"})[
        "documentation_scope"
    ] == "all"
    with pytest.raises(ConfigError, match="documentation_scope"):
        load_config(tmp_path, {"documentation_scope": "wide"})

    parser = build_parser()
    assert parser.parse_args(["--documentation-scope", "all"]).documentation_scope == "all"
    assert parser.parse_args([]).documentation_scope is None


def test_dry_run_scope_stats_and_cost_units(tmp_path):
    _project(tmp_path)

    entry_stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "dry_run": True, "documentation_scope": "entry"},
    )
    all_stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "dry_run": True, "documentation_scope": "all"},
    )

    assert entry_stats["selected"] == 2
    assert entry_stats["entry_excluded"] == 1
    assert entry_stats["disconnected_paid_files"] == 0
    assert all_stats["selected"] == 3
    assert all_stats["entry_reachable"] == 2
    assert all_stats["entry_disconnected"] == 1
    assert all_stats["entry_excluded"] == 0
    assert all_stats["disconnected_paid_files"] == 1
    # 0.10.0: default single mode → one planned initial call per disconnected file.
    assert all_stats["disconnected_planned_calls"] == 1


class _RoutingProvider:
    """Fake provider recording every project-relative path routed to it."""

    def __init__(self, name: str) -> None:
        self.provider_name = name
        self.routed: set[str] = set()
        self.calls = 0

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

    def complete_json(self, prompt, system=""):
        self.calls += 1
        match = re.search(r"^File: (.+)$", prompt, re.MULTILINE)
        if match:
            self.routed.add(match.group(1).strip())
        if "key_concepts" in prompt:
            return json.dumps(
                {
                    "description": "Documented.",
                    "role_in_system": "test",
                    "key_concepts": [],
                    "usage_example": "",
                }
            )
        if "dependencies_analysis" in prompt:
            return json.dumps({"dependencies_analysis": {"internal": [], "external": []}})
        return json.dumps({"functions": [], "classes": [], "exports": []})


def _make_routing_project(tmp_path, name: str):
    project = tmp_path / name
    project.mkdir()
    (project / "main.py").write_text("import helper\n", encoding="utf-8")
    (project / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (project / "orphan.py").write_text("y = 2\n", encoding="utf-8")
    return project


def test_prior_all_output_does_not_leak_full_scope_into_default_run(tmp_path, monkeypatch):
    """A prior `all` run must not make a later default run pay for orphans."""
    project = _make_routing_project(tmp_path, "leak")

    first = _RoutingProvider("OpenAI(test)")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: first)
    run_pipeline(
        project,
        {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    output = project / "codedoc" / "codedoc.json"
    documented = {f["path"] for f in json.loads(output.read_text(encoding="utf-8"))["files"]}
    assert documented == {"main.py", "helper.py", "orphan.py"}

    # Later run with NO scope override returns to the conservative entry default.
    second = _RoutingProvider("OpenAI(test)")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: second)
    stats = run_pipeline(
        project,
        {
            "entry_file": "main.py",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )

    assert stats["documentation_scope"] == "entry"
    # The disconnected file is never routed to the provider in the default run.
    assert "orphan.py" not in second.routed
    # Stale disconnected records are dropped from the final public output.
    documented = {f["path"] for f in json.loads(output.read_text(encoding="utf-8"))["files"]}
    assert documented == {"main.py", "helper.py"}


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


def test_documentation_scope_precedence_cli_over_config_file(tmp_path):
    """An absent CLI flag preserves the config-file value; a flag overrides it."""
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"documentation_scope": "all"}), encoding="utf-8"
    )
    assert load_config(tmp_path, None)["documentation_scope"] == "all"
    assert load_config(tmp_path, {})["documentation_scope"] == "all"
    assert (
        load_config(tmp_path, {"documentation_scope": "entry"})["documentation_scope"]
        == "entry"
    )


def test_switching_entry_to_all_reuses_cache_and_pays_only_disconnected(
    tmp_path, monkeypatch
):
    project = _make_routing_project(tmp_path, "switch")
    first = _RoutingProvider("p")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: first)
    run_pipeline(
        project,
        {
            "entry_file": "main.py",
            "documentation_scope": "entry",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    assert first.routed == {"main.py", "helper.py"}

    second = _RoutingProvider("p")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: second)
    stats = run_pipeline(
        project,
        {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    # Reachable files are reused from cache; only the disconnected file is paid for.
    assert second.routed == {"orphan.py"}
    assert stats["disconnected_paid_files"] == 1
    output = project / "codedoc" / "codedoc.json"
    documented = {f["path"] for f in json.loads(output.read_text(encoding="utf-8"))["files"]}
    assert documented == {"main.py", "helper.py", "orphan.py"}


def test_force_disconnected_file_warns_and_is_excluded_under_entry(
    tmp_path, monkeypatch, caplog
):
    import logging

    project = _make_routing_project(tmp_path, "force-entry")
    provider = _RoutingProvider("p")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    with caplog.at_level(logging.WARNING):
        run_pipeline(
            project,
            {
                "entry_file": "main.py",
                "documentation_scope": "entry",
                "force_files": ["orphan.py"],
                "parallel_agents": False,
                "max_parallel_files": 1,
                "propagate_changes": False,
            },
        )
    warnings = " ".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    assert "force_files" in warnings and "orphan.py" in warnings
    assert "orphan.py" not in provider.routed
    output = project / "codedoc" / "codedoc.json"
    documented = {f["path"] for f in json.loads(output.read_text(encoding="utf-8"))["files"]}
    assert "orphan.py" not in documented


def test_force_disconnected_file_is_documented_under_all(tmp_path, monkeypatch):
    project = _make_routing_project(tmp_path, "force-all")
    provider = _RoutingProvider("p")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    run_pipeline(
        project,
        {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "force_files": ["orphan.py"],
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    assert "orphan.py" in provider.routed
    output = project / "codedoc" / "codedoc.json"
    documented = {f["path"] for f in json.loads(output.read_text(encoding="utf-8"))["files"]}
    assert "orphan.py" in documented


def test_pipeline_plan_canonical_documented_field_with_selected_alias(tmp_path):
    # 0.10.0: documented_rels is now the canonical dataclass field; selected_rels
    # is retained as a read-only delegating compatibility property.
    _project(tmp_path)
    graph, file_map = _graph_and_map(tmp_path)
    plan, _ = build_pipeline_plan(
        file_map=file_map,
        graph=graph,
        selected_rels={"main.py", "helper.py"},
        entry_rel="main.py",
        existing_docs={},
        forced_paths=[],
        config={"propagate_changes": True, "max_files": 0},
    )
    # Canonical field direction.
    assert "documented_rels" in {field.name for field in fields(plan)}
    assert "selected_rels" not in {field.name for field in fields(plan)}
    assert asdict(plan)["documented_rels"] == plan.documented_rels
    match plan:
        case PipelinePlan(documented_rels=documented):
            assert documented == plan.documented_rels
    # Retained compatibility alias.
    assert plan.selected_rels == plan.documented_rels


def test_no_supported_files_real_stats_keep_scope_and_compatibility_keys(tmp_path):
    stats = run_pipeline(tmp_path, {})
    assert stats["entry_excluded"] == 0
    assert stats["documentation_scope"] == "entry"
    assert stats["entry_reachable"] == 0
    assert stats["entry_disconnected"] == 0
    assert stats["disconnected_paid_files"] == 0
    assert stats["disconnected_planned_calls"] == 0
