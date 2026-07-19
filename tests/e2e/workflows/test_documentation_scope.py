"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import re
from codedoc.pipeline import run_pipeline

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
