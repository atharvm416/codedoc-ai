"""Tests organized by feature ownership."""

from __future__ import annotations

import threading
import pytest
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.pipeline_usage import FakeProvider
from tests.support.pipeline_usage import write_py
from tests.support.pipeline_usage import make_graph

pytestmark = pytest.mark.future_split_execution

def patch_provider(monkeypatch) -> None:
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: FakeProvider())


def test_split_usage_reconciles_chunk_and_synthesis_calls(tmp_path, monkeypatch):
    """Split is only valid in analysis_mode 'single' (D2); the hierarchical
    reduction tree means total planned calls is leaf + reduction + synthesis,
    never a flat chunks*multiplier + 1."""
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    provider = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "parallel_agents": False,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
        },
    )

    expected = (
        stats["unit_documentation_calls_planned"]
        + stats["file_reduction_calls_planned"]
        + stats["synthesis_calls_planned"]
    )
    assert stats["unit_documentation_calls_planned"] == stats["split_chunks"]
    assert stats["file_reduction_calls_planned"] > 0
    assert stats["synthesis_calls_planned"] == 1
    assert stats["total_calls_planned"] == expected
    assert stats["attempted_logical_calls"] == expected
    assert stats["attempted_calls"] == expected
    assert stats["successful_calls"] == expected
    assert stats["failed_calls"] == stats["additional_attempts"] == 0
    assert stats["planned_calls_not_attempted"] == 0


def test_split_execution_never_calls_initial_calls_per_file(tmp_path, monkeypatch):
    """workstream 6/D9: split execution never calls
    `initial_calls_per_file(...)` — dry-run, a real run, an empty-project
    dry-run under a split-configured strategy, and a split-configured run
    whose files include an under-threshold ordinary file (the exact
    "split-path use" the plan singles out). Ordinary `truncate` accounting
    is unaffected and still uses the helper (proven by the pre-existing,
    unmodified single/triple accounting tests in this module continuing to
    pass unchanged)."""
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    def _raise(*_args, **_kwargs):
        raise AssertionError("split execution must never call initial_calls_per_file")

    monkeypatch.setattr("codedoc.pipeline.initial_calls_per_file", _raise)

    # 1. Empty project, split-configured, dry-run.
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_stats = run_pipeline(
        empty_dir,
        {
            "dry_run": True,
            "entry_file": None,
            "auto_entry_candidates": [],
            "large_file_strategy": "split",
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )
    assert empty_stats["scanned"] == 0
    assert empty_stats["initial_calls_per_file"] == 0

    # 2. Split dry-run with an oversized (split-only) file.
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    split_only_dir = tmp_path / "split_only"
    split_only_dir.mkdir()
    (split_only_dir / "main.py").write_text(source, encoding="utf-8", newline="")
    dry_stats = run_pipeline(
        split_only_dir,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )
    assert dry_stats["split_chunks"] > 0
    assert dry_stats["initial_calls_per_file"] == 0

    # 3. Split real-run with the same oversized file.
    provider = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    real_stats = run_pipeline(
        split_only_dir,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )
    assert real_stats["checked"] == 1
    assert real_stats["initial_calls_per_file"] == 0

    # 4. Split-configured run with BOTH an oversized split file and an
    # under-threshold ordinary file. The ordinary file's per-file count is
    # still reported (derived from the manifest), never from
    # initial_calls_per_file(...).
    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    (mixed_dir / "main.py").write_text(source, encoding="utf-8", newline="")
    (mixed_dir / "small.py").write_text("x = 1\n", encoding="utf-8")
    mixed_stats = run_pipeline(
        mixed_dir,
        {
            "dry_run": True,
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )
    assert mixed_stats["split_ordinary_files"] == 1
    assert mixed_stats["split_divided_files"] == 1
    assert mixed_stats["initial_calls_per_file"] == 1


def test_ordinary_truncate_accounting_still_uses_initial_calls_per_file(tmp_path):
    """Preserve the ordinary `truncate` path: `initial_calls_per_file(...)`
    remains the source of `initial_calls_per_file` for both single and triple
    mode when no split is configured."""
    from codedoc.pipeline import run_pipeline

    for mode, expected in (("single", 1), ("triple", 3)):
        project = tmp_path / mode
        project.mkdir()
        (project / "main.py").write_text("x = 1\n", encoding="utf-8")
        stats = run_pipeline(
            project,
            {
                "dry_run": True,
                "entry_file": "main.py",
                "analysis_mode": mode,
                "propagate_changes": False,
                "output_dir": "docs",
            },
        )
        assert stats["initial_calls_per_file"] == expected


def test_max_files_counts_only_agent_files(tmp_path):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.planning import build_pipeline_plan

    for rel, content in (("cached.py", "x = 1\n"), ("new.py", "y = 2\n")):
        write_py(tmp_path / rel, content)
    file_map = {
        rel: {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
        for rel in ("cached.py", "new.py")
    }
    graph = make_graph(*file_map)
    existing = {
        "old-name.py": {
            "path": "old-name.py",
            "hash": compute_file_hash(tmp_path / "cached.py"),
            "language": "python",
            "_analysis_revision": ANALYSIS_REVISION,
            "_analysis_mode": "single",
        }
    }
    plan, _ = build_pipeline_plan(
        file_map,
        graph,
        set(file_map),
        None,
        existing,
        [],
        {"propagate_changes": False, "max_files": 1},
    )
    assert len(plan.process_rels) == 2
    assert len(plan.identical_reuse_rels) == 1
    assert len(plan.agent_rels) == 1
    assert plan.max_files_exceeded is False

def test_usage_counts_success_failure_parse_error_and_is_thread_safe():
    from codedoc.agents.base_agent import BaseAgent
    from codedoc.core.usage import UsageAccumulator, estimate_tokens

    class Agent(BaseAgent):
        agent_name = "Agent"

        def run(self, file_path, content, imports, language):
            return {}

    class Provider:
        def __init__(self):
            self.responses = iter(["not-json", RuntimeError("boom"), "{}"])

        def complete_json(self, prompt, system=""):
            result = next(self.responses)
            if isinstance(result, Exception):
                raise result
            return result

    usage = UsageAccumulator()
    agent = Agent(Provider(), usage=usage)
    raw = agent._call_llm("user", system="system")
    with pytest.raises(Exception):
        agent._parse_json(raw, "x.py")
    with pytest.raises(Exception):
        agent._call_llm("user", system="system")
    assert agent._call_llm("user", system="system") == "{}"

    threads = [threading.Thread(target=usage.record_success, args=("abcd",)) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = usage.snapshot()
    assert snapshot["successful_calls"] == 22
    assert snapshot["failed_calls"] == 1
    assert snapshot["attempted_calls"] == 23
    assert snapshot["estimated_input_tokens"] == 3 * (
        estimate_tokens("system") + estimate_tokens("user")
    )

def test_usage_accounting_failure_does_not_change_provider_success():
    from codedoc.agents.base_agent import BaseAgent

    class Agent(BaseAgent):
        def run(self, file_path, content, imports, language):
            return {}

    class Provider:
        def complete_json(self, prompt, system=""):
            return "{}"

    class BrokenUsage:
        def record_input(self, *texts):
            raise RuntimeError("accounting failed")

        def record_success(self, output):
            raise RuntimeError("accounting failed")

    assert Agent(Provider(), usage=BrokenUsage())._call_llm("prompt") == "{}"

def test_real_run_reports_planned_and_actual_usage(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    write_py(tmp_path / "main.py")
    patch_provider(monkeypatch)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "parallel_agents": False, "propagate_changes": False},
    )

    # 0.10.0: default single mode → one combined call per file.
    assert stats["planned_calls"] == 1
    assert stats["successful_calls"] == 1
    assert stats["failed_calls"] == 0
    assert stats["attempted_calls"] == 1
    assert stats["estimated_input_tokens"] > 0
    assert stats["estimated_output_tokens"] > 0
