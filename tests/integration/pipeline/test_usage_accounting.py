"""Tests organized by feature ownership."""

from __future__ import annotations

import threading
import pytest
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.pipeline_usage import FakeProvider
from tests.support.pipeline_usage import write_py
from tests.support.pipeline_usage import make_graph

def patch_provider(monkeypatch) -> None:
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: FakeProvider())

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
