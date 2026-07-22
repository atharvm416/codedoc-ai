"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import re
import pytest
from codedoc.agents.orchestrator import Orchestrator
from tests.support.execution_requests import make_execution_request
from tests.support.one_call_cases import _COMBINED_JSON
from tests.support.one_call_cases import _CountingProvider

_FLAT_KEYS = {
    "file_path", "language", "extension", "imports", "description",
    "role_in_system", "functions", "classes", "exports", "structure",
    "dependencies_analysis", "key_concepts", "usage_example", "documentation",
    "state",
}

def test_single_mode_makes_exactly_one_call(tmp_path):
    provider = _CountingProvider()
    orch = Orchestrator(provider, analysis_mode="single")
    orch.process(make_execution_request(tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",)))
    assert provider.calls == 1

def test_triple_mode_makes_exactly_three_calls(tmp_path):
    provider = _CountingProvider()
    orch = Orchestrator(provider, parallel=False, analysis_mode="triple")
    request = make_execution_request(
        tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",), analysis_mode="triple"
    )
    orch.process(request)
    assert provider.calls == 3

def test_both_modes_produce_identical_top_level_keys(tmp_path):
    single = Orchestrator(_CountingProvider(), analysis_mode="single").process(
        make_execution_request(tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",))
    )
    triple = Orchestrator(
        _CountingProvider(), parallel=False, analysis_mode="triple"
    ).process(
        make_execution_request(
            tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",), analysis_mode="triple"
        )
    )
    assert set(single) - {"_analysis_revision", "_analysis_mode"} == _FLAT_KEYS
    assert set(triple) - {"_analysis_revision", "_analysis_mode"} == _FLAT_KEYS

def test_single_mode_compatibility_views(tmp_path):
    result = Orchestrator(_CountingProvider(), analysis_mode="single").process(
        make_execution_request(tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",))
    )
    assert result["structure"] == {
        "description": "A documented module.",
        "role_in_system": "entry point",
        "functions": [{"name": "main", "description": "runs"}],
        "classes": [{"name": "C", "description": "a class"}],
        "exports": ["main"],
    }
    assert result["documentation"] == {
        "description": "A documented module.",
        "role_in_system": "entry point",
        "key_concepts": ["startup"],
        "usage_example": "import mod",
    }
    # dependencies_analysis is exactly the cleaned dict, not a wrapper.
    assert result["dependencies_analysis"] == {
        "external": ["requests"], "dependency_refs": ["requests"]
    }
    assert result["state"] == "checked"

@pytest.mark.parametrize("mode,expected_calls", [("single", 2), ("triple", 6)])
def test_partial_failure_preserves_successful_output_in_both_modes(
    tmp_path, monkeypatch, mode, expected_calls
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "good.py").write_text("from bad import value\nresult = value\n")
    (tmp_path / "bad.py").write_text("value = 1\n")

    class OneFileFails(_CountingProvider):
        def complete_json(self, prompt, system=""):
            self.calls += 1
            match = re.search(r"^File: (.+)$", prompt, re.MULTILINE)
            assert match is not None
            return "not json" if match.group(1) == "bad.py" else _COMBINED_JSON

    provider = OneFileFails()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "good.py",
            "analysis_mode": mode,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 0,
            "allow_partial": True,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 1
    assert stats["failed"] == 1
    assert stats["attempted_calls"] == expected_calls
    assert provider.calls == expected_calls
    payload = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert [item["path"] for item in payload["files"]] == ["good.py"]

def _fake_provider():
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return _json.dumps({
                "description": "Documented.", "role_in_system": "r",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {},
                "key_concepts": [], "usage_example": "",
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()

def _patch_provider(monkeypatch):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: _fake_provider())

@pytest.mark.parametrize("parallel", [True, False])
def test_no_parallel_has_no_per_file_effect_in_single_mode(tmp_path, monkeypatch, parallel):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _patch_provider(monkeypatch)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "analysis_mode": "single",
         "parallel_agents": parallel, "propagate_changes": False},
    )
    # single mode makes one call regardless of the parallel_agents setting.
    assert stats["checked"] == 1
    assert stats["initial_calls_per_file"] == 1
    assert stats["attempted_calls"] == 1

def test_real_run_reports_mode(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _patch_provider(monkeypatch)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "analysis_mode": "single", "propagate_changes": False},
    )
    assert stats["analysis_mode"] == "single"
    assert stats["initial_calls_per_file"] == 1
