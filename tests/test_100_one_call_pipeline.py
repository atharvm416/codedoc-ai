"""0.10.0 Gate 2 + Gate 3 — mode dispatch, record parity, accounting, cache identity."""

from __future__ import annotations

import json
import re

import pytest

from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.execution import _agent_errors
from tests.test_095_provider_contracts import (
    _install_anthropic,
    _install_gemini,
    _install_openai,
)

_COMBINED = {
    "description": "A documented module.",
    "role_in_system": "entry point",
    "functions": [{"name": "main", "description": "runs"}],
    "classes": [{"name": "C", "description": "a class"}],
    "exports": ["main"],
    "dependencies_analysis": {"external": ["requests"], "dependency_refs": ["requests"]},
    "key_concepts": ["startup"],
    "usage_example": "import mod",
}
_COMBINED_JSON = json.dumps(_COMBINED)


class _CountingProvider:
    provider_name = "fake"

    def __init__(self, raw=_COMBINED_JSON):
        self._raw = raw
        self.calls = 0

    def complete_json(self, prompt, system=""):
        self.calls += 1
        return self._raw

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _descriptor():
    return {"rel_path": "pkg/mod.py", "language": "python", "extension": ".py"}


_FLAT_KEYS = {
    "file_path", "language", "extension", "imports", "description",
    "role_in_system", "functions", "classes", "exports", "structure",
    "dependencies_analysis", "key_concepts", "usage_example", "documentation",
    "state",
}


# ---------------------------------------------------------------------------
# Orchestrator dispatch + record parity
# ---------------------------------------------------------------------------

def test_single_mode_makes_exactly_one_call():
    provider = _CountingProvider()
    orch = Orchestrator(provider, analysis_mode="single")
    orch.process(_descriptor(), "x = 1\n", ["os"])
    assert provider.calls == 1


def test_triple_mode_makes_exactly_three_calls():
    provider = _CountingProvider()
    orch = Orchestrator(provider, parallel=False, analysis_mode="triple")
    orch.process(_descriptor(), "x = 1\n", ["os"])
    assert provider.calls == 3


def test_both_modes_produce_identical_top_level_keys():
    single = Orchestrator(_CountingProvider(), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
    )
    triple = Orchestrator(
        _CountingProvider(), parallel=False, analysis_mode="triple"
    ).process(_descriptor(), "x = 1\n", ["os"])
    assert set(single) - {"_analysis_revision", "_analysis_mode"} == _FLAT_KEYS
    assert set(triple) - {"_analysis_revision", "_analysis_mode"} == _FLAT_KEYS


def test_single_mode_compatibility_views():
    result = Orchestrator(_CountingProvider(), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
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


def test_deterministic_fields_cannot_be_replaced_by_model_output():
    raw = json.dumps({
        **_COMBINED,
        "file_path": "EVIL.py", "language": "evil", "imports": ["malware"],
        "extension": ".evil",
    })
    result = Orchestrator(_CountingProvider(raw), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
    )
    assert result["file_path"] == "pkg/mod.py"
    assert result["language"] == "python"
    assert result["extension"] == ".py"
    assert result["imports"] == ["os"]


def test_generated_record_carries_cache_identity():
    result = Orchestrator(_CountingProvider(), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
    )
    assert result["_analysis_revision"] == ANALYSIS_REVISION
    assert result["_analysis_mode"] == "single"


# ---------------------------------------------------------------------------
# Combined-agent failure shape
# ---------------------------------------------------------------------------

def test_combined_failure_produces_one_agent_error_via_documentation():
    result = Orchestrator(_CountingProvider("not json"), analysis_mode="single").process(
        _descriptor(), "x = 1\n", ["os"]
    )
    assert result["structure"] == {}
    assert result["dependencies_analysis"] == {}
    assert result["documentation"]["agent"] == "FileDocumentationAgent"
    assert result["documentation"]["error"]
    # Identity preserved on the failed flat record.
    assert result["file_path"] == "pkg/mod.py"
    # _agent_errors detects exactly one failure (through the documentation key).
    assert len(_agent_errors(result)) == 1
    # A failed result is not marked as a successful cache record.
    assert "_analysis_revision" not in result


# ---------------------------------------------------------------------------
# Gate 2 — provider parity (single mode, fake SDK clients)
# ---------------------------------------------------------------------------

_PROVIDER_MATRIX = [
    ("openai", "gpt-test", _install_openai, "content"),
    ("anthropic", "claude-test", _install_anthropic, "text"),
    ("gemini", "gemini-test", _install_gemini, "text"),
]


def _base_config(provider_name, model):
    return {
        "entry_file": "main.py",
        "llm_provider": provider_name,
        "model_name": model,
        "api_key": "test-key",
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
    }


@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_provider_executes_one_call_contract(tmp_path, monkeypatch, provider_name, model, installer, kw):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, **{kw: _COMBINED_JSON})
    stats = run_pipeline(project, _base_config(provider_name, model))

    assert stats["checked"] == 1
    # Exactly one provider request for the single file.
    calls = rec.get("create_calls") or rec.get("generate_calls")
    assert len(calls) == 1
    output = json.loads((project / "codedoc" / "codedoc.json").read_text(encoding="utf-8"))
    assert output["files"][0]["description"] == "A documented module."


@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_provider_malformed_response_fails_file(tmp_path, monkeypatch, provider_name, model, installer, kw):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, **{kw: "this is not json"})
    stats = run_pipeline(project, _base_config(provider_name, model))
    assert stats["checked"] == 0
    assert stats["failed"] == 1


@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_provider_error_propagates_to_failure(tmp_path, monkeypatch, provider_name, model, installer, kw):
    from codedoc.pipeline import run_pipeline

    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, error=RuntimeError("transient boom"))
    stats = run_pipeline(project, _base_config(provider_name, model))
    assert stats["checked"] == 0
    assert stats["failed"] == 1


# ---------------------------------------------------------------------------
# Gate 3 — cache identity across runs
# ---------------------------------------------------------------------------

def _pipeline_provider(monkeypatch):
    provider = _CountingProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: provider)
    return provider


def test_steady_state_reuse_skips_provider(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    cfg = {"entry_file": "main.py", "analysis_mode": "single", "propagate_changes": False}

    _pipeline_provider(monkeypatch)
    first = run_pipeline(tmp_path, cfg)
    assert first["checked"] == 1

    # Second run must not even create a provider.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda c: pytest.fail("provider created though all files were reusable"),
    )
    second = run_pipeline(tmp_path, cfg)
    assert second["checked"] == 0


def test_mode_switch_invalidates_reuse(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    _pipeline_provider(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                            "propagate_changes": False})

    # Switching to triple changes the cache identity → reprocess once.
    provider = _pipeline_provider(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "triple",
                                    "parallel_agents": False, "propagate_changes": False})
    assert stats["checked"] == 1
    assert provider.calls == 3


def test_legacy_record_without_identity_reprocessed_once(tmp_path, monkeypatch):
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "codedoc"
    out.mkdir()
    # Pre-0.10.0 record: matching hash, no cache-identity keys.
    out.joinpath("codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [{"path": "main.py", "hash": compute_file_hash(main),
                   "language": "python", "description": "legacy"}],
    }), encoding="utf-8")

    provider = _pipeline_provider(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                    "propagate_changes": False})
    assert stats["checked"] == 1
    assert provider.calls == 1
    # After reprocessing, the record carries the current identity.
    rec = json.loads(out.joinpath("codedoc.json").read_text(encoding="utf-8"))["files"][0]
    assert rec["_analysis_revision"] == ANALYSIS_REVISION
    assert rec["_analysis_mode"] == "single"


@pytest.mark.parametrize("mode,expected_calls", [("single", 2), ("triple", 6)])
def test_file_retry_repeats_the_resolved_modes_full_call_set(
    tmp_path, monkeypatch, mode, expected_calls
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    class FirstCallMalformed(_CountingProvider):
        def complete_json(self, prompt, system=""):
            self.calls += 1
            return "not json" if self.calls == 1 else _COMBINED_JSON

    provider = FirstCallMalformed()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": mode,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert stats["planned_calls"] == (1 if mode == "single" else 3)
    assert stats["attempted_calls"] == expected_calls
    assert provider.calls == expected_calls


@pytest.mark.parametrize("source", ["same_path", "identical", "checkpoint"])
@pytest.mark.parametrize(
    ("identity_change", "identity_value"),
    [
        (None, None),
        ("_analysis_revision", None),
        ("_analysis_revision", "stale-revision"),
        ("_analysis_mode", None),
        ("_analysis_mode", "triple"),
    ],
)
def test_every_reuse_source_requires_complete_matching_identity(
    tmp_path, source, identity_change, identity_value
):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan

    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    content_hash = compute_file_hash(target)
    identity = {
        "_analysis_revision": ANALYSIS_REVISION,
        "_analysis_mode": "single",
    }
    if identity_change is not None:
        if identity_value is None:
            identity.pop(identity_change)
        else:
            identity[identity_change] = identity_value

    record = {
        "path": "cached.py",
        "hash": content_hash,
        "description": "cached",
        **identity,
    }
    existing_docs = {}
    checkpoints = {}
    if source == "same_path":
        existing_docs["main.py"] = record
    elif source == "identical":
        existing_docs["cached.py"] = record
    else:
        checkpoints["main.py"] = {
            **record,
            "_checkpoint_hash": content_hash,
        }

    graph = DependencyGraph()
    graph.add_file("main.py")
    plan, _materials = build_pipeline_plan(
        file_map={
            "main.py": {
                "path": target,
                "rel_path": "main.py",
                "language": "python",
                "extension": ".py",
            }
        },
        graph=graph,
        selected_rels={"main.py"},
        entry_rel="main.py",
        existing_docs=existing_docs,
        checkpoint_records=checkpoints,
        forced_paths=[],
        config={
            "analysis_mode": "single",
            "propagate_changes": False,
            "max_files": 0,
        },
    )

    identity_matches = identity_change is None
    if identity_matches and source == "same_path":
        assert plan.unchanged_rels == frozenset({"main.py"})
    elif identity_matches and source == "identical":
        assert plan.identical_reuse_rels == frozenset({"main.py"})
    elif identity_matches:
        assert plan.checkpoint_reuse_rels == frozenset({"main.py"})
    else:
        assert plan.agent_rels == frozenset({"main.py"})


def test_checkpoint_reuse_routes_through_central_predicate(tmp_path, monkeypatch):
    import codedoc.core.planning as planning
    from codedoc.core.db import compute_file_hash
    from codedoc.core.graph import DependencyGraph

    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    content_hash = compute_file_hash(target)
    calls = []
    original = planning._record_is_reusable

    def observed(stored, expected_hash, expected_identity):
        calls.append((stored, expected_hash, expected_identity))
        return original(stored, expected_hash, expected_identity)

    monkeypatch.setattr(planning, "_record_is_reusable", observed)
    graph = DependencyGraph()
    graph.add_file("main.py")
    plan, _materials = planning.build_pipeline_plan(
        file_map={
            "main.py": {
                "path": target,
                "rel_path": "main.py",
                "language": "python",
                "extension": ".py",
            }
        },
        graph=graph,
        selected_rels={"main.py"},
        entry_rel="main.py",
        existing_docs={},
        checkpoint_records={
            "main.py": {
                "_checkpoint_hash": content_hash,
                "_analysis_revision": ANALYSIS_REVISION,
                "_analysis_mode": "single",
            }
        },
        forced_paths=[],
        config={"analysis_mode": "single", "propagate_changes": False},
    )
    assert plan.checkpoint_reuse_rels == frozenset({"main.py"})
    assert any(
        isinstance(stored, dict) and stored.get("hash") == content_hash
        for stored, _, _ in calls
    )


@pytest.mark.parametrize("mode", ["single", "triple"])
def test_matching_live_recovery_is_reused_in_both_modes(tmp_path, monkeypatch, mode):
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    output = tmp_path / "codedoc"
    output.mkdir()
    (output / "crash_recovery_codedoc.json").write_text(
        json.dumps(
            {
                "_crash_safety": "INCOMPLETE RUN - test",
                "_codedoc": {
                    "entry_file": "main.py",
                    "schema_version": "1.4",
                    "status": "in_progress",
                },
                "schema_version": "1.4",
                "files": [
                    {
                        "path": "main.py",
                        "hash": compute_file_hash(target),
                        "language": "python",
                        "description": "Recovered.",
                        "_analysis_revision": ANALYSIS_REVISION,
                        "_analysis_mode": mode,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: pytest.fail("matching recovery created a provider"),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": mode,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 0
    assert stats["skipped"] == 1
    final = json.loads((output / "codedoc.json").read_text(encoding="utf-8"))
    assert final["files"][0]["description"] == "Recovered."
    assert final["files"][0]["_analysis_mode"] == mode


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


# ---------------------------------------------------------------------------
# Deterministic multi-language public-output quality gate
# ---------------------------------------------------------------------------

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
