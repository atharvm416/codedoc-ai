"""Section 2 route coverage: every publication route reaches the one shared
source-backed structural authority (findings F-3 / F-4 / F-5).

Routes exercised end to end:

* single ordinary        -- ``Orchestrator.process`` (analysis_mode="single")
* triple ordinary        -- ``Orchestrator.process`` (analysis_mode="triple")
* single truncate        -- an oversized single-mode file (full source still
  drives reconciliation even though the model sees a truncated head)
* triple truncate        -- an oversized triple-mode file
* split final synthesis  -- ``run_pipeline`` with a real split + a leaf
  provider that over-reports; the assembled record is reconciled
* recovery re-assembly    -- a second zero-call run over the completed split
"""

from __future__ import annotations

import json

import pytest

from codedoc.agents.orchestrator import Orchestrator
from codedoc.pipeline import run_pipeline
from tests.support.execution_requests import make_execution_request
from tests.support.structure_extra import requires_structure_pack


# ---------------------------------------------------------------------------
# ordinary + truncate, single + triple
# ---------------------------------------------------------------------------

_OVER_REPORTING_COMBINED = {
    "description": "d",
    "role_in_system": "r",
    "functions": [
        {"name": "realArrow", "description": "arrow"},
        {"name": "realArrow", "description": "arrow dup"},
        {"name": "DPRTab", "description": "component"},
        {"name": "updateEnvVersion", "description": "invented"},
        {"name": "useState", "description": "imported"},
    ],
    "classes": [
        {"name": "DPRTab", "description": "wrongly a class"},
        {"name": "RealClass", "description": "a class"},
        {"name": "GhostClass", "description": "invented"},
    ],
    "exports": ["realArrow", "GhostExport"],
    "key_concepts": ["k"],
    "usage_example": "u",
    "dependencies_analysis": {"internal": [], "external": []},
}

_TSX_SOURCE = (
    "import React from 'react';\n"
    "import { useState } from 'react';\n"
    "export const realArrow = () => { return 1; };\n"
    "const DPRTab: React.FC = () => { return null; };\n"
    "class RealClass { method() {} }\n"
)


class _OverReportingProvider:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        if "Analyse the imports" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "dependency_refs": [],
                        "catalog_updates": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        # Combined agent (single) and the triple StructureAgent/DocumentationAgent
        # all reach this over-reporting shape; each sub-cleaner keeps only its
        # own fields.
        return json.dumps(_OVER_REPORTING_COMBINED)

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _assert_reconciled(result: dict) -> None:
    assert [f["name"] for f in result["functions"]] == [
        "realArrow",
        "DPRTab",
    ]
    assert result["functions"][0]["description"] == "arrow"  # first wins, deduped
    assert [c["name"] for c in result["classes"]] == ["RealClass"]
    assert result["exports"] == ["realArrow"]
    assert "updateEnvVersion" not in json.dumps(result)
    assert "GhostClass" not in json.dumps(result)
    assert "GhostExport" not in json.dumps(result)


@pytest.mark.parametrize("mode", ["single", "triple"])
def test_ordinary_route_reaches_the_shared_authority(tmp_path, mode):
    orch = Orchestrator(
        _OverReportingProvider(), parallel=False, analysis_mode=mode
    )
    request = make_execution_request(
        tmp_path, "src/Panel.tsx", _TSX_SOURCE, language="tsx", analysis_mode=mode
    )
    _assert_reconciled(orch.process(request))


@pytest.mark.parametrize("mode", ["single", "triple"])
def test_truncate_route_reconciles_against_the_full_source(tmp_path, mode):
    # The declarations sit past a tiny ceiling, so the model only ever sees a
    # truncated head -- reconciliation must still use the full planned source.
    filler = "// padding line that is only here to exceed the ceiling\n" * 60
    oversized = filler + _TSX_SOURCE
    orch = Orchestrator(
        _OverReportingProvider(), parallel=False, analysis_mode=mode
    )
    request = make_execution_request(
        tmp_path,
        "src/Panel.tsx",
        oversized,
        language="tsx",
        analysis_mode=mode,
        max_content_chars=200,
    )
    _assert_reconciled(orch.process(request))


_OVERLOAD_PY = (
    "def dispatch(a):\n    return a\n\n\n"
    "def dispatch(a, b):\n    return a + b\n\n\n"
    "class Solo:\n    pass\n"
)


class _OverloadProvider:
    """Reports the two ``dispatch`` overloads in a caller-chosen order, each
    with a signature that isolates exactly one source declaration."""

    provider_name = "fake"

    def __init__(self, reversed_order: bool = False):
        one = {"name": "dispatch", "description": "takes one", "signature": "dispatch(a)"}
        two = {"name": "dispatch", "description": "takes two", "signature": "dispatch(a, b)"}
        self._functions = [two, one] if reversed_order else [one, two]

    def complete_json(self, prompt, system=""):
        if "Analyse the imports" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "dependency_refs": [],
                        "catalog_updates": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        return json.dumps(
            {
                "description": "d",
                "role_in_system": "r",
                "functions": list(self._functions),
                "classes": [{"name": "Solo", "description": "solo"}],
                "exports": [],
                "key_concepts": ["k"],
                "usage_example": "u",
            }
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


@requires_structure_pack
@pytest.mark.parametrize("mode", ["single", "triple"])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_real_route_signature_disambiguates_overloads_and_keeps_source_order(
    tmp_path, mode, reversed_order
):
    orch = Orchestrator(
        _OverloadProvider(reversed_order=reversed_order),
        parallel=False,
        analysis_mode=mode,
    )
    request = make_execution_request(
        tmp_path, "svc.py", _OVERLOAD_PY, language="python", analysis_mode=mode
    )
    result = orch.process(request)

    funcs = result["functions"]
    assert [f["name"] for f in funcs] == ["dispatch", "dispatch"]
    # Source order: dispatch(a) is declared first, dispatch(a, b) second --
    # independent of the provider's ordering.
    assert funcs[0]["description"] == "takes one"
    assert funcs[1]["description"] == "takes two"
    assert [c["name"] for c in result["classes"]] == ["Solo"]

    # Public schema stays {name, description?} on both the top level and the
    # nested ``structure`` mirror; no signature/range/id leaks anywhere.
    for entry in list(funcs) + list(result["structure"]["functions"]):
        assert set(entry) <= {"name", "description"}
    blob = json.dumps(result)
    for leaked in (
        "signature",
        "dispatch(a)",
        "dispatch(a, b)",
        "start_byte",
        "symbol_id",
        "_provenance",
    ):
        assert leaked not in blob


def test_overload_route_without_structure_pack_is_safe_bare(tmp_path, monkeypatch):
    # With no parser signatures the model signature cannot be lined up against
    # an authoritative one; both real declarations still publish, bare, and no
    # transient signature leaks.
    import sys

    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    orch = Orchestrator(_OverloadProvider(), parallel=False, analysis_mode="single")
    request = make_execution_request(
        tmp_path, "svc.py", _OVERLOAD_PY, language="python"
    )
    result = orch.process(request)
    assert [f["name"] for f in result["functions"]] == ["dispatch", "dispatch"]
    assert all("description" not in f for f in result["functions"])
    assert "signature" not in json.dumps(result)


_CAP_SOURCE = "\n\n".join(
    f"def f{index}():\n    return {index}" for index in range(12)
)


class _SignatureHeavyProvider:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return json.dumps(
            {
                "description": "d",
                "functions": [
                    {
                        "name": f"f{index}",
                        "description": "documented",
                        "signature": "s" * 2000,
                    }
                    for index in range(12)
                ],
            }
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def test_single_route_transient_signatures_cannot_evict_public_symbols(tmp_path):
    orch = Orchestrator(
        _SignatureHeavyProvider(), parallel=False, analysis_mode="single"
    )
    request = make_execution_request(
        tmp_path, "cap.py", _CAP_SOURCE, language="python", analysis_mode="single"
    )

    result = orch.process(request)

    assert [item["name"] for item in result["functions"]] == [
        f"f{index}" for index in range(12)
    ]
    assert all(item.get("description") == "documented" for item in result["functions"])
    assert "signature" not in json.dumps(result)


def test_failure_result_is_not_reconciled_and_stays_empty(tmp_path):
    class _FailingProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return "not json at all"

        complete = complete_json

    orch = Orchestrator(_FailingProvider(), parallel=False, analysis_mode="single")
    request = make_execution_request(
        tmp_path, "src/Panel.tsx", _TSX_SOURCE, language="tsx"
    )
    result = orch.process(request)
    assert result["functions"] == []
    assert result["classes"] == []
    assert result["documentation"].get("error")


# ---------------------------------------------------------------------------
# split final synthesis + recovery re-assembly
# ---------------------------------------------------------------------------

_SPLIT_TSX = (
    "import React from 'react';\n"
    + "".join(
        f"export const item_{i:03d} = () => {{ return {i}; }};\n" for i in range(120)
    )
    + "class BigPanel {\n"
    + "".join(f"  method_{i:03d}() {{ return {i}; }}\n" for i in range(60))
    + "}\n"
)


class _SplitLeafProvider:
    """Every leaf over-reports one ghost function and one ghost export."""

    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        if "Analyse the imports" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "dependency_refs": [],
                        "catalog_updates": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        if "This is one bounded fragment of a larger" in prompt:
            return json.dumps(
                {
                    "description": "A bounded fragment.",
                    "functions": [
                        {"name": "item_000", "description": "a real item"},
                        {"name": "ghostLeafFn", "description": "invented in a leaf"},
                    ],
                    "classes": [
                        {"name": "BigPanel", "description": "the panel"},
                        {"name": "GhostLeafClass", "description": "invented"},
                    ],
                    "exports": ["item_000", "ghostLeafExport"],
                }
            )
        if "Refine one combined narrative from" in prompt:
            return json.dumps({"narrative": "A refined narrative."})
        return json.dumps(
            {
                "description": "A file.",
                "role_in_system": "core",
                "functions": [{"name": "ghostFinalFn", "description": "invented"}],
                "classes": [{"name": "GhostFinalClass", "description": "invented"}],
                "exports": ["ghostFinalExport"],
                "key_concepts": ["split"],
                "usage_example": "import x",
            }
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _run_split(tmp_path, monkeypatch):
    (tmp_path / "big.tsx").write_text(_SPLIT_TSX, encoding="utf-8", newline="")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _cfg: _SplitLeafProvider()
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "big.tsx",
            "documentation_scope": "entry",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    record = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    return stats, record


def test_split_final_record_reaches_the_shared_authority(tmp_path, monkeypatch):
    stats, record = _run_split(tmp_path, monkeypatch)
    assert stats["checked"] == 1 and stats["failed"] == 0

    blob = json.dumps(record)
    # Every invented ledger / synthesis declaration is gone.
    for ghost in (
        "ghostLeafFn",
        "GhostLeafClass",
        "ghostLeafExport",
        "ghostFinalFn",
        "GhostFinalClass",
        "ghostFinalExport",
    ):
        assert ghost not in blob, ghost
    # The real declarations survive, kind-correct and source-ordered.
    function_names = [f["name"] for f in record["functions"]]
    assert function_names == sorted(function_names)
    assert "item_000" in function_names
    # Every leaf that contains ``item_000`` reports it, but it is one real
    # declaration -> exactly one published entry (split duplicate-count).
    assert function_names.count("item_000") == 1
    assert record["classes"] == [{"name": "BigPanel", "description": "the panel"}]
    assert "item_000" in record["exports"]
    assert record["exports"].count("item_000") == 1
    assert "BigPanel" not in function_names  # class stays a class


class _OrderVariedSplitLeafProvider(_SplitLeafProvider):
    """Same leaf payloads, but the fragment response lists the real item
    *after* a ghost and in a different key order, so leaf/model ordering cannot
    be what produces the final source order."""

    def complete_json(self, prompt, system=""):
        if "This is one bounded fragment of a larger" in prompt:
            return json.dumps(
                {
                    "exports": ["ghostLeafExport", "item_000"],
                    "classes": [
                        {"name": "GhostLeafClass", "description": "invented"},
                        {"name": "BigPanel", "description": "the panel"},
                    ],
                    "functions": [
                        {"name": "ghostLeafFn", "description": "invented in a leaf"},
                        {"name": "item_000", "description": "a real item"},
                    ],
                    "description": "A bounded fragment.",
                }
            )
        return super().complete_json(prompt, system)


def test_split_output_order_is_source_order_not_leaf_or_model_order(tmp_path, monkeypatch):
    (tmp_path / "big.tsx").write_text(_SPLIT_TSX, encoding="utf-8", newline="")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: _OrderVariedSplitLeafProvider(),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "big.tsx",
            "documentation_scope": "entry",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 1 and stats["failed"] == 0
    record = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    function_names = [f["name"] for f in record["functions"]]
    assert function_names == sorted(function_names)
    assert function_names.count("item_000") == 1
    assert "ghostLeafFn" not in json.dumps(record)


def test_split_recovery_reassembly_is_identically_reconciled(tmp_path, monkeypatch):
    _stats1, record1 = _run_split(tmp_path, monkeypatch)

    # A second run reuses the completed split with zero provider calls and
    # re-assembles the final record through the same authority.
    def _forbidden(_cfg):
        raise AssertionError("second run constructed a provider")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _forbidden)
    stats2 = run_pipeline(
        tmp_path,
        {
            "entry_file": "big.tsx",
            "documentation_scope": "entry",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "max_parallel_files": 1,
            "propagate_changes": False,
        },
    )
    record2 = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert stats2["checked"] == 0
    assert record2["functions"] == record1["functions"]
    assert record2["classes"] == record1["classes"]
    assert record2["exports"] == record1["exports"]
