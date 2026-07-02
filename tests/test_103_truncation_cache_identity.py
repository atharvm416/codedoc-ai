"""0.10.3 — truncation parameters participate in cache identity.

Before 0.10.3, changing ``max_content_chars`` or ``truncation_head_ratio``
changed the truncated prompt sent for an oversized file but did *not* invalidate
its cached record, so an incremental re-run silently reused stale documentation
(the remedy the truncation warning recommends, "raise max_content_chars", had no
effect on a cached run).  These tests pin the fix: a file large enough to be
truncated carries a ``_max_context_revision`` encoding the effective ceiling and
head ratio, the planner reprocesses exactly those files when the ceiling/ratio
changes, and files that fit the ceiling stay reusable.
"""

from __future__ import annotations

import json

from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.db import compute_file_hash, source_char_count
from codedoc.core.graph import DependencyGraph
from codedoc.core.planning import build_pipeline_plan
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    MAX_CONTEXT_REVISION,
    expected_max_context_revision,
)

_COMBINED_JSON = json.dumps({
    "description": "A documented module.",
    "role_in_system": "entry point",
    "functions": [{"name": "main", "description": "runs"}],
    "classes": [{"name": "C", "description": "a class"}],
    "exports": ["main"],
    "dependencies_analysis": {"external": ["requests"], "dependency_refs": ["requests"]},
    "key_concepts": ["startup"],
    "usage_example": "import mod",
})

# Sentinel: omit ``_max_context_revision`` from the stored record (a legacy
# pre-0.10.3 record).
_OMIT = object()


class _FakeProvider:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return _COMBINED_JSON

    def complete(self, prompt, system="", temperature=0.1):
        return _COMBINED_JSON


# ---------------------------------------------------------------------------
# A. Pure revision helper
# ---------------------------------------------------------------------------

def test_revision_is_none_when_file_fits_ceiling():
    assert expected_max_context_revision(1000, max_chars=1000, head_ratio=0.70) is None
    assert expected_max_context_revision(999, max_chars=1000, head_ratio=0.70) is None


def test_revision_encodes_ceiling_and_head_ratio_when_truncated():
    assert (
        expected_max_context_revision(1001, max_chars=1000, head_ratio=0.70)
        == "truncate-v1:max=1000:head=0.7000"
    )
    # A different ceiling and ratio produce a different identity.
    assert (
        expected_max_context_revision(9999, max_chars=8000, head_ratio=0.85)
        == "truncate-v1:max=8000:head=0.8500"
    )


def test_revision_token_and_registry():
    assert MAX_CONTEXT_REVISION == "truncate-v1"
    assert "_max_context_revision" in CACHE_IDENTITY_KEYS


# ---------------------------------------------------------------------------
# B. source_char_count short-circuit
# ---------------------------------------------------------------------------

def test_source_char_count_short_circuits_small_files(tmp_path):
    p = tmp_path / "small.py"
    p.write_text("x = 1\n", encoding="utf-8")
    # Byte size (6) <= ceiling: returns a value <= ceiling, never above it.
    assert source_char_count(p, ceiling=1000) <= 1000


def test_source_char_count_exact_for_large_files(tmp_path):
    p = tmp_path / "big.py"
    p.write_text("x" * 2000, encoding="utf-8")
    assert source_char_count(p, ceiling=1000) == 2000


# ---------------------------------------------------------------------------
# C. Planning reuse for oversized files — the core fix
# ---------------------------------------------------------------------------

def _oversized_plan(tmp_path, stored_mcr, *, max_chars=1000, head_ratio=0.70):
    """Plan one oversized (2000-char) file whose cached record carries *stored_mcr*.

    Pass ``_OMIT`` to leave ``_max_context_revision`` off the record entirely.
    """
    src = tmp_path / "main.py"
    src.write_text("x" * 2000, encoding="utf-8")  # 2000 chars > 1000 ceiling
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    record = {
        "path": "main.py",
        "hash": compute_file_hash(src),
        "description": "cached",
        "_analysis_revision": "file-doc-v2",
        "_analysis_mode": "single",
    }
    if stored_mcr is not _OMIT:
        record["_max_context_revision"] = stored_mcr
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "max_content_chars": max_chars, "truncation_head_ratio": head_ratio,
    }
    plan, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
    )
    return plan


def test_oversized_file_with_matching_revision_is_reused(tmp_path):
    plan = _oversized_plan(tmp_path, "truncate-v1:max=1000:head=0.7000")
    assert "main.py" in plan.unchanged_rels
    assert "main.py" not in plan.agent_rels


def test_legacy_oversized_record_without_revision_is_reprocessed_once(tmp_path):
    plan = _oversized_plan(tmp_path, _OMIT)
    assert "main.py" in plan.agent_rels
    assert "main.py" not in plan.unchanged_rels


def test_raising_ceiling_reprocesses_truncated_file(tmp_path):
    # Cached under ceiling 1000; now running with ceiling 1500 (file is still
    # 2000 chars, so still truncated, but under a new identity).
    plan = _oversized_plan(tmp_path, "truncate-v1:max=1000:head=0.7000", max_chars=1500)
    assert "main.py" in plan.agent_rels
    assert "main.py" not in plan.unchanged_rels


def test_changing_head_ratio_reprocesses_truncated_file(tmp_path):
    plan = _oversized_plan(tmp_path, "truncate-v1:max=1000:head=0.7000", head_ratio=0.85)
    assert "main.py" in plan.agent_rels
    assert "main.py" not in plan.unchanged_rels


# ---------------------------------------------------------------------------
# D. Files that fit the ceiling are unaffected by ceiling/ratio changes
# ---------------------------------------------------------------------------

def _small_plan(tmp_path, *, max_chars, head_ratio=0.70):
    src = tmp_path / "main.py"
    src.write_text("x = 1\n", encoding="utf-8")  # 6 chars, never truncated
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    # A small file would never carry _max_context_revision.
    record = {
        "path": "main.py", "hash": compute_file_hash(src), "description": "cached",
        "_analysis_revision": "file-doc-v2", "_analysis_mode": "single",
    }
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "max_content_chars": max_chars, "truncation_head_ratio": head_ratio,
    }
    plan, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
    )
    return plan


def test_small_file_reusable_across_ceiling_and_ratio_changes(tmp_path):
    assert "main.py" in _small_plan(tmp_path, max_chars=1000).unchanged_rels
    assert "main.py" in _small_plan(
        tmp_path, max_chars=5000, head_ratio=0.85
    ).unchanged_rels


# ---------------------------------------------------------------------------
# E. The orchestrator stamps exactly what the planner expects (round-trip)
# ---------------------------------------------------------------------------

def _process(content, *, max_chars=1000, head_ratio=0.70):
    orch = Orchestrator(
        _FakeProvider(), analysis_mode="single",
        max_content_chars=max_chars, truncation_head_ratio=head_ratio,
    )
    return orch.process(
        {"rel_path": "pkg/mod.py", "language": "python", "extension": ".py"},
        content, ["os"],
    )


def test_orchestrator_stamps_revision_for_oversized_file():
    result = _process("x" * 5000)
    assert result["state"] == "checked"
    assert result["_max_context_revision"] == "truncate-v1:max=1000:head=0.7000"


def test_orchestrator_omits_revision_for_small_file():
    result = _process("x = 1\n")
    assert result["state"] == "checked"
    assert "_max_context_revision" not in result
