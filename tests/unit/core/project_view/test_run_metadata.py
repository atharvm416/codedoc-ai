"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from tests.support.pipeline_scenarios import patch_provider
from tests.support.pipeline_scenarios import md_meta
import pytest
from codedoc.core.markdown_view import markdown_from_view
from codedoc.core.output import write_summary
from codedoc.core.planning import PipelinePlan
from codedoc.core.project_view import build_project_view
from codedoc.pipeline import _set_plan_counters
from tests.support.run_metadata_cases import _records
from tests.support.run_metadata_cases import _stats
from tests.support.run_metadata_cases import _view
from tests.support.run_metadata_cases import _partition_sum

def test_F1_md_metadata_contains_file_hashes(tmp_path, monkeypatch):
    """F1: written MD output contains file_hashes in the codedoc-ai comment."""
    patch_provider(monkeypatch)
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    real_hash = compute_file_hash(src)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                             "propagate_changes": False, "parallel_agents": False})
    meta = md_meta(tmp_path / "codedoc" / "codedoc.md")
    assert "file_hashes" in meta
    assert meta["file_hashes"].get("main.py") == real_hash

def test_F2_md_metadata_contains_entry_file(tmp_path, monkeypatch):
    """F2: written MD metadata comment contains entry_file."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                             "propagate_changes": False, "parallel_agents": False})
    meta = md_meta(tmp_path / "codedoc" / "codedoc.md")
    assert meta.get("entry_file") == "main.py"

def test_F3_json_last_run_contains_entry_file(tmp_path, monkeypatch):
    """F3: written JSON last_run block contains entry_file."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                             "propagate_changes": False, "parallel_agents": False})
    data = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text())
    assert data["last_run"]["entry_file"] == "main.py"
    assert "_codedoc" not in data
    assert "project" not in data
    assert "run" not in data

def _plan(scanned, selected, agent):
    return PipelinePlan(
        scanned_rels=frozenset(scanned),
        documented_rels=frozenset(selected),
        changed_rels=frozenset(),
        forced_rels=frozenset(),
        process_rels=frozenset(agent),
        unchanged_rels=frozenset(),
        identical_reuse_rels=frozenset(),
        agent_rels=frozenset(agent),
        entry_rel=None,
        max_files=0,
        max_files_exceeded=False,
    )

def test_last_run_is_truthful_and_legacy_wrappers_are_removed():
    view = _view()

    assert "run" not in view
    assert "project" not in view
    assert view["last_run"] == {
        "entry_file": "main.py",
        "entry_source": "explicit",
        "documentation_scope": "entry",
        "analysis_mode": "single",
        "files_scanned": 7,
        "files_selected": 6,
        "files_documented_by_llm": 1,
        "files_failed": 1,
        "files_unattempted": 1,
        "files_skipped_insufficient_source": 0,
        "files_reused_unchanged": 2,
        "files_reused_identical_content": 1,
        "files_resumed_from_recovery": 1,
    }
    assert (
        view["last_run"]["files_selected"]
        == view["last_run"]["files_reused_unchanged"]
        + view["last_run"]["files_reused_identical_content"]
        + view["last_run"]["files_documented_by_llm"]
        + view["last_run"]["files_failed"]
        + view["last_run"]["files_unattempted"]
        + view["last_run"]["files_skipped_insufficient_source"]
    )
    assert len(view["files"]) < view["last_run"]["files_selected"]
    assert view["last_run"]["files_resumed_from_recovery"] <= view["last_run"][
        "files_reused_unchanged"
    ]

def test_markdown_and_summary_render_truthful_labels(tmp_path):
    md = markdown_from_view(_view())
    summary = write_summary(_stats(), tmp_path).read_text(encoding="utf-8")

    for text in (md, summary):
        assert "Files reused from cache" not in text
        assert "Files documented by LLM: 1" in text
        assert "Files reused (unchanged): 2" in text
        assert "Files reused (identical content): 1" in text
        assert "Files resumed from recovery: 1" in text

@pytest.mark.parametrize(
    ("shape", "counts", "selected"),
    [
        # checked, failed, skipped, reused, unattempted
        ("all_reused", (0, 0, 5, 0, 0), 5),
        ("fresh_full", (5, 0, 0, 0, 0), 5),
        ("mixed", (3, 0, 2, 1, 0), 6),
        ("with_failures", (2, 2, 0, 0, 0), 4),
        ("health_check_abort", (2, 1, 0, 0, 2), 5),
    ],
)
def test_partition_invariant_holds_for_every_run_shape(shape, counts, selected):
    checked, failed, skipped, reused, unattempted = counts
    stats = {
        "checked": checked,
        "failed": failed,
        "skipped": skipped,
        "reused": reused,
        "resumed": 0,
        "files_scanned": selected,
        "files_selected": selected,
        "unattempted_files": unattempted,
    }
    view = build_project_view(_records(), stats, entry_file="main.py")
    assert view["last_run"]["files_selected"] == _partition_sum(view["last_run"]) == selected

def test_set_plan_counters_computes_unattempted_and_keeps_partition(tmp_path):
    # 10 scanned, 8 selected, 5 routed to the agent; 3 checked + 1 failed leaves
    # 1 unattempted (the health-check / early-abort case D7 targets).
    plan = _plan(
        scanned=[f"f{i}.py" for i in range(10)],
        selected=[f"f{i}.py" for i in range(8)],
        agent=[f"f{i}.py" for i in range(5)],
    )
    stats = {"checked": 3, "failed": 1, "skipped": 2, "reused": 1, "resumed": 0}
    _set_plan_counters(stats, plan)

    assert stats["files_scanned"] == 10
    assert stats["files_selected"] == 8
    assert stats["unattempted_files"] == 1

    view = build_project_view(_records(), stats, entry_file="main.py")
    assert view["last_run"]["files_unattempted"] == 1
    assert view["last_run"]["files_selected"] == _partition_sum(view["last_run"]) == 8

def test_resumed_is_a_subset_and_must_not_be_summed_into_the_partition():
    # _stats(): resumed=1 is already inside reused_unchanged=2.  The true sum equals
    # files_selected; naively adding resumed overcounts — this pins the invariant so
    # nobody "fixes" it by summing resumed in (D5).
    lr = _view()["last_run"]
    assert lr["files_resumed_from_recovery"] <= lr["files_reused_unchanged"]
    assert lr["files_selected"] == _partition_sum(lr)
    assert _partition_sum(lr) + lr["files_resumed_from_recovery"] > lr["files_selected"]
