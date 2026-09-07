"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from tests.support.pipeline_scenarios import patch_provider
from tests.support.pipeline_scenarios import md_meta
import pytest
from codedoc.core.markdown_view import markdown_from_view
from codedoc.core.output import write_summary
from codedoc.core.planning import PipelinePlan
from codedoc.core.project_view import build_project_view, sanitize_public_view
from codedoc.pipeline import _set_plan_counters
from tests.support.run_metadata_cases import _records
from tests.support.run_metadata_cases import _stats
from tests.support.run_metadata_cases import _view
from tests.support.run_metadata_cases import _partition_sum
from tests.support.run_metadata_cases import _split_record, _split_stats

_OPTIONAL_SPLIT_COUNTERS = (
    "split_completed_files_reused",
    "split_partial_files_resumed",
    "split_unpaid_nodes",
    "split_reexecuted_nodes",
    "split_quarantined_nodes",
    "split_recovery_conflict_files",
)

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


def test_current_split_run_emits_all_six_optional_counters():
    last_run = build_project_view([_split_record()], _split_stats())["last_run"]
    assert {key: last_run[key] for key in _OPTIONAL_SPLIT_COUNTERS} == {
        "split_completed_files_reused": 0,
        "split_partial_files_resumed": 0,
        "split_unpaid_nodes": 3,
        "split_reexecuted_nodes": 1,
        "split_quarantined_nodes": 2,
        "split_recovery_conflict_files": 1,
    }


def test_predecessor_split_view_does_not_acquire_optional_counters_on_sanitize():
    predecessor = build_project_view([_split_record()], _split_stats())
    for key in _OPTIONAL_SPLIT_COUNTERS:
        predecessor["last_run"].pop(key)

    sanitized = sanitize_public_view(predecessor)

    assert all(key not in sanitized["last_run"] for key in _OPTIONAL_SPLIT_COUNTERS)


# ===========================================================================
# Section 5.8 correction round: the granular provider-free preflight details
# (per-file split/truncate/blocked descriptor arrays, the two scanner-skip
# categories, and every new path-bearing granular array) are DELIBERATELY
# EPHEMERAL -- they belong to the dry-run / real-run stats surface and the
# ``plan_reporter`` snapshot only, and must never be persisted into the public
# ``last_run`` block of codedoc.json / codedoc.md. No project_view.py change is
# authorized; these assertions pin the current (correct) behaviour.
# ===========================================================================

_S8_EPHEMERAL_GRANULAR_KEYS = (
    "split_plan_details", "split_plan_details_total", "split_plan_details_retained",
    "split_plan_details_omitted", "split_plan_details_digest",
    "truncate_plan_details", "truncate_plan_details_total",
    "truncate_plan_details_retained", "truncate_plan_details_omitted",
    "truncate_plan_details_digest",
    "split_blocked_details", "split_blocked_details_total",
    "split_blocked_details_retained", "split_blocked_details_omitted",
    "split_blocked_details_digest",
    "scanner_size_skip_details", "scanner_size_skip_details_total",
    "scanner_size_skip_details_retained", "scanner_size_skip_details_omitted",
    "scanner_size_skip_details_digest",
    "scanner_admission_skip_details", "scanner_admission_skip_details_total",
    "scanner_admission_skip_details_retained", "scanner_admission_skip_details_omitted",
    "scanner_admission_skip_details_digest",
    # Section 6.3: the two EPHEMERAL recovery-transition counts are preflight /
    # CLI observability only -- unlike the allowlisted split_reexecuted_nodes /
    # split_recovery_conflict_files they are never persisted into last_run.
    "split_recovery_discarded_predecessor_nodes",
    "split_recovery_replacement_nodes_planned",
    # billing fields are runtime disclosure, not persisted schema
    "initial_provider_calls_planned", "prompt_review_calls_planned",
    "initial_documentation_calls_planned", "correction_calls_possible_max",
    "provider_calls_max_before_retries", "file_retry_attempts",
    "retries_included_in_ceiling", "max_planned_calls_applies_to",
    "call_manifest_digest",
)


def _s8_enriched_split_stats() -> dict:
    stats = _split_stats()
    stats.update(
        {
            "split_plan_details": [
                {"path": "src/huge.py", "initial_calls": 5,
                 "units": [{"ordinal": 0}], "leaves": [{"payload_chars": 600}]}
            ],
            "split_plan_details_total": 1,
            "split_plan_details_retained": 1,
            "split_plan_details_omitted": 0,
            "split_plan_details_digest": "sha256:" + "a" * 64,
            "truncate_plan_details": [
                {"path": "src/wide.py", "retained_head_chars": 700,
                 "retained_tail_chars": 300, "omitted_chars": 4000}
            ],
            "truncate_plan_details_total": 1,
            "truncate_plan_details_retained": 1,
            "truncate_plan_details_omitted": 0,
            "truncate_plan_details_digest": "sha256:" + "b" * 64,
            "split_blocked_details": [
                {"path": "src/blocked.py", "phase": "division-packing",
                 "reason": "chunk-cap", "observed": 9, "limit": 1,
                 "guidance_code": "raise-source-ceiling-or-split-source"}
            ],
            "split_blocked_details_total": 1,
            "split_blocked_details_retained": 1,
            "split_blocked_details_omitted": 0,
            "split_blocked_details_digest": "sha256:" + "c" * 64,
            "scanner_size_skip_details": [
                {"path": "src/big.bin.py", "phase": "scanner-byte",
                 "observed": 900000, "limit": 512000,
                 "guidance_code": "raise-scan-byte-limit-or-exclude"}
            ],
            "scanner_size_skip_details_total": 1,
            "scanner_size_skip_details_retained": 1,
            "scanner_size_skip_details_omitted": 0,
            "scanner_size_skip_details_digest": "sha256:" + "d" * 64,
            "scanner_admission_skip_details": [
                {"path": "src/gone.py", "phase": "scanner-admission",
                 "reason": "missing", "guidance_code": "fix-entry-path"}
            ],
            "scanner_admission_skip_details_total": 1,
            "scanner_admission_skip_details_retained": 1,
            "scanner_admission_skip_details_omitted": 0,
            "scanner_admission_skip_details_digest": "sha256:" + "e" * 64,
            "split_recovery_discarded_predecessor_nodes": 5,
            "split_recovery_replacement_nodes_planned": 4,
            "initial_provider_calls_planned": 4,
            "prompt_review_calls_planned": 0,
            "initial_documentation_calls_planned": 4,
            "correction_calls_possible_max": 4,
            "provider_calls_max_before_retries": 8,
            "file_retry_attempts": 1,
            "retries_included_in_ceiling": False,
            "max_planned_calls_applies_to": "initial_provider_calls_planned",
            "call_manifest_digest": "f" * 64,
        }
    )
    return stats


def test_s8_granular_preflight_details_never_reach_persisted_last_run():
    last_run = build_project_view([_split_record()], _s8_enriched_split_stats())["last_run"]
    for key in _S8_EPHEMERAL_GRANULAR_KEYS:
        assert key not in last_run, key
    # a spot-check that legitimate persisted aggregates DO remain
    assert "split_divided_files" in last_run
    assert "split_chunks" in last_run
    # the allowlisted §6.3 counters survive; the two ephemeral ones are gone
    assert last_run["split_reexecuted_nodes"] == 1
    assert last_run["split_recovery_conflict_files"] == 1
    assert "split_recovery_discarded_predecessor_nodes" not in last_run
    assert "split_recovery_replacement_nodes_planned" not in last_run


def _s8_persisted_last_run(tmp_path):
    written = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    return written["last_run"]


def _s8_assert_no_granular_keys(last_run):
    for key in _S8_EPHEMERAL_GRANULAR_KEYS:
        assert key not in last_run, key
    # and no *_details / *_digest / *_details_* granular key under any name
    for key in last_run:
        assert not key.endswith("_details"), key
        assert not key.endswith("_details_digest"), key
        assert not key.endswith("_details_total"), key
        assert not key.endswith("_details_retained"), key
        assert not key.endswith("_details_omitted"), key


def test_s8_written_codedoc_json_last_run_has_no_granular_preflight_details(
    tmp_path, monkeypatch
):
    """Real ``large_file_strategy: "split"`` run: both oversized modules take the
    SPLIT route (no truncate route is exercised here -- that is a separate run
    below) and one module is byte-size skipped. The run genuinely populates the
    ``split_plan`` and ``scanner_size_skip`` categories on its stats surface;
    the persisted ``last_run`` block still contains none of the granular /
    ephemeral preflight keys, and neither of the two ephemeral §6.3 recovery
    counters. The remaining granular categories (truncate / blocked / admission)
    are proven strippable by the synthetic-injection test above and by the
    truncate-route run below.
    """
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    (tmp_path / "main.py").write_text(
        "import wide\nimport huge\nVALUE = 1\n", encoding="utf-8"
    )
    (tmp_path / "huge.py").write_text(
        "\n".join(f"def fn_{i}(): return {i}" for i in range(400)) + "\n",
        encoding="utf-8", newline="",
    )
    (tmp_path / "wide.py").write_text(
        "\n".join(f"w{i} = {i}" for i in range(400)) + "\n",
        encoding="utf-8", newline="",
    )
    (tmp_path / "skipme.py").write_text(
        "\n".join(f"s{i} = {i}" for i in range(40000)), encoding="utf-8"
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py", "documentation_scope": "all",
            "large_file_strategy": "split", "max_content_chars": 2000,
            "max_file_size_kb": 500, "parallel_agents": False,
            "propagate_changes": False,
        },
    )
    # the two categories this run actually exercises are non-empty on the
    # stats surface (we are not claiming a zero-population category).
    assert stats["split_plan_details_total"] >= 1
    assert stats["scanner_size_skip_details_total"] >= 1
    assert stats["truncate_plan_details_total"] == 0

    last_run = _s8_persisted_last_run(tmp_path)
    _s8_assert_no_granular_keys(last_run)
    assert "split_recovery_discarded_predecessor_nodes" not in last_run
    assert "split_recovery_replacement_nodes_planned" not in last_run


def test_s8_written_codedoc_json_last_run_truncate_route_has_no_granular_details(
    tmp_path, monkeypatch
):
    """Companion genuine ``large_file_strategy: "truncate"`` run: an oversized
    module takes the TRUNCATE route, populating the ``truncate_plan`` category on
    the stats surface. The persisted ``last_run`` block still carries none of the
    granular / ephemeral preflight keys.
    """
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    (tmp_path / "main.py").write_text("import wide\nVALUE = 1\n", encoding="utf-8")
    (tmp_path / "wide.py").write_text(
        "\n".join(f"w{i} = {i}" for i in range(4000)) + "\n",
        encoding="utf-8", newline="",
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: SmartFake())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py", "documentation_scope": "all",
            "large_file_strategy": "truncate", "max_content_chars": 2000,
            "parallel_agents": False, "propagate_changes": False,
        },
    )
    assert stats["truncate_plan_details_total"] >= 1
    assert stats["large_files_routed_truncate"] >= 1

    last_run = _s8_persisted_last_run(tmp_path)
    _s8_assert_no_granular_keys(last_run)
    assert "split_recovery_discarded_predecessor_nodes" not in last_run
    assert "split_recovery_replacement_nodes_planned" not in last_run
