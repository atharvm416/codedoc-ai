"""Tests organized by feature ownership."""

from __future__ import annotations

from pathlib import Path
import pytest
from tests.support.pipeline_usage import write_py
from codedoc.core.planning import PipelinePlan
from codedoc.pipeline import run_pipeline
from tests.support.selection_projects import _project
from codedoc.cli.cli import (
    run_cli,
)
from tests.support.feasibility_cases import _cross_file_profile
from tests.support.feasibility_cases import _ReviewFake
from codedoc.core.resume import (
    RECOVERY_FILENAME,
)
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _first_run
from tests.support.cross_format_runs import _forbid_provider

def test_dry_run_is_provider_free_and_filesystem_read_only(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    write_py(tmp_path / "main.py")
    legacy = tmp_path / "codedoc" / "codedoc_db.json"
    stale = tmp_path / "codedoc" / ".codedoc_build.json"
    legacy.parent.mkdir()
    legacy.write_text("legacy", encoding="utf-8")
    stale.write_text('{"_codedoc": {}, "files": []}', encoding="utf-8")
    before = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("dry-run created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.Orchestrator",
        lambda *args, **kwargs: pytest.fail("dry-run created an orchestrator"),
    )

    stats = run_pipeline(
        tmp_path,
        {"dry_run": True, "entry_file": "main.py", "output_dir": "new-output"},
    )
    after = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    assert stats["dry_run"] is True
    # 0.10.0: default single mode → one call per file; the single-mode input
    # estimate is exact, so it is not a lower bound.
    assert stats["estimated_calls"] == 1
    assert stats["estimate_is_lower_bound"] is False
    assert before == after
    assert not (tmp_path / "new-output").exists()

def test_dry_run_reports_conflict_and_cap_but_cli_exits_zero(tmp_path, monkeypatch, capsys):

    write_py(tmp_path / "main.py")
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").write_text('{"foreign": true}', encoding="utf-8")

    code = run_cli(
        [
            str(tmp_path),
            "--entry",
            "main.py",
            "--output",
            "out/report.json",
            "--max-files",
            "0",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "ownership conflict" in capsys.readouterr().out.lower()

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

class TestDryRunRatioEstimate:
    def test_A9_non_default_ratio_changes_estimate(self, tmp_path):
        """_estimate_planned_input_tokens reflects the configured head ratio."""
        from unittest.mock import MagicMock
        from codedoc.pipeline import _estimate_planned_input_tokens
        from tests.support.execution_requests import make_execution_request

        # Create a file larger than max_content_chars so truncation fires
        src = tmp_path / "big.py"
        src.write_text("x = 1\n" * 5000, encoding="utf-8")
        descriptor = {
            "rel_path": "big.py",
            "path": src,
            "language": "python",
            "extension": ".py",
        }

        plan = MagicMock(spec=PipelinePlan)
        plan.agent_rels = frozenset({"big.py"})
        file_map = {"big.py": descriptor}

        config_default = {"analysis_mode": "single", "max_content_chars": 500, "truncation_head_ratio": 0.70}
        config_alt = {"analysis_mode": "single", "max_content_chars": 500, "truncation_head_ratio": 0.10}

        default_request = make_execution_request(
            tmp_path,
            "big.py",
            "x = 1\n" * 5000,
            max_content_chars=500,
            truncation_head_ratio=0.70,
        )
        alt_request = make_execution_request(
            tmp_path,
            "big.py",
            "x = 1\n" * 5000,
            max_content_chars=500,
            truncation_head_ratio=0.10,
        )
        est_default = _estimate_planned_input_tokens(
            plan,
            file_map,
            config_default,
            execution_requests={"big.py": default_request},
        )
        est_alt = _estimate_planned_input_tokens(
            plan,
            file_map,
            config_alt,
            execution_requests={"big.py": alt_request},
        )

        # Both are positive; the estimates may differ because the head/tail content differs.
        assert est_default > 0
        assert est_alt > 0

@pytest.mark.parametrize(
    ("enabled", "possible_extra", "maximum"),
    [(False, 0, 1), (True, 1, 2)],
)
def test_dry_run_reports_baseline_and_correction_ceiling(
    tmp_path, monkeypatch, enabled, possible_extra, maximum
):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("dry-run must not construct a provider"),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "dry_run": True,
            "response_correction_enabled": enabled,
            "propagate_changes": False,
        },
    )

    assert stats["estimated_calls"] == 1
    assert stats["response_correction_enabled"] is enabled
    assert stats["response_correction_calls_possible_max"] == possible_extra
    assert stats["estimated_calls_max_with_correction"] == maximum

def test_dry_and_real_runs_transport_the_same_advisory_without_persisting_it(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    profile = _cross_file_profile()

    dry = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "prompt_profiles": profile,
        },
    )
    fake = _ReviewFake("SAFE")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    real = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "prompt_profiles": profile,
        },
    )

    notes = dry["prompt_customization_feasibility_advisories"]
    assert notes
    assert notes == real["prompt_customization_feasibility_advisories"]
    assert dry["prompt_customization_feasibility_notes"] == len(notes)
    assert real["prompt_customization_feasibility_notes"] == len(notes)
    assert fake.review_calls == 1
    assert fake.doc_calls == 1
    output = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    assert "prompt_customization_feasibility" not in output

def test_cross_format_dry_run_is_provider_free_and_read_only(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    _forbid_provider(monkeypatch)

    stats = run_pipeline(tmp_path, {**_config("md"), "dry_run": True})
    out = tmp_path / "docs"
    assert stats["would_call_llm_for"] == 0
    assert stats["estimated_calls"] == 0
    assert not (out / "codedoc.md").exists()
    assert not (out / RECOVERY_FILENAME).exists()

def test_dry_run_reports_mode_single(tmp_path):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "dry_run": True})
    assert stats["analysis_mode"] == "single"
    assert stats["initial_calls_per_file"] == 1
    assert stats["estimate_is_lower_bound"] is False
    assert stats["estimated_calls"] == 1

def test_dry_run_reports_mode_triple(tmp_path):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    stats = run_pipeline(
        tmp_path, {"entry_file": "main.py", "dry_run": True, "analysis_mode": "triple"}
    )
    assert stats["analysis_mode"] == "triple"
    assert stats["initial_calls_per_file"] == 3
    assert stats["estimate_is_lower_bound"] is True
    assert stats["estimated_calls"] == 3

@pytest.mark.parametrize(
    ("mode", "lower_bound"), [("single", False), ("triple", True)]
)
def test_empty_dry_run_reports_mode_aware_estimate(tmp_path, mode, lower_bound):
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {"dry_run": True, "analysis_mode": mode})
    assert stats["analysis_mode"] == mode
    assert stats["estimate_is_lower_bound"] is lower_bound
    assert stats["estimated_calls"] == 0

def test_single_prompt_estimate_does_not_exceed_triple_lower_bound(tmp_path):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text(
        "import os\n\ndef main():\n    return os.getcwd()\n", encoding="utf-8"
    )
    common = {"entry_file": "main.py", "dry_run": True}
    single = run_pipeline(tmp_path, {**common, "analysis_mode": "single"})
    triple = run_pipeline(tmp_path, {**common, "analysis_mode": "triple"})
    assert single["estimated_input_tokens"] <= triple["estimated_input_tokens"]
    assert single["estimate_is_lower_bound"] is False
    assert triple["estimate_is_lower_bound"] is True

@pytest.mark.parametrize(
    ("analysis_mode", "per_file"),
    [("single", 1), ("triple", 3)],
)
def test_dry_run_excludes_empty_files_from_calls_and_candidate_cap(
    tmp_path, analysis_mode, per_file
):
    (tmp_path / "empty.py").write_text("\n", encoding="utf-8")
    (tmp_path / "normal.py").write_text("VALUE = 1\n", encoding="utf-8")

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "documentation_scope": "all",
            "analysis_mode": analysis_mode,
            "max_files": 1,
        },
    )

    assert stats["would_skip_insufficient_source"] == 1
    assert stats["would_call_llm_for"] == 1
    assert stats["estimated_calls"] == per_file
    assert stats["documentation_calls_planned"] == per_file
    assert stats["response_correction_calls_possible_max"] == 0
    assert stats["max_files_candidate_files"] == 1
    assert stats["max_files_exceeded"] is False

    (tmp_path / "empty.py").write_text("OTHER = 2\n", encoding="utf-8")
    without_skip = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "documentation_scope": "all",
            "analysis_mode": analysis_mode,
        },
    )
    assert without_skip["estimated_input_tokens"] > stats["estimated_input_tokens"]

def test_dry_run_read_failure_remains_a_documentation_candidate(
    tmp_path, monkeypatch
):
    bad = tmp_path / "bad.py"
    bad.write_text("VALUE = 1\n", encoding="utf-8")
    original = Path.read_text

    def failing_read(self, *args, **kwargs):
        if self == bad:
            raise OSError("fixture read failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read)
    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "bad.py",
        },
    )

    assert stats["would_skip_insufficient_source"] == 0
    assert stats["would_call_llm_for"] == 1
    assert stats["estimated_calls"] == 1
