"""0.10.0 — selectable analysis mode: config, CLI, precedence, interactions."""

from __future__ import annotations

import json

import pytest

from codedoc.agents.orchestrator import initial_calls_per_file
from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError


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


# ---------------------------------------------------------------------------
# initial_calls_per_file helper
# ---------------------------------------------------------------------------

def test_initial_calls_per_file_values():
    assert initial_calls_per_file("single") == 1
    assert initial_calls_per_file("triple") == 3
    # Anything not explicitly triple resolves to one call (single default).
    assert initial_calls_per_file("anything-else") == 1


# ---------------------------------------------------------------------------
# Configuration resolution and precedence
# ---------------------------------------------------------------------------

def test_default_resolves_to_single(tmp_path):
    config = load_config(tmp_path)
    assert config["analysis_mode"] == "single"


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, overrides={"analysis_mode": "quadruple"})


def test_json_config_value_used(tmp_path):
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"analysis_mode": "triple"}), encoding="utf-8"
    )
    config = load_config(tmp_path)
    assert config["analysis_mode"] == "triple"


def test_env_overrides_json(tmp_path, monkeypatch):
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"analysis_mode": "single"}), encoding="utf-8"
    )
    monkeypatch.setenv("CODEDOC_ANALYSIS_MODE", "triple")
    config = load_config(tmp_path)
    assert config["analysis_mode"] == "triple"


def test_overrides_take_highest_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDOC_ANALYSIS_MODE", "single")
    config = load_config(tmp_path, overrides={"analysis_mode": "triple"})
    assert config["analysis_mode"] == "triple"


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------

def test_cli_flag_absent_does_not_add_override(monkeypatch):
    from codedoc.cli.cli import main

    captured = {}

    def fake_run_pipeline(root, config_overrides=None):
        captured["config"] = config_overrides
        return {"checked": 0, "failed": 0, "reused": 0,
                "output_dir": "d", "output_files": []}

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)
    main(["run", "--entry", "main.py"])
    assert "analysis_mode" not in captured["config"]


def test_cli_flag_sets_override(monkeypatch):
    from codedoc.cli.cli import main

    captured = {}

    def fake_run_pipeline(root, config_overrides=None):
        captured["config"] = config_overrides
        return {"checked": 0, "failed": 0, "reused": 0,
                "output_dir": "d", "output_files": []}

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)
    main(["run", "--analysis-mode", "triple"])
    assert captured["config"]["analysis_mode"] == "triple"


@pytest.mark.parametrize("alias", ["--single-call", "--three-call"])
def test_unsupported_aliases_rejected(alias):
    from codedoc.cli.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([alias])


def test_cli_only_accepts_single_or_triple():
    from codedoc.cli.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--analysis-mode", "double"])


# ---------------------------------------------------------------------------
# parallel_agents interaction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

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


def test_cli_summaries_show_resolved_mode_and_call_count(capsys):
    from codedoc.cli.cli import _print_dry_run_summary, _print_run_summary

    _print_dry_run_summary(
        {
            "analysis_mode": "triple",
            "initial_calls_per_file": 3,
            "estimate_is_lower_bound": True,
        }
    )
    dry_output = capsys.readouterr().out
    assert "Analysis mode          : triple" in dry_output
    assert "Initial calls per file : 3" in dry_output
    assert "approximate lower bound" in dry_output

    _print_run_summary(
        {
            "checked": 0,
            "failed": 0,
            "output_dir": "out",
            "analysis_mode": "single",
            "initial_calls_per_file": 1,
        }
    )
    run_output = capsys.readouterr().out
    assert "Analysis mode    : single" in run_output
    assert "Initial calls/file: 1" in run_output


def test_single_dry_run_summary_does_not_claim_lower_bound(capsys):
    from codedoc.cli.cli import _print_dry_run_summary

    _print_dry_run_summary(
        {
            "analysis_mode": "single",
            "initial_calls_per_file": 1,
            "estimate_is_lower_bound": False,
        }
    )
    output = capsys.readouterr().out
    assert "(approximate — character heuristic" in output
    assert "approximate lower bound" not in output
