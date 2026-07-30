"""Tests organized by feature ownership."""

from __future__ import annotations

import json

import pytest
from codedoc.cli.cli import run_cli
from codedoc.core.loader import load_config
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_usage import write_py

pytestmark = pytest.mark.future_split_execution


def _write_files(root, count):
    for i in range(count):
        write_py(root / f"f{i}.py")


def test_split_call_cap_counts_every_chunk_and_synthesis_before_side_effects(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text(
        "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n",
        encoding="utf-8",
        newline="",
    )
    no_llm(monkeypatch)
    confirm_calls = []

    with pytest.raises(
        ConfigError, match="leaf documentation.*file reduction.*file synthesis"
    ):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "large_file_strategy": "split",
                "max_content_chars": 2000,
                "max_planned_calls": 1,
                "output_dir": "docs",
            },
            confirm_risky=lambda warnings: confirm_calls.append(warnings) or True,
        )

    assert confirm_calls == []
    assert not (tmp_path / "docs").exists()


def test_cap_failure_touches_no_provider_usage_writer_or_callback(tmp_path, monkeypatch):
    """A max_planned_calls failure occurs before any provider-side effect —
    mirrors the equivalent max_files proof."""
    _write_files(tmp_path, 2)
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter",
        lambda *args, **kwargs: pytest.fail("cap failure initialized SafeWriter"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("cap failure created provider"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.UsageAccumulator",
        lambda *args, **kwargs: pytest.fail("cap failure created a usage accumulator"),
    )
    confirm_calls = []

    with pytest.raises(ConfigError, match="max_planned_calls"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "documentation_scope": "all",
                "max_planned_calls": 1,
                "output_dir": "never-created",
            },
            confirm_risky=lambda warnings: confirm_calls.append(warnings) or True,
        )

    assert confirm_calls == []
    assert not (tmp_path / "never-created").exists()


def test_dry_run_and_real_plan_values_match(tmp_path, monkeypatch):
    """The manifest counts a dry run reports are identical to what the
    corresponding real (unlimited-cap) run reports, since both come from the
    same immutable plan."""
    _write_files(tmp_path, 2)
    no_llm(monkeypatch)

    dry = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "dry_run": True,
        },
    )

    class _Fake:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return json.dumps({"description": "d"})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _Fake())
    real = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
        },
    )

    assert dry["total_calls_planned"] == real["total_calls_planned"] == 2
    assert dry["max_planned_calls"] == real["max_planned_calls"] == 0
    assert dry["max_planned_calls_exceeded"] == real["max_planned_calls_exceeded"] is False
    assert dry["call_manifest_digest"] == real["call_manifest_digest"]


def test_max_files_and_max_planned_calls_are_independent_in_a_real_run(tmp_path, monkeypatch):
    _write_files(tmp_path, 2)
    no_llm(monkeypatch)

    # max_planned_calls is unlimited (0) but max_files is exceeded -> max_files error.
    with pytest.raises(ConfigError, match="max_files"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "documentation_scope": "all",
                "max_files": 1,
                "max_planned_calls": 0,
            },
        )

    # max_files is unlimited (0) but max_planned_calls is exceeded -> the other error.
    with pytest.raises(ConfigError, match="max_planned_calls"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "documentation_scope": "all",
                "max_files": 0,
                "max_planned_calls": 1,
            },
        )


def test_equal_cap_passes_but_one_below_fails(tmp_path, monkeypatch):
    # Two independent projects (not the same tree across two runs) so the
    # second run's plan is not affected by the first run's cached output.
    passing = tmp_path / "passing"
    failing = tmp_path / "failing"
    passing.mkdir()
    failing.mkdir()
    _write_files(passing, 2)
    _write_files(failing, 2)

    class _Fake:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return json.dumps({"description": "d"})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _Fake())
    stats = run_pipeline(
        passing,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "max_planned_calls": 2,
        },
    )
    assert stats["checked"] == 2
    assert stats["max_planned_calls_exceeded"] is False

    no_llm(monkeypatch)
    with pytest.raises(ConfigError, match="max_planned_calls"):
        run_pipeline(
            failing,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "documentation_scope": "all",
                "max_planned_calls": 1,
            },
        )


def test_cli_flag_overrides_config_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDOC_MAX_PLANNED_CALLS", "9")
    config = load_config(tmp_path, {})
    assert config["max_planned_calls"] == 9

    parser_config = load_config(tmp_path, {"max_planned_calls": 3})
    assert parser_config["max_planned_calls"] == 3


def test_cli_exit_code_two_on_cap_violation(tmp_path, monkeypatch, capsys):
    _write_files(tmp_path, 2)
    no_llm(monkeypatch)

    code = run_cli(
        [
            str(tmp_path),
            "--documentation-scope", "all",
            "--max-planned-calls", "1",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "max_planned_calls" in err


def test_reconciliation_fields_present_after_a_completed_run(tmp_path, monkeypatch):
    _write_files(tmp_path, 2)

    class _Fake:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return json.dumps({"description": "d"})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _Fake())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
        },
    )
    assert stats["attempted_logical_calls"] == 2
    assert stats["planned_calls_not_attempted"] == 0
    assert stats["additional_attempts"] == 0
