"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from codedoc.cli.cli import _confirm_risky_prompt_customization, build_parser, run_cli
from pathlib import Path
from codedoc.utils.errors import ConfigError
from codedoc.cli.cli import (
    _print_feasibility_advisories,
    _print_prompt_profile_dry_run,
    _print_prompt_profile_run,
)
from tests.support.feasibility_cases import _cross_file_profile
from tests.support.feasibility_cases import _ReviewFake
from codedoc.core.loader import load_config
import codedoc.core.execution as ex
from codedoc.utils.errors import (
    UnrecoverableProviderError,
)

@pytest.mark.parametrize(
    "argv",
    [
        ["--init-config", "project"],
        ["--init-config", "--format", "md"],
        ["--init-config", "--analysis-mode", "triple"],
        ["--init-config", "--dry-run"],
        ["--force"],
    ],
)
def test_invalid_initializer_combinations_fail_without_writing(
    tmp_path, monkeypatch, argv
):
    monkeypatch.chdir(tmp_path)
    assert run_cli(argv) == 2
    assert not (tmp_path / "codedoc.config.json").exists()

def test_initializer_fixed_target_refusal_and_merge_safe_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_cli(["--init-config"]) == 0
    path = tmp_path / "codedoc.config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["output_format"] = "md"
    data["prompt_profiles"] = None
    path.write_text(json.dumps(data), encoding="utf-8")
    assert run_cli(["--init-config"]) == 2
    assert run_cli(["--init-config", "--force"]) == 0
    refreshed = json.loads(path.read_text(encoding="utf-8"))
    assert refreshed["output_format"] == "md"
    assert "schema_version" not in refreshed["prompt_profiles"]

def test_removed_utilities_are_absent_from_help():
    help_text = build_parser().format_help()
    assert "--describe-prompt-schema" not in help_text
    assert "--init-instructions" not in help_text

def test_cli_confirmation_is_default_no_and_requires_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _confirm_risky_prompt_customization(("warning",)) is False
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    assert _confirm_risky_prompt_customization(("warning",)) is True

def test_cli_run_alias_passes_current_directory_and_overrides(monkeypatch):

    from codedoc.cli.cli import main

    captured = {}

    def fake_run_pipeline(root, config_overrides=None, **_kwargs):
        captured["root"] = root
        captured["config"] = config_overrides
        return {
            "checked": 0,
            "failed": 0,
            "reused": 0,
            "output_dir": "docs_output",
            "output_files": [],
        }

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)

    main(["run", "--format", "md", "--max-parallel-files", "3"])

    # The ``run`` alias passes the current working directory unchanged, so the
    # captured root must equal CWD — not any hard-coded checkout directory name.
    assert Path(captured["root"]).resolve() == Path.cwd().resolve()
    assert captured["config"]["output_format"] == "md"
    assert captured["config"]["max_parallel_files"] == 3

@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (None, 0),
        ("failed", 1),
        ("partial", 0),
        ("config", 2),
        ("output", 1),
        ("fatal", 1),
        ("interrupt", 130),
    ],
)
def test_cli_exit_code_contract(tmp_path, monkeypatch, exception, expected):
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import OutputError

    def fake_pipeline(*args, **kwargs):
        if exception == "config":
            raise ConfigError("bad config")
        if exception == "output":
            raise OutputError("out.json", "write failed")
        if exception == "fatal":
            raise RuntimeError("unexpected")
        if exception == "interrupt":
            raise KeyboardInterrupt()
        return {
            "checked": 0,
            "failed": 1 if exception in {"failed", "partial"} else 0,
            "reused": 0,
            "output_dir": str(tmp_path / "out"),
            "output_files": [],
            "allow_partial": exception == "partial",
        }

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)
    assert run_cli([str(tmp_path)]) == expected

def test_cli_missing_root_and_provider_init_return_two(tmp_path, monkeypatch):
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import ProviderInitError

    assert run_cli([str(tmp_path / "missing")]) == 2
    monkeypatch.setattr(
        "codedoc.pipeline.run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProviderInitError("bad key")),
    )
    assert run_cli([str(tmp_path)]) == 2

def test_run_cli_returns_two_for_invalid_cli_input():
    from codedoc.cli.cli import run_cli

    assert run_cli(["--max-files", "not-an-int"]) == 2

def test_C14_cli_skip_dirs_replaces_defaults():
    """C14: --skip-dirs replaces the default list."""
    from codedoc.cli.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--skip-dirs", "a", "b", "--entry", "main.py"])
    assert args.skip_dirs == ["a", "b"]
    assert args.add_skip_dirs == []
    assert args.remove_skip_dirs == []

def test_C15_cli_add_skip_dir_is_repeatable():
    """C15: --add-skip-dir can be repeated to build a list."""
    from codedoc.cli.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "--add-skip-dir", "generated",
        "--add-skip-dir", "vendor",
        "--entry", "main.py",
    ])
    assert args.add_skip_dirs == ["generated", "vendor"]

def test_C16_cli_remove_skip_dir_is_repeatable():
    """C16: --remove-skip-dir can be repeated to build a list."""
    from codedoc.cli.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "--remove-skip-dir", "codedoc",
        "--remove-skip-dir", "dist",
        "--entry", "main.py",
    ])
    assert args.remove_skip_dirs == ["codedoc", "dist"]

def test_C16_cli_remove_skip_dir_wired_to_overrides(tmp_path, monkeypatch, capsys):
    """C16b: --remove-skip-dir codedoc removes 'codedoc' from resolved skip_dirs."""
    (tmp_path / "main.py").write_text("x=1\n")

    # Capture the resolved skip_dirs by intercepting run_pipeline
    captured = {}

    def fake_run(root, config_overrides=None, **_kwargs):
        cfg = load_config(root, config_overrides)
        captured["skip_dirs"] = cfg["skip_dirs"]
        return {"checked": 0, "failed": 0, "skipped": 0, "reused": 0,
                "output_dir": str(root), "output_files": [],
                "rate_limit_warnings": [], "issues_recorded": 0,
                "error_log": None, "live_backup_path": None}

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run)
    monkeypatch.setattr("codedoc.cli.cli.sys.argv", [
        "codedoc", "run", str(tmp_path),
        "--entry", "main.py",
        "--remove-skip-dir", "codedoc",
    ])

    from codedoc.cli.cli import main
    try:
        main(["run", str(tmp_path), "--entry", "main.py", "--remove-skip-dir", "codedoc"])
    except SystemExit:
        pass

    assert "codedoc" not in captured.get("skip_dirs", ["codedoc"]), (
        "'codedoc' must be removed from skip_dirs via --remove-skip-dir"
    )

@pytest.mark.parametrize("flag", ["--describe-prompt-schema", "--init-instructions"])
def test_removed_config_utilities_are_rejected(flag):
    assert flag not in build_parser().format_help()
    assert run_cli([flag]) == 2

def test_force_without_init_utility_is_rejected():
    assert run_cli(["--force"]) == 2

def test_cli_helper_and_profile_presenters_print_advisory(capsys):
    note = "single/combined/* description: bounded note"
    stats = {
        "prompt_profile_source": "inline",
        "prompt_customization_feasibility_advisories": (note,),
    }

    _print_feasibility_advisories(stats)
    _print_prompt_profile_dry_run(stats)
    _print_prompt_profile_run(stats)

    output = capsys.readouterr().out
    assert output.count("Feasibility advisory (non-blocking):") == 3
    assert output.count(f"- {note}") == 3

def test_cli_helper_ignores_missing_or_non_dict_stats(capsys):
    _print_feasibility_advisories(None)
    _print_feasibility_advisories({})
    assert capsys.readouterr().out == ""

def test_cli_prints_advisory_on_review_block(tmp_path, monkeypatch, capsys):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = _ReviewFake("TOO_RISKY")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    config = {
        "prompt_profiles": _cross_file_profile(),
        "entry_file": "main.py",
    }
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    assert run_cli([str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "Feasibility advisory (non-blocking):" in captured.err
    assert "[different_file: different file]" in captured.err

def test_cli_flag_absent_does_not_add_override(monkeypatch):
    from codedoc.cli.cli import main

    captured = {}

    def fake_run_pipeline(root, config_overrides=None, **_kwargs):
        captured["config"] = config_overrides
        return {"checked": 0, "failed": 0, "reused": 0,
                "output_dir": "d", "output_files": []}

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)
    main(["run", "--entry", "main.py"])
    assert "analysis_mode" not in captured["config"]

def test_cli_flag_sets_override(monkeypatch):
    from codedoc.cli.cli import main

    captured = {}

    def fake_run_pipeline(root, config_overrides=None, **_kwargs):
        captured["config"] = config_overrides
        return {"checked": 0, "failed": 0, "reused": 0,
                "output_dir": "d", "output_files": []}

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)
    main(["run", "--analysis-mode", "triple"])
    assert captured["config"]["analysis_mode"] == "triple"

def test_cli_only_accepts_single_or_triple():
    from codedoc.cli.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--analysis-mode", "double"])

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

def test_cli_prints_recovery_path_when_attached(tmp_path, monkeypatch, capsys):
    import codedoc.pipeline as pipeline_mod

    def raise_with_path(*a, **k):
        exc = KeyboardInterrupt()
        exc.recovery_path = str(tmp_path / "codedoc" / "crash_recovery.json")
        raise exc

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_with_path)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "crash_recovery.json" in err
    assert "left untouched" in err

def test_cli_generic_message_when_no_recovery_path(tmp_path, monkeypatch, capsys):
    import codedoc.pipeline as pipeline_mod

    def raise_plain(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_plain)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "crash-recovery file was created or confirmed" in err
    assert not list(tmp_path.glob("**/crash_recovery.json"))

@pytest.fixture(autouse=True)
def _no_parse(monkeypatch):
    monkeypatch.setattr(ex, "parse_file", lambda descriptor: [])

@pytest.mark.parametrize(
    "category, expected_code",
    [("terminal", 2), ("rate_limit_exhausted", 1)],
)
def test_cli_exit_codes_for_unrecoverable_provider_error(
    tmp_path, monkeypatch, capsys, category, expected_code
):
    from codedoc.cli.cli import run_cli

    def fake_pipeline(*args, **kwargs):
        raise UnrecoverableProviderError("openai", "stopped: doomed run", category)

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == expected_code

    err = capsys.readouterr().err
    # A safe-stop message, NOT the generic crash fallthrough.
    assert "Fatal error:" not in err
    # Resume hint is always printed.
    assert "re-run" in err.lower()
    assert "crash_recovery.json" in err
