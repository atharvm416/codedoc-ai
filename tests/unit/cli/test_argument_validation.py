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
from codedoc.utils.errors import (
    LLMError,
)
from tests.support.logging_sentinels import (
    assert_no_sentinels_leaked,
    sentinel_bearing_exception,
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


def test_cli_help_exposes_the_large_file_split_reuse_and_recovery_boundary():
    help_text = " ".join(build_parser().format_help().split())

    for required in (
        "large-file split execution:",
        "analysis-mode single",
        "paid execution",
        "triple plus split",
        "completed split reuse",
        "node recovery",
        "current schema-4 checkpoints",
        "zero calls",
        "complete source coverage",
        "atom-cap",
        "symbol-cap",
        "unit-cap",
        "chunk-cap",
        "reduction-envelope-cap",
        "reduction-fan-in-cap",
        "reduction-depth-cap",
        "final-synthesis-envelope-cap",
    ):
        assert required in help_text

def test_cli_confirmation_is_default_no_and_requires_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert _confirm_risky_prompt_customization(("warning",)) is False
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    assert _confirm_risky_prompt_customization(("warning",)) is True

def test_cli_large_file_strategy_flag_reaches_run_pipeline(monkeypatch, tmp_path):
    """``--large-file-strategy split`` must arrive as a config override.

    The flag defaults to ``None`` so an unset flag never overrides config or
    environment; only an explicit value is forwarded.
    """
    captured = {}

    def fake_run_pipeline(root, config_overrides=None, **_kwargs):
        captured["config"] = config_overrides
        return {
            "checked": 0,
            "failed": 0,
            "reused": 0,
            "output_dir": "docs",
            "output_files": [],
        }

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)

    assert (
        run_cli(
            [
                str(tmp_path),
                "--dry-run",
                "--large-file-strategy",
                "split",
            ]
        )
        == 0
    )
    assert captured["config"]["large_file_strategy"] == "split"
    assert captured["config"]["dry_run"] is True

    captured.clear()
    assert run_cli([str(tmp_path)]) == 0
    assert "large_file_strategy" not in captured["config"]


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


@pytest.mark.parametrize(
    ("detail", "expected_contact_note"),
    [
        (
            "No documentation call was made, but the prompt-customization "
            "review already ran and was billed.",
            "review already ran and was billed",
        ),
        ("No provider was contacted.", "No provider was contacted"),
    ],
)
def test_cli_preserves_output_error_contact_truth_without_appending_a_claim(
    tmp_path, monkeypatch, capsys, detail, expected_contact_note
):
    from codedoc.utils.errors import OutputError

    def fake_pipeline(*_args, **_kwargs):
        raise OutputError(str(tmp_path / "docs"), detail)

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == 1
    stderr = capsys.readouterr().err
    assert expected_contact_note in stderr
    assert stderr.count("No provider was contacted") == (
        1 if expected_contact_note == "No provider was contacted" else 0
    )
    assert "Choose a writable output directory" in stderr


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:cli-secret@example.com:notaport/v1",
        "ftp://user:cli-secret@example.com/v1",
        "https://[::1/v1?token=cli-secret",
    ],
)
def test_cli_malformed_api_base_url_returns_two_without_leaking_url(
    tmp_path, monkeypatch, capsys, endpoint
):
    (tmp_path / "main.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setenv("API_BASE_URL", endpoint)

    assert run_cli(
        [str(tmp_path), "--entry", "main.py", "--dry-run", "--verbose"]
    ) == 2

    stderr = capsys.readouterr().err
    assert "valid HTTP or HTTPS URL" in stderr
    assert "cli-secret" not in stderr
    assert endpoint not in stderr


def test_cli_persistent_winerror_5_reports_permission_guidance(
    tmp_path, monkeypatch, capsys
):
    """A bounded atomic-replace retry may act on WinError 5, but if it still
    escapes, the CLI must describe a permission problem rather than claiming
    another process temporarily locked the file."""
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import OutputError

    def fake_pipeline(*_args, **_kwargs):
        root = PermissionError("Access is denied")
        root.winerror = 5
        root.errno = 13
        root.strerror = "Access is denied"
        raise OutputError("out.json", "write failed") from root

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == 1
    stderr = capsys.readouterr().err
    assert "Choose a writable output directory" in stderr
    assert "transient file lock" not in stderr

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

def test_split_cli_summary_uses_category_counts_and_synthesis_bound(capsys):
    from codedoc.cli.cli import _print_dry_run_summary

    _print_dry_run_summary(
        {
            "analysis_mode": "single",
            "initial_calls_per_file": 1,
            "large_file_strategy": "split",
            "split_ordinary_files": 1,
            "split_syntax_files": 1,
            "split_lexical_files": 0,
            "split_blocked_files": 1,
            "split_blocked_by_reason": {"chunk-cap": 1},
            "split_blocked_pairs": (("src/huge.py", "chunk-cap"),),
            "split_units": 2,
            "split_chunks": 4,
            "split_continuation_groups": 1,
            "split_unit_consolidation_levels": 1,
            "split_unit_consolidation_calls_planned": 2,
            "split_general_reduction_levels": 0,
            "split_general_reduction_calls_planned": 0,
            "split_final_synthesis_calls_planned": 1,
            "split_restored_complete_chunks": 1,
            "split_restored_unit_consolidation_calls": 0,
            "split_restored_general_reduction_calls": 0,
            "split_restored_final_synthesis_calls": 0,
            "file_documentation_calls_planned": 6,
            "unit_documentation_calls_planned": 9,
            "file_reduction_calls_planned": 2,
            "synthesis_calls_planned": 1,
            "split_synthesis_input_estimate": "upper-bound",
            "estimated_input_tokens": 100,
            "max_planned_calls_exceeded": True,
            "total_calls_planned": 18,
            "max_planned_calls": 15,
        }
    )

    output = capsys.readouterr().out
    assert "Calls per under-threshold file: 1" in output
    assert "Initial calls per file" not in output
    assert (
        "Planned call categories: 6 file / 9 leaf / 2 reduction / 1 synthesis"
        in output
    )
    assert "Synthesis input estimate: upper-bound from configured ceiling" in output
    assert "Blocked path/reason pairs:" in output
    assert "src/huge.py (chunk-cap)" in output
    assert (
        "6 file documentation, 9 leaf documentation, 2 file reduction, "
        "1 file synthesis" in output
    )
    assert "approximate mixed bound; synthesis uses the configured ceiling" in output


def test_cli_prints_split_complexity_advisory_only_when_present(capsys):
    from codedoc.cli.cli import _print_dry_run_summary

    base_stats = {
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "split_ordinary_files": 0,
        "split_syntax_files": 1,
        "split_lexical_files": 0,
        "split_blocked_files": 0,
        "split_blocked_by_reason": {},
        "split_units": 1,
        "split_chunks": 30,
        "split_continuation_groups": 1,
        "split_unit_consolidation_levels": 0,
        "split_unit_consolidation_calls_planned": 0,
        "split_general_reduction_levels": 4,
        "split_general_reduction_calls_planned": 14,
        "split_final_synthesis_calls_planned": 1,
        "split_restored_complete_chunks": 0,
        "split_restored_unit_consolidation_calls": 0,
        "split_restored_general_reduction_calls": 0,
        "split_restored_final_synthesis_calls": 0,
        "file_documentation_calls_planned": 0,
        "unit_documentation_calls_planned": 30,
        "file_reduction_calls_planned": 14,
        "synthesis_calls_planned": 1,
        "split_synthesis_input_estimate": "deterministic-worst-case-envelope",
        "estimated_input_tokens": 100,
    }

    _print_dry_run_summary(
        {**base_stats, "split_complexity_advisory": "A higher-capability model may help."}
    )
    with_advisory = capsys.readouterr().out
    assert "Advisory (non-blocking): A higher-capability model may help." in with_advisory

    _print_dry_run_summary({**base_stats, "split_complexity_advisory": None})
    without_advisory = capsys.readouterr().out
    assert "Advisory (non-blocking):" not in without_advisory


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
    assert "completed ordinary and split records may be reused" in err
    assert "compatible current schema-4 split node checkpoints may resume" in err

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
    assert "completed ordinary and split records may be reused" in err
    assert "compatible current schema-4 split node checkpoints may resume" in err
    assert not list(tmp_path.glob("**/crash_recovery.json"))

@pytest.mark.parametrize(
    "category, expected_code",
    [("terminal", 2), ("rate_limit_exhausted", 1)],
)
def test_cli_exit_codes_for_unrecoverable_provider_error(
    tmp_path, monkeypatch, capsys, category, expected_code
):
    from codedoc.cli.cli import run_cli
    from codedoc.core.error_classifier import (
        _build_rate_limit_exhausted_abort,
        _build_terminal_abort,
    )

    def fake_pipeline(*args, **kwargs):
        if category == "terminal":
            raise _build_terminal_abort(
                LLMError("openai", "insufficient_quota"),
                "openai",
                "terminal_billing",
            )
        raise _build_rate_limit_exhausted_abort("openai")

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == expected_code

    err = capsys.readouterr().err
    # A safe-stop message, NOT the generic crash fallthrough.
    assert "Fatal error:" not in err
    # Resume hint is always printed.
    assert "re-run" in err.lower()
    assert "crash_recovery.json" in err
    assert "completed ordinary and split records may be reused" in err
    assert "compatible current schema-4 split node checkpoints may resume" in err
    assert "resumes the unfinished files" not in err


def test_verbose_bounded_trace_never_renders_foreign_type_or_message() -> None:
    from codedoc.cli.cli import _bounded_traceback

    try:
        raise sentinel_bearing_exception("foreign-provider-exception")
    except RuntimeError as cause:
        try:
            raise ConfigError("bounded outer reason") from cause
        except ConfigError as outer:
            rendered = _bounded_traceback(outer)

    assert "Bounded diagnostic trace" in rendered
    assert "ConfigError" in rendered
    assert "unknown-error" in rendered
    assert "RuntimeError" not in rendered
    assert "foreign-provider-exception" not in rendered
    assert_no_sentinels_leaked(rendered)


def test_cli_locked_output_explains_fresh_split_recovery_boundary(
    tmp_path, monkeypatch, capsys
):
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import OutputError

    def fake_pipeline(*_args, **_kwargs):
        root = PermissionError("The process cannot access the file")
        root.winerror = 32
        root.errno = 13
        raise OutputError("out.json", "atomic replace failed") from root

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "transient file lock" in err
    assert "Any crash-recovery file already created" in err
    assert "can also occur before one is created" in err
    assert "Completed work is preserved" not in err
    assert "completed ordinary and split records may be reused" in err
    assert "compatible current schema-4 split node checkpoints may resume" in err


def test_cli_non_lock_output_explains_fresh_split_recovery_boundary(
    tmp_path, monkeypatch, capsys
):
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import OutputError

    def fake_pipeline(*_args, **_kwargs):
        raise OutputError("out.json", "permission denied") from PermissionError(
            "permission denied"
        )

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)

    assert run_cli([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "Choose a writable output directory" in err
    assert "failure can also occur before one exists" in err
    assert "completed ordinary and split records may be reused" in err
    assert "compatible current schema-4 split node checkpoints may resume" in err
