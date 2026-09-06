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
from tests.support.logging_sentinels import (
    assert_no_sentinels_leaked,
    sentinel_bearing_exception,
)
from tests.support.provider_failures import provider_failure_error

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
        "in-progress split checkpoints",
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
    # The retired uncapped split_blocked_pairs presenter is gone: even when a
    # retired split_blocked_pairs stat is injected, NO path-bearing category is
    # rendered. (Part 2 restores a bounded, JSON-escaped `split_blocked`.)
    assert "Blocked path/reason pairs:" not in output
    assert "src/huge.py" not in output
    # The path-free split_blocked_by_reason aggregate is still rendered.
    assert "Blocked reasons         : chunk-cap=1" in output
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
    assert "compatible in-progress split checkpoints may resume" in err

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
    assert "compatible in-progress split checkpoints may resume" in err
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
                provider_failure_error("openai", "provider-quota-exhausted", status=429),
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
    assert "compatible in-progress split checkpoints may resume" in err
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
    assert "compatible in-progress split checkpoints may resume" in err


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
    assert "compatible in-progress split checkpoints may resume" in err


# ===========================================================================
# Section 9 Part 1: --max-content-chars CLI surface, non-misleading billing
# labels, absence of the retired split_blocked_pairs loop, and public-only
# help. (Plan sections 5.8 / 5.9 / 7.1 / 7.2.)
# ===========================================================================


def _capture_overrides(monkeypatch):
    """Patch run_pipeline to capture the config_overrides run_cli builds."""
    captured = {}

    def fake_run_pipeline(root, config_overrides=None, **_kwargs):
        captured["config"] = config_overrides
        return {
            "checked": 0, "failed": 0, "reused": 0,
            "output_dir": "docs", "output_files": [],
        }

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)
    return captured


def test_cli_max_content_chars_flag_reaches_resolved_config(monkeypatch, tmp_path):
    """--max-content-chars N is forwarded as a config override and resolves."""
    captured = _capture_overrides(monkeypatch)
    assert run_cli([str(tmp_path), "--max-content-chars", "3000"]) == 0
    assert captured["config"]["max_content_chars"] == 3000
    resolved = load_config(tmp_path, captured["config"])
    assert resolved["max_content_chars"] == 3000

    # Unset flag never overrides config or environment.
    captured.clear()
    assert run_cli([str(tmp_path)]) == 0
    assert "max_content_chars" not in captured["config"]


def test_cli_max_content_chars_rejects_non_integer_and_below_minimum(
    tmp_path, capsys
):
    """Non-integers are rejected at parse time; a value below 1000 is a
    classified ConfigError (exit 2), never a raw traceback. Neither path
    reaches a provider (an empty project would otherwise error on no files)."""
    # Non-integer: argparse rejects before any pipeline work.
    assert run_cli([str(tmp_path), "--max-content-chars", "not-an-int"]) == 2
    capsys.readouterr()

    # Below the minimum: load_config raises a classified ConfigError.
    assert run_cli([str(tmp_path), "--max-content-chars", "999"]) == 2
    err = capsys.readouterr().err
    assert "max_content_chars must be at least 1000" in err
    assert "Traceback (most recent call last)" not in err


def test_cli_max_content_chars_precedence_cli_over_env_over_config_over_default(
    monkeypatch, tmp_path
):
    """Resolution order: CLI override > CODEDOC_MAX_CONTENT_CHARS > config file
    > default (12000)."""
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"max_content_chars": 7000}), encoding="utf-8"
    )

    monkeypatch.delenv("CODEDOC_MAX_CONTENT_CHARS", raising=False)
    assert load_config(tmp_path, {})["max_content_chars"] == 7000        # config file

    monkeypatch.setenv("CODEDOC_MAX_CONTENT_CHARS", "5000")
    assert load_config(tmp_path, {})["max_content_chars"] == 5000        # env beats file

    assert (
        load_config(tmp_path, {"max_content_chars": 3000})["max_content_chars"] == 3000
    )                                                                    # CLI beats env

    monkeypatch.delenv("CODEDOC_MAX_CONTENT_CHARS", raising=False)
    (tmp_path / "codedoc.config.json").unlink()
    assert load_config(tmp_path, {})["max_content_chars"] == 12000       # default


def test_cli_max_content_chars_is_rejected_with_init_config(
    tmp_path, monkeypatch, capsys
):
    """--max-content-chars is a documentation-run option: a *valid* value
    combined with --init-config is rejected by the run-only-option guard (not
    merely as an unrecognized argument), and no config is written."""
    monkeypatch.chdir(tmp_path)
    assert run_cli(["--init-config", "--max-content-chars", "2000"]) == 2
    err = capsys.readouterr().err
    assert "--init-config can be combined only with --force" in err
    assert "unrecognized arguments" not in err
    assert not (tmp_path / "codedoc.config.json").exists()


def test_cli_adds_no_balance_or_tolerance_knob():
    """Section 5.8 line 1133: no new balance / tolerance / fan-in / overlap /
    reducer setting was introduced alongside --max-content-chars."""
    parser = build_parser()
    option_strings = {
        opt for action in parser._actions for opt in action.option_strings
    }
    assert "--max-content-chars" in option_strings
    for forbidden in (
        "--balance", "--balance-window", "--tolerance", "--min-chunk-chars",
        "--fan-in", "--reducer-fan-in", "--overlap", "--chunk-overlap",
        "--reducer-calls", "--min-content-chars", "--chunk-size",
    ):
        assert forbidden not in option_strings
    help_text = " ".join(parser.format_help().split())
    for word in ("balance window", "tolerance", "fan-in knob", "overlap"):
        assert word not in help_text


def test_cli_split_blocked_pairs_presenter_loop_is_absent(capsys):
    """Feeding a retired split_blocked_pairs stat produces NO path output; the
    uncapped raw loop is gone (section 5.8 lines 1182-1187; section 12)."""
    from codedoc.cli.cli import _print_dry_run_summary

    _print_dry_run_summary(
        {
            "analysis_mode": "single",
            "initial_calls_per_file": 1,
            "large_file_strategy": "split",
            "split_blocked_files": 1,
            "split_blocked_by_reason": {"chunk-cap": 1},
            "split_blocked_pairs": (
                ("src/huge.py", "chunk-cap"),
                ("src/second\ninjected line", "unit-cap"),
            ),
            "estimated_input_tokens": 10,
        }
    )
    output = capsys.readouterr().out
    assert "Blocked path/reason pairs:" not in output
    assert "src/huge.py" not in output
    assert "src/second" not in output
    assert "injected line" not in output


def test_cli_split_blocked_by_reason_aggregate_is_still_rendered(capsys):
    """The closed, path-free split_blocked_by_reason aggregate is preserved
    (section 5.8 lines 1184-1186)."""
    from codedoc.cli.cli import _print_dry_run_summary

    _print_dry_run_summary(
        {
            "analysis_mode": "single",
            "initial_calls_per_file": 1,
            "large_file_strategy": "split",
            "split_blocked_files": 2,
            "split_blocked_by_reason": {"chunk-cap": 1, "atom-cap": 1},
            "estimated_input_tokens": 10,
        }
    )
    output = capsys.readouterr().out
    assert "Blocked reasons         : atom-cap=1, chunk-cap=1" in output


def test_cli_billing_labels_are_not_misleading(capsys):
    """Section 5.9: the headline is provider_calls_max_before_retries; the two
    legacy documentation-only keys still appear but are never labelled total or
    worst-case."""
    from codedoc.cli.cli import _print_dry_run_summary, _print_prompt_profile_dry_run

    _print_dry_run_summary(
        {
            "analysis_mode": "single",
            "initial_calls_per_file": 1,
            "would_call_llm_for": 1,
            "estimated_calls": 1,
            "estimated_calls_max_with_correction": 2,
            "response_correction_enabled": True,
            "response_correction_calls_possible_max": 1,
            "provider_calls_max_before_retries": 2,
            "estimated_input_tokens": 10,
        }
    )
    out = capsys.readouterr().out
    assert "Worst-case LLM calls" not in out
    assert "Total paid calls" not in out
    # Headline value is provider_calls_max_before_retries.
    assert "Provider calls before retries : 2" in out
    assert "exact initial manifest" in out
    assert "transport and file retries are additional" in out
    # Both legacy keys still shown, documentation-scoped, not total/worst-case.
    assert "Estimated documentation calls : 1" in out
    assert "Estimated documentation calls, with correction : 2" in out

    _print_prompt_profile_dry_run(
        {
            "prompt_profile_source": "inline",
            "prompt_profile_active": True,
            "prompt_profile_affected_files": 1,
            "documentation_calls_planned": 1,
            "prompt_customization_security_review_calls_planned": 1,
        }
    )
    profile_out = capsys.readouterr().out
    assert "Total paid calls" not in profile_out
    assert "Initial provider calls: 2 planned" in profile_out


def test_cli_help_has_no_internal_schema_vocabulary():
    """Section 5.8 lines 1477-1478: user-facing help names no internal recovery
    schema generation."""
    help_text = build_parser().format_help()
    assert "schema-4" not in help_text
    assert "schema_4" not in help_text
    assert "schema-3" not in help_text
    # The behaviour is still described in user terms.
    assert "in-progress split checkpoint" in " ".join(help_text.split())


def test_cli_run_help_names_source_and_synthesis_ceilings_separately():
    """Section 7.1 lines 1846-1848: run help no longer claims max_content_chars
    bounds reducer/final manifests; it names the source ceiling and the
    automatic split synthesis ceiling separately."""
    help_text = " ".join(build_parser().format_help().split())
    assert "max_content_chars is the ordinary/leaf source ceiling" in help_text
    assert "separate automatic split synthesis ceiling" in help_text
    assert "12,000 characters" in help_text
    assert "max_content_chars bounds each planned leaf, reducer manifest" not in help_text


# ===========================================================================
# Section 9 Part 2: the real-run preflight reporter. The CLI hands
# run_pipeline() a plan_reporter that renders the pipeline's ONE immutable
# snapshot before any provider is constructed. The presenter is a pure read of
# that snapshot (section 5.8 lines 1365-1380 / section 5.9 lines 1482-1517).
# ===========================================================================


class _ProviderSentinel(Exception):
    pass


def _fake_provider(monkeypatch):
    from tests.support.providers import SmartFake

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: SmartFake()
    )


def _no_provider(monkeypatch):
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: (_ for _ in ()).throw(
            _ProviderSentinel("provider must not be constructed")
        ),
    )


def _flat_snapshot(prefix, records, *, total=None, retained=None, omitted=None):
    """Minimal synthetic preflight snapshot exercising one detail category.
    Only the fields the presenter reads are populated."""
    total = len(records) if total is None else total
    retained = len(records) if retained is None else retained
    omitted = (total - retained) if omitted is None else omitted
    return {
        "provider_calls_max_before_retries": 0,
        "initial_provider_calls_planned": 0,
        "prompt_review_calls_planned": 0,
        "initial_documentation_calls_planned": 0,
        "file_retry_attempts": 0,
        "estimated_calls": 0,
        "large_file_strategy_resolved": "split",
        "large_file_source_ceiling_chars": 2000,
        "split_internal_manifest_budget_chars": 12000,
        f"{prefix}_details": records,
        f"{prefix}_details_total": total,
        f"{prefix}_details_retained": retained,
        f"{prefix}_details_omitted": omitted,
        f"{prefix}_details_digest": "sha256:" + "e" * 64,
    }


# --- 1. Ordering: summary before provider construction ----------------------

def test_preflight_summary_prints_before_provider_construction(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    # This test checks the reporter ORDERING and one headline value, not the
    # correction default; pin it off so the headline stays 1 (Section 10).
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"response_correction_enabled": False}), encoding="utf-8"
    )
    _no_provider(monkeypatch)

    rc = run_cli([str(tmp_path), "--entry", "main.py"])

    out = capsys.readouterr().out
    # The summary is in stdout even though create_provider raised: the pipeline
    # invoked the reporter before constructing the provider.
    assert "Planned provider work (before calls)" in out
    assert "Provider calls before retries : 1" in out
    assert rc == 1  # the sentinel is a generic fatal error


# --- 2. Exactly once, and never on --dry-run -------------------------------

def test_preflight_reporter_fires_once_and_not_on_dry_run(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    _fake_provider(monkeypatch)
    import codedoc.cli.cli as cli_mod

    calls = {"n": 0}
    real = cli_mod._print_preflight_summary
    monkeypatch.setattr(
        cli_mod,
        "_print_preflight_summary",
        lambda snap, **kw: (calls.__setitem__("n", calls["n"] + 1), real(snap, **kw))[1],
    )

    assert run_cli([str(tmp_path), "--entry", "main.py"]) == 0
    assert calls["n"] == 1

    calls["n"] = 0
    assert run_cli([str(tmp_path), "--entry", "main.py", "--dry-run"]) == 0
    assert calls["n"] == 0  # dry-run keeps its own review-and-exit summary


# --- 3. The presenter does not re-plan; it echoes snapshot fields verbatim --

def test_preflight_presenter_does_not_re_plan_and_echoes_snapshot_verbatim(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    _fake_provider(monkeypatch)
    import codedoc.cli.cli as cli_mod
    import codedoc.pipeline as pipeline_mod

    real_bpp = pipeline_mod.build_pipeline_plan
    plan_calls = {"n": 0}

    def _counting(*a, **k):
        plan_calls["n"] += 1
        return real_bpp(*a, **k)

    monkeypatch.setattr(pipeline_mod, "build_pipeline_plan", _counting)

    seen = {}
    real = cli_mod._print_preflight_summary
    monkeypatch.setattr(
        cli_mod,
        "_print_preflight_summary",
        lambda snap, **kw: (seen.__setitem__("snap", dict(snap)), real(snap, **kw))[1],
    )

    assert run_cli([str(tmp_path), "--entry", "main.py"]) == 0
    out = capsys.readouterr().out

    assert plan_calls["n"] == 1  # planned once; the presenter triggers no second plan
    snap = seen["snap"]
    assert (
        f"Provider calls before retries : {snap['provider_calls_max_before_retries']}"
        in out
    )
    assert (
        f"Source / synthesis ceiling    : {snap['large_file_source_ceiling_chars']} / "
        f"{snap['split_internal_manifest_budget_chars']} chars"
        in out
    )
    assert (
        f"Initial provider calls        : {snap['initial_provider_calls_planned']}"
        in out
    )


# --- 4. Both ceilings, distinct --------------------------------------------

def test_preflight_names_both_ceilings_distinctly(tmp_path, monkeypatch, capsys):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    _fake_provider(monkeypatch)

    rc = run_cli(
        [
            str(tmp_path),
            "--entry", "main.py",
            "--large-file-strategy", "split",
            "--max-content-chars", "1000",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Source / synthesis ceiling    : 1000 / 12000 chars" in out


# --- 5. Default form caps at 20 and states the omitted count --------------

def test_preflight_default_form_caps_at_twenty_records(capsys):
    from codedoc.cli.cli import _PREFLIGHT_DISPLAY_CAP, _print_preflight_summary

    records = [
        {
            "path": f"src/blocked_{i:03d}.py",
            "reason": "chunk-cap",
            "phase": "division-packing",
            "observed": 300 + i,
            "limit": 256,
            "guidance_code": "raise-source-ceiling-or-split-source",
        }
        for i in range(25)
    ]
    _print_preflight_summary(_flat_snapshot("split_blocked", records), verbose=False)
    out = capsys.readouterr().out
    shown = [ln for ln in out.splitlines() if "src/blocked_" in ln]
    assert len(shown) == _PREFLIGHT_DISPLAY_CAP == 20
    assert "5 more not shown (display cap 20; use --verbose)" in out
    assert "sha256:" + "e" * 64 in out


# --- 6. --verbose shows all retained + explicit cap disclosure -----------

def test_preflight_verbose_shows_all_retained_and_discloses_snapshot_cap(capsys):
    from codedoc.cli.cli import _print_preflight_summary

    records = [
        {
            "path": f"src/t_{i:03d}.py",
            "source_chars": 9000 + i,
            "retained_head_chars": 700,
            "retained_tail_chars": 300,
            "omitted_chars": 8000 + i,
            "initial_calls": 1,
        }
        for i in range(25)
    ]

    # No snapshot-cap loss -> verbose is complete, no omitted notice.
    _print_preflight_summary(
        _flat_snapshot("truncate_plan", records, omitted=0), verbose=True
    )
    out = capsys.readouterr().out
    assert len([ln for ln in out.splitlines() if "src/t_" in ln]) == 25
    assert "snapshot cap" not in out
    assert "not shown" not in out

    # The snapshot's 4096-item cap dropped rows -> verbose discloses it + digest.
    _print_preflight_summary(
        _flat_snapshot("truncate_plan", records, total=5000, retained=25, omitted=4975),
        verbose=True,
    )
    out2 = capsys.readouterr().out
    assert len([ln for ln in out2.splitlines() if "src/t_" in ln]) == 25
    assert "4975 dropped by the 4096-item snapshot cap" in out2
    assert "sha256:" + "e" * 64 in out2


# --- 7. split_blocked guidance-code prose ---------------------------------

def test_preflight_renders_guidance_prose_for_every_closed_code(capsys):
    """The closed guidance vocabulary is nine codes: the five capacity /
    scanner-byte codes (section 5.8 lines 1294-1304) and the four
    scanner-admission codes (section 5.8 lines 1440-1445). Coverage here is
    exhaustive BY CONSTRUCTION -- the admission codes are read from production,
    never hard-coded -- so a future addition to either set fails this test
    loudly instead of silently rendering a raw kebab-case code.
    """
    from codedoc.cli.cli import _GUIDANCE_PROSE, _print_preflight_summary
    from codedoc.core.scanner import _ADMISSION_REASON_GUIDANCE

    # (1) Admission codes come from production, not a literal list.
    admission_codes = set(_ADMISSION_REASON_GUIDANCE.values())
    # (2) _GUIDANCE_PROSE is exactly the union of both frozen sets.
    capacity_byte_codes = {
        "simplify-or-exclude",
        "raise-source-ceiling-or-split-source",
        "report-planning-capacity-defect",
        "inspect-authoritative-metadata-or-exclude",
        "raise-scan-byte-limit-or-exclude",
    }
    assert set(_GUIDANCE_PROSE) == capacity_byte_codes | admission_codes

    # (3a) Capacity codes through the split_blocked category (its real shape).
    blocked = [
        ("atom-cap", "division-structure", "simplify-or-exclude"),
        ("chunk-cap", "division-packing", "raise-source-ceiling-or-split-source"),
        ("reduction-depth-cap", "reduction-depth", "report-planning-capacity-defect"),
        (
            "final-synthesis-envelope-cap",
            "final-synthesis",
            "inspect-authoritative-metadata-or-exclude",
        ),
    ]
    blocked_records = [
        {
            "path": f"src/b_{reason}.py",
            "reason": reason,
            "phase": phase,
            "observed": 999,
            "limit": 256,
            "guidance_code": code,
        }
        for reason, phase, code in blocked
    ]
    _print_preflight_summary(
        _flat_snapshot("split_blocked", blocked_records), verbose=True
    )
    blocked_out = capsys.readouterr().out
    for reason, _phase, code in blocked:
        assert _GUIDANCE_PROSE[code] in blocked_out
        assert code not in blocked_out  # never the raw kebab-case code alone
        assert json.dumps(f"src/b_{reason}.py", ensure_ascii=True) in blocked_out

    # (3b) The scanner-byte code through the scanner_size_skip category.
    _print_preflight_summary(
        _flat_snapshot(
            "scanner_size_skip",
            [
                {
                    "path": "src/huge.bin",
                    "phase": "scanner-byte",
                    "observed": 900000,
                    "limit": 512000,
                    "guidance_code": "raise-scan-byte-limit-or-exclude",
                }
            ],
        ),
        verbose=True,
    )
    size_out = capsys.readouterr().out
    assert _GUIDANCE_PROSE["raise-scan-byte-limit-or-exclude"] in size_out
    assert "raise-scan-byte-limit-or-exclude" not in size_out

    # (3c) EVERY admission code through the scanner_admission_skip category
    # (descriptor shape: path, phase="scanner-admission", reason, guidance_code).
    admission_records = [
        {
            "path": f"src/a_{reason}.py",
            "phase": "scanner-admission",
            "reason": reason,
            "guidance_code": code,
        }
        for reason, code in sorted(_ADMISSION_REASON_GUIDANCE.items())
    ]
    _print_preflight_summary(
        _flat_snapshot("scanner_admission_skip", admission_records), verbose=True
    )
    admission_out = capsys.readouterr().out
    for reason, code in _ADMISSION_REASON_GUIDANCE.items():
        assert _GUIDANCE_PROSE[code] in admission_out
        assert code not in admission_out  # (4) raw code never rendered on its own
        assert (
            json.dumps(f"src/a_{reason}.py", ensure_ascii=True) in admission_out
        )


# --- 8. Terminal-injection matrix for every rendered category ------------

_HOSTILE = "pkg/ev\nil\ta\x1b‮\"x\\y.py"

_INJECTION_CASES = (
    (
        "split_plan",
        {
            "path": _HOSTILE,
            "source_chars": 5000,
            "structural_mode": "syntax",
            "source_ceiling_chars": 2000,
            "synthesis_manifest_ceiling_chars": 12000,
            "reduction_levels": 1,
            "reduction_calls": 1,
            "final_calls": 1,
            "initial_calls": 3,
            "units": [],
            "leaves": [],
        },
    ),
    (
        "truncate_plan",
        {
            "path": _HOSTILE,
            "source_chars": 9000,
            "retained_head_chars": 700,
            "retained_tail_chars": 300,
            "omitted_chars": 8000,
            "initial_calls": 1,
        },
    ),
    (
        "split_blocked",
        {
            "path": _HOSTILE,
            "reason": "chunk-cap",
            "phase": "division-packing",
            "observed": 300,
            "limit": 256,
            "guidance_code": "raise-source-ceiling-or-split-source",
        },
    ),
    (
        "scanner_size_skip",
        {
            "path": _HOSTILE,
            "phase": "scanner-byte",
            "observed": 900000,
            "limit": 512000,
            "guidance_code": "raise-scan-byte-limit-or-exclude",
        },
    ),
    (
        "scanner_admission_skip",
        {
            "path": _HOSTILE,
            "phase": "scanner-admission",
            "reason": "unsupported",
            "guidance_code": "raise-scan-byte-limit-or-exclude",
        },
    ),
)


@pytest.mark.parametrize("prefix,record", _INJECTION_CASES)
def test_preflight_paths_render_as_one_escaped_json_string(capsys, prefix, record):
    from codedoc.cli.cli import _print_preflight_summary

    _print_preflight_summary(_flat_snapshot(prefix, [record]), verbose=True)
    out = capsys.readouterr().out

    encoded = json.dumps(_HOSTILE, ensure_ascii=True)
    assert encoded in out
    # The raw path, with its real control characters, was never interpolated.
    assert _HOSTILE not in out
    for raw_char in ("\t", "\x1b", "‮"):
        assert raw_char not in out
    # Exactly one output line carries the encoded path -- no injected line.
    carriers = [ln for ln in out.splitlines() if encoded in ln]
    assert len(carriers) == 1


# --- 9. Report-before-error on cap-exceeded and division-blocked --------

def test_preflight_reported_before_max_files_error(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    _no_provider(monkeypatch)

    rc = run_cli([str(tmp_path), "--documentation-scope", "all", "--max-files", "1"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "Planned provider work (before calls)" in out
    assert "exceed --max-files 1" in out  # WARNING printed before the ConfigError


def test_preflight_reported_before_division_blocked_error(
    tmp_path, monkeypatch, capsys
):
    # One 300k-char assignment line: a single semantic unit needing ~300
    # continuation pieces at a 1000-char ceiling -> chunk-cap capacity block.
    (tmp_path / "main.py").write_text(
        "x = " + "1" * 300_000 + "\n", encoding="utf-8"
    )
    _no_provider(monkeypatch)

    rc = run_cli(
        [
            str(tmp_path),
            "--entry", "main.py",
            "--large-file-strategy", "split",
            "--max-content-chars", "1000",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 2
    assert "Planned provider work (before calls)" in out
    assert "Capacity-blocked files (1):" in out
    from codedoc.cli.cli import _GUIDANCE_PROSE

    assert _GUIDANCE_PROSE["raise-source-ceiling-or-split-source"] in out


# --- 10. Section 5.9 billing presenter regressions through the CLI ------

@pytest.mark.parametrize("with_profile, before_retries", [(False, 2), (True, 3)])
def test_preflight_billing_headline_is_before_retries_value(
    tmp_path, monkeypatch, capsys, with_profile, before_retries
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = {"entry_file": "main.py", "response_correction_enabled": True}
    if with_profile:
        config["prompt_profiles"] = _cross_file_profile()
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider", lambda _config: _ReviewFake("SAFE")
        )
    else:
        _fake_provider(monkeypatch)
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    assert run_cli([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    preflight = out.split("Planned provider work (before calls)", 1)[1].split(
        "\ncodedoc complete.", 1
    )[0]
    assert f"Provider calls before retries : {before_retries} " in preflight
    assert "Possible correction calls     : 1 " in preflight
    assert "Worst-case LLM calls" not in out
    assert "Total paid calls" not in out


# --- 11. F2/F3 regressions: capped-piece disclosure and payable-vs-topology --

def _split_plan_snapshot(records, *, files_total=None):
    """Minimal synthetic preflight snapshot exercising the nested
    ``split_plan`` category. Only the fields the presenter reads are
    populated; ``large_file_strategy_resolved`` must be set or
    ``_print_preflight_summary`` skips the whole strategy block."""
    files_total = len(records) if files_total is None else files_total
    return {
        "large_file_strategy_resolved": "split",
        "split_plan_details": records,
        "large_files_routed_split": files_total,
        "split_plan_details_omitted": 0,
        "split_plan_details_digest": "sha256:" + "f" * 64,
    }


def _split_plan_record(**overrides):
    record = {
        "path": "main.py",
        "source_chars": 2010,
        "structural_mode": "syntax",
        "source_ceiling_chars": 1000,
        "synthesis_manifest_ceiling_chars": 12000,
        "initial_calls": 5,
        "reduction_calls": 1,
        "final_calls": 1,
        "units": (),
        "leaves": (),
    }
    record.update(overrides)
    return record


def _split_unit(**overrides):
    unit = {
        "unit_ordinal": 0,
        "natural_source_chars": 2010,
        "pieces": (),
        "pieces_total": 0,
        "pieces_retained": 0,
        "pieces_omitted": 0,
        "pieces_digest": "sha256:" + "0" * 64,
        "arithmetic_piece_count": 1,
        "crlf_safe_piece_count": 1,
        "atomicity_extra_piece_count": 0,
    }
    unit.update(overrides)
    return unit


def test_verbose_split_plan_preserves_the_pinned_fully_retained_descriptor(capsys):
    """F2 control case: a fully retained (uncapped) subdivided unit must
    still render exactly the plan-pinned `2010 -> 670 + 670 + 670` string
    (section 5.8 line 1311) -- proving the fix does not alter the common,
    uncapped case."""
    from codedoc.cli.cli import _print_preflight_summary

    unit = _split_unit(
        pieces=(
            {"payload_chars": 670},
            {"payload_chars": 670},
            {"payload_chars": 670},
        ),
        pieces_total=3, pieces_retained=3, pieces_omitted=0,
        arithmetic_piece_count=3, crlf_safe_piece_count=3,
    )
    record = _split_plan_record(units=(unit,))
    _print_preflight_summary(_split_plan_snapshot([record]), verbose=True)
    out = capsys.readouterr().out
    assert "2010 -> 670 + 670 + 670" in out
    assert "not subdivided" not in out


def test_verbose_split_plan_shows_not_subdivided_only_for_a_genuine_single_unit(
    capsys,
):
    """F2: ``pieces_total == 0`` -- not an empty retained list -- is the
    correct test for "this unit was never subdivided", verified directly
    from ``file_division.py``'s own piece-counting instrumentation: a
    fitting unit's branch in ``_iter_split_unit_items`` never yields a
    "piece" tuple at all, so its ``pieces_total`` can only be 0."""
    from codedoc.cli.cli import _print_preflight_summary

    unit = _split_unit(natural_source_chars=500)
    record = _split_plan_record(units=(unit,))
    _print_preflight_summary(_split_plan_snapshot([record]), verbose=True)
    out = capsys.readouterr().out
    assert "500 -> (not subdivided)" in out


def test_verbose_split_plan_discloses_a_fully_capped_unit_instead_of_claiming_not_subdivided(
    capsys,
):
    """F2 case 1 (the audited defect): ``pieces_retained == 0`` with
    ``pieces_total > 0`` -- a genuinely subdivided unit whose every piece was
    dropped by the 4096-item snapshot cap. Reproduced by the audit with 1025
    genuine plans: ``pieces_total=3, pieces_retained=0, pieces_omitted=3``.
    The presenter must not print "(not subdivided)" for this unit."""
    from codedoc.cli.cli import _print_preflight_summary

    unit = _split_unit(
        pieces_total=3, pieces_retained=0, pieces_omitted=3,
        pieces_digest="sha256:" + "b" * 64,
        arithmetic_piece_count=3, crlf_safe_piece_count=3,
    )
    record = _split_plan_record(units=(unit,))
    _print_preflight_summary(_split_plan_snapshot([record]), verbose=True)
    out = capsys.readouterr().out
    assert "not subdivided" not in out
    assert "3 of 3 piece(s) omitted" in out
    assert "b" * 64 in out


def test_verbose_split_plan_discloses_a_partially_capped_unit_instead_of_understating_it(
    capsys,
):
    """F2 case 2 (the audited defect, second half): ``0 < pieces_retained <
    pieces_total`` prints a complete-looking ``670 + 670`` descriptor that
    silently drops a real third piece -- equally false, and a global
    omission notice elsewhere in the output does not repair a per-unit
    claim. The disclosure must be local to this unit."""
    from codedoc.cli.cli import _print_preflight_summary

    unit = _split_unit(
        pieces=({"payload_chars": 670}, {"payload_chars": 670}),
        pieces_total=3, pieces_retained=2, pieces_omitted=1,
        pieces_digest="sha256:" + "c" * 64,
        arithmetic_piece_count=3, crlf_safe_piece_count=3,
    )
    record = _split_plan_record(units=(unit,))
    _print_preflight_summary(_split_plan_snapshot([record]), verbose=True)
    out = capsys.readouterr().out
    assert "670 + 670" in out
    assert "1 of 3 piece(s) omitted" in out
    assert "c" * 64 in out
    # The old defect: a bare "670 + 670" with no disclosure on the same line.
    for line in out.splitlines():
        if "670 + 670" in line:
            assert "omitted" in line


def test_split_plan_final_only_resume_line_is_not_arithmetic(capsys):
    """F3: ``initial_calls`` is the payable count for this run;
    ``reduction_calls`` and ``final_calls`` are full-tree topology
    (``final_calls`` is hard-coded ``1`` at ``file_division.py:3387``). A
    recovered file with only the final call left to pay must not render as
    if ``1 == 1 + 1``."""
    from codedoc.cli.cli import _print_preflight_summary

    record = _split_plan_record(initial_calls=1, reduction_calls=1, final_calls=1)
    _print_preflight_summary(_split_plan_snapshot([record]), verbose=False)
    out = capsys.readouterr().out
    assert "1 initial call(s) payable now" in out
    assert "full tree: 1 reduction + 1 final call(s)" in out
    # The old defect read as an arithmetic decomposition of the lead figure.
    assert "1 initial call(s) (1 reduction + 1 final)" not in out
