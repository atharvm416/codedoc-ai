"""Strict CLI utility-combination and lifecycle tests."""

import json

import pytest

from codedoc.cli.cli import _confirm_risky_prompt_customization, build_parser, run_cli


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
