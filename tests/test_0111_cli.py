"""0.11.1 — CLI surface: version-2 stdout export, describe format rules, and the
distinct single-to-triple conversion terminal results.

Covers Test Plan items #32, #38, #40 and Review Addenda 10/12. Inline profiles
reach the CLI through ``codedoc.config.json`` (there is no inline CLI flag).
"""

import json

from codedoc.cli.cli import (
    _print_conversion_summary,
    _print_prompt_profile_dry_run,
    _print_prompt_profile_run,
    run_cli,
)

from tests.test_0111_conversion import ConversionFake, CUSTOM_SINGLE_V2


def _project_with_inline(tmp_path, inline, *, analysis_mode="triple"):
    (tmp_path / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    config = {"analysis_mode": analysis_mode, "entry_file": "main.py",
              "prompt_profiles": inline}
    (tmp_path / "codedoc.config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Describe / export (#32, #40, Addendum 12)
# ---------------------------------------------------------------------------

def test_describe_rejects_format_both():
    assert run_cli(["--describe-prompt-schema", "--format", "both"]) == 2


def test_describe_accepts_json_and_md(capsys):
    assert run_cli(["--describe-prompt-schema", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["supported_schema_versions"] == [1, 2]
    assert run_cli(["--describe-prompt-schema", "--format", "md"]) == 0
    assert "requested_shape" in capsys.readouterr().out


def test_stdout_export_modes(capsys):
    assert run_cli(["--export-prompt-profile"]) == 0
    both = json.loads(capsys.readouterr().out)["prompt_profiles"]
    assert "single" in both and "triple" in both
    assert run_cli(["--export-prompt-profile", "--analysis-mode", "single"]) == 0
    one = json.loads(capsys.readouterr().out)["prompt_profiles"]
    assert "single" in one and "triple" not in one


def test_path_export_remains_v1_external(tmp_path):
    target = tmp_path / "profile.json"
    assert run_cli(["--export-prompt-profile", str(target)]) == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert "fields" in data["single"]


def test_explicit_profile_cost_summaries(capsys):
    common = {
        "prompt_profile_source": "inline",
        "prompt_profile_active": True,
        "prompt_profile_affected_files": 2,
    }
    _print_prompt_profile_dry_run({
        **common,
        "documentation_calls_planned": 2,
        "prompt_customization_security_review_calls_planned": 1,
    })
    dry = capsys.readouterr().out
    assert "Documentation calls   : 2 planned" in dry
    assert "Security-review calls : 1 planned" in dry
    assert "Total paid calls      : 3 planned" in dry

    _print_prompt_profile_run({
        **common,
        "prompt_customization_security_review": "safe",
        "documentation_calls_attempted": 2,
        "prompt_customization_security_review_calls_attempted": 1,
        "prompt_customization_security_review_calls_completed": 1,
        "prompt_profile_conversion_calls_attempted": 0,
    })
    real = capsys.readouterr().out
    assert "Security review       : safe (1 attempted, 1 completed)" in real
    assert "Documentation calls   : 2 attempted" in real
    assert "Total attempted calls : 3" in real


def test_no_profile_cost_summary_stays_silent(capsys):
    _print_prompt_profile_dry_run({"prompt_profile_source": "absent"})
    _print_prompt_profile_run({"prompt_profile_source": "disabled"})
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Conversion terminal results (#38, Addendum 10)
# ---------------------------------------------------------------------------

def test_cli_conversion_dry_run(monkeypatch, tmp_path, capsys):
    project = _project_with_inline(tmp_path, CUSTOM_SINGLE_V2)

    def boom(_cfg):
        raise AssertionError("no provider in dry-run")

    monkeypatch.setattr("codedoc.pipeline.create_provider", boom)
    code = run_cli([str(project), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "conversion (dry run)" in out
    assert "Total paid proposal calls   : 2" in out


def test_cli_conversion_proposal_prints_fragment(monkeypatch, tmp_path, capsys):
    project = _project_with_inline(tmp_path, CUSTOM_SINGLE_V2)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: ConversionFake())
    code = run_cli([str(project)])
    out = capsys.readouterr().out
    assert code == 0
    assert "conversion proposal" in out
    assert '"prompt_profiles"' in out
    assert '"schema_version": 2' in out
    # the printed fragment is valid JSON
    start = out.index('{', out.index('"prompt_profiles"') - 5)
    json.loads(out[start: out.rindex("}") + 1])


def test_cli_conversion_fail_closed_exit_2(monkeypatch, tmp_path, capsys):
    project = _project_with_inline(tmp_path, CUSTOM_SINGLE_V2)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda c: ConversionFake(routing_raw="{not json"),
    )
    code = run_cli([str(project)])
    err = capsys.readouterr().err
    assert code == 2
    assert "Paid calls before stop" in err
    assert err.index("Paid calls before stop") < err.index("Error:")
    assert not (project / "codedoc").exists()


def test_new_cli_surfaces_are_cp1252_safe(capsys):
    assert run_cli(["--describe-prompt-schema", "--format", "md"]) == 0
    capsys.readouterr().out.encode("cp1252")

    _print_conversion_summary(
        {
            "analysis_mode": "triple",
            "prompt_profile_conversion": "pending",
            "prompt_customization_security_review_calls_planned": 1,
            "prompt_profile_conversion_calls_planned": 1,
        }
    )
    capsys.readouterr().out.encode("cp1252")
