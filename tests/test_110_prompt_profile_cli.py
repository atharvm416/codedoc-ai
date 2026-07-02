"""0.11.0 — CLI describe/export utilities and end-to-end pipeline integration.

Covers Tests #9, #10, #11 (pipeline ordering / outcomes) and #21 (describe/export
determinism, no-overwrite safety, backup-before-force, project-free execution).
"""

import json

import pytest

import codedoc.pipeline as pipe
from codedoc.cli.cli import run_cli
from codedoc.utils.errors import ConfigError, PromptCustomizationValidationError

INLINE = {"schema_version": 1, "single": {"common": {"fields": [
    {"key": "description", "type": "string", "instruction": "Describe the file."},
    {"key": "key_concepts", "type": "string_list", "instruction": "List concepts."},
]}}}


class SmartFake:
    """Serves both the review verdict and documentation responses."""

    provider_name = "fake"

    def __init__(self, verdict="SAFE"):
        self.verdict = verdict
        self.review_calls = 0
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            review_id = next(
                line.split(": ", 1)[1]
                for line in prompt.splitlines()
                if line.startswith("review_id: ")
            )
            ordinal, count = next(
                line.split(": ", 1)[1].split("/", 1)
                for line in prompt.splitlines()
                if line.startswith("batch: ")
            )
            return json.dumps({
                "review_id": review_id,
                "batch_index": int(ordinal),
                "batch_count": int(count),
                "verdict": self.verdict,
                "reasons": ["bad"] if self.verdict == "TOO_RISKY" else [],
                "warnings": ["warn"] if self.verdict == "RISKY" else [],
            })
        self.doc_calls += 1
        return json.dumps({
            "description": "A file.", "role_in_system": "core",
            "functions": [{"name": "f", "description": "does f"}],
            "key_concepts": ["kc"], "usage_example": "import x",
        })

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _run(monkeypatch, project, cfg, fake):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: fake)
    return pipe.run_pipeline(project, cfg)


def _output(project):
    return project / "codedoc" / "codedoc.json"


def _recovery(project):
    return project / "codedoc" / "crash_recovery.json"


# ---------------------------------------------------------------------------
# End-to-end pipeline integration (Tests #9, #10, #11)
# ---------------------------------------------------------------------------

def test_no_profile_makes_no_review_and_no_digest(monkeypatch, project):
    fake = SmartFake()
    stats = _run(monkeypatch, project, {"entry_file": "main.py"}, fake)
    assert fake.review_calls == 0 and stats["checked"] == 1
    assert stats["prompt_customization_security_review"] == "not-required"
    assert stats["documentation_calls_attempted"] == stats["attempted_calls"] == 1
    assert stats["prompt_customization_security_review_calls_attempted"] == 0
    rec = json.loads(_output(project).read_text())["files"][0]
    assert "_prompt_profile_digest" not in rec


def test_active_profile_safe_reviews_filters_and_stamps(monkeypatch, project):
    fake = SmartFake("SAFE")
    stats = _run(monkeypatch, project, {"entry_file": "main.py", "prompt_profiles": INLINE}, fake)
    assert fake.review_calls == 1 and fake.doc_calls == 1
    assert stats["prompt_customization_security_review"] == "safe"
    assert stats["prompt_profile_active"] is True
    assert stats["prompt_profile_source"] == "inline"
    category_total = (
        stats["documentation_calls_attempted"]
        + stats["prompt_customization_security_review_calls_attempted"]
    )
    assert category_total == stats["attempted_calls"] == 2
    rec = json.loads(_output(project).read_text())["files"][0]
    assert rec["description"] == "A file."
    assert "functions" not in rec          # profile omitted -> filtered
    assert "_prompt_profile_digest" in rec


def test_too_risky_blocks_and_writes_no_artifacts(monkeypatch, project):
    fake = SmartFake("TOO_RISKY")
    with pytest.raises(PromptCustomizationValidationError, match="TOO RISKY") as caught:
        _run(monkeypatch, project, {"entry_file": "main.py", "prompt_profiles": INLINE}, fake)
    assert caught.value.stats["prompt_customization_security_review"] == "too-risky-blocked"
    assert caught.value.stats["prompt_customization_security_review_calls_completed"] == 1
    # Review ran before any mutation: no output, no recovery file, no documentation call.
    assert fake.doc_calls == 0
    assert not _output(project).exists()
    assert not _recovery(project).exists()


def test_paid_review_warning_names_resolved_provider_and_model(
    monkeypatch, project, capsys
):
    _run(monkeypatch, project, {
        "entry_file": "main.py", "prompt_profiles": INLINE,
        "llm_provider": "anthropic", "model_name": "claude-test",
    }, SmartFake("SAFE"))
    out = capsys.readouterr().out
    assert "provider=anthropic" in out
    assert "model=claude-test" in out


def test_dry_run_reports_pending_and_creates_no_provider(monkeypatch, project):
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda c: pytest.fail("provider created during dry-run"),
    )
    stats = pipe.run_pipeline(
        project, {"entry_file": "main.py", "prompt_profiles": INLINE, "dry_run": True}
    )
    assert stats["prompt_customization_security_review"] == "pending"
    assert stats["prompt_customization_security_review_calls_planned"] == 1
    assert not _output(project).exists()


def test_cap_fails_before_review_or_billing(monkeypatch, project):
    (project / "b.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    fake = SmartFake("SAFE")
    with pytest.raises(ConfigError, match="max_files"):
        _run(monkeypatch, project, {
            "documentation_scope": "all", "prompt_profiles": INLINE, "max_files": 1}, fake)
    assert fake.review_calls == 0 and fake.doc_calls == 0


def test_first_activation_invalidates_then_reuses(monkeypatch, project):
    _run(monkeypatch, project, {"entry_file": "main.py"}, SmartFake())
    fake2 = SmartFake("SAFE")
    s2 = _run(monkeypatch, project, {"entry_file": "main.py", "prompt_profiles": INLINE}, fake2)
    assert s2["checked"] == 1 and fake2.doc_calls == 1  # reprocessed once on activation
    fake3 = SmartFake("SAFE")
    s3 = _run(monkeypatch, project, {"entry_file": "main.py", "prompt_profiles": INLINE}, fake3)
    assert s3["checked"] == 0 and fake3.doc_calls == 0 and fake3.review_calls == 0  # reused
    assert s3["documentation_calls_attempted"] == 0
    assert s3["prompt_customization_security_review_calls_attempted"] == 0


# ---------------------------------------------------------------------------
# Describe / export utilities (Test #21)
# ---------------------------------------------------------------------------

def test_describe_schema_json_is_project_free(capsys):
    assert run_cli(["--describe-prompt-schema", "--analysis-mode", "single"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["modes"]) == {"single"}


def test_describe_schema_markdown(capsys):
    assert run_cli(["--describe-prompt-schema", "--format", "md"]) == 0
    out = capsys.readouterr().out
    assert "BEGIN CODEDOC PROMPT SCHEMA" in out


def test_describe_is_deterministic(capsys):
    run_cli(["--describe-prompt-schema"])
    first = capsys.readouterr().out
    run_cli(["--describe-prompt-schema"])
    assert capsys.readouterr().out == first


def test_describe_schema_shows_common_envelope(capsys):
    assert run_cli(["--describe-prompt-schema", "--format", "md"]) == 0
    out = capsys.readouterr().out
    assert '"common"' in out
    assert "--export-prompt-profile" not in out
    assert "codedoc-prompt-profiles.json" not in out


def test_describe_rejects_format_both(capsys):
    assert run_cli(["--describe-prompt-schema", "--format", "both"]) == 2
    err = capsys.readouterr().err
    assert "both" in err.lower()


def test_utilities_are_mutually_exclusive_with_run_options():
    # describe + a documentation run option -> usage error
    assert run_cli(["--describe-prompt-schema", "--entry", "main.py"]) == 2


def test_force_without_init_utility_is_rejected():
    assert run_cli(["--force"]) == 2
