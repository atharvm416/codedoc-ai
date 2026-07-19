"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.prompt_profile_runs import project  # noqa: F401, F811


from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedProfile,
    build_review_batches,
    validate_profile,
)
from codedoc.pipeline import run_pipeline
import pytest
import codedoc.pipeline as pipe
from codedoc.utils.errors import ConfigError
from tests.support.profiles import INLINE
from tests.support.providers import SmartFake
from tests.support.prompt_profile_runs import _run
from tests.support.prompt_profile_runs import _output

def _project(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "mod.js").write_text("const y = 2;\n", encoding="utf-8")

_JS_DESC = "Explain the JavaScript module for a reviewer, in detail."

def _profile(js_desc=_JS_DESC):
    return {
        "single": {
            "common": {"requested_shape": {"description": "Common summary."}},
            "per_extension": {".js": {"requested_shape": {"description": js_desc}}},
        }
    }

def _dry_run(tmp_path, overrides=None, monkeypatch=None):
    if monkeypatch is not None:
        def _boom(_config):
            raise AssertionError("dry-run must not create a provider")
        monkeypatch.setattr("codedoc.pipeline.create_provider", _boom)
    config = {
        "dry_run": True,
        "documentation_scope": "all",
        "prompt_profiles": _profile(),
    }
    if overrides:
        config.update(overrides)
    return run_pipeline(tmp_path, config)

def test_scope_counts_sum_to_would_call_llm_for(tmp_path, monkeypatch):
    _project(tmp_path)
    stats = _dry_run(tmp_path, monkeypatch=monkeypatch)
    counts = stats["prompt_profile_scope_counts"]
    assert set(counts) == {"extension", "common", "built-in"}
    assert sum(counts.values()) == stats["would_call_llm_for"]
    # main.py -> common (customized), mod.js -> extension.
    assert counts["extension"] == 1
    assert counts["common"] == 1
    assert counts["built-in"] == 0

def test_dry_run_reports_exact_review_batch_count_without_provider(tmp_path, monkeypatch):
    _project(tmp_path)
    stats = _dry_run(tmp_path, monkeypatch=monkeypatch)
    # Independently compute the expected batch count for the two reachable scopes.
    profile = validate_profile(
        _profile(), active_mode="single", known_extensions=frozenset({".py", ".js"}),
        source="inline", source_path=None,
    )
    resolved = ResolvedProfile("single", profile)
    scopes = frozenset({FileScope(basename="main.py"), FileScope(basename="mod.js")})
    expected = len(build_review_batches(resolved, scopes))
    assert expected >= 1
    assert stats["prompt_customization_security_review_calls_planned"] == expected
    assert stats["prompt_customization_security_review"] == "pending"
    assert stats["prompt_customization_security_review_calls_completed"] == 0

def test_token_estimate_reflects_the_resolved_override_block(tmp_path, monkeypatch):
    _project(tmp_path)

    def _run(js_desc):
        def _boom(_config):
            raise AssertionError("dry-run must not create a provider")
        monkeypatch.setattr("codedoc.pipeline.create_provider", _boom)
        config = {
            "dry_run": True,
            "documentation_scope": "all",
            "prompt_profiles": {
                "single": {
                    "common": {"requested_shape": {"description": "Common summary."}},
                    "per_extension": {".js": {"requested_shape": {"description": js_desc}}},
                }
            },
        }
        return run_pipeline(tmp_path, config)["estimated_input_tokens"]

    short = _run("Short.")
    long = _run("A considerably longer instruction " * 40)
    # The .js file's estimate uses its resolved override block, so a longer
    # override instruction raises the estimate.
    assert long > short > 0

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
