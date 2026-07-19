"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.prompt_profile_runs import project  # noqa: F401, F811


import json
import pytest
from codedoc.utils.errors import PromptCustomizationValidationError
from tests.support.profiles import INLINE
from tests.support.providers import SmartFake
from tests.support.prompt_profile_runs import _run
from tests.support.prompt_profile_runs import _output

def _recovery(project):
    return project / "codedoc" / "crash_recovery.json"

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
