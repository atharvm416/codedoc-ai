"""Focused 0.11.3 tests for mandatory non-bypassable semantic review."""

import json

import pytest

from codedoc.agents.prompt_customization_validation_agent import (
    PromptCustomizationValidationAgent,
)
from codedoc.core.prompt_profiles import ResolvedProfile, build_review_batches, validate_profile
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError, PromptCustomizationValidationError


INLINE = {
    "schema_version": 1,
    "single": {
        "common": {
            "fields": [
                {"key": "description", "type": "string", "instruction": "Custom description."},
            ]
        }
    },
}


class ReviewFake:
    provider_name = "fake"

    def __init__(self, verdict):
        self.verdict = verdict
        self.review_calls = 0
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            review_id = next(line.split(": ", 1)[1] for line in prompt.splitlines() if line.startswith("review_id: "))
            ordinal, count = next(line.split(": ", 1)[1].split("/", 1) for line in prompt.splitlines() if line.startswith("batch: "))
            return json.dumps({
                "review_id": review_id,
                "batch_index": int(ordinal),
                "batch_count": int(count),
                "verdict": self.verdict,
                "reasons": ["unsafe"] if self.verdict == "TOO_RISKY" else [],
                "warnings": ["review"] if self.verdict == "RISKY" else [],
            })
        self.doc_calls += 1
        return json.dumps({"description": "documented"})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _batches():
    profile = validate_profile(
        INLINE,
        active_mode="single",
        known_languages=frozenset({"python"}),
        source="inline",
        source_path=None,
    )
    return build_review_batches(ResolvedProfile("single", profile), frozenset({"python"}))


@pytest.mark.parametrize("verdict", ["SAFE", "RISKY", "TOO_RISKY"])
def test_review_agent_preserves_highest_verdict(verdict):
    outcome = PromptCustomizationValidationAgent(ReviewFake(verdict)).review(_batches())
    assert outcome.verdict == verdict
    assert outcome.calls_completed == 1


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '```json\n{"verdict":"SAFE"}\n```',
        '{"verdict":"SAFE","verdict":"TOO_RISKY"}',
    ],
)
def test_malformed_or_duplicate_review_response_fails_closed(raw):
    class RawProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return raw

    with pytest.raises(PromptCustomizationValidationError, match="failed closed"):
        PromptCustomizationValidationAgent(RawProvider()).review(_batches())


def test_review_binding_mismatch_fails_closed():
    class WrongBinding(ReviewFake):
        def complete_json(self, prompt, system=""):
            payload = json.loads(super().complete_json(prompt, system))
            payload["review_id"] = "rev-wrong"
            return json.dumps(payload)

    with pytest.raises(PromptCustomizationValidationError, match="binding mismatch"):
        PromptCustomizationValidationAgent(WrongBinding("SAFE")).review(_batches())


def test_too_risky_always_blocks_before_mutation_or_documentation(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    fake = ReviewFake("TOO_RISKY")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    with pytest.raises(PromptCustomizationValidationError, match="TOO RISKY"):
        run_pipeline(tmp_path, {"entry_file": "main.py", "prompt_profiles": INLINE})
    assert fake.review_calls == 1
    assert fake.doc_calls == 0
    assert not (tmp_path / "codedoc").exists()


def test_removed_risky_override_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="prompt_customization_allow_risky"):
        run_pipeline(tmp_path, {"prompt_customization_allow_risky": True, "dry_run": True})
