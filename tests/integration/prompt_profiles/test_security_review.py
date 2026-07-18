"""Tests organized by feature ownership."""

import json
import pytest
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import PromptCustomizationValidationError
from tests.support.feasibility_cases import _cross_file_profile
from tests.support.feasibility_cases import _ReviewFake
from tests.support.security_review_cases import INLINE as MANDATORY_SECURITY_INLINE
from tests.support.security_review_cases import ReviewFake

INLINE = {
    "schema_version": 1,
    "single": {
        "common": {
            "fields": [
                {
                    "key": "description",
                    "type": "string",
                    "instruction": "Custom description.",
                }
            ]
        }
    },
}

class RiskyProvider:
    provider_name = "fake"

    def __init__(self):
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
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
            return json.dumps(
                {
                    "review_id": review_id,
                    "batch_index": int(ordinal),
                    "batch_count": int(count),
                    "verdict": "RISKY",
                    "reasons": [],
                    "warnings": ["review this"],
                }
            )
        self.doc_calls += 1
        return json.dumps({"description": "documented"})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

@pytest.mark.parametrize("callback", [None, lambda _warnings: False])
def test_medium_risk_without_confirmation_blocks_before_mutation(
    tmp_path, monkeypatch, callback
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    provider = RiskyProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    with pytest.raises(PromptCustomizationValidationError, match="confirmation"):
        run_pipeline(
            tmp_path,
            {"entry_file": "main.py", "prompt_profiles": INLINE},
            confirm_risky=callback,
        )
    assert provider.doc_calls == 0
    assert not (tmp_path / "codedoc").exists()

def test_medium_risk_explicit_confirmation_allows_documentation(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    provider = RiskyProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "prompt_profiles": INLINE},
        confirm_risky=lambda warnings: warnings == ("review this",),
    )
    assert provider.doc_calls == 1
    assert stats["prompt_customization_security_review"] == "risky-confirmed"

@pytest.mark.parametrize(
    ("verdict", "malformed", "confirm_risky", "expected_status"),
    [
        ("SAFE", True, None, "failed-closed"),
        ("RISKY", False, lambda _warnings: False, "risky-confirmation-blocked"),
        ("TOO_RISKY", False, None, "too-risky-blocked"),
    ],
)
def test_blocking_review_paths_carry_advisories(
    tmp_path,
    monkeypatch,
    verdict,
    malformed,
    confirm_risky,
    expected_status,
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = _ReviewFake(verdict, malformed=malformed)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)

    with pytest.raises(PromptCustomizationValidationError) as caught:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "prompt_profiles": _cross_file_profile(),
            },
            confirm_risky=confirm_risky,
        )

    assert (
        caught.value.stats["prompt_customization_security_review"]
        == expected_status
    )
    assert caught.value.stats["prompt_customization_feasibility_advisories"]
    assert fake.doc_calls == 0

def test_too_risky_always_blocks_before_mutation_or_documentation(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    fake = ReviewFake("TOO_RISKY")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    with pytest.raises(PromptCustomizationValidationError, match="TOO RISKY"):
        run_pipeline(tmp_path, {"entry_file": "main.py", "prompt_profiles": MANDATORY_SECURITY_INLINE})
    assert fake.review_calls == 1
    assert fake.doc_calls == 0
    assert not (tmp_path / "codedoc").exists()

class _TooRiskyFake:
    provider_name = "fake"

    def __init__(self):
        self.review_calls = 0
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            review_id = next(
                line.split(": ", 1)[1] for line in prompt.splitlines()
                if line.startswith("review_id: ")
            )
            ordinal, count = next(
                line.split(": ", 1)[1].split("/", 1) for line in prompt.splitlines()
                if line.startswith("batch: ")
            )
            return json.dumps({
                "review_id": review_id,
                "batch_index": int(ordinal),
                "batch_count": int(count),
                "verdict": "TOO_RISKY",
                "reasons": ["unsafe extension override"],
                "warnings": [],
            })
        self.doc_calls += 1
        return json.dumps({"description": "documented"})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

def test_too_risky_extension_block_blocks_before_documentation(tmp_path, monkeypatch):
    (tmp_path / "main.js").write_text("const x = 1;\n", encoding="utf-8")
    fake = _TooRiskyFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    profile = {
        "single": {
            "common": {"requested_shape": {"description": "<clear paragraph describing what this file does and why it exists>"}},
            "per_extension": {
                ".js": {"requested_shape": {"description": "Explain the JS module for a reviewer."}}
            },
        }
    }
    with pytest.raises(PromptCustomizationValidationError, match="TOO RISKY"):
        run_pipeline(tmp_path, {"entry_file": "main.js", "prompt_profiles": profile})
    assert fake.review_calls >= 1
    assert fake.doc_calls == 0
    assert not (tmp_path / "codedoc").exists()
