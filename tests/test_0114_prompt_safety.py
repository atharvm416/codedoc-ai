"""Medium-risk customization confirmation boundary."""

import json

import pytest

from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import PromptCustomizationValidationError


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
