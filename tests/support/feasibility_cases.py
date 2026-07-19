"""Shared test support extracted from mapped source modules."""

import json
from codedoc.core.prompt_profiles import (
    default_prompt_profiles,
)

class _ReviewFake:
    provider_name = "fake"

    def __init__(self, verdict="SAFE", *, malformed=False):
        self.verdict = verdict
        self.malformed = malformed
        self.review_calls = 0
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            if self.malformed:
                return "{}"
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
                    "verdict": self.verdict,
                    "reasons": ["blocked"] if self.verdict == "TOO_RISKY" else [],
                    "warnings": ["confirm"] if self.verdict == "RISKY" else [],
                }
            )
        self.doc_calls += 1
        return json.dumps({"description": "Documented file."})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

def _cross_file_profile() -> dict:
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["requested_shape"]["description"] = (
        "Add a reference of a different file."
    )
    return raw
