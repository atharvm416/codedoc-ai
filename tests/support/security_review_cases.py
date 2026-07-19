"""Shared test support extracted from mapped source modules."""

import json

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
