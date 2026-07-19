"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import json

class _Fake:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return json.dumps(
            {"description": "A file.", "role_in_system": "r",
             "functions": [{"name": "f", "description": "d"}]}
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)
