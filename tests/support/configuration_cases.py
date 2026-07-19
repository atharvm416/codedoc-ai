"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _fake_provider():
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({"description": "ok", "role_in_system": "r",
                                    "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _json.dumps({"description": "ok", "role_in_system": "r",
                                "functions": [], "classes": [], "exports": []})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()

def _load(tmp_path, **overrides):
    from codedoc.core.loader import load_config
    return load_config(tmp_path, overrides if overrides else None)
