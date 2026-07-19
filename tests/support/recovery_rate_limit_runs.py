"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _make_fake_provider(description: str = "Documented.", fail_after: int = -1):
    """Return a fake LLM provider.

    Parameters
    ----------
    fail_after:
        If >= 0, raise ``LLMError`` with a rate-limit signal on every call
        after the first *fail_after* successful calls.  -1 = always succeed.
    """
    import json as _json
    from codedoc.utils.errors import LLMError

    call_count = {"n": 0}

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if fail_after >= 0 and call_count["n"] > fail_after:
                raise LLMError("fake", "429 rate_limit_exceeded tokens per min")
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": description,
                    "role_in_system": "test",
                    "key_concepts": [],
                    "usage_example": "",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({
                    "dependencies_analysis": {
                        "internal": [], "external": [],
                        "dependency_refs": [], "catalog_updates": [],
                        "usage_notes": [], "warnings": [],
                    }
                })
            return _json.dumps({
                "description": description,
                "role_in_system": "test",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()

def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)

def _run(tmp_path, monkeypatch, provider=None, **cfg):
    """Helper to run the pipeline with a fake provider."""
    if provider is None:
        provider = _make_fake_provider()
    _patch_provider(monkeypatch, provider)
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)
