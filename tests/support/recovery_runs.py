"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import json
from pathlib import Path

def _fake_provider(observer=None):
    """A fake provider that returns valid agent JSON.

    ``observer`` (optional) is invoked with no arguments on every
    ``complete_json`` call, letting a test inspect on-disk state *mid-run*.
    """
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if observer is not None:
                observer()
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": "Documented.",
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
                "description": "Documented.",
                "role_in_system": "test",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()

def _run(tmp_path, monkeypatch, provider=None, **cfg):
    if provider is None:
        provider = _fake_provider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)

def _write_codedoc_json(path: Path, files: list, status: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"entry_file": "main.py", "schema_version": "1.4"}
    if status:
        meta["status"] = status
    payload: dict = {"_codedoc": meta, "schema_version": "1.4", "files": files}
    if status == "in_progress":
        payload = {"_crash_safety": "INCOMPLETE RUN - test", **payload}
    path.write_text(json.dumps(payload), encoding="utf-8")
