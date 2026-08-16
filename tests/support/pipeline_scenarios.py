"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
import json
import re
from pathlib import Path

def _cache_identity(rel_path: str = "main.py", analysis_mode: str = "single") -> dict:
    """Cache-identity keys a prior CodeDoc run would have persisted for *rel_path*.

    Includes ``_ordinary_path_identity`` (0.14.4) so a fixture record built
    with this helper is same-path reusable, not merely revision/mode-matched;
    ordinary cross-path reuse is refused regardless of this key.
    """
    from codedoc.core.record_meta import (
        expected_analysis_identity,
        expected_ordinary_path_identity,
    )

    return {
        **expected_analysis_identity(analysis_mode),
        "_ordinary_path_identity": expected_ordinary_path_identity(rel_path),
    }

def make_fake_provider(description="Documented."):
    """Return a provider instance whose complete_json always succeeds."""
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            # 0.10.0: a single combined response shape works for both the default
            # single-call combined agent and each legacy triple-mode agent (each
            # extracts only its own fields).
            return _json.dumps({
                "description": description,
                "role_in_system": "test role",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {
                    "internal": [], "external": [],
                    "dependency_refs": [], "catalog_updates": [],
                    "usage_notes": [], "warnings": [],
                },
                "key_concepts": [],
                "usage_example": "",
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()

def md_meta(md_path: Path) -> dict:
    content = md_path.read_text(encoding="utf-8")
    m = re.search(r"<!-- codedoc-ai: (\{.*?\}) -->", content, re.DOTALL)
    assert m, f"No metadata comment in {md_path}"
    return json.loads(m.group(1))

def no_llm(monkeypatch):
    def fail(_config):
        raise AssertionError("LLM must not be called in this scenario")
    monkeypatch.setattr("codedoc.pipeline.create_provider", fail)

def patch_provider(monkeypatch, description="Documented."):
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: make_fake_provider(description),
    )

def write_existing_json(
    path: Path, file_hash: str, description: str = "Cached.", analysis_mode: str = "single"
):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_record = {
        "path": "main.py", "hash": file_hash,
        "language": "python", "description": description,
        **_cache_identity("main.py", analysis_mode),
    }
    path.write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [file_record],
    }), encoding="utf-8")

def write_existing_md(
    path: Path, file_hash: str, description: str = "Cached.", analysis_mode: str = "single"
):
    """Write a prior-run MD file with a lossless embedded view carrying the
    0.10.0 cache-identity keys, so steady-state reuse skips the provider."""
    from codedoc.core.markdown_view import markdown_from_view

    view = {
        "schema_version": "1.4",
        "project": {
            "entry_file": "main.py", "file_count": 1,
            "languages": ["python"], "folders": [],
        },
        "run": {
            "files_checked": 1, "files_failed": 0, "files_skipped": 0,
            "files_reused": 0, "files_documented": 1,
        },
        "tree": {},
        "folders": [],
        "dependency_catalog": [],
        "dependency_graph": [],
        "files": [{
            "path": "main.py", "hash": file_hash,
            "language": "python", "description": description,
            **_cache_identity("main.py", analysis_mode),
        }],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_from_view(view), encoding="utf-8")
