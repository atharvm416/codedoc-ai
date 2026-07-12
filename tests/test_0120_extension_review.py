"""0.12.0 — mandatory safety review over reachable extension scopes."""

import json

import pytest

from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedProfile,
    build_review_batches,
    build_review_units,
    default_prompt_profiles,
    validate_profile,
)
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import PromptCustomizationValidationError

_KNOWN = frozenset({".py", ".js", ".rb"})


def _resolved(common_desc, per_extension):
    raw = {
        "single": {
            "common": {"requested_shape": {"description": common_desc}},
            "per_extension": per_extension,
        }
    }
    return ResolvedProfile(
        "single",
        validate_profile(raw, active_mode="single", known_extensions=_KNOWN,
                         source="inline", source_path=None),
    )


def _resolved_default_common(per_extension):
    """A developer-standard (inert) common block plus per_extension overrides."""
    raw = default_prompt_profiles("single")
    raw["single"]["per_extension"] = per_extension
    return ResolvedProfile(
        "single",
        validate_profile(raw, active_mode="single", known_extensions=_KNOWN,
                         source="inline", source_path=None),
    )


def _scopes(*names):
    return frozenset(FileScope(basename=name) for name in names)


def test_unused_extension_entry_creates_zero_review_units():
    # Common is developer-standard (inert); only an unreachable .rb override is
    # customized.  A planned .py file reaches only the inert common block.
    resolved = _resolved_default_common({".rb": {"requested_shape": {"description": "RB"}}})
    units, components = build_review_units(resolved, _scopes("a.py"))
    assert units == []
    assert components == {}


def test_reachable_common_and_extension_create_two_components():
    resolved = _resolved("Common desc.", {".js": {"requested_shape": {"description": "JS desc."}}})
    units, components = build_review_units(resolved, _scopes("a.js", "b.py"))
    assert set(components) == {"single/combined/*", "single/combined/ext:.js"}
    # Each component contributes its own units.
    assert {u.component for u in units} == set(components)


def test_byte_identical_blocks_across_scopes_collapse():
    # The .js override renders identically to common -> deduplicated to one.
    resolved = _resolved("Same desc.", {".js": {"requested_shape": {"description": "Same desc."}}})
    units, components = build_review_units(resolved, _scopes("a.js", "b.py"))
    assert set(components) == {"single/combined/*"}
    assert {u.component for u in units} == {"single/combined/*"}


def test_review_id_and_stream_digest_are_deterministic():
    resolved = _resolved("Common desc.", {".js": {"requested_shape": {"description": "JS desc."}}})
    b1 = build_review_batches(resolved, _scopes("a.js", "b.py"))
    b2 = build_review_batches(resolved, _scopes("a.js", "b.py"))
    assert [b.review_id for b in b1] == [b.review_id for b in b2]
    assert [b.stream_digest for b in b1] == [b.stream_digest for b in b2]
    assert b1 and b1[0].review_id.startswith("rev-")


def test_empty_planned_scopes_yields_no_review():
    resolved = _resolved("Common desc.", {".js": {"requested_shape": {"description": "JS desc."}}})
    assert build_review_batches(resolved, frozenset()) == []
    assert build_review_units(resolved, frozenset()) == ([], {})


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
