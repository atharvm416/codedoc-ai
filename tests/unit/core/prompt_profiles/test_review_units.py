"""Tests organized by feature ownership."""

import json
import pytest
from codedoc.agents.prompt_customization_validation_agent import (
    PromptCustomizationValidationAgent,
)
from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedProfile,
    build_review_batches,
    validate_profile,
)
from codedoc.utils.errors import PromptCustomizationValidationError
from tests.support.security_review_cases import INLINE
from tests.support.security_review_cases import ReviewFake
from codedoc.core.prompt_profiles import (
    build_review_units,
    default_prompt_profiles,
)

def _batches():
    profile = validate_profile(
        INLINE,
        active_mode="single",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )
    return build_review_batches(
        ResolvedProfile("single", profile), frozenset({FileScope(basename="main.py")})
    )

@pytest.mark.parametrize("verdict", ["SAFE", "RISKY", "TOO_RISKY"])
def test_review_agent_preserves_highest_verdict(verdict):
    outcome = PromptCustomizationValidationAgent(ReviewFake(verdict)).review(_batches())
    assert outcome.verdict == verdict
    assert outcome.calls_completed == 1

@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '```json\n{"verdict":"SAFE"}\n```',
        '{"verdict":"SAFE","verdict":"TOO_RISKY"}',
    ],
)
def test_malformed_or_duplicate_review_response_fails_closed(raw):
    class RawProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return raw

    with pytest.raises(PromptCustomizationValidationError, match="failed closed"):
        PromptCustomizationValidationAgent(RawProvider()).review(_batches())

def test_review_binding_mismatch_fails_closed():
    class WrongBinding(ReviewFake):
        def complete_json(self, prompt, system=""):
            payload = json.loads(super().complete_json(prompt, system))
            payload["review_id"] = "rev-wrong"
            return json.dumps(payload)

    with pytest.raises(PromptCustomizationValidationError, match="binding mismatch"):
        PromptCustomizationValidationAgent(WrongBinding("SAFE")).review(_batches())

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
