"""0.12.0 — golden prompt bytes and digest stability.

Proves that the extension-scope release did not change the developer-standard
prompt bytes for any file that has no matching override, nor the per-file digest
of a ``common``-only profile.  The SHA-256 pins and the embedded literal block
are the exact values produced by the rendering path, which this release left
untouched, so existing records stay reusable.
"""

import hashlib

from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    default_prompt_profiles,
    render_default_shape_block,
    validate_profile,
)

# Byte-exact SHA-256 of every developer-standard requested-shape block.  A change
# here means the frozen prompt bytes drifted and cache identity would silently
# break for every existing record.
_GOLDEN_BLOCK_SHA256 = {
    ("single", "combined"): "79eab7cf04e9ac6b7e2971065810e0a43448366f88b10630b6ddddf7f91217bd",
    ("triple", "structure"): "0b3b137f5e29a053f117a1eded2946cfba2dd66b4a08daa932b3766264396417",
    ("triple", "dependency"): "6ade2ed4a599164dfec6caaecf8f283f538752e8091b5404e199cddb8667b20a",
    ("triple", "documentation"): "13afc0f2a631df40b1f9a2ae496f0e610786d25eefc1b6db67c1e4c5d9972283",
}

# The exact single/combined default block, embedded verbatim so any visible drift
# in the most-scrutinized prompt fails loudly rather than only via a hash.
_SINGLE_COMBINED_GOLDEN = (
    'Return EXACTLY this JSON shape:\n'
    '{\n'
    '  "description": "<clear paragraph describing what this file does and why it exists>",\n'
    '  "role_in_system": "<how this file connects to and supports the rest of the codebase>",\n'
    '  "functions": [\n'
    '    {"name": "<function or method defined IN this file>", "description": "<what it does>"}\n'
    '  ],\n'
    '  "classes": [\n'
    '    {"name": "<class defined IN this file>", "description": "<what it does>"}\n'
    '  ],\n'
    '  "exports": ["<symbol this module deliberately exposes, including intentional re-exports>"],\n'
    '  "dependencies_analysis": {\n'
    '    "internal": ["<relative import paths that are project files>"],\n'
    '    "external": ["<package names from node_modules / pip / pub / maven>"],\n'
    '    "dependency_refs": ["<normalized dependency names used by this file>"],\n'
    '    "catalog_updates": [\n'
    '      {"name": "<normalized dependency name>", "type": "internal|external", "used_for": "<stable project-level purpose>"}\n'
    '    ],\n'
    '    "usage_notes": [\n'
    '      {"import": "<import string>", "used_for": "<file-specific note only>"}\n'
    '    ],\n'
    '    "warnings": ["<any dependency concern, e.g. unused import, potential cycle>"]\n'
    '  },\n'
    '  "key_concepts": ["<important concept or pattern used in this file>"],\n'
    '  "usage_example": "<one-line example of how another file imports or uses this file, or empty string>"\n'
    '}'
)

# A ``common``-only profile digest, pinned to the exact pre-edit value.
_COMMON_ONLY_DIGEST = "pp-v1:eb3ccbfafeb5b164bc67a9c84552d2fd3a159e82870d6b47e9135c12ac912ad0"


def test_default_blocks_are_byte_identical_per_mode_agent():
    for (mode, agent), expected in _GOLDEN_BLOCK_SHA256.items():
        block = render_default_shape_block(mode, agent)
        assert hashlib.sha256(block.encode("utf-8")).hexdigest() == expected


def test_single_combined_default_block_literal_unchanged():
    assert render_default_shape_block("single", "combined") == _SINGLE_COMBINED_GOLDEN


def test_common_only_profile_digest_is_stable_and_scope_independent():
    raw = {
        "single": {
            "common": {
                "requested_shape": {"description": "Golden fixed description for digest pin."}
            }
        }
    }
    profile = validate_profile(
        raw, active_mode="single", known_extensions=frozenset({".py"}),
        source="inline", source_path=None,
    )
    resolved = ResolvedProfile("single", profile)
    # No per_extension entry -> every basename resolves to the same common digest,
    # equal to the empty-basename digest.
    assert resolved.file_digest("anything.py") == _COMMON_ONLY_DIGEST
    assert resolved.file_digest("") == _COMMON_ONLY_DIGEST
    assert resolved.file_digest("x.rs") == _COMMON_ONLY_DIGEST


def test_no_profile_renders_defaults_and_no_digest():
    resolved = ResolvedProfile("single", None)
    assert resolved.file_digest("a.py") == NO_PROMPT_PROFILE_DIGEST
    block = resolved.resolve_block("combined", "a.py")
    assert block.text == render_default_shape_block("single", "combined")
    assert not block.active


def test_generated_default_profile_is_inert():
    # The generated default carries an empty per_extension and is developer-
    # standard-equivalent, so it stamps no digest.
    profile = validate_profile(
        default_prompt_profiles("single"),
        active_mode="single", known_extensions=frozenset({".py"}),
        source="inline", source_path=None,
    )
    resolved = ResolvedProfile("single", profile)
    assert resolved.file_digest("a.py") == NO_PROMPT_PROFILE_DIGEST
