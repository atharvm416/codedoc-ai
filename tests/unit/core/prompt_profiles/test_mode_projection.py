"""Tests organized by feature ownership."""

import pytest
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    PROFILE_ACTION_EXECUTABLE,
    build_resolved_profile,
    classify_profile_action,
    default_prompt_profiles,
    documentation_projectable_paths,
    validate_profile,
)
from codedoc.utils.errors import ConfigError

def _custom_single_profile():
    raw = default_prompt_profiles("single")
    shape = raw["single"]["common"]["requested_shape"]
    shape["description"] = "Describe this file for a maintainer."
    return validate_profile(
        raw,
        active_mode="triple",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )

def _custom_triple_missing_documentation():
    raw = default_prompt_profiles("triple")
    del raw["triple"]["common"]["documentation"]
    raw["triple"]["common"]["structure"]["requested_shape"]["description"] = (
        "Custom structure description."
    )
    return raw

def test_single_only_triple_profile_projects_without_conversion():
    action = classify_profile_action(_custom_single_profile(), "triple")
    assert action.action == PROFILE_ACTION_EXECUTABLE
    resolved = build_resolved_profile(action, "triple")
    assert not resolved.resolve_block("structure", "a.py").active
    assert not resolved.resolve_block("dependency", "a.py").active
    documentation = resolved.resolve_block("documentation", "a.py")
    assert documentation.active
    assert set(documentation.requested_field_paths) <= documentation_projectable_paths()
    assert "Describe this file for a maintainer." in documentation.text
    assert resolved.file_digest("a.py") != NO_PROMPT_PROFILE_DIGEST

def test_default_equivalent_single_only_profile_uses_builtin_triple_defaults():
    profile = validate_profile(
        default_prompt_profiles("single"),
        active_mode="triple",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )
    resolved = build_resolved_profile(classify_profile_action(profile, "triple"), "triple")
    assert resolved.file_digest("a.py") == NO_PROMPT_PROFILE_DIGEST
    assert not resolved.resolve_block("documentation", "a.py").active

def test_explicit_triple_may_omit_documentation():
    """triple.common requires only structure/dependency; documentation is optional
    there and falls back to built-in defaults when no single section is present."""
    raw = _custom_triple_missing_documentation()
    profile = validate_profile(
        raw,
        active_mode="triple",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )
    assert sorted(profile.triple.keys()) == ["dependency", "structure"]
    resolved = build_resolved_profile(classify_profile_action(profile, "triple"), "triple")
    assert resolved.resolve_block("structure", "a.py").active
    assert not resolved.resolve_block("documentation", "a.py").active

def test_explicit_triple_missing_documentation_projects_from_single():
    """When both 'single' and 'triple' are present and triple.common omits
    documentation, documentation resolves via projection of the single section."""
    raw = _custom_triple_missing_documentation()
    raw["single"] = default_prompt_profiles("single")["single"]
    raw["single"]["common"]["requested_shape"]["description"] = (
        "Projected description from single."
    )
    profile = validate_profile(
        raw,
        active_mode="triple",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )
    resolved = build_resolved_profile(classify_profile_action(profile, "triple"), "triple")
    documentation = resolved.resolve_block("documentation", "a.py")
    assert documentation.active
    assert "Projected description from single." in documentation.text
    assert set(documentation.requested_field_paths) <= documentation_projectable_paths()

@pytest.mark.parametrize("missing_agent", ["structure", "dependency"])
def test_explicit_triple_missing_structure_or_dependency_rejects(missing_agent):
    raw = default_prompt_profiles("triple")
    del raw["triple"]["common"][missing_agent]
    with pytest.raises(ConfigError, match=missing_agent):
        validate_profile(
            raw,
            active_mode="triple",
            known_extensions=frozenset({".py"}),
            source="inline",
            source_path=None,
        )

def test_documentation_projectable_paths_drift_guard():
    """Pins the registry-derived single/documentation-compatible field set. If
    this fails, the single or documentation registry changed and the projection
    ownership boundary (Workstream D) must be re-reviewed, not silently accepted."""
    assert documentation_projectable_paths() == frozenset(
        {"description", "role_in_system", "key_concepts", "usage_example"}
    )

def test_single_only_triple_profile_projects_per_extension_overrides():
    """Projection must carry per-extension overrides, not just the base block."""
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["requested_shape"]["description"] = "Base description."
    raw["single"]["per_extension"] = {
        ".js": {
            "requested_shape": {
                **raw["single"]["common"]["requested_shape"],
                "description": "JS-specific description.",
            }
        }
    }
    profile = validate_profile(
        raw,
        active_mode="triple",
        known_extensions=frozenset({".py", ".js"}),
        source="inline",
        source_path=None,
    )
    resolved = build_resolved_profile(classify_profile_action(profile, "triple"), "triple")
    py_doc = resolved.resolve_block("documentation", "a.py")
    js_doc = resolved.resolve_block("documentation", "a.js")
    assert "Base description." in py_doc.text
    assert "JS-specific description." in js_doc.text
    assert "Base description." not in js_doc.text

def test_triple_per_extension_documentation_requires_common_documentation():
    raw = _custom_triple_missing_documentation()
    raw["triple"]["per_extension"] = {
        ".py": {
            "structure": raw["triple"]["common"]["structure"],
            "dependency": raw["triple"]["common"]["dependency"],
            "documentation": {"requested_shape": {"description": "x"}},
        }
    }
    with pytest.raises(ConfigError, match="per_extension"):
        validate_profile(
            raw,
            active_mode="triple",
            known_extensions=frozenset({".py"}),
            source="inline",
            source_path=None,
        )
