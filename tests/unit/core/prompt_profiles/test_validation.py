"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
from codedoc.core.prompt_profiles import (
    MAX_COMMENT_LIST_ITEMS,
    MAX_EXTENSION_OVERRIDES_PER_MODE,
    validate_profile,
)
from codedoc.utils.errors import ConfigError
from codedoc.core.loader import DEFAULTS, load_config
from codedoc.core.prompt_profiles import default_prompt_profiles
from codedoc.agents.dependency_agent import build_prompt as build_dependency_prompt
from codedoc.agents.documentation_agent import build_prompt as build_documentation_prompt
from codedoc.agents.file_documentation_agent import build_prompt as build_combined_prompt
from codedoc.agents.structure_agent import build_prompt as build_structure_prompt
from tests.support.versionless_documents import _assert_versionless
from tests.support.versionless_documents import _validate as validate_versionless

KNOWN = frozenset({".py", ".js", ".cs", ".ts"})

def _single(per_extension):
    return {
        "single": {
            "common": {"requested_shape": {"description": "base description"}},
            "per_extension": per_extension,
        }
    }

def _override(desc="ext description"):
    return {"requested_shape": {"description": desc}}

def _validate(raw, known=KNOWN, mode="single"):
    return validate_profile(
        raw, active_mode=mode, known_extensions=known,
        source="inline", source_path=None,
    )

@pytest.mark.parametrize("key", [".py", ".js", ".cs", ".d.ts", ".test.js"])
def test_accepted_extension_keys(key):
    profile = _validate(_single({key: _override()}))
    assert set(profile.single.per_extension) == {key}

@pytest.mark.parametrize(
    "key",
    [
        ".PY",       # uppercase
        "js",        # no leading dot
        ".",         # dot only
        ".py.",      # trailing dot
        ".p y",      # whitespace
        ".p/y",      # path separator
        ".p*",       # glob metacharacter
        "a.py",      # does not start with a dot
        "",          # empty
        "." + "a" * 33,  # 34 chars, exceeds the 32-char cap
        ".pyy",      # final segment not a configured extension
    ],
)
def test_rejected_extension_keys(key):
    with pytest.raises(ConfigError):
        _validate(_single({key: _override()}))

def test_too_many_extension_overrides_rejected():
    overrides = {f".a{i}.py": _override() for i in range(MAX_EXTENSION_OVERRIDES_PER_MODE + 1)}
    with pytest.raises(ConfigError, match="at most"):
        _validate(_single(overrides))

def test_exactly_max_extension_overrides_accepted():
    overrides = {f".a{i}.py": _override() for i in range(MAX_EXTENSION_OVERRIDES_PER_MODE)}
    profile = _validate(_single(overrides))
    assert len(profile.single.per_extension) == MAX_EXTENSION_OVERRIDES_PER_MODE

def test_comment_list_boundary_is_preserved():
    at_limit = _single({})
    at_limit["$comment"] = ["comment"] * MAX_COMMENT_LIST_ITEMS
    _validate(at_limit)

    over_limit = _single({})
    over_limit["$comment"] = ["comment"] * (MAX_COMMENT_LIST_ITEMS + 1)
    with pytest.raises(ConfigError, match="comment list has too many items"):
        _validate(over_limit)

def test_serialized_byte_cap_runs_before_section_validation():
    # A ~320 KB profile with two per_extension entries is rejected by the byte-cap
    # message (ordering, not precedence) — long before any per-entry validation.
    big = "x" * 160_000
    raw = _single({
        ".py": {"requested_shape": {"description": big}},
        ".js": {"requested_shape": {"description": big}},
    })
    with pytest.raises(ConfigError, match="exceeds .* bytes"):
        _validate(raw)

def test_single_override_missing_shape_key_rejected():
    with pytest.raises(ConfigError, match="requested_shape"):
        _validate(_single({".py": {}}))

def test_triple_override_missing_agent_key_rejected():
    raw = {
        "triple": {
            "common": {
                "structure": {"requested_shape": {}},
                "dependency": {"requested_shape": {}},
                "documentation": {"requested_shape": {"description": "d"}},
            },
            "per_extension": {
                ".cs": {
                    "structure": {"requested_shape": {}},
                    "dependency": {"requested_shape": {}},
                    # documentation missing
                }
            },
        }
    }
    with pytest.raises(ConfigError, match="three agent keys"):
        _validate(raw, mode="triple")

def test_triple_per_extension_requires_common_documentation():
    raw = {
        "triple": {
            "common": {
                "structure": {"requested_shape": {}},
                "dependency": {"requested_shape": {}},
            },
            "per_extension": {
                ".cs": {
                    "structure": {"requested_shape": {}},
                    "dependency": {"requested_shape": {}},
                    "documentation": {"requested_shape": {"description": "d"}},
                }
            },
        }
    }
    with pytest.raises(ConfigError, match="documentation"):
        _validate(raw, mode="triple")

def test_mixed_syntax_override_rejected():
    # common is requested_shape (v2); a per_extension override using v1 'fields'
    # is caught by the wrong-version-key guard.
    raw = {
        "single": {
            "common": {"requested_shape": {"description": "base"}},
            "per_extension": {
                ".py": {"fields": [
                    {"key": "description", "type": "string", "instruction": "x"}
                ]}
            },
        }
    }
    with pytest.raises(ConfigError, match="version"):
        _validate(raw)

@pytest.mark.parametrize("bad_key", ["per_file", "per_category"])
def test_unknown_section_keys_rejected_naming_accepted_set(bad_key):
    raw = {
        "single": {
            "common": {"requested_shape": {"description": "base"}},
            bad_key: {},
        }
    }
    with pytest.raises(ConfigError, match=r"\{'common', 'per_extension'\}"):
        _validate(raw)

def test_alternate_config_and_external_profile_are_not_discovered(tmp_path):
    alternate = tmp_path / "config.json"
    external = tmp_path / "codedoc-prompt-profiles.json"
    alternate.write_text("not json", encoding="utf-8")
    external.write_text("not json", encoding="utf-8")
    before = {path: path.read_bytes() for path in (alternate, external)}
    resolved = load_config(tmp_path)
    assert resolved["analysis_mode"] == DEFAULTS["analysis_mode"]
    assert {path: path.read_bytes() for path in before} == before

@pytest.mark.parametrize(
    "key",
    [
        "safe_mode",
        "manage_output_gitignore",
        "output_gitignore_filename",
        "prompt_profile_file",
        "prompt_profile_auto_detect",
        "prompt_profile_disabled",
        "prompt_customization_allow_risky",
    ],
)
def test_removed_config_keys_fail_with_targeted_guidance(tmp_path, key):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: True})

@pytest.mark.parametrize("version", [1, 2])
def test_both_inline_schema_versions_validate_under_common(version):
    raw = default_prompt_profiles(schema_version=version)
    profile = validate_profile(
        raw,
        active_mode="triple",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )
    assert profile.schema_version == version

def test_flat_layout_is_rejected_and_per_extension_is_accepted():
    raw = default_prompt_profiles("single")
    raw["single"] = raw["single"]["common"]
    with pytest.raises(ConfigError, match="common"):
        validate_profile(raw, active_mode="single", known_extensions=frozenset(), source="inline", source_path=None)

    # An empty per_extension is accepted (the generated default carries it).
    raw = default_prompt_profiles("single")
    profile = validate_profile(
        raw, active_mode="single", known_extensions=frozenset({".py"}),
        source="inline", source_path=None,
    )
    assert profile.single.per_extension == {}

    # A populated, registry-valid per_extension override is accepted.
    raw = default_prompt_profiles("single")
    raw["single"]["per_extension"] = {
        ".js": {"requested_shape": {"description": "JS-specific summary."}}
    }
    profile = validate_profile(
        raw, active_mode="single", known_extensions=frozenset({".py", ".js"}),
        source="inline", source_path=None,
    )
    assert set(profile.single.per_extension) == {".js"}

@pytest.mark.parametrize("bad_key", ["per_file", "per_category"])
def test_unknown_section_keys_name_the_accepted_set(bad_key):
    raw = default_prompt_profiles("single")
    raw["single"][bad_key] = {}
    with pytest.raises(ConfigError, match=r"\{'common', 'per_extension'\}"):
        validate_profile(
            raw, active_mode="single", known_extensions=frozenset({".py"}),
            source="inline", source_path=None,
        )

@pytest.mark.parametrize("version", [1, 2])
def test_explicit_legacy_profile_versions_remain_readable(version):
    raw = default_prompt_profiles("single", schema_version=version)
    _assert_versionless(raw)
    raw["schema_version"] = version
    assert raw["schema_version"] == version
    assert validate_versionless(raw).schema_version == version

def test_mixed_versionless_profile_syntax_fails_closed():
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["fields"] = []
    with pytest.raises(ConfigError, match="both 'fields'.*'requested_shape'"):
        validate_versionless(raw)

def test_provider_requested_shapes_are_versionless():
    prompts = [
        build_combined_prompt("main.py", "x = 1", [], "python"),
        build_structure_prompt("main.py", "x = 1", [], "python"),
        build_dependency_prompt("main.py", "x = 1", [], "python"),
        build_documentation_prompt("main.py", "x = 1", "python", {}, {}),
    ]
    for system, prompt in prompts:
        assert "schema_version" not in system
        assert "schema_version" not in prompt
