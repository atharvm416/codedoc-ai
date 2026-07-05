"""Focused 0.11.3 tests for the one-config/no-discovery contract."""

import pytest

from codedoc.core.loader import DEFAULTS, load_config
from codedoc.core.prompt_profiles import default_prompt_profiles, validate_profile
from codedoc.utils.errors import ConfigError


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
        known_languages=frozenset({"python"}),
        source="inline",
        source_path=None,
    )
    assert profile.schema_version == version


def test_flat_layout_and_reserved_per_extension_are_rejected():
    raw = default_prompt_profiles("single")
    raw["single"] = raw["single"]["common"]
    with pytest.raises(ConfigError, match="common"):
        validate_profile(raw, active_mode="single", known_languages=frozenset(), source="inline", source_path=None)

    raw = default_prompt_profiles("single")
    raw["single"]["per_extension"] = {}
    with pytest.raises(ConfigError, match="per_extension"):
        validate_profile(raw, active_mode="single", known_languages=frozenset(), source="inline", source_path=None)

