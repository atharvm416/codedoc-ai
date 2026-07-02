"""Focused 0.11.3 tests for canonical config generation."""

import json

import pytest

from codedoc.core.config_template import (
    GENERATOR_EXCLUDED_KEYS,
    PUBLIC_CONFIG_KEYS,
    build_default_config,
    init_config,
    init_instructions,
)
from codedoc.core.loader import DEFAULTS, load_config
from codedoc.utils.errors import ConfigError


def test_public_default_registry_is_complete_and_generated_config_round_trips(tmp_path):
    generated = build_default_config()
    public_keys = [key for key, _description in PUBLIC_CONFIG_KEYS]
    assert len(public_keys) == len(set(public_keys))
    assert set(DEFAULTS) == set(public_keys) | (set(DEFAULTS) & GENERATOR_EXCLUDED_KEYS)
    assert list(generated) == public_keys
    assert generated["api_key"] is None
    assert "supported_extensions" not in generated
    assert not (set(generated) & GENERATOR_EXCLUDED_KEYS)
    profiles = generated["prompt_profiles"]
    assert profiles["schema_version"] == 2
    assert set(profiles["single"]) == {"common", "per_language"}
    assert set(profiles["triple"]) == {"common", "per_language"}
    assert "per_extension" not in json.dumps(profiles)

    init_config(tmp_path, None, False)
    resolved = load_config(tmp_path)
    assert resolved["analysis_mode"] == DEFAULTS["analysis_mode"]
    assert resolved["prompt_profiles"] == profiles


def test_custom_name_normalization_and_force_create_no_backup(tmp_path):
    result = init_config(tmp_path, "team-example", False)
    assert result.path.name == "team-example.json"
    assert "reads only codedoc.config.json" in result.message
    with pytest.raises(ConfigError, match="--force"):
        init_config(tmp_path, "team-example.json", False)
    result.path.write_text("old", encoding="utf-8")
    init_config(tmp_path, "team-example.json", True)
    assert json.loads(result.path.read_text(encoding="utf-8"))["api_key"] is None
    assert not list(tmp_path.glob("*.bak-*"))


@pytest.mark.parametrize("name", ["../escape", "folder/name", "CON", "x.json.json"])
def test_custom_name_rejects_nonportable_or_double_extension(name, tmp_path):
    with pytest.raises(ConfigError):
        init_config(tmp_path, name, False)


def test_instruction_initializer_preserves_other_settings(tmp_path):
    path = tmp_path / "codedoc.config.json"
    path.write_text('{"output_format":"md","prompt_profiles":null}', encoding="utf-8")
    init_instructions(tmp_path, "single", False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["output_format"] == "md"
    assert set(data["prompt_profiles"]) == {"schema_version", "single"}

