"""Focused 0.11.3 tests for canonical config generation."""

import json

import pytest

from codedoc.core.config_template import (
    GENERATOR_EXCLUDED_KEYS,
    PUBLIC_CONFIG_KEYS,
    build_default_config,
    init_config,
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

    init_config(tmp_path)
    resolved = load_config(tmp_path)
    assert resolved["analysis_mode"] == DEFAULTS["analysis_mode"]
    assert resolved["prompt_profiles"] == profiles


def test_force_refreshes_only_profiles_and_creates_no_backup(tmp_path):
    result = init_config(tmp_path)
    assert result.path.name == "codedoc.config.json"
    with pytest.raises(ConfigError, match="--force"):
        init_config(tmp_path)
    data = json.loads(result.path.read_text(encoding="utf-8"))
    data["output_format"] = "md"
    data["prompt_profiles"]["single"]["common"]["requested_shape"]["description"] = (
        "custom"
    )
    result.path.write_text(json.dumps(data), encoding="utf-8")
    init_config(tmp_path, force=True)
    refreshed = json.loads(result.path.read_text(encoding="utf-8"))
    assert refreshed["output_format"] == "md"
    assert refreshed["prompt_profiles"] == build_default_config()["prompt_profiles"]
    assert not list(tmp_path.glob("*.bak-*"))
