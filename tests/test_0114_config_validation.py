"""Fail-closed configuration validation regressions."""

import math

import pytest

from codedoc.core.config_template import init_config
from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError


def test_unknown_keys_are_rejected_from_file_and_overrides(tmp_path):
    path = tmp_path / "codedoc.config.json"
    path.write_text('{"output_fomat":"md","dryrun":true}', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"(?s)dryrun.*output_fomat"):
        load_config(tmp_path)
    path.unlink()
    with pytest.raises(ConfigError, match="max_file"):
        load_config(tmp_path, {"max_file": 1})


@pytest.mark.parametrize(
    "key",
    [
        "parallel_agents",
        "follow_symlinks",
        "propagate_changes",
        "rate_limit_adaptive",
        "respect_retry_after",
        "dry_run",
        "allow_partial",
    ],
)
@pytest.mark.parametrize("value", ["garbage", 0, 1, 2, None, [], {}])
def test_all_boolean_settings_are_strict(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: value})


@pytest.mark.parametrize(
    "key",
    [
        "max_file_size_kb",
        "max_parallel_files",
        "file_retry_attempts",
        "max_consecutive_failures",
        "retry_after_cap_s",
        "max_content_chars",
        "max_files",
    ],
)
@pytest.mark.parametrize("value", [True, 1.0])
def test_integer_settings_reject_booleans_and_floats(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: value})


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_numbers_are_rejected(tmp_path, token):
    (tmp_path / "codedoc.config.json").write_text(
        f'{{"truncation_head_ratio":{token}}}', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="non-finite"):
        load_config(tmp_path)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_in_memory_numbers_are_rejected(tmp_path, value):
    with pytest.raises(ConfigError, match="non-finite"):
        load_config(tmp_path, {"rate_limit_backoff_s": value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("skip_dirs", [1]),
        ("ignore_paths", [""]),
        ("extension_language_map", {"py": "python"}),
        ("extension_language_map_add", {".x": ""}),
        ("provider_prefixes", {"openai": "gpt-"}),
        ("provider_prefixes_add", {"openai": [1]}),
        ("rate_limit_signals_add", [None]),
    ],
)
def test_nested_collection_members_are_validated(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: value})


def test_generated_config_round_trips(tmp_path):
    init_config(tmp_path)
    assert "schema_version" not in load_config(tmp_path)["prompt_profiles"]
