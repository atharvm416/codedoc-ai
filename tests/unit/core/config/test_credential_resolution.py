"""0.14.4 P1: focused regressions for the two-phase api_key precedence fix.

``load_config()``'s phase-2 credential resolution selects among three
sources in increasing precedence -- the ``codedoc.config.json`` ``api_key``,
the ``LLM_API_KEY`` environment variable, and the programmatic
``config_overrides`` ``api_key`` -- with the last *present* source winning.
Presence must be judged by ``is not None``, never by truthiness: a
present-but-empty or present-but-wrong-type candidate at any precedence
level must be validated and rejected, never silently treated as absent and
skipped in favor of a lower-precedence source.
"""

from __future__ import annotations

import json

import pytest

from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError


def test_empty_string_api_key_in_config_file_alone_is_rejected(tmp_path):
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"api_key": ""}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="api_key must be a non-empty string"):
        load_config(tmp_path, {"dry_run": True})


def test_zero_api_key_in_config_file_alone_is_rejected(tmp_path):
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"api_key": 0}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="api_key must be a non-empty string"):
        load_config(tmp_path, {"dry_run": True})


def test_zero_api_key_in_overrides_alone_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="api_key must be a non-empty string"):
        load_config(tmp_path, {"api_key": 0, "dry_run": True})


def test_null_api_key_is_accepted_as_genuinely_absent(tmp_path, monkeypatch):
    """None (JSON null, or simply omitted) is the one value that means
    'absent': it must never itself raise, and resolution proceeds exactly as
    if the key had not been set at all."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = load_config(tmp_path, {"api_key": None, "dry_run": True})
    assert config["api_key"] is None


def test_empty_override_fails_rather_than_deferring_to_a_valid_lower_precedence_env(
    tmp_path, monkeypatch
):
    """The exact bug: config_overrides.api_key="" is the highest-precedence
    candidate and is present. It must fail on its own terms -- never be
    treated as absent and silently pass resolution to LLM_API_KEY."""
    monkeypatch.setenv("LLM_API_KEY", "sk-lower-precedence-must-not-be-used")
    with pytest.raises(ConfigError, match="api_key must be a non-empty string"):
        load_config(tmp_path, {"api_key": "", "dry_run": True})


def test_empty_llm_api_key_env_fails_rather_than_deferring_to_a_valid_config_file(
    tmp_path, monkeypatch
):
    """LLM_API_KEY="" (explicitly set -- e.g. an unresolved CI secret) is
    present at its precedence level and outranks a valid lower-precedence
    config-file value; it must fail rather than silently letting the file
    value win."""
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"api_key": "sk-from-file-must-not-be-used"}), encoding="utf-8"
    )
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(ConfigError, match="api_key must be a non-empty string"):
        load_config(tmp_path, {"dry_run": True})


def test_valid_higher_precedence_sources_still_win_in_order(tmp_path, monkeypatch):
    """Non-regression: the ordinary case -- 'last present source wins', file
    lowest, then LLM_API_KEY, then config_overrides -- still resolves
    normally once every candidate is valid."""
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"api_key": "sk-from-file"}), encoding="utf-8"
    )
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    config = load_config(tmp_path, {"dry_run": True})
    assert config["api_key"] == "sk-from-file"

    monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
    config = load_config(tmp_path, {"dry_run": True})
    assert config["api_key"] == "sk-from-env"

    config = load_config(tmp_path, {"api_key": "sk-from-overrides", "dry_run": True})
    assert config["api_key"] == "sk-from-overrides"
