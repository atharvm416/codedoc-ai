"""0.9.6 — configuration bound hardening and the follow_symlinks safety key."""

from __future__ import annotations

import pytest

from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError


# ---------------------------------------------------------------------------
# max_file_size_kb
# ---------------------------------------------------------------------------

def test_zero_max_file_size_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"max_file_size_kb": 0})


def test_negative_max_file_size_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"max_file_size_kb": -10})


def test_boolean_max_file_size_rejected(tmp_path):
    # True would otherwise coerce to 1; reject it explicitly.
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"max_file_size_kb": True})


def test_valid_max_file_size_unchanged(tmp_path):
    config = load_config(tmp_path, {"max_file_size_kb": 250})
    assert config["max_file_size_kb"] == 250


# ---------------------------------------------------------------------------
# retry_after_cap_s
# ---------------------------------------------------------------------------

def test_negative_retry_after_cap_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"retry_after_cap_s": -1})


def test_boolean_retry_after_cap_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"retry_after_cap_s": True})


def test_zero_retry_after_cap_valid(tmp_path):
    config = load_config(tmp_path, {"retry_after_cap_s": 0})
    assert config["retry_after_cap_s"] == 0


def test_positive_retry_after_cap_valid(tmp_path):
    config = load_config(tmp_path, {"retry_after_cap_s": 45})
    assert config["retry_after_cap_s"] == 45


# ---------------------------------------------------------------------------
# follow_symlinks
# ---------------------------------------------------------------------------

def test_follow_symlinks_defaults_to_false(tmp_path):
    config = load_config(tmp_path)
    assert config["follow_symlinks"] is False


def test_follow_symlinks_accepts_bool(tmp_path):
    assert load_config(tmp_path, {"follow_symlinks": True})["follow_symlinks"] is True


@pytest.mark.parametrize("value,expected", [
    ("true", True),
    ("1", True),
    ("yes", True),
    ("false", False),
    ("0", False),
    ("no", False),
])
def test_follow_symlinks_accepts_documented_strings(tmp_path, value, expected):
    config = load_config(tmp_path, {"follow_symlinks": value})
    assert config["follow_symlinks"] is expected


def test_follow_symlinks_rejects_unrecognized_string(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"follow_symlinks": "sometimes"})
