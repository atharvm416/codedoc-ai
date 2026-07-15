"""0.12.1 — response_correction_enabled config key.

Default is ``false``; explicit ``true`` enables correction; the documented
boolean strings coerce through ``_coerce_strict_bool``; a non-boolean value and
``null`` fail closed; the generated config contains the key; and all four
maintained fixtures load with the key absent (defaulting to ``false``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoc.core.config_template import PUBLIC_CONFIG_KEYS, build_default_config
from codedoc.core.loader import DEFAULTS, load_config
from codedoc.utils.errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent


def test_default_is_false():
    assert DEFAULTS["response_correction_enabled"] is False
    assert load_config(ROOT)["response_correction_enabled"] is False


def test_explicit_true_enables():
    assert load_config(ROOT, {"response_correction_enabled": True})[
        "response_correction_enabled"
    ] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("yes", True), ("0", False), ("false", False), ("no", False)],
)
def test_boolean_strings_coerce(value, expected):
    resolved = load_config(ROOT, {"response_correction_enabled": value})
    assert resolved["response_correction_enabled"] is expected


@pytest.mark.parametrize("bad", ["maybe", "enabled", 1, 0, 2, None])
def test_non_boolean_and_null_fail_closed(bad):
    with pytest.raises(ConfigError):
        load_config(ROOT, {"response_correction_enabled": bad})


def test_generated_config_contains_key():
    generated = build_default_config()
    assert generated["response_correction_enabled"] is False
    assert "response_correction_enabled" in dict(PUBLIC_CONFIG_KEYS)


@pytest.mark.parametrize(
    "name", ["default", "alternative", "medium-risk", "per-extension"]
)
def test_maintained_fixtures_default_false_when_absent(name):
    raw = json.loads(
        (ROOT / "test_instructions" / f"codedoc.config.{name}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "response_correction_enabled" not in raw
    resolved = load_config(ROOT, raw)
    assert resolved["response_correction_enabled"] is False
