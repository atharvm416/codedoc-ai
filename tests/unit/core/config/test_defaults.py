"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from codedoc.core.config_template import (
    PUBLIC_CONFIG_KEYS,
    build_default_config,
    init_config,
)
from codedoc.core.loader import DEFAULTS, load_config
from codedoc.utils.errors import ConfigError
from codedoc.agents.orchestrator import initial_calls_per_file

ROOT = Path(__file__).resolve().parents[4]


# --- Section 5.5 / 8.G: the four configuration states, proven individually ---

def test_state_1_absent_key_and_no_override_resolves_to_true():
    """No project key, no override -> the new default is on."""
    assert DEFAULTS["response_correction_enabled"] is True
    # The repository-root codedoc.config.json is tracked and does NOT contain
    # the key, so this resolves through DEFAULTS.
    assert load_config(ROOT)["response_correction_enabled"] is True


def test_state_2_explicit_true_resolves_to_true():
    assert load_config(ROOT, {"response_correction_enabled": True})[
        "response_correction_enabled"
    ] is True


def test_state_3_explicit_false_resolves_to_false():
    assert load_config(ROOT, {"response_correction_enabled": False})[
        "response_correction_enabled"
    ] is False


def test_state_4_older_generated_config_with_explicit_false_stays_false_unrewritten(
    tmp_path,
):
    """An older complete generated configuration whose then-default value was
    written as ``false`` remains ``False`` and is NOT rewritten by normal
    loading (section 5.5 lines 775-788)."""
    older = build_default_config()
    older["response_correction_enabled"] = False  # simulate the pre-flip default
    target = tmp_path / "codedoc.config.json"
    target.write_text(json.dumps(older, indent=2) + "\n", encoding="utf-8")
    before = target.read_bytes()

    assert load_config(tmp_path)["response_correction_enabled"] is False
    # Normal loading never migrates the value or touches the file.
    assert target.read_bytes() == before


def test_init_config_force_carries_an_explicit_false_forward(tmp_path):
    """A user-invoked ``--init-config --force`` rewrite may canonicalize the
    file but must carry an existing explicit ``false`` forward -- CodeDoc cannot
    tell an old generated value from a deliberate cost opt-out (section 5.5
    lines 783-788)."""
    cfg = build_default_config()
    cfg["response_correction_enabled"] = False
    target = tmp_path / "codedoc.config.json"
    target.write_text(json.dumps(cfg), encoding="utf-8")  # deliberately un-canonical

    result = init_config(tmp_path, force=True)
    assert result.path == target

    rewritten = json.loads(target.read_text(encoding="utf-8"))
    assert rewritten["response_correction_enabled"] is False
    assert load_config(tmp_path)["response_correction_enabled"] is False


# --- coercion / fail-closed behaviour is unchanged by the flip ---

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


# --- the generated template states the new default and its cost ---

def test_generated_config_emits_true_and_documents_the_cost():
    generated = build_default_config()
    assert generated["response_correction_enabled"] is True
    assert "response_correction_enabled" in dict(PUBLIC_CONFIG_KEYS)

    description = dict(PUBLIC_CONFIG_KEYS)["response_correction_enabled"]
    lowered = description.lower()
    assert "enabled by default" in lowered
    assert "one" in lowered and "per rejected" in lowered  # one extra call per rejection
    assert "100%" in description                            # the real worst case
    assert "2.6%" not in description                        # not the historical figure
    assert "retries" in lowered and "additional" in lowered
    assert "max_planned_calls" in description
    assert "not a hard" in lowered and "ceiling" in lowered
    assert "false" in lowered and (
        "opt-out" in lowered or "zero-correction" in lowered
    )
    # The description is Python-only guidance, never emitted into the strict
    # commentless generated JSON.
    assert description not in json.dumps(generated)


@pytest.mark.parametrize(
    "name", ["default", "alternative", "medium-risk", "per-extension"]
)
def test_maintained_fixtures_default_true_when_key_absent(name):
    raw = json.loads(
        (ROOT / "test_instructions" / f"codedoc.config.{name}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "response_correction_enabled" not in raw
    resolved = load_config(ROOT, raw)
    assert resolved["response_correction_enabled"] is True


def test_initial_calls_per_file_values():
    assert initial_calls_per_file("single") == 1
    assert initial_calls_per_file("triple") == 3
    # Anything not explicitly triple resolves to one call (single default).
    assert initial_calls_per_file("anything-else") == 1


def test_large_file_strategy_is_public_and_defaults_to_truncate():
    generated = build_default_config()

    assert DEFAULTS["large_file_strategy"] == "truncate"
    assert generated["large_file_strategy"] == "truncate"
    assert "large_file_strategy" in dict(PUBLIC_CONFIG_KEYS)
