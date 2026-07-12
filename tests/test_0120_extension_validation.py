"""0.12.0 — deterministic ``per_extension`` validation at config-load time."""

import pytest

from codedoc.core.prompt_profiles import (
    MAX_COMMENT_LIST_ITEMS,
    MAX_EXTENSION_OVERRIDES_PER_MODE,
    validate_profile,
)
from codedoc.utils.errors import ConfigError

# Final segments of every accepted key below must be configured.
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
