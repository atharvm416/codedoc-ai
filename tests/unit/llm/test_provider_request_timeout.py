"""Section 5.5: ``provider_request_timeout_s`` across config, environment,
CLI, init-config generation, and the resolved factory-to-adapter wiring.

A transport timeout per connect/read/write/pool phase, not a wall-clock
deadline for the whole call: OpenAI/Anthropic clients receive
``timeout=<float>, max_retries=0``; Gemini receives millisecond
``HttpOptions.timeout`` and ``HttpRetryOptions(attempts=1)`` -- so every
adapter invocation is one transport attempt after SDK retries are disabled.
"""

from __future__ import annotations

import pytest

from codedoc.utils.errors import ConfigError
from tests.support.configuration_cases import _load

# ---------------------------------------------------------------------------
# Config layer: default, valid forms, strict rejection
# ---------------------------------------------------------------------------


def test_default_is_120(tmp_path):
    cfg = _load(tmp_path)
    assert cfg["provider_request_timeout_s"] == 120.0


def test_explicit_null_fails_boundedly_without_echoing(tmp_path):
    """Section 12.1 C3: explicit JSON/Python null is a distinct, invalid
    value -- it must never silently become the default -- and the error
    must not echo the rejected value."""
    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, provider_request_timeout_s=None)
    assert excinfo.value.reason == (
        "provider_request_timeout_s must be a number between 1 and 600 (inclusive)."
    )


def test_empty_string_fails_boundedly_without_echoing(tmp_path):
    """Section 12.1 C3: an empty string from the config file or CLI text is
    invalid, not absent -- unlike an empty environment variable, which is
    filtered out upstream before it ever reaches this validation."""
    with pytest.raises(ConfigError) as excinfo:
        _load(tmp_path, provider_request_timeout_s="")
    assert excinfo.value.reason == (
        "provider_request_timeout_s must be a number between 1 and 600 (inclusive)."
    )
    assert "''" not in str(excinfo.value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, 1.0),
        (600, 600.0),
        (45, 45.0),
        (45.5, 45.5),
        ("45", 45.0),
        ("45.5", 45.5),
        ("1", 1.0),
        ("600", 600.0),
    ],
)
def test_accepts_ascii_decimal_strings_and_non_boolean_numbers(tmp_path, value, expected):
    cfg = _load(tmp_path, provider_request_timeout_s=value)
    assert cfg["provider_request_timeout_s"] == expected
    assert isinstance(cfg["provider_request_timeout_s"], float)


@pytest.mark.parametrize(
    "bad_value",
    [
        True,
        False,
        "",
        "0",
        "0.5",
        "601",
        "-5",
        "+5",
        "1e2",
        "1E2",
        "5_0",
        "5.0.0",
        "abc",
        "۵۰",  # extended Arabic-Indic digits "50"
        "٥٠",  # Arabic-Indic digits "50"
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_rejects_everything_outside_the_strict_domain(tmp_path, bad_value):
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        _load(tmp_path, provider_request_timeout_s=bad_value)


def test_error_message_never_echoes_rejected_input(tmp_path):
    from codedoc.utils.errors import ConfigError

    sentinel = "not-a-number-sentinel-xyz"
    with pytest.raises(ConfigError) as caught:
        _load(tmp_path, provider_request_timeout_s=sentinel)
    assert sentinel not in str(caught.value)


def test_out_of_range_is_rejected_without_echo(tmp_path):
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError) as caught:
        _load(tmp_path, provider_request_timeout_s=700)
    assert "700" not in str(caught.value)


def test_empty_env_value_retains_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDOC_PROVIDER_REQUEST_TIMEOUT_S", "")
    cfg = _load(tmp_path)
    assert cfg["provider_request_timeout_s"] == 120.0


def test_set_env_value_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDOC_PROVIDER_REQUEST_TIMEOUT_S", "200")
    cfg = _load(tmp_path)
    assert cfg["provider_request_timeout_s"] == 200.0


# ---------------------------------------------------------------------------
# CLI: flag flows into overrides, --init-config guard
# ---------------------------------------------------------------------------


def test_cli_flag_is_defined_and_flows_to_overrides():
    from codedoc.cli.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["some/project", "--provider-request-timeout-s", "77"])
    assert args.provider_request_timeout_s == "77"


def test_cli_flag_combined_with_init_config_is_rejected(capsys):
    from codedoc.cli.cli import run_cli

    code = run_cli(["--init-config", "--provider-request-timeout-s", "77"])
    assert code != 0
    err = capsys.readouterr().err
    assert "--init-config" in err


# ---------------------------------------------------------------------------
# init-config generation: template key present with the correct default
# ---------------------------------------------------------------------------


def test_init_config_template_includes_the_key_with_default():
    from codedoc.core.config_template import (
        PUBLIC_CONFIG_KEYS,
        build_default_config,
    )

    keys = [key for key, _description in PUBLIC_CONFIG_KEYS]
    assert "provider_request_timeout_s" in keys

    generated = build_default_config()
    assert generated["provider_request_timeout_s"] == 120


# ---------------------------------------------------------------------------
# Factory-to-adapter wiring: resolved timeout reaches each SDK client
# ---------------------------------------------------------------------------


def test_openai_adapter_receives_resolved_timeout_and_disables_sdk_retries(monkeypatch):
    from tests.support.providers import _install_openai
    from codedoc.llm.factory import create_provider

    rec = {}
    _install_openai(monkeypatch, rec)
    create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "openai",
            "model_name": "gpt-test",
            "api_key": "k",
            "provider_request_timeout_s": 45.0,
        }
    )
    assert rec["init"]["timeout"] == 45.0
    assert rec["init"]["max_retries"] == 0


def test_anthropic_adapter_receives_resolved_timeout_and_disables_sdk_retries(monkeypatch):
    from tests.support.providers import _install_anthropic
    from codedoc.llm.factory import create_provider

    rec = {}
    _install_anthropic(monkeypatch, rec)
    create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "anthropic",
            "model_name": "claude-test",
            "api_key": "k",
            "provider_request_timeout_s": 45.0,
        }
    )
    assert rec["init"]["timeout"] == 45.0
    assert rec["init"]["max_retries"] == 0


def test_gemini_adapter_receives_millisecond_timeout_and_single_attempt(monkeypatch):
    from tests.support.providers import _install_gemini
    from codedoc.llm.factory import create_provider

    rec = {}
    _install_gemini(monkeypatch, rec)
    create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "gemini",
            "model_name": "gemini-test",
            "api_key": "k",
            "provider_request_timeout_s": 45.0,
        }
    )
    http_options = rec["init"]["http_options"]
    assert http_options.__dict__["timeout"] == 45000
    retry_options = http_options.__dict__["retry_options"]
    assert retry_options.__dict__["attempts"] == 1


def test_default_timeout_reaches_adapters_when_config_omits_it(monkeypatch):
    from tests.support.providers import _install_openai
    from codedoc.llm.factory import create_provider

    rec = {}
    _install_openai(monkeypatch, rec)
    create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "openai",
            "model_name": "gpt-test",
            "api_key": "k",
        }
    )
    assert rec["init"]["timeout"] == 120


def test_direct_adapter_construction_uses_safe_default_without_timeout_kwarg(monkeypatch):
    """Adapter constructor parameters are keyword-only with safe defaults for
    direct library construction (section 5.5) -- a caller that never passes
    timeout/max_retries still gets a working, correctly-defaulted client."""
    from tests.support.providers import _install_openai
    from codedoc.llm.api_provider import OpenAIProvider

    rec = {}
    _install_openai(monkeypatch, rec)
    OpenAIProvider(api_key="k", model="gpt-test")
    assert rec["init"]["timeout"] == 120.0
    assert rec["init"]["max_retries"] == 0


def test_gemini_direct_construction_default_retry_attempts_is_one(monkeypatch):
    from tests.support.providers import _install_gemini
    from codedoc.llm.api_provider import GeminiProvider

    rec = {}
    _install_gemini(monkeypatch, rec)
    GeminiProvider(api_key="k", model="gemini-test")
    http_options = rec["init"]["http_options"]
    assert http_options.__dict__["timeout"] == 120000
    assert http_options.__dict__["retry_options"].__dict__["attempts"] == 1
