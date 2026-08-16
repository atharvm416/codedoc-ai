"""0.14.4 P0: focused regressions for the runtime endpoint-authorization gate.

A project-controlled codedoc.config.json can set api_base_url; nothing in it
may ever authorize sending the user's credential, source, or prompts there.
Authorization is accepted only from --trust-api-base-url or
CODEDOC_TRUST_API_BASE_URL, resolved before any credential is read or
selected, and covers dry runs identically to real runs.
"""

from __future__ import annotations

import os

import pytest

from codedoc.core.loader import ResolvedConfig, load_config, validate_config_data
from codedoc.llm.factory import (
    ENDPOINT_TRUST_ATTRIBUTE,
    EndpointTrustAttestation,
    create_provider,
    effective_endpoint_identity,
)
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError, ProviderInitError


_URL = "http://example.com:9999/v1"
_OTHER_URL = "http://other.example.com/v1"


# ---------------------------------------------------------------------------
# load_config(): the authorization gate itself
# ---------------------------------------------------------------------------


def test_default_endpoint_needs_no_approval(tmp_path):
    config = load_config(tmp_path, {"dry_run": True})
    assert isinstance(config, ResolvedConfig)
    assert config.endpoint_trust is None


def test_approval_absent_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no runtime approval"):
        load_config(tmp_path, {"api_base_url": _URL, "dry_run": True})


def test_approval_present_but_mismatched_digest_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="does not match"):
        load_config(
            tmp_path,
            {"api_base_url": _URL, "dry_run": True},
            trust_api_base_url=_OTHER_URL,
        )


def test_approval_present_and_matching_succeeds(tmp_path):
    config = load_config(
        tmp_path,
        {"api_base_url": _URL, "dry_run": True},
        trust_api_base_url=_URL,
    )
    assert config.endpoint_trust is not None
    assert config.endpoint_trust.digest == effective_endpoint_identity(_URL)
    assert config.endpoint_trust.mechanism == "--trust-api-base-url"


def test_approval_via_environment_variable_succeeds_and_cli_wins_when_both_set(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CODEDOC_TRUST_API_BASE_URL", _URL)
    config = load_config(tmp_path, {"api_base_url": _URL, "dry_run": True})
    assert config.endpoint_trust.mechanism == "CODEDOC_TRUST_API_BASE_URL"

    config = load_config(
        tmp_path,
        {"api_base_url": _URL, "dry_run": True},
        trust_api_base_url=_URL,
    )
    assert config.endpoint_trust.mechanism == "--trust-api-base-url"


@pytest.mark.parametrize("key", ["trust_api_base_url", "trusted_api_base_url", "_endpoint_trust"])
def test_approval_supplied_through_config_file_is_rejected(tmp_path, key):
    (tmp_path / "codedoc.config.json").write_text(
        f'{{"api_base_url": "{_URL}", "{key}": "{_URL}"}}', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="endpoint-trust"):
        load_config(tmp_path, {"dry_run": True})


@pytest.mark.parametrize("key", ["trust_api_base_url", "trusted_api_base_url", "_endpoint_trust"])
def test_approval_supplied_through_config_overrides_is_rejected(tmp_path, key):
    with pytest.raises(ConfigError, match="endpoint-trust"):
        load_config(tmp_path, {"api_base_url": _URL, "dry_run": True, key: _URL})


@pytest.mark.parametrize("bad_value", [123, 4.5, True, [], {}])
def test_wrong_type_cli_trust_value_raises_config_error_not_attribute_error(
    tmp_path, bad_value
):
    """0.14.4 audit fix: a non-string --trust-api-base-url (e.g. a caller
    passing an int) must raise ConfigError, never a raw AttributeError from
    blindly calling .strip() on it."""
    with pytest.raises(ConfigError, match="non-empty string"):
        load_config(
            tmp_path,
            {"api_base_url": _URL, "dry_run": True},
            trust_api_base_url=bad_value,
        )


@pytest.mark.parametrize("blank_value", ["", "   "])
def test_blank_cli_trust_value_raises_and_never_falls_back_to_the_environment(
    tmp_path, monkeypatch, blank_value
):
    """0.14.4 audit fix: a present-but-blank --trust-api-base-url must fail
    on its own terms. Reproduces the exact P0 regression: the environment
    carries a *valid, matching* approval, so the old truthiness check would
    have silently fallen through and authorized the run anyway."""
    monkeypatch.setenv("CODEDOC_TRUST_API_BASE_URL", _URL)
    with pytest.raises(ConfigError, match="non-empty string"):
        load_config(
            tmp_path,
            {"api_base_url": _URL, "dry_run": True},
            trust_api_base_url=blank_value,
        )


def test_present_cli_trust_value_wins_even_when_it_makes_authorization_fail(
    tmp_path, monkeypatch
):
    """Presence, not validity, decides precedence: an explicitly supplied
    (mismatched) CLI approval must still win over a valid environment
    approval and cause the expected mismatch failure -- not quietly succeed
    by using the environment's value instead."""
    monkeypatch.setenv("CODEDOC_TRUST_API_BASE_URL", _URL)
    with pytest.raises(ConfigError, match="does not match"):
        load_config(
            tmp_path,
            {"api_base_url": _URL, "dry_run": True},
            trust_api_base_url=_OTHER_URL,
        )


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://user:pass@example.com/v1",
        "http://user@example.com/v1",
        "http://example.com/v1?token=x",
        "http://example.com/v1#frag",
    ],
)
def test_api_base_url_carrying_forbidden_components_is_rejected(tmp_path, bad_url):
    with pytest.raises(ConfigError, match="username, password, query string"):
        load_config(tmp_path, {"api_base_url": bad_url, "dry_run": True})


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://user:pass@example.com:9999/v1",
        "http://example.com:9999/v1?token=x",
        "http://example.com:9999/v1#frag",
    ],
)
def test_approval_url_carrying_forbidden_components_is_rejected(tmp_path, bad_url):
    with pytest.raises(ConfigError, match="username, password, query string"):
        load_config(
            tmp_path,
            {"api_base_url": _URL, "dry_run": True},
            trust_api_base_url=bad_url,
        )


def test_approval_supplied_while_api_base_url_empty_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no api_base_url is configured"):
        load_config(tmp_path, {"dry_run": True}, trust_api_base_url=_URL)


def test_config_file_supplying_endpoint_and_key_together_still_refused(tmp_path):
    """A project-controlled config file can set api_base_url and api_key
    together; neither field, alone or combined, ever satisfies the gate."""
    (tmp_path / "codedoc.config.json").write_text(
        f'{{"api_base_url": "{_URL}", "api_key": "sk-from-config-file"}}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="no runtime approval"):
        load_config(tmp_path, {"dry_run": True})


def test_validate_config_data_accepts_syntax_without_authorizing():
    """A static syntax validator: valid api_base_url syntax passes even
    though no runtime approval was ever supplied -- this function alone can
    never authorize a run."""
    validate_config_data({"api_base_url": _URL})  # must not raise

    with pytest.raises(ConfigError, match="username, password, query string"):
        validate_config_data({"api_base_url": "http://user:pass@example.com/v1"})

    with pytest.raises(ConfigError, match="endpoint-trust"):
        validate_config_data({"api_base_url": _URL, "trust_api_base_url": _URL})


def test_refusal_error_text_names_no_raw_url_and_no_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-must-never-appear-in-any-message")
    with pytest.raises(ConfigError) as caught:
        load_config(
            tmp_path,
            {"api_base_url": _URL, "dry_run": True},
            trust_api_base_url=_OTHER_URL,
        )
    message = str(caught.value)
    assert _URL not in message
    assert _OTHER_URL not in message
    assert "sk-must-never-appear-in-any-message" not in message


# ---------------------------------------------------------------------------
# create_provider(): the second, independent enforcement boundary
# ---------------------------------------------------------------------------


def test_create_provider_refuses_plain_dict_with_unattested_endpoint():
    """create_provider(custom_plain_dict) cannot bypass authorization: a
    plain dict simply has no endpoint_trust attribute."""
    with pytest.raises(ProviderInitError, match="not authorized at runtime"):
        create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "openai",
                "model_name": "gpt-4o-mini",
                "api_key": "k",
                "api_base_url": _URL,
            }
        )


def test_create_provider_refuses_mismatched_attestation():
    config = ResolvedConfig(
        {
            "llm_mode": "api",
            "llm_provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "k",
            "api_base_url": _URL,
        },
        endpoint_trust=EndpointTrustAttestation(
            digest=effective_endpoint_identity(_OTHER_URL), mechanism="--trust-api-base-url"
        ),
    )
    with pytest.raises(ProviderInitError, match="not authorized at runtime"):
        create_provider(config)


def test_create_provider_accepts_attribute_carrying_the_correct_digest():
    """The attribute is duck-typed (getattr, not isinstance ResolvedConfig):
    any object exposing the right attribute name and digest satisfies it."""

    class _Carrier(dict):
        pass

    carrier = _Carrier(
        {
            "llm_mode": "api",
            "llm_provider": "anthropic",
            "model_name": "claude-haiku-4-5-20251001",
            "api_key": "k",
            "api_base_url": _URL,
        }
    )
    setattr(
        carrier,
        ENDPOINT_TRUST_ATTRIBUTE,
        EndpointTrustAttestation(
            digest=effective_endpoint_identity(_URL), mechanism="--trust-api-base-url"
        ),
    )
    # anthropic ignores api_base_url entirely and construction does not
    # validate the key against the network, so this only proves the gate
    # itself passes (never raises "not authorized") -- it does not exercise
    # the openai-specific base_url wiring covered elsewhere.
    try:
        create_provider(carrier)
    except ProviderInitError as exc:
        assert "not authorized at runtime" not in str(exc)


# ---------------------------------------------------------------------------
# run_pipeline(): no credential read, no provider constructed, real + dry run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dry_run", [False, True])
def test_refusal_reads_no_credential_env_var_and_constructs_no_provider(
    tmp_path, monkeypatch, dry_run
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("LLM_API_KEY", "sk-must-never-be-read")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-never-be-read-either")

    original_get = os.environ.get
    read_keys: list[str] = []

    def tracking_get(key, *args, **kwargs):
        read_keys.append(key)
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(os.environ, "get", tracking_get)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("refusal path must not construct a provider"),
    )

    with pytest.raises(ConfigError) as caught:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "api_base_url": _URL,
                "dry_run": dry_run,
                "propagate_changes": False,
            },
        )

    assert "LLM_API_KEY" not in read_keys
    assert "OPENAI_API_KEY" not in read_keys
    assert "sk-must-never-be-read" not in str(caught.value)


@pytest.mark.parametrize("dry_run", [False, True])
def test_matching_approval_permits_the_run_on_the_same_terms_dry_or_real(
    tmp_path, monkeypatch, dry_run
):
    """A dry run never reports a plan a real run would refuse, and vice
    versa: the gate is evaluated identically regardless of dry_run."""
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    calls = {"n": 0}

    class _Fake:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            calls["n"] += 1
            return '{"description": "d"}'

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: _Fake())

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "api_base_url": _URL,
            "dry_run": dry_run,
            "propagate_changes": False,
        },
        trust_api_base_url=_URL,
    )
    assert stats.get("dry_run", False) is dry_run
