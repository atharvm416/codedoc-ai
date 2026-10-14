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
from codedoc.llm.api_provider import (
    _ANTHROPIC_DEFAULT_BASE_URL,
    _GEMINI_DEFAULT_BASE_URL,
    _OPENAI_DEFAULT_BASE_URL,
)
from codedoc.llm.factory import (
    ENDPOINT_TRUST_ATTRIBUTE,
    _SDK_ENDPOINT_OVERRIDE_VARS,
    EndpointTrustAttestation,
    _sdk_endpoint_override,
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


# ===========================================================================
# FIX-4 (section 5.5): the pinned SDKs' own environment-override fallback
# (OPENAI_BASE_URL, ANTHROPIC_BASE_URL, GOOGLE_GEMINI_BASE_URL,
# GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_GENAI_USE_ENTERPRISE) is checked and
# refused, before any credential lookup or SDK construction, when it would
# route the selected provider's traffic to a different endpoint than
# CodeDoc's configured one.
# ===========================================================================

_SYNTHETIC_OPENAI_URL = "https://synthetic-openai.invalid/v1"
_SYNTHETIC_ANTHROPIC_URL = "https://synthetic-anthropic.invalid"
_SYNTHETIC_GEMINI_URL = "https://synthetic-gemini.invalid/"


def _fail_credential_lookup(monkeypatch):
    monkeypatch.setattr(
        "codedoc.llm.factory._provider_api_key",
        lambda *a, **k: pytest.fail("credential lookup must not run"),
    )


def _fail_sdk_construction(monkeypatch):
    monkeypatch.setattr(
        "codedoc.llm.factory._make_api",
        lambda *a, **k: pytest.fail("SDK construction must not run"),
    )


@pytest.mark.parametrize(
    ("provider_kind", "model", "var", "value"),
    [
        ("openai", "gpt-4o-mini", "OPENAI_BASE_URL", _SYNTHETIC_OPENAI_URL),
        ("openai", "gpt-4o-mini", "OPENAI_BASE_URL", "not-a-valid-url"),
        ("anthropic", "claude-haiku-4-5-20251001", "ANTHROPIC_BASE_URL", _SYNTHETIC_ANTHROPIC_URL),
        ("anthropic", "claude-haiku-4-5-20251001", "ANTHROPIC_BASE_URL", "not-a-valid-url"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GEMINI_BASE_URL", _SYNTHETIC_GEMINI_URL),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GEMINI_BASE_URL", "not-a-valid-url"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GENAI_USE_VERTEXAI", "true"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GENAI_USE_VERTEXAI", "1"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GENAI_USE_VERTEXAI", "TRUE"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GENAI_USE_VERTEXAI", "True"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GENAI_USE_ENTERPRISE", "true"),
        ("gemini", "gemini-2.5-flash", "GOOGLE_GENAI_USE_ENTERPRISE", "1"),
    ],
)
def test_conflicting_sdk_override_is_refused_before_credential_or_sdk_construction(
    monkeypatch, provider_kind, model, var, value,
):
    """No ``api_key`` is supplied: ``config.get("api_key") or
    _provider_api_key(...)`` would otherwise short-circuit on a truthy
    supplied key regardless of where the gate sits, so this test would pass
    even if the gate ran *after* credential lookup. Omitting it forces the
    ``or`` branch to actually reach ``_provider_api_key`` unless the gate
    itself stops the call first -- genuinely proving gate-before-credential
    ordering."""
    monkeypatch.setenv(var, value)
    _fail_credential_lookup(monkeypatch)
    _fail_sdk_construction(monkeypatch)

    with pytest.raises(ProviderInitError) as caught:
        create_provider(
            {
                "llm_mode": "api",
                "llm_provider": provider_kind,
                "model_name": model,
            }
        )
    message = str(caught.value)
    assert var in message
    assert value not in message


@pytest.mark.parametrize(
    ("var", "value", "provider_kind"),
    [
        ("OPENAI_BASE_URL", "", "openai"),
        ("OPENAI_BASE_URL", "   ", "openai"),
        ("OPENAI_BASE_URL", _OPENAI_DEFAULT_BASE_URL, "openai"),
        ("OPENAI_BASE_URL", "HTTPS://API.OPENAI.COM/v1", "openai"),
        ("ANTHROPIC_BASE_URL", "", "anthropic"),
        ("ANTHROPIC_BASE_URL", _ANTHROPIC_DEFAULT_BASE_URL, "anthropic"),
        ("ANTHROPIC_BASE_URL", "HTTPS://API.ANTHROPIC.COM", "anthropic"),
        ("GOOGLE_GEMINI_BASE_URL", "", "gemini"),
        ("GOOGLE_GEMINI_BASE_URL", _GEMINI_DEFAULT_BASE_URL, "gemini"),
        ("GOOGLE_GEMINI_BASE_URL", _GEMINI_DEFAULT_BASE_URL.rstrip("/"), "gemini"),
        ("GOOGLE_GENAI_USE_VERTEXAI", "", "gemini"),
        ("GOOGLE_GENAI_USE_VERTEXAI", "false", "gemini"),
        ("GOOGLE_GENAI_USE_VERTEXAI", "0", "gemini"),
        ("GOOGLE_GENAI_USE_ENTERPRISE", "", "gemini"),
        ("GOOGLE_GENAI_USE_ENTERPRISE", "false", "gemini"),
        ("GOOGLE_GENAI_USE_ENTERPRISE", "0", "gemini"),
        # A variable belonging to a different provider than the one selected.
        ("OPENAI_BASE_URL", _SYNTHETIC_OPENAI_URL, "anthropic"),
        ("ANTHROPIC_BASE_URL", _SYNTHETIC_ANTHROPIC_URL, "gemini"),
        ("GOOGLE_GEMINI_BASE_URL", _SYNTHETIC_GEMINI_URL, "openai"),
        ("GOOGLE_GENAI_USE_VERTEXAI", "true", "openai"),
        # Leading/trailing whitespace around a true-shaped value: the pinned
        # SDK does not strip before comparing (`value.lower() in
        # ['true', '1']`), so " true "/" 1 " are read as NOT true by the real
        # SDK -- the gate must mirror that exactly (section 5.7 item 2).
        ("GOOGLE_GENAI_USE_VERTEXAI", " true ", "gemini"),
        ("GOOGLE_GENAI_USE_VERTEXAI", "true ", "gemini"),
        ("GOOGLE_GENAI_USE_VERTEXAI", " true", "gemini"),
        ("GOOGLE_GENAI_USE_VERTEXAI", " 1 ", "gemini"),
        ("GOOGLE_GENAI_USE_ENTERPRISE", " true ", "gemini"),
        ("GOOGLE_GENAI_USE_ENTERPRISE", " 1 ", "gemini"),
    ],
)
def test_blank_default_equivalent_false_and_other_provider_overrides_are_allowed(
    monkeypatch, var, value, provider_kind,
):
    monkeypatch.setenv(var, value)
    assert _sdk_endpoint_override(provider_kind, None) is None


def test_premise_the_real_sdk_does_not_strip_whitespace_around_a_vertex_value(
    monkeypatch,
):
    """Premise: confirms the installed google-genai SDK itself treats
    " true " (with spaces) as NOT true, so the gate's no-strip behavior is
    mirroring real SDK parsing, not an arbitrary choice."""
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", " true ")
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key="fake-key", http_options=types.HttpOptions(timeout=30000)
    )
    assert client.vertexai is False


def test_allowed_openai_environment_leaves_the_real_sdk_transport_at_the_default(
    monkeypatch,
):
    """Fake key, real installed SDK client, no request: with no conflicting
    override, the recorded identity and the actual constructed transport
    both stay at the default."""
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    provider = create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "fake-key",
        }
    )
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(_OPENAI_DEFAULT_BASE_URL)


def test_allowed_anthropic_environment_leaves_the_real_sdk_transport_at_the_default(
    monkeypatch,
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _ANTHROPIC_DEFAULT_BASE_URL)
    provider = create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "anthropic",
            "model_name": "claude-haiku-4-5-20251001",
            "api_key": "fake-key",
        }
    )
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(_ANTHROPIC_DEFAULT_BASE_URL)


def test_allowed_gemini_environment_leaves_the_real_sdk_transport_at_the_default(
    monkeypatch,
):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", _GEMINI_DEFAULT_BASE_URL.rstrip("/"))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    provider = create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "gemini",
            "model_name": "gemini-2.5-flash",
            "api_key": "fake-key",
        }
    )
    http_options = provider._client._api_client._http_options
    assert effective_endpoint_identity(
        http_options.base_url
    ) == effective_endpoint_identity(_GEMINI_DEFAULT_BASE_URL)
    assert provider._client.vertexai is False


def test_approved_api_base_url_wins_over_conflicting_openai_base_url_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", _SYNTHETIC_OPENAI_URL)
    approved_url = "http://approved.example.com:9999/v1"
    config = ResolvedConfig(
        {
            "llm_mode": "api",
            "llm_provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "fake-key",
            "api_base_url": approved_url,
        },
        endpoint_trust=EndpointTrustAttestation(
            digest=effective_endpoint_identity(approved_url),
            mechanism="--trust-api-base-url",
        ),
    )
    provider = create_provider(config)
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(approved_url)


def test_disabling_the_factory_gate_still_leaves_real_transport_at_the_default(
    monkeypatch,
):
    """Defense in depth: even if the factory's own override detector were
    disabled or buggy, the explicit SDK constructor arguments in
    api_provider.py are an independent layer that still pins transport to
    the default -- a malicious environment override never reaches the wire
    either way."""
    monkeypatch.setattr(
        "codedoc.llm.factory._sdk_endpoint_override", lambda *a, **k: None
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", _SYNTHETIC_ANTHROPIC_URL)
    provider = create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "anthropic",
            "model_name": "claude-haiku-4-5-20251001",
            "api_key": "fake-key",
        }
    )
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(_ANTHROPIC_DEFAULT_BASE_URL)


def test_premise_unpinned_real_anthropic_sdk_honors_the_synthetic_override(
    monkeypatch,
):
    """Premise check: the installed anthropic SDK really does fall back to
    ANTHROPIC_BASE_URL when CodeDoc passes no explicit base_url -- proving
    FIX-4's own explicit-argument pinning is the mechanism closing the gap,
    not an SDK that never had this fallback to begin with."""
    import anthropic

    monkeypatch.setenv("ANTHROPIC_BASE_URL", _SYNTHETIC_ANTHROPIC_URL)
    unpinned_client = anthropic.Anthropic(api_key="fake-key", timeout=30, max_retries=0)
    assert effective_endpoint_identity(
        str(unpinned_client.base_url)
    ) == effective_endpoint_identity(_SYNTHETIC_ANTHROPIC_URL)


def test_premise_unpinned_real_openai_sdk_honors_the_synthetic_override(monkeypatch):
    import openai

    monkeypatch.setenv("OPENAI_BASE_URL", _SYNTHETIC_OPENAI_URL)
    unpinned_client = openai.OpenAI(api_key="fake-key", timeout=30, max_retries=0)
    assert effective_endpoint_identity(
        str(unpinned_client.base_url)
    ) == effective_endpoint_identity(_SYNTHETIC_OPENAI_URL)


def test_premise_unpinned_real_gemini_sdk_honors_the_synthetic_overrides(monkeypatch):
    from google import genai
    from google.genai import types

    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", _SYNTHETIC_GEMINI_URL)
    http_options = types.HttpOptions(timeout=30000)
    unpinned_client = genai.Client(api_key="fake-key", http_options=http_options)
    assert effective_endpoint_identity(
        unpinned_client._api_client._http_options.base_url
    ) == effective_endpoint_identity(_SYNTHETIC_GEMINI_URL)

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    unpinned_vertex_client = genai.Client(
        api_key="fake-key", http_options=types.HttpOptions(timeout=30000)
    )
    assert unpinned_vertex_client.vertexai is True


def test_autouse_fixture_leaves_the_environment_clean_at_test_start():
    for variables in _SDK_ENDPOINT_OVERRIDE_VARS.values():
        for name in variables:
            assert name not in os.environ


@pytest.fixture(scope="class")
def _sdk_overrides_preset_once_per_class():
    """Sets every SDK endpoint-override variable once for the whole class,
    through a class-scoped ``pytest.MonkeyPatch`` (not the function-scoped
    ``monkeypatch`` fixture, which cannot outlive a single test). Because
    scope="class" is broader than the conftest autouse fixture's default
    function scope, pytest sets this one up first for each test in the
    class; the function-scoped autouse scrub then runs immediately before
    the test body, so a correctly-working scrub must have already removed
    every value this fixture set, before the test even starts -- proven by
    fixture nesting within one test, not by execution order across two
    separate tests."""
    mp = pytest.MonkeyPatch()
    for variables in _SDK_ENDPOINT_OVERRIDE_VARS.values():
        for name in variables:
            value = (
                "true" if name.startswith("GOOGLE_GENAI_USE_")
                else "https://preexisting.invalid/v1"
            )
            mp.setenv(name, value)
    yield
    mp.undo()


class TestAutouseFixtureScrubsPreexistingOverrides:
    @pytest.mark.usefixtures("_sdk_overrides_preset_once_per_class")
    def test_autouse_fixture_removed_every_preset_override_before_the_test_body_ran(
        self,
    ):
        for variables in _SDK_ENDPOINT_OVERRIDE_VARS.values():
            for name in variables:
                assert name not in os.environ
