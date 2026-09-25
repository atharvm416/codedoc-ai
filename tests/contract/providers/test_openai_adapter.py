"""Tests organized by feature ownership."""

from __future__ import annotations

import inspect
import sys
import types
import typing

import pytest

from tests.support.providers import _install_openai
from tests.support.provider_contract_cases import _make
from codedoc.utils.errors import LLMError, find_provider_failure

class TestAPIProviders:
    def test_openai_provider_uses_chat_completions(self, monkeypatch):
        from codedoc.llm.api_provider import OpenAIProvider

        captured = {}

        class FakeMessage:
            content = '{"ok": true}'

        class FakeChoice:
            message = FakeMessage()

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(choices=[FakeChoice()])

        class FakeOpenAI:
            def __init__(self, api_key, base_url=None, timeout=None, max_retries=None):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                captured["timeout"] = timeout
                captured["max_retries"] = max_retries
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions()
                )

        monkeypatch.setitem(
            sys.modules,
            "openai",
            types.SimpleNamespace(OpenAI=FakeOpenAI),
        )

        provider = OpenAIProvider(api_key="openai-key", model="gpt-test")
        result = provider.complete_json("Return JSON")

        assert result == '{"ok": true}'
        assert captured["api_key"] == "openai-key"
        assert captured["model"] == "gpt-test"
        assert captured["temperature"] == 0.0

def test_openai_sends_native_json_response_format(monkeypatch):
    rec = {}
    _install_openai(monkeypatch, rec)
    provider = _make("OpenAIProvider")(api_key="k", model="gpt-4o-mini")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert rec["create_kwargs"]["response_format"] == {"type": "json_object"}

def test_openai_compatible_base_url_stays_on_openai_adapter(monkeypatch):
    rec = {}
    _install_openai(monkeypatch, rec)
    provider = _make("OpenAIProvider")(
        api_key="k",
        model="custom-model",
        base_url="https://gateway.example/v1",
    )
    assert provider.provider_name == "OpenAI(custom-model)"
    assert rec["init"]["base_url"] == "https://gateway.example/v1"


# ---------------------------------------------------------------------------
# Shared fake SDK builder: full control over finish_reason/refusal, unlike
# the shared ``_install_openai`` installer.
# ---------------------------------------------------------------------------


def _install_openai_shaped(monkeypatch, rec, *, content, finish_reason="stop", refusal=None):
    mod = types.ModuleType("openai")

    class _Message:
        def __init__(self):
            self.content = content
            self.refusal = refusal

    class _Choice:
        def __init__(self):
            self.finish_reason = finish_reason
            self.message = _Message()

    class _Completions:
        def create(self, **kwargs):
            rec["create_kwargs"] = kwargs
            return types.SimpleNamespace(choices=[_Choice()])

    class OpenAI:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            rec["init"] = {"api_key": api_key}
            self.chat = types.SimpleNamespace(completions=_Completions())

    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)


# ---------------------------------------------------------------------------
# FIX-1: exact-model capability-based request compatibility (no prefix
# leakage across the GPT-5 family)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
def test_exact_gpt5_family_omits_temperature(monkeypatch, model):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(api_key="k", model=model)
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert "temperature" not in rec["create_kwargs"]
    assert rec["create_kwargs"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5.2"])
def test_gpt51_and_gpt52_keep_the_legacy_byte_identical_request(monkeypatch, model):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(api_key="k", model=model)
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert rec["create_kwargs"]["temperature"] == 0.0


def _audited_main_openai_json_kwargs(model: str) -> dict:
    """The exact ``complete_json`` request at commit 49221c2 (audited main):
    ``json_system = system + "\\n\\n" + <JSON instruction>``, sent as
    ``{model, messages=[system, user], temperature=0.0,
    response_format={"type": "json_object"}}``. Hardcoded from that audited
    baseline, never derived by calling into the 0.15.0 implementation under
    test."""
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "system\n\n"
                    "Respond ONLY with valid JSON. No markdown fences, no "
                    "explanation, no preamble."
                ),
            },
            {"role": "user", "content": "prompt"},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


def test_unknown_openai_model_full_kwargs_are_byte_identical_to_the_audited_main_request(
    monkeypatch,
):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(api_key="k", model="gpt-some-future-model")
    provider.complete_json("prompt", "system")
    assert rec["create_kwargs"] == _audited_main_openai_json_kwargs("gpt-some-future-model")


@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5.2"])
def test_gpt51_and_gpt52_full_kwargs_are_byte_identical_to_the_audited_main_request(
    monkeypatch, model,
):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(api_key="k", model=model)
    provider.complete_json("prompt", "system")
    assert rec["create_kwargs"] == _audited_main_openai_json_kwargs(model)


def test_custom_endpoint_full_kwargs_are_byte_identical_to_the_audited_main_request(
    monkeypatch,
):
    """A custom OpenAI-compatible endpoint, even with model string "gpt-5",
    keeps the exact audited-main request shape -- the regression this whole
    fix addresses."""
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(
        api_key="k", model="gpt-5", base_url="https://gateway.example/v1",
    )
    provider.complete_json("prompt", "system")
    assert rec["create_kwargs"] == _audited_main_openai_json_kwargs("gpt-5")


def test_gpt5_on_a_custom_endpoint_keeps_the_legacy_temperature(monkeypatch):
    """Regression: a custom OpenAI-compatible ``base_url`` (a local or
    self-hosted endpoint) must never inherit the exact-model GPT-5 family's
    temperature omission -- a model string of "gpt-5" served through a
    gateway is not necessarily the real OpenAI model. Only the default
    OpenAI endpoint applies the omission."""
    default_rec = {}
    _install_openai_shaped(monkeypatch, default_rec, content='{"ok": true}')
    default_provider = _make("OpenAIProvider")(api_key="k", model="gpt-5")
    default_provider.complete_json("prompt", "system")

    custom_rec = {}
    _install_openai_shaped(monkeypatch, custom_rec, content='{"ok": true}')
    custom_provider = _make("OpenAIProvider")(
        api_key="k", model="gpt-5", base_url="https://gateway.example/v1",
    )
    custom_provider.complete_json("prompt", "system")

    assert "temperature" not in default_rec["create_kwargs"]
    assert custom_rec["create_kwargs"]["temperature"] == 0.0
    # Every other field is identical between the two requests -- only the
    # endpoint changes the outcome, not some other divergent code path.
    assert (
        {k: v for k, v in default_rec["create_kwargs"].items() if k != "temperature"}
        == {k: v for k, v in custom_rec["create_kwargs"].items() if k != "temperature"}
    )


def test_unlisted_dated_snapshot_does_not_inherit_the_family_omission(monkeypatch):
    """Only the documented dated snapshot gets the omission."""
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(api_key="k", model="gpt-5-mini-2099-01-01")
    provider.complete_json("prompt", "system")
    assert rec["create_kwargs"]["temperature"] == 0.0


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
    ],
)
def test_new_exact_openai_ids_omit_json_default_but_keep_native_format(
    monkeypatch, model,
):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    _make("OpenAIProvider")(api_key="k", model=model).complete_json("prompt", "system")
    assert "temperature" not in rec["create_kwargs"]
    assert rec["create_kwargs"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
    ],
)
def test_new_openai_ids_on_custom_endpoints_keep_the_legacy_json_request(
    monkeypatch, model,
):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}')
    provider = _make("OpenAIProvider")(
        api_key="k", model=model, base_url="https://gateway.example/v1"
    )
    provider.complete_json("prompt", "system")
    assert rec["create_kwargs"] == _audited_main_openai_json_kwargs(model)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5", "gpt-5-mini", "gpt-5-nano",
        "gpt-5-2025-08-07", "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna",
    ],
)
@pytest.mark.parametrize("system", ["", "system"])
def test_public_complete_omits_only_an_unsupplied_default_on_an_exact_id(
    monkeypatch, model, system,
):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content="free text")
    provider = _make("OpenAIProvider")(api_key="k", model=model)
    assert provider.complete("prompt", system) == "free text"
    assert rec["create_kwargs"] == {
        "model": model,
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": "prompt"},
        ],
    }


@pytest.mark.parametrize("temperature", [0.1, 0.7])
def test_public_complete_forwards_every_explicit_temperature(monkeypatch, temperature):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content="free text")
    provider = _make("OpenAIProvider")(api_key="k", model="gpt-6-sol")
    provider.complete("prompt", temperature=temperature)
    assert rec["create_kwargs"]["temperature"] == temperature
    assert type(rec["create_kwargs"]["temperature"]) is float


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-5.1", "gpt-some-future-model"])
def test_public_complete_legacy_request_stays_a_plain_float(monkeypatch, model):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content="free text")
    _make("OpenAIProvider")(api_key="k", model=model).complete("prompt", "system")
    assert rec["create_kwargs"] == {
        "model": model,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "prompt"},
        ],
        "temperature": 0.1,
    }
    assert type(rec["create_kwargs"]["temperature"]) is float


def test_public_complete_custom_endpoint_keeps_plain_float(monkeypatch):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content="free text")
    provider = _make("OpenAIProvider")(
        api_key="k", model="gpt-6-sol", base_url="https://gateway.example/v1"
    )
    provider.complete("prompt", "system")
    assert rec["create_kwargs"]["temperature"] == 0.1
    assert type(rec["create_kwargs"]["temperature"]) is float


def test_public_complete_temperature_signatures_remain_float_defaults():
    from codedoc.llm.api_provider import AnthropicProvider, GeminiProvider, OpenAIProvider

    for cls in (AnthropicProvider, OpenAIProvider, GeminiProvider):
        parameter = inspect.signature(cls.complete).parameters["temperature"]
        assert typing.get_type_hints(cls.complete)["temperature"] is float
        assert parameter.default == 0.1
        assert repr(parameter.default) == "0.1"


def test_public_complete_always_forwards_caller_temperature_even_for_gpt5(
    monkeypatch,
):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content="free text")
    provider = _make("OpenAIProvider")(api_key="k", model="gpt-5")
    out = provider.complete("prompt", temperature=0.9)
    assert out == "free text"
    assert rec["create_kwargs"]["temperature"] == 0.9


# ---------------------------------------------------------------------------
# FIX-3: stop/refusal signal detection before returning text
# ---------------------------------------------------------------------------


def test_length_finish_reason_maps_to_output_truncated(monkeypatch):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content="partial", finish_reason="length")
    provider = _make("OpenAIProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-output-truncated"


def test_content_filter_finish_reason_maps_to_input_rejected(monkeypatch):
    rec = {}
    _install_openai_shaped(
        monkeypatch, rec, content=None, finish_reason="content_filter",
    )
    provider = _make("OpenAIProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-input-rejected"


def test_nonempty_refusal_maps_to_input_rejected(monkeypatch):
    rec = {}
    _install_openai_shaped(
        monkeypatch, rec, content=None, finish_reason="stop",
        refusal="SENTINEL-REFUSAL-TEXT",
    )
    provider = _make("OpenAIProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-input-rejected"
    assert "SENTINEL-REFUSAL-TEXT" not in str(caught.value)
    assert "SENTINEL-REFUSAL-TEXT" not in repr(caught.value)


def test_length_signal_present_alongside_text_still_fails(monkeypatch):
    rec = {}
    _install_openai_shaped(
        monkeypatch, rec, content='{"description": "cut off mid', finish_reason="length",
    )
    provider = _make("OpenAIProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-output-truncated"


def test_no_content_and_no_signal_raises_response_malformed(monkeypatch):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content=None, finish_reason="stop")
    provider = _make("OpenAIProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-response-malformed"


def test_stop_finish_reason_with_text_succeeds(monkeypatch):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content='{"ok": true}', finish_reason="stop")
    provider = _make("OpenAIProvider")(api_key="k")
    assert provider.complete_json("prompt") == '{"ok": true}'


def test_signal_failure_is_not_remapped_to_request_failed(monkeypatch):
    rec = {}
    _install_openai_shaped(monkeypatch, rec, content=None, finish_reason="content_filter")
    provider = _make("OpenAIProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code != "provider-request-failed"


# ---------------------------------------------------------------------------
# FIX-4 (section 5.5): explicit SDK endpoint pinning
# ---------------------------------------------------------------------------


def test_fake_sdk_records_the_pinned_default_base_url_when_unconfigured(monkeypatch):
    rec = {}
    _install_openai(monkeypatch, rec)
    _make("OpenAIProvider")(api_key="k")
    from codedoc.llm.api_provider import _OPENAI_DEFAULT_BASE_URL

    assert rec["init"]["base_url"] == _OPENAI_DEFAULT_BASE_URL


def test_direct_construction_with_real_sdk_ignores_a_synthetic_override(monkeypatch):
    from codedoc.llm.api_provider import OpenAIProvider, _OPENAI_DEFAULT_BASE_URL
    from codedoc.llm.factory import effective_endpoint_identity

    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic-openai.invalid/v1")
    provider = OpenAIProvider(api_key="fake-key")
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(_OPENAI_DEFAULT_BASE_URL)


def test_supplied_custom_url_wins_over_its_own_environment_variable(monkeypatch):
    """A caller-supplied base_url (an approved OpenAI-compatible endpoint)
    must win over OPENAI_BASE_URL, which would otherwise point somewhere
    else entirely."""
    from codedoc.llm.api_provider import OpenAIProvider
    from codedoc.llm.factory import effective_endpoint_identity

    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic-openai.invalid/v1")
    custom_url = "https://gateway.example.com/v1"
    provider = OpenAIProvider(api_key="fake-key", base_url=custom_url)
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(custom_url)


def test_premise_unpinned_real_sdk_would_have_honored_the_override(monkeypatch):
    """Premise: the installed openai SDK really falls back to
    OPENAI_BASE_URL when no base_url argument is given."""
    import openai

    monkeypatch.setenv("OPENAI_BASE_URL", "https://synthetic-openai.invalid/v1")
    unpinned = openai.OpenAI(api_key="fake-key")
    assert "synthetic-openai.invalid" in str(unpinned.base_url)
