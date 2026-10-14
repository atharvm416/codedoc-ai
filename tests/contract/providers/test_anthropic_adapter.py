"""Tests organized by feature ownership."""

from __future__ import annotations

import sys
import types

import pytest

from tests.support.providers import _install_anthropic
from tests.support.provider_contract_cases import _JSON_HINT
from tests.support.provider_contract_cases import _make
from codedoc.utils.errors import LLMError, find_provider_failure

class TestAPIProviders:
    def test_anthropic_provider_uses_messages_api(self, monkeypatch):
        from codedoc.llm.api_provider import AnthropicProvider

        captured = {}

        class FakeMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text='{"ok": true}')],
                    stop_reason="end_turn",
                )

        class FakeAnthropicClient:
            def __init__(self, api_key, base_url=None, timeout=None, max_retries=None):
                captured["api_key"] = api_key
                captured["base_url"] = base_url
                captured["timeout"] = timeout
                captured["max_retries"] = max_retries
                self.messages = FakeMessages()

        monkeypatch.setitem(
            sys.modules,
            "anthropic",
            types.SimpleNamespace(Anthropic=FakeAnthropicClient),
        )

        provider = AnthropicProvider(api_key="anthropic-key", model="claude-test")
        result = provider.complete_json("Return JSON")

        assert result == '{"ok": true}'
        assert captured["api_key"] == "anthropic-key"
        assert captured["model"] == "claude-test"
        assert captured["temperature"] == 0.0

def test_anthropic_receives_json_instruction_and_keeps_token_allowance(monkeypatch):
    rec = {}
    _install_anthropic(monkeypatch, rec)
    provider = _make("AnthropicProvider")(api_key="k")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    # JSON-only instruction is carried in the top-level system parameter.
    assert _JSON_HINT in rec["create_kwargs"]["system"]
    # The output token allowance is preserved (not reduced by JSON mode).
    assert rec["create_kwargs"]["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Shared fake SDK builder for the tests below: full control over content
# blocks and stop_reason, unlike the shared ``_install_anthropic`` installer.
# ---------------------------------------------------------------------------


def _install_anthropic_shaped(monkeypatch, rec, *, content, stop_reason="end_turn"):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs):
            rec["create_kwargs"] = kwargs
            return types.SimpleNamespace(content=content, stop_reason=stop_reason)

    class Anthropic:
        def __init__(self, api_key=None, base_url=None, timeout=None, max_retries=None):
            rec["init"] = {"api_key": api_key, "base_url": base_url}
            self.messages = _Messages()

    mod.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)


def _text_block(text: str):
    return types.SimpleNamespace(type="text", text=text)


def _thinking_block(thinking: str):
    return types.SimpleNamespace(type="thinking", thinking=thinking)


# ---------------------------------------------------------------------------
# FIX-1: exact-model capability-based request compatibility
# ---------------------------------------------------------------------------


def test_exact_opus_5_omits_temperature_and_uses_its_capped_max_tokens(monkeypatch):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block('{"ok": true}')])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-opus-5")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert "temperature" not in rec["create_kwargs"]
    assert 0 < rec["create_kwargs"]["max_tokens"] <= 21_000
    assert rec["create_kwargs"]["max_tokens"] != 4096


def test_exact_opus_5_5_omits_temperature_and_uses_its_capped_max_tokens(monkeypatch):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block('{"ok": true}')])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-opus-5-5")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert "temperature" not in rec["create_kwargs"]


@pytest.mark.parametrize(
    ("model", "cap"),
    [
        ("claude-opus-4-7", 4096),
        ("claude-opus-4-8", 4096),
        ("claude-sonnet-5", 8192),
        ("claude-fable-5", 8192),
        ("claude-fable-5-1", 8192),
        ("claude-mythos-5", 8192),
        ("claude-mythos-5-1", 8192),
    ],
)
def test_new_exact_anthropic_ids_omit_json_default_with_their_cap(
    monkeypatch, model, cap,
):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block('{"ok": true}')])
    provider = _make("AnthropicProvider")(api_key="k", model=model)
    provider.complete_json("prompt", "system")
    assert "temperature" not in rec["create_kwargs"]
    assert rec["create_kwargs"]["max_tokens"] == cap
    assert _JSON_HINT in rec["create_kwargs"]["system"]


def test_unknown_model_keeps_legacy_request_with_temperature(monkeypatch):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block('{"ok": true}')])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-some-future-snapshot")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    assert rec["create_kwargs"]["temperature"] == 0.0
    assert rec["create_kwargs"]["max_tokens"] == 4096


def test_haiku_default_full_kwargs_are_byte_identical_to_the_audited_main_request(
    monkeypatch,
):
    """Full-kwargs regression: at commit 49221c2 (audited main), Anthropic's
    ``complete_json`` was inherited from ``LLMProvider.complete_json``, which
    built ``json_system = system + "\\n\\n" + <JSON instruction>`` and called
    the old ``complete()``, sending exactly ``{model, max_tokens=4096,
    temperature=0.0, messages=[{role: user, content: prompt}], system}``.
    This expected dict is hardcoded from that audited baseline, never derived
    by calling into the 0.15.0 implementation under test."""
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block('{"ok": true}')])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-haiku-4-5-20251001")
    provider.complete_json("prompt", "system")

    expected = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "prompt"}],
        "system": (
            "system\n\n"
            "Respond ONLY with valid JSON. No markdown fences, no explanation, "
            "no preamble."
        ),
    }
    assert rec["create_kwargs"] == expected


def test_unknown_anthropic_model_full_kwargs_are_byte_identical_to_the_audited_main_request(
    monkeypatch,
):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block('{"ok": true}')])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-some-future-snapshot")
    provider.complete_json("prompt", "system")

    expected = {
        "model": "claude-some-future-snapshot",
        "max_tokens": 4096,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "prompt"}],
        "system": (
            "system\n\n"
            "Respond ONLY with valid JSON. No markdown fences, no explanation, "
            "no preamble."
        ),
    }
    assert rec["create_kwargs"] == expected


def test_public_complete_always_forwards_caller_temperature_even_for_opus_5(
    monkeypatch,
):
    """An explicitly supplied value is forwarded even for an omit model."""
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block("free text")])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-opus-5")
    out = provider.complete("prompt", temperature=0.7)
    assert out == "free text"
    assert rec["create_kwargs"]["temperature"] == 0.7
    assert rec["create_kwargs"]["max_tokens"] == 4096


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5",
        "claude-opus-5", "claude-opus-5-5", "claude-fable-5",
        "claude-fable-5-1", "claude-mythos-5", "claude-mythos-5-1",
    ],
)
@pytest.mark.parametrize("system", ["", "system"])
def test_public_complete_omits_unsupplied_temperature_for_every_omit_id(
    monkeypatch, model, system,
):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block("free text")])
    provider = _make("AnthropicProvider")(api_key="k", model=model)
    assert provider.complete("prompt", system) == "free text"
    expected = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "prompt"}],
    }
    if system:
        expected["system"] = system
    assert rec["create_kwargs"] == expected


@pytest.mark.parametrize("temperature", [0.1, 0.7])
def test_public_complete_forwards_explicit_temperature(monkeypatch, temperature):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block("free text")])
    provider = _make("AnthropicProvider")(api_key="k", model="claude-sonnet-5")
    provider.complete("prompt", temperature=temperature)
    assert rec["create_kwargs"]["temperature"] == temperature
    assert type(rec["create_kwargs"]["temperature"]) is float


@pytest.mark.parametrize(
    "model", ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-unknown"]
)
def test_public_complete_legacy_request_uses_a_plain_float(monkeypatch, model):
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block("free text")])
    provider = _make("AnthropicProvider")(api_key="k", model=model)
    provider.complete("prompt", "system")
    assert rec["create_kwargs"] == {
        "model": model,
        "max_tokens": 4096,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": "prompt"}],
        "system": "system",
    }
    assert type(rec["create_kwargs"]["temperature"]) is float


# ---------------------------------------------------------------------------
# FIX-2: safe text extraction by content-block type
# ---------------------------------------------------------------------------


def test_thinking_block_before_text_is_ignored_and_text_is_read(monkeypatch):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec,
        content=[_thinking_block("SENTINEL-THINKING-1"), _text_block('{"ok": true}')],
    )
    provider = _make("AnthropicProvider")(api_key="k")
    out = provider.complete_json("prompt")
    assert out == '{"ok": true}'
    assert "SENTINEL-THINKING-1" not in out


def test_single_text_block_preserves_bytes(monkeypatch):
    rec = {}
    text = '{"description": "exact bytes, no change"}'
    _install_anthropic_shaped(monkeypatch, rec, content=[_text_block(text)])
    provider = _make("AnthropicProvider")(api_key="k")
    assert provider.complete_json("prompt") == text


def test_multiple_text_blocks_are_joined_with_empty_string(monkeypatch):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec,
        content=[_text_block('{"a": '), _text_block("1}")],
    )
    provider = _make("AnthropicProvider")(api_key="k")
    assert provider.complete_json("prompt") == '{"a": 1}'


def test_zero_text_blocks_raises_response_malformed(monkeypatch):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec, content=[_thinking_block("only thinking, no answer")],
    )
    provider = _make("AnthropicProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    failure = find_provider_failure(caught.value)
    assert failure is not None
    assert failure.reason_code == "provider-response-malformed"


def test_thinking_content_never_leaks_into_the_zero_text_error(monkeypatch):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec, content=[_thinking_block("SENTINEL-THINKING-LEAK")],
    )
    provider = _make("AnthropicProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert "SENTINEL-THINKING-LEAK" not in str(caught.value)
    assert "SENTINEL-THINKING-LEAK" not in repr(caught.value)


# ---------------------------------------------------------------------------
# FIX-3: stop/refusal signal detection before returning text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded"])
def test_truncation_stop_reasons_map_to_output_truncated(monkeypatch, stop_reason):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec, content=[_text_block("partial")], stop_reason=stop_reason,
    )
    provider = _make("AnthropicProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    failure = find_provider_failure(caught.value)
    assert failure is not None
    assert failure.reason_code == "provider-output-truncated"


def test_refusal_stop_reason_maps_to_input_rejected(monkeypatch):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec, content=[], stop_reason="refusal",
    )
    provider = _make("AnthropicProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    failure = find_provider_failure(caught.value)
    assert failure is not None
    assert failure.reason_code == "provider-input-rejected"


def test_truncation_signal_present_alongside_text_still_fails(monkeypatch):
    """Section 5.2: check the signal even when text is present -- a
    max_tokens stop reason fails the call even though a (possibly
    incomplete) text block accompanies it."""
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec,
        content=[_text_block('{"description": "cut off mid')],
        stop_reason="max_tokens",
    )
    provider = _make("AnthropicProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-output-truncated"


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence", "tool_use", "pause_turn"])
def test_other_stop_reasons_with_text_succeed(monkeypatch, stop_reason):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec, content=[_text_block('{"ok": true}')], stop_reason=stop_reason,
    )
    provider = _make("AnthropicProvider")(api_key="k")
    assert provider.complete_json("prompt") == '{"ok": true}'


def test_none_stop_reason_is_not_a_signal():
    """An absent or None stop_reason means 'no signal', never an error."""
    from codedoc.llm.api_provider import _extract_anthropic_text

    response = types.SimpleNamespace(
        content=[_text_block('{"ok": true}')], stop_reason=None,
    )
    assert _extract_anthropic_text(response) == '{"ok": true}'


def test_missing_stop_reason_attribute_is_not_a_signal():
    from codedoc.llm.api_provider import _extract_anthropic_text

    response = types.SimpleNamespace(content=[_text_block('{"ok": true}')])
    assert _extract_anthropic_text(response) == '{"ok": true}'


def test_signal_failure_is_not_remapped_to_request_failed(monkeypatch):
    """The typed LLMError this module raises for a post-response signal is
    never re-caught by this same module's own SDK-exception mapper and
    relabeled provider-request-failed."""
    rec = {}
    _install_anthropic_shaped(monkeypatch, rec, content=[], stop_reason="refusal")
    provider = _make("AnthropicProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code != "provider-request-failed"


def test_sentinel_key_and_prompt_never_leak_into_signal_error(monkeypatch):
    rec = {}
    _install_anthropic_shaped(
        monkeypatch, rec, content=[_text_block("partial")], stop_reason="max_tokens",
    )
    provider = _make("AnthropicProvider")(api_key="SENTINEL-FAKE-KEY-XYZ")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("SENTINEL-PROMPT-TEXT")
    assert "SENTINEL-FAKE-KEY-XYZ" not in str(caught.value)
    assert "SENTINEL-PROMPT-TEXT" not in str(caught.value)
    assert "SENTINEL-FAKE-KEY-XYZ" not in repr(caught.value)
    assert "SENTINEL-PROMPT-TEXT" not in repr(caught.value)


# ---------------------------------------------------------------------------
# FIX-4 (section 5.5): explicit SDK endpoint pinning
# ---------------------------------------------------------------------------


def test_fake_sdk_records_the_pinned_base_url(monkeypatch):
    rec = {}
    _install_anthropic(monkeypatch, rec)
    _make("AnthropicProvider")(api_key="k")
    from codedoc.llm.api_provider import _ANTHROPIC_DEFAULT_BASE_URL

    assert rec["init"]["base_url"] == _ANTHROPIC_DEFAULT_BASE_URL


def test_direct_construction_with_real_sdk_ignores_a_synthetic_override(monkeypatch):
    """Real installed anthropic SDK, fake key, no request: a synthetic
    ANTHROPIC_BASE_URL set in the environment must not change the client's
    actual resolved transport."""
    from codedoc.llm.api_provider import _ANTHROPIC_DEFAULT_BASE_URL, AnthropicProvider
    from codedoc.llm.factory import effective_endpoint_identity

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://synthetic-anthropic.invalid")
    provider = AnthropicProvider(api_key="fake-key")
    assert effective_endpoint_identity(
        str(provider._client.base_url)
    ) == effective_endpoint_identity(_ANTHROPIC_DEFAULT_BASE_URL)


def test_premise_unpinned_real_sdk_would_have_honored_the_override(monkeypatch):
    """Premise: the installed anthropic SDK really falls back to
    ANTHROPIC_BASE_URL when no base_url argument is given -- confirming
    FIX-4's explicit argument is the mechanism, not a no-op."""
    import anthropic

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://synthetic-anthropic.invalid")
    unpinned = anthropic.Anthropic(api_key="fake-key")
    assert "synthetic-anthropic.invalid" in str(unpinned.base_url)
