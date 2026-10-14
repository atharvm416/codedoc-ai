"""Tests organized by feature ownership."""

from __future__ import annotations

import sys
import types

import pytest

from tests.support.providers import _install_gemini
from tests.support.provider_contract_cases import _JSON_HINT
from tests.support.provider_contract_cases import _make
from codedoc.utils.errors import LLMError, find_provider_failure

class TestAPIProviders:
    def test_gemini_provider_uses_generate_content(self, monkeypatch):
        from codedoc.llm.api_provider import GeminiProvider

        captured = {}

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeHttpRetryOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeHttpOptions:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(text='{"ok": true}')

        class FakeClient:
            def __init__(self, api_key, http_options=None, vertexai=None):
                captured["api_key"] = api_key
                captured["http_options"] = http_options
                captured["vertexai"] = vertexai
                self.models = FakeModels()

        fake_types = types.SimpleNamespace(
            GenerateContentConfig=FakeGenerateContentConfig,
            HttpRetryOptions=FakeHttpRetryOptions,
            HttpOptions=FakeHttpOptions,
        )
        fake_genai = types.SimpleNamespace(Client=FakeClient, types=fake_types)
        fake_google = types.SimpleNamespace(genai=fake_genai)

        monkeypatch.setitem(sys.modules, "google", fake_google)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

        provider = GeminiProvider(api_key="gemini-key", model="gemini-test")
        result = provider.complete_json("Return JSON")

        assert result == '{"ok": true}'
        assert captured["api_key"] == "gemini-key"
        assert captured["model"] == "gemini-test"
        assert captured["config"].kwargs["temperature"] == 0.0
        assert captured["config"].kwargs["response_mime_type"] == "application/json"

def test_gemini_sends_json_mime_type_and_system_instruction(monkeypatch):
    rec = {}
    _install_gemini(monkeypatch, rec)
    provider = _make("GeminiProvider")(api_key="k")
    out = provider.complete_json("prompt", "system")
    assert out == '{"ok": true}'
    config = rec["generate"]["config"]
    assert config.response_mime_type == "application/json"
    assert _JSON_HINT in config.system_instruction


# ---------------------------------------------------------------------------
# Shared fake SDK builder: full control over candidates/finish_reason/
# prompt_feedback, unlike the shared ``_install_gemini`` installer (which
# only shapes ``.text``).
# ---------------------------------------------------------------------------


def _candidate(finish_reason=None):
    return types.SimpleNamespace(finish_reason=finish_reason)


def _prompt_feedback(block_reason=None):
    return types.SimpleNamespace(block_reason=block_reason)


def _install_gemini_shaped(
    monkeypatch, rec, *, text, candidates=None, prompt_feedback=None,
):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    gtypes = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class HttpRetryOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class HttpOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    gtypes.GenerateContentConfig = GenerateContentConfig
    gtypes.HttpRetryOptions = HttpRetryOptions
    gtypes.HttpOptions = HttpOptions

    class _Models:
        def generate_content(self, model, contents, config):
            rec["generate"] = {"model": model, "contents": contents, "config": config}
            return types.SimpleNamespace(
                text=text, candidates=candidates, prompt_feedback=prompt_feedback,
            )

    class Client:
        def __init__(self, api_key=None, http_options=None, vertexai=None):
            rec["init"] = {
                "api_key": api_key, "http_options": http_options, "vertexai": vertexai,
            }
            self.models = _Models()

    genai.Client = Client
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)


# ---------------------------------------------------------------------------
# FIX-3: stop/refusal/prompt-block signal detection before returning text
# ---------------------------------------------------------------------------


def test_max_tokens_finish_reason_maps_to_output_truncated(monkeypatch):
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text="partial", candidates=[_candidate("MAX_TOKENS")],
    )
    provider = _make("GeminiProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-output-truncated"


@pytest.mark.parametrize(
    "finish_reason",
    [
        "SAFETY",
        "RECITATION",
        "LANGUAGE",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    ],
)
def test_every_policy_finish_reason_maps_to_input_rejected(monkeypatch, finish_reason):
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text=None, candidates=[_candidate(finish_reason)],
    )
    provider = _make("GeminiProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-input-rejected"


@pytest.mark.parametrize(
    "block_reason",
    ["SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY",
     "MODEL_ARMOR", "JAILBREAK"],
)
def test_every_prompt_block_reason_maps_to_input_rejected(monkeypatch, block_reason):
    """A blocked prompt has no candidate and therefore no finish reason."""
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text=None, candidates=None,
        prompt_feedback=_prompt_feedback(block_reason),
    )
    provider = _make("GeminiProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-input-rejected"


def test_blocked_reason_unspecified_is_not_a_signal(monkeypatch):
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text='{"ok": true}',
        candidates=[_candidate("STOP")],
        prompt_feedback=_prompt_feedback("BLOCKED_REASON_UNSPECIFIED"),
    )
    provider = _make("GeminiProvider")(api_key="k")
    assert provider.complete_json("prompt") == '{"ok": true}'


def test_max_tokens_signal_present_alongside_text_still_fails(monkeypatch):
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text='{"description": "cut off mid',
        candidates=[_candidate("MAX_TOKENS")],
    )
    provider = _make("GeminiProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-output-truncated"


def test_no_text_and_no_signal_raises_response_malformed(monkeypatch):
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text=None, candidates=[_candidate("STOP")],
    )
    provider = _make("GeminiProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code == "provider-response-malformed"


def test_absent_candidates_and_prompt_feedback_is_not_a_signal(monkeypatch):
    rec = {}
    _install_gemini_shaped(monkeypatch, rec, text='{"ok": true}')
    provider = _make("GeminiProvider")(api_key="k")
    assert provider.complete_json("prompt") == '{"ok": true}'


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-some-future-model"])
def test_gemini_full_kwargs_are_byte_identical_to_the_audited_main_request(
    monkeypatch, model,
):
    """0.15.0 makes no request-shape change for Gemini (plan section 5.1:
    'preserve the current request unless an exact documented incompatibility
    requires a change' -- none was found). Full-kwargs regression against the
    audited commit 49221c2 request, for both the default and an unlisted
    model: ``generate_content(model, contents=prompt, config)`` where
    ``config`` carries ``temperature=0.0``, the JSON system instruction, and
    ``response_mime_type="application/json"``. Hardcoded from that audited
    baseline, never derived from the 0.15.0 implementation under test."""
    rec = {}
    _install_gemini_shaped(monkeypatch, rec, text='{"ok": true}')
    provider = _make("GeminiProvider")(api_key="k", model=model)
    provider.complete_json("prompt", "system")

    call = rec["generate"]
    assert call["model"] == model
    assert call["contents"] == "prompt"
    assert call["config"].__dict__ == {
        "temperature": 0.0,
        "system_instruction": (
            "system\n\n"
            "Respond ONLY with valid JSON. No markdown fences, no explanation, "
            "no preamble."
        ),
        "response_mime_type": "application/json",
    }


def test_signal_failure_is_not_remapped_to_request_failed(monkeypatch):
    rec = {}
    _install_gemini_shaped(
        monkeypatch, rec, text=None, candidates=[_candidate("SAFETY")],
    )
    provider = _make("GeminiProvider")(api_key="k")
    with pytest.raises(LLMError) as caught:
        provider.complete_json("prompt")
    assert find_provider_failure(caught.value).reason_code != "provider-request-failed"


# ---------------------------------------------------------------------------
# FIX-4 (section 5.5): explicit SDK endpoint pinning
# ---------------------------------------------------------------------------


def test_fake_sdk_records_the_pinned_base_url_and_vertexai_false(monkeypatch):
    rec = {}
    _install_gemini(monkeypatch, rec)
    _make("GeminiProvider")(api_key="k")
    from codedoc.llm.api_provider import _GEMINI_DEFAULT_BASE_URL

    assert rec["init"]["http_options"].base_url == _GEMINI_DEFAULT_BASE_URL
    assert rec["init"]["vertexai"] is False


def test_direct_construction_with_real_sdk_ignores_synthetic_overrides(monkeypatch):
    from codedoc.llm.api_provider import GeminiProvider, _GEMINI_DEFAULT_BASE_URL
    from codedoc.llm.factory import effective_endpoint_identity

    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://synthetic-gemini.invalid/")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    provider = GeminiProvider(api_key="fake-key")
    http_options = provider._client._api_client._http_options
    assert effective_endpoint_identity(
        http_options.base_url
    ) == effective_endpoint_identity(_GEMINI_DEFAULT_BASE_URL)
    assert provider._client.vertexai is False


def test_pinned_transport_is_byte_identical_to_the_prior_unpinned_construction(
    monkeypatch,
):
    """Clean environment, no overrides set: compare the pinned GeminiProvider
    construction against the exact prior (pre-FIX-4) unpinned construction
    -- base URL, API version, header names, and timeout must all match."""
    from google import genai
    from google.genai import types

    from codedoc.llm.api_provider import GeminiProvider

    pinned = GeminiProvider(api_key="fake-key", timeout=45, max_retries=2)
    pinned_options = pinned._client._api_client._http_options

    # The exact pre-FIX-4 construction (codedoc/llm/api_provider.py at
    # commit 49221c2): no base_url, no vertexai argument.
    prior_http_options = types.HttpOptions(
        timeout=int(45 * 1000),
        retry_options=types.HttpRetryOptions(attempts=2),
    )
    prior_client = genai.Client(api_key="fake-key", http_options=prior_http_options)
    prior_options = prior_client._api_client._http_options

    assert pinned_options.base_url == prior_options.base_url
    assert pinned_options.api_version == prior_options.api_version
    assert set(pinned_options.headers or {}) == set(prior_options.headers or {})
    assert pinned_options.timeout == prior_options.timeout


def test_premise_unpinned_real_sdk_would_have_honored_the_overrides(monkeypatch):
    """Premise: the installed google-genai SDK really falls back to
    GOOGLE_GEMINI_BASE_URL and the Vertex-switch variables when no explicit
    base_url/vertexai argument is given."""
    from google import genai
    from google.genai import types

    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://synthetic-gemini.invalid/")
    unpinned = genai.Client(
        api_key="fake-key", http_options=types.HttpOptions(timeout=30000)
    )
    assert "synthetic-gemini.invalid" in unpinned._api_client._http_options.base_url

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    unpinned_vertex = genai.Client(
        api_key="fake-key", http_options=types.HttpOptions(timeout=30000)
    )
    assert unpinned_vertex.vertexai is True
