"""Unit tests for the exact-model capability policy (0.15.0 FIX-1).

Covers the pure, network-free lookup in isolation: exact-ID matching (never a
prefix/family match), the closed set of models that omit CodeDoc's default
``temperature``, and that every table entry carries a non-empty citation.
"""

from __future__ import annotations

import pytest

from codedoc.llm.model_capabilities import (
    OPENAI_TEMPERATURE_OMITTED_CITATIONS,
    OPENAI_TEMPERATURE_PRESERVED_MODELS,
    AnthropicModelCapability,
    anthropic_capability,
    openai_omits_temperature,
)


# ---------------------------------------------------------------------------
# Anthropic exact-ID lookup
# ---------------------------------------------------------------------------


def test_haiku_capability_is_unchanged_from_the_historical_default():
    capability = anthropic_capability("claude-haiku-4-5-20251001")
    assert capability == AnthropicModelCapability(
        model_id="claude-haiku-4-5-20251001",
        supports_temperature=True,
        max_tokens=4096,
        citation=capability.citation,
    )
    assert capability.citation


def test_opus_5_omits_temperature_and_has_a_conservative_cap():
    capability = anthropic_capability("claude-opus-5")
    assert capability is not None
    assert capability.supports_temperature is False
    # Within Claude Opus 5's documented 128k max output tokens, and safely
    # under the Python SDK's non-streaming ceiling (~21k) since CodeDoc never
    # streams.
    assert 0 < capability.max_tokens <= 21_000
    assert capability.citation


def test_opus_5_5_omits_temperature_and_has_a_conservative_cap():
    capability = anthropic_capability("claude-opus-5-5")
    assert capability is not None
    assert capability.supports_temperature is False
    assert 0 < capability.max_tokens <= 21_000
    assert capability.citation


def test_unknown_anthropic_model_returns_none():
    """An unlisted model keeps the legacy request shape -- signaled by a
    ``None`` lookup result, never a guessed/default capability."""
    assert anthropic_capability("claude-test") is None
    assert anthropic_capability("claude-haiku-4-5") is None  # no exact match
    assert anthropic_capability("claude-opus-4-6") is None
    assert anthropic_capability("") is None


def test_exact_match_only_no_prefix_leakage_between_opus_snapshots():
    """'claude-opus-5' must not determine the policy for a differently-named
    snapshot, and vice versa -- exact ID matching only."""
    assert anthropic_capability("claude-opus-5-preview") is None
    assert anthropic_capability("claude-opus-5x") is None
    assert anthropic_capability("claude-opus-55") is None


def test_every_anthropic_entry_has_a_citation_and_valid_max_tokens():
    for model_id in (
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5",
        "claude-mythos-5-1",
    ):
        capability = anthropic_capability(model_id)
        assert capability is not None, model_id
        assert capability.model_id == model_id
        assert isinstance(capability.citation, str) and capability.citation
        assert capability.max_tokens > 0


@pytest.mark.parametrize(
    ("model_id", "cap"),
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
def test_new_anthropic_ids_have_the_exact_policy_and_primary_citation(model_id, cap):
    capability = anthropic_capability(model_id)
    assert capability is not None
    assert capability.model_id == model_id
    assert capability.supports_temperature is False
    assert capability.max_tokens == cap
    assert capability.citation.startswith("https://platform.claude.com/")


@pytest.mark.parametrize(
    "model_id",
    [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-sonnet-5-preview",
        "claude-mythos-preview",
    ],
)
def test_near_miss_anthropic_ids_keep_legacy_policy(model_id):
    assert anthropic_capability(model_id) is None


# ---------------------------------------------------------------------------
# OpenAI exact-ID temperature policy -- no prefix leakage across the family
# ---------------------------------------------------------------------------


def test_exact_gpt5_family_omits_temperature():
    assert openai_omits_temperature("gpt-5") is True
    assert openai_omits_temperature("gpt-5-mini") is True
    assert openai_omits_temperature("gpt-5-nano") is True


def test_gpt51_and_gpt52_preserve_temperature():
    assert openai_omits_temperature("gpt-5.1") is False
    assert openai_omits_temperature("gpt-5.2") is False
    assert {"gpt-5.1", "gpt-5.2"} == set(OPENAI_TEMPERATURE_PRESERVED_MODELS)


def test_no_prefix_leakage_between_gpt5_gpt51_and_dated_snapshots():
    """'gpt-5' must not determine the policy for 'gpt-5.1' or 'gpt-5-mini'
    beyond the exact, explicitly audited set -- an unlisted dated snapshot
    or unrelated model keeps the legacy request."""
    assert openai_omits_temperature("gpt-5.1") is False
    assert openai_omits_temperature("gpt-5-mini-2025-08-07") is True
    assert openai_omits_temperature("gpt-5.2-chat-latest") is False
    assert openai_omits_temperature("gpt-4o-mini") is False
    assert openai_omits_temperature("gpt-4.1") is False
    assert openai_omits_temperature("o3") is False
    assert openai_omits_temperature("") is False
    assert openai_omits_temperature("custom-openai-compatible-model") is False


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-6-astra",
        "gpt-6-sol",
        "gpt-6-luna",
    ],
)
def test_new_openai_ids_have_exact_omission_and_primary_citation(model_id):
    assert openai_omits_temperature(model_id)
    assert OPENAI_TEMPERATURE_OMITTED_CITATIONS[model_id].startswith(
        "https://developers.openai.com/"
    )


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.5", "gpt-5.6-sol", "gpt-6", "gpt-6-astra-mini"],
)
def test_near_miss_openai_ids_keep_legacy_policy(model_id):
    assert openai_omits_temperature(model_id) is False
