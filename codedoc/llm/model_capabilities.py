"""Exact-model capability policy for CodeDoc-owned default provider requests.

A small, pure, network-free lookup table of *exact* documented model IDs
(never a prefix or family match) that chooses CodeDoc's internal default
request fields and, for Anthropic JSON calls, a per-model output-token cap.
An unknown or unlisted model ID
keeps the historical legacy request shape unchanged.

An unsupplied ``temperature`` on the public ``complete`` path follows this
exact-model policy too. A caller-supplied value is always forwarded unchanged.

Sources are cited next to each entry and were verified against official
documentation as of the read date given beside each citation. Re-check
before trusting an entry well after that date; provider docs, limits, and
model availability change.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnthropicModelCapability:
    """CodeDoc-owned request policy for one exact Anthropic model ID.

    ``max_tokens`` is CodeDoc's own conservative output cap for this model,
    not the model's full documented maximum -- see the cap-calculation note
    beside each entry below.
    """

    model_id: str
    supports_temperature: bool
    max_tokens: int
    citation: str


# Output-cap calculation for the 8192-token entries below. Opus 4.7 and 4.8
# keep the legacy 4096-token cap because thinking is off by default.
#
# CodeDoc does not stream, and constructs the Anthropic client with an
# explicit ``timeout`` (the resolved ``provider_request_timeout_s``, default
# 120s) -- so the SDK's own default non-streaming timeout guard, which only
# activates when the client uses the SDK's *default* timeout, does not
# protect this call. The cap must therefore be chosen so a full
# thinking-plus-JSON response can plausibly complete inside 120s.
#
# Anthropic does not publish an exact standard-speed (non-fast-mode) output
# tokens/second figure for any model; Fast mode's own documentation only
# states a relative "up to 2.5x" multiplier over standard speed without
# giving the baseline. In the absence of a published figure, this is
# CodeDoc's own conservative internal assumption, not an Anthropic claim:
#
#   assumed sustained throughput ........... 100 output tokens/second
#   reserved overhead (connection/TTFT) ..... 20 seconds of the 120s budget
#   generation budget ....................... 100 seconds
#   raw budget-driven cap ................... 100s x 100 tok/s = 10,000 tokens
#   chosen cap (headroom under raw budget) .. 8,192 tokens
#
# 8,192 leaves roughly 18% margin under the raw calculation (82s of assumed
# generation time against a 100s budget), stays far under the Python SDK's
# non-streaming ceiling (~21k tokens; the SDK's own docs recommend streaming
# above that), and stays far under Claude Opus 5's documented 128k max output
# tokens. It also gives thinking (on by default, billed as output tokens and
# counted toward this same max_tokens ceiling) meaningful room ahead of the
# final JSON text, unlike merely reusing Haiku 4.5's unrelated 4096 cap. This
# is a latency *expectation*, not a guarantee.
_ANTHROPIC_NEW_MODEL_OUTPUT_CAP = 8192

_ANTHROPIC_MODELS: dict[str, AnthropicModelCapability] = {
    # Existing default. Recorded here so lookups resolve through one path;
    # NOT a new capability-table entry and NOT subject to the cap calculation
    # above -- this is the historical request, kept byte-identical intact.
    "claude-haiku-4-5-20251001": AnthropicModelCapability(
        model_id="claude-haiku-4-5-20251001",
        supports_temperature=True,
        max_tokens=4096,
        citation=(
            "Unchanged historical CodeDoc default (codedoc/llm/api_provider.py "
            "AnthropicProvider.complete); not re-derived from new provider "
            "documentation."
        ),
    ),
    "claude-opus-4-7": AnthropicModelCapability(
        model_id="claude-opus-4-7",
        supports_temperature=False,
        max_tokens=4096,
        citation=(
            "https://platform.claude.com/docs/en/build-with-claude/thinking "
            "and https://platform.claude.com/docs/en/about-claude/model-deprecations "
            "(read 2026-09-24): non-default sampling returns 400; exact ID active. "
            "The thinking page documents 128K max output; thinking is off by "
            "default (https://platform.claude.com/docs/en/build-with-claude/"
            "thinking-troubleshooting)."
        ),
    ),
    "claude-opus-4-8": AnthropicModelCapability(
        model_id="claude-opus-4-8",
        supports_temperature=False,
        max_tokens=4096,
        citation=(
            "https://platform.claude.com/docs/en/build-with-claude/thinking "
            "and https://platform.claude.com/docs/en/about-claude/model-deprecations "
            "(read 2026-09-24): non-default sampling returns 400; exact ID active. "
            "The thinking page documents 128K max output; thinking is off by "
            "default (https://platform.claude.com/docs/en/build-with-claude/"
            "thinking-troubleshooting)."
        ),
    ),
    "claude-sonnet-5": AnthropicModelCapability(
        model_id="claude-sonnet-5",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/models/sonnet-5/overview "
            "(read 2026-09-24): exact Claude API ID, adaptive thinking on by "
            "default, 128K max output, non-default sampling returns 400."
        ),
    ),
    "claude-opus-5": AnthropicModelCapability(
        model_id="claude-opus-5",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/models/opus-5/migration-guide "
            "(read 2026-09-23): \"Setting temperature, top_p, or top_k to a "
            "non-default value returns a 400 error on Claude Opus 5, the same "
            "as on Claude Opus 4.7.\" Documented max output: 128k tokens and "
            "thinking is on by default, billed as output tokens counted toward "
            "max_tokens (https://platform.claude.com/docs/en/models/opus-5/"
            "whats-new-opus-5, same read date)."
        ),
    ),
    "claude-opus-5-5": AnthropicModelCapability(
        model_id="claude-opus-5-5",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/build-with-claude/"
            "working-with-messages (read 2026-09-24): \"The temperature, "
            "top_p, and top_k sampling parameters are not supported on "
            "Claude 4.7 and later models and Claude Mythos Preview. Setting "
            "them to a non-default value returns a 400 error.\" The page's "
            "own example request uses model=\"claude-opus-5-5\" directly. "
            "Adaptive thinking is always on; the effort parameter controls "
            "thinking depth in place of temperature."
        ),
    ),
    "claude-fable-5": AnthropicModelCapability(
        model_id="claude-fable-5",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/models/fable-5/overview "
            "and https://platform.claude.com/docs/en/build-with-claude/thinking "
            "(read 2026-09-24): exact Claude API ID, always-on adaptive thinking, "
            "128K max output, non-default sampling returns 400."
        ),
    ),
    "claude-fable-5-1": AnthropicModelCapability(
        model_id="claude-fable-5-1",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/models/fable-5-1/overview "
            "and https://platform.claude.com/docs/en/build-with-claude/thinking "
            "(read 2026-09-24): exact Claude API ID, always-on adaptive thinking, "
            "128K max output, non-default sampling returns 400; slower latency."
        ),
    ),
    "claude-mythos-5": AnthropicModelCapability(
        model_id="claude-mythos-5",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/models/mythos-5/overview "
            "and https://platform.claude.com/docs/en/build-with-claude/thinking "
            "(read 2026-09-24): exact invite-only Claude API ID, always-on "
            "adaptive thinking, 128K max output, non-default sampling returns 400."
        ),
    ),
    "claude-mythos-5-1": AnthropicModelCapability(
        model_id="claude-mythos-5-1",
        supports_temperature=False,
        max_tokens=_ANTHROPIC_NEW_MODEL_OUTPUT_CAP,
        citation=(
            "https://platform.claude.com/docs/en/models/mythos-5-1/overview "
            "and https://platform.claude.com/docs/en/build-with-claude/thinking "
            "(read 2026-09-24): exact invite-only Claude API ID, always-on "
            "adaptive thinking, 128K max output, non-default sampling returns 400; "
            "slower latency."
        ),
    ),
}


def anthropic_capability(model: str) -> AnthropicModelCapability | None:
    """Exact-ID lookup only -- never a prefix match.

    Returns ``None`` for any model not in the closed table above, including
    every unlisted Anthropic snapshot: callers must keep the legacy request
    shape (``temperature`` sent, ``max_tokens=4096``) for an unknown model and
    let a real incompatibility surface as an ordinary provider error.
    """
    return _ANTHROPIC_MODELS.get(model)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

# Exact IDs for which CodeDoc omits its own default temperature on the
# default OpenAI endpoint. Each entry carries its evidence; dated GPT-5
# snapshot behavior is an inference from the documented alias-to-snapshot
# mapping. No prefix matching or policy inheritance is used.
#
# Source: https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2
# (read 2026-09-23): "The following parameters are only supported when using
# GPT-5.2 with reasoning effort set to none: temperature, top_p, logprobs.
# Requests to GPT-5.2 or GPT-5.1 with any other reasoning effort setting, or
# to older GPT-5 models -- for example, gpt-5, gpt-5-mini, or gpt-5-nano --
# that include these fields will raise an error." Confirmed separately
# (https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5,
# same date): "reasoning.effort supports minimal, low, medium, and high" for
# gpt-5/gpt-5-mini/gpt-5-nano -- there is no `none` level for this family, so
# the rejection is unconditional for these three exact IDs.
OPENAI_TEMPERATURE_OMITTED_CITATION = (
    "https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.2 "
    "(read 2026-09-23): exact gpt-5, gpt-5-mini, gpt-5-nano raise an error "
    "when temperature/top_p/logprobs are sent, at any (all non-none) "
    "reasoning effort these models support."
)

OPENAI_TEMPERATURE_OMITTED_CITATIONS: dict[str, str] = {
    "gpt-5": OPENAI_TEMPERATURE_OMITTED_CITATION,
    "gpt-5-mini": OPENAI_TEMPERATURE_OMITTED_CITATION,
    "gpt-5-nano": OPENAI_TEMPERATURE_OMITTED_CITATION,
    "gpt-5-2025-08-07": (
        "https://developers.openai.com/api/docs/models/gpt-5 and "
        "https://developers.openai.com/api/docs/deprecations "
        "(read 2026-09-24): inference from the documented gpt-5 snapshot "
        "mapping and its sampling restriction; shutdown 2026-12-11."
    ),
    "gpt-5-mini-2025-08-07": (
        "https://developers.openai.com/api/docs/models/gpt-5-mini and "
        "https://developers.openai.com/api/docs/deprecations "
        "(read 2026-09-24): inference from the documented gpt-5-mini "
        "snapshot mapping and its sampling restriction; shutdown 2026-12-11."
    ),
    "gpt-5-nano-2025-08-07": (
        "https://developers.openai.com/api/docs/models/gpt-5-nano and "
        "https://developers.openai.com/api/docs/deprecations "
        "(read 2026-09-24): inference from the documented gpt-5-nano "
        "snapshot mapping and its sampling restriction; shutdown 2026-12-11."
    ),
    "gpt-6-astra": (
        "https://developers.openai.com/api/docs/changelog "
        "(2026-09-03): no custom temperature; no none reasoning effort."
    ),
    "gpt-6-sol": (
        "https://developers.openai.com/api/docs/models/gpt-6-sol and "
        "https://developers.openai.com/api/docs/guides/latest-model "
        "(read 2026-09-24): default effort medium; remove temperature "
        "when effort is not none."
    ),
    "gpt-6-luna": (
        "https://developers.openai.com/api/docs/models/gpt-6-luna and "
        "https://developers.openai.com/api/docs/guides/latest-model "
        "(read 2026-09-24): default effort medium; remove temperature "
        "when effort is not none."
    ),
}

# Exact IDs confirmed to keep accepting the existing temperature=0.0 request
# byte-identical: default reasoning effort is `none` (GPT-5.1/5.2) and
# CodeDoc sends no `reasoning.effort`, so these run at `none` by default,
# the one level at which temperature/top_p/logprobs remain accepted.
#
# Source: https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.1
# (read 2026-09-23): "reasoning.effort supports none (the default), low,
# medium, and high." GPT-5.2 page (same date): "With GPT-5.2, the lowest
# setting is none ... This is the default setting in GPT-5.2." Recorded for
# documentation/contract-testing; these IDs need no table entry since they
# take the unmodified legacy request path identically to an unknown model.
OPENAI_TEMPERATURE_PRESERVED_MODELS: frozenset[str] = frozenset({"gpt-5.1", "gpt-5.2"})


def openai_omits_temperature(model: str) -> bool:
    """True only for audited exact IDs that omit the default temperature.

    Every other model -- including gpt-5.1, gpt-5.2, and unlisted snapshots --
    keeps the legacy request (temperature sent) and, if genuinely
    incompatible, surfaces the resulting provider error normally rather than
    being guessed at here.

    This is a model-ID judgment only: it has no notion of endpoint. A model
    string of ``"gpt-5"`` served through a custom OpenAI-compatible
    ``base_url`` (a local or self-hosted endpoint) is not necessarily the
    real OpenAI model and must not be assumed to share its capability
    policy. The caller (``codedoc.llm.api_provider.OpenAIProvider``) applies
    this lookup only when it is also using the default OpenAI endpoint.
    """
    return model in OPENAI_TEMPERATURE_OMITTED_CITATIONS
