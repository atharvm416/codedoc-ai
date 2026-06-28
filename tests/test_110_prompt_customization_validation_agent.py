"""0.11.0 — bounded standards/safety review: batching, aggregation, fail-closed.

Covers Tests #9, #10, #11 of the plan (the provider-isolated parts; the
pipeline-ordering parts are in test_110_prompt_profile_cli.py).
"""

import json

import pytest

from codedoc.agents.prompt_customization_validation_agent import (
    PromptCustomizationValidationAgent,
)
from codedoc.core import prompt_profiles as pp
from codedoc.core.prompt_profiles import (
    MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCH_CHARS,
    MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCHES,
    MAX_PROMPT_PROFILE_FILE_BYTES,
    ResolvedProfile,
    ReviewUnit,
    build_review_batches,
    build_review_units,
    clean_review_verdict,
    export_default_profile_dict,
    pack_review_batches,
    validate_profile,
)
from codedoc.utils.errors import PromptCustomizationValidationError

LANGS = frozenset({"python", "typescript", "java"})


def _resolved(raw, mode="single", languages=LANGS):
    return ResolvedProfile(
        mode,
        validate_profile(
            raw, active_mode=mode, known_languages=languages,
            source="inline", source_path=None,
        ),
    )


def _active_single():
    raw = export_default_profile_dict("single")
    raw["single"]["fields"][0]["instruction"] = "custom description"
    return _resolved(raw)


class SequenceProvider:
    """Returns a queued verdict per call (one per batch)."""

    provider_name = "fake"

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    def complete_json(self, prompt, system=""):
        payload = dict(self._verdicts[self.calls])
        review_id = next(
            line.split(": ", 1)[1]
            for line in prompt.splitlines()
            if line.startswith("review_id: ")
        )
        ordinal, count = next(
            line.split(": ", 1)[1].split("/", 1)
            for line in prompt.splitlines()
            if line.startswith("batch: ")
        )
        payload.setdefault("review_id", review_id)
        payload.setdefault("batch_index", int(ordinal))
        payload.setdefault("batch_count", int(count))
        self.calls += 1
        return json.dumps(payload)

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


class RaisingProvider:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        raise RuntimeError("transport boom")

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def _verdict(v, reasons=None, warnings=None):
    return {"verdict": v, "reasons": reasons or [], "warnings": warnings or []}


def _clean(payload):
    return clean_review_verdict(
        {"review_id": "rev-test", "batch_index": 1, "batch_count": 1, **payload},
        expected_review_id="rev-test",
        expected_batch_index=1,
        expected_batch_count=1,
    )


# ---------------------------------------------------------------------------
# Review reachability / no-review cases (Test #11)
# ---------------------------------------------------------------------------

def test_no_review_for_no_profile_or_no_languages_or_default_equivalent():
    assert build_review_batches(ResolvedProfile("single", None), frozenset({"python"})) == []
    assert build_review_batches(_active_single(), frozenset()) == []
    default_equiv = _resolved(export_default_profile_dict("single"))
    assert build_review_batches(default_equiv, frozenset({"python"})) == []


def test_ordinary_active_profile_makes_one_batch():
    batches = build_review_batches(_active_single(), frozenset({"python"}))
    assert len(batches) == 1
    assert batches[0].ordinal == 1 and batches[0].count == 1
    assert "custom description" in batches[0].text
    assert "description (string)" in batches[0].text
    assert '"review_id": "rev-' in batches[0].text


def test_base_block_deduplicated_across_languages():
    # Two planned languages with no override share one base component.
    units, comps = build_review_units(_active_single(), frozenset({"python", "java"}))
    assert list(comps) == ["single/combined/*"]


def test_only_selected_mode_is_reviewed():
    raw = {
        "single": {"fields": [
            {"key": "description", "type": "string", "instruction": "single custom"}]},
        "triple": {
            "structure": {"fields": [
                {"key": "description", "type": "string", "instruction": "tri custom"}]},
            "dependency": {"fields": [
                {"key": "dependencies_analysis.internal", "type": "string_list",
                 "instruction": "i"}]},
            "documentation": {"fields": [
                {"key": "description", "type": "string", "instruction": "tri doc"}]},
        },
    }
    single_units, _ = build_review_units(_resolved(raw, "single"), frozenset({"python"}))
    assert all(u.component.startswith("single/") for u in single_units)
    triple_units, _ = build_review_units(_resolved(raw, "triple"), frozenset({"python"}))
    assert all(u.component.startswith("triple/") for u in triple_units)


# ---------------------------------------------------------------------------
# Batch packing boundaries (Test #11)
# ---------------------------------------------------------------------------

def test_large_profile_packs_into_multiple_bounded_batches():
    fields = [
        {"key": f.path, "type": f.type, "instruction": "X" * 1900}
        for f in pp.iter_fields("single", "combined")
    ]
    batches = build_review_batches(_resolved({"single": {"fields": fields}}), frozenset({"python"}))
    assert len(batches) >= 2
    assert all(len(b.text) <= MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCH_CHARS for b in batches)
    # Every unit appears exactly once across batches (none truncated or omitted).
    total_units = sum(len(b.units) for b in batches)
    assert total_units == len(fields)
    assert all(b.count == len(batches) for b in batches)
    # Every stateless batch has the complete manifest for represented components.
    assert all("description (string)" in batch.text for batch in batches)


def test_file_ceiling_packs_within_max_batches():
    # A profile near the file-size ceiling: full base + several language overrides,
    # all planned, must never need more than MAX batches.
    langs = ["python", "typescript", "tsx", "javascript", "jsx", "dart", "java", "csharp"]
    known = frozenset(langs)
    base = [{"key": f.path, "type": f.type, "instruction": "X" * 1900}
            for f in pp.iter_fields("single", "combined")]
    per_language = {
        lang: {"fields": [
            {"key": f.path, "type": f.type, "instruction": (lang[:1] or "x") * 1900}
            for f in pp.iter_fields("single", "combined")]}
        for lang in langs
    }
    raw = {"single": {"fields": base, "per_language": per_language}}
    size = len(json.dumps(raw).encode("utf-8"))
    assert size <= MAX_PROMPT_PROFILE_FILE_BYTES
    batches = build_review_batches(_resolved(raw, languages=known), known)
    assert 0 < len(batches) <= MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCHES
    assert all(len(b.text) <= MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCH_CHARS for b in batches)


def test_indivisible_oversized_unit_rejected():
    huge = ReviewUnit(
        component="single/combined/*", field_path="description",
        field_type="string", instruction="Z" * (MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCH_CHARS + 100),
    )
    comps = {"single/combined/*": pp.ReviewComponent(
        component="single/combined/*", block_text="x",
        fields=(("description", "string"),))}
    with pytest.raises(PromptCustomizationValidationError, match="too large"):
        pack_review_batches([huge], comps, "rev-x")


def test_exceeding_max_batches_rejected(monkeypatch):
    monkeypatch.setattr(pp, "MAX_PROMPT_CUSTOMIZATION_REVIEW_BATCHES", 1)
    fields = [
        {"key": f.path, "type": f.type, "instruction": "X" * 1900}
        for f in pp.iter_fields("single", "combined")
    ]
    with pytest.raises(PromptCustomizationValidationError, match="hard limit"):
        build_review_batches(_resolved({"single": {"fields": fields}}), frozenset({"python"}))


# ---------------------------------------------------------------------------
# Three-tier aggregation (Test #10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["SAFE", "RISKY", "TOO_RISKY"])
def test_single_batch_outcomes(verdict):
    payload = _verdict(
        verdict,
        reasons=["blocked"] if verdict == "TOO_RISKY" else [],
        warnings=["warn"] if verdict == "RISKY" else [],
    )
    batches = build_review_batches(_active_single(), frozenset({"python"}))
    outcome = PromptCustomizationValidationAgent(SequenceProvider([payload])).review(batches)
    assert outcome.verdict == verdict
    assert outcome.calls_completed == len(batches)


def test_highest_severity_aggregation_and_warning_dedup():
    fields = [
        {"key": f.path, "type": f.type, "instruction": "X" * 1900}
        for f in pp.iter_fields("single", "combined")
    ]
    batches = build_review_batches(_resolved({"single": {"fields": fields}}), frozenset({"python"}))
    assert len(batches) >= 2
    # batch1 RISKY(warn "dup"), batch2 TOO_RISKY(warn "dup", reason "r") -> TOO_RISKY,
    # warnings de-duplicated.
    provider = SequenceProvider([
        _verdict("RISKY", warnings=["dup"]),
        _verdict("TOO_RISKY", reasons=["r"], warnings=["dup"]),
    ] + [_verdict("SAFE")] * 8)
    outcome = PromptCustomizationValidationAgent(provider).review(batches)
    assert outcome.verdict == "TOO_RISKY"
    assert outcome.reasons == ("r",)
    assert outcome.warnings == ("dup",)


def test_run_level_warning_aggregation_is_globally_bounded():
    fields = [
        {"key": f.path, "type": f.type, "instruction": "X" * 1900}
        for f in pp.iter_fields("single", "combined")
    ]
    batches = build_review_batches(
        _resolved({"single": {"fields": fields}}), frozenset({"python"})
    )
    assert len(batches) >= 2
    payloads = [
        _verdict(
            "RISKY",
            warnings=[f"batch-{batch_index}-warning-{i}" for i in range(16)],
        )
        for batch_index in range(len(batches))
    ]
    outcome = PromptCustomizationValidationAgent(SequenceProvider(payloads)).review(
        batches
    )
    assert len(outcome.warnings) == pp.MAX_PROFILE_SECURITY_WARNINGS


# ---------------------------------------------------------------------------
# Fail-closed (Test #10)
# ---------------------------------------------------------------------------

def test_transport_failure_fails_closed():
    batches = build_review_batches(_active_single(), frozenset({"python"}))
    with pytest.raises(PromptCustomizationValidationError, match="failed closed"):
        PromptCustomizationValidationAgent(RaisingProvider()).review(batches)


def test_contradictory_and_unknown_verdicts_fail_closed():
    batches = build_review_batches(_active_single(), frozenset({"python"}))
    for bad in (
        _verdict("SAFE", reasons=["x"]),       # SAFE must not carry reasons
        _verdict("TOO_RISKY"),                  # TOO_RISKY needs a reason
        {"verdict": "MAYBE", "reasons": [], "warnings": []},
    ):
        with pytest.raises(PromptCustomizationValidationError):
            PromptCustomizationValidationAgent(SequenceProvider([bad])).review(batches)


def test_batch_binding_mismatch_fails_closed():
    batches = build_review_batches(_active_single(), frozenset({"python"}))
    bad = {**_verdict("SAFE"), "review_id": "rev-wrong"}
    with pytest.raises(PromptCustomizationValidationError, match="binding mismatch"):
        PromptCustomizationValidationAgent(SequenceProvider([bad])).review(batches)


def test_preamble_or_fenced_response_fails_closed():
    class Raw:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return '```json\n{"verdict":"SAFE","reasons":[],"warnings":[]}\n```'

    batches = build_review_batches(_active_single(), frozenset({"python"}))
    with pytest.raises(PromptCustomizationValidationError):
        PromptCustomizationValidationAgent(Raw()).review(batches)


@pytest.mark.parametrize(
    ("installer", "class_name", "response_kw"),
    [
        ("_install_openai", "OpenAIProvider", "content"),
        ("_install_anthropic", "AnthropicProvider", "text"),
        ("_install_gemini", "GeminiProvider", "text"),
    ],
)
def test_provider_adapters_receive_identical_security_review_batches(
    monkeypatch, installer, class_name, response_kw
):
    from codedoc.llm import api_provider
    from tests import test_095_provider_contracts as contracts

    rec = {}
    batches = build_review_batches(_active_single(), frozenset({"python"}))
    response = json.dumps({
        **_verdict("SAFE"),
        "review_id": batches[0].review_id,
        "batch_index": 1,
        "batch_count": 1,
    })
    getattr(contracts, installer)(monkeypatch, rec, **{response_kw: response})
    provider = getattr(api_provider, class_name)(api_key="k")
    outcome = PromptCustomizationValidationAgent(provider).review(batches)
    assert outcome.verdict == "SAFE"
    if class_name == "OpenAIProvider":
        sent = rec["create_kwargs"]["messages"][-1]["content"]
    elif class_name == "AnthropicProvider":
        sent = rec["create_kwargs"]["messages"][0]["content"]
    else:
        sent = rec["generate"]["contents"]
    assert sent == batches[0].text


# ---------------------------------------------------------------------------
# Verdict cleaner (Test #10) — bounded while cleaning, de-duplicated.
# ---------------------------------------------------------------------------

def test_verdict_cleaner_clamps_and_dedupes():
    v = _clean({"verdict": "RISKY", "warnings": ["a", "a", "b"] + [f"w{i}" for i in range(40)]})
    assert v["verdict"] == "RISKY"
    assert v["warnings"][:2] == ["a", "b"]
    assert len(v["warnings"]) == pp.MAX_PROFILE_SECURITY_WARNINGS
    # missing lists default to []; extra keys ignored.
    assert _clean({"verdict": "SAFE", "x": 1}) == {
        "review_id": "rev-test", "batch_index": 1, "batch_count": 1,
        "verdict": "SAFE", "reasons": [], "warnings": []}
