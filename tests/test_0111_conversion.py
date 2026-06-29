"""0.11.1 — single-to-triple conversion: classification, dry-run, paid proposal,
fail-closed routing, attempt accounting, CLI summaries, and cycle-warning dedup.

Covers Test Plan items #13-#23, #36, #38, #41, #42, and Workstream H. Providers
are always fakes; no test makes a network call.
"""

import json
import logging

import pytest

import codedoc.pipeline as pipe
from codedoc.core import prompt_profiles as pp
from codedoc.core.graph import DependencyGraph
from codedoc.core.prompt_profiles import (
    AgentProfile,
    ShapeFieldSpec,
    build_routing_request,
    classify_profile_action,
    routing_conversion_id,
    validate_profile,
)
from codedoc.utils.errors import ConfigError, PromptCustomizationValidationError

KL = frozenset({"python"})

# A customized single structure (v2) that is NOT developer-standard-equivalent.
CUSTOM_SINGLE_V2 = {"single": {"requested_shape": {"description": "Custom explain."}}}
CUSTOM_SINGLE_V1 = {"single": {"fields": [
    {"key": "description", "type": "string", "instruction": "Custom explain."}]}}


def _default_triple_response(cid):
    return {
        "conversion_id": cid,
        "triple": {
            "structure": {"requested_shape": {"description": "Structure of the file."}},
            "dependency": {"requested_shape": {
                "dependencies_analysis": {"internal": ["Project dep."]}}},
            "documentation": {"requested_shape": {"description": "Docs for maintainers."}},
        },
        "factors": ["split by concern", "split by concern", ""],
    }


class ConversionFake:
    """Serves review verdicts and one routing response, branching on the prompt."""

    provider_name = "fake"

    def __init__(self, *, review_verdict="SAFE", routing_raw=None, triple=None):
        self.review_verdict = review_verdict
        self.routing_raw = routing_raw
        self.triple = triple
        self.review_calls = 0
        self.routing_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            review_id = next(
                line.split(": ", 1)[1] for line in prompt.splitlines()
                if line.startswith("review_id: ")
            )
            ordinal, count = next(
                line.split(": ", 1)[1].split("/", 1) for line in prompt.splitlines()
                if line.startswith("batch: ")
            )
            return json.dumps({
                "review_id": review_id, "batch_index": int(ordinal),
                "batch_count": int(count), "verdict": self.review_verdict,
                "reasons": ["bad"] if self.review_verdict == "TOO_RISKY" else [],
                "warnings": ["w"] if self.review_verdict == "RISKY" else [],
            })
        if "single-to-triple prompt-profile routing" in prompt:
            self.routing_calls += 1
            if self.routing_raw is not None:
                return self.routing_raw
            cid = next(
                line.split(": ", 1)[1] for line in prompt.splitlines()
                if line.startswith("conversion_id: ")
            )
            resp = _default_triple_response(cid)
            if self.triple is not None:
                resp["triple"] = self.triple
            return json.dumps(resp)
        raise AssertionError(f"unexpected prompt: {prompt[:60]}")

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _run(monkeypatch, project, cfg, fake):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: fake)
    return pipe.run_pipeline(project, cfg)


def _no_provider(monkeypatch):
    def boom(_cfg):
        raise AssertionError("provider must not be created")
    monkeypatch.setattr("codedoc.pipeline.create_provider", boom)


# ---------------------------------------------------------------------------
# Classification (#13, #14, #15, #21, #22, #24)
# ---------------------------------------------------------------------------

def _classify(raw, mode, source="inline"):
    profile = validate_profile(
        raw, active_mode=mode, known_languages=KL, source=source, source_path=None
    )
    return classify_profile_action(profile, mode)


def test_single_mode_uses_single_directly():
    assert _classify(CUSTOM_SINGLE_V2, "single").action == pp.PROFILE_ACTION_EXECUTABLE


def test_explicit_triple_is_executable():
    raw = {"triple": {
        "structure": {"requested_shape": {"description": "s"}},
        "dependency": {"requested_shape": {"dependencies_analysis": {"internal": ["i"]}}},
        "documentation": {"requested_shape": {"description": "d"}},
    }}
    assert _classify(raw, "triple").action == pp.PROFILE_ACTION_EXECUTABLE


def test_default_equivalent_single_in_triple_is_local_default():
    raw = {"single": {"requested_shape": pp._default_requested_shape("single", "combined")}}
    assert _classify(raw, "triple").action == pp.PROFILE_ACTION_LOCAL_DEFAULT


def test_customized_single_in_triple_requires_conversion():
    assert _classify(CUSTOM_SINGLE_V2, "triple").action == pp.PROFILE_ACTION_CONVERSION_REQUIRED
    assert _classify(CUSTOM_SINGLE_V1, "triple").action == pp.PROFILE_ACTION_CONVERSION_REQUIRED


def test_single_only_with_language_override_rejects_conversion():
    raw = {"single": {
        "requested_shape": {"description": "custom"},
        "per_language": {"python": {"requested_shape": {"description": "py"}}},
    }}
    with pytest.raises(ConfigError, match="per-language overrides"):
        _classify(raw, "triple")


def test_external_single_only_in_triple_keeps_legacy_rejection(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(CUSTOM_SINGLE_V1), encoding="utf-8")
    with pytest.raises(ConfigError, match="no 'triple' section"):
        pp.resolve_profile_source(
            {"prompt_profile_file": "p.json"}, tmp_path,
            known_languages=KL, active_mode="triple",
        )


# ---------------------------------------------------------------------------
# Dry-run conversion (#16)
# ---------------------------------------------------------------------------

def test_dry_run_conversion_reports_pending_no_provider(monkeypatch, project):
    _no_provider(monkeypatch)
    stats = pipe.run_pipeline(project, {
        "entry_file": "main.py", "analysis_mode": "triple",
        "prompt_profiles": CUSTOM_SINGLE_V2, "dry_run": True,
    })
    assert stats["prompt_profile_conversion"] == "pending"
    assert stats["prompt_profile_conversion_calls_planned"] == 1
    assert stats["prompt_customization_security_review_calls_planned"] == 1
    assert stats["total_paid_proposal_calls"] == 2
    assert not (project / "codedoc").exists()


def test_conversion_branch_precedes_entry_scan_graph_and_planning(monkeypatch, project):
    """Addendum 11: proposal planning must branch before project inspection."""
    _no_provider(monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary documentation pipeline was reached")

    for name in (
        "_resolve_entry_and_docs",
        "select_active_recovery_path",
        "inspect_output_ownership",
        "_load_existing_file_docs",
        "scan_files",
        "_build_graph",
        "build_pipeline_plan",
    ):
        monkeypatch.setattr(pipe, name, forbidden)

    stats = pipe.run_pipeline(
        project,
        {
            "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
            "dry_run": True,
        },
    )
    assert stats["prompt_profile_conversion"] == "pending"


# ---------------------------------------------------------------------------
# Real conversion proposal (#17, #18, #41)
# ---------------------------------------------------------------------------

def test_conversion_proposal_success(monkeypatch, project):
    fake = ConversionFake()
    stats = _run(monkeypatch, project, {
        "entry_file": "main.py", "analysis_mode": "triple",
        "prompt_profiles": CUSTOM_SINGLE_V2,
    }, fake)
    assert stats["prompt_profile_conversion"] == "generated-awaiting-confirmation"
    assert fake.review_calls == 1 and fake.routing_calls == 1
    assert stats["prompt_profile_conversion_calls_completed"] == 1
    assert stats["prompt_customization_security_review_calls_completed"] == 1
    assert stats["documentation_calls_attempted"] == 0
    # factors de-duplicated + empties dropped
    assert stats["prompt_profile_conversion_factors"] == ["split by concern"]
    # no artifacts written
    assert not (project / "codedoc").exists()
    # fragment is valid JSON and accepted unchanged on rerun
    fragment = json.loads(stats["prompt_profile_conversion_fragment"])
    assert set(fragment) == {"prompt_profiles"}
    profiles = fragment["prompt_profiles"]
    assert profiles["schema_version"] == 2
    assert set(profiles["triple"]) == {"structure", "dependency", "documentation"}
    reparsed = validate_profile(
        profiles, active_mode="triple", known_languages=KL,
        source="inline", source_path=None,
    )
    assert classify_profile_action(reparsed, "triple").action == pp.PROFILE_ACTION_EXECUTABLE


def test_conversion_preserves_comment(monkeypatch, project):
    cfg_inline = {"$comment": "keep me", "single": {"requested_shape": {"description": "Custom explain."}}}
    fake = ConversionFake()
    stats = _run(monkeypatch, project, {
        "entry_file": "main.py", "analysis_mode": "triple",
        "prompt_profiles": cfg_inline,
    }, fake)
    fragment = json.loads(stats["prompt_profile_conversion_fragment"])
    assert fragment["prompt_profiles"]["$comment"] == "keep me"


def test_conversion_attempt_accounting_reconciles(monkeypatch, project):
    fake = ConversionFake()
    stats = _run(monkeypatch, project, {
        "entry_file": "main.py", "analysis_mode": "triple",
        "prompt_profiles": CUSTOM_SINGLE_V2,
    }, fake)
    total = (
        stats["documentation_calls_attempted"]
        + stats["prompt_customization_security_review_calls_attempted"]
        + stats["prompt_profile_conversion_calls_attempted"]
    )
    assert total == stats["attempted_calls"]


def test_v1_source_converts_and_emits_v2(monkeypatch, project):
    fake = ConversionFake()
    stats = _run(monkeypatch, project, {
        "entry_file": "main.py", "analysis_mode": "triple",
        "prompt_profiles": CUSTOM_SINGLE_V1,
    }, fake)
    fragment = json.loads(stats["prompt_profile_conversion_fragment"])
    assert fragment["prompt_profiles"]["schema_version"] == 2
    # the normalized single block is rendered in v2 requested_shape form
    assert "requested_shape" in fragment["prompt_profiles"]["single"]


# ---------------------------------------------------------------------------
# Fail-closed routing (#19, #20, #23)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    '```json\n{}\n```',                                   # fenced
    'sure! {"conversion_id":"x"}',                        # preamble
    '{"conversion_id":"x"} {"conversion_id":"y"}',        # multiple objects
    '{not json',                                           # malformed
])
def test_routing_malformed_fails_closed(monkeypatch, project, raw):
    fake = ConversionFake(routing_raw=raw)
    with pytest.raises(PromptCustomizationValidationError) as caught:
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, fake)
    stats = caught.value.stats
    assert stats["attempted_calls"] == 2
    assert stats["prompt_customization_security_review_calls_attempted"] == 1
    assert stats["prompt_profile_conversion_calls_attempted"] == 1
    assert stats["documentation_calls_attempted"] == 0
    assert not (project / "codedoc").exists()


def test_routing_conversion_id_mismatch_fails_closed(monkeypatch, project):
    fake = ConversionFake(routing_raw=json.dumps(_default_triple_response("route-wrong")))
    with pytest.raises(PromptCustomizationValidationError, match="conversion_id mismatch"):
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, fake)


def test_routing_nested_duplicate_key_fails_closed(monkeypatch, project):
    class NestedDuplicateFake(ConversionFake):
        def complete_json(self, prompt, system=""):
            if "single-to-triple prompt-profile routing" in prompt:
                self.routing_calls += 1
                conversion_id = next(
                    line.split(": ", 1)[1] for line in prompt.splitlines()
                    if line.startswith("conversion_id: ")
                )
                raw = json.dumps(_default_triple_response(conversion_id))
                return raw.replace(
                    '"structure": {"requested_shape":',
                    '"structure": {"requested_shape": {}, "requested_shape":',
                    1,
                )
            return super().complete_json(prompt, system)

    with pytest.raises(PromptCustomizationValidationError, match="duplicate key"):
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, NestedDuplicateFake())


@pytest.mark.parametrize("mutation", ["missing-factors", "extra-key"])
def test_routing_root_contract_is_exact(monkeypatch, project, mutation):
    raw = _default_triple_response("placeholder")
    if mutation == "missing-factors":
        raw.pop("factors")
    else:
        raw["unsupported"] = True

    class RootContractFake(ConversionFake):
        def complete_json(self, prompt, system=""):
            if "single-to-triple prompt-profile routing" in prompt:
                self.routing_calls += 1
                conversion_id = next(
                    line.split(": ", 1)[1] for line in prompt.splitlines()
                    if line.startswith("conversion_id: ")
                )
                raw["conversion_id"] = conversion_id
                return json.dumps(raw)
            return super().complete_json(prompt, system)

    with pytest.raises(PromptCustomizationValidationError, match="exactly the keys"):
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, RootContractFake())


def test_routing_invented_field_fails_closed(monkeypatch, project):
    bad_triple = {
        "structure": {"requested_shape": {"description": "s", "bogus": "x"}},
        "dependency": {"requested_shape": {"dependencies_analysis": {"internal": ["i"]}}},
        "documentation": {"requested_shape": {"description": "d"}},
    }
    fake = ConversionFake(triple=bad_triple)
    with pytest.raises(PromptCustomizationValidationError, match="not a registered"):
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, fake)


def test_routing_missing_doc_description_fails_closed(monkeypatch, project):
    bad_triple = {
        "structure": {"requested_shape": {"description": "s"}},
        "dependency": {"requested_shape": {"dependencies_analysis": {"internal": ["i"]}}},
        "documentation": {"requested_shape": {"role_in_system": "r"}},  # no description
    }
    fake = ConversionFake(triple=bad_triple)
    with pytest.raises(PromptCustomizationValidationError, match="required field 'description'"):
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, fake)


def test_too_risky_source_blocks_conversion(monkeypatch, project):
    fake = ConversionFake(review_verdict="TOO_RISKY")
    with pytest.raises(PromptCustomizationValidationError, match="TOO RISKY") as caught:
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, fake)
    assert fake.routing_calls == 0           # never reached routing
    stats = caught.value.stats
    assert stats["attempted_calls"] == 1
    assert stats["prompt_customization_security_review_calls_attempted"] == 1
    assert stats["prompt_profile_conversion_calls_attempted"] == 0
    assert not (project / "codedoc").exists()


def test_too_risky_override_reaches_routing(monkeypatch, project):
    fake = ConversionFake(review_verdict="TOO_RISKY")
    stats = _run(monkeypatch, project, {
        "entry_file": "main.py", "analysis_mode": "triple",
        "prompt_profiles": CUSTOM_SINGLE_V2,
        "prompt_customization_allow_risky": True,
    }, fake)
    assert stats["prompt_profile_conversion"] == "generated-awaiting-confirmation"
    assert fake.routing_calls == 1


def test_routing_response_over_ceiling_fails_before_parse(monkeypatch, project):
    huge = "x" * (pp.MAX_PROMPT_PROFILE_ROUTING_RESPONSE_CHARS + 10)
    fake = ConversionFake(routing_raw=huge)
    with pytest.raises(PromptCustomizationValidationError, match="ceiling"):
        _run(monkeypatch, project, {
            "entry_file": "main.py", "analysis_mode": "triple",
            "prompt_profiles": CUSTOM_SINGLE_V2,
        }, fake)


def test_routing_transport_failure_counts_attempt_and_writes_nothing(
    monkeypatch, project
):
    class TransportFailureFake(ConversionFake):
        def complete_json(self, prompt, system=""):
            if "single-to-triple prompt-profile routing" in prompt:
                self.routing_calls += 1
                raise RuntimeError("transport unavailable")
            return super().complete_json(prompt, system)

    with pytest.raises(PromptCustomizationValidationError) as caught:
        _run(
            monkeypatch,
            project,
            {
                "analysis_mode": "triple",
                "prompt_profiles": CUSTOM_SINGLE_V2,
            },
            TransportFailureFake(),
        )
    stats = caught.value.stats
    assert stats["attempted_calls"] == 2
    assert stats["prompt_customization_security_review_calls_attempted"] == 1
    assert stats["prompt_profile_conversion_calls_attempted"] == 1
    assert stats["prompt_profile_conversion_calls_completed"] == 0
    assert not (project / "codedoc").exists()


def test_routing_factor_bounds_and_structural_failures():
    response = _default_triple_response("route-test")
    response["factors"] = [
        "  same  ",
        "same",
        "X" * (pp.MAX_PROMPT_PROFILE_ROUTING_FACTOR_CHARS + 10),
        "",
        *[f"factor-{i}" for i in range(pp.MAX_PROMPT_PROFILE_ROUTING_FACTORS + 5)],
    ]
    cleaned = pp.validate_routing_response(
        response, expected_conversion_id="route-test"
    )["factors"]
    assert cleaned[0] == "same"
    assert len(cleaned[1]) == pp.MAX_PROMPT_PROFILE_ROUTING_FACTOR_CHARS
    assert len(cleaned) == pp.MAX_PROMPT_PROFILE_ROUTING_FACTORS

    for invalid in ("not-an-array", ["valid", 3]):
        response["factors"] = invalid
        with pytest.raises(PromptCustomizationValidationError, match="factors"):
            pp.validate_routing_response(
                response, expected_conversion_id="route-test"
            )


def test_routing_request_within_ceiling_for_max_single():
    # Build a maximal single (every field at the instruction limit) and prove the
    # rendered routing request stays under the ceiling.
    specs = []
    for fld in pp.iter_fields("single", "combined"):
        specs.append(ShapeFieldSpec(key=fld.path, type=fld.type, instruction="X" * pp.MAX_INSTRUCTION_CHARS))
    profile = AgentProfile(fields=tuple(specs), per_language={})
    cid = routing_conversion_id(profile)
    request = build_routing_request(profile, cid)
    assert len(request) <= pp.MAX_PROMPT_PROFILE_ROUTING_REQUEST_CHARS


def test_routing_request_over_ceiling_rejects_locally():
    profile = AgentProfile(
        fields=(
            ShapeFieldSpec(
                key="description",
                type="string",
                instruction="X" * pp.MAX_PROMPT_PROFILE_ROUTING_REQUEST_CHARS,
            ),
        ),
        per_language={},
    )
    with pytest.raises(PromptCustomizationValidationError, match="ceiling"):
        build_routing_request(profile, routing_conversion_id(profile))


# ---------------------------------------------------------------------------
# Local-default branch runs ordinary documentation (#15, #41)
# ---------------------------------------------------------------------------

class DocFake:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        assert "standards/safety review" not in prompt
        assert "single-to-triple" not in prompt
        return json.dumps({
            "description": "A file.", "role_in_system": "core",
            "functions": [{"name": "f", "description": "does f"}],
        })

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def test_local_default_runs_documentation_without_review_or_conversion(monkeypatch, project):
    raw = {"single": {"requested_shape": pp._default_requested_shape("single", "combined")}}
    stats = _run(monkeypatch, project, {
        "entry_file": "main.py", "analysis_mode": "triple", "prompt_profiles": raw,
    }, DocFake())
    assert stats["prompt_profile_conversion"] == "not-required"
    assert stats["prompt_customization_security_review"] == "not-required"
    assert stats["checked"] == 1
    assert (project / "codedoc" / "codedoc.json").exists()


# ---------------------------------------------------------------------------
# build_resolved_profile defensive guard (Addendum 11)
# ---------------------------------------------------------------------------

def test_build_resolved_profile_rejects_conversion_required():
    plan = classify_profile_action(
        validate_profile(CUSTOM_SINGLE_V2, active_mode="triple", known_languages=KL,
                         source="inline", source_path=None),
        "triple",
    )
    with pytest.raises(PromptCustomizationValidationError, match="conversion-required"):
        pp.build_resolved_profile(plan, "triple")


def test_no_files_return_keeps_all_paid_call_category_keys(monkeypatch, tmp_path):
    _no_provider(monkeypatch)
    stats = pipe.run_pipeline(tmp_path, {"dry_run": True})
    for key in (
        "documentation_calls_planned",
        "documentation_calls_attempted",
        "prompt_profile_conversion",
        "prompt_profile_conversion_calls_planned",
        "prompt_profile_conversion_calls_attempted",
        "prompt_profile_conversion_calls_completed",
        "prompt_customization_security_review_calls_attempted",
    ):
        assert key in stats


# ---------------------------------------------------------------------------
# Cycle-warning dedup (Workstream H, #33)
# ---------------------------------------------------------------------------

def test_cycle_warning_emitted_once(caplog):
    g = DependencyGraph()
    g.add_dependency("a.py", "b.py")
    g.add_dependency("b.py", "a.py")
    with caplog.at_level(logging.WARNING, logger="codedoc.core.graph"):
        order1 = g.topological_order()
        order2 = g.topological_order()
    warnings = [r for r in caplog.records if "Dependency cycle detected" in r.message]
    assert len(warnings) == 1
    # ordering unchanged across calls
    assert order1 == order2
    assert set(order1) == {"a.py", "b.py"}
