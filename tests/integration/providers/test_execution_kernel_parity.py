"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest

from codedoc.pipeline import run_pipeline
from tests.support.one_call_cases import _COMBINED_JSON
from tests.support.providers import _install_anthropic, _install_gemini, _install_openai

_PROVIDER_MATRIX = [
    ("openai", "gpt-test", _install_openai, "content"),
    ("anthropic", "claude-test", _install_anthropic, "text"),
    ("gemini", "gemini-test", _install_gemini, "text"),
]


def _base_config(provider_name, model, **overrides):
    config = {
        "llm_provider": provider_name,
        "model_name": model,
        "api_key": "test-key",
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
    }
    config.update(overrides)
    return config


def _sdk_calls(rec):
    return rec.get("create_calls") or rec.get("generate_calls") or []


def test_call_manifest_digest_and_planned_counts_are_provider_neutral(tmp_path, monkeypatch):
    """The call manifest is built on the provider-free planning path; its
    digest and every planned/reconciliation count must be byte-identical
    across OpenAI, Anthropic, and Gemini for the same file set, and the number
    of real SDK calls must equal the planned documentation-call count exactly
    (one logical call per planned entry, regardless of provider)."""
    digests = []
    for provider_name, model, installer, kw in _PROVIDER_MATRIX:
        project = tmp_path / provider_name
        project.mkdir()
        (project / "a.py").write_text("x = 1\n", encoding="utf-8")
        (project / "b.py").write_text("y = 2\n", encoding="utf-8")

        rec = {}
        installer(monkeypatch, rec, **{kw: _COMBINED_JSON})
        stats = run_pipeline(
            project,
            _base_config(
                provider_name,
                model,
                entry_file=None,
                auto_entry_candidates=[],
                documentation_scope="all",
            ),
        )

        assert stats["checked"] == 2
        assert stats["failed"] == 0
        assert stats["total_calls_planned"] == 2
        assert stats["attempted_logical_calls"] == 2
        assert stats["planned_calls_not_attempted"] == 0
        assert stats["additional_attempts"] == 0
        assert len(_sdk_calls(rec)) == 2
        digests.append(stats["call_manifest_digest"])

    assert digests[0] == digests[1] == digests[2]
    assert digests[0] != ""


@pytest.mark.parametrize("provider_name,model,installer,kw", _PROVIDER_MATRIX)
def test_malformed_response_failure_reconciles_identically_across_providers(
    tmp_path, monkeypatch, provider_name, model, installer, kw
):
    """A deterministic response-contract failure is one attempted logical call
    (not a planned-but-unattempted one) with no additional attempts, identically
    for every real provider adapter."""
    project = tmp_path / provider_name
    project.mkdir()
    (project / "main.py").write_text("x = 1\n", encoding="utf-8")

    rec = {}
    installer(monkeypatch, rec, **{kw: "this is not json"})
    stats = run_pipeline(
        project,
        _base_config(provider_name, model, entry_file="main.py", file_retry_attempts=0),
    )

    assert stats["checked"] == 0
    assert stats["failed"] == 1
    assert stats["total_calls_planned"] == 1
    assert stats["attempted_logical_calls"] == 1
    assert stats["planned_calls_not_attempted"] == 0
    assert stats["additional_attempts"] == 0
    assert len(_sdk_calls(rec)) == 1
