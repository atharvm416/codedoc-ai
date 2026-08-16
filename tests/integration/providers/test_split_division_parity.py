"""Split mode (analysis_mode 'single' only — D2) must behave identically
across every real provider adapter: identical prompts, identical hierarchical
leaf/reduction/final call counts, identical retry/correction/terminal-failure
and node-keyed recovery behavior."""

from __future__ import annotations

import json
import re
import sys
import types
from types import SimpleNamespace

import pytest
from codedoc.core.file_division import build_division_plan, build_reduction_tree
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import UnrecoverableProviderError
from tests.support.one_call_cases import _COMBINED_JSON

_PROVIDERS = (
    ("openai", "openai", "gpt-test", None),
    (
        "openai-compatible",
        "openai",
        "compatible-test",
        "https://compatible.invalid/v1",
    ),
    ("anthropic", "anthropic", "claude-test", None),
    ("gemini", "gemini", "gemini-test", None),
)


def _prompts(provider_name: str, calls: dict) -> list[str]:
    if provider_name == "gemini":
        return [call["contents"] for call in calls["generate_calls"]]
    return [call["messages"][-1]["content"] for call in calls["create_calls"]]


def _normalized_prompts(prompts: list[str]) -> list[str]:
    return [
        re.sub(r"LLMError \[(?:OpenAI|Anthropic|Gemini)\]", "LLMError [provider]", prompt)
        for prompt in prompts
    ]


def _install_scripted_sdk(monkeypatch, calls: dict, provider_name: str, responder):
    if provider_name == "openai":
        module = types.ModuleType("openai")

        class Completions:
            def create(self, **kwargs):
                calls.setdefault("create_calls", []).append(kwargs)
                content = responder(kwargs["messages"][-1]["content"])
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content)
                        )
                    ]
                )

        class OpenAI:
            def __init__(self, api_key=None, base_url=None):
                calls["init"] = {"api_key": api_key, "base_url": base_url}
                self.chat = SimpleNamespace(completions=Completions())

        module.OpenAI = OpenAI
        monkeypatch.setitem(sys.modules, "openai", module)
        return

    if provider_name == "anthropic":
        module = types.ModuleType("anthropic")

        class Messages:
            def create(self, **kwargs):
                calls.setdefault("create_calls", []).append(kwargs)
                text = responder(kwargs["messages"][-1]["content"])
                return SimpleNamespace(content=[SimpleNamespace(text=text)])

        class Anthropic:
            def __init__(self, api_key=None):
                calls["init"] = {"api_key": api_key}
                self.messages = Messages()

        module.Anthropic = Anthropic
        monkeypatch.setitem(sys.modules, "anthropic", module)
        return

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            calls.setdefault("configs", []).append(kwargs)

    class Models:
        def generate_content(self, model, contents, config):
            call = {"model": model, "contents": contents, "config": config}
            calls.setdefault("generate_calls", []).append(call)
            return SimpleNamespace(text=responder(contents))

    class Client:
        def __init__(self, api_key=None):
            calls["init"] = {"api_key": api_key}
            self.models = Models()

    genai_types.GenerateContentConfig = GenerateContentConfig
    genai.Client = Client
    genai.types = genai_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)


def _valid_response(prompt: str) -> str:
    # Every split call uses one of three fixed internal shapes (leaf capsule,
    # reduction capsule) or the final resolved file-level shape — never the
    # old triple-mode dependency-agent shape, which split never issues.
    if "This is one bounded fragment of a larger" in prompt:
        return json.dumps(
            {
                "description": "A fragment.",
                "functions": [{"name": "f", "description": "does f"}],
            }
        )
    if "Refine one combined narrative from" in prompt:
        return json.dumps({"narrative": "A refined narrative."})
    return _COMBINED_JSON


def _config(
    *,
    provider_name: str,
    model: str,
    base_url: str | None,
    **overrides,
) -> dict:
    config = {
        "entry_file": "main.py",
        "llm_provider": provider_name,
        "model_name": model,
        "api_key": "test-key",
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "file_retry_attempts": 0,
        "propagate_changes": False,
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "output_dir": "docs",
    }
    if base_url is not None:
        config["api_base_url"] = base_url
    config.update(overrides)
    return config


def _expected_totals(source: str) -> tuple[int, int, int]:
    """(leaf, reduction, final) call counts for the shared fixture source."""
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=2000
    )
    tree = build_reduction_tree(plan, max_content_chars=2000)
    reduction = len(tree.unit_consolidation_nodes) + len(tree.general_nodes)
    return len(plan.chunks), reduction, 1


def test_real_provider_adapters_have_identical_split_prompts_outputs_and_counts(
    tmp_path,
    monkeypatch,
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    leaf_total, reduction_total, final_total = _expected_totals(source)
    planned = leaf_total + reduction_total + final_total
    outputs = []
    prompt_sets = []
    stat_sets = []

    for case_name, provider_name, model, base_url in _PROVIDERS:
        project = tmp_path / case_name
        project.mkdir()
        (project / "main.py").write_bytes(source.encode("utf-8"))
        calls: dict = {}
        _install_scripted_sdk(monkeypatch, calls, provider_name, _valid_response)

        stats = run_pipeline(
            project,
            _config(provider_name=provider_name, model=model, base_url=base_url),
            trust_api_base_url=base_url,
        )
        payload = json.loads(
            (project / "docs" / "codedoc.json").read_text(encoding="utf-8")
        )
        prompts = _prompts(provider_name, calls)

        assert len(prompts) == planned
        assert sum("Synthesize one final file-level" in prompt for prompt in prompts) == 1
        assert all("File: main.py" in prompt for prompt in prompts)
        assert stats["file_documentation_calls_planned"] == 0
        assert stats["unit_documentation_calls_planned"] == leaf_total
        assert stats["file_reduction_calls_planned"] == reduction_total
        assert stats["synthesis_calls_planned"] == final_total
        assert stats["total_calls_planned"] == planned
        assert stats["attempted_calls"] == len(prompts)
        assert stats["failed_calls"] == 0
        assert not (project / "docs" / "crash_recovery.json").exists()
        if case_name == "openai-compatible":
            assert calls["init"]["base_url"] == base_url

        outputs.append(payload["files"][0])
        prompt_sets.append(_normalized_prompts(prompts))
        stat_sets.append(
            (
                stats["call_manifest_digest"],
                stats["total_calls_planned"],
                stats["split_units"],
                stats["split_chunks"],
            )
        )

    assert outputs[0] == outputs[1] == outputs[2] == outputs[3]
    assert prompt_sets[0] == prompt_sets[1] == prompt_sets[2] == prompt_sets[3]
    assert stat_sets[0] == stat_sets[1] == stat_sets[2] == stat_sets[3]


@pytest.mark.parametrize("scenario", ["retry", "correction"])
def test_split_retry_and_correction_are_provider_adapter_neutral(
    tmp_path,
    monkeypatch,
    scenario,
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    leaf_total, reduction_total, final_total = _expected_totals(source)
    planned = leaf_total + reduction_total + final_total
    stat_sets = []
    prompt_sets = []

    for case_name, provider_name, model, base_url in _PROVIDERS:
        project = tmp_path / f"{case_name}-{scenario}"
        project.mkdir()
        (project / "main.py").write_bytes(source.encode("utf-8"))
        state = {"triggered": False}

        def scripted_response(prompt: str) -> str:
            is_correction = "Previous response (verbatim" in prompt
            is_synthesis = "Synthesize one final file-level" in prompt
            if not state["triggered"] and not is_correction and not is_synthesis:
                state["triggered"] = True
                if scenario == "retry":
                    raise RuntimeError("temporary provider outage")
                return "{}"
            return _valid_response(prompt)

        calls: dict = {}
        _install_scripted_sdk(
            monkeypatch, calls, provider_name, scripted_response
        )
        callback_calls = []
        stats = run_pipeline(
            project,
            _config(
                provider_name=provider_name,
                model=model,
                base_url=base_url,
                file_retry_attempts=1 if scenario == "retry" else 0,
                response_correction_enabled=scenario == "correction",
            ),
            confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
            trust_api_base_url=base_url,
        )
        prompts = _prompts(provider_name, calls)

        assert stats["checked"] == 1
        assert stats["failed"] == 0
        assert stats["total_calls_planned"] == planned
        assert stats["attempted_logical_calls"] == planned
        assert stats["planned_calls_not_attempted"] == 0
        if scenario == "retry":
            # The one call that hit a transient error is repeated once; every
            # other planned call attempts and succeeds exactly once.
            assert stats["attempted_calls"] == planned + 1
            assert stats["additional_attempts"] == 1
            assert stats["successful_calls"] == planned
            assert stats["failed_calls"] == 1
            assert stats["response_contract_failures"] == 0
        else:
            assert stats["attempted_calls"] == planned + 1
            assert stats["additional_attempts"] == 1
            assert stats["successful_calls"] == planned + 1
            assert stats["failed_calls"] == 0
            assert stats["response_contract_failures"] == 1
            assert stats["response_correction_calls_attempted"] == 1
            assert stats["response_correction_calls_succeeded"] == 1
        assert len(prompts) == stats["attempted_calls"]
        assert callback_calls == []
        assert not (project / "docs" / "crash_recovery.json").exists()
        assert json.loads(
            (project / "docs" / "codedoc.json").read_text(encoding="utf-8")
        )["files"][0]["description"] == "A documented module."
        stat_sets.append(
            (
                stats["call_manifest_digest"],
                stats["attempted_calls"],
                stats["successful_calls"],
                stats["failed_calls"],
                stats["additional_attempts"],
            )
        )
        prompt_sets.append(_normalized_prompts(prompts))

    assert stat_sets[0] == stat_sets[1] == stat_sets[2] == stat_sets[3]
    assert prompt_sets[0] == prompt_sets[1] == prompt_sets[2] == prompt_sets[3]


def test_terminal_split_failure_and_recovery_are_provider_adapter_neutral(
    tmp_path,
    monkeypatch,
) -> None:
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    leaf_total, reduction_total, final_total = _expected_totals(source)
    planned = leaf_total + reduction_total + final_total
    recovery_call_counts = []

    for case_name, provider_name, model, base_url in _PROVIDERS:
        project = tmp_path / f"{case_name}-terminal"
        project.mkdir()
        (project / "main.py").write_bytes(source.encode("utf-8"))
        state = {"leaf_calls": 0}

        def terminal_after_one_leaf(prompt: str) -> str:
            # All leaves execute (in order) before any reduction or final
            # call is ever issued, so failing on the second leaf call
            # guarantees exactly one completed, checkpointed leaf node.
            if "This is one bounded fragment of a larger" in prompt:
                state["leaf_calls"] += 1
                if state["leaf_calls"] == 2:
                    raise RuntimeError("Your credit balance is too low to continue")
            return _valid_response(prompt)

        failed_calls: dict = {}
        _install_scripted_sdk(
            monkeypatch,
            failed_calls,
            provider_name,
            terminal_after_one_leaf,
        )
        callback_calls = []
        with pytest.raises(UnrecoverableProviderError):
            run_pipeline(
                project,
                _config(provider_name=provider_name, model=model, base_url=base_url),
                confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
                trust_api_base_url=base_url,
            )

        recovery_path = project / "docs" / "crash_recovery.json"
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        partial = recovery["_codedoc"]["partial_files"]["main.py"]
        assert partial["schema_version"] == 4
        leaf_nodes = [
            node for node in partial["nodes"] if node["node_type"] == "leaf"
        ]
        assert len(leaf_nodes) == 1
        assert recovery["files"] == []
        assert callback_calls == []
        assert not (project / "docs" / "codedoc.json").exists()

        resumed_calls: dict = {}
        _install_scripted_sdk(
            monkeypatch, resumed_calls, provider_name, _valid_response
        )
        stats = run_pipeline(
            project,
            _config(provider_name=provider_name, model=model, base_url=base_url),
            confirm_risky=lambda warnings: callback_calls.append(warnings) or True,
            trust_api_base_url=base_url,
        )
        expected_remaining = planned - 1
        assert stats["resumed"] == 1
        assert stats["split_restored_complete_chunks"] == 1
        assert stats["total_calls_planned"] == expected_remaining
        assert stats["attempted_calls"] == expected_remaining
        assert len(_prompts(provider_name, resumed_calls)) == expected_remaining
        assert callback_calls == []
        assert not recovery_path.exists()
        assert json.loads(
            (project / "docs" / "codedoc.json").read_text(encoding="utf-8")
        )["files"][0]["description"] == "A documented module."
        recovery_call_counts.append(expected_remaining)

    assert len(set(recovery_call_counts)) == 1
