"""Focused regression tests for the codedoc-ai 0.9.2 release."""

from __future__ import annotations

import json
import logging
import os
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest


class FakeProvider:
    provider_name = "fake"

    def complete_json(self, prompt: str, system: str = "") -> str:
        if "key_concepts" in prompt:
            return json.dumps(
                {
                    "description": "Test.",
                    "role_in_system": "test",
                    "key_concepts": [],
                    "usage_example": "",
                }
            )
        if "dependencies_analysis" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        return json.dumps(
            {
                "description": "Test.",
                "role_in_system": "test",
                "functions": [],
                "classes": [],
                "exports": [],
            }
        )


def write_py(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def patch_provider(monkeypatch) -> None:
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: FakeProvider())


def make_graph(*paths: str, edges: tuple[tuple[str, str], ...] = ()):
    from codedoc.core.graph import DependencyGraph

    graph = DependencyGraph()
    for path in paths:
        graph.add_file(path)
    for source, dependency in edges:
        graph.add_dependency(source, dependency)
    return graph


def test_config_defaults_env_and_validation(tmp_path, monkeypatch):
    from codedoc.core.loader import load_config
    from codedoc.utils.errors import ConfigError

    monkeypatch.setenv("CODEDOC_DRY_RUN", "yes")
    monkeypatch.setenv("CODEDOC_MAX_FILES", "7")
    monkeypatch.setenv("CODEDOC_FORCE_FILES", "a.py; src/b.py ;")
    monkeypatch.setenv("CODEDOC_ALLOW_PARTIAL", "true")
    config = load_config(tmp_path)

    assert config["dry_run"] is True
    assert config["max_files"] == 7
    assert config["force_files"] == ["a.py", "src/b.py"]
    assert config["allow_partial"] is True

    for invalid in (-1, True, 1.5, "1.5", "+-5", "-+5", object()):
        with pytest.raises(ConfigError):
            load_config(tmp_path, {"max_files": invalid})
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"force_files": [""]})


def test_force_path_normalization_and_root_escape(tmp_path):
    from codedoc.core.planning import normalize_force_files
    from codedoc.utils.errors import ConfigError

    source = tmp_path / "src" / "a.py"
    write_py(source)
    normalized = normalize_force_files(
        ["src/a.py", str(source.resolve()), "src/./a.py", r"src\a.py"],
        tmp_path,
    )

    # The expected result depends on whether the backslash is a path separator
    # on the running platform.  ``normalize_force_files`` is correct per
    # platform; only the assertion must be platform-aware.
    backslash_is_separator = "\\" in (os.sep, os.altsep)
    if backslash_is_separator:
        # ``src\a.py`` collapses into the same entry as ``src/a.py``.
        assert normalized == ["src/a.py"]
    else:
        # The backslash is a legal filename character, so ``src\a.py`` is a
        # distinct relative path and is preserved as its own entry.
        assert normalized == ["src/a.py", r"src\a.py"]

    with pytest.raises(ConfigError):
        normalize_force_files(["../outside.py"], tmp_path)


def test_forcing_precedes_propagation_and_only_forced_file_bypasses_reuse(tmp_path):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.planning import build_pipeline_plan

    write_py(tmp_path / "dep.py", "same = 1\n")
    write_py(tmp_path / "main.py", "from dep import same\n")
    file_map = {
        rel: {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
        for rel in ("dep.py", "main.py")
    }
    graph = make_graph("dep.py", "main.py", edges=(("main.py", "dep.py"),))
    existing = {
        rel: {
            "path": rel,
            "hash": compute_file_hash(tmp_path / rel),
            "description": "cached",
        }
        for rel in file_map
    }

    plan, _ = build_pipeline_plan(
        file_map,
        graph,
        set(file_map),
        "main.py",
        existing,
        {},
        ["dep.py"],
        {"propagate_changes": True, "max_files": 0},
    )

    assert plan.forced_rels == frozenset({"dep.py"})
    assert plan.process_rels == frozenset({"dep.py", "main.py"})
    assert plan.agent_rels == frozenset({"dep.py"})
    assert plan.identical_reuse_rels == frozenset({"main.py"})


def test_missing_and_unselected_forced_paths_warn_once(tmp_path, caplog):
    from codedoc.core.planning import build_pipeline_plan

    for rel in ("main.py", "other.py"):
        write_py(tmp_path / rel)
    file_map = {
        rel: {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
        for rel in ("main.py", "other.py")
    }
    graph = make_graph("main.py", "other.py")

    with caplog.at_level(logging.WARNING, logger="codedoc.core.planning"):
        plan, _ = build_pipeline_plan(
            file_map,
            graph,
            {"main.py"},
            "main.py",
            {},
            {},
            ["missing.py", "other.py"],
            {"propagate_changes": True, "max_files": 0},
        )

    assert not plan.forced_rels
    assert caplog.text.count("missing.py") == 1
    assert caplog.text.count("other.py") == 1


def test_checkpoint_hash_reuse_and_live_backup_precedence(tmp_path, monkeypatch):
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    write_py(tmp_path / "main.py")
    output = tmp_path / "codedoc"
    output.mkdir()
    checkpoint = {
        "codedoc_checkpoint": True,
        "results": {
            "main.py": {
                "_checkpoint_hash": compute_file_hash(tmp_path / "main.py"),
                "description": "checkpoint",
                "language": "python",
            }
        },
    }
    (output / ".codedoc_progress.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )

    stats = run_pipeline(
        tmp_path,
        {"dry_run": True, "entry_file": "main.py", "propagate_changes": False},
    )
    assert stats["would_resume"] == 1
    assert stats["would_call_llm_for"] == 0

    live = {
        "_codedoc": {"status": "in_progress"},
        "files": [{"path": "other.py", "hash": "hash"}],
    }
    (output / "codedoc.json").write_text(json.dumps(live), encoding="utf-8")
    stats = run_pipeline(
        tmp_path,
        {"dry_run": True, "entry_file": "main.py", "propagate_changes": False},
    )
    assert stats["would_resume"] == 0
    assert stats["would_call_llm_for"] == 1


def test_dry_run_is_provider_free_and_filesystem_read_only(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    write_py(tmp_path / "main.py")
    legacy = tmp_path / "codedoc" / "codedoc_db.json"
    stale = tmp_path / "codedoc" / ".codedoc_build.json"
    legacy.parent.mkdir()
    legacy.write_text("legacy", encoding="utf-8")
    stale.write_text('{"_codedoc": {}, "files": []}', encoding="utf-8")
    before = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("dry-run created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.Orchestrator",
        lambda *args, **kwargs: pytest.fail("dry-run created an orchestrator"),
    )

    stats = run_pipeline(
        tmp_path,
        {"dry_run": True, "entry_file": "main.py", "output_dir": "new-output"},
    )
    after = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    assert stats["dry_run"] is True
    assert stats["estimated_calls"] == 3
    assert stats["estimate_is_lower_bound"] is True
    assert before == after
    assert not (tmp_path / "new-output").exists()


def test_dry_run_reports_conflict_and_cap_but_cli_exits_zero(tmp_path, monkeypatch, capsys):
    from codedoc.cli.cli import run_cli

    write_py(tmp_path / "main.py")
    output = tmp_path / "out"
    output.mkdir()
    (output / "report.json").write_text('{"foreign": true}', encoding="utf-8")

    code = run_cli(
        [
            str(tmp_path),
            "--entry",
            "main.py",
            "--output",
            "out/report.json",
            "--max-files",
            "0",
            "--dry-run",
        ]
    )
    assert code == 0
    assert "ownership conflict" in capsys.readouterr().out.lower()


def test_real_cap_fails_before_mutation_writer_or_provider(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    write_py(tmp_path / "a.py")
    write_py(tmp_path / "b.py")
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter",
        lambda *args, **kwargs: pytest.fail("cap failure initialized SafeWriter"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("cap failure created provider"),
    )

    with pytest.raises(ConfigError, match="exceeds"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": None,
                "auto_entry_candidates": [],
                "max_files": 1,
                "output_dir": "never-created",
            },
        )
    assert not (tmp_path / "never-created").exists()


def test_max_files_counts_only_agent_files(tmp_path):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.planning import build_pipeline_plan

    for rel, content in (("cached.py", "x = 1\n"), ("new.py", "y = 2\n")):
        write_py(tmp_path / rel, content)
    file_map = {
        rel: {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
        for rel in ("cached.py", "new.py")
    }
    graph = make_graph(*file_map)
    existing = {
        "old-name.py": {
            "path": "old-name.py",
            "hash": compute_file_hash(tmp_path / "cached.py"),
        }
    }
    plan, _ = build_pipeline_plan(
        file_map,
        graph,
        set(file_map),
        None,
        existing,
        {},
        [],
        {"propagate_changes": False, "max_files": 1},
    )
    assert len(plan.process_rels) == 2
    assert len(plan.identical_reuse_rels) == 1
    assert len(plan.agent_rels) == 1
    assert plan.max_files_exceeded is False


def test_usage_counts_success_failure_parse_error_and_is_thread_safe():
    from codedoc.agents.base_agent import BaseAgent
    from codedoc.core.usage import UsageAccumulator, estimate_tokens

    class Agent(BaseAgent):
        agent_name = "Agent"

        def run(self, file_path, content, imports, language):
            return {}

    class Provider:
        def __init__(self):
            self.responses = iter(["not-json", RuntimeError("boom"), "{}"])

        def complete_json(self, prompt, system=""):
            result = next(self.responses)
            if isinstance(result, Exception):
                raise result
            return result

    usage = UsageAccumulator()
    agent = Agent(Provider(), usage=usage)
    raw = agent._call_llm("user", system="system")
    with pytest.raises(Exception):
        agent._parse_json(raw, "x.py")
    with pytest.raises(Exception):
        agent._call_llm("user", system="system")
    assert agent._call_llm("user", system="system") == "{}"

    threads = [threading.Thread(target=usage.record_success, args=("abcd",)) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = usage.snapshot()
    assert snapshot["successful_calls"] == 22
    assert snapshot["failed_calls"] == 1
    assert snapshot["attempted_calls"] == 23
    assert snapshot["estimated_input_tokens"] == 3 * (
        estimate_tokens("system") + estimate_tokens("user")
    )


def test_usage_accounting_failure_does_not_change_provider_success():
    from codedoc.agents.base_agent import BaseAgent

    class Agent(BaseAgent):
        def run(self, file_path, content, imports, language):
            return {}

    class Provider:
        def complete_json(self, prompt, system=""):
            return "{}"

    class BrokenUsage:
        def record_input(self, *texts):
            raise RuntimeError("accounting failed")

        def record_success(self, output):
            raise RuntimeError("accounting failed")

    assert Agent(Provider(), usage=BrokenUsage())._call_llm("prompt") == "{}"


def test_real_run_reports_planned_and_actual_usage(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    write_py(tmp_path / "main.py")
    patch_provider(monkeypatch)
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "parallel_agents": False, "propagate_changes": False},
    )

    assert stats["planned_calls"] == 3
    assert stats["successful_calls"] == 3
    assert stats["failed_calls"] == 0
    assert stats["attempted_calls"] == 3
    assert stats["estimated_input_tokens"] > 0
    assert stats["estimated_output_tokens"] > 0


def test_provider_construction_errors_are_classified_as_setup_errors(monkeypatch):
    from codedoc.llm.factory import create_provider
    from codedoc.utils.errors import LLMError, ProviderInitError

    monkeypatch.setattr(
        "codedoc.llm.factory._make_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(LLMError("fake", "bad auth")),
    )
    with pytest.raises(ProviderInitError):
        create_provider(
            {
                "llm_mode": "api",
                "llm_provider": "openai",
                "model_name": "gpt-test",
                "api_key": "test",
            }
        )


def test_orchestrator_truncates_once_and_all_agents_receive_same_text(caplog):
    from codedoc.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator(FakeProvider(), parallel=False, max_content_chars=1000)
    seen: list[str] = []

    def structure(file_path, content, imports, language):
        seen.append(content)
        return {}

    def dependency(file_path, content, imports, language):
        seen.append(content)
        return {}

    def documentation(file_path, content, imports, language, structure, dependencies):
        seen.append(content)
        return {}

    orchestrator._structure_agent._safe_run = structure
    orchestrator._dependency_agent._safe_run = dependency
    orchestrator._doc_agent._safe_run_with_context = documentation

    with caplog.at_level(logging.WARNING, logger="codedoc.agents.orchestrator"):
        orchestrator.process(
            {"rel_path": "large.py", "language": "python", "extension": ".py"},
            "x" * 2000,
            [],
        )

    assert len(seen) == 3
    assert seen[0] == seen[1] == seen[2]
    assert len(seen[0]) == 1000
    assert seen[0].endswith("[truncated]")
    assert caplog.text.count("Content truncated") == 1


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (None, 0),
        ("failed", 1),
        ("partial", 0),
        ("config", 2),
        ("output", 1),
        ("fatal", 1),
        ("interrupt", 130),
    ],
)
def test_cli_exit_code_contract(tmp_path, monkeypatch, exception, expected):
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import ConfigError, OutputError

    def fake_pipeline(*args, **kwargs):
        if exception == "config":
            raise ConfigError("bad config")
        if exception == "output":
            raise OutputError("out.json", "write failed")
        if exception == "fatal":
            raise RuntimeError("unexpected")
        if exception == "interrupt":
            raise KeyboardInterrupt()
        return {
            "checked": 0,
            "failed": 1 if exception in {"failed", "partial"} else 0,
            "reused": 0,
            "output_dir": str(tmp_path / "out"),
            "output_files": [],
            "allow_partial": exception == "partial",
        }

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_pipeline)
    assert run_cli([str(tmp_path)]) == expected


def test_cli_missing_root_and_provider_init_return_two(tmp_path, monkeypatch):
    from codedoc.cli.cli import run_cli
    from codedoc.utils.errors import ProviderInitError

    assert run_cli([str(tmp_path / "missing")]) == 2
    monkeypatch.setattr(
        "codedoc.pipeline.run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(ProviderInitError("bad key")),
    )
    assert run_cli([str(tmp_path)]) == 2


def test_run_cli_returns_two_for_invalid_cli_input():
    from codedoc.cli.cli import run_cli

    assert run_cli(["--max-files", "not-an-int"]) == 2


def test_safe_mode_hidden_but_accepted_and_warns_once(tmp_path, monkeypatch, capsys):
    from codedoc.cli.cli import build_parser
    from codedoc.pipeline import run_pipeline

    help_text = build_parser().format_help()
    assert "--safe-mode" not in help_text
    write_py(tmp_path / "main.py")
    patch_provider(monkeypatch)
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "safe_mode": True, "propagate_changes": False},
    )
    assert capsys.readouterr().out.count("--safe-mode") == 1


def test_workflow_is_manual_safe_and_packaged_in_metadata():
    repo = Path(__file__).resolve().parent.parent
    workflow = (repo / "codedoc" / "templates" / "github-actions-codedoc.yml").read_text(
        encoding="utf-8"
    )
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (repo / "MANIFEST.in").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "\npush:" not in workflow and "\npull_request:" not in workflow
    assert "contents: read" in workflow
    assert workflow.index("--dry-run") < workflow.index("Generate documentation")
    assert workflow.count('            --max-files "$CODEDOC_MAX_FILES"') == 2
    assert 'args=(' in workflow and '"$CODEDOC_PROJECT_ROOT"' in workflow
    assert '"$CODEDOC_OUTPUT_PATH"' in workflow
    assert workflow.count("${{ inputs.") == 3
    run_blocks = workflow.split("run: |")[1:]
    assert run_blocks
    assert all("${{ inputs." not in block for block in run_blocks)
    assert "git push" not in workflow and "git commit" not in workflow
    assert "path: ${{ env.CODEDOC_OUTPUT_PATH }}" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert 'codedoc = ["templates/*.yml"]' in pyproject
    assert "recursive-include codedoc/templates *.yml" in manifest


def test_built_distributions_contain_workflow_template():
    """Inspect release artifacts when present; the release check builds them."""
    repo = Path(__file__).resolve().parent.parent
    dist = repo / "dist"
    wheels = sorted(dist.glob("codedoc_ai-0.9.2-*.whl"))
    sdists = sorted(dist.glob("codedoc_ai-0.9.2.tar.gz"))
    if not wheels or not sdists:
        pytest.skip("0.9.2 release artifacts have not been built yet")

    packaged = "codedoc/templates/github-actions-codedoc.yml"
    try:
        with zipfile.ZipFile(wheels[-1]) as archive:
            assert packaged in archive.namelist()
        with tarfile.open(sdists[-1]) as archive:
            assert any(name.endswith("/" + packaged) for name in archive.getnames())
    except PermissionError:
        pytest.skip("managed sandbox cannot read externally built artifacts")
