from __future__ import annotations

import json

import pytest

from codedoc.core.loader import DEFAULTS, load_config
from codedoc.utils.errors import ConfigError


def test_large_file_strategy_defaults_to_truncate(tmp_path) -> None:
    assert load_config(tmp_path)["large_file_strategy"] == "truncate"
    assert DEFAULTS["large_file_strategy"] == "truncate"


def test_large_file_strategy_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODEDOC_LARGE_FILE_STRATEGY", "split")
    monkeypatch.setenv("CODEDOC_DRY_RUN", "true")

    assert load_config(tmp_path)["large_file_strategy"] == "split"


def test_large_file_strategy_precedence_chain(tmp_path, monkeypatch) -> None:
    """defaults < codedoc.config.json < environment < explicit override.

    Each step is asserted while every lower-precedence source still supplies a
    different value, so a stage that silently stopped applying would fail here
    rather than be masked by an agreeing lower source.
    """
    monkeypatch.delenv("CODEDOC_LARGE_FILE_STRATEGY", raising=False)

    # 1. defaults
    assert load_config(tmp_path)["large_file_strategy"] == "truncate"

    # 2. codedoc.config.json beats defaults
    config_file = tmp_path / "codedoc.config.json"
    config_file.write_text(
        json.dumps({"large_file_strategy": "split", "dry_run": True}),
        encoding="utf-8",
    )
    assert load_config(tmp_path)["large_file_strategy"] == "split"

    # 3. environment beats the config file
    monkeypatch.setenv("CODEDOC_LARGE_FILE_STRATEGY", "truncate")
    assert load_config(tmp_path)["large_file_strategy"] == "truncate"

    # 4. an explicit programmatic override beats the environment
    assert (
        load_config(tmp_path, {"large_file_strategy": "split"})[
            "large_file_strategy"
        ]
        == "split"
    )


def test_large_file_strategy_rejects_case_or_whitespace(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"large_file_strategy": "Split"})
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"large_file_strategy": " split "})


def test_large_file_strategy_split_rejects_triple_analysis_mode(tmp_path) -> None:
    """D2: split is only valid in analysis_mode 'single'; combined with
    'triple' it must fail closed before scan, recovery, writer, or provider
    work, with a remedy naming both the offending values."""
    with pytest.raises(ConfigError, match="split.*triple"):
        load_config(
            tmp_path,
            {"large_file_strategy": "split", "analysis_mode": "triple"},
        )


def test_large_file_strategy_split_accepts_single_analysis_mode(tmp_path) -> None:
    config = load_config(
        tmp_path,
        {
            "large_file_strategy": "split",
            "analysis_mode": "single",
            "dry_run": True,
        },
    )
    assert config["large_file_strategy"] == "split"
    assert config["analysis_mode"] == "single"
    assert config["dry_run"] is True


def test_large_file_strategy_split_rejects_real_run(tmp_path) -> None:
    with pytest.raises(ConfigError, match="planning preview.*dry_run"):
        load_config(
            tmp_path,
            {
                "large_file_strategy": "split",
                "analysis_mode": "single",
                "dry_run": False,
            },
        )


def test_large_file_strategy_real_split_rejects_before_pipeline_side_effects(
    tmp_path, monkeypatch
) -> None:
    import codedoc.pipeline as pipeline_module
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_module,
        "scan_files",
        lambda *_a, **_k: pytest.fail("scanner ran before split release gate"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_recovery_records_if_compatible",
        lambda *_a, **_k: pytest.fail("recovery ran before split release gate"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_output_accessibility",
        lambda *_a, **_k: pytest.fail("output preflight ran before split release gate"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_provider",
        lambda *_a, **_k: pytest.fail("provider ran before split release gate"),
    )

    with pytest.raises(ConfigError, match="planning preview.*dry_run"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "large_file_strategy": "split",
                "dry_run": False,
                "output_dir": "docs",
            },
        )

    assert not (tmp_path / "docs").exists()


def test_large_file_strategy_split_rejects_triple_from_environment(
    tmp_path, monkeypatch
) -> None:
    """D2: the rejection applies regardless of which precedence layer
    supplied the conflicting values — not only an explicit programmatic
    override."""
    monkeypatch.setenv("CODEDOC_LARGE_FILE_STRATEGY", "split")
    monkeypatch.setenv("CODEDOC_ANALYSIS_MODE", "triple")

    with pytest.raises(ConfigError, match="split.*triple"):
        load_config(tmp_path)


def test_large_file_strategy_split_rejects_triple_from_cli(tmp_path, monkeypatch) -> None:
    """D2: the CLI surface rejects the combination before scanning, recovery,
    provider, or output work — exit code 2, matching every other ConfigError."""
    import codedoc.pipeline as pipeline_module
    from codedoc.cli.cli import run_cli

    def fail_if_scanned(*_args, **_kwargs):
        pytest.fail("scanner ran before the triple+split rejection")

    # Patch the aliases the lifecycle coordinator actually calls. Patching the
    # defining scanner module before importing codedoc.pipeline can make the
    # pipeline capture the test double permanently after monkeypatch teardown,
    # causing order-dependent failures in unrelated tests.
    monkeypatch.setattr(pipeline_module, "scan_files", fail_if_scanned)
    monkeypatch.setattr(
        pipeline_module,
        "create_provider",
        lambda _config: pytest.fail("provider created before the triple+split rejection"),
    )

    exit_code = run_cli(
        [
            str(tmp_path),
            "--analysis-mode",
            "triple",
            "--large-file-strategy",
            "split",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "codedoc").exists()


def test_large_file_strategy_split_rejects_triple_before_any_side_effect(
    tmp_path, monkeypatch
) -> None:
    """D2: rejection is the first normal-run guard — before the scanner,
    recovery inspection, SafeWriter initialization, or provider creation."""
    import codedoc.pipeline as pipeline_module
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline_module,
        "scan_files",
        lambda *_a, **_k: pytest.fail("scanner ran before the triple+split rejection"),
    )
    monkeypatch.setattr(
        "codedoc.core.safe_writer.SafeWriter.__init__",
        lambda *_a, **_k: pytest.fail("SafeWriter was constructed before the rejection"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_provider",
        lambda _config: pytest.fail("provider created before the triple+split rejection"),
    )

    with pytest.raises(ConfigError, match="split.*triple"):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "analysis_mode": "triple",
                "large_file_strategy": "split",
                "propagate_changes": False,
            },
        )

    assert not (tmp_path / "codedoc").exists()


def test_direct_planning_bypass_of_loader_produces_no_split_artifact(tmp_path) -> None:
    """D2 defense-in-depth: even if `codedoc.core.loader._validate()` were
    bypassed entirely (a config dict built and passed directly to planning,
    never going through `load_config`), an oversized file under
    `analysis_mode: 'triple'` never receives a division plan, a split
    identity, a split call manifest entry, split statistics, or any other
    split treatment — it is safely excluded rather than incorrectly split."""
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan
    from codedoc.pipeline import _split_division_stats

    src = tmp_path / "main.py"
    src.write_text("x" * 2000, encoding="utf-8")  # far above the 1000-char ceiling
    file_map = {
        "main.py": {
            "path": src,
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    config = {
        "propagate_changes": False,
        "max_files": 0,
        "analysis_mode": "triple",
        "large_file_strategy": "split",
        "max_content_chars": 1000,
    }

    plan, materials = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {}, [], config,
    )

    assert "main.py" not in plan.division_plan_rels
    assert "main.py" not in materials.division_plans
    assert "main.py" not in materials.reduction_trees
    assert not plan.division_blocked
    # The file is safely dropped from this run rather than incorrectly routed
    # through the split path under a rejected mode combination.
    assert "main.py" not in plan.agent_rels
    # No split call manifest entries: every call in the (empty-here) manifest
    # is one of the ordinary categories only.
    assert plan.unit_documentation_calls_planned == 0
    assert plan.file_reduction_calls_planned == 0
    # No split statistics — not even the bare `large_file_strategy` marker —
    # for the invalid mode combination, despite the requested string being
    # exactly "split".
    assert _split_division_stats(config, materials, plan) == {}
    # Split-partial recovery loading/writing requires SafeWriter — proven
    # never constructed for this exact invalid combination by
    # test_large_file_strategy_split_rejects_triple_before_any_side_effect.


@pytest.mark.parametrize(
    ("source_chars", "expects_split"),
        (
            (1999, False),
            (2000, False),
            (2001, True),
    ),
)
def test_split_routing_uses_inclusive_exact_character_threshold(
    tmp_path, monkeypatch, source_chars, expects_split
) -> None:
    """D1: `max_content_chars` is inclusive. The exact max-1/max/max+1
    boundary drives the complete dry-run call manifest."""
    prefix = "VALUE = "
    suffix = "\n"
    source = prefix + ("1" * (source_chars - len(prefix) - len(suffix))) + suffix
    assert len(source) == source_chars
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("exact-threshold dry run created a provider"),
    )

    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "dry_run": True,
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    assert stats["would_call_llm_for"] == 1
    assert stats["split_divided_files"] == int(expects_split)
    assert stats["split_ordinary_files"] == int(not expects_split)
    assert stats["file_documentation_calls_planned"] == int(not expects_split)
    assert stats["split_final_synthesis_calls_planned"] == int(expects_split)
    if expects_split:
        assert stats["unit_documentation_calls_planned"] > 0
    else:
        assert stats["unit_documentation_calls_planned"] == 0
    assert not (tmp_path / "docs").exists()
