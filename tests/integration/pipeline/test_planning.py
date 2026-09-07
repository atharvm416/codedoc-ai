"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
import hashlib
import json
from tests.support.pipeline_scenarios import patch_provider
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_scenarios import _cache_identity
from tests.support.pipeline_scenarios import write_existing_json
from tests.support.pipeline_scenarios import write_existing_md
import logging
from pathlib import Path
import pytest
from codedoc.core.record_meta import ANALYSIS_REVISION, expected_ordinary_path_identity
from tests.support.pipeline_usage import write_py
from tests.support.pipeline_usage import make_graph
from dataclasses import asdict, fields
from codedoc.core.loader import load_config
from codedoc.core.execution_model import build_call_manifest
from codedoc.core.planning import PipelinePlan, build_pipeline_plan
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.selection_projects import _project
from tests.support.selection_projects import _graph_and_map
from tests.support.markdown_cases import _fake_provider as markdown_fake_provider
from tests.support.configuration_cases import _fake_provider as configuration_fake_provider
import codedoc.core.file_division as file_division
from codedoc.core.discovery import _resolve_entry_and_docs
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.pipeline import _final_entry_source
from tests.support.run_metadata_cases import _records
from tests.support.run_metadata_cases import _stats
from tests.support.feasibility_cases import _ReviewFake, _cross_file_profile


def test_retained_split_nodes_are_excluded_from_exact_unpaid_manifest(tmp_path):
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    source_path = tmp_path / "main.py"
    source_path.write_text(source, encoding="utf-8", newline="")
    config = load_config(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": "single",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
        },
    )
    division = file_division.build_division_plan(
        rel_path="main.py",
        language="python",
        content=source,
        source_budget_chars=2000,
    )
    # Build the recovered container's tree with the exact synthesis budget
    # production planning carries (the automatic 12,000 floor), so its
    # reduction-tree digest matches what build_pipeline_plan computes -- a
    # mismatched digest would (correctly) route this same-plan recovery into
    # cross-plan fresh-preserve carry instead of node validation.
    tree = file_division.build_reduction_tree(
        division,
        synthesis_manifest_chars=max(
            2000, file_division.MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
        ),
        language="python",
    )
    content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    provider_identity = file_division.provider_execution_identity(config)
    retained_chunks = division.chunks[:2]
    recovered = file_division.SplitTreeState(
        schema_version=file_division.SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=content_hash,
        division_plan_digest=division.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=tuple(
            file_division.tree_node_state(
                node_id=chunk.chunk_id,
                node_type="leaf",
                rel_path="main.py",
                content_hash=content_hash,
                division_plan_digest=division.plan_digest,
                input_digest=file_division.leaf_input_digest(
                    rel_path="main.py",
                    language="python",
                    chunk=chunk,
                    unit_indexes=division.unit_positions(chunk),
                    unit_count=len(division.units),
                ),
                execution_identity_digest=file_division.leaf_execution_identity(
                    rel_path="main.py",
                    content_hash=content_hash,
                    division_plan_digest=division.plan_digest,
                    provider_identity=provider_identity,
                    chunk=chunk,
                ),
                unit_id=None,
                child_ids=(),
                coverage_leaf_ids=(chunk.chunk_id,),
                result={
                    "description": f"retained {index}",
                    "chunk_id": chunk.chunk_id,
                    "unit_id": chunk.unit_id,
                },
            )
            for index, chunk in enumerate(retained_chunks)
        ),
    )
    graph = make_graph("main.py")
    plan, materials = build_pipeline_plan(
        {
            "main.py": {
                "path": source_path,
                "rel_path": "main.py",
                "language": "python",
                "extension": ".py",
            }
        },
        graph,
        {"main.py"},
        "main.py",
        {},
        [],
        config,
        recovered_partials={"main.py": recovered},
    )

    manifest = build_call_manifest(
        [],
        plan.agent_rels,
        "single",
        materials.division_plans,
        materials.reduction_trees,
        materials.tree_states,
    )
    planned = plan.with_call_manifest(
        manifest,
        0,
        materials.division_plans,
        materials.reduction_trees,
        materials.tree_states,
    )

    retained_ids = set(materials.tree_states["main.py"].by_id())
    assert retained_ids == {chunk.chunk_id for chunk in retained_chunks}
    assert all(call.owner not in retained_ids for call in manifest.calls)
    assert planned.total_calls_planned == (
        planned.unit_documentation_calls_planned
        + planned.file_reduction_calls_planned
        + planned.synthesis_calls_planned
    )
    assert planned.total_calls_planned == (
        len(division.chunks) + len(tree.all_nodes) - len(retained_ids)
    )


def test_split_dry_run_plans_chunks_and_synthesis_without_provider_or_output(
    tmp_path, monkeypatch
):
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    no_llm(monkeypatch)

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    assert stats["split_divided_files"] == 1
    assert stats["unit_documentation_calls_planned"] == stats["split_chunks"]
    assert stats["synthesis_calls_planned"] == 1
    assert stats["split_final_synthesis_calls_planned"] == 1
    # Hierarchical reduction: file_reduction_calls_planned covers every
    # unit-consolidation + general-reduction node below the final synthesis.
    assert stats["file_reduction_calls_planned"] == (
        stats["split_unit_consolidation_calls_planned"]
        + stats["split_general_reduction_calls_planned"]
    )
    assert stats["file_reduction_calls_planned"] > 0
    assert stats["total_calls_planned"] == (
        stats["unit_documentation_calls_planned"]
        + stats["file_reduction_calls_planned"]
        + stats["synthesis_calls_planned"]
    )
    assert not (tmp_path / "docs").exists()


def test_split_dry_run_final_estimate_uses_reserved_synopsis_bound(
    tmp_path,
    monkeypatch,
):
    source = "\n".join(f"value_{index} = {index}" for index in range(1200)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    no_llm(monkeypatch)
    config = {
        "dry_run": True,
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 12000,
        "propagate_changes": False,
        "output_dir": "docs",
    }
    baseline = run_pipeline(tmp_path, config)
    original = file_division.worst_case_final_synthesis_chars

    def expanded_bound(**kwargs):
        return original(**kwargs) + 4

    monkeypatch.setattr(
        file_division,
        "worst_case_final_synthesis_chars",
        expanded_bound,
    )
    expanded = run_pipeline(tmp_path, config)

    assert expanded["estimated_input_tokens"] == (
        baseline["estimated_input_tokens"] + 1
    )
    assert not (tmp_path / "docs").exists()


def test_planning_computes_the_effective_synthesis_budget_once_and_carries_it(tmp_path):
    """Section 5.7 / 8.0B / 9.1 item 7: ``build_pipeline_plan`` computes
    ``effective_split_manifest_chars = max(max_content_chars, 12000)`` a single
    time and carries the identical value into both the frozen
    ``AgentCallContext`` and the reduction tree. At ``B = 1000`` that is 12000;
    a source ceiling above 12000 is carried through unchanged. Leaf/source
    validation keeps using the raw source ceiling."""
    source = (
        "\n".join(f"def fn_{index}(): return {index}" for index in range(1500)) + "\n"
    )
    source_path = tmp_path / "main.py"
    source_path.write_text(source, encoding="utf-8", newline="")
    graph = make_graph("main.py")
    file_map = {
        "main.py": {
            "path": source_path,
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }

    def _materials_for(max_content_chars):
        config = load_config(
            tmp_path,
            {
                "entry_file": "main.py",
                "analysis_mode": "single",
                "large_file_strategy": "split",
                "max_content_chars": max_content_chars,
                "propagate_changes": False,
            },
        )
        _plan, materials = build_pipeline_plan(
            dict(file_map), graph, {"main.py"}, "main.py", {}, [], config
        )
        return materials

    # B = 1000: the source ceiling is below the floor, so the synthesis budget
    # is floored to 12000 once and both consumers carry that one value.
    materials = _materials_for(1000)
    request = materials.execution_requests["main.py"]
    tree = materials.reduction_trees["main.py"]
    division = materials.division_plans["main.py"]
    assert request.context.max_content_chars == 1000
    assert request.context.synthesis_manifest_chars == 12000
    assert tree.synthesis_manifest_chars == 12000
    assert (
        request.context.synthesis_manifest_chars == tree.synthesis_manifest_chars
    )
    # Leaf/source validation still uses the raw source ceiling.
    assert division.chunks
    assert all(len(chunk.payload) <= 1000 for chunk in division.chunks)
    # Reducer/final planning used the carried 12000: a complete final node that
    # covers every planned leaf, built without a starved-budget block.
    assert tree.final_node is not None
    assert tuple(sorted(tree.final_node.leaf_ids)) == tuple(
        sorted(chunk.chunk_id for chunk in division.chunks)
    )

    # A source ceiling above the floor is carried through unchanged.
    materials_hi = _materials_for(15000)
    request_hi = materials_hi.execution_requests["main.py"]
    tree_hi = materials_hi.reduction_trees["main.py"]
    assert request_hi.context.max_content_chars == 15000
    assert request_hi.context.synthesis_manifest_chars == 15000
    assert tree_hi.synthesis_manifest_chars == 15000
    assert all(
        len(chunk.payload) <= 15000
        for chunk in materials_hi.division_plans["main.py"].chunks
    )


def test_split_dry_run_final_estimate_uses_the_carried_synthesis_budget_not_the_source(
    tmp_path, monkeypatch
):
    """Section 5.7 / 9.1 item 7: in a provider-free split dry-run at a source
    ceiling of 1000, the real final worst-case token-estimation path is handed
    ``max_chars == 12000`` (the carried synthesis budget), never 1000."""
    source = (
        "\n".join(f"def fn_{index}(): return {index}" for index in range(400)) + "\n"
    )
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    no_llm(monkeypatch)

    captured_final_max_chars: list = []
    original = file_division.worst_case_final_synthesis_chars

    def _recording(**kwargs):
        captured_final_max_chars.append(kwargs.get("max_chars"))
        return original(**kwargs)

    monkeypatch.setattr(
        file_division, "worst_case_final_synthesis_chars", _recording
    )

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 1000,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    assert stats["split_divided_files"] == 1
    assert stats["estimated_input_tokens"] > 0
    # The final worst-case estimation path was reached, and every call carried
    # the automatic 12,000 synthesis budget -- never the 1,000 source ceiling.
    assert captured_final_max_chars
    assert set(captured_final_max_chars) == {12000}
    assert 1000 not in captured_final_max_chars
    assert not (tmp_path / "docs").exists()


def test_complex_split_plan_reports_a_provider_free_non_blocking_advisory(
    tmp_path, monkeypatch
):
    """D6a: a plan whose chunk count or reduction depth exceeds its frozen
    threshold carries a deterministic, provider-free advisory. It never
    creates a provider, blocks the run, or changes the resolved model/provider
    selection — it is purely an informational dry-run/CLI surface.

    A single oversized statement (rather than many small top-level
    statements) keeps the exact chunk count independent of whether the
    optional structural grammar extra is available: with a real parser,
    adjacent bare statements can merge into one shared "gap" unit, while a
    lone oversized statement is always its own unit under both lexical
    fallback and syntax-mode parsing."""
    source = "x = " + ("1" * 78001) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    no_llm(monkeypatch)
    config = {
        "dry_run": True,
        "entry_file": "main.py",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "llm_provider": "auto",
        "model_name": "",
        "propagate_changes": False,
        "output_dir": "docs",
    }

    stats = run_pipeline(tmp_path, config)

    assert stats["split_chunks"] > 24
    advisory = stats["split_complexity_advisory"]
    assert advisory is not None
    assert "higher-capability model" in advisory
    assert "max_content_chars" in advisory
    # Purely informational: the config this run actually used is unchanged.
    assert config["llm_provider"] == "auto"
    assert config["model_name"] == ""
    assert not (tmp_path / "docs").exists()


def test_simple_split_plan_reports_no_complexity_advisory(tmp_path, monkeypatch):
    # See test_complex_split_plan_reports_a_provider_free_non_blocking_advisory
    # for why this uses a single oversized statement rather than many small
    # top-level statements.
    source = "x = " + ("1" * 7001) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    no_llm(monkeypatch)

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "propagate_changes": False,
            "output_dir": "docs",
        },
    )

    assert stats["split_chunks"] <= 24
    assert stats["split_complexity_advisory"] is None


def test_pipeline_no_entry_no_docs_uses_auto_detection(tmp_path):
    """0.8.1: pipeline with no --entry and no existing docs must NOT raise 'No entry point
    specified'.  Instead _resolve_entry_and_docs() returns quietly and lets
    detect_entry_file() handle auto-detection at scan time.
    """
    from codedoc.pipeline import _resolve_entry_and_docs

    (tmp_path / "main.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    config = load_config(tmp_path, {"output_dir": "docs_output", "output_format": "json",
                                    "propagate_changes": False})
    config["entry_file"] = None  # simulate no --entry flag

    # Must NOT raise ConfigError — leaves entry_file unset for detect_entry_file()
    _resolve_entry_and_docs(tmp_path, config)
    assert config.get("entry_file") is None, (
        "_resolve_entry_and_docs() must leave entry_file unset so detect_entry_file() "
        "can handle auto-detection later in the pipeline"
    )

def test_pipeline_reads_entry_from_existing_json_metadata(tmp_path, monkeypatch):
    import json

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    output_dir = tmp_path / "docs_output"
    output_dir.mkdir()

    # Pre-write a public JSON that includes both the metadata block (for entry_file
    # discovery) and the file's documentation (so _build_documentation_records can
    # recover it without calling the LLM).
    file_hash = compute_file_hash(main)
    (output_dir / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "main.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "main.py",
                    "hash": file_hash,
                    "description": "Resumed from metadata.",
                    "language": "python",
                    "format": "py",
                    **_PRIOR_RUN_IDENTITY,
                }
            ],
        }),
        encoding="utf-8",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached metadata resume")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert "Resumed from metadata." in (
        output_dir / "codedoc.json"
    ).read_text(encoding="utf-8")

def test_select_files_raises_when_entry_not_in_file_map(tmp_path, monkeypatch):
    """A2: when an explicit entry exists on disk but is not picked up by the
    scanner (e.g. unsupported extension), the run must raise ConfigError instead
    of silently falling back to documenting all files."""
    import json as _json
    import pytest

    from codedoc.pipeline import run_pipeline

    # A .py file the scanner will pick up
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
    # Entry file physically exists but has an unsupported extension —
    # scanner will ignore it, so it won't appear in file_map.
    (tmp_path / "entry.txt").write_text("entrypoint\n", encoding="utf-8")

    def fake_provider(config):
        class P:
            provider_name = "fake"
            def complete_json(self, prompt, system=""): return _json.dumps({
                "description": "x", "role_in_system": "x",
                "functions": [], "classes": [], "exports": [],
                "key_concepts": [], "usage_example": "",
                "dependencies_analysis": {"internal": [], "external": []},
            })
            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)
        return P()

    monkeypatch.setattr("codedoc.pipeline.create_provider", fake_provider)

    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(
            tmp_path,
            {
                "output_dir": "docs_output",
                "output_format": "json",
                "entry_file": "entry.txt",   # exists but unsupported extension → not in file_map
                "propagate_changes": False,
                "max_parallel_files": 1,
                "parallel_agents": False,
            },
        )

    assert "scanned file set" in str(exc_info.value)

def test_A2_explicit_entry_with_zero_scanned_files_raises(tmp_path):
    """A2: an explicit entry with no supported files scanned must raise, not exit
    successfully having documented nothing."""
    import pytest
    from codedoc.pipeline import run_pipeline
    # Empty project (no supported files at all).
    with pytest.raises(ConfigError):
        run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                "parallel_agents": False})

def test_A2_entry_outside_project_root_raises(tmp_path):
    """A2: an explicit entry resolving outside the project root raises ConfigError
    (not a leaked ValueError)."""
    import pytest
    from codedoc.pipeline import run_pipeline

    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("y = 1\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(root, {"entry_file": str(outside), "output_format": "json",
                            "parallel_agents": False})
    assert "outside" in str(exc_info.value).lower()

def test_A8_no_entry_no_docs_proceeds_to_auto_detection(tmp_path):
    """A8 (0.8.1): no --entry, no existing docs → pipeline must NOT raise early.
    _resolve_entry_and_docs() returns quietly; detect_entry_file() handles
    auto-detection at scan time.  The old 'No entry point specified' error
    was removed in 0.8.1 so first runs work without an explicit --entry flag.
    """
    from codedoc.pipeline import _resolve_entry_and_docs

    (tmp_path / "some_file.py").write_text("x=1\n")
    config = load_config(tmp_path, {"output_dir": "docs_output", "output_format": "json"})
    config["entry_file"] = None

    # Must NOT raise — pipeline proceeds to detect_entry_file() / process-all-files fallback
    _resolve_entry_and_docs(tmp_path, config)
    assert config.get("entry_file") is None, (
        "entry_file must remain None so detect_entry_file() can attempt auto-detection"
    )

def test_D4_entry_auto_read_from_custom_json(tmp_path, monkeypatch):
    """D4: no --entry, --output docs/api.json exists with _codedoc.entry_file → entry resolved."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "docs" / "api.json", compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"output_dir": "docs/api.json",
                                     "propagate_changes": False})
    assert stats["checked"] == 0

def test_D5_entry_auto_read_from_custom_md(tmp_path, monkeypatch):
    """D5: no --entry, --output docs/api.md exists with codedoc-ai comment → entry resolved."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "docs" / "api.md", compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"output_dir": "docs/api.md",
                                     "propagate_changes": False})
    assert stats["checked"] == 0

def test_E1_entry_resolved_from_default_json(tmp_path, monkeypatch):
    """E1: no --entry, codedoc/codedoc.json exists → entry resolved from JSON metadata."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"output_format": "json", "propagate_changes": False})
    assert stats["checked"] == 0

def test_E2_entry_resolved_from_default_md(tmp_path, monkeypatch):
    """E2: no --entry, only codedoc/codedoc.md exists, --format md → entry from MD metadata."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "codedoc" / "codedoc.md",
                      compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"output_format": "md", "propagate_changes": False})
    assert stats["checked"] == 0

def test_E3_entry_not_found_no_docs_raises(tmp_path):
    """E3: no --entry, no existing docs → ConfigError."""
    from codedoc.pipeline import run_pipeline
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"output_format": "json"})
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass

def test_E4_entry_unsupported_extension_raises(tmp_path, monkeypatch):
    """E4 (A2): --entry entry.txt (exists, unsupported ext, so not scanned) must
    raise ConfigError, not silently document the whole repo."""
    import pytest

    patch_provider(monkeypatch)
    (tmp_path / "other.py").write_text("x=1\n")
    (tmp_path / "entry.txt").write_text("entrypoint\n")
    from codedoc.pipeline import run_pipeline
    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(tmp_path, {"entry_file": "entry.txt",
                                "output_format": "json",
                                "propagate_changes": False,
                                "parallel_agents": False})
    assert "scanned file set" in str(exc_info.value)

def test_E5_entry_missing_file_raises(tmp_path, monkeypatch):
    """E5 (A2): --entry nonexistent.py must raise ConfigError instead of falling
    back to documenting all files."""
    import pytest

    patch_provider(monkeypatch)
    (tmp_path / "other.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(tmp_path, {"entry_file": "nonexistent.py",
                                "output_format": "json",
                                "propagate_changes": False,
                                "parallel_agents": False})
    assert "was not found" in str(exc_info.value)

def test_E6_no_entry_still_documents_all_files(tmp_path, monkeypatch):
    """E6 (A2 guard): with NO entry specified, auto-detection finding nothing must
    still fall back to documenting all files (the legitimate path)."""
    patch_provider(monkeypatch)
    (tmp_path / "other.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"output_format": "json",
                                    "propagate_changes": False,
                                    "parallel_agents": False})
    assert stats["checked"] >= 1

def test_I1_propagate_changes_true_reimports_updated(tmp_path, monkeypatch):
    """I1: propagate_changes=True (default): A changed → B (imports A) is pulled into
    process_rels via propagation, so it is NOT skipped.
    B's own content didn't change so it is reused from docs_by_hash rather than
    re-sent to the LLM — but the key point is skipped=0 (neither file bypassed
    the process set)."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    (tmp_path / "a.py").write_text("def helper(): pass\n")
    (tmp_path / "b.py").write_text("import a\ndef main(): a.helper()\n")

    b_hash = compute_file_hash(tmp_path / "b.py")

    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "b.py", "schema_version": "1.4"},
        "files": [
            {"path": "a.py", "hash": "old_a_hash", "language": "python",
             "description": "Old A.", **_cache_identity("a.py")},
            {"path": "b.py", "hash": b_hash, "language": "python",
             "description": "Old B.", **_cache_identity("b.py")},
        ],
    }), encoding="utf-8")

    patch_provider(monkeypatch, "Updated.")

    stats = run_pipeline(tmp_path, {"entry_file": "b.py", "output_format": "json",
                                     "propagate_changes": True, "parallel_agents": False})
    # a.py was changed → LLM re-processed (checked=1)
    # b.py was propagated into process_rels, content hash unchanged → reused (not skipped)
    assert stats["checked"] == 1           # only a.py sent to LLM
    assert stats["skipped"] == 0           # b.py was NOT skipped (it was in process_rels)
    assert stats.get("reused", 0) == 1     # b.py reused from its matching hash in docs_by_hash

def test_I2_propagate_changes_false_only_changed(tmp_path, monkeypatch):
    """I2: propagate_changes=False: only A re-processed, B (imports A) skipped."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    (tmp_path / "a.py").write_text("def helper(): pass\n")
    (tmp_path / "b.py").write_text("import a\ndef main(): a.helper()\n")

    b_hash = compute_file_hash(tmp_path / "b.py")
    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "b.py", "schema_version": "1.4"},
        "files": [
            {"path": "a.py", "hash": "old_a_hash", "language": "python", "description": "Old A.", **_cache_identity("a.py")},
            {"path": "b.py", "hash": b_hash, "language": "python", "description": "Old B.", **_cache_identity("b.py")},
        ],
    }), encoding="utf-8")

    patch_provider(monkeypatch, "Updated A only.")
    stats = run_pipeline(tmp_path, {"entry_file": "b.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats["checked"] == 1  # only a.py

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
                "language": "python",
                "_analysis_revision": ANALYSIS_REVISION,
            "_analysis_mode": "single",
            "_ordinary_path_identity": expected_ordinary_path_identity(rel),
        }
        for rel in file_map
    }

    plan, _ = build_pipeline_plan(
        file_map,
        graph,
        set(file_map),
        "main.py",
        existing,
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
            ["missing.py", "other.py"],
            {"propagate_changes": True, "max_files": 0},
        )

    assert not plan.forced_rels
    assert caplog.text.count("missing.py") == 1
    assert caplog.text.count("other.py") == 1


def test_source_snapshot_race_rebuilds_planning_once(tmp_path, monkeypatch):
    source = tmp_path / "main.py"
    write_py(source, "old = 1\n")
    file_map = {
        "main.py": {
            "path": source,
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }
    graph = make_graph("main.py")
    routing_hashes = iter(("route-v1", "route-v2"))
    snapshots = iter(
        (
            ("snapshot-v2", "intermediate = 2\n"),
            ("route-v2", "stable = 3\n"),
        )
    )
    monkeypatch.setattr(
        "codedoc.core.planning.compute_file_hash",
        lambda _path: next(routing_hashes),
    )
    monkeypatch.setattr(
        "codedoc.core.planning.read_source_snapshot",
        lambda _path: next(snapshots),
    )

    plan, materials = build_pipeline_plan(
        file_map,
        graph,
        {"main.py"},
        "main.py",
        {},
        [],
        {"propagate_changes": True, "max_files": 0},
    )

    assert plan.agent_rels == frozenset({"main.py"})
    request = materials.execution_requests["main.py"]
    assert request.content_hash == "route-v2"
    assert request.content == "stable = 3\n"


def test_source_snapshot_second_race_fails_deterministically(tmp_path, monkeypatch):
    source = tmp_path / "main.py"
    write_py(source, "old = 1\n")
    file_map = {
        "main.py": {
            "path": source,
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }
    graph = make_graph("main.py")
    routing_hashes = iter(("route-v1", "route-v3"))
    snapshots = iter(
        (
            ("snapshot-v2", "intermediate = 2\n"),
            ("snapshot-v4", "still_changing = 4\n"),
        )
    )
    monkeypatch.setattr(
        "codedoc.core.planning.compute_file_hash",
        lambda _path: next(routing_hashes),
    )
    monkeypatch.setattr(
        "codedoc.core.planning.read_source_snapshot",
        lambda _path: next(snapshots),
    )

    with pytest.raises(ConfigError, match="changed again on the retry"):
        build_pipeline_plan(
            file_map,
            graph,
            {"main.py"},
            "main.py",
            {},
            [],
            {"propagate_changes": True, "max_files": 0},
        )

def test_stale_source_rebuilds_dependency_routing_not_only_snapshots(
    tmp_path, monkeypatch
):
    """The dependency graph is parsed before routing hashes are taken, so a
    detected stale revision means routing itself may have observed it. The
    pipeline — which owns scanning and graph construction — must rebuild scan,
    parse, graph, and entry selection once, not re-read snapshots against the
    graph that already saw the stale revision."""
    import codedoc.core.planning as planning_mod
    import codedoc.pipeline as pipeline_mod

    main = tmp_path / "main.py"
    # Pre-edit revision: main.py imports nothing, so the first graph has no edge.
    write_py(main, "VALUE = 1\n")
    write_py(tmp_path / "helper.py", "HELPER = 2\n")

    graphs_built: list[dict[str, list[str]]] = []
    real_build_graph = pipeline_mod._build_graph

    def _recording_build_graph(all_files, root, error_reporter):
        graph, file_map, unresolved = real_build_graph(all_files, root, error_reporter)
        graphs_built.append(
            {rel: sorted(graph.dependencies_of(rel)) for rel in sorted(file_map)}
        )
        return graph, file_map, unresolved

    monkeypatch.setattr(pipeline_mod, "_build_graph", _recording_build_graph)

    # Introduce the concurrent edit inside planning, after main.py's routing
    # hash has been taken but before its canonical snapshot is read — the exact
    # window the stale-revision check exists to detect.
    real_hash = planning_mod.compute_file_hash
    edited = {"done": False}

    def _hash_then_edit(path):
        digest = real_hash(path)
        if not edited["done"] and Path(path).name == "main.py":
            edited["done"] = True
            write_py(main, "import helper\nVALUE = 1\n")
        return digest

    monkeypatch.setattr(planning_mod, "compute_file_hash", _hash_then_edit)

    class _Fake:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            return json.dumps({"description": "d"})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr(pipeline_mod, "create_provider", lambda _cfg: _Fake())
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": None,
            "auto_entry_candidates": [],
            "documentation_scope": "all",
            "max_parallel_files": 1,
        },
    )

    # Exactly one rebuild: the graph was constructed twice, never more.
    assert len(graphs_built) == 2
    # The first graph observed the stale, import-free revision; the rebuilt one
    # carries the edge the concurrent edit introduced.
    assert graphs_built[0]["main.py"] == []
    assert graphs_built[1]["main.py"] == ["helper.py"]
    assert stats["checked"] == 2

def test_second_stale_revision_fails_before_every_provider_side_effect(
    tmp_path, monkeypatch
):
    """A source that changes again on the rebuild is a concurrent-modification
    error, reported before usage accounting, provider creation, any
    confirmation callback, or writer initialization."""
    import codedoc.core.planning as planning_mod

    main = tmp_path / "main.py"
    write_py(main, "VALUE = 1\n")

    real_hash = planning_mod.compute_file_hash
    edits = {"count": 0}

    def _hash_then_edit(path):
        digest = real_hash(path)
        if Path(path).name == "main.py":
            edits["count"] += 1
            write_py(main, f"VALUE = {edits['count'] + 1}\n")
        return digest

    monkeypatch.setattr(planning_mod, "compute_file_hash", _hash_then_edit)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: pytest.fail("concurrent-modification error created a provider"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.UsageAccumulator",
        lambda *a, **k: pytest.fail("concurrent-modification error created usage"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter",
        lambda *a, **k: pytest.fail("concurrent-modification error initialized a writer"),
    )
    confirmations = []

    with pytest.raises(ConfigError, match="changed again on the retry"):
        run_pipeline(
            tmp_path,
            {"entry_file": "main.py", "output_dir": "never-created"},
            confirm_risky=lambda warnings: confirmations.append(warnings) or True,
        )

    # Two detections: the initial pass and the post-rebuild pass.
    assert edits["count"] == 2
    assert confirmations == []
    assert not (tmp_path / "never-created").exists()

def test_real_cap_fails_before_mutation_writer_or_provider(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

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

def test_pipeline_plan_canonical_documented_field_with_selected_alias(tmp_path):
    # 0.10.0: documented_rels is now the canonical dataclass field; selected_rels
    # is retained as a read-only delegating compatibility property.
    _project(tmp_path)
    graph, file_map = _graph_and_map(tmp_path)
    plan, _ = build_pipeline_plan(
        file_map=file_map,
        graph=graph,
        selected_rels={"main.py", "helper.py"},
        entry_rel="main.py",
        existing_docs={},
        forced_paths=[],
        config={"propagate_changes": True, "max_files": 0},
    )
    # Canonical field direction.
    assert "documented_rels" in {field.name for field in fields(plan)}
    assert "selected_rels" not in {field.name for field in fields(plan)}
    assert asdict(plan)["documented_rels"] == plan.documented_rels
    match plan:
        case PipelinePlan(documented_rels=documented):
            assert documented == plan.documented_rels
    # Retained compatibility alias.
    assert plan.selected_rels == plan.documented_rels

def test_no_supported_files_real_stats_keep_scope_and_compatibility_keys(tmp_path):
    stats = run_pipeline(tmp_path, {})
    assert stats["entry_excluded"] == 0
    assert stats["documentation_scope"] == "entry"
    assert stats["entry_reachable"] == 0
    assert stats["entry_disconnected"] == 0
    assert stats["disconnected_paid_files"] == 0
    assert stats["disconnected_planned_calls"] == 0

def test_A10_first_run_no_entry_no_output_does_not_raise(tmp_path):
    """A10: First run without --entry and no existing output must not raise ConfigError."""
    from codedoc.pipeline import _resolve_entry_and_docs

    # No output files exist, no entry_file in config
    config = load_config(tmp_path, {"output_format": "json"})
    config.pop("entry_file", None)
    config["entry_file"] = None

    # Must not raise
    _resolve_entry_and_docs(tmp_path, config)
    # entry_file should still be None (auto-detection left to detect_entry_file)
    assert config.get("entry_file") is None

def test_A10_first_run_pipeline_without_entry_uses_auto_detection(tmp_path, monkeypatch):
    """A10b: Pipeline first run without --entry reaches detect_entry_file() and processes all files."""
    # Write a file with a common auto-detect name
    (tmp_path / "main.py").write_text("x=1\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: markdown_fake_provider())
    from codedoc.pipeline import run_pipeline

    # No entry_file provided — should auto-detect main.py
    stats = run_pipeline(tmp_path, {
        "parallel_agents": False,
        "propagate_changes": False,
        # no entry_file
    })
    assert stats["checked"] >= 1 or stats.get("reused", 0) >= 1, (
        "Pipeline must process files even without an explicit --entry"
    )

def test_C9_custom_auto_entry_detected(tmp_path, monkeypatch):
    """C9: A custom auto_entry_candidates_add entry is found and used."""
    (tmp_path / "app_start.py").write_text("pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: configuration_fake_provider())
    from codedoc.pipeline import run_pipeline

    # No --entry; should auto-detect "app_start.py" from custom candidates
    run_pipeline(tmp_path, {
        "auto_entry_candidates_add": ["app_start.py"],
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result.get("last_run", {}).get("entry_file") == "app_start.py", (
        "Custom auto-entry candidate must be detected and stored as entry_file"
    )

def test_C10_default_auto_entry_main_py_detected(tmp_path, monkeypatch):
    """C10: Default auto-entry 'main.py' is found and used on first run."""
    (tmp_path / "main.py").write_text("def main(): pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: configuration_fake_provider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "parallel_agents": False,
        "propagate_changes": False,
        # No entry_file — auto-detection should find main.py
    })

    assert stats["checked"] >= 1 or stats.get("reused", 0) >= 1

def test_C11_no_entry_no_auto_candidate_processes_all(tmp_path, monkeypatch):
    """C11: Without --entry and no auto-entry match, all supported files are processed."""
    (tmp_path / "helper.py").write_text("x = 1\n")
    (tmp_path / "util.py").write_text("y = 2\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: configuration_fake_provider())
    from codedoc.pipeline import run_pipeline

    # Remove all default auto_entry_candidates so none match
    stats = run_pipeline(tmp_path, {
        "auto_entry_candidates": [],  # no candidates at all
        "parallel_agents": False,
        "propagate_changes": False,
    })

    # All .py files should be processed (entry_file = None → all files)
    total = stats["checked"] + stats.get("reused", 0)
    assert total >= 2, (
        f"All files must be processed when no entry is detected, got stats={stats}"
    )

def test_entry_source_none_exactly_when_no_project_entry():
    stats = {
        "checked": 2,
        "failed": 0,
        "skipped": 0,
        "reused": 0,
        "resumed": 0,
        "files_scanned": 2,
        "files_selected": 2,
    }
    view = build_project_view(_records(), stats, entry_file=None)
    assert view["last_run"]["entry_file"] is None
    assert view["last_run"]["entry_source"] == "none"
    assert view["last_run"]["documentation_scope"] == "all"

def test_resolve_entry_source_recovered_from_prior_document(tmp_path):
    # A prior completed codedoc.json carrying an entry, and no --entry supplied,
    # resolves as "recovered" and writes the entry back into config.
    (tmp_path / "codedoc.json").write_text(
        json_from_view(build_project_view(_records(), _stats(), entry_file="main.py")),
        encoding="utf-8",
    )
    config = {"output_dir": str(tmp_path), "output_format": "json"}

    source = _resolve_entry_and_docs(tmp_path, config)

    assert source == "recovered"
    assert config["entry_file"] == "main.py"
    # The pipeline keeps "recovered" verbatim regardless of later auto-detection.
    assert _final_entry_source(source, "main.py") == "recovered"

def test_final_entry_source_maps_pending_to_auto_detected_or_none():
    assert _final_entry_source("pending", "main.py") == "auto-detected"
    assert _final_entry_source("pending", None) == "none"
    assert _final_entry_source("explicit", None) == "explicit"

# ===========================================================================
# Section 5.8 correction round: the real-run ``plan_reporter`` receives ONE
# deeply-immutable provider-free preflight snapshot, invoked exactly once and
# strictly before division-blocked / cap enforcement, provider construction,
# prompt-profile review, recovery/writer mutation, output probing, or any
# documentation call. Without a reporter the pipeline logs the COMPLETE bounded
# value-safe aggregate contract at INFO. Zero-work planning uses the identical
# mechanism.
# ===========================================================================


def _s8_large_source(defs: int = 400) -> str:
    return "\n".join(f"def fn_{i}(): return {i}" for i in range(defs)) + "\n"


def _s8_norm(value):
    """Recursively convert a (possibly frozen) snapshot into plain
    dict/list/scalar so two snapshots can be compared field-for-field."""
    from types import MappingProxyType

    if isinstance(value, (dict, MappingProxyType)):
        return {k: _s8_norm(v) for k, v in dict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_s8_norm(v) for v in value]
    return value


def _s8_event_spy(monkeypatch, events):
    def _spy(_config):
        events.append("provider")
        return markdown_fake_provider()

    monkeypatch.setattr("codedoc.pipeline.create_provider", _spy)


def _s8_no_side_effects(monkeypatch):
    """Hard sentinels: any provider / writer / output probe is an immediate
    failure."""
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("provider constructed before/instead of the report"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter",
        lambda *a, **k: pytest.fail("SafeWriter constructed"),
    )
    monkeypatch.setattr(
        "codedoc.pipeline.preflight_output_accessibility",
        lambda *a, **k: pytest.fail("output accessibility probed"),
    )


# --- reporter immutability -------------------------------------------------

def test_s8_reporter_snapshot_rejects_top_level_mutation(tmp_path, monkeypatch):
    import operator
    from collections.abc import Mapping

    write_py(tmp_path / "main.py")
    _s8_event_spy(monkeypatch, [])
    seen: dict = {}

    def _reporter(snap):
        seen["mapping"] = isinstance(snap, Mapping)
        for attempt in (
            lambda: snap.__setitem__("total_calls_planned", 999),
            lambda: snap.__setitem__("injected_key", 1),
            lambda: snap.__delitem__("call_manifest_digest"),
            lambda: snap.update({"x": 1}),
            lambda: operator.setitem(snap, "z", 1),
        ):
            try:
                attempt()
                seen.setdefault("leaks", []).append("MUTATED")
            except (TypeError, AttributeError):
                pass

    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "parallel_agents": False, "propagate_changes": False},
        plan_reporter=_reporter,
    )
    assert seen["mapping"] is True
    assert "leaks" not in seen


def test_s8_reporter_snapshot_rejects_nested_mutation(tmp_path, monkeypatch):
    from collections.abc import Mapping

    (tmp_path / "big.py").write_text(_s8_large_source(), encoding="utf-8", newline="")
    _s8_event_spy(monkeypatch, [])
    outcome: dict = {"leaks": [], "truncate_items_seen": 0, "deep_seen": 0}

    def _visit(node):
        if isinstance(node, Mapping):
            assert not isinstance(node, dict), type(node)
            try:
                node["__hack__"] = 1  # type: ignore[index]
                outcome["leaks"].append("mapping-set")
            except TypeError:
                pass
            for v in node.values():
                _visit(v)
        elif isinstance(node, tuple):
            for v in node:
                _visit(v)
        elif isinstance(node, list):
            outcome["leaks"].append("mutable-list")

    def _reporter(snap):
        for key in (
            "split_plan_details", "truncate_plan_details", "split_blocked_details",
            "scanner_size_skip_details", "scanner_admission_skip_details",
            "ownership_conflicts", "output_files",
        ):
            coll = snap[key]
            assert isinstance(coll, tuple), (key, type(coll))
            try:
                coll.append({"x": 1})  # type: ignore[attr-defined]
                outcome["leaks"].append(f"{key}:append")
            except AttributeError:
                pass
        outcome["truncate_items_seen"] = len(snap["truncate_plan_details"])
        for item in snap["truncate_plan_details"]:
            try:
                item["path"] = "hacked"
                outcome["leaks"].append("truncate-item")
            except TypeError:
                pass
        _visit(snap)

    run_pipeline(
        tmp_path,
        {
            "entry_file": "big.py", "large_file_strategy": "truncate",
            "max_content_chars": 1000, "parallel_agents": False,
            "propagate_changes": False,
        },
        plan_reporter=_reporter,
    )
    assert outcome["leaks"] == []
    assert outcome["truncate_items_seen"] >= 1  # the run actually had a truncate detail


def test_s8_reporter_cannot_corrupt_the_dry_run_return(tmp_path, monkeypatch):
    write_py(tmp_path / "main.py")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("dry-run constructed a provider"),
    )

    def _reporter(snap):
        for key, val in (("total_calls_planned", 999),
                         ("call_manifest_digest", "deadbeef"),
                         ("injected", True)):
            try:
                snap[key] = val
            except (TypeError, AttributeError):
                pass

    dry = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "dry_run": True, "propagate_changes": False},
        plan_reporter=_reporter,
    )
    assert dry["total_calls_planned"] == 1
    assert dry["call_manifest_digest"] and dry["call_manifest_digest"] != "deadbeef"
    assert "injected" not in dry
    plain = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "dry_run": True, "propagate_changes": False},
    )
    assert plain["call_manifest_digest"] == dry["call_manifest_digest"]


@pytest.mark.parametrize("zero_work", [False, True])
def test_s8_reporter_exception_aborts_before_provider_and_mutation(
    tmp_path, monkeypatch, zero_work
):
    if not zero_work:
        write_py(tmp_path / "main.py")
    _s8_no_side_effects(monkeypatch)

    class _Boom(RuntimeError):
        pass

    cfg = {"parallel_agents": False, "propagate_changes": False, "output_dir": "out"}
    if zero_work:
        cfg.update(entry_file=None, auto_entry_candidates=[])
    else:
        cfg["entry_file"] = "main.py"
    with pytest.raises(_Boom):
        run_pipeline(tmp_path, cfg, plan_reporter=lambda _s: (_ for _ in ()).throw(_Boom()))
    assert not (tmp_path / "out").exists()


# --- INFO fallback contract ---------------------------------------------------

_S8_INFO_REQUIRED_FIELDS = (
    "initial_provider_calls_planned", "prompt_review_calls_planned",
    "initial_documentation_calls_planned", "correction_calls_possible_max",
    "provider_calls_max_before_retries", "file_retry_attempts",
    "retries_included_in_ceiling", "max_planned_calls_applies_to",
    "total_calls_planned", "max_planned_calls", "max_planned_calls_exceeded",
    "call_manifest_digest",
    "large_file_strategy_resolved", "large_file_source_ceiling_chars",
    "large_files_over_source_ceiling", "large_files_routed_split",
    "large_files_routed_truncate", "truncate_retained_source_chars",
    "truncate_omitted_source_chars", "split_internal_manifest_budget_chars",
    "split_oversized_units", "split_continuation_chunks",
    "split_crlf_atomicity_extra_chunks", "split_boundary_cuts_syntax",
    "split_boundary_cuts_physical_line", "split_boundary_cuts_balanced_codepoint",
    "split_boundary_constrained_small_chunks", "split_closures_source_ceiling",
    "split_closures_metadata_ceiling", "split_closures_source_and_metadata_ceiling",
    "split_closures_oversized_unit_isolation", "split_closures_continuation",
    "split_closures_end_of_file", "split_metadata_limited_files",
    "split_metadata_limited_closures", "split_chunk_payload_chars_min",
    "split_chunk_payload_chars_max",
    "split_plan_details_total", "split_plan_details_retained",
    "split_plan_details_omitted", "split_plan_details_digest",
    "truncate_plan_details_total", "truncate_plan_details_retained",
    "truncate_plan_details_omitted", "truncate_plan_details_digest",
    "split_blocked_details_total", "split_blocked_details_retained",
    "split_blocked_details_omitted", "split_blocked_details_digest",
    "scanner_size_skip_details_total", "scanner_size_skip_details_retained",
    "scanner_size_skip_details_omitted", "scanner_size_skip_details_digest",
    "scanner_admission_skip_details_total", "scanner_admission_skip_details_retained",
    "scanner_admission_skip_details_omitted", "scanner_admission_skip_details_digest",
    "split_recovery_discarded_predecessor_nodes",
    "split_recovery_replacement_nodes_planned",
)


# ``prompt_profile_scope_counts`` is a route-wide closed mapping;
# ``split_blocked_by_reason`` is published only on a genuine split route.
_S8_INFO_CLOSED_MAPPINGS_ALWAYS = ("prompt_profile_scope_counts",)
_S8_INFO_CLOSED_MAPPINGS_SPLIT_ONLY = ("split_blocked_by_reason",)

# Only a genuine split route (``large_file_strategy: "split"``) also publishes the
# two split-scoped recovery counters; a truncate / no-oversize / zero-work route
# omits them entirely. The two EPHEMERAL transition counts above stay route-wide.
_S8_INFO_SPLIT_ONLY_FIELDS = ("split_recovery_conflict_files", "split_reexecuted_nodes")


def _s8_info_check_closed_mapping(line, fields, kind, mk):
    """Assert one closed aggregate mapping is emitted as canonical compact JSON."""
    assert f"{mk}={{" in line, (kind, mk)
    val = fields[mk]
    assert val.startswith("{") and val.endswith("}")
    parsed = json.loads(val)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in parsed.items())
    assert val == json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return parsed


def _s8_info_fields(caplog):
    lines = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("Planned provider work (before calls):")
    ]
    assert len(lines) == 1, lines
    payload = lines[0].split("Planned provider work (before calls): ", 1)[1]
    return lines[0], dict(kv.split("=", 1) for kv in payload.split("; "))


def _s8_info_scenario(kind, tmp_path, monkeypatch):
    """Install the provider / constants for ``kind`` and return its run config.

    Every branch drives the pipeline as far as the pre-call reporter site (which
    emits the INFO aggregate); ``_blocks`` marks a scenario that then aborts with
    a ``ConfigError`` *after* that site.
    """
    if kind == "active_profile":
        # A genuinely active cross-file profile needs a review-capable provider
        # so prompt review clears; the reporter still precedes it.
        (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider", lambda _c: _ReviewFake("SAFE")
        )
        return {"entry_file": "main.py", "prompt_profiles": _cross_file_profile(),
                "parallel_agents": False, "propagate_changes": False}
    _s8_event_spy(monkeypatch, [])
    if kind == "normal_no_oversize":
        write_py(tmp_path / "main.py")
        return {"entry_file": "main.py", "parallel_agents": False,
                "propagate_changes": False}
    if kind == "resolved_split":
        (tmp_path / "main.py").write_text(
            _s8_large_source(), encoding="utf-8", newline="",
        )
        return {"entry_file": "main.py", "large_file_strategy": "split",
                "max_content_chars": 2000, "parallel_agents": False,
                "propagate_changes": False}
    if kind == "split_blocked":
        # Force every oversized file past the per-file chunk cap so the split
        # route resolves but division is blocked with reason "chunk-cap".
        monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 1)
        (tmp_path / "main.py").write_text("V = 1\n", encoding="utf-8")
        (tmp_path / "zeta.py").write_text(
            _s8_large_source(), encoding="utf-8", newline="",
        )
        return {"entry_file": "main.py", "documentation_scope": "all",
                "large_file_strategy": "split", "max_content_chars": 2000,
                "parallel_agents": False, "propagate_changes": False,
                "_blocks": True}
    if kind == "zero_work":
        return {"entry_file": None, "auto_entry_candidates": [],
                "propagate_changes": False}
    if kind == "scanner_skipped_zero_work":
        (tmp_path / "huge.py").write_text(
            "\n".join(f"v{i} = {i}" for i in range(60000)), encoding="utf-8"
        )
        return {"entry_file": None, "auto_entry_candidates": [],
                "max_file_size_kb": 1, "propagate_changes": False}
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    ["normal_no_oversize", "resolved_split", "split_blocked", "active_profile",
     "zero_work", "scanner_skipped_zero_work"],
)
def test_s8_info_fallback_publishes_the_complete_bounded_aggregate_contract(
    tmp_path, monkeypatch, caplog, kind
):
    cfg = _s8_info_scenario(kind, tmp_path, monkeypatch)
    blocks = cfg.pop("_blocks", False)
    with caplog.at_level(logging.INFO, logger="codedoc.pipeline"):
        if blocks:
            with pytest.raises(ConfigError):
                run_pipeline(tmp_path, cfg)
        else:
            run_pipeline(tmp_path, cfg)
    line, fields = _s8_info_fields(caplog)
    for name in _S8_INFO_REQUIRED_FIELDS:
        assert name in fields, (kind, name)
    split_route = fields["large_file_strategy_resolved"] == "split"
    for name in _S8_INFO_SPLIT_ONLY_FIELDS:
        assert (name in fields) == split_route, (kind, name)
    for mk in _S8_INFO_CLOSED_MAPPINGS_ALWAYS:
        _s8_info_check_closed_mapping(line, fields, kind, mk)
    for mk in _S8_INFO_CLOSED_MAPPINGS_SPLIT_ONLY:
        if split_route:
            parsed = _s8_info_check_closed_mapping(line, fields, kind, mk)
            if kind == "split_blocked":
                assert parsed.get("chunk-cap", 0) >= 1
        else:
            assert mk not in fields, (kind, mk)
    assert fields["retries_included_in_ceiling"] == "False"
    assert fields["max_planned_calls_applies_to"] == "initial_provider_calls_planned"
    assert "_details=[" not in line and "_details={" not in line
    assert "output_dir" not in fields and "output_files" not in fields
    assert "ownership_conflicts" not in fields
    assert str(tmp_path) not in line
    assert "\n" not in line and "\t" not in line
    if kind.endswith("zero_work"):
        assert fields["initial_provider_calls_planned"] == "0"
        assert fields["total_calls_planned"] == "0"
        assert fields["split_chunk_payload_chars_min"] == "0"
    if kind == "scanner_skipped_zero_work":
        assert int(fields["scanner_size_skip_details_total"]) == 1


# --- reporter ordering matrix ------------------------------------------------

def test_s8_reporter_precedes_provider_without_prompt_review(tmp_path, monkeypatch):
    write_py(tmp_path / "main.py")
    events: list = []
    _s8_event_spy(monkeypatch, events)
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "parallel_agents": False, "propagate_changes": False},
        plan_reporter=lambda _s: events.append("report"),
    )
    assert events[0] == "report"
    assert events.index("report") < events.index("provider")
    assert events.count("report") == 1


def test_s8_reporter_precedes_provider_and_review_with_active_profile(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    profile = _cross_file_profile()
    events: list = []
    fake = _ReviewFake("SAFE")

    def _spy(_config):
        events.append("provider")
        return fake

    monkeypatch.setattr("codedoc.pipeline.create_provider", _spy)
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "prompt_profiles": profile, "parallel_agents": False,
         "propagate_changes": False},
        plan_reporter=lambda _s: events.append("report"),
    )
    assert events[0] == "report"
    assert events.index("report") < events.index("provider")
    assert events.count("report") == 1
    assert fake.review_calls == 1 and fake.doc_calls == 1


def test_s8_reporter_precedes_the_division_blocked_error(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "zeta.py").write_text(_s8_large_source(), encoding="utf-8", newline="")
    monkeypatch.setattr(file_division, "MAX_CHUNKS_PER_FILE", 1)
    _s8_no_side_effects(monkeypatch)
    reports: list = []
    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py", "documentation_scope": "all",
                "large_file_strategy": "split", "max_content_chars": 2000,
                "parallel_agents": False, "propagate_changes": False,
            },
            plan_reporter=lambda s: reports.append(s),
        )
    assert len(reports) == 1
    assert reports[0]["split_blocked_details_total"] >= 1
    assert reports[0]["split_blocked_details"][0]["reason"] == "chunk-cap"


def test_s8_reporter_precedes_the_max_planned_calls_error(tmp_path, monkeypatch):
    write_py(tmp_path / "a.py")
    write_py(tmp_path / "b.py")
    _s8_no_side_effects(monkeypatch)
    reports: list = []
    with pytest.raises(ConfigError, match="max_planned_calls"):
        run_pipeline(
            tmp_path,
            {"entry_file": None, "auto_entry_candidates": [], "max_planned_calls": 1,
             "output_dir": "never-made"},
            plan_reporter=lambda s: reports.append(s),
        )
    assert len(reports) == 1
    assert reports[0]["max_planned_calls"] == 1
    assert reports[0]["max_planned_calls_exceeded"] is True
    assert not (tmp_path / "never-made").exists()


def test_s8_reporter_precedes_the_max_files_error(tmp_path, monkeypatch):
    write_py(tmp_path / "a.py")
    write_py(tmp_path / "b.py")
    _s8_no_side_effects(monkeypatch)
    reports: list = []
    with pytest.raises(ConfigError, match="max_files"):
        run_pipeline(
            tmp_path,
            {"entry_file": None, "auto_entry_candidates": [], "max_files": 1,
             "output_dir": "never-made"},
            plan_reporter=lambda s: reports.append(s),
        )
    assert len(reports) == 1
    assert reports[0]["max_files_exceeded"] is True
    assert not (tmp_path / "never-made").exists()


def test_s8_reporter_precedes_the_output_accessibility_probe(tmp_path, monkeypatch):
    write_py(tmp_path / "main.py")
    events: list = []
    monkeypatch.setattr(
        "codedoc.pipeline.preflight_output_accessibility",
        lambda *a, **k: events.append("probe"),
    )

    def _spy(_config):
        events.append("provider")
        return markdown_fake_provider()

    monkeypatch.setattr("codedoc.pipeline.create_provider", _spy)
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "parallel_agents": False, "propagate_changes": False},
        plan_reporter=lambda _s: events.append("report"),
    )
    assert events[0] == "report"
    assert events.index("report") < events.index("probe")


# --- explicit-entry admission five-case matrix ------------------------------

_S8_ADMISSION_CASES = {
    "size": {
        "setup": lambda d: (d / "main.py").write_text(
            "\n".join(f"v{i} = {i}" for i in range(40000)), encoding="utf-8"
        ),
        "cfg": {"entry_file": "main.py", "max_file_size_kb": 1},
        "category": "scanner_size_skip",
        "path": "main.py", "phase": "scanner-byte",
        "guidance": "raise-scan-byte-limit-or-exclude", "reason": None,
    },
    "ignored": {
        "setup": lambda d: (d / "main.py").write_text("x = 1\n", encoding="utf-8"),
        "cfg": {"entry_file": "main.py", "ignore_paths": ["main.py"]},
        "category": "scanner_admission_skip",
        "path": "main.py", "phase": "scanner-admission",
        "guidance": "adjust-ignore-or-entry", "reason": "ignored",
    },
    "unsupported": {
        "setup": lambda d: (d / "main.txt").write_text("entrypoint\n", encoding="utf-8"),
        "cfg": {"entry_file": "main.txt"},
        "category": "scanner_admission_skip",
        "path": "main.txt", "phase": "scanner-admission",
        "guidance": "configure-extension-or-entry", "reason": "unsupported",
    },
    "missing": {
        "setup": lambda d: None,
        "cfg": {"entry_file": "main.py"},
        "category": "scanner_admission_skip",
        "path": "main.py", "phase": "scanner-admission",
        "guidance": "fix-entry-path", "reason": "missing",
    },
}


@pytest.mark.parametrize("case_name", sorted(_S8_ADMISSION_CASES))
@pytest.mark.parametrize("dry_run", [True, False])
def test_s8_explicit_entry_admission_matrix(tmp_path, monkeypatch, case_name, dry_run):
    from codedoc.core.file_division import canonical_stream_digest

    spec = _S8_ADMISSION_CASES[case_name]
    spec["setup"](tmp_path)
    _s8_no_side_effects(monkeypatch)

    events: list = []
    reports: list = []

    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {**spec["cfg"], "dry_run": dry_run, "propagate_changes": False,
             "output_dir": "out"},
            plan_reporter=lambda s: (events.append("report"), reports.append(s)),
        )

    assert events == ["report"]
    assert not (tmp_path / "out" / "codedoc.json").exists()
    assert not (tmp_path / "out" / "crash_recovery.json").exists()

    snap = reports[0]
    assert snap["initial_provider_calls_planned"] == 0
    assert snap["prompt_review_calls_planned"] == 0
    assert snap["total_calls_planned"] == 0

    cat = spec["category"]
    other = "scanner_admission_skip" if cat == "scanner_size_skip" else "scanner_size_skip"
    details = [dict(x) for x in snap[cat + "_details"]]
    assert snap[cat + "_details_total"] == 1
    assert snap[cat + "_details_retained"] == 1
    assert snap[cat + "_details_omitted"] == 0
    assert len(details) == 1
    d0 = details[0]
    assert d0["path"] == spec["path"]
    assert d0["phase"] == spec["phase"]
    assert d0["guidance_code"] == spec["guidance"]
    if spec["reason"] is not None:
        assert d0["reason"] == spec["reason"]
    else:
        assert "reason" not in d0
        assert d0["observed"] > d0["limit"]
    assert snap[cat + "_details_digest"] == canonical_stream_digest(details)
    assert snap[other + "_details_total"] == 0
    assert snap[other + "_details"] == ()
    blob = json.dumps(_s8_norm(snap))
    assert str(tmp_path) not in blob
    for bad in ("\n", "\t", "\x1b", "\\"):
        assert bad not in d0["path"]


@pytest.mark.parametrize("dry_run", [True, False])
def test_s8_explicit_entry_unreadable_admission(tmp_path, monkeypatch, dry_run):
    """The bulk walk classifies an unreadable explicit entry (EACCES on stat,
    not ENOENT); the post-walk fold does not double-record it."""
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    real_stat = Path.stat

    def _denied(self, *a, **k):
        if self.name == "main.py" and str(tmp_path) in str(self):
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _denied)
    _s8_no_side_effects(monkeypatch)
    reports: list = []
    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {"entry_file": "main.py", "dry_run": dry_run, "propagate_changes": False,
             "output_dir": "out"},
            plan_reporter=lambda s: reports.append(s),
        )
    snap = reports[0]
    details = [dict(x) for x in snap["scanner_admission_skip_details"]]
    assert snap["scanner_admission_skip_details_total"] == 1
    assert details[0]["reason"] == "unreadable"
    assert details[0]["path"] == "main.py"
    assert details[0]["phase"] == "scanner-admission"
    assert details[0]["guidance_code"] == "fix-permissions-or-exclude"
    assert snap["scanner_size_skip_details_total"] == 0


def test_s8_explicit_entry_admission_survives_a_stale_source_rescan(tmp_path):
    """Final-generation replacement: two back-to-back scans on one diagnostics
    instance publish only the last generation (one descriptor, not a merge)."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    diag = ScanDiagnostics()
    diag.explicit_entry_hint = "missing.py"
    scan_files(tmp_path, extension_language_map={".py": "python"}, diagnostics=diag)
    scan_files(tmp_path, extension_language_map={".py": "python"}, diagnostics=diag)
    assert diag.scanner_admission_skip["details_total"] == 1
    assert [d["reason"] for d in diag.scanner_admission_skip["details"]] == ["missing"]
    assert diag.scanner_admission_skip["details"][0]["path"] == "missing.py"


# --- Correction round 3: explicit-entry zero-admission WITH other admitted -----
# The explicit target itself yields no admitted source, but unrelated project
# files WERE admitted. The pipeline must still fire the canonical empty reporter
# exactly once BEFORE _select_files()'s established deterministic error, and
# before any provider / writer / output probe / recovery init / output file.
# Dry and real preflight snapshots must be recursively identical except dry_run.

_S8_MIXED_CASES = {
    "size": {
        "make": lambda d: (d / "entry.py").write_text(
            "\n".join(f"v{i} = {i}" for i in range(40000)), encoding="utf-8"
        ),
        "cfg": {"entry_file": "entry.py", "max_file_size_kb": 1},
        "category": "scanner_size_skip", "paths": ["entry.py"], "reason": None,
    },
    "ignored": {
        "make": lambda d: (d / "entry.py").write_text("x = 1\n", encoding="utf-8"),
        "cfg": {"entry_file": "entry.py", "ignore_paths": ["entry.py"]},
        "category": "scanner_admission_skip", "paths": ["entry.py"], "reason": "ignored",
    },
    "unsupported": {
        "make": lambda d: (d / "entry.txt").write_text("entrypoint\n", encoding="utf-8"),
        "cfg": {"entry_file": "entry.txt"},
        "category": "scanner_admission_skip", "paths": ["entry.txt"], "reason": "unsupported",
    },
    "missing": {
        "make": lambda d: None,
        "cfg": {"entry_file": "entry.py"},
        "category": "scanner_admission_skip", "paths": ["entry.py"], "reason": "missing",
    },
    "empty_dir": {
        "make": lambda d: (d / "entry").mkdir(),
        "cfg": {"entry_file": "entry"},
        "category": None, "paths": [], "reason": None,
    },
    "excluded_dir": {
        "make": lambda d: (
            (d / "entry").mkdir(),
            (d / "entry" / "a.txt").write_text("x", encoding="utf-8"),
            (d / "entry" / "b.txt").write_text("x", encoding="utf-8"),
        ),
        "cfg": {"entry_file": "entry"},
        "category": "scanner_admission_skip",
        "paths": ["entry/a.txt", "entry/b.txt"], "reason": "unsupported",
        "siblings": [],
    },
    "hidden_ancestor_file": {
        "make": lambda d: (
            (d / ".hidden").mkdir(),
            (d / ".hidden" / "entry.py").write_text("x = 1\n", encoding="utf-8"),
            (d / ".hidden" / "sibling.py").write_text("y = 1\n", encoding="utf-8"),
        ),
        "cfg": {"entry_file": ".hidden/entry.py"},
        "category": "scanner_admission_skip",
        "paths": [".hidden/entry.py"], "reason": "ignored",
        "siblings": [".hidden/sibling.py"],
    },
    "skip_dirs_ancestor_file": {
        "make": lambda d: (
            (d / "vendor").mkdir(),
            (d / "vendor" / "entry.py").write_text("x = 1\n", encoding="utf-8"),
            (d / "vendor" / "sibling.py").write_text("y = 1\n", encoding="utf-8"),
        ),
        "cfg": {"entry_file": "vendor/entry.py", "skip_dirs": ["vendor"]},
        "category": "scanner_admission_skip",
        "paths": ["vendor/entry.py"], "reason": "ignored",
        "siblings": ["vendor/sibling.py"],
    },
    "missing_under_skipped_ancestor": {
        "make": lambda d: (
            (d / ".hidden").mkdir(),
            (d / ".hidden" / "sibling.py").write_text("y = 1\n", encoding="utf-8"),
        ),
        "cfg": {"entry_file": ".hidden/missing.py"},
        "category": "scanner_admission_skip",
        "paths": [".hidden/missing.py"], "reason": "missing",
        "siblings": [".hidden/sibling.py"],
    },
    "explicit_dir_under_skipped_ancestor": {
        "make": lambda d: (
            (d / ".hidden").mkdir(),
            (d / ".hidden" / "target").mkdir(),
            (d / ".hidden" / "target" / "a.py").write_text("x = 1\n", encoding="utf-8"),
            (d / ".hidden" / "outside.py").write_text("y = 1\n", encoding="utf-8"),
        ),
        "cfg": {"entry_file": ".hidden/target"},
        "category": "scanner_admission_skip",
        "paths": [".hidden/target/a.py"], "reason": "ignored",
        "siblings": [".hidden/outside.py", ".hidden/target"],
    },
}


def _s8_mixed_run(tmp_path, cfg, *, dry):
    """Run one mixed explicit-entry pipeline; capture (snapshot|None, events, err).

    Any provider / SafeWriter / output-accessibility probe is a hard failure -- a
    zero-admission explicit-entry error must reach the reporter and the
    ConfigError without constructing or probing any of them.
    """
    events: list = []
    snaps: list = []

    def _boom(name):
        def _f(*_a, **_k):
            events.append(name)
            raise AssertionError(f"{name} constructed/probed during zero-admission path")
        return _f

    err = None
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr("codedoc.pipeline.create_provider", _boom("provider"))
        mp.setattr("codedoc.pipeline.SafeWriter", _boom("writer"))
        mp.setattr("codedoc.pipeline.preflight_output_accessibility", _boom("probe"))
        try:
            run_pipeline(
                tmp_path, {**cfg, "dry_run": dry, "propagate_changes": False,
                           "output_dir": "out"},
                plan_reporter=lambda s: (events.append("report"), snaps.append(_s8_norm(s))),
            )
        except ConfigError as exc:
            err = exc
    finally:
        mp.undo()
    snap = snaps[0] if snaps else None
    if snap is not None:
        snap.pop("dry_run", None)
    return snap, events, err


@pytest.mark.parametrize("case_name", sorted(_S8_MIXED_CASES))
def test_s8_mixed_explicit_entry_zero_admission_reports_before_error(tmp_path, case_name):
    from codedoc.core.file_division import (
        EMPTY_PLAN_DETAILS_DIGEST,
        canonical_stream_digest,
    )

    spec = _S8_MIXED_CASES[case_name]
    (tmp_path / "other.py").write_text("value = 1\n", encoding="utf-8")
    spec["make"](tmp_path)

    dry_snap, dry_events, dry_err = _s8_mixed_run(tmp_path, spec["cfg"], dry=True)
    real_snap, real_events, real_err = _s8_mixed_run(tmp_path, spec["cfg"], dry=False)

    # reporter fired exactly once, first, in BOTH modes; then the ConfigError.
    assert dry_events == ["report"], (case_name, dry_events)
    assert real_events == ["report"], (case_name, real_events)
    assert isinstance(dry_err, ConfigError) and isinstance(real_err, ConfigError)
    assert str(dry_err) == str(real_err)
    # no output file, no recovery file, in either mode.
    assert not (tmp_path / "out" / "codedoc.json").exists()
    assert not (tmp_path / "out" / "crash_recovery.json").exists()

    # empty payable manifest.
    assert dry_snap is not None and real_snap is not None
    assert dry_snap["total_calls_planned"] == 0
    assert dry_snap["initial_provider_calls_planned"] == 0
    assert dry_snap["prompt_review_calls_planned"] == 0

    # scanner evidence is EXACTLY the target-owned paths -- an off-path
    # sibling of the explicit target is never target evidence.
    _all_paths = [
        d["path"]
        for c in ("scanner_admission_skip", "scanner_size_skip")
        for d in [dict(x) for x in dry_snap[c + "_details"]]
    ]
    for _sib in spec.get("siblings", []):
        assert _sib not in _all_paths, (case_name, _sib, _all_paths)
    if spec["category"] is not None:
        cat = spec["category"]
        other_cat = (
            "scanner_size_skip" if cat == "scanner_admission_skip"
            else "scanner_admission_skip"
        )
        details = [dict(x) for x in dry_snap[cat + "_details"]]
        got = [d["path"] for d in details]
        assert sorted(got) == sorted(spec["paths"]), (case_name, got)
        assert dry_snap[cat + "_details_total"] == len(spec["paths"])
        assert dry_snap[cat + "_details_retained"] == len(spec["paths"])
        assert dry_snap[cat + "_details_omitted"] == 0
        assert dry_snap[cat + "_details_digest"] == canonical_stream_digest(details)
        assert dry_snap[other_cat + "_details_total"] == 0
        assert dry_snap[other_cat + "_details_digest"] == EMPTY_PLAN_DETAILS_DIGEST
        if spec["reason"] is not None:
            assert all(d.get("reason") == spec["reason"] for d in details)
    else:
        # empty explicit directory: no invented path record.
        assert dry_snap["scanner_admission_skip_details_total"] == 0
        assert dry_snap["scanner_size_skip_details_total"] == 0

    # dry and real preflight snapshots are field-for-field identical (minus mode).
    assert dry_snap == real_snap


@pytest.mark.parametrize("dry_run", [True, False])
def test_s8_mixed_explicit_entry_unreadable_reports_before_error(tmp_path, monkeypatch, dry_run):
    (tmp_path / "other.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "entry.py").write_text("x = 1\n", encoding="utf-8")
    real_stat = Path.stat

    def _denied(self, *a, **k):
        if self.name == "entry.py" and str(tmp_path) in str(self):
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _denied)
    snap, events, err = _s8_mixed_run(tmp_path, {"entry_file": "entry.py"}, dry=dry_run)
    assert events == ["report"]
    assert isinstance(err, ConfigError)
    details = [dict(x) for x in snap["scanner_admission_skip_details"]]
    assert {d["path"] for d in details} == {"entry.py"}
    assert details[0]["reason"] == "unreadable"
    assert snap["scanner_size_skip_details_total"] == 0
    assert snap["total_calls_planned"] == 0


# --- zero-work full-snapshot parity ---------------------------------------

_S8_ZERO_WORK_SNAPSHOT_KEYS = (
    "analysis_mode", "initial_calls_per_file", "documentation_scope",
    "estimated_calls", "estimate_is_lower_bound",
    "initial_provider_calls_planned", "prompt_review_calls_planned",
    "initial_documentation_calls_planned", "correction_calls_possible_max",
    "provider_calls_max_before_retries", "file_retry_attempts",
    "retries_included_in_ceiling", "max_planned_calls_applies_to",
    "total_calls_planned", "max_planned_calls", "call_manifest_digest",
    "large_file_strategy_resolved", "large_file_source_ceiling_chars",
    "large_files_over_source_ceiling", "scanner_size_skip_details_total",
    "scanner_size_skip_details_digest", "scanner_admission_skip_details_total",
    "scanner_admission_skip_details_digest", "split_plan_details_digest",
    "truncate_plan_details_digest", "split_blocked_details_digest",
    "split_recovery_discarded_predecessor_nodes",
    "split_recovery_replacement_nodes_planned",
)


def test_s8_truly_empty_project_full_snapshot_parity_and_ordering(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _c: pytest.fail("provider")
    )
    monkeypatch.setattr(
        "codedoc.pipeline.SafeWriter", lambda *a, **k: pytest.fail("writer")
    )
    probe: list = []
    monkeypatch.setattr(
        "codedoc.pipeline.preflight_output_accessibility",
        lambda *a, **k: probe.append("probe"),
    )
    reports: list = []
    cfg = {"entry_file": None, "auto_entry_candidates": [], "propagate_changes": False}
    dry = run_pipeline(tmp_path, {**cfg, "dry_run": True},
                       plan_reporter=lambda s: reports.append(("dry", _s8_norm(s))))
    assert probe == []  # dry-run never probes output
    real = run_pipeline(tmp_path, cfg,
                        plan_reporter=lambda s: reports.append(("real", _s8_norm(s))))
    assert [t for t, _ in reports] == ["dry", "real"]
    dry_snap, real_snap = reports[0][1], reports[1][1]
    dry_snap.pop("dry_run")
    real_snap.pop("dry_run")
    assert dry_snap == real_snap
    assert dry_snap["scanner_size_skip_details_total"] == 0
    assert dry_snap["scanner_admission_skip_details_total"] == 0
    assert dry_snap["initial_provider_calls_planned"] == 0
    assert {k: _s8_norm(dry[k]) for k in _S8_ZERO_WORK_SNAPSHOT_KEYS} == {
        k: _s8_norm(real[k]) for k in _S8_ZERO_WORK_SNAPSHOT_KEYS
    }
    for _split_only in _S8_INFO_SPLIT_ONLY_FIELDS:
        assert _split_only not in dry and _split_only not in real
    assert dry["split_recovery_discarded_predecessor_nodes"] == 0
    assert dry["split_recovery_replacement_nodes_planned"] == 0
    assert dry["scanned"] == 0 and dry["initial_calls_per_file"] == 0
    assert real["checked"] == 0 and real["live_backup_path"] is None
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()


def test_s8_all_scanner_skipped_project_full_snapshot_parity(tmp_path, monkeypatch):
    from codedoc.core.file_division import canonical_stream_digest

    big = "\n".join(f"value_{i} = {i}" for i in range(60000)) + "\n"
    (tmp_path / "huge.py").write_text(big, encoding="utf-8")
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _c: pytest.fail("provider")
    )
    reports: list = []
    cfg = {"entry_file": None, "auto_entry_candidates": [], "max_file_size_kb": 1,
           "propagate_changes": False}
    dry = run_pipeline(tmp_path, {**cfg, "dry_run": True},
                       plan_reporter=lambda s: reports.append(_s8_norm(s)))
    real = run_pipeline(tmp_path, cfg,
                        plan_reporter=lambda s: reports.append(_s8_norm(s)))
    d, r = reports
    d.pop("dry_run")
    r.pop("dry_run")
    assert d == r
    assert d["scanner_size_skip_details_total"] == 1
    assert d["scanner_size_skip_details"][0]["path"] == "huge.py"
    assert d["scanner_size_skip_details_digest"] == canonical_stream_digest(
        d["scanner_size_skip_details"]
    )
    assert d["scanner_admission_skip_details_total"] == 0
    assert dry["files_skipped_large"] == real["files_skipped_large"] == 1


# --- Correction round 5: alternate-casing explicit entry is a normal plan -----

def _cr5_ci_fs(tmp_path):
    probe = tmp_path / "_cr5_ci_probe.py"
    probe.write_text("x = 1\n", encoding="utf-8")
    try:
        return (tmp_path / "_CR5_CI_PROBE.PY").exists()
    finally:
        probe.unlink()


def test_s8_cr5_alternate_casing_supported_entry_is_not_zero_work(tmp_path, monkeypatch):
    (tmp_path / "entry.py").write_text("VALUE = 1\n", encoding="utf-8")
    _s8_event_spy(monkeypatch, [])
    reports: list = []
    if not _cr5_ci_fs(tmp_path):
        # case-sensitive host: ENTRY.PY is genuinely absent -> ConfigError.
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "ENTRY.PY", "dry_run": True,
                                    "propagate_changes": False},
                         plan_reporter=lambda s: reports.append(s))
        return
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "ENTRY.PY", "dry_run": True, "propagate_changes": False,
         "parallel_agents": False},
        plan_reporter=lambda s: reports.append(s),
    )
    # a real, non-zero dry-run plan for the existing supported target.
    assert stats["initial_provider_calls_planned"] == 1
    assert stats["total_calls_planned"] >= 1
    assert stats["scanned"] == 1
    assert stats["scanner_admission_skip_details_total"] == 0
    assert stats["scanner_size_skip_details_total"] == 0
    assert len(reports) == 1


def test_s8_cr5_alternate_casing_entry_real_run_documents_the_file(tmp_path, monkeypatch):
    if not _cr5_ci_fs(tmp_path):
        import pytest as _pytest
        _pytest.skip("case-sensitive host: alt casing is a different path")
    (tmp_path / "entry.py").write_text("VALUE = 1\n", encoding="utf-8")
    _s8_event_spy(monkeypatch, [])
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "ENTRY.PY", "propagate_changes": False, "parallel_agents": False},
    )
    assert stats["checked"] == 1 and stats["failed"] == 0
    out = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8"))
    assert any(f["path"] == "entry.py" for f in out["files"])
    assert out["last_run"]["entry_file"] == "entry.py"


def test_s8_cr5_alternate_casing_entry_survives_stale_source_rebuild(tmp_path, monkeypatch):
    if not _cr5_ci_fs(tmp_path):
        import pytest as _pytest
        _pytest.skip("case-sensitive host: alt casing is a different path")
    (tmp_path / "entry.py").write_text("import dep\nVALUE = 1\n", encoding="utf-8")
    (tmp_path / "dep.py").write_text("D = 1\n", encoding="utf-8")
    _s8_event_spy(monkeypatch, [])
    run_pipeline(tmp_path, {"entry_file": "ENTRY.PY", "documentation_scope": "all",
                            "propagate_changes": False, "parallel_agents": False})
    # mutate a source so the next run triggers the stale-source rebuild path.
    (tmp_path / "dep.py").write_text("D = 2\nE = 3\n", encoding="utf-8")
    stats = run_pipeline(tmp_path, {"entry_file": "ENTRY.PY", "documentation_scope": "all",
                                    "propagate_changes": True, "parallel_agents": False})
    assert stats["failed"] == 0
    assert stats["checked"] >= 1


# ===========================================================================
# Section 7 (Defect 2): the canonical project-relative entry-hint rule must be
# the FIRST thing run_pipeline applies to config["entry_file"] -- before any
# entry-derived resolve/stat/exists probe and before the generated-target
# collision check (_reject_generated_target_collision). The existing coverage
# in tests/unit/core/scanner/test_rules.py calls detect_entry_file() directly,
# so the production ordering inside run_pipeline was never observed: an invalid
# spelling used to reach the filesystem, and an invalid non-project-relative
# spelling that normalized onto a generated target used to lose to the
# collision error (which echoed the raw absolute/traversal spelling).
# ===========================================================================

# Distinctive spellings use the token ``zzhint`` / ``zzoutside`` -- absent from
# the repo path and from any pytest tmp-dir name -- so a Path.resolve() call on
# an entry-derived path is unambiguously detectable.
_D2_INVALID_ENTRY_HINTS = [
    "C:/zzabs/zzhint.py",            # absolute / drive-qualified
    "C:main.py",                     # drive-relative, no separator
    "C:zzrel/zzhint.py",             # drive-relative with a path
    "//host/share/zzhint.py",        # UNC
    "../zzoutside.py",               # parent traversal
    "a/../../zzoutside.py",          # parent traversal mid-path
    ".",                            # dot-only
    "./.",                         # dot segments only
    "./C:/zzhint.py",              # post-normalization drive bypass
]


def _d2_instrument(tmp_path, monkeypatch):
    """Record every ``exclude_path_key`` call and ``Path.resolve`` target, and
    make provider construction an assertion failure. Returns the two lists."""
    import pathlib

    import codedoc.pipeline as pipeline_mod

    exclude_calls: list = []
    real_exclude = pipeline_mod.exclude_path_key
    monkeypatch.setattr(
        pipeline_mod,
        "exclude_path_key",
        lambda p: (exclude_calls.append(str(p)), real_exclude(p))[1],
    )
    resolve_targets: list = []
    real_resolve = pathlib.Path.resolve
    monkeypatch.setattr(
        pathlib.Path,
        "resolve",
        lambda self, *a, **k: (
            resolve_targets.append(str(self)),
            real_resolve(self, *a, **k),
        )[1],
    )

    def _no_provider(_config):
        raise AssertionError("no provider may be constructed before entry validation")

    monkeypatch.setattr(pipeline_mod, "create_provider", _no_provider)
    return exclude_calls, resolve_targets


@pytest.mark.parametrize("hint", _D2_INVALID_ENTRY_HINTS)
def test_d2_invalid_entry_hint_rejected_by_run_pipeline_before_any_probe(
    tmp_path, monkeypatch, hint
):
    from codedoc.core.scanner import _ENTRY_HINT_NOT_PROJECT_RELATIVE

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    exclude_calls, resolve_targets = _d2_instrument(tmp_path, monkeypatch)

    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(
            tmp_path,
            {
                "entry_file": hint,
                "output_dir": "docs",
                "parallel_agents": False,
                "propagate_changes": False,
            },
        )

    # The canonical project-relative rule wins -- its reason text is
    # byte-identical to the shared constant, and it is NOT the generated-target
    # collision error.
    assert exc_info.value.reason == _ENTRY_HINT_NOT_PROJECT_RELATIVE
    message = str(exc_info.value)
    assert "generated output target" not in message
    # The raw spelling is never echoed: the exact reason match above already
    # proves it, and the collision error's quoted echo form is absent too.
    assert f"'{hint}'" not in message
    # No collision check / scan-exclude-set build ran: validation is strictly
    # first, so the pipeline-imported exclude_path_key was never called.
    assert exclude_calls == []
    # No entry-derived resolve fired: no controlled token from an invalid hint
    # reached Path.resolve().
    assert not any("zz" in target for target in resolve_targets), resolve_targets
    # No provider, writer, or output artifact was produced.
    assert not (tmp_path / "docs").exists()
    assert list(tmp_path.rglob("codedoc.json")) == []
    assert list(tmp_path.rglob("crash_recovery.json")) == []


@pytest.mark.parametrize("target", ["codedoc.json", "codedoc.md", "crash_recovery.json"])
def test_d2_traversal_onto_generated_target_loses_to_the_canonical_rule(
    tmp_path, monkeypatch, target
):
    """A non-project-relative ('..') spelling that normalizes exactly onto a
    generated output target must be rejected by the canonical rule, NOT by the
    collision check echoing the raw traversal spelling."""
    from codedoc.core.scanner import _ENTRY_HINT_NOT_PROJECT_RELATIVE

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    exclude_calls, _resolve_targets = _d2_instrument(tmp_path, monkeypatch)
    hint = f"../{tmp_path.name}/docs/{target}"

    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(
            tmp_path,
            {
                "entry_file": hint,
                "output_dir": "docs",
                "parallel_agents": False,
                "propagate_changes": False,
            },
        )

    assert exc_info.value.reason == _ENTRY_HINT_NOT_PROJECT_RELATIVE
    message = str(exc_info.value)
    assert "generated output target" not in message
    assert hint not in message and target not in message
    assert exclude_calls == []
    assert not (tmp_path / "docs").exists()


def test_d2_blank_entry_hint_is_rejected_before_any_probe(tmp_path, monkeypatch):
    """A whitespace-only entry hint is rejected early (config validation) with a
    ConfigError -- never the collision error, never a filesystem probe, and
    never a raw echo of the blank spelling."""
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    exclude_calls, _resolve_targets = _d2_instrument(tmp_path, monkeypatch)

    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "   ",
                "output_dir": "docs",
                "parallel_agents": False,
                "propagate_changes": False,
            },
        )

    message = str(exc_info.value)
    assert "generated output target" not in message
    assert exclude_calls == []
    assert not (tmp_path / "docs").exists()


def test_d2_valid_normalizable_entry_hints_still_run_through_the_pipeline(
    tmp_path, monkeypatch
):
    """Positive control: './main.py' and 'pkg//sub/./found.py' point inside the
    project root, so the early canonical check accepts them and the run
    proceeds and canonicalizes exactly as before."""
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    pkg = tmp_path / "pkg" / "sub"
    pkg.mkdir(parents=True)
    (pkg / "found.py").write_text("y = 2\n", encoding="utf-8")

    patch_provider(monkeypatch)
    stats_dot = run_pipeline(
        tmp_path,
        {
            "entry_file": "./main.py",
            "output_dir": "d1",
            "parallel_agents": False,
            "propagate_changes": False,
        },
    )
    assert stats_dot["failed"] == 0
    assert (tmp_path / "d1" / "codedoc.json").exists()

    stats_redundant = run_pipeline(
        tmp_path,
        {
            "entry_file": "pkg//sub/./found.py",
            "output_dir": "d2",
            "parallel_agents": False,
            "propagate_changes": False,
        },
    )
    assert stats_redundant["failed"] == 0
    out = json.loads((tmp_path / "d2" / "codedoc.json").read_text(encoding="utf-8"))
    assert any(f["path"] == "pkg/sub/found.py" for f in out["files"])
