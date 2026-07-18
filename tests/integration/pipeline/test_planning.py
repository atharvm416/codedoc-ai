"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
import json
from tests.support.pipeline_scenarios import patch_provider
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_scenarios import _cache_identity
from tests.support.pipeline_scenarios import write_existing_json
from tests.support.pipeline_scenarios import write_existing_md
import logging
import pytest
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.pipeline_usage import write_py
from tests.support.pipeline_usage import make_graph
from dataclasses import asdict, fields
from codedoc.core.loader import load_config
from codedoc.core.planning import PipelinePlan, build_pipeline_plan
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.selection_projects import _project
from tests.support.selection_projects import _graph_and_map
from tests.support.markdown_cases import _fake_provider as markdown_fake_provider
from tests.support.configuration_cases import _fake_provider as configuration_fake_provider
from codedoc.core.discovery import _resolve_entry_and_docs
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.pipeline import _final_entry_source
from tests.support.run_metadata_cases import _records
from tests.support.run_metadata_cases import _stats

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
             "description": "Old A.", **_cache_identity()},
            {"path": "b.py", "hash": b_hash, "language": "python",
             "description": "Old B.", **_cache_identity()},
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
            {"path": "a.py", "hash": "old_a_hash", "language": "python", "description": "Old A.", **_cache_identity()},
            {"path": "b.py", "hash": b_hash, "language": "python", "description": "Old B.", **_cache_identity()},
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
            "_analysis_revision": ANALYSIS_REVISION,
            "_analysis_mode": "single",
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
