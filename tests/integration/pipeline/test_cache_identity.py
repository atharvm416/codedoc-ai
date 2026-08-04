"""Tests organized by feature ownership."""

from __future__ import annotations

import hashlib
from pathlib import Path

from codedoc.core.record_meta import expected_analysis_identity
from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
import json
from tests.support.pipeline_scenarios import make_fake_provider
from tests.support.pipeline_scenarios import _cache_identity
import pytest
from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.execution_requests import make_execution_request
from tests.support.one_call_cases import _CountingProvider
from tests.support.response_correction_cases import RoutingProvider
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.pipeline import run_pipeline
from tests.support.profiles import INLINE
from tests.support.providers import SmartFake
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _first_run
from codedoc.core.db import compute_file_hash
from codedoc.core.file_division import (
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.planning import build_pipeline_plan
from codedoc.core.record_meta import (
    expected_large_file_identity,
    normalized_identity_value,
)

def test_pipeline_reuses_identical_file_content_without_llm(tmp_path, monkeypatch):
    import json

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    entry = tmp_path / "entry.py"
    content = "def shared():\n    return 1\n"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    entry.write_text("import first\nimport second\n", encoding="utf-8")

    # Pre-write the public JSON with first.py and entry.py docs and their hashes,
    # so that second.py (identical content to first.py) can be reused by hash.
    docs_output = tmp_path / "docs_output"
    docs_output.mkdir()
    first_hash = compute_file_hash(first)
    entry_hash = compute_file_hash(entry)
    (docs_output / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "entry.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "entry.py",
                    "hash": entry_hash,
                    "description": "Entry module.",
                    "language": "python",
                    "format": "py",
                    "imports": ["first", "second"],
                    **_PRIOR_RUN_IDENTITY,
                },
                {
                    "path": "first.py",
                    "hash": first_hash,
                    "description": "Shared helper.",
                    "language": "python",
                    "format": "py",
                    **_PRIOR_RUN_IDENTITY,
                },
            ],
        }),
        encoding="utf-8",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for identical cached content")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "entry.py",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["reused"] == 1
    output = (tmp_path / "docs_output" / "codedoc.json").read_text(encoding="utf-8")
    assert '"path": "first.py"' in output
    assert '"path": "second.py"' in output
    assert '"description": "Shared helper."' in output

    # Second run: public JSON still exists, all files are up-to-date, nothing to reuse
    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "entry.py",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["reused"] == 0
    assert (tmp_path / "docs_output" / "codedoc.json").exists()

def test_H1_identical_content_files_reuse_docs(tmp_path, monkeypatch):
    """H1: two files with byte-for-byte identical content — the second reuses the
    documented output of the first without an LLM call (docs_by_hash dedup).

    main.py imports both helper_a and helper_b.  helper_a is pre-documented in the
    JSON; helper_b is new this run but has identical content to helper_a → reused."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    shared = "def helper(): pass\n"
    (tmp_path / "main.py").write_text("import helper_a\nimport helper_b\n")
    (tmp_path / "helper_a.py").write_text(shared)
    (tmp_path / "helper_b.py").write_text(shared)  # byte-for-byte identical to helper_a

    shared_hash = compute_file_hash(tmp_path / "helper_a.py")
    assert shared_hash == compute_file_hash(tmp_path / "helper_b.py")
    main_hash = compute_file_hash(tmp_path / "main.py")

    # Pre-write JSON: main + helper_a documented; helper_b is NEW this run
    (tmp_path / "codedoc").mkdir()
    (tmp_path / "codedoc" / "codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [
            {"path": "main.py",     "hash": main_hash,   "language": "python", "description": "Entry.", **_cache_identity()},
            {"path": "helper_a.py", "hash": shared_hash, "language": "python", "description": "Shared helper.", **_cache_identity()},
        ],
    }), encoding="utf-8")

    call_count = {"n": 0}
    original = make_fake_provider("Shared helper.")
    orig_complete = original.complete_json
    def counting_complete(prompt, system=""):
        call_count["n"] += 1
        return orig_complete(prompt, system)
    original.complete_json = counting_complete
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: original)

    # main + helper_a: hashes match → skipped (not in process_rels)
    # helper_b: new → in process_rels → content hash matches helper_a in docs_by_hash → reused
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats.get("reused", 0) >= 1
    assert call_count["n"] == 0  # LLM never called

def _pipeline_provider(monkeypatch):
    provider = _CountingProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: provider)
    return provider

def test_generated_record_carries_cache_identity(tmp_path):
    result = Orchestrator(_CountingProvider(), analysis_mode="single").process(
        make_execution_request(tmp_path, "pkg/mod.py", "x = 1\n", imports=("os",))
    )
    assert result["_analysis_revision"] == ANALYSIS_REVISION
    assert result["_analysis_mode"] == "single"

def test_steady_state_reuse_skips_provider(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    cfg = {"entry_file": "main.py", "analysis_mode": "single", "propagate_changes": False}

    _pipeline_provider(monkeypatch)
    first = run_pipeline(tmp_path, cfg)
    assert first["checked"] == 1

    # Second run must not even create a provider.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda c: pytest.fail("provider created though all files were reusable"),
    )
    second = run_pipeline(tmp_path, cfg)
    assert second["checked"] == 0


def test_final_output_hash_remains_bound_to_the_documented_source_snapshot(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "main.py"
    planned_source = "ORIGINAL = 1\n"
    later_source = "CHANGED_AFTER_PLANNING = 2\n"
    source_path.write_text(planned_source, encoding="utf-8")
    expected_snapshot_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    class MutateAfterPlanning(SmartFake):
        def __init__(self) -> None:
            super().__init__()
            self.mutated = False

        def complete_json(self, prompt, system=""):
            if not self.mutated and "standards/safety review" not in prompt:
                self.mutated = True
                source_path.write_text(later_source, encoding="utf-8")
            return super().complete_json(prompt, system)

    provider = MutateAfterPlanning()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: provider
    )

    first = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "output_dir": "docs",
            "propagate_changes": False,
        },
    )

    record = json.loads(
        (tmp_path / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert first["checked"] == 1
    assert source_path.read_text(encoding="utf-8") == later_source
    assert record["hash"] == expected_snapshot_hash
    assert record["hash"] != compute_file_hash(source_path)

    second_provider = SmartFake()
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider", lambda _config: second_provider
    )
    second = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "output_dir": "docs",
            "propagate_changes": False,
        },
    )
    assert second["checked"] == 1
    assert second_provider.doc_calls == 1


def test_mode_switch_invalidates_reuse(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    _pipeline_provider(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                            "propagate_changes": False})

    # Switching to triple changes the cache identity → reprocess once.
    provider = _pipeline_provider(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "triple",
                                    "parallel_agents": False, "propagate_changes": False})
    assert stats["checked"] == 1
    assert provider.calls == 3

def test_legacy_record_without_identity_reprocessed_once(tmp_path, monkeypatch):
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "codedoc"
    out.mkdir()
    # Pre-0.10.0 record: matching hash, no cache-identity keys.
    out.joinpath("codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [{"path": "main.py", "hash": compute_file_hash(main),
                   "language": "python", "description": "legacy"}],
    }), encoding="utf-8")

    provider = _pipeline_provider(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                    "propagate_changes": False})
    assert stats["checked"] == 1
    assert provider.calls == 1
    # After reprocessing, the record carries the current identity.
    rec = json.loads(out.joinpath("codedoc.json").read_text(encoding="utf-8"))["files"][0]
    assert rec["_analysis_revision"] == ANALYSIS_REVISION
    assert rec["_analysis_mode"] == "single"

@pytest.mark.parametrize("source", ["same_path", "identical"])
@pytest.mark.parametrize("stored_language", ["python", "javascript", None])
@pytest.mark.parametrize(
    ("identity_change", "identity_value"),
    [
        (None, None),
        ("_analysis_revision", None),
        ("_analysis_revision", "stale-revision"),
        ("_analysis_mode", None),
        ("_analysis_mode", "triple"),
    ],
)
def test_every_reuse_source_requires_complete_matching_identity_and_language(
    tmp_path, source, stored_language, identity_change, identity_value
):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan

    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    content_hash = compute_file_hash(target)
    identity = {
        "_analysis_revision": ANALYSIS_REVISION,
        "_analysis_mode": "single",
    }
    if identity_change is not None:
        if identity_value is None:
            identity.pop(identity_change)
        else:
            identity[identity_change] = identity_value

    record = {
        "path": "cached.py",
        "hash": content_hash,
        "description": "cached",
        **identity,
    }
    if stored_language is not None:
        record["language"] = stored_language
    existing_docs = {}
    if source == "same_path":
        existing_docs["main.py"] = record
    else:  # identical
        existing_docs["cached.py"] = record

    graph = DependencyGraph()
    graph.add_file("main.py")
    plan, _materials = build_pipeline_plan(
        file_map={
            "main.py": {
                "path": target,
                "rel_path": "main.py",
                "language": "python",
                "extension": ".py",
            }
        },
        graph=graph,
        selected_rels={"main.py"},
        entry_rel="main.py",
        existing_docs=existing_docs,
        forced_paths=[],
        config={
            "analysis_mode": "single",
            "propagate_changes": False,
            "max_files": 0,
        },
    )

    reuse_matches = identity_change is None and stored_language == "python"
    if reuse_matches and source == "same_path":
        assert plan.unchanged_rels == frozenset({"main.py"})
    elif reuse_matches:
        assert plan.identical_reuse_rels == frozenset({"main.py"})
    else:
        assert plan.agent_rels == frozenset({"main.py"})

def test_cache_identity_is_v2():
    assert ANALYSIS_REVISION == "file-doc-v3"
    assert expected_analysis_identity("single") == {
        "_analysis_revision": "file-doc-v3",
        "_analysis_mode": "single",
    }

def test_v1_record_is_invalidated_once_under_v2(tmp_path):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    file_map = {
        "main.py": {
            "path": tmp_path / "main.py",
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    file_hash = compute_file_hash(tmp_path / "main.py")

    def _plan(revision):
        existing = {
            "main.py": {
                    "path": "main.py",
                    "hash": file_hash,
                    "description": "cached",
                    "language": "python",
                    "_analysis_revision": revision,
                "_analysis_mode": "single",
            }
        }
        plan, _ = build_pipeline_plan(
            file_map, graph, {"main.py"}, "main.py", existing, [],
            {"propagate_changes": False, "max_files": 0, "analysis_mode": "single"},
        )
        return plan

    # A current-revision record with an unchanged hash is reused (no LLM call):
    # it is skipped as unchanged, never routed to an agent.
    current = _plan(ANALYSIS_REVISION)
    assert "main.py" in current.unchanged_rels
    assert "main.py" not in current.agent_rels
    # A stale file-doc-v1 record is invalidated and reprocessed once.
    stale = _plan("file-doc-v1")
    assert "main.py" in stale.agent_rels
    assert "main.py" not in stale.unchanged_rels

def test_corrected_successful_record_is_reused_from_cache(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    prov = RoutingProvider(fail_agents={"combined"})
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: prov)
    first = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "response_correction_enabled": True,
            "propagate_changes": False,
        },
    )
    assert first["checked"] == 1
    assert first["response_correction_calls_succeeded"] == 1

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _c: pytest.fail("unchanged corrected record must be reusable"),
    )
    second = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "response_correction_enabled": True,
            "propagate_changes": False,
        },
    )
    assert second["checked"] == 0
    assert second["documentation_calls_attempted"] == 0

def test_profile_identity_change_still_reprocesses_fallback(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")

    fake = SmartFake("SAFE")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: fake)
    config = {**_config("md"), "prompt_profiles": INLINE}
    stats = run_pipeline(tmp_path, config)

    assert fake.review_calls == 1
    assert fake.doc_calls == 1
    assert stats["checked"] == 1

def test_analysis_mode_change_reprocesses_cross_format_fallback(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")

    fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: fake)
    stats = run_pipeline(tmp_path, {**_config("md"), "analysis_mode": "triple"})

    assert stats["checked"] == 1
    assert fake.doc_calls == 3
    record = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert record["_analysis_mode"] == "triple"

def test_truncation_identity_change_reprocesses_cross_format_fallback(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x" * 2000, encoding="utf-8")
    first_fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: first_fake)
    run_pipeline(tmp_path, {**_config("json"), "max_content_chars": 2000})
    assert first_fake.doc_calls == 1

    second_fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: second_fake)
    stats = run_pipeline(tmp_path, {**_config("md"), "max_content_chars": 1500})

    assert stats["checked"] == 1
    assert second_fake.doc_calls == 1
    record = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert record["_max_context_revision"] == "truncate-v1:max=1500:head=0.7000"

_OMIT = object()

def _oversized_plan(tmp_path, stored_mcr, *, max_chars=1000, head_ratio=0.70):
    """Plan one oversized (2000-char) file whose cached record carries *stored_mcr*.

    Pass ``_OMIT`` to leave ``_max_context_revision`` off the record entirely.
    """
    src = tmp_path / "main.py"
    src.write_text("x" * 2000, encoding="utf-8")  # 2000 chars > 1000 ceiling
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    record = {
        "path": "main.py",
        "hash": compute_file_hash(src),
        "description": "cached",
        "language": "python",
        "_analysis_revision": "file-doc-v3",
        "_analysis_mode": "single",
    }
    if stored_mcr is not _OMIT:
        record["_max_context_revision"] = stored_mcr
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "max_content_chars": max_chars, "truncation_head_ratio": head_ratio,
    }
    plan, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
    )
    return plan

def _small_plan(tmp_path, *, max_chars, head_ratio=0.70):
    src = tmp_path / "main.py"
    src.write_text("x = 1\n", encoding="utf-8")  # 6 chars, never truncated
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    # A small file would never carry _max_context_revision.
    record = {
        "path": "main.py", "hash": compute_file_hash(src), "description": "cached",
        "language": "python",
        "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
    }
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "max_content_chars": max_chars, "truncation_head_ratio": head_ratio,
    }
    plan, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
    )
    return plan

def test_oversized_file_with_matching_revision_is_reused(tmp_path):
    plan = _oversized_plan(tmp_path, "truncate-v1:max=1000:head=0.7000")
    assert "main.py" in plan.unchanged_rels
    assert "main.py" not in plan.agent_rels

def test_legacy_oversized_record_without_revision_is_reprocessed_once(tmp_path):
    plan = _oversized_plan(tmp_path, _OMIT)
    assert "main.py" in plan.agent_rels
    assert "main.py" not in plan.unchanged_rels

def test_raising_ceiling_reprocesses_truncated_file(tmp_path):
    # Cached under ceiling 1000; now running with ceiling 1500 (file is still
    # 2000 chars, so still truncated, but under a new identity).
    plan = _oversized_plan(tmp_path, "truncate-v1:max=1000:head=0.7000", max_chars=1500)
    assert "main.py" in plan.agent_rels
    assert "main.py" not in plan.unchanged_rels

def test_changing_head_ratio_reprocesses_truncated_file(tmp_path):
    plan = _oversized_plan(tmp_path, "truncate-v1:max=1000:head=0.7000", head_ratio=0.85)
    assert "main.py" in plan.agent_rels
    assert "main.py" not in plan.unchanged_rels

def test_small_file_reusable_across_ceiling_and_ratio_changes(tmp_path):
    assert "main.py" in _small_plan(tmp_path, max_chars=1000).unchanged_rels
    assert "main.py" in _small_plan(
        tmp_path, max_chars=5000, head_ratio=0.85
    ).unchanged_rels

def test_analysis_revision_is_v3():
    assert ANALYSIS_REVISION == "file-doc-v3"

def test_v2_record_is_invalidated():
    # A stored v2 record no longer matches the current v3 identity.
    assert normalized_identity_value("_analysis_revision", {"_analysis_revision": "file-doc-v2"}) == (
        "file-doc-v2"
    )
    assert ANALYSIS_REVISION != "file-doc-v2"

def _split_plan(tmp_path, *, max_chars=2000, head_ratio=0.70):
    """One reusable completed split record under the current release policy."""
    src = tmp_path / "main.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source, source_budget_chars=max_chars
    )
    tree = build_reduction_tree(
        plan,
        max_content_chars=max_chars,
        language="python",
    )
    identity = expected_large_file_identity(
        source_chars=len(source),
        max_chars=max_chars,
        rel_path="main.py",
        division_plan_digest=plan.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode,
        imports_digest=deterministic_imports_digest(()),
    )
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    record = {
        "path": "main.py",
        "hash": compute_file_hash(src),
        "description": "cached",
        "language": "python",
        "_analysis_revision": "file-doc-v3",
        "_analysis_mode": "single",
        "_large_file_identity": identity,
    }
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": max_chars, "truncation_head_ratio": head_ratio,
    }
    plan, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
    )
    return plan

def test_split_file_with_matching_identity_is_reused(tmp_path):
    assert "main.py" in _split_plan(tmp_path).unchanged_rels

def test_split_file_identity_is_invariant_to_truncation_head_ratio(tmp_path):
    """A truncate-only head-ratio change does not invalidate current split reuse."""
    assert "main.py" in _split_plan(tmp_path, head_ratio=0.70).unchanged_rels
    assert "main.py" in _split_plan(tmp_path, head_ratio=0.85).unchanged_rels


def test_actual_predecessor_completed_split_record_is_rejected_as_stale(tmp_path):
    """The frozen record was produced by the reviewed 0.14.1 commit, rather
    than reconstructed in this test. Both predecessor identity values must
    cause current planning to schedule the file as unpaid work."""
    fixture_dir = Path(__file__).resolve().parents[2] / "fixtures" / "split_state"
    predecessor = json.loads(
        (fixture_dir / "completed_0_14_1.json").read_text(encoding="utf-8")
    )
    record = predecessor["files"][0]
    assert record["_split_reuse_contract"] == "fresh-only-v1"
    assert record["_large_file_identity"].startswith("large-file-v2:")

    rel_path = record["path"]
    source = (Path(__file__).with_name(rel_path)).read_bytes()
    src = tmp_path / rel_path
    src.write_bytes(source)
    assert compute_file_hash(src) == record["hash"]
    file_map = {
        rel_path: {
            "path": src, "rel_path": rel_path,
            "language": "python", "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file(rel_path)
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2500, "truncation_head_ratio": 0.70,
    }
    plan_result, _ = build_pipeline_plan(
        file_map, graph, {rel_path}, rel_path, {rel_path: record}, [], config,
    )
    assert rel_path not in plan_result.unchanged_rels
    assert rel_path in plan_result.changed_rels


def test_actual_predecessor_completed_split_recovery_has_no_partial_files():
    fixture_dir = Path(__file__).resolve().parents[2] / "fixtures" / "split_state"
    payload = json.loads(
        (fixture_dir / "recovery_0_14_1_completed_split.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["_codedoc"]["status"] == "in_progress"
    assert "partial_files" not in payload["_codedoc"]
    assert len(payload["files"]) == 1
    record = payload["files"][0]
    assert record["_split_reuse_contract"] == "fresh-only-v1"
    assert record["_large_file_identity"].startswith("large-file-v2:")
