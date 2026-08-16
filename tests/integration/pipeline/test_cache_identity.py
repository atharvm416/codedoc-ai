"""Tests organized by feature ownership."""

from __future__ import annotations

import hashlib
from pathlib import Path

from codedoc.core.record_meta import expected_analysis_identity
from tests.support.pipeline_identity import _prior_run_identity
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
from codedoc.core.db import compute_file_hash, read_source_text
from codedoc.core.file_division import (
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.planning import build_pipeline_plan
from codedoc.core import record_meta
from codedoc.core.record_meta import (
    expected_large_file_identity,
    expected_ordinary_path_identity,
    normalized_identity_value,
)
from tests.support.fixture_paths import FIXTURES_ROOT
from tests.support.structure_extra import requires_structure_pack

def test_pipeline_same_path_reuse_free_but_cross_path_content_match_is_not(
    tmp_path, monkeypatch
):
    """Same-path records (entry.py, first.py) reuse for free; second.py has
    byte-identical content to first.py but is a different path, so ordinary
    cross-path reuse is refused (0.14.4) and it is documented fresh."""
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

    # Pre-write the public JSON with first.py and entry.py docs and their
    # hashes, so both are same-path reusable. second.py (identical content to
    # first.py, but a different path) has no prior record of its own.
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
                    **_prior_run_identity("entry.py"),
                },
                {
                    "path": "first.py",
                    "hash": first_hash,
                    "description": "Shared helper.",
                    "language": "python",
                    "format": "py",
                    **_prior_run_identity("first.py"),
                },
            ],
        }),
        encoding="utf-8",
    )

    fake = make_fake_provider("Second helper, freshly documented.")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: fake)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "entry.py",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 1
    assert stats["reused"] == 0
    output = (tmp_path / "docs_output" / "codedoc.json").read_text(encoding="utf-8")
    assert '"path": "first.py"' in output
    assert '"path": "second.py"' in output
    assert '"description": "Shared helper."' in output
    assert '"description": "Second helper, freshly documented."' in output

    # Second run: public JSON now covers second.py under its own path, so
    # every file is same-path unchanged and nothing is reprocessed.
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("provider must not be created; nothing changed"),
    )
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

def test_H1_identical_content_cross_path_is_not_reused(tmp_path, monkeypatch):
    """H1 (corrected for 0.14.4): two files with byte-for-byte identical
    content no longer share documentation across paths.

    main.py imports both helper_a and helper_b. helper_a is pre-documented in
    the JSON and unchanged, so it is reused for free; helper_b is new this
    run and has identical content to helper_a, but ordinary cross-path reuse
    is refused, so it is documented by a fresh provider call."""
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
            {"path": "main.py",     "hash": main_hash,   "language": "python", "description": "Entry.", **_cache_identity("main.py")},
            {"path": "helper_a.py", "hash": shared_hash, "language": "python", "description": "Shared helper.", **_cache_identity("helper_a.py")},
        ],
    }), encoding="utf-8")

    call_count = {"n": 0}
    original = make_fake_provider("Freshly documented helper_b.")
    orig_complete = original.complete_json
    def counting_complete(prompt, system=""):
        call_count["n"] += 1
        return orig_complete(prompt, system)
    original.complete_json = counting_complete
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: original)

    # main + helper_a: hashes match → skipped (not in process_rels)
    # helper_b: new → in process_rels → byte-identical to helper_a in
    # docs_by_hash, but a different path, so ordinary cross-path reuse is
    # refused and it is documented fresh.
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats.get("reused", 0) == 0
    assert stats.get("checked", 0) == 1
    assert call_count["n"] == 1  # LLM called exactly once, for helper_b only
    record = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    helper_b = next(f for f in record["files"] if f["path"] == "helper_b.py")
    assert helper_b["description"] == "Freshly documented helper_b."

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

def test_pre_0_14_4_record_stays_invalid_until_successfully_replaced(tmp_path, monkeypatch):
    """A pre-0.14.4 record with every other cache-identity field matching
    (hash, _analysis_revision, _analysis_mode, language) but no
    _ordinary_path_identity is invalid and is regenerated exactly once; once
    regenerated, same-path reuse costs zero calls on the next run, and the
    regenerated _ordinary_path_identity round-trips unchanged into the
    Markdown embedded view without a provider call."""
    from codedoc.core.db import compute_file_hash
    from codedoc.core.project_view import read_embedded_view
    from codedoc.core.record_meta import expected_ordinary_path_identity
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "codedoc"
    out.mkdir()
    out.joinpath("codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [{
            "path": "main.py", "hash": compute_file_hash(main),
            "language": "python", "description": "pre-0.14.4",
            "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
        }],
    }), encoding="utf-8")

    provider = _pipeline_provider(monkeypatch)
    first = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                     "propagate_changes": False})
    assert first["checked"] == 1
    assert provider.calls == 1
    rec = json.loads(out.joinpath("codedoc.json").read_text(encoding="utf-8"))["files"][0]
    assert rec["_ordinary_path_identity"] == expected_ordinary_path_identity("main.py")

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: pytest.fail("second run must not create a provider"),
    )
    second = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                      "propagate_changes": False, "output_format": "md"})
    assert second["checked"] == 0
    md_text = (out / "codedoc.md").read_text(encoding="utf-8")
    embedded = read_embedded_view(md_text)
    embedded_rec = next(f for f in embedded["files"] if f["path"] == "main.py")
    assert embedded_rec["_ordinary_path_identity"] == expected_ordinary_path_identity("main.py")

    # Convert back MD -> JSON: still zero calls, and _ordinary_path_identity
    # round-trips unchanged into the freshly written JSON too.
    third = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                     "propagate_changes": False, "output_format": "json"})
    assert third["checked"] == 0
    rec_after_round_trip = json.loads(
        out.joinpath("codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    assert rec_after_round_trip["_ordinary_path_identity"] == expected_ordinary_path_identity(
        "main.py"
    )

def test_pre_0_14_4_record_stays_invalid_after_a_failed_regeneration_attempt(
    tmp_path, monkeypatch
):
    """0.14.4 audit fix: the test above never actually proves 'stays invalid
    until successfully replaced' -- its one regeneration attempt always
    succeeds. Here the regeneration's write fails, so the stale legacy
    record must not be mistaken for a successful replacement: it stays on
    disk exactly as it was, and a following run must regenerate it fully
    from scratch rather than skip it as already fixed."""
    from codedoc.core.db import compute_file_hash
    import codedoc.core.safe_writer as safe_writer_mod
    from codedoc.utils.errors import LiveBackupWriteError

    main = tmp_path / "main.py"
    main.write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "codedoc"
    out.mkdir()
    out.joinpath("codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [{
            "path": "main.py", "hash": compute_file_hash(main),
            "language": "python", "description": "pre-0.14.4",
            "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
        }],
    }), encoding="utf-8")
    before_bytes = out.joinpath("codedoc.json").read_bytes()

    _pipeline_provider(monkeypatch)
    original_atomic_write_text = safe_writer_mod.atomic_write_text

    def boom(path, text):
        raise OSError("simulated disk failure during regeneration")

    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", boom)
    with pytest.raises(LiveBackupWriteError):
        run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                 "propagate_changes": False})

    # The failed attempt must not silently persist a stamped replacement.
    assert out.joinpath("codedoc.json").read_bytes() == before_bytes
    after_failure = json.loads(before_bytes)["files"][0]
    assert "_ordinary_path_identity" not in after_failure

    # Write path healthy again: the record is still invalid, so a following
    # run regenerates it fully -- never skips it as if already replaced.
    monkeypatch.setattr(safe_writer_mod, "atomic_write_text", original_atomic_write_text)
    provider = _pipeline_provider(monkeypatch)
    second = run_pipeline(tmp_path, {"entry_file": "main.py", "analysis_mode": "single",
                                      "propagate_changes": False})
    assert second["checked"] == 1
    assert provider.calls == 1
    rec = json.loads(out.joinpath("codedoc.json").read_text(encoding="utf-8"))["files"][0]
    assert rec["_ordinary_path_identity"] == expected_ordinary_path_identity("main.py")

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
    from codedoc.core.record_meta import expected_ordinary_path_identity

    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    content_hash = compute_file_hash(target)
    # The stored record's own path must match its destination for same_path
    # reuse to even be eligible for consideration; "identical" stores it under
    # a genuinely different path, since that is what cross-path candidate
    # selection actually looks like.
    stored_path = "main.py" if source == "same_path" else "cached.py"
    identity = {
        "_analysis_revision": ANALYSIS_REVISION,
        "_analysis_mode": "single",
        "_ordinary_path_identity": expected_ordinary_path_identity(stored_path),
    }
    if identity_change is not None:
        if identity_value is None:
            identity.pop(identity_change)
        else:
            identity[identity_change] = identity_value

    record = {
        "path": stored_path,
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
    else:
        # A same-path record with any mismatched identity/language field is
        # reprocessed. A cross-path candidate (source == "identical") is
        # always refused regardless of identity/language match: ordinary
        # identical-content reuse is same-path only (0.14.4).
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
                "_ordinary_path_identity": expected_ordinary_path_identity("main.py"),
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
        "_ordinary_path_identity": expected_ordinary_path_identity("main.py"),
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
        "_ordinary_path_identity": expected_ordinary_path_identity("main.py"),
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
    fixture_dir = FIXTURES_ROOT / "split_state"
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


@requires_structure_pack
def test_actual_predecessor_completed_split_record_is_stale_under_v7(tmp_path, monkeypatch):
    """The frozen record's ``_large_file_identity`` is not a placeholder: it
    is the real value ``expected_large_file_identity`` produces for this
    exact source under the genuine reconstructed division plan, reduction
    tree, and deterministic-imports digest -- with ``LEAF_CAPSULE_SCHEMA_REVISION``
    and ``MAX_LEAF_CAPSULE_CANONICAL_CHARS`` monkeypatched back to the
    reviewed 0.14.2 values (``leaf-capsule-v5`` / 150,656) that produced it.
    Every other bound/revision is genuinely unchanged.

    This is section 20A item 2's two-direction proof, not merely a
    single-direction staleness check: a fixture that fails to match
    *either* mode would prove nothing, since any arbitrary wrong hash is
    trivially "stale" under the current identity too. The plan/tree
    reconstruction uses the real parser, so this depends on the optional
    `structure` extra exactly as the frozen fixture does (recorded via
    ``@requires_structure_pack``, matching section 20A item 1's treatment of
    the sibling schema-4 fixture) -- the absence-simulation lexical-fallback
    contract is proven elsewhere and is not weakened by this skip.

    0.14.2 no longer stamps ``_split_reuse_contract`` (retired in the 0.14.2
    completed-split-reuse work), so this fixture omits it, unlike the 0.14.1
    fixture above."""
    fixture_dir = FIXTURES_ROOT / "split_state"
    predecessor = json.loads(
        (fixture_dir / "completed_0_14_2.json").read_text(encoding="utf-8")
    )
    record = predecessor["files"][0]
    assert record["_large_file_identity"].startswith("large-file-v3:")
    assert "_split_reuse_contract" not in record

    rel_path = record["path"]
    source_bytes = (Path(__file__).with_name(rel_path)).read_bytes()
    src = tmp_path / rel_path
    src.write_bytes(source_bytes)
    assert compute_file_hash(src) == record["hash"]

    # The canonical pipeline encoding (utf-8-sig, universal newlines), not a
    # raw-bytes decode -- read_source_text is what planning and execution
    # actually feed into build_division_plan, so this must match it exactly
    # or the reconstructed plan/tree digests silently diverge from a real run.
    source = read_source_text(src)
    plan = build_division_plan(
        rel_path=rel_path, language="python", content=source, source_budget_chars=2500,
    )
    tree = build_reduction_tree(plan, max_content_chars=2500, language="python")
    imports_digest = deterministic_imports_digest(tuple(record["imports"]))

    # Section 20A item 5: the fixture's own last_run structural statistics
    # must agree with this same reconstructed plan/tree, not merely its
    # _large_file_identity. A fixture whose identity says "real syntax-mode
    # plan" but whose last_run counts still describe a different (e.g.
    # lexical-fallback) execution could not have been emitted by any single
    # genuine run -- checked here so a future hand-edit or partial
    # regeneration that touches one without the other fails loudly instead
    # of silently drifting back into that incoherent state.
    last_run = predecessor["last_run"]
    assert (last_run["split_syntax_files"], last_run["split_lexical_files"]) == (
        (1, 0) if plan.structural_mode == "syntax" else (0, 1)
    )
    assert last_run["split_units"] == len(plan.units)
    assert last_run["split_chunks"] == len(plan.chunks)
    assert last_run["split_unit_consolidation_levels"] == (
        1 if tree.unit_consolidation_nodes else 0
    )
    assert last_run["split_general_reduction_levels"] == (
        1 if tree.general_nodes else 0
    )
    assert last_run["split_final_synthesis_calls_planned"] == (
        1 if tree.final_node is not None else 0
    )

    file_map = {
        rel_path: {
            "path": src, "rel_path": rel_path,
            "language": "python", "extension": ".py",
        }
    }
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2500, "truncation_head_ratio": 0.70,
    }

    # First direction: with the module-level bound/revision monkeypatched
    # back to the reviewed 0.14.2 values, both the identity computation AND
    # current planning's own internal recomputation (build_pipeline_plan
    # calls expected_large_file_identity again to compare) see v5 -- so the
    # build_pipeline_plan call proving compatible reuse must happen while
    # the patch is still active, not after it is undone.
    monkeypatch.setattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v5")
    monkeypatch.setattr(record_meta, "MAX_LEAF_CAPSULE_CANONICAL_CHARS", 150656)
    v5_identity = record_meta.expected_large_file_identity(
        source_chars=len(source), max_chars=2500, rel_path=rel_path,
        division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode, imports_digest=imports_digest,
    )
    assert v5_identity == record["_large_file_identity"], (
        "fixture does not match a genuine predecessor v5/150,656 computation"
    )

    graph_v5 = DependencyGraph()
    graph_v5.add_file(rel_path)
    plan_result_v5, _ = build_pipeline_plan(
        file_map, graph_v5, {rel_path}, rel_path, {rel_path: record}, [], config,
    )
    assert rel_path in plan_result_v5.unchanged_rels
    assert rel_path not in plan_result_v5.changed_rels

    # Second direction: undo the patch, restoring the real current (v7)
    # bound/revision, and confirm the same record is now genuinely stale.
    monkeypatch.undo()
    assert record_meta.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v7"
    assert record_meta.MAX_LEAF_CAPSULE_CANONICAL_CHARS == 448672

    v7_identity = record_meta.expected_large_file_identity(
        source_chars=len(source), max_chars=2500, rel_path=rel_path,
        division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode, imports_digest=imports_digest,
    )
    assert v7_identity != v5_identity

    graph_v7 = DependencyGraph()
    graph_v7.add_file(rel_path)
    plan_result_v7, _ = build_pipeline_plan(
        file_map, graph_v7, {rel_path}, rel_path, {rel_path: record}, [], config,
    )
    assert rel_path not in plan_result_v7.unchanged_rels
    assert rel_path in plan_result_v7.changed_rels


def test_actual_predecessor_completed_split_recovery_has_no_partial_files():
    fixture_dir = FIXTURES_ROOT / "split_state"
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
