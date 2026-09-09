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
import codedoc.core.file_division as file_division
import codedoc.core.planning as planning_mod
from codedoc.core.file_division import (
    MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS,
    SPLIT_PARTIAL_SCHEMA_VERSION,
    SplitTreeState,
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
    leaf_execution_identity,
    leaf_input_digest,
    provider_execution_identity,
    reduction_execution_identity,
    reduction_input_digest,
    tree_node_state,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.execution_model import build_call_manifest
from codedoc.core.planning import build_pipeline_plan
from codedoc.core import record_meta
from codedoc.core.record_meta import (
    expected_large_file_identity,
    expected_ordinary_path_identity,
    normalized_identity_value,
)
from tests.support.fixture_paths import FIXTURES_ROOT
from tests.support.structure_extra import requires_structure_pack


def _historical_crlf_source_bytes(path: Path) -> bytes:
    """Reconstruct the exact CRLF byte stream the frozen predecessor records
    below were hashed from, independent of how Git materialized line endings in
    the current checkout.

    The completed-predecessor fixtures freeze the raw SHA-256 of
    ``test_config_precedence.py`` as it existed with CRLF endings
    (``f4cdd10b...696bb``). ``.gitattributes`` deliberately pins an EOL policy
    only for ``tests/fixtures/**``, so a normal ``.py`` checkout is LF on Linux
    and CRLF on Windows and would otherwise hash two different ways. Collapsing
    every CRLF and every lone CR to LF and then re-emitting CRLF yields the one
    canonical historical stream on every platform, while any non-EOL content
    change still moves the hash and keeps the frozen value a real oracle.

    This is a test-local materialization only: it never touches
    ``compute_file_hash``, ``read_source_text``, source decoding, cache
    identity, ``.gitattributes``, or the live file on disk.
    """
    raw = Path(path).read_bytes()
    lf_only = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return lf_only.replace(b"\n", b"\r\n")


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

def test_analysis_identity_is_the_current_revision():
    assert ANALYSIS_REVISION == "file-doc-v4"
    assert expected_analysis_identity("single") == {
        "_analysis_revision": "file-doc-v4",
        "_analysis_mode": "single",
    }

def test_v1_record_is_invalidated_once_under_the_current_revision(tmp_path):
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
        "_analysis_revision": ANALYSIS_REVISION,
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
        "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
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

def test_analysis_revision_is_current():
    assert ANALYSIS_REVISION == "file-doc-v4"

def test_a_predecessor_analysis_record_is_invalidated():
    # A stored predecessor record (here file-doc-v2) no longer matches the
    # current analysis identity.
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
    # No FileExecutionRequest here: reconstruct the exact synthesis budget
    # production planning carries -- the automatic floor -- so this expected
    # identity matches what build_pipeline_plan computes for the same config.
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
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
        "_analysis_revision": ANALYSIS_REVISION,
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
    source = _historical_crlf_source_bytes(Path(__file__).with_name(rel_path))
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
def test_actual_predecessor_completed_split_record_is_stale_under_current_identity(
    tmp_path,
):
    """The frozen record's ``_large_file_identity`` is a genuine 0.14.2
    predecessor value, and it is genuinely stale under the current identity.

    Provenance -- why the frozen value is trusted even though current code
    can no longer re-derive it. It was reproduced *exactly* by real 0.14.2
    code at commit ``ae22733`` ("0.14.2 hashlib issue resolved"). Recipe,
    run read-only (e.g. ``git archive ae22733 | tar -x`` into a scratch dir,
    that dir first on ``sys.path``):

        source = read_source_text(<repo>/tests/integration/pipeline/test_config_precedence.py)
        plan   = build_division_plan(rel_path="test_config_precedence.py",
                                     language="python", content=source,
                                     source_budget_chars=2500)
        tree   = build_reduction_tree(plan, max_content_chars=2500, language="python")
        idig   = deterministic_imports_digest(tuple(record["imports"]))
        record_meta.expected_large_file_identity(...)  # -> the frozen value

    with 0.14.2's own constants -- ``division-packer-v5`` /
    ``leaf-capsule-v5`` / ``reduction-packing-v4`` / ``file-reduction-v1``
    and ``MAX_LEAF_CAPSULE_CANONICAL_CHARS`` 150,656 -- over a mode=syntax
    plan of 8 units / 2 chunks. This is deliberately NOT re-run as a test:
    doing so would make the suite depend on git history and offline archive
    extraction for no added safety.

    Current code cannot reconstruct that value by monkeypatching the four
    revision constants back, so this test no longer tries to:

    - ``_plan_payload`` now unconditionally writes
      ``"leaf_prompt_signature_hint_chars"`` into the division-plan digest
      payload (``codedoc/core/file_division.py:2012``; 0 occurrences at
      HEAD). ``monkeypatch.setattr`` on a constant cannot remove a key
      literal from a function body, so the payload *shape* differs from
      0.14.2's regardless of revision values.
    - ``pack_chunks`` changed algorithm under the packer v5 -> v6 advance.
      Patching the revision *string* back relabels the digest; it does not
      restore the earlier chunk boundaries.

    What is proven here instead: reconstructing the plan/tree with *current*
    code and comparing ``expected_large_file_identity`` -- and
    ``build_pipeline_plan``'s own classification -- against the frozen
    fixture value shows the completed record is stale and is scheduled as
    changed work, never reused unchanged. The reconstruction uses the real
    parser (``structural_mode == "syntax"`` for this source), so it depends
    on the optional ``structure`` extra exactly as the frozen fixture's own
    ``last_run`` statistics do (recorded via ``@requires_structure_pack``);
    a base install cannot reproduce a syntax-mode plan and skips rather than
    failing for an environment the fixture was never generated in.

    0.14.2 no longer stamps ``_split_reuse_contract`` (retired in the 0.14.2
    completed-split-reuse work), so this fixture omits it, unlike the 0.14.1
    fixture above."""
    fixture_dir = FIXTURES_ROOT / "split_state"
    predecessor = json.loads(
        (fixture_dir / "completed_0_14_2.json").read_text(encoding="utf-8")
    )
    record = predecessor["files"][0]
    assert record["_large_file_identity"].startswith("large-file-v3:")
    # Drift tripwire on the one field whose reconstruction guard cannot
    # survive the current payload shape (see Provenance above). Everything
    # below proves the record is *stale* -- which any wrong hash would also
    # satisfy -- so this is what proves it is the *right* stale value: the
    # one real 0.14.2 code at commit ``ae22733`` produced. Not evidence of a
    # computation (the docstring records that separately); a freeze, like the
    # sibling ``partial_schema4_0_14_6.json`` fixture pin.
    assert record["_large_file_identity"] == (
        "large-file-v3:f3819f3552084f7c4555546ad78401d61cc15fb3e455ad1c29d6e84ea3ff9be2"
    )
    assert "_split_reuse_contract" not in record

    rel_path = record["path"]
    source_bytes = _historical_crlf_source_bytes(Path(__file__).with_name(rel_path))
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

    # Local sanity guard: the current identity inputs are what this test was
    # written against. If any of these move, the staleness measured below is
    # against the wrong baseline and this test needs revisiting rather than
    # silently passing. Advanced to leaf-capsule-v11 / file-reduction-v4 by the
    # plan section 5.6.2 revision advance (0.14.9); the frozen 0.14.2
    # ``large-file-v3:f3819f35...`` predecessor value is even further from the
    # current identity now, so both staleness directions below still hold.
    assert record_meta.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v11"
    assert record_meta.MAX_LEAF_CAPSULE_CANONICAL_CHARS == 986272
    assert record_meta.REDUCER_PROMPT_REVISION == "file-reduction-v4"
    assert file_division.PACKER_SCHEMA_REVISION == "division-packer-v6"
    assert file_division.REDUCTION_PACKING_REVISION == "reduction-packing-v5"

    # Staleness, direction 1 -- the identity function itself: a current
    # reconstruction's expected_large_file_identity does not match the
    # frozen predecessor value.
    current_identity = record_meta.expected_large_file_identity(
        source_chars=len(source), max_chars=2500, rel_path=rel_path,
        division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode, imports_digest=imports_digest,
    )
    assert current_identity != record["_large_file_identity"]

    # Staleness, direction 2 -- planning's own classification agrees: the
    # completed record is scheduled as changed work, never reused unchanged.
    graph = DependencyGraph()
    graph.add_file(rel_path)
    plan_result, _ = build_pipeline_plan(
        file_map, graph, {rel_path}, rel_path, {rel_path: record}, [], config,
    )
    assert rel_path not in plan_result.unchanged_rels
    assert rel_path in plan_result.changed_rels


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


def test_completed_leaf_capsule_v7_record_is_planned_as_unpaid_work(tmp_path, monkeypatch):
    """0.14.6: a *completed* 0.14.5 split record must stop being reused.

    The sibling above proves the same thing for the frozen `0.14.2` (`v5`)
    predecessor, and a unit test proves the identity digest moves between
    `v7` and `v8`. Neither shows the planner acting on the `v7` -> `v8`
    advance specifically, which is the migration real users hit. This builds
    a record stamped with a genuine `v7` `large-file-v3:` identity -- produced
    by `expected_large_file_identity` itself with the constant patched back,
    never hand-written -- and runs it through `build_pipeline_plan` twice.

    Two directions, so it cannot pass vacuously: reusable while the constant
    reads `leaf-capsule-v7`, unpaid work under the real current revision."""
    src = tmp_path / "main.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    max_chars = 2000
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=max_chars,
    )
    # Reconstruct the automatic synthesis floor production planning carries,
    # so this expected identity matches build_pipeline_plan for the same config.
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    identity_kwargs = dict(
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
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": max_chars, "truncation_head_ratio": 0.70,
    }

    def plan_for(record):
        graph = DependencyGraph()
        graph.add_file("main.py")
        result, _ = build_pipeline_plan(
            file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
        )
        return result

    monkeypatch.setattr(
        record_meta, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v7"
    )
    v7_identity = record_meta.expected_large_file_identity(**identity_kwargs)
    assert v7_identity.startswith("large-file-v3:")
    record = {
        "path": "main.py",
        "hash": compute_file_hash(src),
        "description": "documented by 0.14.5",
        "language": "python",
        "_analysis_revision": ANALYSIS_REVISION,
        "_analysis_mode": "single",
        "_large_file_identity": v7_identity,
    }

    # Direction one: this really is a valid completed record of its own release.
    under_v7 = plan_for(record)
    assert "main.py" in under_v7.unchanged_rels
    assert "main.py" not in under_v7.changed_rels

    # Direction two: the identical record under the real current revision.
    monkeypatch.undo()
    current_identity = record_meta.expected_large_file_identity(**identity_kwargs)
    assert current_identity != v7_identity
    assert current_identity.startswith("large-file-v3:")

    under_current = plan_for(record)
    assert "main.py" not in under_current.unchanged_rels
    assert "main.py" in under_current.changed_rels


def test_completed_leaf_capsule_v8_record_is_planned_as_unpaid_work(tmp_path, monkeypatch):
    """0.14.7: a *completed* 0.14.6 split record must stop being reused.

    The exact sibling of the `v7` proof above, one revision later, for the
    same reason: neither the frozen-fixture staleness test nor a bare
    constant pin shows the planner acting on the `v8` -> `v9` advance this
    release makes specifically. Unlike
    `test_actual_predecessor_completed_split_record_is_stale_under_current_identity`,
    which measures staleness against a frozen fixture value pinned literally
    and a current-revision baseline guard, this test's own two directions
    dynamically recompute "current" via `expected_large_file_identity`
    itself, so a source-level revert of `LEAF_CAPSULE_SCHEMA_REVISION` back
    to `v8` collapses "current" onto the "v8" direction and fails these
    `build_pipeline_plan` assertions for a genuine mechanical reason, not a
    hardcoded string comparison.

    Two directions, so it cannot pass vacuously: reusable while the constant
    reads `leaf-capsule-v8`, unpaid work under the real current revision."""
    src = tmp_path / "main.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    max_chars = 2000
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=max_chars,
    )
    # Reconstruct the automatic synthesis floor production planning carries,
    # so this expected identity matches build_pipeline_plan for the same config.
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    identity_kwargs = dict(
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
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": max_chars, "truncation_head_ratio": 0.70,
    }

    def plan_for(record):
        graph = DependencyGraph()
        graph.add_file("main.py")
        result, _ = build_pipeline_plan(
            file_map, graph, {"main.py"}, "main.py", {"main.py": record}, [], config,
        )
        return result

    monkeypatch.setattr(
        record_meta, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v8"
    )
    v8_identity = record_meta.expected_large_file_identity(**identity_kwargs)
    assert v8_identity.startswith("large-file-v3:")
    record = {
        "path": "main.py",
        "hash": compute_file_hash(src),
        "description": "documented by 0.14.6",
        "language": "python",
        "_analysis_revision": ANALYSIS_REVISION,
        "_analysis_mode": "single",
        "_large_file_identity": v8_identity,
    }

    # Direction one: this really is a valid completed record of its own release.
    under_v8 = plan_for(record)
    assert "main.py" in under_v8.unchanged_rels
    assert "main.py" not in under_v8.changed_rels

    # Direction two: the identical record under the real current revision.
    monkeypatch.undo()
    current_identity = record_meta.expected_large_file_identity(**identity_kwargs)
    assert current_identity != v8_identity
    assert current_identity.startswith("large-file-v3:")

    under_current = plan_for(record)
    assert "main.py" not in under_current.unchanged_rels
    assert "main.py" in under_current.changed_rels


_OLD_FOUR_REVISIONS = {
    "PACKER_SCHEMA_REVISION": "division-packer-v5",
    "LEAF_CAPSULE_SCHEMA_REVISION": "leaf-capsule-v8",
    "REDUCTION_PACKING_REVISION": "reduction-packing-v4",
    "REDUCER_PROMPT_REVISION": "file-reduction-v2",
}


def _fail_if_validated(*_args, **_kwargs):
    raise AssertionError(
        "validate_recovered_tree must not run for a cross-plan transition."
    )


def test_true_v5_v8_v4_v2_predecessor_is_unpaid_completed_and_cross_plan_carry_partial(
    tmp_path, monkeypatch
):
    """Section 9.22 / 6.3: a predecessor whose four internal revisions were
    ``division-packer-v5`` / ``leaf-capsule-v8`` / ``reduction-packing-v4`` /
    ``file-reduction-v2`` -- every digest, node ID, and execution identity
    produced by the production functions with those four constants patched
    back, then evaluated with the patches undone -- is:

      (1) planned as *unpaid work* through ``build_pipeline_plan`` when it is a
          completed record; and
      (2) carried byte-for-byte into cross-plan fresh-preserve when it is a
          schema-4 partial, reusing zero predecessor nodes and scheduling the
          whole current split fresh.

    No hash is hand-authored. The frozen/hash-pinned deferred cache test is
    left untouched (this is a separate dynamic proof).
    """
    src = tmp_path / "main.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    content_hash = compute_file_hash(src)
    budget = 2000
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": budget, "truncation_head_ratio": 0.70,
    }
    file_map = {
        "main.py": {
            "path": src, "rel_path": "main.py",
            "language": "python", "extension": ".py",
        }
    }

    # ---- Build the genuine four-revision predecessor artifacts -------------
    with monkeypatch.context() as mp:
        for name, value in _OLD_FOUR_REVISIONS.items():
            mp.setattr(file_division, name, value)
            if hasattr(record_meta, name):
                mp.setattr(record_meta, name, value)
        old_plan = build_division_plan(
            rel_path="main.py", language="python", content=source,
            source_budget_chars=budget,
        )
        # The genuine coupled pre-0.14.7 synthesis value: the raw source budget.
        old_tree = build_reduction_tree(
            old_plan, max_content_chars=budget, language="python"
        )
        old_provider_identity = provider_execution_identity(config)
        old_completed_identity = record_meta.expected_large_file_identity(
            source_chars=len(source), max_chars=budget, rel_path="main.py",
            division_plan_digest=old_plan.plan_digest,
            reduction_tree_digest=old_tree.tree_digest,
            structural_mode=old_plan.structural_mode,
            imports_digest=deterministic_imports_digest(()),
        )
        old_leaves = tuple(
            tree_node_state(
                node_id=chunk.chunk_id, node_type="leaf", rel_path="main.py",
                content_hash=content_hash,
                division_plan_digest=old_plan.plan_digest,
                input_digest=leaf_input_digest(
                    rel_path="main.py", language="python", chunk=chunk,
                    unit_indexes=old_plan.unit_positions(chunk),
                    unit_count=len(old_plan.units),
                ),
                execution_identity_digest=leaf_execution_identity(
                    rel_path="main.py", content_hash=content_hash,
                    division_plan_digest=old_plan.plan_digest,
                    provider_identity=old_provider_identity, chunk=chunk,
                ),
                unit_id=None, child_ids=(), coverage_leaf_ids=(chunk.chunk_id,),
                result={
                    "description": f"v5 leaf {i}", "chunk_id": chunk.chunk_id,
                    "unit_id": chunk.unit_id,
                },
            )
            for i, chunk in enumerate(old_plan.chunks)
        )
        # A genuine v2 reducer checkpoint: its execution identity is built by
        # reduction_execution_identity while REDUCER_PROMPT_REVISION reads
        # "file-reduction-v2", so v2 is behaviorally represented in the partial.
        old_uc = old_tree.unit_consolidation_nodes[0]
        old_child_narratives = tuple(
            f"v5 leaf {i}" for i in range(len(old_plan.chunks))
        )
        assert file_division.REDUCER_PROMPT_REVISION == "file-reduction-v2"
        old_v2_reducer_identity = reduction_execution_identity(
            rel_path="main.py", content_hash=content_hash,
            division_plan_digest=old_plan.plan_digest,
            reduction_tree_digest=old_tree.tree_digest,
            provider_identity=old_provider_identity, node=old_uc,
        )
        old_reducer = tree_node_state(
            node_id=old_uc.node_id, node_type=old_uc.phase, rel_path="main.py",
            content_hash=content_hash,
            division_plan_digest=old_plan.plan_digest,
            input_digest=reduction_input_digest(
                rel_path="main.py", phase=old_uc.phase, level=old_uc.level,
                unit_id=old_uc.unit_id, child_count=len(old_uc.child_ids),
                ordered_child_narratives=old_child_narratives,
            ),
            execution_identity_digest=old_v2_reducer_identity,
            unit_id=old_uc.unit_id, child_ids=old_uc.child_ids,
            coverage_leaf_ids=old_uc.leaf_ids,
            result={"narrative": "v2 combined narrative"},
        )
        old_state = SplitTreeState(
            schema_version=SPLIT_PARTIAL_SCHEMA_VERSION, owner="codedoc-ai",
            rel_path="main.py", content_hash=content_hash,
            division_plan_digest=old_plan.plan_digest,
            reduction_tree_digest=old_tree.tree_digest,
            nodes=old_leaves + (old_reducer,),
        )
        # Before leaving the patch context: the partial genuinely carries both
        # node types and the reducer identity was generated under v2.
        assert {node.node_type for node in old_state.nodes} == {
            "leaf", "unit-consolidation",
        }
        assert old_reducer.execution_identity_digest == old_v2_reducer_identity

    # ---- Patches undone: current v6/v9/v5/v3 behavior below ---------------
    current_plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=budget,
    )
    current_tree = build_reduction_tree(
        current_plan,
        synthesis_manifest_chars=max(budget, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    current_completed_identity = record_meta.expected_large_file_identity(
        source_chars=len(source), max_chars=budget, rel_path="main.py",
        division_plan_digest=current_plan.plan_digest,
        reduction_tree_digest=current_tree.tree_digest,
        structural_mode=current_plan.structural_mode,
        imports_digest=deterministic_imports_digest(()),
    )
    assert old_completed_identity.startswith("large-file-v3:")
    assert old_completed_identity != current_completed_identity
    assert old_plan.plan_digest != current_plan.plan_digest  # packer v5 -> v6
    assert old_state.reduction_tree_digest != current_tree.tree_digest
    # The stored v2 reducer identity is genuinely stale under current v3.
    current_v3_reducer_identity = reduction_execution_identity(
        rel_path="main.py", content_hash=content_hash,
        division_plan_digest=current_plan.plan_digest,
        reduction_tree_digest=current_tree.tree_digest,
        provider_identity=old_provider_identity,
        node=current_tree.all_intermediate_nodes[0],
    )
    assert old_v2_reducer_identity != current_v3_reducer_identity

    # ---- (1) completed predecessor record -> unpaid work -----------------
    completed_record = {
        "path": "main.py", "hash": content_hash,
        "description": "documented by a v5/v8/v4/v2 build",
        "language": "python", "_analysis_revision": "file-doc-v3",
        "_analysis_mode": "single", "_large_file_identity": old_completed_identity,
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    completed_result, _ = build_pipeline_plan(
        file_map, graph, {"main.py"}, "main.py",
        {"main.py": completed_record}, [], config,
    )
    assert "main.py" not in completed_result.unchanged_rels
    assert "main.py" in completed_result.changed_rels

    # ---- (2) partial predecessor -> cross-plan carry --------------------
    monkeypatch.setattr(planning_mod, "validate_recovered_tree", _fail_if_validated)
    graph2 = DependencyGraph()
    graph2.add_file("main.py")
    carry_result, materials = build_pipeline_plan(
        file_map, graph2, {"main.py"}, "main.py", {}, [], config,
        recovered_partials={"main.py": old_state},
    )
    assert materials.carry_states["main.py"] is old_state
    assert "main.py" not in materials.tree_states
    assert materials.recovery_conflict_files == 1
    assert materials.reexecuted_nodes == 0
    old_ids = {n.node_id for n in old_state.nodes}
    # Discarded == every old paid unique node ID, INCLUDING the genuine v2
    # reducer; the chosen topology has zero overlap with current IDs.
    assert old_reducer.node_id in old_ids
    assert materials.recovery_discarded_predecessor_nodes == len(old_ids)
    assert materials.recovery_discarded_predecessor_nodes == len(old_plan.chunks) + 1
    current_ids = {c.chunk_id for c in current_plan.chunks} | {
        n.node_id for n in current_tree.all_nodes
    }
    assert materials.recovery_replacement_nodes_planned == len(current_ids)
    assert old_ids & current_ids == set()
    assert "main.py" in carry_result.division_plan_rels
    assert "main.py" not in carry_result.completed_split_reuse_rels


# ---------------------------------------------------------------------------
# 0.14.9 section 5.6.2 -- a COMPLETED split-file record stamped under either
# predecessor revision regenerates (section 9.1 items 31-33); with no partial
# state it reports zero quarantine; dependency propagation reprocesses selected
# dependents only when enabled; current-identity records still reuse.
# ---------------------------------------------------------------------------

_S3_PRED_REVS = {
    "leaf": ("LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v10"),
    "reducer": ("REDUCER_PROMPT_REVISION", "file-reduction-v3"),
}


def _s3_split_identity_kwargs(source, max_chars):
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=max_chars,
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    return dict(
        source_chars=len(source), max_chars=max_chars, rel_path="main.py",
        division_plan_digest=plan.plan_digest, reduction_tree_digest=tree.tree_digest,
        structural_mode=plan.structural_mode,
        imports_digest=deterministic_imports_digest(()),
    )


def _s3_predecessor_split_identity(monkeypatch, attr, old, kwargs):
    with monkeypatch.context() as mp:
        mp.setattr(record_meta, attr, old)
        if hasattr(file_division, attr):
            mp.setattr(file_division, attr, old)
        return record_meta.expected_large_file_identity(**kwargs)


@pytest.mark.parametrize("which", ["leaf", "reducer"])
def test_s3_completed_split_record_under_a_predecessor_revision_regenerates(
    tmp_path, monkeypatch, which
):
    """Objective 6 / item 31: a completed record stamped by a real predecessor
    (``leaf-capsule-v10`` OR ``file-reduction-v3``) identity is rejected by the
    completed identity and scheduled as changed work; with no partial state the
    quarantine count is zero -- partial-recovery quarantine accounting does not
    apply to a completed-record rerun.

    P2-3 (completed-record rerun): the reject is proven at the CALL-MANIFEST
    level, not only via ``changed_rels`` membership. The regenerated record is
    absent from ``completed_split_reuse_rels`` and the manifest carries the
    file's FULL current split call set -- one ``unit-documentation`` call per
    leaf chunk, one ``file-reduction`` call per reducer node, one
    ``file-synthesis`` call -- with no ``file-documentation`` call, no unrelated
    owner, and a canonical digest."""
    attr, old = _S3_PRED_REVS[which]
    src = tmp_path / "main.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(1000)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    max_chars = 2000
    kwargs = _s3_split_identity_kwargs(source, max_chars)
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=max_chars,
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    reducer_ids = [
        n.node_id for n in tree.unit_consolidation_nodes + tree.general_nodes
    ]
    chunk_ids = [c.chunk_id for c in plan.chunks]
    assert len(chunk_ids) >= 2 and len(reducer_ids) >= 1

    pred_identity = _s3_predecessor_split_identity(monkeypatch, attr, old, kwargs)
    current_identity = record_meta.expected_large_file_identity(**kwargs)
    assert pred_identity.startswith("large-file-v3:")
    assert pred_identity != current_identity

    record = {
        "path": "main.py", "hash": compute_file_hash(src),
        "description": "documented by a predecessor build", "language": "python",
        "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
        "_large_file_identity": pred_identity,
    }
    file_map = {
        "main.py": {"path": src, "rel_path": "main.py", "language": "python",
                    "extension": ".py"},
    }
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split", "max_content_chars": max_chars,
        "truncation_head_ratio": 0.70,
    }

    def plan_for():
        g = DependencyGraph()
        g.add_file("main.py")
        return build_pipeline_plan(
            file_map, g, {"main.py"}, "main.py", {"main.py": record}, [], config,
        )

    # Direction 1: while the constant reads the predecessor value, the identical
    # record is a valid completed record of its own release.
    with monkeypatch.context() as mp:
        mp.setattr(record_meta, attr, old)
        if hasattr(file_division, attr):
            mp.setattr(file_division, attr, old)
        under_pred, _ = plan_for()
    assert "main.py" in under_pred.unchanged_rels
    assert "main.py" not in under_pred.changed_rels

    # Direction 2: identical record under the real current revision -> scheduled
    # as changed, with NO partial-recovery quarantine (no partial state exists).
    under_cur, materials = plan_for()
    assert "main.py" not in under_cur.unchanged_rels
    assert "main.py" in under_cur.changed_rels
    assert "main.py" not in materials.tree_states
    assert sum(len(s.quarantine) for s in materials.tree_states.values()) == 0

    # P2-3: the completed-record reject is a full split re-plan, proven at the
    # manifest level. No partial state -> not a completed-split reuse.
    assert "main.py" not in under_cur.completed_split_reuse_rels
    assert "main.py" in under_cur.division_plan_rels
    manifest = build_call_manifest(
        [], sorted(under_cur.agent_rels), "single",
        division_plans=materials.division_plans,
        reduction_trees=materials.reduction_trees,
        tree_states=materials.tree_states,
    )
    by_cat: dict[str, list[str]] = {}
    for call in manifest.calls:
        by_cat.setdefault(call.category, []).append(call.owner)
    assert sorted(under_cur.agent_rels) == ["main.py"]                 # nothing else
    assert sorted(by_cat["unit-documentation"]) == sorted(chunk_ids)   # every leaf
    assert sorted(by_cat["file-reduction"]) == sorted(reducer_ids)     # every reducer
    assert by_cat["file-synthesis"] == ["main.py"]                     # the final
    assert by_cat.get("file-documentation", []) == []                  # not whole-file
    assert len(manifest.calls) == len(chunk_ids) + len(reducer_ids) + 1
    assert manifest.digest == hashlib.sha256(
        "\n".join(c.call_id for c in manifest.calls).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("propagate", [True, False])
def test_s3_stale_completed_split_record_propagation_is_mode_dependent(
    tmp_path, monkeypatch, propagate
):
    """Objective 7 / item 32: a stale completed split record enters
    ``changed_rels`` identically in both modes; a selected dependent is
    scheduled ONLY when dependency propagation is enabled.

    P2-3 (completed-record rerun): the propagation-mode difference is proven to
    be a SCHEDULING difference (``process_rels`` vs ``unchanged_rels``
    membership), NOT a planned-call difference. The split file's full current
    call set is planned identically in both modes and the current-identity
    dependent contributes NO manifest call in either mode -- with propagation on
    it is pulled into ``process_rels`` and then resolved by identical-content
    reuse, exactly as ``tests/integration/pipeline/test_planning.py``
    ::``test_I1_propagate_changes_true_reimports_updated`` establishes (a
    propagated dependent whose own content is unchanged is reused, not re-sent).
    ``codedoc/core/planning.py`` states the same in situ: "Propagated dependents
    keep normal reuse behaviour". Forcing a paid dependent call on a
    dependency-only change would be a production behaviour change and is out of
    scope here."""
    src = tmp_path / "main.py"
    dep = tmp_path / "dependent.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(1000)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    dep.write_text("import main\n", encoding="utf-8")
    max_chars = 2000
    kwargs = _s3_split_identity_kwargs(source, max_chars)
    pred_identity = _s3_predecessor_split_identity(
        monkeypatch, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v10", kwargs
    )
    plan = build_division_plan(
        rel_path="main.py", language="python", content=source,
        source_budget_chars=max_chars,
    )
    tree = build_reduction_tree(
        plan,
        synthesis_manifest_chars=max(max_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS),
        language="python",
    )
    split_node_ids = {c.chunk_id for c in plan.chunks} | {
        n.node_id for n in tree.unit_consolidation_nodes + tree.general_nodes
    }
    expected_split_calls = len(plan.chunks) + len(
        tree.unit_consolidation_nodes + tree.general_nodes
    ) + 1

    main_rec = {
        "path": "main.py", "hash": compute_file_hash(src),
        "description": "predecessor split", "language": "python",
        "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
        "_large_file_identity": pred_identity,
    }
    dep_rec = {
        "path": "dependent.py", "hash": compute_file_hash(dep),
        "description": "dependent", "language": "python",
        "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
        "_ordinary_path_identity": expected_ordinary_path_identity("dependent.py"),
    }
    file_map = {
        "main.py": {"path": src, "rel_path": "main.py", "language": "python",
                    "extension": ".py"},
        "dependent.py": {"path": dep, "rel_path": "dependent.py",
                         "language": "python", "extension": ".py"},
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    graph.add_file("dependent.py")
    graph.add_dependency("dependent.py", "main.py")   # dependent imports main
    config = {
        "propagate_changes": propagate, "max_files": 0, "analysis_mode": "single",
        "large_file_strategy": "split", "max_content_chars": max_chars,
        "truncation_head_ratio": 0.70,
    }
    result, materials = build_pipeline_plan(
        file_map, graph, {"main.py", "dependent.py"}, "main.py",
        {"main.py": main_rec, "dependent.py": dep_rec}, [], config,
    )

    # Direct identity invalidation of main.py is identical either way: it is
    # changed work and is sent to an agent in both modes.
    assert "main.py" in result.changed_rels
    assert "main.py" in result.agent_rels
    # The selected dependent is pulled into process_rels (not skipped) ONLY with
    # propagation enabled; with propagation off it stays unchanged/skipped.
    assert ("dependent.py" in result.process_rels) is propagate
    assert ("dependent.py" in result.unchanged_rels) is (not propagate)
    # ...but in NEITHER mode is the current-identity dependent sent to an agent.
    assert "dependent.py" not in result.agent_rels
    assert sorted(result.agent_rels) == ["main.py"]

    # The planned call manifest is byte-identical in both modes: the full
    # current split call set for main.py, nothing owned by dependent.py.
    manifest = build_call_manifest(
        [], sorted(result.agent_rels), "single",
        division_plans=materials.division_plans,
        reduction_trees=materials.reduction_trees,
        tree_states=materials.tree_states,
    )
    by_cat: dict[str, list[str]] = {}
    for call in manifest.calls:
        by_cat.setdefault(call.category, []).append(call.owner)
    assert len(manifest.calls) == expected_split_calls
    assert sorted(by_cat["unit-documentation"]) == sorted(
        c.chunk_id for c in plan.chunks
    )
    assert sorted(by_cat["file-reduction"]) == sorted(
        n.node_id for n in tree.unit_consolidation_nodes + tree.general_nodes
    )
    assert by_cat["file-synthesis"] == ["main.py"]
    assert by_cat.get("file-documentation", []) == []
    non_synth_owners = {
        call.owner for call in manifest.calls if call.category != "file-synthesis"
    }
    assert non_synth_owners <= split_node_ids           # no dependent.py owner
    assert "dependent.py" not in {call.owner for call in manifest.calls}
    assert manifest.digest == hashlib.sha256(
        "\n".join(c.call_id for c in manifest.calls).encode("utf-8")
    ).hexdigest()


def _s3_completed_split_tree(root, monkeypatch, *, propagate):
    """A project tree with a COMPLETED split record for ``main.py`` stamped
    under the ``leaf-capsule-v10`` predecessor identity plus a current-identity
    ``dependent.py`` importing it, ready for a real ``run_pipeline`` rerun. The
    predecessor identity is stamped through ``monkeypatch.context()`` so the
    live constant is restored before ``run_pipeline`` executes."""
    root.mkdir(parents=True, exist_ok=True)
    src = root / "main.py"
    dep = root / "dependent.py"
    source = "\n".join(f"value_{i} = {i}" for i in range(1000)) + "\n"
    src.write_text(source, encoding="utf-8", newline="")
    dep.write_text("import main\nq = main.value_0\n", encoding="utf-8")
    max_chars = 2000
    kwargs = _s3_split_identity_kwargs(source, max_chars)
    pred_identity = _s3_predecessor_split_identity(
        monkeypatch, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v10", kwargs
    )
    assert file_division.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v11"  # restored
    out = root / "docs"
    out.mkdir()
    out.joinpath("codedoc.json").write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [
            {"path": "main.py", "hash": compute_file_hash(src),
             "description": "documented by a predecessor build", "language": "python",
             "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
             "_large_file_identity": pred_identity},
            {"path": "dependent.py", "hash": compute_file_hash(dep),
             "description": "dependent", "language": "python",
             "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single",
             "_ordinary_path_identity": expected_ordinary_path_identity("dependent.py")},
        ],
    }), encoding="utf-8")
    return {
        "entry_file": "main.py", "analysis_mode": "single",
        "large_file_strategy": "split", "max_content_chars": max_chars,
        "output_dir": "docs", "documentation_scope": "all",
        "propagate_changes": propagate, "parallel_agents": False,
    }, pred_identity


class _S3CountingSmartFake(SmartFake):
    def __init__(self) -> None:
        super().__init__()
        self.completions = 0

    def complete_json(self, prompt, system=""):
        self.completions += 1
        return super().complete_json(prompt, system)


def test_s3_completed_split_record_rerun_paid_work_is_propagation_mode_independent(
    tmp_path, monkeypatch
):
    """P2-3 (completed-record rerun) end to end through a real fake-provider
    pipeline: a completed split record stamped ``leaf-capsule-v10`` is rejected
    and fully re-documented; a current-identity ``dependent.py`` importing it is
    REUSED with propagation on and SKIPPED with propagation off -- never
    re-sent to the provider. The provider call count for the rerun is IDENTICAL
    in both propagation modes: propagation changes which files are examined, not
    how much paid work runs (mirrors
    ``tests/integration/pipeline/test_planning.py``
    ::``test_I1_propagate_changes_true_reimports_updated``)."""
    completions: dict[bool, int] = {}
    for propagate in (True, False):
        root = tmp_path / f"prop_{propagate}"
        config, pred_identity = _s3_completed_split_tree(
            root, monkeypatch, propagate=propagate
        )
        provider = _S3CountingSmartFake()
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider", lambda _c, _p=provider: _p
        )
        stats = run_pipeline(root, config)
        completions[propagate] = provider.completions

        assert stats["failed"] == 0
        # main.py: the predecessor split identity was rejected and regenerated.
        records = records_by_path(
            read_codedoc_document(root / "docs" / "codedoc.json")
        )
        main_rec = records["main.py"]
        assert main_rec["_large_file_identity"] != pred_identity
        assert main_rec["_large_file_identity"].startswith("large-file-v3:")
        assert main_rec["_analysis_revision"] == ANALYSIS_REVISION
        # dependent.py: never re-sent. Reused (in process_rels) with propagation
        # on; skipped (never entered process_rels) with propagation off.
        assert stats.get("reused", 0) == (1 if propagate else 0)
        assert stats.get("skipped", 0) == (0 if propagate else 1)

    # The paid work is the same either way -- propagation is a scheduling
    # concern, not a call-count one.
    assert completions[True] == completions[False]
    assert completions[True] > 0


def test_s3_current_identity_split_ordinary_and_truncate_records_still_reuse(tmp_path):
    """Objective 8 / item 33: the two advances do not over-invalidate. A
    current-identity completed split record, a current ordinary record, and a
    current truncate record are all still reused unpaid under v11/v4, while the
    two advanced constants read their new values and a representative slice of
    section 5.6.3 is unchanged by value (the full slice is asserted in
    ``tests/unit/agents/test_response_correction.py``)."""
    assert file_division.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v11"
    assert file_division.REDUCER_PROMPT_REVISION == "file-reduction-v4"
    assert record_meta.ANALYSIS_REVISION == "file-doc-v4"
    assert file_division.FINAL_SYNTHESIS_REVISION == "file-synthesis-v4"
    assert file_division.REDUCTION_PACKING_REVISION == "reduction-packing-v5"
    assert file_division.PACKER_SCHEMA_REVISION == "division-packer-v6"
    assert file_division.LEAF_INPUT_DIGEST_REVISION == "leaf-input-v1"
    assert file_division.REDUCTION_INPUT_DIGEST_REVISION == "reduction-input-v1"
    assert file_division.FINAL_INPUT_DIGEST_REVISION == "final-input-v1"

    assert "main.py" in _split_plan(tmp_path).unchanged_rels
    assert "main.py" in _small_plan(tmp_path, max_chars=1000).unchanged_rels
    assert "main.py" in _oversized_plan(
        tmp_path, "truncate-v1:max=1000:head=0.7000"
    ).unchanged_rels
