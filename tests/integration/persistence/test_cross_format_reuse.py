"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
import json
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_scenarios import write_existing_md
import pytest
from tests.support.recovery_rate_limit_runs import _run
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.record_meta import CACHE_IDENTITY_KEYS
from codedoc.core.resume import (
    RECOVERY_FILENAME,
    _load_existing_file_docs,
)
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.providers import SmartFake
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _first_run
from tests.support.cross_format_runs import _forbid_provider


@pytest.mark.parametrize(
    ("first_format", "second_format", "first_name", "second_name"),
    [
        ("json", "md", "codedoc.json", "codedoc.md"),
        ("md", "json", "codedoc.md", "codedoc.json"),
    ],
)
def test_split_fields_survive_provider_free_cross_format_conversion(
    tmp_path, monkeypatch, first_format, second_format, first_name, second_name
):
    source = "\n".join(f"value_{index} = {index}" for index in range(220)) + "\n"
    (tmp_path / "main.py").write_text(source, encoding="utf-8", newline="")
    first_config = {
        **_config(first_format),
        "large_file_strategy": "split",
        "max_content_chars": 2000,
    }
    provider = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: provider)
    first_stats = run_pipeline(tmp_path, first_config)
    assert first_stats["checked"] == 1

    _forbid_provider(monkeypatch)
    second_stats = run_pipeline(
        tmp_path,
        {
            **_config(second_format),
            "large_file_strategy": "split",
            "max_content_chars": 2000,
        },
    )

    output_dir = tmp_path / "docs"
    first_record = records_by_path(
        read_codedoc_document(output_dir / first_name)
    )["main.py"]
    second_record = records_by_path(
        read_codedoc_document(output_dir / second_name)
    )["main.py"]
    assert second_stats["checked"] == 0
    # Internal division/reduction content is never public (D9/D14); only the
    # final documentation and its provider-free large-file identity round-trip.
    assert "division" not in first_record
    assert "division" not in second_record
    assert "documentation_units" not in first_record
    assert "documentation_units" not in second_record
    assert first_record["description"] == second_record["description"]
    assert first_record["_large_file_identity"] == second_record[
        "_large_file_identity"
    ]
    assert not (output_dir / RECOVERY_FILENAME).exists()


def test_stale_0_14_2_split_identity_reprocesses_while_ordinary_sibling_converts_free(
    tmp_path, monkeypatch
):
    """Section 6: the cross-format matrix must treat a preserved `0.14.2`
    completed split record (bound to the retired `leaf-capsule-v5` /
    150,656-char payload) as stale under the current v6 identity and
    reprocess it, while an ordinary compatible sibling in the very same
    conversion still reuses with zero provider calls."""
    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.core.record_meta import ANALYSIS_REVISION, expected_large_file_identity

    large_source = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
    (tmp_path / "main.py").write_text(large_source, encoding="utf-8", newline="")
    (tmp_path / "helper.py").write_text("def helper(): pass\n", encoding="utf-8")
    main_hash = compute_file_hash(tmp_path / "main.py")
    helper_hash = compute_file_hash(tmp_path / "helper.py")

    current_identity = expected_large_file_identity(
        source_chars=len(large_source),
        max_chars=2000,
        rel_path="main.py",
        division_plan_digest="division-plan:" + "1" * 64,
        reduction_tree_digest="reduction-tree:" + "2" * 64,
        structural_mode="syntax",
        imports_digest="imports:" + "3" * 64,
    )
    # Reproduce exactly what a preserved 0.14.2 record's identity payload
    # would hash to: identical inputs, but leaf_capsule/leaf_capsule_chars
    # reverted to their retired 0.14.2 values (section 5).
    import hashlib
    import json as _json

    from codedoc.core.file_division import (
        FINAL_SYNTHESIS_REVISION,
        LEDGER_SCHEMA_REVISION,
        MAX_ATOMS_PER_FILE,
        MAX_CHUNKS_PER_FILE,
        MAX_KNOWN_SYMBOLS_PER_CHUNK,
        MAX_LEAF_PROMPT_METADATA_CHARS,
        MAX_LEDGER_SYNOPSIS_CHARS,
        MAX_REDUCTION_CAPSULE_CANONICAL_CHARS,
        MAX_REDUCTION_NARRATIVE_CHARS,
        MAX_REDUCTION_TREE_DEPTH,
        MAX_SYMBOLS_PER_FILE,
        MAX_UNITS_PER_FILE,
        PACKER_SCHEMA_REVISION,
        REDUCER_PROMPT_REVISION,
        REDUCTION_CAPSULE_SCHEMA_REVISION,
        REDUCTION_ENVELOPE_OVERHEAD_CHARS,
        REDUCTION_PACKING_REVISION,
        STRUCTURE_SCHEMA_REVISION,
        UNIT_SCHEMA_REVISION,
    )
    from codedoc.parser.tree_sitter_structure import PARSER_PACKAGE_VERSION

    stale_payload = {
        "revision": "large-file-identity-v3",
        "requested_strategy": "split",
        "effective_strategy": "split",
        "source_budget": 2000,
        "path": "main.py",
        "division_plan_digest": "division-plan:" + "1" * 64,
        "reduction_tree_digest": "reduction-tree:" + "2" * 64,
        "structural_mode": "syntax",
        "imports_digest": "imports:" + "3" * 64,
        "parser_package_version": PARSER_PACKAGE_VERSION,
        "grammar_availability_mode": "bundled-grammar-or-complete-lexical-fallback-v1",
        "bounds": {
            "atoms": MAX_ATOMS_PER_FILE, "symbols": MAX_SYMBOLS_PER_FILE,
            "units": MAX_UNITS_PER_FILE, "chunks": MAX_CHUNKS_PER_FILE,
            "known_symbols_per_chunk": MAX_KNOWN_SYMBOLS_PER_CHUNK,
            "leaf_prompt_metadata_chars": MAX_LEAF_PROMPT_METADATA_CHARS,
        },
        "reduction_bounds": {
            "leaf_capsule_chars": 150656,
            "reduction_capsule_chars": MAX_REDUCTION_CAPSULE_CANONICAL_CHARS,
            "reduction_envelope_overhead": REDUCTION_ENVELOPE_OVERHEAD_CHARS,
            "final_narrative_chars": MAX_REDUCTION_NARRATIVE_CHARS,
            "final_ledger_synopsis_chars": MAX_LEDGER_SYNOPSIS_CHARS,
            "final_envelope": "exact-worst-case-v2",
            "max_tree_depth": MAX_REDUCTION_TREE_DEPTH,
        },
        "revisions": {
            "structure": STRUCTURE_SCHEMA_REVISION, "units": UNIT_SCHEMA_REVISION,
            "packer": PACKER_SCHEMA_REVISION, "leaf_capsule": "leaf-capsule-v5",
            "ledger": LEDGER_SCHEMA_REVISION,
            "reduction_capsule": REDUCTION_CAPSULE_SCHEMA_REVISION,
            "reduction_packing": REDUCTION_PACKING_REVISION,
            "reducer_prompt": REDUCER_PROMPT_REVISION,
            "final_synthesis": FINAL_SYNTHESIS_REVISION,
        },
    }
    stale_identity = "large-file-v3:" + hashlib.sha256(
        _json.dumps(stale_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert stale_identity != current_identity

    records = [
        {
            "hash": helper_hash,
            "file_path": "helper.py",
            "language": "python",
            "documentation": {
                "file_path": "helper.py", "language": "python", "description": "Helper.",
            },
            **_PRIOR_RUN_IDENTITY,
        },
        {
            "hash": main_hash,
            "file_path": "main.py",
            "language": "python",
            "documentation": {
                "file_path": "main.py", "language": "python",
                "description": "0.14.2-produced split documentation.",
                "_analysis_revision": ANALYSIS_REVISION,
                "_analysis_mode": "single",
                "_large_file_identity": stale_identity,
            },
        },
    ]
    docs_dir = tmp_path / "docs"
    write_project_outputs(records, {"checked": 2, "failed": 0, "skipped": 0}, docs_dir, output_format="json")

    provider = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "helper.py",
            "documentation_scope": "all",
            "output_dir": "docs",
            "output_format": "md",
            "large_file_strategy": "split",
            "max_content_chars": 2000,
            "parallel_agents": False,
            "propagate_changes": False,
        },
    )

    # helper.py: ordinary compatible reuse, zero calls (skipped as
    # unchanged). main.py: stale split identity, reprocessed under the
    # current v6 leaf identity.
    assert stats["checked"] == 1
    assert stats["skipped"] == 1
    assert provider.doc_calls > 0
    record = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert record["_large_file_identity"] != stale_identity


def test_cross_format_sibling_is_used_for_zero_call_conversion(tmp_path, monkeypatch):
    """--output docs/claude.json after a previous --format md run that wrote
    docs/claude.md must read the entry from claude.md and resume without error."""
    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.pipeline import run_pipeline

    src = tmp_path / "main.py"
    src.write_text("def main(): pass\n", encoding="utf-8")
    real_hash = compute_file_hash(src)

    docs_dir = tmp_path / "docs"

    # Simulate a previous --format md run that wrote docs/claude.md
    records = [
        {
            "hash": real_hash,
            "file_path": "main.py",
            "language": "python",
            "documentation": {
                "file_path": "main.py",
                "language": "python",
                "description": "Entry module.",
            },
            **_PRIOR_RUN_IDENTITY,
        }
    ]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        docs_dir,
        output_format="md",
        entry_file="main.py",
        md_filename="claude.md",
    )

    assert (docs_dir / "claude.md").exists()
    assert not (docs_dir / "claude.json").exists()

    calls = {"count": 0}

    class Provider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            calls["count"] += 1
            return json.dumps({
                "description": "Fresh exact-target analysis.",
                "role_in_system": "entry",
                "functions": [],
                "classes": [],
                "exports": [],
                "key_concepts": [],
                "usage_example": "",
                "dependencies_analysis": {"internal": [], "external": []},
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    def fail_if_llm_used(config):
        raise AssertionError("LLM must not be called — file is unchanged")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: Provider())

    # Now run with --output docs/claude.json — should find claude.md as sibling
    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs/claude.json",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert calls["count"] == 0
    assert (docs_dir / "claude.json").exists()
    json_content = (docs_dir / "claude.json").read_text(encoding="utf-8")
    assert "Entry module." in json_content
    assert "Fresh exact-target analysis." not in json_content

def test_D3_named_cross_format_reuse_json_after_md(tmp_path, monkeypatch):
    """A named JSON target reuses only its same-stem Markdown counterpart."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "docs" / "claude.md",
                      compute_file_hash(src), "Cross format cached.")
    assert not (tmp_path / "docs" / "claude.json").exists()
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"output_dir": "docs/claude.json",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "docs" / "claude.json").exists()

def test_3_named_md_output_uses_json_sibling(tmp_path, monkeypatch):
    """Test 3: --output docs/report.md → docs/report.json (not docs/codedoc.json)."""
    (tmp_path / "main.py").write_text("x=1\n")
    _run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        output_dir="docs/report.md",
    )

    md = tmp_path / "docs" / "report.md"
    sibling = tmp_path / "docs" / "report.json"
    default_json = tmp_path / "docs" / "codedoc.json"

    assert md.exists(), "report.md must exist"
    # After clean success the JSON sibling is removed
    assert not sibling.exists(), "report.json must be removed after clean MD write"
    assert not default_json.exists(), "docs/codedoc.json must NOT be created"

@pytest.mark.parametrize(
    ("first_format", "second_format", "first_name", "second_name"),
    [
        ("md", "json", "codedoc.md", "codedoc.json"),
        ("json", "md", "codedoc.json", "codedoc.md"),
    ],
)
def test_unchanged_format_switch_converts_without_provider(
    tmp_path, monkeypatch, first_format, second_format, first_name, second_name
):
    (tmp_path / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    first_fake, _ = _first_run(tmp_path, monkeypatch, first_format)
    assert first_fake.doc_calls == 1

    out = tmp_path / "docs"
    stable_sibling = out / first_name
    before = stable_sibling.read_bytes()
    _forbid_provider(monkeypatch)
    stats = run_pipeline(tmp_path, _config(second_format))

    assert (out / second_name).exists()
    assert stable_sibling.read_bytes() == before
    assert stats["checked"] == 0
    assert stats["documentation_calls_attempted"] == 0
    assert stats["attempted_calls"] == 0
    assert not (out / RECOVERY_FILENAME).exists()

    old = records_by_path(read_codedoc_document(stable_sibling))["main.py"]
    new = records_by_path(read_codedoc_document(out / second_name))["main.py"]
    assert old["hash"] == new["hash"]
    for key in CACHE_IDENTITY_KEYS:
        assert old.get(key) == new.get(key)

def test_format_switch_processes_only_changed_file(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("y = 1\n", encoding="utf-8")
    first_fake, _ = _first_run(tmp_path, monkeypatch, "json")
    assert first_fake.doc_calls == 2

    (tmp_path / "helper.py").write_text("y = 2\n", encoding="utf-8")
    second_fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: second_fake)
    stats = run_pipeline(tmp_path, _config("md"))

    assert second_fake.doc_calls == 1
    assert stats["checked"] == 1
    assert stats["skipped"] == 1
    assert (tmp_path / "docs" / "codedoc.md").exists()

def test_existing_selected_target_is_authoritative_and_sibling_is_not_read(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    foreign = tmp_path / "docs" / "codedoc.md"
    foreign.write_text("not a codedoc document", encoding="utf-8")

    _forbid_provider(monkeypatch)
    stats = run_pipeline(tmp_path, _config("json"))
    assert stats["checked"] == 0
    assert foreign.read_text(encoding="utf-8") == "not a codedoc document"

def test_missing_both_candidates_returns_no_existing_records(tmp_path):
    assert _load_existing_file_docs(
        tmp_path / "missing.json", tmp_path / "missing.md", "json"
    ) == {}
    assert _load_existing_file_docs(
        tmp_path / "missing.json", tmp_path / "missing.md", "md"
    ) == {}

@pytest.mark.parametrize(
    ("first_output", "second_output", "first_name", "second_name"),
    [
        ("docs/report.md", "docs/report.json", "report.md", "report.json"),
        ("docs/report.json", "docs/report.md", "report.json", "report.md"),
    ],
)
def test_named_output_switch_uses_same_stem_sibling(
    tmp_path, monkeypatch, first_output, second_output, first_name, second_name
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, first_name.rsplit(".", 1)[1], first_output)
    unrelated = tmp_path / "docs" / (
        "codedoc.json" if second_name.endswith(".json") else "codedoc.md"
    )
    unrelated.write_text("foreign unrelated file", encoding="utf-8")

    _forbid_provider(monkeypatch)
    stats = run_pipeline(
        tmp_path, _config(second_name.rsplit(".", 1)[1], second_output)
    )
    assert stats["checked"] == 0
    assert (tmp_path / "docs" / first_name).exists()
    assert (tmp_path / "docs" / second_name).exists()
    assert unrelated.read_text(encoding="utf-8") == "foreign unrelated file"

def test_both_mode_cross_document_hash_conflict_blocks(tmp_path, monkeypatch):
    """A both-mode JSON/Markdown pair whose per-file content hash disagrees is a
    deterministic pre-provider conflict, not a silently-picked-by-mtime target."""
    import json as json_mod

    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "output_format": "both", "propagate_changes": False},
    )
    output = tmp_path / "codedoc"
    json_target = output / "codedoc.json"
    md_target = output / "codedoc.md"
    assert json_target.exists() and md_target.exists()

    doc = json_mod.loads(json_target.read_text(encoding="utf-8"))
    doc["files"][0]["hash"] = "0" * 64
    json_target.write_text(json_mod.dumps(doc), encoding="utf-8")

    with pytest.raises(ConfigError, match="content hash"):
        _load_existing_file_docs(json_target, md_target, "both")

def test_both_mode_cross_document_entry_conflict_blocks(tmp_path, monkeypatch):
    """A both-mode pair whose recorded entry file disagrees is a deterministic
    pre-provider conflict."""
    import json as json_mod

    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "output_format": "both", "propagate_changes": False},
    )
    output = tmp_path / "codedoc"
    json_target = output / "codedoc.json"
    md_target = output / "codedoc.md"

    doc = json_mod.loads(json_target.read_text(encoding="utf-8"))
    doc["last_run"]["entry_file"] = "other.py"
    json_target.write_text(json_mod.dumps(doc), encoding="utf-8")

    with pytest.raises(ConfigError, match="entry file"):
        _load_existing_file_docs(json_target, md_target, "both")
