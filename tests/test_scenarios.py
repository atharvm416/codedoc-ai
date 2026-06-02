"""
Comprehensive scenario validation for codedoc 0.7.0.
Every meaningful CLI / API combination is exercised here.
Results are printed as a pass/fail table.
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_provider(description="Documented."):
    """Return a provider instance whose complete_json always succeeds."""
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": description,
                    "role_in_system": "test role",
                    "key_concepts": [],
                    "usage_example": "",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({
                    "dependencies_analysis": {
                        "internal": [], "external": [],
                        "dependency_refs": [], "catalog_updates": [],
                        "usage_notes": [], "warnings": [],
                    }
                })
            return _json.dumps({
                "description": description,
                "role_in_system": "test role",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


def patch_provider(monkeypatch, description="Documented."):
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: make_fake_provider(description),
    )


def no_llm(monkeypatch):
    def fail(_config):
        raise AssertionError("LLM must not be called in this scenario")
    monkeypatch.setattr("codedoc.pipeline.create_provider", fail)


def md_meta(md_path: Path) -> dict:
    content = md_path.read_text(encoding="utf-8")
    m = re.search(r"<!-- codedoc-ai: (\{.*?\}) -->", content, re.DOTALL)
    assert m, f"No metadata comment in {md_path}"
    return json.loads(m.group(1))


def write_existing_json(path: Path, file_hash: str, description: str = "Cached."):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "files": [{"path": "main.py", "hash": file_hash,
                   "language": "python", "description": description}],
    }), encoding="utf-8")


def write_existing_md(path: Path, file_hash: str, description: str = "Cached."):
    """Write a 0.7.0-format MD file with file_hashes in the metadata comment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({
        "entry_file": "main.py",
        "schema_version": "1.4",
        "generated_at": "2026-05-24T00:00:00+00:00",
        "file_hashes": {"main.py": file_hash},
    })
    path.write_text(
        f"<!-- codedoc-ai: {meta} -->\n"
        f"# codedoc project documentation\n\n"
        f"## Files\n\n"
        f"### main.py\n\n"
        f"**Language:** python  \n\n"
        f"**Description:** {description}\n\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Group A — First run (no existing docs)
# ---------------------------------------------------------------------------

def test_A1_first_run_default_output(tmp_path, monkeypatch):
    """A1: first run, no flags → writes codedoc/codedoc.json only."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "propagate_changes": False,
                                     "parallel_agents": False})
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert not (tmp_path / "codedoc" / "codedoc.md").exists()
    assert stats["output_files"] == [str(tmp_path / "codedoc" / "codedoc.json")]


def test_A2_first_run_format_md(tmp_path, monkeypatch):
    """A2: first run with --format md → writes codedoc/codedoc.md only."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    assert stats["output_files"] == [str(tmp_path / "codedoc" / "codedoc.md")]


def test_A3_first_run_format_both(tmp_path, monkeypatch):
    """A3: first run with --format both → writes codedoc.json AND codedoc.md."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "both",
                                     "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert len(stats["output_files"]) == 2


def test_A4_first_run_custom_directory(tmp_path, monkeypatch):
    """A4: --output custom_dir → writes custom_dir/codedoc.json."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "my_docs",
                             "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "my_docs" / "codedoc.json").exists()
    assert not (tmp_path / "my_docs" / "codedoc.md").exists()


def test_A5_first_run_named_json_file(tmp_path, monkeypatch):
    """A5: --output docs/report.json → writes docs/report.json."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "docs/report.json",
                             "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "docs" / "report.json").exists()
    assert not (tmp_path / "docs" / "codedoc.json").exists()


def test_A6_first_run_named_md_file(tmp_path, monkeypatch):
    """A6: --output docs/report.md → writes docs/report.md only."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "docs/report.md",
                             "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "docs" / "report.md").exists()
    assert not (tmp_path / "docs" / "report.json").exists()


def test_A7_unsupported_extension_raises_error(tmp_path):
    """A7: --output report.txt → ConfigError (unsupported extension)."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "docs/report.txt"})
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "txt" in str(e).lower() or "unsupported" in str(e).lower()


def test_A8_no_entry_no_docs_proceeds_to_auto_detection(tmp_path):
    """A8 (0.8.1): no --entry, no existing docs → pipeline must NOT raise early.
    _resolve_entry_and_docs() returns quietly; detect_entry_file() handles
    auto-detection at scan time.  The old 'No entry point specified' error
    was removed in 0.8.1 so first runs work without an explicit --entry flag.
    """
    from codedoc.core.loader import load_config
    from codedoc.pipeline import _resolve_entry_and_docs

    (tmp_path / "some_file.py").write_text("x=1\n")
    config = load_config(tmp_path, {"output_dir": "docs_output", "output_format": "json"})
    config["entry_file"] = None

    # Must NOT raise — pipeline proceeds to detect_entry_file() / process-all-files fallback
    _resolve_entry_and_docs(tmp_path, config)
    assert config.get("entry_file") is None, (
        "entry_file must remain None so detect_entry_file() can attempt auto-detection"
    )


def test_A9_format_both_named_file_raises_error(tmp_path):
    """A9: --format both --output report.md → ConfigError (not a directory)."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"entry_file": "main.py",
                                 "output_dir": "docs/report.md", "output_format": "both"})
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "both" in str(e).lower()


def test_A10_format_both_named_json_raises_error(tmp_path):
    """A10: --format both --output report.json → ConfigError (not a directory)."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"entry_file": "main.py",
                                 "output_dir": "docs/report.json", "output_format": "both"})
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "both" in str(e).lower()


def test_A11_conflicting_format_extension_warns_uses_extension(tmp_path, monkeypatch, caplog):
    """A11: --output report.json --format md → warning logged, JSON is written (extension wins)."""
    import logging
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    with caplog.at_level(logging.WARNING):
        run_pipeline(tmp_path, {"entry_file": "main.py",
                                 "output_dir": "docs/report.json",
                                 "output_format": "md",
                                 "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "docs" / "report.json").exists()
    assert any("implies format" in r.message or "takes precedence" in r.message
               for r in caplog.records)


# ---------------------------------------------------------------------------
# Group B — Incremental / subsequent runs (JSON state)
# ---------------------------------------------------------------------------

def test_B1_second_run_json_no_changes_skips_llm(tmp_path, monkeypatch):
    """B1: second run, --format json, file unchanged → LLM not called, JSON rewritten."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Cached content.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    out = (tmp_path / "codedoc" / "codedoc.json").read_text()
    assert "Cached content." in out


def test_B2_second_run_json_file_changed_reprocesses(tmp_path, monkeypatch):
    """B2: second run, file changed → only changed file re-processed."""
    patch_provider(monkeypatch, "Fresh content.")
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        "stale_hash_that_will_mismatch", "Old content.")
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats["checked"] == 1
    out = (tmp_path / "codedoc" / "codedoc.json").read_text()
    assert "Fresh content." in out


def test_B3_second_run_md_reads_json_state(tmp_path, monkeypatch):
    """B3: --format md run, but codedoc.json exists from a previous run → reads JSON for hashes,
    writes only MD, JSON still present."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "From JSON cache.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert "From JSON cache." in (tmp_path / "codedoc" / "codedoc.md").read_text()
    # JSON is NOT removed (format switch doesn't delete other format)
    assert (tmp_path / "codedoc" / "codedoc.json").exists()


def test_B4_second_run_md_only_state_incremental(tmp_path, monkeypatch):
    """B4: --format md, only codedoc.md exists with file_hashes → reads hashes from MD,
    skips unchanged file, writes MD only."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "codedoc" / "codedoc.md",
                      compute_file_hash(src), "From MD cache.")
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()


def test_B5_second_run_old_md_no_hashes_reprocesses_once(tmp_path, monkeypatch):
    """B5: --format md, MD exists but no file_hashes (pre-0.7.0 format) → all files
    re-processed, new MD has file_hashes embedded."""
    patch_provider(monkeypatch, "Re-documented.")
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    # Old-format MD without file_hashes in metadata
    old_meta = json.dumps({"entry_file": "main.py", "schema_version": "1.3",
                            "generated_at": "2026-05-01T00:00:00"})
    (tmp_path / "codedoc").mkdir()
    (tmp_path / "codedoc" / "codedoc.md").write_text(
        f"<!-- codedoc-ai: {old_meta} -->\n# codedoc\n\n## Files\n\n### main.py\n\n"
        "**Language:** python  \n\n**Description:** Old doc.\n\n",
        encoding="utf-8",
    )
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats["checked"] == 1  # re-processed because no hashes
    new_md = (tmp_path / "codedoc" / "codedoc.md").read_text()
    meta = md_meta(tmp_path / "codedoc" / "codedoc.md")
    assert "file_hashes" in meta
    assert "main.py" in meta["file_hashes"]


def test_B6_second_run_format_both_reads_json(tmp_path, monkeypatch):
    """B6: --format both on run 2, codedoc.json exists → reads JSON, writes both files."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Both cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "both",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert len(stats["output_files"]) == 2


# ---------------------------------------------------------------------------
# Group C — Format switching
# ---------------------------------------------------------------------------

def test_C1_switch_json_to_md(tmp_path, monkeypatch):
    """C1: have codedoc.json, run --format md → reads JSON for hashes, writes MD,
    JSON stays (not deleted)."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "JSON switch.")
    no_llm(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                             "propagate_changes": False})
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert (tmp_path / "codedoc" / "codedoc.json").exists()  # NOT deleted


def test_C2_switch_md_to_json(tmp_path, monkeypatch):
    """C2: have codedoc.md (with hashes), run --format json → reads MD hashes,
    writes JSON, MD stays."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "codedoc" / "codedoc.md",
                      compute_file_hash(src), "MD switch.")
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()  # NOT deleted


def test_C3_switch_both_to_json(tmp_path, monkeypatch):
    """C3: have both codedoc.json + codedoc.md, run --format json → reads JSON, MD stays."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Both to JSON.")
    write_existing_md(tmp_path / "codedoc" / "codedoc.md",
                      compute_file_hash(src), "Both to JSON.")
    no_llm(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                             "propagate_changes": False})
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()


def test_C4_switch_both_to_md(tmp_path, monkeypatch):
    """C4: have both files, run --format md → reads JSON for hashes (preferred), writes MD."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Both to MD.")
    no_llm(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                             "propagate_changes": False})
    assert (tmp_path / "codedoc" / "codedoc.md").exists()


# ---------------------------------------------------------------------------
# Group D — Custom named files
# ---------------------------------------------------------------------------

def test_D1_custom_json_incremental(tmp_path, monkeypatch):
    """D1: --output docs/api.json on run 2, api.json exists → incremental, writes api.json."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "docs" / "api.json",
                        compute_file_hash(src), "API docs cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py",
                                     "output_dir": "docs/api.json",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "docs" / "api.json").exists()


def test_D2_custom_md_incremental(tmp_path, monkeypatch):
    """D2: --output docs/api.md on run 2, api.md exists with hashes → incremental, writes api.md."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "docs" / "api.md",
                      compute_file_hash(src), "API md cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py",
                                     "output_dir": "docs/api.md",
                                     "propagate_changes": False})
    assert stats["checked"] == 0


def test_D3_cross_format_resume_json_after_md(tmp_path, monkeypatch):
    """D3: --output docs/claude.json, only docs/claude.md exists → reads entry + hashes
    from claude.md, writes claude.json (cross-format resume)."""
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


# ---------------------------------------------------------------------------
# Group E — Entry resolution
# ---------------------------------------------------------------------------

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
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"output_format": "json"})
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass


def test_E4_entry_unsupported_extension_warns_all_files(tmp_path, monkeypatch, caplog):
    """E4: --entry entry.txt (exists, unsupported ext) → warning logged, all files documented."""
    import logging
    patch_provider(monkeypatch)
    (tmp_path / "other.py").write_text("x=1\n")
    (tmp_path / "entry.txt").write_text("entrypoint\n")
    from codedoc.pipeline import run_pipeline
    with caplog.at_level(logging.WARNING, logger="codedoc.pipeline"):
        stats = run_pipeline(tmp_path, {"entry_file": "entry.txt",
                                         "output_format": "json",
                                         "propagate_changes": False,
                                         "parallel_agents": False})
    assert stats["checked"] >= 1
    assert any("not found in the scanned file set" in r.message for r in caplog.records)


def test_E5_entry_missing_file_all_files_documented(tmp_path, monkeypatch):
    """E5: --entry nonexistent.py → scanner warns, detect_entry_file returns None,
    pipeline falls back to documenting all files."""
    patch_provider(monkeypatch)
    (tmp_path / "other.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "nonexistent.py",
                                     "output_format": "json",
                                     "propagate_changes": False,
                                     "parallel_agents": False})
    # Should not raise — documents all found files
    assert stats["checked"] >= 1


# ---------------------------------------------------------------------------
# Group F — MD metadata correctness
# ---------------------------------------------------------------------------

def test_F1_md_metadata_contains_file_hashes(tmp_path, monkeypatch):
    """F1: written MD output contains file_hashes in the codedoc-ai comment."""
    patch_provider(monkeypatch)
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    real_hash = compute_file_hash(src)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                             "propagate_changes": False, "parallel_agents": False})
    meta = md_meta(tmp_path / "codedoc" / "codedoc.md")
    assert "file_hashes" in meta
    assert meta["file_hashes"].get("main.py") == real_hash


def test_F2_md_metadata_contains_entry_file(tmp_path, monkeypatch):
    """F2: written MD metadata comment contains entry_file."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                             "propagate_changes": False, "parallel_agents": False})
    meta = md_meta(tmp_path / "codedoc" / "codedoc.md")
    assert meta.get("entry_file") == "main.py"


def test_F3_json_metadata_contains_entry_file(tmp_path, monkeypatch):
    """F3: written JSON _codedoc block contains entry_file."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                             "propagate_changes": False, "parallel_agents": False})
    data = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text())
    assert data["_codedoc"]["entry_file"] == "main.py"


# ---------------------------------------------------------------------------
# Group G — Legacy migration
# ---------------------------------------------------------------------------

def test_G1_legacy_db_deleted_on_run(tmp_path, monkeypatch):
    """G1: codedoc_db.json in output dir is deleted automatically on any run."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    legacy = out / "codedoc_db.json"
    legacy.write_text('{"files": {}}', encoding="utf-8")
    write_existing_json(out / "codedoc.json", compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                             "propagate_changes": False})
    assert not legacy.exists()


# ---------------------------------------------------------------------------
# Group H — Content-based deduplication
# ---------------------------------------------------------------------------

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
            {"path": "main.py",     "hash": main_hash,   "language": "python", "description": "Entry."},
            {"path": "helper_a.py", "hash": shared_hash, "language": "python", "description": "Shared helper."},
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


# ---------------------------------------------------------------------------
# Group I — Propagate changes
# ---------------------------------------------------------------------------

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
             "description": "Old A."},
            {"path": "b.py", "hash": b_hash, "language": "python",
             "description": "Old B."},
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
            {"path": "a.py", "hash": "old_a_hash", "language": "python", "description": "Old A."},
            {"path": "b.py", "hash": b_hash, "language": "python", "description": "Old B."},
        ],
    }), encoding="utf-8")

    patch_provider(monkeypatch, "Updated A only.")
    stats = run_pipeline(tmp_path, {"entry_file": "b.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats["checked"] == 1  # only a.py


# ---------------------------------------------------------------------------
# Group J — Output file count guarantees
# ---------------------------------------------------------------------------

def test_J1_format_json_exactly_one_output_file(tmp_path, monkeypatch):
    """J1: --format json → exactly 1 output file path in stats."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert len(stats["output_files"]) == 1
    assert stats["output_files"][0].endswith(".json")


def test_J2_format_md_exactly_one_output_file(tmp_path, monkeypatch):
    """J2: --format md → exactly 1 output file path in stats."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False, "parallel_agents": False})
    assert len(stats["output_files"]) == 1
    assert stats["output_files"][0].endswith(".md")


def test_J3_format_both_exactly_two_output_files(tmp_path, monkeypatch):
    """J3: --format both → exactly 2 output file paths in stats."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "both",
                                     "propagate_changes": False, "parallel_agents": False})
    assert len(stats["output_files"]) == 2
    exts = {Path(p).suffix for p in stats["output_files"]}
    assert exts == {".json", ".md"}


def test_J4_no_codedoc_db_ever_written(tmp_path, monkeypatch):
    """J4: after any run, codedoc_db.json must never exist in the output dir."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    for fmt in ("json", "md", "both"):
        run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": fmt,
                                 "propagate_changes": False, "parallel_agents": False})
        assert not (tmp_path / "codedoc" / "codedoc_db.json").exists(), \
            f"codedoc_db.json found after --format {fmt}"


# ---------------------------------------------------------------------------
# Group K — Recovery / ownership hardening (0.7.2 fixes)
# ---------------------------------------------------------------------------

def _codedoc_json(path: Path, records: list[dict], status: str | None = None) -> None:
    meta: dict = {
        "entry_file": "main.py",
        "schema_version": "1.4",
        "generated_at": "2026-05-30T00:00:00+00:00",
    }
    if status:
        meta["status"] = status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_codedoc": meta, "files": records}), encoding="utf-8")


def test_K1_safewriter_raises_on_malformed_target(tmp_path):
    """K1 (Scenario R): SafeWriter refuses to overwrite a malformed target file."""
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ConfigError

    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "codedoc.json").write_text("{ not valid json", encoding="utf-8")

    sw = SafeWriter(out / "codedoc.json", "json", "main.py", {})
    try:
        sw.load()
        assert False, "load() should have raised on a malformed target file"
    except ConfigError:
        pass


def test_K2_safewriter_raises_on_foreign_target(tmp_path):
    """K2 (Scenario P): SafeWriter refuses a valid-JSON file with no _codedoc block."""
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ConfigError

    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "codedoc.json").write_text(json.dumps({"name": "not-codedoc"}), encoding="utf-8")

    sw = SafeWriter(out / "codedoc.json", "json", "main.py", {})
    try:
        sw.load()
        assert False, "load() should have raised on a foreign file"
    except ConfigError:
        pass


def test_K3_safewriter_preloads_completed_records(tmp_path):
    """K3 (Scenario Q): a completed codedoc.json is preserved across a safe-mode interrupt.

    After preloading, the first record() flush must keep the prior records, not
    overwrite the file with only the newly processed one.
    """
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    _codedoc_json(out / "codedoc.json", [
        {"path": "a.py", "hash": "HA"},
        {"path": "b.py", "hash": "HB"},
    ])  # completed output — no in_progress status

    sw = SafeWriter(out / "codedoc.json", "json", "main.py", {})
    sw.load()
    sw.record("c.py", {"language": "python"}, file_hash="HC")  # then "interrupt"

    after = json.loads((out / "codedoc.json").read_text(encoding="utf-8"))
    paths = sorted(f["path"] for f in after["files"])
    assert paths == ["a.py", "b.py", "c.py"], paths


def test_K4_stale_build_file_does_not_override_newer_json(tmp_path):
    """K4 (Scenario S): a build file older than codedoc.json is treated as stale."""
    from codedoc.pipeline import _load_existing_file_docs
    from codedoc.core.output import BUILD_FILENAME
    import time

    out = tmp_path / "codedoc"
    _codedoc_json(out / BUILD_FILENAME, [{"path": "main.py", "hash": "OLD"}])
    time.sleep(0.05)
    _codedoc_json(out / "codedoc.json", [{"path": "main.py", "hash": "NEW"}])

    res = _load_existing_file_docs(out, "codedoc.json")
    assert res["main.py"]["hash"] == "NEW"
    assert not (out / BUILD_FILENAME).exists()  # stale build file removed


def test_K5_newer_build_file_overrides_older_json(tmp_path):
    """K5 (Scenario N): a build file newer than codedoc.json still wins per-file."""
    from codedoc.pipeline import _load_existing_file_docs
    from codedoc.core.output import BUILD_FILENAME
    import time

    out = tmp_path / "codedoc"
    _codedoc_json(out / "codedoc.json", [{"path": "main.py", "hash": "OLD"}])
    time.sleep(0.05)
    _codedoc_json(out / BUILD_FILENAME, [{"path": "main.py", "hash": "NEW"}], status="in_progress")

    res = _load_existing_file_docs(out, "codedoc.json")
    assert res["main.py"]["hash"] == "NEW"


def test_K6_old_checkpoint_without_hash_is_reprocessed(tmp_path, monkeypatch):
    """K6 (Scenario F): a checkpoint entry with no _checkpoint_hash is reprocessed, not resumed.

    Checkpoints written before 0.7.2 carry no hash and cannot be verified, so the
    routing loop must re-send those files to the LLM rather than silently
    restoring potentially stale documentation.
    """
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    (out / ".codedoc_progress.json").write_text(json.dumps({
        "codedoc_checkpoint": True,
        "version": "1",
        "created_at": "2026-05-29T00:00:00+00:00",
        "updated_at": "2026-05-29T00:00:00+00:00",
        "entry_file": "main.py",
        "total_recorded": 1,
        "results": {"main.py": {"language": "python", "description": "stale"}},
    }), encoding="utf-8")

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats["checked"] == 1          # re-sent to the LLM
    assert stats.get("resumed", 0) == 0   # not restored from the hash-less checkpoint
