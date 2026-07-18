"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
import json
from pathlib import Path
from tests.support.pipeline_scenarios import patch_provider
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_scenarios import md_meta
from tests.support.pipeline_scenarios import write_existing_json
from tests.support.pipeline_scenarios import write_existing_md
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.recovery_rate_limit_runs import _make_fake_provider
from tests.support.recovery_rate_limit_runs import _patch_provider

def test_md_only_incremental_skips_unchanged_files(tmp_path, monkeypatch):
    """Second --format md run must not call the LLM for unchanged files.
    This verifies that file_hashes from the MD metadata comment are used
    for the incremental hash check when no JSON exists."""
    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.pipeline import run_pipeline

    # Write a source file
    src = tmp_path / "app.py"
    src.write_text("def app(): pass\n", encoding="utf-8")
    real_hash = compute_file_hash(src)

    output_dir = tmp_path / "docs_output"

    # Simulate what a first MD run would have produced: an MD file with
    # file_hashes embedded in the metadata comment.  No JSON file exists.
    records = [
        {
            "hash": real_hash,
            "file_path": "app.py",
            "language": "python",
            "documentation": {
                "file_path": "app.py",
                "language": "python",
                "description": "The app module.",
            },
            **_PRIOR_RUN_IDENTITY,
        }
    ]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        output_format="md",
        entry_file="app.py",
    )
    # Confirm: only MD exists, no JSON
    assert (output_dir / "codedoc.md").exists()
    assert not (output_dir / "codedoc.json").exists()

    def fail_if_llm_used(config):
        raise AssertionError("LLM must not be called — file is unchanged")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_used)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "app.py",
            "output_dir": "docs_output",
            "output_format": "md",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert (output_dir / "codedoc.md").exists()
    assert not (output_dir / "codedoc.json").exists()

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

def test_B3_second_run_md_reuses_json_state(tmp_path, monkeypatch):
    """A missing Markdown target reuses its exact JSON sibling."""
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
    # JSON is NOT removed (format switch doesn't touch the other format)
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
    (tmp_path / "codedoc" / "codedoc.md").read_text()
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

def _codedoc_json(path: Path, files: list, status: str | None = None):
    """Write a minimal codedoc-owned JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"entry_file": "main.py", "schema_version": "1.4"}
    if status:
        meta["status"] = status
    payload: dict = {"_codedoc": meta, "files": files}
    if status == "in_progress":
        payload = {
            "_crash_safety": "INCOMPLETE RUN - test",
            **payload,
        }
    path.write_text(json.dumps(payload), encoding="utf-8")

def test_7_resume_skips_unchanged_files(tmp_path, monkeypatch):
    """Test 7: re-run skips files already in live backup with matching hash."""
    main_py = tmp_path / "main.py"
    main_py.write_text("x=1\n")

    from codedoc.core.db import compute_file_hash
    h = compute_file_hash(main_py)

    # Write a live backup that already has main.py with the correct hash
    _codedoc_json(
        tmp_path / "codedoc" / "codedoc.json",
        [{"path": "main.py", "hash": h, "language": "python",
          "_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": "single"}],
        status="in_progress",
    )
    # Add _crash_safety to make it look like an in-progress run
    data = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text())
    data["_crash_safety"] = "INCOMPLETE RUN - test"
    (tmp_path / "codedoc" / "codedoc.json").write_text(json.dumps(data))

    call_count = {"n": 0}
    real_provider = _make_fake_provider()
    original_complete = real_provider.complete_json

    def counted_complete(prompt, system=""):
        call_count["n"] += 1
        return original_complete(prompt, system)

    real_provider.complete_json = counted_complete
    _patch_provider(monkeypatch, real_provider)

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    assert call_count["n"] == 0, "LLM must not be called for unchanged files"
    assert stats["skipped"] == 1 or stats["reused"] == 1 or stats["checked"] == 0

def test_7b_changed_hash_triggers_reprocess_and_replaces_slot(tmp_path, monkeypatch):
    """Test 7b: changed file hash triggers re-process; result replaces in queue slot."""
    main_py = tmp_path / "main.py"
    main_py.write_text("x=1\n")

    # Live backup has main.py with OLD description and STALE hash
    _codedoc_json(
        tmp_path / "codedoc" / "codedoc.json",
        [{"path": "main.py", "hash": "stale_hash_000", "language": "python",
          "description": "OLD description"}],
        status="in_progress",
    )
    data = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text())
    data["_crash_safety"] = "INCOMPLETE"
    (tmp_path / "codedoc" / "codedoc.json").write_text(json.dumps(data))

    # Use a provider that returns a distinctively new description
    provider = _make_fake_provider(description="NEW description from provider")
    _patch_provider(monkeypatch, provider)

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    # The stale hash causes re-processing
    assert stats["checked"] == 1, "main.py must be re-processed due to hash mismatch"

    out_json = tmp_path / "codedoc" / "codedoc.json"
    assert out_json.exists()
    result = json.loads(out_json.read_text(encoding="utf-8"))
    # No _crash_safety in final clean output
    assert "_crash_safety" not in result
    # One file in the result
    assert len(result.get("files", [])) == 1
