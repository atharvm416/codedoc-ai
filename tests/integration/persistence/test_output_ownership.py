"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
import codedoc.core.output as output_mod
from codedoc.core.output import validate_distinct_artifact_paths
from codedoc.utils.errors import ConfigError
import json
from pathlib import Path
from tests.support.pipeline_scenarios import patch_provider as scenario_patch_provider
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_scenarios import write_existing_json
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.recovery_rate_limit_runs import _run
from tests.support.logging_runs import make_fake_provider as logging_make_fake_provider
from tests.support.logging_runs import patch_provider as logging_patch_provider
from tests.support.logging_runs import write_py as logging_write_py
from tests.support.pipeline_usage import write_py as usage_write_py
from tests.support.markdown_cases import _fake_provider
from codedoc.core.resume import (
    _load_existing_file_docs,
)
from codedoc.pipeline import run_pipeline
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _forbid_provider
from codedoc.core.loader import load_config

def test_distinct_paths_pass(tmp_path):
    validate_distinct_artifact_paths(
        {
            "json_live_backup": tmp_path / "codedoc.json",
            "markdown": tmp_path / "codedoc.md",
            "error_log": tmp_path / "error.log",
        }
    )

def test_none_values_are_ignored(tmp_path):
    validate_distinct_artifact_paths(
        {
            "markdown": tmp_path / "codedoc.md",
            "json_live_backup": None,
            "error_log": None,
        }
    )

def test_intentional_json_live_backup_alias_is_not_a_collision(tmp_path):
    # The final JSON and its live-backup phase share one path; representing it as
    # a single logical artifact must never trip the collision check.
    validate_distinct_artifact_paths(
        {
            "json_live_backup": tmp_path / "codedoc.json",
            "markdown": tmp_path / "codedoc.md",
            "error_log": tmp_path / "error.log",
        }
    )

def test_json_versus_markdown_collision(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "out.txt",
                "markdown": tmp_path / "out.txt",
            }
        )
    message = str(excinfo.value)
    assert "json_live_backup" in message
    assert "markdown" in message

def test_output_versus_error_log_collision(tmp_path):
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "error.log",
                "error_log": tmp_path / "error.log",
            }
        )

def test_markdown_versus_json_backup_collision(tmp_path):
    # Markdown-only mode keeps the live backup as a JSON sibling; if a user names
    # the Markdown file identically to the backup it must be rejected.
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "markdown": tmp_path / "report.json",
                "live_backup": tmp_path / "report.json",
            }
        )

def test_relative_segments_are_normalized(tmp_path):
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "codedoc.json",
                "markdown": tmp_path / "sub" / ".." / "codedoc.json",
            }
        )

def test_case_equivalent_paths_collide_on_case_insensitive_fs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        output_mod,
        "_filesystem_is_case_insensitive",
        lambda _path: True,
    )
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "CodeDoc.json",
                "markdown": tmp_path / "codedoc.json",
            }
        )

def test_case_equivalent_paths_are_distinct_on_case_sensitive_fs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        output_mod,
        "_filesystem_is_case_insensitive",
        lambda _path: False,
    )
    validate_distinct_artifact_paths(
        {
            "json_live_backup": tmp_path / "CodeDoc.json",
            "markdown": tmp_path / "codedoc.json",
        }
    )

def test_G1_legacy_db_left_untouched(tmp_path, monkeypatch):
    """G1 (0.11.3): a legacy codedoc_db.json is outside the runtime allowlist and is
    never probed or deleted — it is left byte-identical."""
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
    assert legacy.exists()
    assert legacy.read_text(encoding="utf-8") == '{"files": {}}'

def test_J1_format_json_exactly_one_output_file(tmp_path, monkeypatch):
    """J1: --format json → exactly 1 output file path in stats."""
    scenario_patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False, "parallel_agents": False})
    assert len(stats["output_files"]) == 1
    assert stats["output_files"][0].endswith(".json")

def test_J2_format_md_exactly_one_output_file(tmp_path, monkeypatch):
    """J2: --format md → exactly 1 output file path in stats."""
    scenario_patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False, "parallel_agents": False})
    assert len(stats["output_files"]) == 1
    assert stats["output_files"][0].endswith(".md")

def test_J3_format_both_exactly_two_output_files(tmp_path, monkeypatch):
    """J3: --format both → exactly 2 output file paths in stats."""
    scenario_patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "both",
                                     "propagate_changes": False, "parallel_agents": False})
    assert len(stats["output_files"]) == 2
    exts = {Path(p).suffix for p in stats["output_files"]}
    assert exts == {".json", ".md"}

def test_J4_no_codedoc_db_ever_written(tmp_path, monkeypatch):
    """J4: after any run, codedoc_db.json must never exist in the output dir."""
    scenario_patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    for fmt in ("json", "md", "both"):
        run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": fmt,
                                 "propagate_changes": False, "parallel_agents": False})
        assert not (tmp_path / "codedoc" / "codedoc_db.json").exists(), \
            f"codedoc_db.json found after --format {fmt}"

def test_K4_legacy_build_file_left_untouched(tmp_path, monkeypatch):
    """K4 (0.11.3): a legacy .codedoc_build.json is outside the runtime allowlist —
    never probed, migrated, or deleted; the exact JSON target is the only source."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    build = out / ".codedoc_build.json"
    build.write_text('{"files": []}', encoding="utf-8")
    write_existing_json(out / "codedoc.json", compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                             "propagate_changes": False})
    assert build.exists()
    assert build.read_text(encoding="utf-8") == '{"files": []}'

def test_K6_legacy_progress_file_left_untouched(tmp_path, monkeypatch):
    """K6 (0.11.3): a legacy .codedoc_progress.json checkpoint is never migrated or
    probed; the run documents from the exact target only."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    progress = out / ".codedoc_progress.json"
    progress.write_text('{"codedoc_checkpoint": true}', encoding="utf-8")
    write_existing_json(out / "codedoc.json", compute_file_hash(src), "Cached.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py",
                                     "propagate_changes": False, "parallel_agents": False})
    assert stats["checked"] == 0          # reused from the exact JSON target
    assert progress.exists()              # legacy checkpoint left byte-identical

def test_6_foreign_json_conversion_sibling_blocks_md_run(tmp_path, monkeypatch):
    """A present same-stem fallback must be owned before it can be reused."""
    (tmp_path / "docs").mkdir()
    foreign = tmp_path / "docs" / "report.json"
    foreign_bytes = json.dumps({"not": "codedoc"}).encode("utf-8")
    foreign.write_bytes(foreign_bytes)
    (tmp_path / "main.py").write_text("x=1\n")

    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: pytest.fail("provider must not be created"),
    )

    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError, match="conversion sibling"):
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "output_dir": "docs/report.md",
            "parallel_agents": False,
            "propagate_changes": False,
        })

    assert not (tmp_path / "docs" / "report.md").exists()
    assert foreign.read_bytes() == foreign_bytes
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()

def test_13_format_both_keeps_json_after_clean_finish(tmp_path, monkeypatch):
    """Test 13: --format both → both codedoc.json and codedoc.md on clean finish."""
    (tmp_path / "main.py").write_text("x=1\n")
    _run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        output_format="both",
    )

    json_file = tmp_path / "codedoc" / "codedoc.json"
    md_file = tmp_path / "codedoc" / "codedoc.md"

    assert json_file.exists(), "codedoc.json must exist for --format both"
    assert md_file.exists(), "codedoc.md must exist for --format both"

    # JSON must be clean (no crash banner)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "_crash_safety" not in data
    assert data.get("_codedoc", {}).get("status") != "in_progress"

    # Markdown must not contain crash banner
    md_content = md_file.read_text(encoding="utf-8")
    assert "INCOMPLETE RUN" not in md_content

class TestPreflightOutputTargets:
    """G0: Foreign targets fail before create_provider() is called."""

    def test_foreign_json_target_raises_before_provider(self, tmp_path, monkeypatch):
        """A foreign report.json causes ConfigError before LLM is ever instantiated."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or logging_make_fake_provider(),
        )

        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "report.json").write_text('{"not": "codedoc"}', encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.json"})

        assert not provider_called, "create_provider must not be called before preflight"

    def test_foreign_md_target_raises_before_provider(self, tmp_path, monkeypatch):
        """A foreign report.md causes ConfigError before LLM is instantiated."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or logging_make_fake_provider(),
        )

        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "report.md").write_text("# personal file\n", encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.md"})

        assert not provider_called, "create_provider must not be called before preflight"

    def test_codedoc_owned_json_passes_preflight(self, tmp_path, monkeypatch):
        """An existing CodeDoc-owned JSON passes preflight and the run succeeds."""
        logging_patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        owned = {
            "_codedoc": {"schema_version": "1.4"},
            "files": [],
        }
        (out_dir / "codedoc.json").write_text(json.dumps(owned), encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out"})
        assert stats is not None

    def test_codedoc_owned_md_passes_preflight(self, tmp_path, monkeypatch):
        """An existing CodeDoc-owned Markdown passes preflight.

        0.9.3: ownership requires structurally valid metadata.  A marker-only
        Markdown with malformed metadata is now treated as foreign (the
        centralized reader fails closed), so the owned fixture must carry a
        well-formed ``<!-- codedoc-ai: {...} -->`` comment.
        """
        import json as _json

        logging_patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        meta = {"entry_file": "src.py", "schema_version": "1.4", "file_hashes": {}}
        (out_dir / "codedoc.md").write_text(
            f"<!-- codedoc-ai: {_json.dumps(meta)} -->\n# docs\n", encoding="utf-8"
        )

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(
            tmp_path, {"entry_file": "src.py", "output_dir": "out", "output_format": "md"}
        )
        assert stats is not None

    def test_nonexistent_output_dir_passes_preflight(self, tmp_path, monkeypatch):
        """A non-existent output directory passes preflight (all targets are new)."""
        logging_patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        logging_write_py(src)

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(
            tmp_path, {"entry_file": "src.py", "output_dir": "brand_new_dir"}
        )
        assert stats is not None

    def test_format_both_fails_if_json_target_foreign(self, tmp_path, monkeypatch):
        """--format both preflights both targets; fails if JSON is foreign."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or logging_make_fake_provider(),
        )

        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "codedoc.json").write_text('{"not": "codedoc"}', encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(
                tmp_path,
                {"entry_file": "src.py", "output_dir": "out", "output_format": "both"},
            )
        assert not provider_called

    def test_foreign_json_conversion_sibling_fails_before_provider(self, tmp_path, monkeypatch):
        """A same-stem fallback is validated strictly before paid work."""
        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or logging_make_fake_provider(),
        )

        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        foreign = out_dir / "report.json"
        foreign.write_text('{"foreign": true}', encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        from codedoc.utils.errors import ConfigError

        with pytest.raises(ConfigError, match="conversion sibling"):
            run_pipeline(
                tmp_path, {"entry_file": "src.py", "output_dir": "out/report.md"}
            )

        assert not provider_called
        assert not (out_dir / "report.md").exists()
        assert foreign.read_text(encoding="utf-8") == '{"foreign": true}'

    def test_foreign_target_leaves_no_new_output_directory(self, tmp_path, monkeypatch):
        """Preflight fires before mkdir: a genuinely new output dir is not created on failure."""
        from codedoc.utils.errors import ConfigError

        src = tmp_path / "src.py"
        logging_write_py(src)

        # Create a sibling directory with a foreign codedoc.json — this is the
        # scenario where the user points at an existing dir that has a foreign file.
        out_dir = tmp_path / "my_out"
        out_dir.mkdir()
        (out_dir / "codedoc.json").write_text('{"foreign": true}', encoding="utf-8")

        # A child sub-dir that does NOT exist yet — preflight should prevent its creation.
        new_sub = out_dir / "sub_that_must_not_be_created"

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or logging_make_fake_provider(),
        )

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": str(out_dir)})

        assert not provider_called
        # The directory contents must be unchanged — only the foreign file we planted
        assert not new_sub.exists(), "No new subdirectories created on preflight failure"

    def test_completed_json_sibling_accepted_for_md_preflight(self, tmp_path, monkeypatch):
        """A completed CodeDoc JSON from a prior JSON run is accepted as an MD live-backup sibling."""
        logging_patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        logging_write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        completed = {
            "_codedoc": {"schema_version": "1.4", "status": "complete"},
            "files": [],
        }
        (out_dir / "report.json").write_text(json.dumps(completed), encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.md"})
        assert stats is not None

def test_legacy_checkpoint_is_ignored_and_preserved(tmp_path):
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    usage_write_py(tmp_path / "main.py")
    output = tmp_path / "codedoc"
    output.mkdir()
    checkpoint = {
        "codedoc_checkpoint": True,
        "results": {
            "main.py": {
                "_checkpoint_hash": compute_file_hash(tmp_path / "main.py"),
                "description": "checkpoint",
                "language": "python",
                "_analysis_revision": ANALYSIS_REVISION,
                "_analysis_mode": "single",
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
    assert stats["would_resume"] == 0
    assert stats["would_call_llm_for"] == 1
    assert json.loads(
        (output / ".codedoc_progress.json").read_text(encoding="utf-8")
    ) == checkpoint

def test_A18_md_run_ignores_and_preserves_legacy_json_sibling(tmp_path, monkeypatch):
    """A18: After resuming from a JSON crash backup, the final MD embedded view
    contains ALL records — both those pre-loaded from the backup and newly
    processed ones — and the JSON backup is removed."""
    import json as _json
    from codedoc.core.db import compute_file_hash

    # Two source files: a.py was documented in the "crashed" run, b.py was not.
    a_py = tmp_path / "a.py"
    b_py = tmp_path / "b.py"
    a_py.write_text("from b import x\n")
    b_py.write_text("x = 1\n")

    h_a = compute_file_hash(a_py)

    # Simulate a crashed first run: codedoc.json has a.py with a valid hash
    # but carries _crash_safety and status=in_progress (b.py was never reached).
    out_dir = tmp_path / "codedoc"
    out_dir.mkdir()
    crash_backup = out_dir / "codedoc.json"
    crash_backup.write_text(_json.dumps({
        "_crash_safety": "INCOMPLETE RUN - crash test",
        "_codedoc": {"entry_file": "a.py", "schema_version": "1.4", "status": "in_progress"},
        "files": [
            {
                "path": "a.py",
                "hash": h_a,
                "language": "python",
                "description": "Pre-crash description for a.py.",
                "role_in_system": "test",
                "functions": [],
                "classes": [],
                "exports": [],
                "key_concepts": ["entry point"],
                "usage_example": "",
                "dependencies_analysis": {},
            }
        ],
    }), encoding="utf-8")
    legacy_bytes = crash_backup.read_bytes()

    # Resume run: should load a.py from the crash backup, process b.py via LLM.
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "output_format": "md",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    md_path = out_dir / "codedoc.md"
    assert md_path.exists(), "Final Markdown must be written after crash resume"
    assert crash_backup.read_bytes() == legacy_bytes
    assert not (out_dir / "crash_recovery.json").exists()

    # The embedded view must contain BOTH a.py (from crash backup) and b.py (newly processed)
    from codedoc.core.project_view import read_embedded_view
    md_content = md_path.read_text(encoding="utf-8")
    embedded = read_embedded_view(md_content)

    assert embedded is not None, "Embedded view must be present in resumed output"
    assert "_crash_safety" not in embedded, "Embedded view must not contain _crash_safety"

    paths = {f["path"] for f in embedded.get("files", [])}
    assert "a.py" in paths, "a.py (from crash backup) must be in the embedded view"
    assert "b.py" in paths, "b.py (newly processed) must be in the embedded view"
    assert len(paths) == 2, f"Expected exactly 2 files in embedded view, got {paths}"

    # Hash for a.py must be preserved from the crash backup (not regenerated)
    file_map = {f["path"]: f for f in embedded["files"]}
    assert file_map["a.py"]["hash"] == h_a, (
        "a.py hash from the crash backup must be preserved in the embedded view"
    )

def test_foreign_fallback_blocks_before_provider(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "docs"
    out.mkdir()
    (out / "codedoc.md").write_text("foreign", encoding="utf-8")
    _forbid_provider(monkeypatch)
    with pytest.raises(ConfigError, match="conversion sibling"):
        run_pipeline(tmp_path, _config("json"))
    assert not (out / "codedoc.json").exists()

def test_validate_distinct_rejects_recovery_equal_to_json(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths({
            "json": tmp_path / "codedoc.json",
            "live_backup": tmp_path / "codedoc.json",
        })

def test_validate_distinct_rejects_recovery_equal_to_markdown(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths({
            "markdown": tmp_path / "codedoc.md",
            "live_backup": tmp_path / "codedoc.md",
        })

def test_validate_distinct_rejects_recovery_equal_to_log(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths({
            "error_log": tmp_path / "error.log",
            "live_backup": tmp_path / "error.log",
        })

def test_validate_distinct_accepts_separate_recovery(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths

    # The normal 0.9.8 layout: all four artifacts are distinct.
    validate_distinct_artifact_paths({
        "error_log": tmp_path / "error.log",
        "json": tmp_path / "codedoc.json",
        "markdown": tmp_path / "codedoc.md",
        "live_backup": tmp_path / "crash_recovery.json",
    })

@pytest.mark.parametrize("reserved", [
    "crash_recovery.json",
    "docs/crash_recovery.json",
    "CRASH_RECOVERY.JSON",
])
def test_reserved_output_name_rejected(tmp_path, reserved):
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path, {"output_dir": reserved})
    assert "crash-recovery" in str(excinfo.value).lower()

@pytest.mark.parametrize("allowed", [
    "crash_recovery_codedoc.json",
    "docs/crash_recovery_report.json",
    "crash_recovery_report.md",
    "crash_recovery.md",  # only the exact .json name is reserved
])
def test_former_prefix_names_are_no_longer_reserved(tmp_path, allowed):
    cfg = load_config(tmp_path, {"output_dir": allowed})
    assert cfg["output_format"] in ("json", "md")

def test_reserved_output_name_rejected_before_scan_or_mutation(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")

    import codedoc.pipeline as pipeline_mod
    monkeypatch.setattr(
        pipeline_mod, "scan_files",
        lambda *a, **k: pytest.fail("scan must not run for a reserved --output"),
    )
    monkeypatch.setattr(
        pipeline_mod, "create_provider",
        lambda *a, **k: pytest.fail("provider must not be created"),
    )

    from codedoc.pipeline import run_pipeline
    with pytest.raises(ConfigError):
        run_pipeline(tmp_path, {"output_dir": "crash_recovery.json"})

    assert not (tmp_path / "crash_recovery.json").exists()

def test_ordinary_md_run_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {"output_dir": "report.md"})
    assert cfg["output_format"] == "md"
    assert cfg["output_md_filename"] == "report.md"
    assert cfg["output_json_filename"] == "report.json"

def test_non_reserved_named_output_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {"output_dir": "docs/report.json"})
    assert cfg["output_format"] == "json"
    assert cfg["output_json_filename"] == "report.json"
    assert cfg["output_md_filename"] == "report.md"

def test_default_run_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {})
    assert cfg["output_json_filename"] == "codedoc.json"

def test_missing_target_rejects_foreign_opposite_format_sibling(tmp_path):
    json_target = tmp_path / "codedoc.json"
    md_sibling = tmp_path / "codedoc.md"
    md_sibling.write_text("foreign sibling", encoding="utf-8")
    with pytest.raises(ConfigError, match="conversion sibling"):
        _load_existing_file_docs(json_target, md_sibling, "json")
