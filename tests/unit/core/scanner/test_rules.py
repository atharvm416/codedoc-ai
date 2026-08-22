"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from tests.support.configuration_cases import _fake_provider

def test_scan_respects_skip_dirs_list(tmp_path):
    """scan_files skips directories that appear in the skip_dirs list."""
    (tmp_path / "main.py").write_text("print('ok')\n")
    env_dir = tmp_path / "myenv"
    env_dir.mkdir()
    (env_dir / "installed.py").write_text("print('skip')\n")

    from codedoc.core.scanner import scan_files

    # Explicitly pass skip_dirs (previously came from the hardcoded SKIP_DIRS)
    files = scan_files(tmp_path, supported_extensions=[".py"], skip_dirs=["myenv"])
    rels = {f["rel_path"] for f in files}

    assert "main.py" in rels
    assert "myenv/installed.py" not in rels

def test_scan_without_skip_dirs_includes_all_non_hidden_dirs(tmp_path):
    """Without skip_dirs, non-hidden directories are not skipped automatically."""
    (tmp_path / "main.py").write_text("print('ok')\n")
    env_dir = tmp_path / "myenv"
    env_dir.mkdir()
    (env_dir / "installed.py").write_text("print('found')\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(tmp_path, supported_extensions=[".py"])
    rels = {f["rel_path"] for f in files}

    # Without skip_dirs, myenv is NOT skipped
    assert "myenv/installed.py" in rels

def test_scan_ignores_strict_project_relative_path(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    generated = tmp_path / "services" / "generated"
    generated.mkdir(parents=True)
    (generated / "client.py").write_text("print('skip')\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(
        tmp_path,
        supported_extensions=[".py"],
        ignore_paths=["/services/generated"],
    )
    rels = {f["rel_path"] for f in files}

    assert "main.py" in rels
    assert "services/generated/client.py" not in rels

def test_scan_ignores_single_file_path(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    (tmp_path / "secret.py").write_text("print('skip')\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(tmp_path, supported_extensions=[".py"], ignore_paths=["secret.py"])
    rels = {f["rel_path"] for f in files}

    assert rels == {"main.py"}

def test_scan_skips_unreadable_directories(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("print('ok')\n")
    blocked = tmp_path / "pytest_cache"
    blocked.mkdir()

    original_iterdir = type(tmp_path).iterdir

    def fake_iterdir(path):
        if path == blocked:
            raise PermissionError("access denied")
        return original_iterdir(path)

    monkeypatch.setattr(type(tmp_path), "iterdir", fake_iterdir)

    from codedoc.core.scanner import scan_files

    files = scan_files(tmp_path, supported_extensions=[".py"])
    rels = {f["rel_path"] for f in files}

    assert rels == {"main.py"}

def test_scan_uses_extension_language_map_for_language_detection(tmp_path):
    """extension_language_map drives both file filtering and language labelling."""
    (tmp_path / "app.svelte").write_text("<script>let x=1;</script>\n")
    (tmp_path / "main.py").write_text("x=1\n")
    (tmp_path / "README.md").write_text("# docs\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(
        tmp_path,
        extension_language_map={".svelte": "svelte", ".py": "python"},
    )
    by_rel = {f["rel_path"]: f for f in files}

    assert "app.svelte" in by_rel
    assert by_rel["app.svelte"]["language"] == "svelte"
    assert "main.py" in by_rel
    assert by_rel["main.py"]["language"] == "python"
    assert "README.md" not in by_rel  # not in the map → not scanned

def test_scanner_walker_state_is_per_scan_A6(tmp_path):
    """A6: two sequential scans must not share state (skipped_dirs counts are
    independent; no leakage via function attributes)."""
    from codedoc.core.scanner import scan_files

    # First tree: one skipped dir.
    root1 = tmp_path / "p1"
    (root1 / "pkg").mkdir(parents=True)
    (root1 / "pkg" / "a.py").write_text("x=1\n")
    skip1 = root1 / "node_modules"
    skip1.mkdir()
    (skip1 / "lib.py").write_text("y=1\n")

    files1 = scan_files(root1, supported_extensions=[".py"], skip_dirs=["node_modules"])
    rels1 = {f["rel_path"] for f in files1}
    assert "pkg/a.py" in rels1
    assert "node_modules/lib.py" not in rels1

    # Second, independent scan with no skipped dirs must return its own files
    # and not be influenced by the first scan's state.
    root2 = tmp_path / "p2"
    root2.mkdir()
    (root2 / "main.py").write_text("z=1\n")
    files2 = scan_files(root2, supported_extensions=[".py"], skip_dirs=["node_modules"])
    rels2 = {f["rel_path"] for f in files2}
    assert rels2 == {"main.py"}

def test_C8_new_extension_scanned_and_labelled(tmp_path, monkeypatch):
    """C8: Adding .svelte via extension_language_map_add scans and labels it."""
    (tmp_path / "App.svelte").write_text("<script>let x = 1;</script>\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    run_pipeline(tmp_path, {
        "entry_file": "App.svelte",
        "extension_language_map_add": {".svelte": "svelte"},
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    files = {f["path"]: f for f in result.get("files", [])}

    assert "App.svelte" in files, "New extension must be scanned"
    assert files["App.svelte"]["language"] == "svelte", "Language must be labelled from the map"

def test_C8_removed_extension_not_scanned(tmp_path, monkeypatch):
    """C8b: Removing .py via extension_language_map_remove prevents scanning .py files."""
    (tmp_path / "main.ts").write_text("const x = 1;\n")
    (tmp_path / "utils.py").write_text("x = 1\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    run_pipeline(tmp_path, {
        "entry_file": "main.ts",
        "extension_language_map_remove": [".py"],
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    result = json.loads(out.read_text(encoding="utf-8"))
    files = {f["path"] for f in result.get("files", [])}

    assert "main.ts" in files, ".ts files must still be scanned"
    assert "utils.py" not in files, ".py files must not be scanned after removal"

def test_P2_scan_files_positional_list_does_not_crash(tmp_path):
    """P2 regression: scan_files(root, ['.py']) positionally must not crash.

    Old callers pass supported_extensions as the second positional argument.
    The new signature makes that position extension_language_map (a dict).
    The guard must detect the list and redirect to the legacy path.
    """
    (tmp_path / "main.py").write_text("x=1\n")

    from codedoc.core.scanner import scan_files

    # Must not raise AttributeError on list.items()
    files = scan_files(tmp_path, [".py"])
    rels = {f["rel_path"] for f in files}
    assert "main.py" in rels

def test_P2_scan_files_positional_tuple_does_not_crash(tmp_path):
    """P2 regression: scan_files(root, ('.py',)) positionally must not crash."""
    (tmp_path / "app.py").write_text("x=1\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(tmp_path, (".py",))
    assert any(f["rel_path"] == "app.py" for f in files)

def test_P2_positional_list_language_resolved_from_fallback_map(tmp_path):
    """P2: When a list is passed positionally, language comes from _FALLBACK_LANGUAGE_MAP."""
    (tmp_path / "main.dart").write_text("void main(){}\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(tmp_path, [".dart"])
    f = next(f for f in files if f["rel_path"] == "main.dart")
    assert f["language"] == "dart"  # from _FALLBACK_LANGUAGE_MAP


# ---------------------------------------------------------------------------
# Section 5.6: exact generated-target exclusion and scan diagnostics
# ---------------------------------------------------------------------------


def test_exclude_paths_matches_by_exact_equality_not_basename_or_prefix(tmp_path):
    """A source file that merely shares the excluded file's basename (in a
    different directory) or sits inside a same-named directory must NOT be
    excluded -- only the exact resolved path is protected."""
    from codedoc.core.scanner import exclude_path_key, scan_files

    excluded_target = tmp_path / "out" / "codedoc.json"
    excluded_target.parent.mkdir()
    excluded_target.write_text("{}\n", encoding="utf-8")

    # Same basename, different directory -- must still be scanned as source
    # if it happened to have a supported extension (it does not here, but
    # prove the exclusion set itself only matches the one exact path).
    lookalike = tmp_path / "other" / "codedoc.json"
    lookalike.parent.mkdir()
    lookalike.write_text("{}\n", encoding="utf-8")

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    exclude_paths = {exclude_path_key(excluded_target)}
    files = scan_files(
        tmp_path,
        extension_language_map={".py": "python", ".json": "json"},
        exclude_paths=exclude_paths,
    )
    rels = {f["rel_path"] for f in files}
    assert "main.py" in rels
    assert "out/codedoc.json" not in rels
    assert "other/codedoc.json" in rels, (
        "A same-named file at a different exact path must not be excluded"
    )


def test_exclude_paths_protects_co_located_generated_targets(tmp_path):
    """Co-located source/output directories remain supported: only the
    exact generated file is excluded, every other file in that same
    directory is still scanned normally."""
    from codedoc.core.scanner import exclude_path_key, scan_files

    (tmp_path / "codedoc.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    exclude_paths = {exclude_path_key(tmp_path / "codedoc.json")}
    files = scan_files(
        tmp_path,
        extension_language_map={".py": "python", ".json": "json"},
        exclude_paths=exclude_paths,
    )
    rels = {f["rel_path"] for f in files}
    assert rels == {"main.py"}


def test_scan_diagnostics_tracks_files_skipped_large(tmp_path):
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "big.py").write_text("x" * 2048, encoding="utf-8")
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")

    diagnostics = ScanDiagnostics()
    files = scan_files(
        tmp_path,
        supported_extensions=[".py"],
        max_file_size_kb=1,
        diagnostics=diagnostics,
    )
    rels = {f["rel_path"] for f in files}
    assert rels == {"small.py"}
    assert diagnostics.files_skipped_large == 1
    assert diagnostics.files_skipped_unreadable == 0


def test_scan_diagnostics_tracks_unreadable_files_and_excludes_them(tmp_path, monkeypatch):
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    blocked = tmp_path / "blocked.py"
    blocked.write_text("x = 1\n", encoding="utf-8")

    original_stat = type(blocked).stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "blocked.py":
            raise PermissionError("access denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(blocked), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    files = scan_files(
        tmp_path,
        supported_extensions=[".py"],
        diagnostics=diagnostics,
    )
    rels = {f["rel_path"] for f in files}
    assert rels == {"good.py"}
    assert diagnostics.files_skipped_unreadable == 1
    assert diagnostics.files_skipped_large == 0


def test_scan_diagnostics_deduplicates_and_bounds_unreadable_warnings(tmp_path, monkeypatch, caplog):
    """Warn individually for at most MAX_UNREADABLE_FILE_WARNINGS files, then
    exactly one aggregate line for the rest -- never one line per file
    beyond the bound, and never double-counted."""
    import logging

    from codedoc.core.scanner import MAX_UNREADABLE_FILE_WARNINGS, ScanDiagnostics, scan_files

    total_unreadable = MAX_UNREADABLE_FILE_WARNINGS + 5
    for i in range(total_unreadable):
        (tmp_path / f"blocked_{i}.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")

    original_stat = type(tmp_path).stat

    def fake_stat(self, *args, **kwargs):
        if self.name.startswith("blocked_"):
            raise PermissionError("access denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        files = scan_files(
            tmp_path,
            supported_extensions=[".py"],
            diagnostics=diagnostics,
        )
    rels = {f["rel_path"] for f in files}
    assert rels == {"good.py"}
    assert diagnostics.files_skipped_unreadable == total_unreadable

    per_file_warnings = [
        r for r in caplog.records if "Skipping unreadable file" in r.message
    ]
    aggregate_warnings = [
        r for r in caplog.records if "more unreadable file" in r.message
    ]
    assert len(per_file_warnings) == MAX_UNREADABLE_FILE_WARNINGS
    assert len(aggregate_warnings) == 1


def test_scan_diagnostics_does_not_double_count_across_a_rescan(
    tmp_path, monkeypatch, caplog
):
    """Section 12.1 C5: a shared ScanDiagnostics instance passed to two
    scan_files calls against the same tree (as a rescan does) must count
    each physical unreadable/large file once for the run, not once per
    call."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "good.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "big.py").write_text("x" * 2048, encoding="utf-8")
    blocked = tmp_path / "blocked.py"
    blocked.write_text("x = 1\n", encoding="utf-8")

    original_stat = type(blocked).stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "blocked.py":
            raise PermissionError("access denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(blocked), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    for _ in range(2):
        files = scan_files(
            tmp_path,
            supported_extensions=[".py"],
            max_file_size_kb=1,
            diagnostics=diagnostics,
        )
        rels = {f["rel_path"] for f in files}
        assert rels == {"good.py"}

    assert diagnostics.files_skipped_unreadable == 1
    assert diagnostics.files_skipped_large == 1
    unreadable_warnings = [
        record
        for record in caplog.records
        if "Skipping unreadable file" in record.message
    ]
    assert len(unreadable_warnings) == 1


# ---------------------------------------------------------------------------
# Section 5.6: pipeline-level skip_dirs / exact-output-exclusion integration
#
# Deliberately placed here rather than in test_config_precedence.py: that
# file's own byte content is frozen source data for
# tests/fixtures/split_state/completed_0_14_1.json,
# completed_0_14_2.json, and recovery_0_14_1_completed_split.json (read via
# Path(__file__).with_name(...) and hash-verified against those fixtures'
# stored "hash" field) -- any edit there, even whitespace-only, invalidates
# those frozen hashes.
# ---------------------------------------------------------------------------


def test_C7_remove_skip_dir_lets_a_co_located_output_directory_be_scanned(
    tmp_path, monkeypatch
):
    """C7, section 5.6: ``--remove-skip-dir`` must actually work -- the
    pipeline no longer unconditionally re-adds the output directory's
    basename to skip_dirs.  A real source file co-located inside the same
    directory as the generated output is scanned and documented; only the
    exact generated targets (codedoc.json/codedoc.md/crash_recovery.json)
    are protected, by exact-path equality, from ever being treated as
    source themselves."""
    from codedoc.pipeline import run_pipeline

    pkg_dir = tmp_path / "codedoc"
    pkg_dir.mkdir()
    (pkg_dir / "helper.py").write_text("pass\n")
    (tmp_path / "main.py").write_text("import codedoc.helper\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    run_pipeline(tmp_path, {
        "entry_file": None,
        "auto_entry_candidates": [],
        "documentation_scope": "all",
        "output_dir": "codedoc",
        "skip_dirs_remove": ["codedoc"],
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    scanned_paths = {f["path"] for f in result.get("files", [])}

    assert "codedoc/helper.py" in scanned_paths
    assert "codedoc/codedoc.json" not in scanned_paths
    assert "codedoc/codedoc.md" not in scanned_paths
    assert "codedoc/crash_recovery.json" not in scanned_paths


def test_C7_default_skip_dirs_still_protects_the_output_directory_by_default(
    tmp_path, monkeypatch
):
    """Without an explicit skip_dirs_remove, the default skip_dirs list
    (which already includes "codedoc") still keeps the whole default output
    directory out of the scan -- the fix only removes the *unconditional*
    re-addition, not the ordinary default behavior."""
    from codedoc.pipeline import run_pipeline

    pkg_dir = tmp_path / "codedoc"
    pkg_dir.mkdir()
    (pkg_dir / "helper.py").write_text("pass\n")
    (tmp_path / "main.py").write_text("pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    scanned_paths = {f["path"] for f in result.get("files", [])}
    assert not any(p.startswith("codedoc/") for p in scanned_paths)


def test_explicit_entry_file_colliding_with_output_target_raises_config_error(
    tmp_path, monkeypatch
):
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    with pytest.raises(ConfigError, match="generated output target"):
        run_pipeline(tmp_path, {
            "entry_file": "docs/codedoc.json",
            "output_dir": "docs",
            "parallel_agents": False,
            "propagate_changes": False,
        })


def test_force_files_colliding_with_output_target_raises_config_error(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    with pytest.raises(ConfigError, match="generated output target"):
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "output_dir": "docs",
            "force_files": ["docs/codedoc.md"],
            "parallel_agents": False,
            "propagate_changes": False,
        })


def test_post_stat_read_failure_fails_boundedly_and_is_counted(tmp_path, monkeypatch):
    """Section 12.1 C5: an ordinary file that passes the scanner's stat-based
    check but fails to read later, during planning (e.g. a permission change
    or deletion racing the scan), must never escape as a raw OSError -- it
    fails boundedly with a ConfigError, detected and counted before any
    provider is constructed."""
    import codedoc.core.planning as planning
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    (tmp_path / "main.py").write_text("import other\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")

    real_read_source_snapshot = planning.read_source_snapshot

    def flaky_read_source_snapshot(path):
        if path.name == "other.py":
            raise PermissionError("access denied")
        return real_read_source_snapshot(path)

    monkeypatch.setattr(planning, "read_source_snapshot", flaky_read_source_snapshot)

    def _forbidden_create_provider(_config):
        raise AssertionError("create_provider must not be called before this failure")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _forbidden_create_provider)

    with pytest.raises(ConfigError, match="other.py"):
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "parallel_agents": False,
            "propagate_changes": False,
        })

    assert not (tmp_path / "codedoc").exists()


def test_unreadable_entry_file_raises_config_error_before_output_mutation(
    tmp_path, monkeypatch
):
    """Section 12.1 C5: an explicitly-specified --entry file that cannot be
    read fails with an actionable ConfigError, never a raw filesystem error,
    and before any recovery file or output is created."""
    import codedoc.core.planning as planning
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    real_read_source_snapshot = planning.read_source_snapshot

    def flaky_read_source_snapshot(path):
        if path.name == "main.py":
            raise PermissionError("access denied")
        return real_read_source_snapshot(path)

    monkeypatch.setattr(planning, "read_source_snapshot", flaky_read_source_snapshot)

    def _forbidden_create_provider(_config):
        raise AssertionError("create_provider must not be called before this failure")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _forbidden_create_provider)

    with pytest.raises(ConfigError, match="Entry file 'main.py' could not be read"):
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "parallel_agents": False,
            "propagate_changes": False,
        })

    assert not (tmp_path / "codedoc").exists()


def test_unreadable_forced_file_raises_config_error_before_output_mutation(
    tmp_path, monkeypatch
):
    """Section 12.1 C5: an unreadable force_files entry fails the same way
    an unreadable entry file does -- an actionable ConfigError, before any
    recovery file or output is created."""
    import codedoc.core.planning as planning
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    (tmp_path / "main.py").write_text("import other\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")

    real_read_source_snapshot = planning.read_source_snapshot

    def flaky_read_source_snapshot(path):
        if path.name == "other.py":
            raise PermissionError("access denied")
        return real_read_source_snapshot(path)

    monkeypatch.setattr(planning, "read_source_snapshot", flaky_read_source_snapshot)

    def _forbidden_create_provider(_config):
        raise AssertionError("create_provider must not be called before this failure")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _forbidden_create_provider)

    with pytest.raises(ConfigError, match="force_files entry 'other.py' could not be read"):
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "force_files": ["other.py"],
            "parallel_agents": False,
            "propagate_changes": False,
        })

    assert not (tmp_path / "codedoc").exists()


def test_only_large_files_early_return_reports_counters_dry_and_real(tmp_path, monkeypatch):
    """Section 12.1 C5: when scanning finds only oversized files (nothing
    left in all_files), both the dry-run and the real-run early-return
    stats must still surface files_skipped_large -- scanning already ran and
    already knows this before either return path."""
    from codedoc.pipeline import run_pipeline

    (tmp_path / "big.py").write_text("x" * 2048, encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    dry_stats = run_pipeline(tmp_path, {
        "max_file_size_kb": 1,
        "parallel_agents": False,
        "propagate_changes": False,
        "dry_run": True,
    })
    assert dry_stats["files_skipped_large"] == 1
    assert dry_stats["files_skipped_unreadable"] == 0

    real_stats = run_pipeline(tmp_path, {
        "max_file_size_kb": 1,
        "parallel_agents": False,
        "propagate_changes": False,
    })
    assert real_stats["files_skipped_large"] == 1
    assert real_stats["files_skipped_unreadable"] == 0


def test_opposite_format_sibling_is_excluded_even_when_not_the_active_format(
    tmp_path, monkeypatch
):
    """Both the active and the opposite-format generated targets are always
    excluded from scanning, regardless of the selected output_format, since
    a cross-format resume can read the opposite-format sibling. A genuine
    codedoc.md from an earlier "both"-format run must never be treated as
    source once a later run selects only "json"."""
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "output_dir": "docs",
        "output_format": "both",
        "parallel_agents": False,
        "propagate_changes": False,
    })
    assert (tmp_path / "docs" / "codedoc.md").exists()

    stats = run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "output_dir": "docs",
        "output_format": "json",
        "extension_language_map_add": {".md": "markdown"},
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "docs" / "codedoc.json"
    result = json.loads(out.read_text(encoding="utf-8"))
    scanned_paths = {f["path"] for f in result.get("files", [])}
    assert "docs/codedoc.md" not in scanned_paths
    assert stats["checked"] + stats.get("reused", 0) + stats.get("skipped", 0) == 1

def test_entry_file_failing_stat_raises_config_error_not_raw_oserror(
    tmp_path, monkeypatch
):
    """Section 5.6 / 12.1 C5, stat-inspection counterpart: an explicitly
    requested entry file whose own metadata cannot be read (EACCES on stat,
    not ENOENT) must fail as an actionable ConfigError before any provider is
    constructed, never as a raw PermissionError escaping detect_entry_file's
    ``Path.exists()`` -- which swallows only the "doesn't exist"-shaped errno
    set and re-raises EACCES."""
    from pathlib import Path

    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("y = 2\n", encoding="utf-8")

    real_stat = Path.stat

    def denied_stat(self, *args, **kwargs):
        if self.name == "main.py":
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)

    def _forbidden_create_provider(_config):
        raise AssertionError("create_provider must not be called before this failure")

    monkeypatch.setattr("codedoc.pipeline.create_provider", _forbidden_create_provider)

    with pytest.raises(ConfigError, match="Entry file 'main.py' could not be read"):
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "parallel_agents": False,
            "propagate_changes": False,
        })

    assert not (tmp_path / "codedoc").exists()

def test_unreadable_auto_entry_candidate_is_skipped_not_fatal(tmp_path, monkeypatch):
    """An auto-detection candidate is a guess, not a user request: one whose
    stat fails is skipped like a missing one, so a single permission-restricted
    file never aborts a run the user never asked to centre on it."""
    from pathlib import Path

    from codedoc.core.scanner import detect_entry_file

    (tmp_path / "index.html").write_text("<html></html>\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    real_stat = Path.stat

    def denied_stat(self, *args, **kwargs):
        if self.name == "index.html":
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)

    resolved = detect_entry_file(tmp_path, None, ["index.html", "main.py"])
    assert resolved is not None
    assert resolved.name == "main.py"
