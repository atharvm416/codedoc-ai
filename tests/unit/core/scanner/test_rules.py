"""Tests organized by feature ownership."""

from __future__ import annotations

import hashlib as _hashlib
import json

import pytest
from codedoc.core.file_division import (
    EMPTY_PLAN_DETAILS_DIGEST,
    canonical_json,
    canonical_stream_digest as _s8_stream_digest,
    MAX_EPHEMERAL_PLAN_DETAIL_ITEMS as _S8_HEAP_CAP,
)
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


def test_scan_diagnostics_rescan_replaces_the_previous_generation(
    tmp_path, monkeypatch, caplog
):
    """Section 5.8 final-generation semantics (replaces the superseded
    rescan-deduplication assertion): a shared ScanDiagnostics instance reused
    for a second ``scan_files`` call builds an INDEPENDENT generation and
    atomically REPLACES the first -- counts, retained details, digests and
    warning allowances are the second generation's alone, never a merge of the
    two. A physical file counted once per generation stays 1 after N scans, and
    every generation gets a fresh warning allowance (so N scans over the same
    unreadable file emit N warning lines, not 1 and not N deduplicated away)."""
    import logging

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
    single_generation_digest_size = None
    single_generation_digest_admission = None
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        for scan_index in range(3):
            files = scan_files(
                tmp_path,
                supported_extensions=[".py"],
                max_file_size_kb=1,
                diagnostics=diagnostics,
            )
            assert {f["rel_path"] for f in files} == {"good.py"}

            # Each generation independently classifies exactly one large and one
            # unreadable file -- REPLACED, never accumulated to 2, 3, ...
            assert diagnostics.files_skipped_large == 1
            assert diagnostics.files_skipped_unreadable == 1
            assert diagnostics.scanner_size_skip["details_total"] == 1
            assert diagnostics.scanner_admission_skip["details_total"] == 1
            assert diagnostics.scanner_size_skip["details_retained"] == 1
            assert diagnostics.scanner_admission_skip["details_retained"] == 1
            assert diagnostics.scanner_size_skip["details_omitted"] == 0

            if scan_index == 0:
                single_generation_digest_size = diagnostics.scanner_size_skip[
                    "details_digest"
                ]
                single_generation_digest_admission = diagnostics.scanner_admission_skip[
                    "details_digest"
                ]
            else:
                # A replaced generation reproduces the single-scan digest
                # exactly -- no doubled descriptor stream.
                assert (
                    diagnostics.scanner_size_skip["details_digest"]
                    == single_generation_digest_size
                )
                assert (
                    diagnostics.scanner_admission_skip["details_digest"]
                    == single_generation_digest_admission
                )

    # Fresh warning allowance per generation: three scans -> three "unreadable"
    # warning lines for blocked.py, not one (dedup) and not zero.
    unreadable_warnings = [
        r for r in caplog.records if "Skipping unreadable file" in r.message
    ]
    assert len(unreadable_warnings) == 3


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


# ===========================================================================
# Section 7 (0.14.7 §5.8): final-generation scanner diagnostics
#
# Each scan_files() call produces ONE fresh, bounded, canonical diagnostic
# generation. A later rescan atomically replaces it -- never merges. Two
# categories, scanner_size_skip and scanner_admission_skip, each publish
# exact details_total / details_retained / details_omitted plus a
# details_digest over the COMPLETE deterministic descriptor stream, with only
# a bounded top-K retained for presentation and warnings capped per class per
# generation.
# ===========================================================================

_SIZE_GUIDANCE = "raise-scan-byte-limit-or-exclude"
_ADMISSION_REASON_GUIDANCE = {
    "unreadable": "fix-permissions-or-exclude",
    "ignored": "adjust-ignore-or-entry",
    "unsupported": "configure-extension-or-entry",
    "missing": "fix-entry-path",
}


def _framed_digest(descriptors):
    """Independent re-implementation of the §5.8 framed streaming digest:
    UTF-8 '[', each descriptor's canonical_json comma-separated, ']'."""
    hasher = _hashlib.sha256()
    hasher.update(b"[")
    for index, descriptor in enumerate(descriptors):
        if index:
            hasher.update(b",")
        hasher.update(canonical_json(descriptor).encode("utf-8"))
    hasher.update(b"]")
    return "sha256:" + hasher.hexdigest()


def _size_descriptor(path, observed, limit):
    return {
        "path": path,
        "phase": "scanner-byte",
        "observed": observed,
        "limit": limit,
        "guidance_code": _SIZE_GUIDANCE,
    }


def _admission_descriptor(path, reason):
    return {
        "path": path,
        "phase": "scanner-admission",
        "reason": reason,
        "guidance_code": _ADMISSION_REASON_GUIDANCE[reason],
    }


def test_s7_reclassified_path_does_not_survive_in_stale_diagnostics(
    tmp_path, monkeypatch
):
    """Proof 2: a path that is large in generation 1 but admitted in
    generation 2 leaves NO stale descriptor, count, or digest contribution."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    target = tmp_path / "swing.py"
    target.write_text("x" * 4096, encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")

    diagnostics = ScanDiagnostics()

    files1 = scan_files(
        tmp_path, supported_extensions=[".py"], max_file_size_kb=1, diagnostics=diagnostics
    )
    assert {f["rel_path"] for f in files1} == {"keep.py"}
    assert diagnostics.scanner_size_skip["details_total"] == 1
    assert diagnostics.scanner_size_skip["details"][0]["path"] == "swing.py"

    # Generation 2: the same file is now under the limit.
    files2 = scan_files(
        tmp_path,
        supported_extensions=[".py"],
        max_file_size_kb=100,
        diagnostics=diagnostics,
    )
    assert {f["rel_path"] for f in files2} == {"keep.py", "swing.py"}
    assert diagnostics.files_skipped_large == 0
    assert diagnostics.scanner_size_skip["details_total"] == 0
    assert diagnostics.scanner_size_skip["details"] == []
    assert diagnostics.scanner_size_skip["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST
    # No stale "swing.py" anywhere in the size category.
    assert all(
        d["path"] != "swing.py" for d in diagnostics.scanner_size_skip["details"]
    )


def test_s7_size_and_admission_exact_counts_and_full_stream_digest(
    tmp_path, monkeypatch
):
    """Proof 3 + 13: exact totals/retained/omitted and a details_digest over
    the COMPLETE canonical descriptor stream for both categories; large-file
    observed/limit are exact bytes."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "a_small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b_big.py").write_text("y" * 3000, encoding="utf-8")
    (tmp_path / "c_big.py").write_text("z" * 5000, encoding="utf-8")
    blocked = tmp_path / "d_blocked.py"
    blocked.write_text("x = 1\n", encoding="utf-8")

    original_stat = type(blocked).stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "d_blocked.py":
            raise PermissionError("nope")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(blocked), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    files = scan_files(
        tmp_path, supported_extensions=[".py"], max_file_size_kb=2, diagnostics=diagnostics
    )
    assert {f["rel_path"] for f in files} == {"a_small.py"}

    size = diagnostics.scanner_size_skip
    assert size["details_total"] == 2
    assert size["details_retained"] == 2
    assert size["details_omitted"] == 0
    # Exact bytes, not KB-rounded.
    by_path = {d["path"]: d for d in size["details"]}
    assert by_path["b_big.py"]["observed"] == 3000
    assert by_path["c_big.py"]["observed"] == 5000
    assert by_path["b_big.py"]["limit"] == 2 * 1024
    # Digest is over the canonical (path-ascending) stream of BOTH descriptors.
    expected_size_digest = _framed_digest(
        [_size_descriptor("b_big.py", 3000, 2048), _size_descriptor("c_big.py", 5000, 2048)]
    )
    assert size["details_digest"] == expected_size_digest

    admission = diagnostics.scanner_admission_skip
    assert admission["details_total"] == 1
    assert admission["details"][0] == _admission_descriptor("d_blocked.py", "unreadable")
    assert admission["details_digest"] == _framed_digest(
        [_admission_descriptor("d_blocked.py", "unreadable")]
    )
    assert diagnostics.files_skipped_unreadable == 1


#
# NOTE: the former four-record ``test_s7_retained_only_digest_mutation_is_detected``
# was replaced by ``test_s7_gap_d_omitted_descriptors_participate_in_the_digest``
# (below), which exercises > MAX_EPHEMERAL_PLAN_DETAIL_ITEMS descriptors so
# omitted records demonstrably participate in the digest -- the four-record
# version proved nothing about omission because all four were retained.
#


def test_s7_snapshot_details_never_exceed_ephemeral_budget():
    """Proof 5: the retained snapshot never exceeds MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    (4096) even when the omitted population is far larger; totals stay exact.
    Driven through the scanner-owned producer seams (no filesystem)."""
    from codedoc.core.file_division import MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    from codedoc.core.scanner import ScanDiagnostics

    count = MAX_EPHEMERAL_PLAN_DETAIL_ITEMS + 137
    diagnostics = ScanDiagnostics()
    diagnostics.begin_scan_generation()
    for i in range(count):
        diagnostics.record("size", f"src/f{i:06d}.py", observed=4242, limit=0)
        diagnostics.record("unreadable", f"pkg/g{i:06d}.py")
    diagnostics.finalize_scan_generation()

    for category, total_key in (
        (diagnostics.scanner_size_skip, "size"),
        (diagnostics.scanner_admission_skip, "admission"),
    ):
        assert category["details_total"] == count, total_key
        assert category["details_retained"] == MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
        assert len(category["details"]) == MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
        assert category["details_omitted"] == count - MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    assert diagnostics.files_skipped_large == count
    assert diagnostics.files_skipped_unreadable == count


def test_s7_size_skip_display_ranking_is_descending_bytes_then_path(tmp_path):
    """Proof 6: retained size descriptors rank by descending actual bytes,
    then path ascending -- while the digest stays path-ascending."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "m_2000.py").write_text("x" * 2000, encoding="utf-8")
    (tmp_path / "a_9000.py").write_text("x" * 9000, encoding="utf-8")
    (tmp_path / "z_5000.py").write_text("x" * 5000, encoding="utf-8")
    (tmp_path / "b_2000.py").write_text("x" * 2000, encoding="utf-8")

    diagnostics = ScanDiagnostics()
    scan_files(
        tmp_path, supported_extensions=[".py"], max_file_size_kb=1, diagnostics=diagnostics
    )
    ranked = [(d["path"], d["observed"]) for d in diagnostics.scanner_size_skip["details"]]
    assert ranked == [
        ("a_9000.py", 9000),
        ("z_5000.py", 5000),
        ("b_2000.py", 2000),  # tie on bytes -> path ascending
        ("m_2000.py", 2000),
    ]
    # Integrity stream is path-ascending, independent of the display ranking.
    assert diagnostics.scanner_size_skip["details_digest"] == _framed_digest(
        [
            _size_descriptor("a_9000.py", 9000, 1024),
            _size_descriptor("b_2000.py", 2000, 1024),
            _size_descriptor("m_2000.py", 2000, 1024),
            _size_descriptor("z_5000.py", 5000, 1024),
        ]
    )


def test_s7_admission_skip_display_ranking_is_frozen_reason_order_then_path():
    """Proof 7: retained admission descriptors rank by the frozen reason order
    (unreadable, ignored, unsupported, missing), then path ascending."""
    from codedoc.core.scanner import ScanDiagnostics

    diagnostics = ScanDiagnostics()
    # Fed via the scanner-owned seam in canonical (path, reason) order.
    diagnostics.begin_scan_generation()
    diagnostics.record("unreadable", "a/x.py")
    diagnostics.record("ignored", "a/y.py")
    diagnostics.record("missing", "b/x.py")
    diagnostics.record("unsupported", "b/z.py")
    diagnostics.record("unreadable", "c/a.py")
    diagnostics.finalize_scan_generation()

    ranked = [
        (d["reason"], d["path"]) for d in diagnostics.scanner_admission_skip["details"]
    ]
    assert ranked == [
        ("unreadable", "a/x.py"),
        ("unreadable", "c/a.py"),
        ("ignored", "a/y.py"),
        ("unsupported", "b/z.py"),
        ("missing", "b/x.py"),
    ]
    assert diagnostics.scanner_admission_skip["details_total"] == 5
    assert all(
        d["guidance_code"] == _ADMISSION_REASON_GUIDANCE[d["reason"]]
        for d in diagnostics.scanner_admission_skip["details"]
    )


def test_s7_traversal_permutation_yields_identical_totals_ranking_and_digests(
    tmp_path, monkeypatch
):
    """Proof 8 (strengthened): permuting the underlying directory iteration
    order changes neither totals, retained ordering, nor digests -- AND the
    resulting digest equals the framed digest independently constructed over
    the descriptors in GLOBAL normalized-path ascending order (not merely some
    stable DFS order). The layout has a directory/file prefix collision so the
    two orders genuinely differ."""
    import random
    from pathlib import Path

    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    layout = [
        ("pkg/big_a.py", 4000),
        ("pkg/big_b.py", 8000),
        ("pkg.py", 6000),        # 'pkg.' 0x2E  < 'pkg/' 0x2F  -> before the subtree
        ("big_c.py", 2000),
        ("big_d.py", 3000),
    ]
    for name, size in layout:
        (tmp_path / name).write_text("x" * size, encoding="utf-8")

    by_bytes = {name: size for name, size in layout}
    canonical_paths = sorted(by_bytes)  # global normalized-POSIX-string order
    expected_digest = _framed_digest(
        [_size_descriptor(p, by_bytes[p], 1024) for p in canonical_paths]
    )
    # display ranking: descending observed bytes, then path
    expected_display = [
        p for p in sorted(canonical_paths, key=lambda p: (-by_bytes[p], p))
    ]

    def run(permute):
        real_iterdir = Path.iterdir

        def permuted_iterdir(self):
            items = list(real_iterdir(self))
            permute(items)
            return iter(items)

        monkeypatch.setattr(Path, "iterdir", permuted_iterdir)
        diagnostics = ScanDiagnostics()
        scan_files(
            tmp_path,
            supported_extensions=[".py"],
            max_file_size_kb=1,
            diagnostics=diagnostics,
        )
        monkeypatch.undo()
        return diagnostics.scanner_size_skip

    forward = run(lambda items: items.sort(key=lambda p: p.name))
    backward = run(lambda items: items.sort(key=lambda p: p.name, reverse=True))
    rng = random.Random(1234)
    shuffled = run(lambda items: rng.shuffle(items))

    for got in (forward, backward, shuffled):
        assert got["details_total"] == 5
        assert [d["path"] for d in got["details"]] == expected_display
        assert got["details_digest"] == expected_digest


def test_s7_warning_cap_is_20_plus_one_aggregate_per_class_per_generation(
    tmp_path, monkeypatch, caplog
):
    """Proof 9 + 10: each path-bearing warning class emits at most 20 lines
    plus exactly one aggregate remainder line, reset per generation."""
    import logging

    from codedoc.core.scanner import (
        MAX_SCANNER_PATH_WARNINGS_PER_CLASS,
        ScanDiagnostics,
        scan_files,
    )

    cap = MAX_SCANNER_PATH_WARNINGS_PER_CLASS
    for i in range(cap + 6):
        (tmp_path / f"big_{i:03d}.py").write_bytes(b"x")
    for i in range(cap + 3):
        (tmp_path / f"blk_{i:03d}.py").write_bytes(b"x")
    (tmp_path / "ok.py").write_bytes(b"")

    original_stat = type(tmp_path).stat

    def fake_stat(self, *args, **kwargs):
        if self.name.startswith("blk_"):
            raise PermissionError("nope")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(
            tmp_path,
            supported_extensions=[".py"],
            max_file_size_kb=0,
            diagnostics=diagnostics,
        )
    large_lines = [r for r in caplog.records if "Skipping large file" in r.message]
    large_aggr = [r for r in caplog.records if "more large file" in r.message]
    unread_lines = [r for r in caplog.records if "Skipping unreadable file" in r.message]
    unread_aggr = [r for r in caplog.records if "more unreadable file" in r.message]
    assert len(large_lines) == cap
    assert len(large_aggr) == 1
    assert f"{6}" in large_aggr[0].message
    assert len(unread_lines) == cap
    assert len(unread_aggr) == 1
    assert f"{3}" in unread_aggr[0].message
    # Totals are still exact despite the capped warnings.
    assert diagnostics.scanner_size_skip["details_total"] == cap + 6
    assert diagnostics.scanner_admission_skip["details_total"] == cap + 3

    # A new generation gets a fresh allowance.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(
            tmp_path,
            supported_extensions=[".py"],
            max_file_size_kb=0,
            diagnostics=diagnostics,
        )
    assert len([r for r in caplog.records if "Skipping large file" in r.message]) == cap


def test_s7_hostile_paths_render_as_one_escaped_field_and_cannot_inject(caplog):
    """Proof 11: newline / tab / escape / quote / backslash / bidi-control /
    non-ASCII in a path render as one JSON-escaped field (ensure_ascii=True) in
    every warning; no raw control byte reaches the log line. Driven through the
    scanner-owned admission seam so it runs on every platform regardless of
    which filenames the local filesystem accepts."""
    import logging

    from codedoc.core.scanner import ScanDiagnostics

    # Only explicit escapes in this source -- newline, tab, ESC, backslash,
    # a double quote, U+202E (RTL override), and U+00E9.
    hostile = chr(101) + chr(118) + chr(10) + chr(105) + chr(108) + chr(9) \
        + chr(34) + "a/b" + chr(92) + "c" + chr(0x202E) + chr(0x1B) + chr(0xE9) + ".py"
    diagnostics = ScanDiagnostics()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        diagnostics.begin_scan_generation()
        diagnostics.record("ignored", hostile)
        diagnostics.finalize_scan_generation()

    records = [r for r in caplog.records if "Skipping" in r.message]
    assert len(records) == 1
    rendered = records[0].getMessage()
    for raw in (chr(10), chr(9), chr(0x1B), chr(7), chr(0x202E)):
        assert raw not in rendered
    normalized = hostile.replace(chr(92), "/")
    assert json.dumps(normalized, ensure_ascii=True) in rendered
    assert (chr(92) + "n") in rendered
    assert (chr(92) + "u202e") in rendered
    assert (chr(92) + "u00e9") in rendered
    desc = diagnostics.scanner_admission_skip["details"][0]
    assert desc["path"] == normalized
    assert ":" not in desc["path"].split("/")[0]
    assert diagnostics.scanner_admission_skip["details_digest"] == _framed_digest(
        [_admission_descriptor(normalized, "ignored")]
    )


def test_s7_no_descriptor_or_warning_contains_absolute_path_or_content(
    tmp_path, monkeypatch, caplog
):
    """Proof 12: descriptors and warnings carry only the normalized
    project-relative POSIX path -- never an absolute path, project root, or
    file content."""
    import logging

    from codedoc.core.scanner import ScanDiagnostics, scan_files

    secret = "SUPER_SECRET_CONTENT_" + "z" * 4096
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "leak.py").write_text(secret, encoding="utf-8")
    blocked = tmp_path / "pkg" / "blk.py"
    blocked.write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")

    original_stat = type(blocked).stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "blk.py":
            raise PermissionError("nope")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(blocked), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(
            tmp_path,
            supported_extensions=[".py"],
            max_file_size_kb=1,
            diagnostics=diagnostics,
        )

    root_str = str(tmp_path)
    for category in (diagnostics.scanner_size_skip, diagnostics.scanner_admission_skip):
        for descriptor in category["details"]:
            blob = canonical_json(descriptor)
            assert root_str not in blob
            assert "SUPER_SECRET_CONTENT_" not in blob
            assert descriptor["path"] in ("pkg/leak.py", "pkg/blk.py")
    for record in caplog.records:
        rendered = record.getMessage()
        assert root_str not in rendered
        assert "SUPER_SECRET_CONTENT_" not in rendered

    assert diagnostics.scanner_size_skip["details"][0]["path"] == "pkg/leak.py"
    assert diagnostics.scanner_admission_skip["details"][0]["path"] == "pkg/blk.py"


def test_s7_duplicate_aliases_obey_existing_scanner_identity_rules(
    tmp_path, monkeypatch
):
    """Proof 14: following a symlink alias to a real oversized file still
    yields at most one descriptor, exactly as admission identity dedup works
    for accepted files."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    real = tmp_path / "real_big.py"
    real.write_text("x" * 8192, encoding="utf-8")
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    alias = tmp_path / "alias_big.py"
    try:
        alias.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/config")

    diagnostics = ScanDiagnostics()
    scan_files(
        tmp_path,
        supported_extensions=[".py"],
        max_file_size_kb=1,
        follow_symlinks=True,
        diagnostics=diagnostics,
    )
    size_paths = [d["path"] for d in diagnostics.scanner_size_skip["details"]]
    assert size_paths.count("real_big.py") + size_paths.count("alias_big.py") == 1
    assert diagnostics.scanner_size_skip["details_total"] == 1


def test_s7_empty_categories_use_the_canonical_empty_stream_digest(tmp_path):
    """Proof 15: a clean scan publishes both categories as empty with the
    canonical '[]' digest -- never a missing key or None."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")

    diagnostics = ScanDiagnostics()
    scan_files(tmp_path, supported_extensions=[".py"], diagnostics=diagnostics)

    for category in (diagnostics.scanner_size_skip, diagnostics.scanner_admission_skip):
        assert category["details"] == []
        assert category["details_total"] == 0
        assert category["details_retained"] == 0
        assert category["details_omitted"] == 0
        assert category["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST
    assert diagnostics.files_skipped_large == 0
    assert diagnostics.files_skipped_unreadable == 0


def test_s7_diagnostic_peak_memory_does_not_scale_with_omitted_population():
    """Proof 16: with the retained budget saturated, growing the OMITTED
    population ~40x must not materially grow diagnostic peak memory -- every
    omitted record is folded into the streaming digest / bounded heap and
    discarded, never retained in an O(N) list/set/string. Driven through the
    scanner-owned producer seam so the measurement is the diagnostics layer,
    not filesystem traversal; nothing is constructed after tracemalloc starts
    beyond the loop counter."""
    import gc
    import tracemalloc

    from codedoc.core.file_division import MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    from codedoc.core.scanner import ScanDiagnostics

    base = MAX_EPHEMERAL_PLAN_DETAIL_ITEMS  # heap saturates in both runs

    def peak_for(total):
        diagnostics = ScanDiagnostics()
        diagnostics.begin_scan_generation()
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        for i in range(total):
            diagnostics.record(
                "size", "src/pkg/mod/f" + format(i, "08d") + ".py", observed=9999, limit=0
            )
        diagnostics.finalize_scan_generation()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert diagnostics.scanner_size_skip["details_total"] == total
        assert diagnostics.scanner_size_skip["details_retained"] == base
        return peak

    small_peak = peak_for(base + 100)
    large_peak = peak_for(base + 20_000)
    # 19,900 extra omitted records must not add O(N) memory: every omitted
    # descriptor is folded into the streaming digest and dropped by the
    # fixed-size heap. An O(N) retained copy of 20k descriptors would cost
    # well over 1 MB; the delta here is heap/GC noise around zero.
    assert large_peak - small_peak < 400_000, (small_peak, large_peak)


def test_s7_walk_records_unreadable_admission_skip_with_frozen_guidance(
    tmp_path, monkeypatch
):
    """The bulk walk feeds reason='unreadable' admission skips with the frozen
    guidance code; the descriptor shape is exactly path/phase/reason/guidance."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    blocked = tmp_path / "no.py"
    blocked.write_text("x = 1\n", encoding="utf-8")
    original_stat = type(blocked).stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "no.py":
            raise PermissionError("nope")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(blocked), "stat", fake_stat)

    diagnostics = ScanDiagnostics()
    scan_files(tmp_path, supported_extensions=[".py"], diagnostics=diagnostics)

    assert diagnostics.scanner_admission_skip["details"] == [
        {
            "path": "no.py",
            "phase": "scanner-admission",
            "reason": "unreadable",
            "guidance_code": "fix-permissions-or-exclude",
        }
    ]


def test_s7_admission_seam_freezes_all_four_reason_guidance_pairs():
    """The scanner-owned seam (for later pipeline explicit-entry evidence)
    records ignored / unsupported / missing / unreadable with exactly the
    frozen guidance pairs and rejects an unknown reason."""
    from codedoc.core.scanner import ScanDiagnostics

    diagnostics = ScanDiagnostics()
    diagnostics.begin_scan_generation()
    for path, reason in [
        ("app/a.py", "unreadable"),
        ("app/b.py", "ignored"),
        ("app/c.py", "unsupported"),
        ("app/d.py", "missing"),
    ]:
        diagnostics.record(reason, path)
    diagnostics.finalize_scan_generation()

    got = {d["path"]: (d["reason"], d["guidance_code"], d["phase"])
           for d in diagnostics.scanner_admission_skip["details"]}
    assert got == {
        "app/a.py": ("unreadable", "fix-permissions-or-exclude", "scanner-admission"),
        "app/b.py": ("ignored", "adjust-ignore-or-entry", "scanner-admission"),
        "app/c.py": ("unsupported", "configure-extension-or-entry", "scanner-admission"),
        "app/d.py": ("missing", "fix-entry-path", "scanner-admission"),
    }
    assert diagnostics.scanner_admission_skip["details_digest"] == _framed_digest(
        [
            _admission_descriptor("app/a.py", "unreadable"),
            _admission_descriptor("app/b.py", "ignored"),
            _admission_descriptor("app/c.py", "unsupported"),
            _admission_descriptor("app/d.py", "missing"),
        ]
    )
    # An unknown reason is rejected with ValueError (checked on a fresh, still
    # open generation -- a record after finalize fails closed with RuntimeError,
    # which is a separate lifecycle contract).
    other = ScanDiagnostics()
    other.begin_scan_generation()
    with pytest.raises(ValueError):
        other.record("not-a-real-reason", "app/x.py")


def test_s7_no_run_lifetime_seen_sets_remain_on_scan_diagnostics():
    """§5.8: the run-lifetime _large_seen / _unreadable_seen dedup sets are
    removed."""
    from codedoc.core.scanner import ScanDiagnostics

    diagnostics = ScanDiagnostics()
    assert not hasattr(diagnostics, "_large_seen")
    assert not hasattr(diagnostics, "_unreadable_seen")


def test_s7_existing_admission_and_positional_behaviour_is_unchanged(tmp_path):
    """Proof 17: admission (which files become descriptors), positional
    compatibility, and generated-target exclusion are untouched by the
    diagnostics rework."""
    from codedoc.core.scanner import exclude_path_key, scan_files

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "codedoc.json").write_text("{}\n", encoding="utf-8")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "util.py").write_text("y = 2\n", encoding="utf-8")

    # Positional list still works.
    positional = scan_files(tmp_path, [".py"])
    assert {f["rel_path"] for f in positional} == {"main.py", "pkg/util.py"}

    # Exact generated-target exclusion still works.
    excluded = scan_files(
        tmp_path,
        extension_language_map={".py": "python", ".json": "json"},
        exclude_paths={exclude_path_key(tmp_path / "codedoc.json")},
    )
    assert {f["rel_path"] for f in excluded} == {"main.py", "pkg/util.py"}


# ===========================================================================
# Section 7 correction round: Defects A / B / C and verification Gap D
# ===========================================================================


def test_s7_defect_a_canonical_digest_is_global_normalized_path_ascending(tmp_path):
    """DEFECT A (P1): the digest stream must be GLOBAL normalized-path ascending,
    not merely per-directory-sorted DFS order. With prefix collisions -- the
    audit's counterexample 'a.py' vs 'a/z.py', plus a dir-prefixed path that
    sorts before a sibling filename -- the DFS order and the string order
    differ. The published digest must equal the framed digest over the
    descriptors in global ``sorted()`` order for BOTH categories."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    # Files 'a.py', 'a/z.py', 'ab.py'. Global POSIX-string order is
    # 'a.py' < 'a/z.py' < 'ab.py'  ('.' 0x2E < '/' 0x2F < 'b' 0x62), but a plain
    # per-directory name sort descends into 'a/' before visiting 'a.py'/'ab.py'.
    (tmp_path / "a").mkdir()
    (tmp_path / "a.py").write_bytes(b"x")
    (tmp_path / "a" / "z.py").write_bytes(b"x")
    (tmp_path / "ab.py").write_bytes(b"x")

    diagnostics = ScanDiagnostics()
    scan_files(
        tmp_path, supported_extensions=[".py"], max_file_size_kb=0, diagnostics=diagnostics
    )

    ordered = ["a.py", "a/z.py", "ab.py"]
    assert sorted(ordered) == ordered  # sanity: this IS the string order
    assert diagnostics.scanner_size_skip["details_total"] == 3
    assert diagnostics.scanner_size_skip["details_digest"] == _framed_digest(
        [_size_descriptor(p, 1, 0) for p in ordered]
    )

    # Admission category, same collision layout, all three unreadable.
    real_stat = type(tmp_path).stat

    def deny(self, *a, **k):
        if self.suffix == ".py":
            raise PermissionError("nope")
        return real_stat(self, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(type(tmp_path), "stat", deny)
    try:
        adm = ScanDiagnostics()
        scan_files(tmp_path, supported_extensions=[".py"], diagnostics=adm)
    finally:
        mp.undo()
    assert adm.scanner_admission_skip["details_total"] == 3
    assert adm.scanner_admission_skip["details_digest"] == _framed_digest(
        [_admission_descriptor(p, "unreadable") for p in ordered]
    )


def test_s7_defect_a_boundary_dir_prefix_vs_sibling_filename(tmp_path):
    """DEFECT A boundary: 'a-b.py' ('-' 0x2D) sorts before 'a/...'; 'ab.py'
    ('b' 0x62) sorts after 'a/...'. Both relationships hold in the digest
    stream and in the retained display order."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "y.py").write_bytes(b"x")
    (tmp_path / "a" / "z.py").write_bytes(b"x")
    (tmp_path / "a-b.py").write_bytes(b"x")
    (tmp_path / "ab.py").write_bytes(b"x")

    diagnostics = ScanDiagnostics()
    scan_files(
        tmp_path, supported_extensions=[".py"], max_file_size_kb=0, diagnostics=diagnostics
    )
    ordered = ["a-b.py", "a/y.py", "a/z.py", "ab.py"]
    assert sorted(ordered) == ordered
    assert diagnostics.scanner_size_skip["details_digest"] == _framed_digest(
        [_size_descriptor(p, 1, 0) for p in ordered]
    )
    # all observed equal -> retained display order is path order
    assert [d["path"] for d in diagnostics.scanner_size_skip["details"]] == ordered


def test_s7_defect_b_record_rejects_non_project_relative_paths(caplog):
    """DEFECT B (P1 privacy/integrity): record() must reject rooted, UNC,
    drive-qualified, parent-traversing, empty and dot-only paths -- never
    serialize or log them -- and must canonicalize redundant separators and
    '.' segments in an otherwise-valid relative path."""
    import logging

    from codedoc.core.scanner import ScanDiagnostics

    rejected = [
        "C:\\private\\secret.py",
        "C:/private/secret.py",
        "C:private.py",
        "\\\\server\\share\\secret.py",
        "//server/share/secret.py",
        "/private/secret.py",
        "../secret.py",
        "a/../../secret.py",
        "",
        "   ",
        ".",
        "..",
        "./.",
    ]
    for bad in rejected:
        d = ScanDiagnostics()
        d.begin_scan_generation()
        with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
            caplog.clear()
            with pytest.raises(ValueError):
                d.record("ignored", bad)
        leaked = [
            r.getMessage() for r in caplog.records
            if "secret" in r.getMessage()
            or "private" in r.getMessage()
            or "server" in r.getMessage()
        ]
        assert leaked == [], leaked
        d.finalize_scan_generation()
        assert d.scanner_admission_skip["details_total"] == 0

    # Safely-normalizable relative paths are accepted and canonicalized.
    d = ScanDiagnostics()
    d.begin_scan_generation()
    d.record("ignored", "a//b/./c.py")
    d.record("ignored", "d\\e/f.py")
    d.finalize_scan_generation()
    assert [x["path"] for x in d.scanner_admission_skip["details"]] == [
        "a/b/c.py",
        "d/e/f.py",
    ]

    # A hostile-but-valid relative filename is preserved and rendered escaped.
    d = ScanDiagnostics()
    d.begin_scan_generation()
    hostile = "ev" + chr(10) + "il" + chr(9) + chr(34) + chr(0x202E) + ".py"
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        caplog.clear()
        d.record("ignored", hostile)
    d.finalize_scan_generation()
    assert d.scanner_admission_skip["details"][0]["path"] == hostile
    line = next(
        r.getMessage() for r in caplog.records if "Skipping ignored" in r.getMessage()
    )
    assert chr(10) not in line and chr(9) not in line and chr(0x202E) not in line
    assert json.dumps(hostile, ensure_ascii=True) in line


def test_s7_defect_c_warning_cap_is_per_reason_class(caplog):
    """DEFECT C (P2): each path-bearing warning kind (large / unreadable /
    ignored / unsupported / missing) is its own warning class -- its own
    20-line allowance and its own exact remainder line with the correct noun.
    One noisy class must not consume another's allowance; a fresh generation
    resets every allowance; totals/digests are unaffected by suppression."""
    import logging

    from codedoc.core.scanner import ScanDiagnostics

    d = ScanDiagnostics()
    d.begin_scan_generation()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        for i in range(21):
            d.record("ignored", f"pkg/ign_{i:03d}.py")
        for i in range(21):
            d.record("unsupported", f"pkg/uns_{i:03d}.py")
        d.finalize_scan_generation()

    msgs = [r.getMessage() for r in caplog.records]
    assert sum("Skipping ignored file" in m for m in msgs) == 20
    assert sum("Skipping unsupported file" in m for m in msgs) == 20
    assert sum(m == "1 more ignored file(s) not shown" for m in msgs) == 1
    assert sum(m == "1 more unsupported file(s) not shown" for m in msgs) == 1
    # Nothing may be described as unreadable / large -- no such records exist.
    assert not any(("unreadable" in m) or ("large file" in m) for m in msgs)

    cat = d.scanner_admission_skip
    assert cat["details_total"] == 42
    assert cat["details_digest"] == _framed_digest(
        [_admission_descriptor(f"pkg/ign_{i:03d}.py", "ignored") for i in range(21)]
        + [_admission_descriptor(f"pkg/uns_{i:03d}.py", "unsupported") for i in range(21)]
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        d.begin_scan_generation()
        for i in range(21):
            d.record("ignored", f"pkg/ign2_{i:03d}.py")
        d.finalize_scan_generation()
    msgs2 = [r.getMessage() for r in caplog.records]
    assert sum("Skipping ignored file" in m for m in msgs2) == 20
    assert sum(m == "1 more ignored file(s) not shown" for m in msgs2) == 1


def test_s7_gap_d_omitted_descriptors_participate_in_the_digest():
    """GAP D (P1 verification): with more than MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    descriptors the published digest must equal the framed digest over the
    COMPLETE canonical stream and must DIFFER from the retained-only digest --
    proving omitted descriptors are hashed. Both categories."""
    from codedoc.core.file_division import MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    from codedoc.core.scanner import ScanDiagnostics

    n = MAX_EPHEMERAL_PLAN_DETAIL_ITEMS + 250

    d = ScanDiagnostics()
    d.begin_scan_generation()
    for i in range(n):
        d.record("size", f"src/s{i:07d}.py", observed=1, limit=0)
    d.finalize_scan_generation()
    cat = d.scanner_size_skip
    assert cat["details_total"] == n
    assert cat["details_retained"] == MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    assert cat["details_omitted"] == n - MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    full = _framed_digest(
        _size_descriptor(f"src/s{i:07d}.py", 1, 0) for i in range(n)
    )
    retained_only = _framed_digest(
        _size_descriptor(f"src/s{i:07d}.py", 1, 0)
        for i in range(MAX_EPHEMERAL_PLAN_DETAIL_ITEMS)
    )
    assert cat["details_digest"] == full
    assert cat["details_digest"] != retained_only

    d = ScanDiagnostics()
    d.begin_scan_generation()
    for i in range(n):
        d.record("ignored", f"src/a{i:07d}.py")
    d.finalize_scan_generation()
    cat = d.scanner_admission_skip
    assert cat["details_total"] == n
    assert cat["details_retained"] == MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    assert cat["details_omitted"] == n - MAX_EPHEMERAL_PLAN_DETAIL_ITEMS
    full = _framed_digest(
        _admission_descriptor(f"src/a{i:07d}.py", "ignored") for i in range(n)
    )
    retained_only = _framed_digest(
        _admission_descriptor(f"src/a{i:07d}.py", "ignored")
        for i in range(MAX_EPHEMERAL_PLAN_DETAIL_ITEMS)
    )
    assert cat["details_digest"] == full
    assert cat["details_digest"] != retained_only


# ===========================================================================
# Section 7 correction round 2: Defects E / F1 / F2 / G
# ===========================================================================


def _seam_gen():
    from codedoc.core.scanner import ScanDiagnostics

    d = ScanDiagnostics()
    d.begin_scan_generation()
    return d


def test_s7_defect_e_out_of_order_size_records_fail_closed():
    """DEFECT E (P1): a size record whose canonical key (normalized path) is
    LESS than the previous one is rejected before it can touch the digest,
    totals, retained heap, or warning counters."""
    d = _seam_gen()
    d.record("size", "z.py", observed=1, limit=0)
    with pytest.raises(ValueError):
        d.record("size", "a.py", observed=1, limit=0)
    d.finalize_scan_generation()
    cat = d.scanner_size_skip
    assert cat["details_total"] == 1
    assert [x["path"] for x in cat["details"]] == ["z.py"]
    assert cat["details_digest"] == _framed_digest([_size_descriptor("z.py", 1, 0)])


def test_s7_defect_e_out_of_order_admission_reasons_fail_closed():
    """DEFECT E: the audit's example -- record('missing','z.py') then
    record('ignored','a.py') -- is rejected; the canonical (path, reason) order
    is accepted."""
    d = _seam_gen()
    d.record("missing", "z.py")
    with pytest.raises(ValueError):
        d.record("ignored", "a.py")
    d.finalize_scan_generation()
    assert d.scanner_admission_skip["details_total"] == 1
    assert d.scanner_admission_skip["details_digest"] == _framed_digest(
        [_admission_descriptor("z.py", "missing")]
    )

    d = _seam_gen()
    d.record("ignored", "a.py")
    d.record("missing", "z.py")
    d.finalize_scan_generation()
    assert d.scanner_admission_skip["details_digest"] == _framed_digest(
        [_admission_descriptor("a.py", "ignored"), _admission_descriptor("z.py", "missing")]
    )


def test_s7_defect_e_same_path_admission_secondary_reason_order():
    """DEFECT E: at one path, the four reasons in frozen order
    (unreadable < ignored < unsupported < missing) are accepted and hashed in
    that order."""
    d = _seam_gen()
    for reason in ("unreadable", "ignored", "unsupported", "missing"):
        d.record(reason, "x/y.py")
    d.finalize_scan_generation()
    assert d.scanner_admission_skip["details_total"] == 4
    assert d.scanner_admission_skip["details_digest"] == _framed_digest(
        [_admission_descriptor("x/y.py", r)
         for r in ("unreadable", "ignored", "unsupported", "missing")]
    )


def test_s7_defect_e_decreasing_same_path_secondary_reason_fails():
    """DEFECT E: at one path, a reason that sorts BEFORE the previous reason is
    rejected."""
    d = _seam_gen()
    d.record("ignored", "x.py")           # reason index 1
    with pytest.raises(ValueError):
        d.record("unreadable", "x.py")    # reason index 0 -> decreasing
    d.finalize_scan_generation()
    assert d.scanner_admission_skip["details_total"] == 1


def test_s7_defect_e_equal_path_reason_duplicates_are_legal_and_each_hashed():
    """DEFECT E: an equal canonical key is permitted -- each duplicate
    occurrence still affects the digest and the total."""
    d = _seam_gen()
    d.record("ignored", "x.py")
    d.record("ignored", "x.py")
    d.finalize_scan_generation()
    cat = d.scanner_admission_skip
    assert cat["details_total"] == 2
    assert cat["details_digest"] == _framed_digest(
        [_admission_descriptor("x.py", "ignored"), _admission_descriptor("x.py", "ignored")]
    )
    assert cat["details_digest"] != _framed_digest(
        [_admission_descriptor("x.py", "ignored")]
    )


def test_s7_defect_e_failure_causes_no_partial_mutation(caplog):
    """DEFECT E: a rejected out-of-order record mutates nothing -- digest,
    total, retained heap, warning counter and warning output all reflect only
    the accepted records."""
    import logging

    d = _seam_gen()
    d.record("size", "m.py", observed=5, limit=0)
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        caplog.clear()
        with pytest.raises(ValueError):
            d.record("size", "a.py", observed=9, limit=0)
    # No warning emitted for the rejected path.
    assert not any("a.py" in r.getMessage() for r in caplog.records)
    d.finalize_scan_generation()
    cat = d.scanner_size_skip
    assert cat["details_total"] == 1
    assert cat["details_retained"] == 1
    assert [x["path"] for x in cat["details"]] == ["m.py"]
    assert cat["details_digest"] == _framed_digest([_size_descriptor("m.py", 5, 0)])


def test_s7_defect_e_filesystem_walk_satisfies_the_ordering_guard(tmp_path, monkeypatch):
    """DEFECT E: the real filesystem walk -- oversized files AND unreadable
    files in a directory/file prefix-collision layout -- feeds records in
    canonical order, so the guard never trips and both digests are canonical."""
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "a").mkdir()
    (tmp_path / "a.py").write_bytes(b"x")
    (tmp_path / "a" / "z.py").write_bytes(b"x")
    (tmp_path / "ab.py").write_bytes(b"x")
    (tmp_path / "b.py").write_bytes(b"x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c.py").write_bytes(b"x")

    real_stat = type(tmp_path).stat

    def deny(self, *a, **k):
        if self.name in ("b.py", "c.py"):
            raise PermissionError("nope")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(type(tmp_path), "stat", deny)
    d = ScanDiagnostics()
    scan_files(tmp_path, supported_extensions=[".py"], max_file_size_kb=0, diagnostics=d)

    assert d.scanner_size_skip["details_digest"] == _framed_digest(
        [_size_descriptor(p, 1, 0) for p in ["a.py", "a/z.py", "ab.py"]]
    )
    assert d.scanner_admission_skip["details_digest"] == _framed_digest(
        [_admission_descriptor(p, "unreadable") for p in ["b.py", "b/c.py"]]
    )


def test_s7_defect_e_finalized_generation_rejects_late_records():
    """DEFECT E lifecycle: once a generation is finalized, a late record fails
    closed -- it does not silently append to a non-canonical combined stream. A
    fresh begin_scan_generation() is required."""
    d = _seam_gen()
    d.record("unreadable", "m.py")
    d.finalize_scan_generation()
    with pytest.raises(RuntimeError):
        d.record("ignored", "a.py")
    # unchanged published generation
    assert d.scanner_admission_skip["details_total"] == 1


# --------------------------------------------------------------------------
# DEFECT F1 -- unreadable-directory warning
# --------------------------------------------------------------------------


def test_s7_defect_f1_unreadable_directory_warnings_bounded_relative_escaped(
    tmp_path, monkeypatch, caplog
):
    """DEFECT F1 (P1 privacy / P2 bounded): >20 unreadable directories produce
    exactly 20 project-relative JSON-escaped path lines plus one exact
    aggregate; no absolute root, no raw exception text, no injected control
    characters, no invented admission descriptor; allowance resets next
    generation."""
    import logging

    from pathlib import Path

    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "keep.py").write_bytes(b"x = 1\n")
    for i in range(25):
        (tmp_path / f"d{i:02d}").mkdir()
        (tmp_path / f"d{i:02d}" / "inner.py").write_bytes(b"y = 2\n")

    real_iterdir = Path.iterdir

    def blocked_iterdir(self):
        if self.name.startswith("d") and self.parent == tmp_path:
            raise PermissionError("boom\ndrop\ttable\x1b[2J secret-root-detail")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", blocked_iterdir)

    diags = ScanDiagnostics()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(tmp_path, supported_extensions=[".py"], diagnostics=diags)

    msgs = [r.getMessage() for r in caplog.records]
    dir_lines = [m for m in msgs if m.startswith("Skipping unreadable directory")]
    aggr = [m for m in msgs if "more unreadable director" in m]
    assert len(dir_lines) == 20
    assert len(aggr) == 1
    assert aggr[0] == "5 more unreadable directory(ies) not shown"
    root_str = str(tmp_path)
    for m in msgs:
        assert root_str not in m
        assert "boom" not in m and "secret-root-detail" not in m
        assert "\n" not in m and "\t" not in m and "\x1b" not in m
    # Each rendered path is exactly one JSON string field, relative.
    for m in dir_lines:
        field = m.split(": ", 1)[1]
        assert field.startswith('"') and field.endswith('"')
        assert json.loads(field) in {f"d{i:02d}" for i in range(25)}
    # No directory turned into an admission descriptor.
    assert all(x["reason"] != "directory"
               for x in diags.scanner_admission_skip["details"])
    assert diags.scanner_admission_skip["details_total"] == 0

    # Allowance resets for the next authoritative generation.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(tmp_path, supported_extensions=[".py"], diagnostics=diags)
    again = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("Skipping unreadable directory")]
    assert len(again) == 20


def test_s7_defect_f1_root_enumeration_failure_uses_dot(tmp_path, monkeypatch, caplog):
    """DEFECT F1: failing to enumerate the project root itself renders the path
    as '.', never the absolute root."""
    import logging

    from pathlib import Path

    from codedoc.core.scanner import ScanDiagnostics, scan_files

    real_iterdir = Path.iterdir

    def blocked_iterdir(self):
        if self == tmp_path:
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", blocked_iterdir)
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(tmp_path, supported_extensions=[".py"], diagnostics=ScanDiagnostics())
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("Skipping unreadable directory")]
    assert lines == ['Skipping unreadable directory: "."']
    assert str(tmp_path) not in "".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# DEFECT F2 -- detect_entry_file missing-entry warning
# --------------------------------------------------------------------------


def test_s7_defect_f2_missing_entry_warning_is_relative_and_escaped(
    tmp_path, caplog
):
    """DEFECT F2 (P1 privacy): the missing-entry warning must not serialize the
    absolute project root, must render the hint as one JSON-escaped field, and
    must not let hostile control characters inject a second log line. The
    return value is unchanged (None for a missing entry)."""
    import logging

    from codedoc.core.scanner import detect_entry_file

    (tmp_path / "real.py").write_bytes(b"x = 1\n")
    hint = "ev\nil\t\x1b" + chr(0x202E) + "no/such.py"

    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        result = detect_entry_file(tmp_path, hint, None)

    assert result is None
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    line = warnings[0]
    assert str(tmp_path) not in line
    for raw in ("\n", "\t", "\x1b", chr(0x202E)):
        assert raw not in line
    field = line.split("Specified entry file ", 1)[1].split(" not found", 1)[0]
    assert field.startswith('"') and field.endswith('"')
    assert json.loads(field) == hint.replace("\\", "/")


# --------------------------------------------------------------------------
# DEFECT G -- finalization lifecycle
# --------------------------------------------------------------------------


def test_s7_defect_g_repeated_finalize_does_not_duplicate_aggregate(caplog):
    """DEFECT G (P2): a second finalize with no new records emits nothing and
    changes no published state."""
    import logging

    d = _seam_gen()
    for i in range(21):
        d.record("ignored", f"pkg/i_{i:03d}.py")
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        caplog.clear()
        d.finalize_scan_generation()
        first_digest = d.scanner_admission_skip["details_digest"]
        d.finalize_scan_generation()
    msgs = [r.getMessage() for r in caplog.records]
    assert sum(m == "1 more ignored file(s) not shown" for m in msgs) == 1
    assert d.scanner_admission_skip["details_digest"] == first_digest
    assert d.scanner_admission_skip["details_total"] == 21


def test_s7_defect_g_record_after_finalize_fails_closed():
    """DEFECT G: a record after finalization fails closed -- never silently
    appended to the closed generation."""
    d = _seam_gen()
    d.record("size", "a.py", observed=1, limit=0)
    d.finalize_scan_generation()
    with pytest.raises(RuntimeError):
        d.record("size", "b.py", observed=1, limit=0)
    assert d.scanner_size_skip["details_total"] == 1


def test_s7_defect_g_begin_after_finalize_creates_clean_replacement(caplog):
    """DEFECT G: an explicit begin after finalize starts a clean replacement
    generation with every warning allowance reset."""
    import logging

    d = _seam_gen()
    for i in range(21):
        d.record("ignored", f"g1/i_{i:03d}.py")
    d.record("size", "g1/a.py", observed=1, limit=0)
    d.finalize_scan_generation()
    gen1_size = dict(d.scanner_size_skip)

    d.begin_scan_generation()
    # Published state is still generation 1 until generation 2 is finalized.
    assert d.scanner_size_skip == gen1_size
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        caplog.clear()
        for i in range(21):
            d.record("ignored", f"g2/i_{i:03d}.py")
        d.record("size", "g2/b.py", observed=2, limit=0)
        d.finalize_scan_generation()
    msgs = [r.getMessage() for r in caplog.records]
    # Fresh allowance: 20 lines + one remainder again.
    assert sum("Skipping ignored file" in m for m in msgs) == 20
    assert sum(m == "1 more ignored file(s) not shown" for m in msgs) == 1
    assert [x["path"] for x in d.scanner_size_skip["details"]] == ["g2/b.py"]
    assert d.scanner_size_skip["details_digest"] == _framed_digest(
        [_size_descriptor("g2/b.py", 2, 0)]
    )


def test_s7_defect_g_published_state_survives_unfinished_replacement():
    """DEFECT G: after generation 1 is finalized, opening generation 2 and NOT
    finalizing it leaves generation 1's published categories intact."""
    d = _seam_gen()
    d.record("size", "one.py", observed=1, limit=0)
    d.record("unreadable", "two.py")
    d.finalize_scan_generation()
    gen1_size = dict(d.scanner_size_skip)
    gen1_adm = dict(d.scanner_admission_skip)

    d.begin_scan_generation()
    d.record("size", "three.py", observed=1, limit=0)
    # generation 2 never finalized
    assert d.scanner_size_skip == gen1_size
    assert d.scanner_admission_skip == gen1_adm


def test_s7_defect_g_aggregate_appears_exactly_once_per_class_per_generation(caplog):
    """DEFECT G: with several noisy classes in one generation, each aggregate
    remainder line appears exactly once; a second finalize adds none."""
    import logging

    d = _seam_gen()
    for i in range(21):
        d.record("ignored", f"p/i_{i:03d}.py")
    for i in range(21):
        d.record("unsupported", f"p/u_{i:03d}.py")
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        caplog.clear()
        d.finalize_scan_generation()
        d.finalize_scan_generation()
    msgs = [r.getMessage() for r in caplog.records]
    assert sum(m == "1 more ignored file(s) not shown" for m in msgs) == 1
    assert sum(m == "1 more unsupported file(s) not shown" for m in msgs) == 1


# ===========================================================================
# Section 7 final completion: F2 -- detect_entry_file() prevalidation + normalization
# ===========================================================================

_F2_INVALID_HINTS = [
    "C:\\private\\secret.py",     # drive-qualified (Windows, backslash)
    "C:/private/secret.py",       # drive-qualified (Windows, slash)
    "C:private.py",               # drive-relative
    "\\\\server\\share\\secret.py",  # UNC
    "//server/share/secret.py",   # UNC / network root
    "/private/secret.py",         # POSIX / rooted
    "../secret.py",               # parent traversal
    "a/../../secret.py",          # parent traversal mid-path
    "   ",                        # whitespace-only explicit hint
    ".",                          # dot-only
    "..",                         # dot-dot
    "./.",                        # dot segments only
]


@pytest.mark.parametrize("hint", _F2_INVALID_HINTS)
def test_s7_f2_invalid_entry_hint_rejected_before_any_filesystem_probe(
    tmp_path, monkeypatch, caplog, hint
):
    """DEFECT F2 (P1): an explicit --entry hint that is drive-qualified,
    drive-relative, UNC, rooted, parent-traversing, or dot-only is rejected
    with ConfigError BEFORE any Path.exists/stat/resolve probe, emits no
    path-bearing warning, and never echoes the absolute/private input or the
    project root."""
    import logging
    from pathlib import Path

    from codedoc.core.scanner import detect_entry_file
    from codedoc.utils.errors import ConfigError

    probed: list = []
    monkeypatch.setattr(Path, "exists", lambda self, *a, **k: probed.append(("exists", str(self))))
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: probed.append(("stat", str(self))))
    monkeypatch.setattr(
        Path, "resolve", lambda self, *a, **k: (probed.append(("resolve", str(self))), self)[1]
    )

    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        with pytest.raises(ConfigError):
            detect_entry_file(tmp_path, hint, None)

    assert probed == [], probed
    joined = "".join(r.getMessage() for r in caplog.records)
    assert "Specified entry file" not in joined
    assert not any(w in joined for w in ("secret", "private", "server"))
    assert str(tmp_path) not in joined


def test_s7_f2_normalizable_relative_hint_uses_canonical_form_for_lookup_and_warning(
    tmp_path, caplog
):
    """DEFECT F2: a valid relative hint with redundant separators / '.' segments
    or backslashes is canonicalized to project-relative POSIX and that canonical
    form is used for BOTH the candidate lookup and the missing-entry warning."""
    import logging

    from codedoc.core.scanner import detect_entry_file

    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "sub" / "found.py").write_bytes(b"x = 1\n")

    # Lookup uses the canonical form.
    got = detect_entry_file(tmp_path, "pkg//sub/./found.py", None)
    assert got == tmp_path / "pkg" / "sub" / "found.py"
    got2 = detect_entry_file(tmp_path, "pkg\\sub\\found.py", None)
    assert got2 == tmp_path / "pkg" / "sub" / "found.py"

    # Missing valid entry -> None, exactly one warning, canonical relative field.
    for raw in ("pkg//sub/./missing.py", "pkg\\sub\\missing.py", "./pkg/sub/missing.py"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
            result = detect_entry_file(tmp_path, raw, None)
        assert result is None
        warnings = [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Specified entry file")
        ]
        assert len(warnings) == 1
        field = warnings[0].split("Specified entry file ", 1)[1].split(" not found", 1)[0]
        assert json.loads(field) == "pkg/sub/missing.py"
        assert str(tmp_path) not in warnings[0]


def test_s7_f2_hostile_but_valid_relative_hint_is_one_escaped_field(tmp_path, caplog):
    """DEFECT F2: a valid relative hint whose filename bytes are hostile
    (newline / tab / ESC / bidi / non-ASCII) is still accepted, rendered as
    exactly one JSON-escaped field, and cannot inject a second log line."""
    import logging

    from codedoc.core.scanner import detect_entry_file

    hint = "pkg/ev" + chr(10) + "il" + chr(9) + chr(0x1B) + chr(0x202E) + chr(0xE9) + ".py"
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        result = detect_entry_file(tmp_path, hint, None)
    assert result is None
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("Specified entry file")]
    assert len(lines) == 1
    line = lines[0]
    for rawch in (chr(10), chr(9), chr(0x1B), chr(0x202E)):
        assert rawch not in line
    field = line.split("Specified entry file ", 1)[1].split(" not found", 1)[0]
    assert field.startswith('"') and field.endswith('"')
    assert json.loads(field) == hint
    assert str(tmp_path) not in line


# ===========================================================================
# Section 7 final bypass fix: leading removable dot segments must not hide a
# Windows drive prefix from detect_entry_file() validation.
# ===========================================================================

_F2_DOT_DRIVE_BYPASSES = [
    "./C:/private/secret.py",
    ".\\C:\\private\\secret.py",
    "./D:/private/secret.py",
    ".\\D:\\private\\secret.py",
    "./C:private.py",
    "././C:/private/secret.py",
    ".\\.\\D:\\private\\secret.py",
    "./c:/private/secret.py",        # lowercase drive
    ".\\d:\\private\\secret.py",     # lowercase drive
    "././c:private.py",              # lowercase drive-relative
    "./C:/x.py",                     # same-drive short form
    "./E:/other/thing.py",           # different drive
]


@pytest.mark.parametrize("hint", _F2_DOT_DRIVE_BYPASSES)
def test_s7_f2_dot_prefixed_drive_bypass_rejected_before_probe(
    tmp_path, monkeypatch, caplog, hint
):
    """FINAL BYPASS (P1): a leading removable '.' (or '.\\') segment must not
    let a Windows drive prefix through -- validation is on the CANONICAL
    components, so `./C:/private/secret.py` is rejected with ConfigError before
    `root / entry_rel`, with zero Path.exists/stat/resolve probes and zero
    warnings, and never echoes the private path or the project root."""
    import logging
    from pathlib import Path

    from codedoc.core.scanner import detect_entry_file
    from codedoc.utils.errors import ConfigError

    probed: list = []
    monkeypatch.setattr(Path, "exists", lambda self, *a, **k: probed.append(("exists", str(self))))
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: probed.append(("stat", str(self))))
    monkeypatch.setattr(
        Path, "resolve", lambda self, *a, **k: (probed.append(("resolve", str(self))), self)[1]
    )

    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        with pytest.raises(ConfigError):
            detect_entry_file(tmp_path, hint, None)

    assert probed == [], probed
    joined = "".join(r.getMessage() for r in caplog.records)
    assert "Specified entry file" not in joined
    assert not any(w in joined for w in ("secret", "private", "other", "thing"))
    assert str(tmp_path) not in joined


def test_s7_f2_leading_dot_segments_stay_valid_for_ordinary_relative_paths(
    tmp_path, caplog
):
    """FINAL BYPASS positive control: leading removable '.' / '.\\' segments in
    front of an ordinary relative path remain valid and canonicalize normally
    -- for both lookup and the missing-entry warning."""
    import logging

    from codedoc.core.scanner import detect_entry_file

    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "sub" / "found.py").write_bytes(b"x = 1\n")

    target = tmp_path / "pkg" / "sub" / "found.py"
    for raw in (
        "./pkg/sub/found.py",
        "././pkg//sub/./found.py",
        ".\\pkg\\sub\\found.py",
        ".\\.\\pkg\\sub\\found.py",
    ):
        assert detect_entry_file(tmp_path, raw, None) == target, raw

    for raw in ("./pkg/sub/missing.py", "././pkg//sub/./missing.py", ".\\pkg\\sub\\missing.py"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
            result = detect_entry_file(tmp_path, raw, None)
        assert result is None
        lines = [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Specified entry file")
        ]
        assert len(lines) == 1
        field = lines[0].split("Specified entry file ", 1)[1].split(" not found", 1)[0]
        assert json.loads(field) == "pkg/sub/missing.py"
        assert str(tmp_path) not in lines[0]


# ===========================================================================
# Section 5.8 correction round 2: an EXPLICIT entry that yields no admitted
# source publishes bounded, canonical, full-stream-digested scanner evidence
# for THAT TARGET'S OWN candidates -- every one size-skipped, unreadable,
# configured-ignored (skip_dirs / hidden / ignore_paths), unsupported, or
# missing. An empty explicit directory has NO invented path record. A directory
# full of excluded candidates is bounded top-K with an exact total and a
# full-stream digest over every candidate, permutation-stable, never an
# unbounded list. No required descriptor is dropped for a canonical-order
# collision (no catch-and-discard). Final-generation atomic replacement holds.
# ===========================================================================


def _s8_run_scan(root, hint, *, ignore_paths=None, skip_dirs=None, max_file_size_kb=500):
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    diag = ScanDiagnostics()
    diag.explicit_entry_hint = hint
    files = scan_files(
        root,
        extension_language_map={".py": "python"},
        ignore_paths=ignore_paths,
        skip_dirs=skip_dirs,
        max_file_size_kb=max_file_size_kb,
        diagnostics=diag,
    )
    return files, diag


_S8_GUIDANCE = {
    "unreadable": "fix-permissions-or-exclude",
    "ignored": "adjust-ignore-or-entry",
    "unsupported": "configure-extension-or-entry",
    "missing": "fix-entry-path",
}


def _s8_adm(reason, path):
    return {"path": path, "phase": "scanner-admission", "reason": reason,
            "guidance_code": _S8_GUIDANCE[reason]}


# --- single-file / missing sole-candidate cases --------------------------

@pytest.mark.parametrize(
    ("reason", "make", "hint", "ignore", "skip"),
    [
        ("ignored", lambda d: (d / "main.py").write_text("x = 1\n"),
         "main.py", ["main.py"], None),
        ("ignored", lambda d: ((d / "vendor").mkdir(),
                               (d / "vendor" / "main.py").write_text("x = 1\n")),
         "vendor/main.py", None, ["vendor"]),
        ("ignored", lambda d: ((d / ".hidden").mkdir(),
                               (d / ".hidden" / "main.py").write_text("x = 1\n")),
         ".hidden/main.py", None, None),
        ("unsupported", lambda d: (d / "main.txt").write_text("x\n"),
         "main.txt", None, None),
        ("missing", lambda d: None, "main.py", None, None),
    ],
)
def test_s8_scan_files_sole_candidate_explicit_entry_admission(
    tmp_path, reason, make, hint, ignore, skip
):
    make(tmp_path)
    files, diag = _s8_run_scan(tmp_path, hint, ignore_paths=ignore, skip_dirs=skip)
    assert files == []
    cat = diag.scanner_admission_skip
    assert (cat["details_total"], cat["details_retained"], cat["details_omitted"]) == (1, 1, 0)
    (d0,) = [dict(x) for x in cat["details"]]
    assert d0 == _s8_adm(reason, hint)
    assert cat["details_digest"] == _s8_stream_digest([d0])
    assert diag.scanner_size_skip["details_total"] == 0
    assert diag.scanner_size_skip["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST


def test_s8_scan_files_size_skipped_explicit_entry_stays_in_size_category_only(tmp_path):
    (tmp_path / "main.py").write_text(
        "\n".join(f"v{i} = {i}" for i in range(20000)), encoding="utf-8"
    )
    files, diag = _s8_run_scan(tmp_path, "main.py", max_file_size_kb=1)
    assert files == []
    s = diag.scanner_size_skip
    assert (s["details_total"], s["details_retained"], s["details_omitted"]) == (1, 1, 0)
    d0 = dict(s["details"][0])
    assert d0["path"] == "main.py" and d0["phase"] == "scanner-byte"
    assert d0["guidance_code"] == "raise-scan-byte-limit-or-exclude"
    assert d0["observed"] > d0["limit"]
    assert s["details_digest"] == _s8_stream_digest([d0])
    assert diag.scanner_admission_skip["details_total"] == 0
    assert diag.scanner_admission_skip["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST


def test_s8_scan_files_unreadable_explicit_entry_is_one_admission_record(
    tmp_path, monkeypatch
):
    from pathlib import Path

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    real_stat = Path.stat

    def _denied(self, *a, **k):
        if self.name == "main.py" and str(tmp_path) in str(self):
            raise PermissionError(13, "denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _denied)
    _files, diag = _s8_run_scan(tmp_path, "main.py")
    cat = diag.scanner_admission_skip
    assert cat["details_total"] == 1
    assert dict(cat["details"][0]) == _s8_adm("unreadable", "main.py")
    assert diag.scanner_size_skip["details_total"] == 0


# --- empty / populated explicit directories -----------------------------

def test_s8_scan_files_empty_explicit_directory_has_no_invented_record(tmp_path):
    (tmp_path / "empty").mkdir()
    files, diag = _s8_run_scan(tmp_path, "empty")
    assert files == []
    assert diag.scanner_admission_skip["details_total"] == 0
    assert diag.scanner_admission_skip["details"] == []
    assert diag.scanner_admission_skip["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST
    assert diag.scanner_size_skip["details_total"] == 0
    assert diag.scanner_size_skip["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST


def test_s8_scan_files_explicit_directory_records_each_excluded_candidate(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "child.txt").write_text("x")
    (tmp_path / "target" / "note.md").write_text("x")
    (tmp_path / "target" / "sub").mkdir()
    (tmp_path / "target" / "sub" / "deep.rst").write_text("x")
    files, diag = _s8_run_scan(tmp_path, "target")
    assert files == []
    cat = diag.scanner_admission_skip
    details = [dict(x) for x in cat["details"]]
    # per-candidate, NOT one record for the directory itself; canonical order.
    assert details == [
        _s8_adm("unsupported", "target/child.txt"),
        _s8_adm("unsupported", "target/note.md"),
        _s8_adm("unsupported", "target/sub/deep.rst"),
    ]
    assert cat["details_total"] == 3
    assert cat["details_digest"] == _s8_stream_digest(details)
    assert "target" not in [d["path"] for d in details]


def test_s8_scan_files_explicit_directory_inside_skip_dirs_records_ignored_candidates(
    tmp_path,
):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "a.py").write_text("x = 1\n")
    (tmp_path / "vendor" / "b.py").write_text("y = 1\n")
    files, diag = _s8_run_scan(tmp_path, "vendor", skip_dirs=["vendor"])
    assert files == []
    details = [dict(x) for x in diag.scanner_admission_skip["details"]]
    assert details == [_s8_adm("ignored", "vendor/a.py"), _s8_adm("ignored", "vendor/b.py")]
    assert diag.scanner_admission_skip["details_digest"] == _s8_stream_digest(details)


# --- directory-scale boundedness --------------------------------------------

def test_s8_scan_files_directory_full_of_candidates_is_bounded_top_k_with_full_digest(
    tmp_path,
):
    from codedoc.core.file_division import canonical_stream_digest

    n = _S8_HEAP_CAP + 40
    (tmp_path / "big").mkdir()
    for i in range(n):
        (tmp_path / "big" / f"c{i:05d}.txt").write_text("x")
    files, diag = _s8_run_scan(tmp_path, "big")
    assert files == []
    cat = diag.scanner_admission_skip
    assert cat["details_total"] == n
    assert cat["details_retained"] == _S8_HEAP_CAP
    assert cat["details_omitted"] == n - _S8_HEAP_CAP
    assert len(cat["details"]) == _S8_HEAP_CAP
    # retained top-K is the first _S8_HEAP_CAP by canonical (path) order.
    assert [d["path"] for d in cat["details"]] == [
        f"big/c{i:05d}.txt" for i in range(_S8_HEAP_CAP)
    ]
    # the digest covers EVERY candidate, including the omitted ones.
    every = [_s8_adm("unsupported", f"big/c{i:05d}.txt") for i in range(n)]
    assert cat["details_digest"] == canonical_stream_digest(every)
    # value-safe: normalized project-relative POSIX, no absolute path.
    for d in cat["details"]:
        assert str(tmp_path) not in d["path"] and not d["path"].startswith("/")


def test_s8_scan_files_directory_candidate_digest_is_permutation_stable(tmp_path):
    import random

    names = [f"f{i:03d}.txt" for i in range(50)]
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    (root_a / "many").mkdir(parents=True)
    (root_b / "many").mkdir(parents=True)
    for nm in names:
        (root_a / "many" / nm).write_text("x")
    shuffled = names[:]
    random.Random(7).shuffle(shuffled)
    for nm in shuffled:
        (root_b / "many" / nm).write_text("x")
    _fa, diag_a = _s8_run_scan(root_a, "many")
    _fb, diag_b = _s8_run_scan(root_b, "many")
    # identical populations -> identical canonical stream, digest and retention,
    # independent of filesystem creation/enumeration order.
    assert diag_a.scanner_admission_skip["details_digest"] == diag_b.scanner_admission_skip["details_digest"]
    assert [dict(x) for x in diag_a.scanner_admission_skip["details"]] == [
        dict(x) for x in diag_b.scanner_admission_skip["details"]
    ]
    assert diag_a.scanner_admission_skip["details_total"] == 50


# --- canonical-order collision: no descriptor dropped ----------------------

def test_s8_scan_files_canonical_order_collision_keeps_both_descriptors(
    tmp_path, monkeypatch
):
    """One authoritative final generation. An unrelated unreadable ``z.py`` in
    the project PLUS a MISSING explicit entry ``a.py`` (whose descriptor sorts
    canonically earlier). Both admission descriptors must survive in canonical
    order (``a.py`` missing, then ``z.py`` unreadable), the total is 2, and the
    complete-stream digest covers both -- no previously recorded evidence may be
    discarded merely to insert the earlier-sorting explicit-entry descriptor."""
    from pathlib import Path

    (tmp_path / "z.py").write_text("x = 1\n", encoding="utf-8")
    real_stat = Path.stat

    def _denied(self, *a, **k):
        if self.name == "z.py":
            raise PermissionError(13, "denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _denied)
    files, diag = _s8_run_scan(tmp_path, "a.py")
    assert files == []
    cat = diag.scanner_admission_skip
    # Display ranking is (frozen-reason index, path): unreadable(0) before
    # missing(3). Integrity/digest order is canonical (path ascending): a.py
    # then z.py. The two orders are deliberately distinct and BOTH descriptors
    # survive in the one final generation.
    assert [dict(x) for x in cat["details"]] == [
        _s8_adm("unreadable", "z.py"), _s8_adm("missing", "a.py"),
    ]
    assert cat["details_total"] == 2
    assert cat["details_retained"] == 2
    assert cat["details_omitted"] == 0
    assert cat["details_digest"] == _s8_stream_digest(
        [_s8_adm("missing", "a.py"), _s8_adm("unreadable", "z.py")]
    )


def test_s8_scanner_code_has_no_catch_and_discard_around_diagnostic_record():
    import inspect

    from codedoc.core import scanner

    src = inspect.getsource(scanner)
    assert "except ValueError:\n" not in src or "pass" not in src.split(
        "except ValueError:\n", 1
    )[-1][:80], "no catch-and-discard around scanner diagnostic recording"


# --- final-generation replacement + no-hint no-op --------------------------

def test_s8_scan_files_explicit_entry_admission_replaced_on_final_generation(tmp_path):
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "a.txt").write_text("x")
    (tmp_path / "target" / "b.txt").write_text("x")
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = "target"
    scan_files(tmp_path, extension_language_map={".py": "python"}, diagnostics=diag)
    scan_files(tmp_path, extension_language_map={".py": "python"}, diagnostics=diag)
    # atomic replacement, not a merge.
    assert diag.scanner_admission_skip["details_total"] == 2
    assert [d["path"] for d in diag.scanner_admission_skip["details"]] == [
        "target/a.txt", "target/b.txt"
    ]


def test_s8_scan_files_without_an_explicit_entry_hint_folds_nothing(tmp_path):
    (tmp_path / "main.txt").write_text("x\n", encoding="utf-8")
    from codedoc.core.scanner import ScanDiagnostics, scan_files

    diag = ScanDiagnostics()
    files = scan_files(
        tmp_path, extension_language_map={".py": "python"}, diagnostics=diag
    )
    assert files == []
    assert diag.scanner_admission_skip["details_total"] == 0
    assert diag.scanner_size_skip["details_total"] == 0


def test_s8_scan_files_explicit_entry_evidence_survives_when_other_files_admitted(
    tmp_path,
):
    """An explicit target that yields no admitted source is still classified in
    the ONE final generation even when unrelated project files ARE admitted. The
    admitted-file set is unchanged; the explicit-entry admission descriptor is
    additional evidence, not a replacement of the real scan."""
    (tmp_path / "main.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("y = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, "main.txt")
    assert [f["rel_path"] for f in files] == ["other.py"]
    cat = diag.scanner_admission_skip
    assert cat["details_total"] == 1
    assert dict(cat["details"][0]) == _s8_adm("unsupported", "main.txt")
    assert cat["details_digest"] == _s8_stream_digest([_s8_adm("unsupported", "main.txt")])
    assert diag.scanner_size_skip["details_total"] == 0


def test_s8_scan_files_explicit_entry_admission_hint_bytes_are_value_safe(tmp_path):
    hostile = "ev" + chr(10) + "il" + chr(9) + chr(0x1B) + chr(0x202E) + '"' + ".py"
    _files, diag = _s8_run_scan(tmp_path, hostile)
    cat = diag.scanner_admission_skip
    assert cat["details_total"] == 1
    d0 = dict(cat["details"][0])
    rendered = json.dumps(d0["path"], ensure_ascii=True)
    for raw in ("\n", "\t", "\x1b", chr(0x202E)):
        assert raw not in rendered
    assert cat["details_digest"] == _s8_stream_digest([d0])


# --- Correction round 3: ONE authoritative final generation --------------------
# The explicit-entry classification is folded into the SAME generation as the
# ordinary project size/unreadable skips (never a post-finalize target-only
# overwrite), so cross-category evidence is complete, warnings emit once, and
# nested hidden candidates are seen.

def _s8_warn_msgs(caplog):
    import logging

    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_s8_scan_files_final_generation_unions_ordinary_size_and_explicit_missing(
    tmp_path, caplog
):
    import logging

    (tmp_path / "z.py").write_text(
        "\n".join(f"v{i} = {i}" for i in range(4000)), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        files, diag = _s8_run_scan(tmp_path, "a.py", max_file_size_kb=1)
    assert files == []
    size, adm = diag.scanner_size_skip, diag.scanner_admission_skip
    assert size["details_total"] == 1
    assert [d["path"] for d in size["details"]] == ["z.py"]
    assert size["details_digest"] == _s8_stream_digest([dict(d) for d in size["details"]])
    assert adm["details_total"] == 1
    assert dict(adm["details"][0]) == _s8_adm("missing", "a.py")
    assert adm["details_digest"] == _s8_stream_digest([_s8_adm("missing", "a.py")])
    # the compatibility scalar + INFO summary count the same one large file.
    assert diag.files_skipped_large == 1
    assert sum("Skipping large file" in m for m in _s8_warn_msgs(caplog)) == 1


def test_s8_scan_files_explicit_large_file_warns_exactly_once(tmp_path, caplog):
    import logging

    (tmp_path / "big.py").write_text(
        "\n".join(f"v{i} = {i}" for i in range(6000)), encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        files, diag = _s8_run_scan(tmp_path, "big.py", max_file_size_kb=1)
    assert files == []
    assert diag.scanner_size_skip["details_total"] == 1
    assert sum("Skipping large file" in m for m in _s8_warn_msgs(caplog)) == 1


def test_s8_scan_files_explicit_directory_large_files_warn_capped_with_one_remainder(
    tmp_path, caplog
):
    import logging

    (tmp_path / "big").mkdir()
    for i in range(25):
        (tmp_path / "big" / f"f{i:02d}.py").write_text(
            "\n".join(f"v{j} = {j}" for j in range(3000)), encoding="utf-8"
        )
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        files, diag = _s8_run_scan(tmp_path, "big", max_file_size_kb=1)
    assert files == []
    assert diag.scanner_size_skip["details_total"] == 25
    msgs = _s8_warn_msgs(caplog)
    assert sum("Skipping large file" in m for m in msgs) == 20
    assert sum("more large file(s) not shown" in m for m in msgs) == 1
    assert sum(m.startswith("5 more large") for m in msgs) == 1


def test_s8_scan_files_explicit_directory_sees_nested_hidden_candidate(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / ".hidden").mkdir()
    (tmp_path / "target" / ".hidden" / "a.py").write_text("x = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, "target")
    assert files == []
    details = [dict(x) for x in diag.scanner_admission_skip["details"]]
    assert details == [_s8_adm("ignored", "target/.hidden/a.py")]
    assert diag.scanner_admission_skip["details_total"] == 1
    assert "target" not in [d["path"] for d in details]
    assert diag.scanner_admission_skip["details_digest"] == _s8_stream_digest(details)


def test_s8_scan_files_explicit_directory_nested_skip_dir_candidate_is_ignored(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "vendor").mkdir()
    (tmp_path / "target" / "vendor" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "target" / "keep.py").write_text("y = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, "target", skip_dirs=["vendor"])
    # keep.py is a normal admitted file; the run is only "zero admission" for the
    # explicit target when it is a real error case -- here the walker still
    # admits keep.py, so this asserts the classification of the excluded one.
    details = [dict(x) for x in diag.scanner_admission_skip["details"]]
    assert _s8_adm("ignored", "target/vendor/a.py") in details
    assert "target/keep.py" in {f["rel_path"] for f in files}


def test_s8_scan_files_explicit_entry_rescan_publishes_only_the_second_generation(
    tmp_path, caplog
):
    import logging

    from codedoc.core.scanner import ScanDiagnostics, scan_files

    tgt = tmp_path / "target"
    tgt.mkdir()
    (tgt / "x.txt").write_text("x\n", encoding="utf-8")
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = "target"
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(
            tmp_path, extension_language_map={".py": "python"},
            max_file_size_kb=1, diagnostics=diag,
        )
        assert diag.scanner_admission_skip["details_total"] == 1
        assert diag.scanner_admission_skip["details"][0]["reason"] == "unsupported"
        (tgt / "x.txt").rename(tgt / "x.py")
        (tgt / "x.py").write_text(
            "\n".join(f"v{i} = {i}" for i in range(6000)), encoding="utf-8"
        )
        caplog.clear()
        scan_files(
            tmp_path, extension_language_map={".py": "python"},
            max_file_size_kb=1, diagnostics=diag,
        )
    # the final snapshot is the COMPLETE second generation only, never a merge.
    assert diag.scanner_admission_skip["details_total"] == 0
    assert diag.scanner_admission_skip["details_digest"] == EMPTY_PLAN_DETAILS_DIGEST
    assert diag.scanner_size_skip["details_total"] == 1
    assert diag.scanner_size_skip["details"][0]["path"] == "target/x.py"
    # the warning allowance reset once: the size skip is warned a single time.
    assert sum("Skipping large file" in m for m in _s8_warn_msgs(caplog)) == 1


# --- Correction round 4: the explicit-target boundary --------------------------
# A skipped directory that is only ON THE PATH to a single explicit file (or a
# deeper directory target) is descended just far enough to reach that target.
# Off-path siblings are NOT target-owned evidence: they are silently discarded
# before any classification, warning, hash, total or sibling-subtree descent.
# The would-skip directory at the normal admission frontier still counts once
# in ``skipped_dirs``.

def _s8_scan_info_line(caplog):
    import logging

    for r in caplog.records:
        m = r.getMessage()
        if r.levelno == logging.INFO and m.startswith("Scanner found "):
            return m
    return ""


def test_s8_explicit_file_under_hidden_ancestor_excludes_siblings(tmp_path, caplog):
    import logging

    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".hidden" / "sibling.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="codedoc.core.scanner"):
        files, diag = _s8_run_scan(tmp_path, ".hidden/main.py")
    assert [f["rel_path"] for f in files] == ["other.py"]
    cat = diag.scanner_admission_skip
    details = [dict(x) for x in cat["details"]]
    # EXACT descriptor universe -- the sibling must not appear anywhere.
    assert details == [_s8_adm("ignored", ".hidden/main.py")]
    assert cat["details_total"] == 1
    assert cat["details_retained"] == 1
    assert cat["details_omitted"] == 0
    assert cat["details_digest"] == _s8_stream_digest([_s8_adm("ignored", ".hidden/main.py")])
    assert diag.scanner_size_skip["details_total"] == 0
    msgs = _s8_warn_msgs(caplog)
    assert not any("sibling.py" in m for m in msgs)
    # the would-skip ancestor still counts once at the normal admission frontier.
    assert "skipped 1 directorie(s)" in _s8_scan_info_line(caplog)


def test_s8_explicit_file_under_skip_dirs_ancestor_excludes_siblings(tmp_path, caplog):
    import logging

    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "vendor" / "sibling.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="codedoc.core.scanner"):
        files, diag = _s8_run_scan(tmp_path, "vendor/main.py", skip_dirs=["vendor"])
    assert [f["rel_path"] for f in files] == ["other.py"]
    cat = diag.scanner_admission_skip
    assert [dict(x) for x in cat["details"]] == [_s8_adm("ignored", "vendor/main.py")]
    assert cat["details_total"] == 1
    assert cat["details_digest"] == _s8_stream_digest([_s8_adm("ignored", "vendor/main.py")])
    assert not any("sibling.py" in m for m in _s8_warn_msgs(caplog))
    assert "skipped 1 directorie(s)" in _s8_scan_info_line(caplog)


def test_s8_missing_explicit_target_under_skipped_ancestor_ignores_siblings(tmp_path, caplog):
    import logging

    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "sibling.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="codedoc.core.scanner"):
        files, diag = _s8_run_scan(tmp_path, ".hidden/missing.py")
    assert [f["rel_path"] for f in files] == ["other.py"]
    cat = diag.scanner_admission_skip
    assert [dict(x) for x in cat["details"]] == [_s8_adm("missing", ".hidden/missing.py")]
    assert cat["details_total"] == 1
    assert cat["details_digest"] == _s8_stream_digest([_s8_adm("missing", ".hidden/missing.py")])
    assert not any("sibling.py" in m for m in _s8_warn_msgs(caplog))


def test_s8_explicit_directory_under_skipped_ancestor_classifies_only_inside(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "target").mkdir()
    (tmp_path / ".hidden" / "target" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".hidden" / "outside.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, ".hidden/target")
    assert [f["rel_path"] for f in files] == ["other.py"]
    cat = diag.scanner_admission_skip
    details = [dict(x) for x in cat["details"]]
    # only descendants INSIDE the explicit directory; the outside sibling and
    # the target directory itself are absent.
    assert details == [_s8_adm("ignored", ".hidden/target/a.py")]
    assert cat["details_total"] == 1
    assert ".hidden/target" not in [d["path"] for d in details]
    assert ".hidden/outside.py" not in [d["path"] for d in details]
    assert cat["details_digest"] == _s8_stream_digest([_s8_adm("ignored", ".hidden/target/a.py")])


def test_s8_explicit_file_under_visible_then_hidden_ancestor_excludes_siblings(tmp_path):
    # A normal visible directory on the path keeps admitting its own files; only
    # the would-skip segment prunes to the target.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "keep.py").write_text("k = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / ".secret").mkdir()
    (tmp_path / "pkg" / ".secret" / "main.py").write_text("m = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / ".secret" / "sibling.py").write_text("s = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, "pkg/.secret/main.py")
    assert "pkg/keep.py" in {f["rel_path"] for f in files}     # admitted-set unchanged
    cat = diag.scanner_admission_skip
    assert [dict(x) for x in cat["details"]] == [_s8_adm("ignored", "pkg/.secret/main.py")]
    assert cat["details_total"] == 1


# --- Correction round 5: case-aware matching + identity isolation -------------

def _s8_ci_fs(tmp_path):
    """True when this host filesystem folds case (Windows / macOS default)."""
    probe = tmp_path / "_cr5_ci_probe.py"
    probe.write_text("x = 1\n", encoding="utf-8")
    try:
        return (tmp_path / "_CR5_CI_PROBE.PY").exists()
    finally:
        probe.unlink()


def test_s8_cr5_hidden_target_alternate_casing(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".hidden" / "sibling.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, ".HIDDEN/MAIN.PY")
    assert [f["rel_path"] for f in files] == ["other.py"]
    c = diag.scanner_admission_skip
    if _s8_ci_fs(tmp_path):
        # matched via host case semantics; descriptor keeps ACTUAL spelling.
        assert [dict(x) for x in c["details"]] == [_s8_adm("ignored", ".hidden/main.py")]
    else:
        # ISOLATION SENTINEL: case-sensitive host -> the alt-cased hint is a
        # distinct, non-existent target: exactly one `missing` descriptor.
        assert [dict(x) for x in c["details"]] == [_s8_adm("missing", ".HIDDEN/MAIN.PY")]
    assert c["details_total"] == 1
    assert c["details_digest"] == _s8_stream_digest([dict(c["details"][0])])
    assert ".hidden/sibling.py" not in [d["path"] for d in c["details"]]
    assert diag.scanner_size_skip["details_total"] == 0


def test_s8_cr5_unsupported_target_alternate_casing(tmp_path):
    (tmp_path / "entry.txt").write_text("entrypoint\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, "ENTRY.TXT")
    assert [f["rel_path"] for f in files] == ["other.py"]
    c = diag.scanner_admission_skip
    if _s8_ci_fs(tmp_path):
        assert [dict(x) for x in c["details"]] == [_s8_adm("unsupported", "entry.txt")]
    else:
        assert [dict(x) for x in c["details"]] == [_s8_adm("missing", "ENTRY.TXT")]
    assert c["details_total"] == 1


def test_s8_cr5_configured_ignored_target_alternate_casing(tmp_path):
    (tmp_path / "entry.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, "ENTRY.PY", ignore_paths=["entry.py"])
    assert [f["rel_path"] for f in files] == ["other.py"]
    c = diag.scanner_admission_skip
    if _s8_ci_fs(tmp_path):
        assert [dict(x) for x in c["details"]] == [_s8_adm("ignored", "entry.py")]
        assert c["details_total"] == 1
    else:
        # case-sensitive: ENTRY.PY missing; entry.py is ignore-matched but is
        # NOT the explicit target, so it is silently excluded (no descriptor).
        assert [dict(x) for x in c["details"]] == [_s8_adm("missing", "ENTRY.PY")]


def test_s8_cr5_directory_boundary_matching_alternate_casing(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "target").mkdir()
    (tmp_path / ".hidden" / "target" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".hidden" / "outside.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    files, diag = _s8_run_scan(tmp_path, ".HIDDEN/TARGET")
    assert [f["rel_path"] for f in files] == ["other.py"]
    c = diag.scanner_admission_skip
    if _s8_ci_fs(tmp_path):
        assert [dict(x) for x in c["details"]] == [_s8_adm("ignored", ".hidden/target/a.py")]
        assert ".hidden/outside.py" not in [d["path"] for d in c["details"]]
        assert ".hidden/target" not in [d["path"] for d in c["details"]]
    else:
        assert [dict(x) for x in c["details"]] == [_s8_adm("missing", ".HIDDEN/TARGET")]


def test_s8_cr5_ignored_directory_hardlink_aliases_dedup(tmp_path):
    import os

    (tmp_path / "target").mkdir()
    a = tmp_path / "target" / "a.py"
    a.write_text("x = 1\n", encoding="utf-8")
    try:
        os.link(a, tmp_path / "target" / "b.py")
    except (OSError, NotImplementedError, AttributeError):
        import pytest as _pytest
        _pytest.skip("hard links unavailable on this filesystem")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = "target"
    files = scan_files(
        tmp_path, extension_language_map={".py": "python"},
        ignore_paths=["target"], follow_symlinks=True, diagnostics=diag,
    )
    assert [f["rel_path"] for f in files] == ["other.py"]
    c = diag.scanner_admission_skip
    # deterministic traversal order -> the first alias only.
    assert [dict(x) for x in c["details"]] == [_s8_adm("ignored", "target/a.py")]
    assert c["details_total"] == 1
    assert c["details_retained"] == 1
    assert c["details_omitted"] == 0
    assert c["details_digest"] == _s8_stream_digest([_s8_adm("ignored", "target/a.py")])


def test_s8_cr5_ignored_dir_hardlink_alias_warns_once(tmp_path, caplog):
    import logging
    import os

    (tmp_path / "target").mkdir()
    a = tmp_path / "target" / "a.py"
    a.write_text("x = 1\n", encoding="utf-8")
    try:
        os.link(a, tmp_path / "target" / "b.py")
    except (OSError, NotImplementedError, AttributeError):
        import pytest as _pytest
        _pytest.skip("hard links unavailable on this filesystem")

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = "target"
    with caplog.at_level(logging.WARNING, logger="codedoc.core.scanner"):
        scan_files(
            tmp_path, extension_language_map={".py": "python"},
            ignore_paths=["target"], follow_symlinks=True, diagnostics=diag,
        )
    assert sum("Skipping ignored file" in m for m in _s8_warn_msgs(caplog)) == 1
    assert diag.scanner_admission_skip["details_total"] == 1


def test_s8_cr5_unreadable_target_under_skipped_ancestor_is_ignored(tmp_path, monkeypatch):
    from pathlib import Path as _P

    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".hidden" / "sibling.py").write_text("y = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")

    _real_stat = _P.stat

    def _denied(self, *a, **k):
        if self.name == "main.py" and ".hidden" in _P(self).as_posix():
            raise PermissionError(13, "denied")
        return _real_stat(self, *a, **k)

    monkeypatch.setattr(_P, "stat", _denied)
    files, diag = _s8_run_scan(tmp_path, ".hidden/main.py")
    assert [f["rel_path"] for f in files] == ["other.py"]
    c = diag.scanner_admission_skip
    # ancestor already excludes it -> `ignored`, never `unreadable`, exactly once.
    assert [dict(x) for x in c["details"]] == [_s8_adm("ignored", ".hidden/main.py")]
    assert c["details_total"] == 1
    assert ".hidden/sibling.py" not in [d["path"] for d in c["details"]]
    assert diag.files_skipped_unreadable == 0


def test_s8_cr5_ordinary_unreadable_target_stays_unreadable(tmp_path, monkeypatch):
    from pathlib import Path as _P

    (tmp_path / "entry.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("z = 1\n", encoding="utf-8")
    _real_stat = _P.stat

    def _denied(self, *a, **k):
        if self.name == "entry.py":
            raise PermissionError(13, "denied")
        return _real_stat(self, *a, **k)

    monkeypatch.setattr(_P, "stat", _denied)
    files, diag = _s8_run_scan(tmp_path, "entry.py")
    c = diag.scanner_admission_skip
    assert [dict(x) for x in c["details"]] == [_s8_adm("unreadable", "entry.py")]
    assert c["details_total"] == 1
