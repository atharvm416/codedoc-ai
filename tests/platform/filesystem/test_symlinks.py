"""Scanner symlink, junction, and recursion-safety tests (0.9.6).

These tests exercise the iterative, symlink-safe walk introduced in 0.9.6.
Individual tests that require creating a link are skipped when the platform or
permission set does not allow link creation; the module as a whole is never
skipped.
"""

from __future__ import annotations

import errno
import os

import pytest

from codedoc.core.scanner import scan_files

pytestmark = pytest.mark.platform


def _write_py(path, text="x = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _can_symlink(tmp_path) -> bool:
    """Return True when this platform/permission set can create a symlink."""
    probe_target = tmp_path / "_probe_target"
    probe_target.mkdir(exist_ok=True)
    probe_link = tmp_path / "_probe_link"
    try:
        os.symlink(probe_target, probe_link, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        return False
    finally:
        try:
            if probe_link.is_symlink() or probe_link.exists():
                probe_link.unlink()
        except OSError:
            pass
    return True


def _rels(files):
    return {f["rel_path"] for f in files}


def test_symlinked_directory_cycle_terminates(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "pkg" / "a.py")
    # Link inside pkg back to the project root → a cycle.
    os.symlink(tmp_path, tmp_path / "pkg" / "loop", target_is_directory=True)

    # follow_symlinks=True must still terminate thanks to visited-identity guard.
    files = scan_files(tmp_path, supported_extensions=[".py"], follow_symlinks=True)
    assert "pkg/a.py" in _rels(files)


def test_symlinks_skipped_by_default_and_counted(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "real" / "mod.py")
    _write_py(tmp_path / "top.py")
    os.symlink(tmp_path / "real", tmp_path / "linkdir", target_is_directory=True)
    os.symlink(tmp_path / "top.py", tmp_path / "link.py")

    files = scan_files(tmp_path, supported_extensions=[".py"])
    rels = _rels(files)

    # Real files included; symlinked aliases excluded.
    assert "real/mod.py" in rels
    assert "top.py" in rels
    assert "linkdir/mod.py" not in rels
    assert "link.py" not in rels


def test_skipped_symlinked_directory_counted(tmp_path, caplog):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "real" / "mod.py")
    os.symlink(tmp_path / "real", tmp_path / "linkdir", target_is_directory=True)

    import logging

    from codedoc.core import scanner

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record)
    scanner.logger.addHandler(handler)
    scanner.logger.setLevel(logging.INFO)
    try:
        files = scan_files(tmp_path, supported_extensions=[".py"])
    finally:
        scanner.logger.removeHandler(handler)

    # The scanner logs the skipped-directory count in its INFO summary line.
    summary = " ".join(r.getMessage() for r in records)
    assert "skipped" in summary
    assert "real/mod.py" in _rels(files)


def test_follow_symlinks_scans_in_root_target_once(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "real" / "mod.py")
    os.symlink(tmp_path / "real", tmp_path / "alias", target_is_directory=True)

    files = scan_files(tmp_path, supported_extensions=[".py"], follow_symlinks=True)
    rels = _rels(files)
    # Both aliases resolve to one real directory, so the file is documented
    # under exactly one project-relative path.  The first encountered alias owns
    # the descriptor; directory encounter order is filesystem-dependent.
    assert rels & {"real/mod.py", "alias/mod.py"}
    descriptors = [f for f in files if f["rel_path"].endswith("mod.py")]
    assert len(descriptors) == 1


def test_out_of_root_symlink_never_included(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    project = tmp_path / "project"
    outside = tmp_path / "outside"
    _write_py(project / "main.py")
    _write_py(outside / "secret.py")

    os.symlink(outside, project / "escape", target_is_directory=True)
    os.symlink(outside / "secret.py", project / "secret_link.py")

    files = scan_files(project, supported_extensions=[".py"], follow_symlinks=True)
    rels = _rels(files)
    assert "main.py" in rels
    assert "escape/secret.py" not in rels
    assert "secret_link.py" not in rels
    assert all("outside" not in r for r in rels)


def test_broken_symlink_skipped_without_aborting(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "main.py")
    os.symlink(tmp_path / "does_not_exist", tmp_path / "broken")

    # follow_symlinks=True: broken link is skipped, scan completes.
    files = scan_files(tmp_path, supported_extensions=[".py"], follow_symlinks=True)
    assert "main.py" in _rels(files)


def test_ignored_symlink_alias_cannot_bypass_ignore(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "secret" / "data.py")
    os.symlink(tmp_path / "secret", tmp_path / "ignored_link", target_is_directory=True)
    _write_py(tmp_path / "main.py")

    files = scan_files(
        tmp_path,
        supported_extensions=[".py"],
        ignore_paths=["ignored_link", "secret"],
        follow_symlinks=True,
    )
    rels = _rels(files)
    assert "main.py" in rels
    assert "ignored_link/data.py" not in rels
    assert "secret/data.py" not in rels


def test_hidden_symlink_alias_skipped(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    _write_py(tmp_path / "real" / "mod.py")
    os.symlink(tmp_path / "real", tmp_path / ".hidden_link", target_is_directory=True)

    files = scan_files(tmp_path, supported_extensions=[".py"], follow_symlinks=True)
    rels = _rels(files)
    assert "real/mod.py" in rels
    assert ".hidden_link/mod.py" not in rels


def test_deeply_nested_acyclic_tree_no_recursion_error(tmp_path):
    # Build a tree far deeper than a recursive walk could survive (each level of
    # the old recursive walk consumed several stack frames, overflowing well
    # before this depth at the default limit of 1000).  The iterative walk uses
    # an explicit stack, so it scans the leaf without a RecursionError.
    depth = 600
    current = tmp_path
    for i in range(depth):
        current = current / f"d{i}"
    try:
        _write_py(current / "leaf.py")
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            pytest.skip(
                "platform PATH_MAX is too small to build a tree deep enough to "
                "exercise the iterative walk (e.g. macOS caps paths at 1024 bytes); "
                "the iterative-walk property is OS-independent and covered on Linux"
            )
        raise

    files = scan_files(tmp_path, supported_extensions=[".py"])
    assert any(f["rel_path"].endswith("leaf.py") for f in files)


def test_positional_compatibility_intact(tmp_path):
    """The legacy positional signature still works (supported_extensions list)."""
    _write_py(tmp_path / "main.py")
    files = scan_files(tmp_path, [".py"])
    assert "main.py" in _rels(files)


def test_existing_skip_dir_and_size_behavior_unchanged(tmp_path):
    _write_py(tmp_path / "main.py")
    _write_py(tmp_path / "node_modules" / "dep.py")
    big = tmp_path / "big.py"
    big.write_text("# " + "x" * (2 * 1024), encoding="utf-8")

    files = scan_files(
        tmp_path,
        supported_extensions=[".py"],
        skip_dirs=["node_modules"],
        max_file_size_kb=1,
    )
    rels = _rels(files)
    assert "main.py" in rels
    assert "node_modules/dep.py" not in rels
    assert "big.py" not in rels


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_windows_junction_skipped_by_default(tmp_path):
    import subprocess

    target = tmp_path / "real"
    _write_py(target / "mod.py")
    junction = tmp_path / "junc"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not junction.exists():
        pytest.skip("could not create a junction in this environment")

    files = scan_files(tmp_path, supported_extensions=[".py"])
    rels = _rels(files)
    assert "real/mod.py" in rels
    assert "junc/mod.py" not in rels


# --- Correction round 5: explicit hint must not change admission -------------

def test_cr5_diagnostic_visit_does_not_suppress_ordinary_alias(tmp_path, monkeypatch):
    """Deterministic (no OS link needed): a skipped explicit directory and a
    visible alias share ONE directory identity. Admitted files must be identical
    with and without ``explicit_entry_hint``; explicit-target diagnostics stay
    exact."""
    import codedoc.core.scanner as _sc

    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "target").mkdir()
    _write_py(tmp_path / ".hidden" / "target" / "a.py")
    (tmp_path / "alias").mkdir()
    _write_py(tmp_path / "alias" / "a.py")
    _write_py(tmp_path / "other.py")

    real_identity = _sc._Walker._identity

    def _shared(self, path):
        p = os.path.normcase(os.path.normpath(str(path)))
        if p.endswith(os.path.normcase(os.path.join(".hidden", "target"))) or \
           p.endswith(os.path.normcase("alias")):
            return ("shared-dir-identity",)
        if p.endswith(os.path.normcase(os.path.join(".hidden", "target", "a.py"))) or \
           p.endswith(os.path.normcase(os.path.join("alias", "a.py"))):
            return ("shared-file-identity",)
        return real_identity(self, path)

    monkeypatch.setattr(_sc._Walker, "_identity", _shared, raising=True)

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    plain = {f["rel_path"] for f in scan_files(
        tmp_path, extension_language_map={".py": "python"}, follow_symlinks=True)}

    diag = ScanDiagnostics()
    diag.explicit_entry_hint = ".hidden/target"
    hinted_files = scan_files(
        tmp_path, extension_language_map={".py": "python"},
        follow_symlinks=True, diagnostics=diag)
    hinted = {f["rel_path"] for f in hinted_files}

    assert hinted == plain, (plain, hinted)
    assert "alias/a.py" in hinted
    cat = diag.scanner_admission_skip
    assert [d["path"] for d in cat["details"]] == [".hidden/target/a.py"]
    assert cat["details_total"] == 1


@pytest.mark.parametrize("alias_first", [True, False])
def test_cr5_shared_identity_both_lexical_orders(tmp_path, monkeypatch, alias_first):
    import codedoc.core.scanner as _sc

    skipped = "aaa_skip" if alias_first else "zzz_skip"
    alias = "zzz_alias" if alias_first else "aaa_alias"
    (tmp_path / skipped).mkdir()
    (tmp_path / skipped / "target").mkdir()
    _write_py(tmp_path / skipped / "target" / "a.py")
    (tmp_path / alias).mkdir()
    _write_py(tmp_path / alias / "a.py")

    real_identity = _sc._Walker._identity

    def _shared(self, path):
        s = str(path).replace("\\", "/")
        if s.endswith(f"{skipped}/target") or s.endswith(f"/{alias}"):
            return ("shared-dir",)
        if s.endswith(f"{skipped}/target/a.py") or s.endswith(f"{alias}/a.py"):
            return ("shared-file",)
        return real_identity(self, path)

    monkeypatch.setattr(_sc._Walker, "_identity", _shared, raising=True)

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    plain = {f["rel_path"] for f in scan_files(
        tmp_path, extension_language_map={".py": "python"},
        skip_dirs=[skipped], follow_symlinks=True)}
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = f"{skipped}/target"
    hinted = {f["rel_path"] for f in scan_files(
        tmp_path, extension_language_map={".py": "python"},
        skip_dirs=[skipped], follow_symlinks=True, diagnostics=diag)}
    assert hinted == plain
    assert f"{alias}/a.py" in hinted
    assert [d["path"] for d in diag.scanner_admission_skip["details"]] == [
        f"{skipped}/target/a.py"]


def test_cr5_real_junction_hint_does_not_change_admission(tmp_path):
    if os.name != "nt":
        pytest.skip("junction test is Windows-only")
    import subprocess

    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "target").mkdir()
    _write_py(tmp_path / ".hidden" / "target" / "a.py")
    _write_py(tmp_path / "other.py")
    junction = tmp_path / "alias"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction),
                        str(tmp_path / ".hidden" / "target")],
                       capture_output=True, text=True)
    if r.returncode != 0 or not junction.exists():
        pytest.skip("could not create a junction here")

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    plain = {f["rel_path"] for f in scan_files(
        tmp_path, extension_language_map={".py": "python"}, follow_symlinks=True)}
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = ".hidden/target"
    hinted = {f["rel_path"] for f in scan_files(
        tmp_path, extension_language_map={".py": "python"},
        follow_symlinks=True, diagnostics=diag)}
    assert hinted == plain
    assert "alias/a.py" in hinted
    assert [d["path"] for d in diag.scanner_admission_skip["details"]] == [
        ".hidden/target/a.py"]


def _cr5_dir_link(src, dst) -> bool:
    """Create a real directory link src<-dst by symlink, else a Windows junction."""
    try:
        os.symlink(src, dst, target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        import subprocess
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           capture_output=True, text=True)
        return r.returncode == 0 and dst.exists()
    return False


def _cr5_file_alias(src, dst) -> bool:
    """Create a real second alias to one physical file (hard link, else symlink)."""
    try:
        os.link(src, dst)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        os.symlink(src, dst)
        return True
    except (OSError, NotImplementedError, AttributeError):
        return False


def test_cr5_diagnostic_target_link_back_to_root_terminates(tmp_path):
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "target").mkdir()
    _write_py(tmp_path / ".hidden" / "target" / "a.py")
    _write_py(tmp_path / "other.py")
    if not _cr5_dir_link(tmp_path, tmp_path / ".hidden" / "target" / "loop"):
        pytest.skip("cannot create a directory link on this host")

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = ".hidden/target"
    files = scan_files(
        tmp_path, extension_language_map={".py": "python"},
        follow_symlinks=True, diagnostics=diag)
    # scan terminates; boundary evidence only; no duplicated / off-boundary paths.
    assert {f["rel_path"] for f in files} == {"other.py"}
    paths = [d["path"] for d in diag.scanner_admission_skip["details"]]
    assert paths == [".hidden/target/a.py"]
    assert all(p.startswith(".hidden/target/") for p in paths)


def test_cr5_real_alias_file_pair_in_ignored_dir_dedup(tmp_path):
    (tmp_path / "target").mkdir()
    _write_py(tmp_path / "target" / "a.py")
    if not _cr5_file_alias(tmp_path / "target" / "a.py", tmp_path / "target" / "b.py"):
        pytest.skip("cannot create a second alias to one file on this host")
    _write_py(tmp_path / "other.py")

    from codedoc.core.scanner import ScanDiagnostics, scan_files
    diag = ScanDiagnostics()
    diag.explicit_entry_hint = "target"
    files = scan_files(
        tmp_path, extension_language_map={".py": "python"},
        ignore_paths=["target"], follow_symlinks=True, diagnostics=diag)
    assert {f["rel_path"] for f in files} == {"other.py"}
    cat = diag.scanner_admission_skip
    assert [d["path"] for d in cat["details"]] == ["target/a.py"]
    assert cat["details_total"] == 1 and cat["details_omitted"] == 0


def test_cr6_pipeline_does_not_replace_skipped_explicit_directory_alias(tmp_path):
    """Case matching must not resolve a skipped junction/symlink to another
    admitted project path when following links is disabled."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    _write_py(tmp_path / "real" / "main.py")
    if not _cr5_dir_link(tmp_path / "real", tmp_path / "alias"):
        pytest.skip("cannot create a directory link on this host")

    reports = []
    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "alias/main.py",
                "dry_run": True,
                "follow_symlinks": False,
                "propagate_changes": False,
            },
            plan_reporter=reports.append,
        )

    assert len(reports) == 1
    assert reports[0]["total_calls_planned"] == 0
    assert not (tmp_path / "codedoc").exists()


def test_cr6_pipeline_ignored_target_is_not_replaced_by_visible_alias(tmp_path):
    """An admitted alias to the same physical file cannot turn an explicitly
    ignored target into payable work."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    target = tmp_path / ".hidden" / "target"
    _write_py(target / "a.py")
    if not _cr5_dir_link(target, tmp_path / "alias"):
        pytest.skip("cannot create a directory link on this host")

    reports = []
    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {
                "entry_file": ".hidden/target/a.py",
                "dry_run": True,
                "follow_symlinks": True,
                "propagate_changes": False,
            },
            plan_reporter=reports.append,
        )

    assert len(reports) == 1
    assert reports[0]["total_calls_planned"] == 0
    assert reports[0]["scanner_admission_skip_details_total"] == 1
    assert reports[0]["scanner_admission_skip_details"][0]["path"] == (
        ".hidden/target/a.py"
    )
    assert not (tmp_path / "codedoc").exists()
