"""Scanner symlink, junction, and recursion-safety tests (0.9.6).

These tests exercise the iterative, symlink-safe walk introduced in 0.9.6.
Individual tests that require creating a link are skipped when the platform or
permission set does not allow link creation; the module as a whole is never
skipped.
"""

from __future__ import annotations

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
    _write_py(current / "leaf.py")

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
