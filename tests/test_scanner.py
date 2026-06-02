"""Tests for project scanning and ignore rules.

0.8.1: skip_dirs and extension_language_map are now caller-supplied (no
hardcoded SKIP_DIRS constant in scanner.py).  Tests that previously relied on
SKIP_DIRS now pass skip_dirs explicitly to verify the same behavior.
"""


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
