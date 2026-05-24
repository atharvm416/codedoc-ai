"""Tests for project scanning and ignore rules."""


def test_scan_skips_virtualenv_name_by_default(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')\n")
    env_dir = tmp_path / "myenv"
    env_dir.mkdir()
    (env_dir / "installed.py").write_text("print('skip')\n")

    from codedoc.core.scanner import scan_files

    files = scan_files(tmp_path, supported_extensions=[".py"])
    rels = {f["rel_path"] for f in files}

    assert "main.py" in rels
    assert "myenv/installed.py" not in rels


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
