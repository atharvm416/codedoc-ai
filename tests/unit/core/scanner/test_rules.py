"""Tests organized by feature ownership."""

from __future__ import annotations

import json
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
