"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from codedoc.core.loader import load_config
from codedoc.pipeline import run_pipeline
from tests.support.configuration_cases import _fake_provider

def test_documentation_scope_precedence_cli_over_config_file(tmp_path):
    """An absent CLI flag preserves the config-file value; a flag overrides it."""
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"documentation_scope": "all"}), encoding="utf-8"
    )
    assert load_config(tmp_path, None)["documentation_scope"] == "all"
    assert load_config(tmp_path, {})["documentation_scope"] == "all"
    assert (
        load_config(tmp_path, {"documentation_scope": "entry"})["documentation_scope"]
        == "entry"
    )

def test_C7_output_dir_auto_appended_to_scan_skip(tmp_path, monkeypatch):
    """C7: Pipeline always adds output dir to scan skip even if removed from skip_dirs."""
    # Create source files in a dir named "codedoc" (like the actual package)
    pkg_dir = tmp_path / "codedoc"
    pkg_dir.mkdir()
    (pkg_dir / "__main__.py").write_text("pass\n")
    (tmp_path / "main.py").write_text("pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())

    # Remove "codedoc" from skip_dirs — the source package should be scanned.
    # But the output goes to a different dir ("docs") so that is auto-skipped.
    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "output_dir": "docs",
        "skip_dirs_remove": ["codedoc"],  # allow scanning codedoc/ source
        "parallel_agents": False,
        "propagate_changes": False,
    })

    # The output dir "docs" must not have been scanned as a source
    out = tmp_path / "docs" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    scanned_paths = {f["path"] for f in result.get("files", [])}
    # docs/ contents must not be in the scanned paths
    assert not any("docs" in p for p in scanned_paths), (
        "Output directory 'docs' must not appear in scanned file paths"
    )

def test_env_overrides_json(tmp_path, monkeypatch):
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"analysis_mode": "single"}), encoding="utf-8"
    )
    monkeypatch.setenv("CODEDOC_ANALYSIS_MODE", "triple")
    config = load_config(tmp_path)
    assert config["analysis_mode"] == "triple"

def test_overrides_take_highest_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEDOC_ANALYSIS_MODE", "single")
    config = load_config(tmp_path, overrides={"analysis_mode": "triple"})
    assert config["analysis_mode"] == "triple"
