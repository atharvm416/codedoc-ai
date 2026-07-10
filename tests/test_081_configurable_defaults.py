"""
0.8.1 — Configurable hardcoded defaults tests (Workstream C).

Plan acceptance criteria:
  C1   Defaults are unchanged when no override keys are set.
  C2   <key> fully replaces defaults.
  C3   <key>_add appends or updates without duplicates.
  C4   <key>_remove removes without error.
  C5   Combined replace/add/remove resolves deterministically.
  C6   --remove-skip-dir codedoc allows scanning codedoc/ source package
       (i.e. "codedoc" is absent from the resolved skip_dirs list).
  C7   Output directory is still appended to scan skip list even when
       "codedoc" is removed from config skip_dirs.
  C8   New extension mapping (e.g. .svelte) scans and labels language correctly.
  C9   Custom auto-entry candidate is used when no --entry is given.
  C10  First run with no --entry and a detectable default candidate succeeds.
  C11  First run with no --entry and no detectable auto-entry processes all files.
  C12  Provider prefixes affect provider selection consistently.
  C13  _provider_api_key uses the same resolved prefixes as _resolve_api_provider.
  C14  CLI --skip-dirs replaces defaults.
  C15  CLI --add-skip-dir appends to skip_dirs.
  C16  CLI --remove-skip-dir removes from skip_dirs.
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(tmp_path, **overrides):
    from codedoc.core.loader import load_config
    return load_config(tmp_path, overrides if overrides else None)


def _fake_provider():
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({"description": "ok", "role_in_system": "r",
                                    "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _json.dumps({"description": "ok", "role_in_system": "r",
                                "functions": [], "classes": [], "exports": []})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


# ---------------------------------------------------------------------------
# C1 — Defaults unchanged when no override keys set
# ---------------------------------------------------------------------------

def test_C1_skip_dirs_default_unchanged(tmp_path):
    """C1a: skip_dirs default list is intact when no overrides are given."""
    cfg = _load(tmp_path, entry_file="main.py")
    from codedoc.core.loader import DEFAULTS
    assert cfg["skip_dirs"] == DEFAULTS["skip_dirs"], (
        "skip_dirs must equal DEFAULTS when no override is set"
    )


def test_C1_extension_language_map_default_intact(tmp_path):
    """C1b: extension_language_map default is intact when no overrides are given."""
    cfg = _load(tmp_path, entry_file="main.py")
    assert cfg["extension_language_map"][".py"] == "python"
    assert cfg["extension_language_map"][".dart"] == "dart"


def test_C1_auto_entry_candidates_default_intact(tmp_path):
    """C1c: auto_entry_candidates default is intact when no overrides are given."""
    cfg = _load(tmp_path, entry_file="main.py")
    from codedoc.core.loader import DEFAULTS
    assert cfg["auto_entry_candidates"] == DEFAULTS["auto_entry_candidates"]


def test_C1_provider_prefixes_default_intact(tmp_path):
    """C1d: provider_prefixes default is intact when no overrides are given."""
    cfg = _load(tmp_path, entry_file="main.py")
    assert "claude" in cfg["provider_prefixes"].get("anthropic", [])
    assert "gemini" in cfg["provider_prefixes"].get("gemini", [])
    assert "gpt-" in cfg["provider_prefixes"].get("openai", [])


# ---------------------------------------------------------------------------
# C2 — <key> fully replaces defaults
# ---------------------------------------------------------------------------

def test_C2_skip_dirs_full_replace(tmp_path):
    """C2a: skip_dirs=[...] replaces the entire default list."""
    cfg = _load(tmp_path, entry_file="main.py", skip_dirs=["custom_dir"])
    assert cfg["skip_dirs"] == ["custom_dir"], (
        "skip_dirs must be replaced entirely when supplied"
    )
    assert "__pycache__" not in cfg["skip_dirs"]


def test_C2_extension_language_map_full_replace(tmp_path):
    """C2b: extension_language_map={...} replaces the entire default map."""
    cfg = _load(tmp_path, entry_file="main.py",
                extension_language_map={".svelte": "svelte"})
    assert cfg["extension_language_map"] == {".svelte": "svelte"}, (
        "extension_language_map must be replaced entirely when supplied"
    )
    assert ".py" not in cfg["extension_language_map"]


def test_C2_auto_entry_candidates_full_replace(tmp_path):
    """C2c: auto_entry_candidates=[...] replaces the entire default list."""
    cfg = _load(tmp_path, entry_file="main.py",
                auto_entry_candidates=["custom_entry.py"])
    assert cfg["auto_entry_candidates"] == ["custom_entry.py"]
    assert "main.py" not in cfg["auto_entry_candidates"]


def test_C2_provider_prefixes_full_replace(tmp_path):
    """C2d: provider_prefixes={...} replaces the entire default dict."""
    cfg = _load(tmp_path, entry_file="main.py",
                provider_prefixes={"custom_provider": ["custom-model-"]})
    assert list(cfg["provider_prefixes"].keys()) == ["custom_provider"]
    assert "anthropic" not in cfg["provider_prefixes"]


# ---------------------------------------------------------------------------
# C3 — <key>_add appends or updates without duplicates
# ---------------------------------------------------------------------------

def test_C3_skip_dirs_add_appends(tmp_path):
    """C3a: skip_dirs_add appends new items without duplicates."""
    cfg = _load(tmp_path, entry_file="main.py", skip_dirs_add=["my_generated"])
    assert "my_generated" in cfg["skip_dirs"]
    # Defaults still present
    assert "__pycache__" in cfg["skip_dirs"]


def test_C3_skip_dirs_add_no_duplicates(tmp_path):
    """C3b: skip_dirs_add does not create duplicates."""
    cfg = _load(tmp_path, entry_file="main.py",
                skip_dirs_add=["codedoc", "extra_dir"])  # "codedoc" already in defaults
    assert cfg["skip_dirs"].count("codedoc") == 1, "No duplicate 'codedoc' entries"
    assert "extra_dir" in cfg["skip_dirs"]


def test_C3_extension_language_map_add_extends(tmp_path):
    """C3c: extension_language_map_add extends the default map."""
    cfg = _load(tmp_path, entry_file="main.py",
                extension_language_map_add={".svelte": "svelte", ".vue": "vue"})
    assert cfg["extension_language_map"][".svelte"] == "svelte"
    assert cfg["extension_language_map"][".vue"] == "vue"
    assert cfg["extension_language_map"][".py"] == "python"  # default preserved


def test_C3_auto_entry_candidates_add_appends(tmp_path):
    """C3d: auto_entry_candidates_add appends new entries."""
    cfg = _load(tmp_path, entry_file="main.py",
                auto_entry_candidates_add=["app.py", "index.py"])
    assert "app.py" in cfg["auto_entry_candidates"]
    assert "index.py" in cfg["auto_entry_candidates"]
    assert "main.py" in cfg["auto_entry_candidates"]  # default preserved


def test_C3_provider_prefixes_add_extends(tmp_path):
    """C3e: provider_prefixes_add appends extra prefixes per provider."""
    cfg = _load(tmp_path, entry_file="main.py",
                provider_prefixes_add={"anthropic": ["claude2"], "custom": ["custom-"]})
    anthropic_prefixes = cfg["provider_prefixes"]["anthropic"]
    assert "claude" in anthropic_prefixes   # default preserved
    assert "claude2" in anthropic_prefixes  # added
    assert "custom-" in cfg["provider_prefixes"]["custom"]


# ---------------------------------------------------------------------------
# C4 — <key>_remove removes without error
# ---------------------------------------------------------------------------

def test_C4_skip_dirs_remove(tmp_path):
    """C4a: skip_dirs_remove drops the specified entry."""
    cfg = _load(tmp_path, entry_file="main.py", skip_dirs_remove=["codedoc"])
    assert "codedoc" not in cfg["skip_dirs"], (
        "skip_dirs_remove must remove 'codedoc' from skip_dirs"
    )
    assert "__pycache__" in cfg["skip_dirs"]  # other defaults intact


def test_C4_skip_dirs_remove_nonexistent_no_error(tmp_path):
    """C4b: Removing an entry that is not in skip_dirs does not raise."""
    cfg = _load(tmp_path, entry_file="main.py",
                skip_dirs_remove=["nonexistent_dir"])  # should not raise
    assert cfg["skip_dirs"]  # list still has defaults


def test_C4_extension_language_map_remove(tmp_path):
    """C4c: extension_language_map_remove drops the specified extension."""
    cfg = _load(tmp_path, entry_file="main.py",
                extension_language_map_remove=[".htm"])
    assert ".htm" not in cfg["extension_language_map"]
    assert ".html" in cfg["extension_language_map"]  # sibling preserved


def test_C4_auto_entry_candidates_remove(tmp_path):
    """C4d: auto_entry_candidates_remove drops the specified entry."""
    cfg = _load(tmp_path, entry_file="main.py",
                auto_entry_candidates_remove=["index.html"])
    assert "index.html" not in cfg["auto_entry_candidates"]
    assert "main.py" in cfg["auto_entry_candidates"]


def test_C4_provider_prefixes_remove(tmp_path):
    """C4e: provider_prefixes_remove drops specific prefixes per provider."""
    cfg = _load(tmp_path, entry_file="main.py",
                provider_prefixes_remove={"openai": ["o1"]})
    assert "o1" not in cfg["provider_prefixes"]["openai"]
    assert "gpt-" in cfg["provider_prefixes"]["openai"]  # other openai prefix intact


# ---------------------------------------------------------------------------
# C5 — Combined replace/add/remove resolves deterministically
# ---------------------------------------------------------------------------

def test_C5_combined_skip_dirs_add_remove(tmp_path):
    """C5: Combined skip_dirs_add + skip_dirs_remove resolves correctly."""
    cfg = _load(tmp_path, entry_file="main.py",
                skip_dirs_add=["my_output"],
                skip_dirs_remove=["codedoc", "dist"])
    assert "my_output" in cfg["skip_dirs"]
    assert "codedoc" not in cfg["skip_dirs"]
    assert "dist" not in cfg["skip_dirs"]
    assert "__pycache__" in cfg["skip_dirs"]  # unaffected default


def test_C5_replace_then_add(tmp_path):
    """C5b: Full replace + _add is resolved: base=replacement, then add applied."""
    cfg = _load(tmp_path, entry_file="main.py",
                skip_dirs=["base_dir"],
                skip_dirs_add=["extra_dir"])
    assert cfg["skip_dirs"] == ["base_dir", "extra_dir"]


def test_C5_extension_map_add_and_remove(tmp_path):
    """C5c: extension_language_map_add + _remove resolves deterministically."""
    cfg = _load(tmp_path, entry_file="main.py",
                extension_language_map_add={".svelte": "svelte"},
                extension_language_map_remove=[".htm"])
    assert ".svelte" in cfg["extension_language_map"]
    assert ".htm" not in cfg["extension_language_map"]
    assert ".html" in cfg["extension_language_map"]


# ---------------------------------------------------------------------------
# C6 — --remove-skip-dir codedoc removes "codedoc" from skip_dirs
# ---------------------------------------------------------------------------

def test_C6_remove_skip_dir_codedoc_removes_from_list(tmp_path):
    """C6: After skip_dirs_remove=['codedoc'], 'codedoc' is not in skip_dirs."""
    cfg = _load(tmp_path, entry_file="main.py", skip_dirs_remove=["codedoc"])
    assert "codedoc" not in cfg["skip_dirs"], (
        "'codedoc' must be absent from skip_dirs after skip_dirs_remove"
    )


# ---------------------------------------------------------------------------
# C7 — Output directory is always in scan skip list
# ---------------------------------------------------------------------------

def test_C7_output_dir_auto_appended_to_scan_skip(tmp_path, monkeypatch):
    """C7: Pipeline always adds output dir to scan skip even if removed from skip_dirs."""
    # Create source files in a dir named "codedoc" (like the actual package)
    pkg_dir = tmp_path / "codedoc"
    pkg_dir.mkdir()
    (pkg_dir / "__main__.py").write_text("pass\n")
    (tmp_path / "main.py").write_text("pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

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


# ---------------------------------------------------------------------------
# C8 — New extension mapping scans and labels language correctly
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# C9 — Custom auto-entry candidate is used
# ---------------------------------------------------------------------------

def test_C9_custom_auto_entry_detected(tmp_path, monkeypatch):
    """C9: A custom auto_entry_candidates_add entry is found and used."""
    (tmp_path / "app_start.py").write_text("pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    # No --entry; should auto-detect "app_start.py" from custom candidates
    run_pipeline(tmp_path, {
        "auto_entry_candidates_add": ["app_start.py"],
        "parallel_agents": False,
        "propagate_changes": False,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result.get("last_run", {}).get("entry_file") == "app_start.py", (
        "Custom auto-entry candidate must be detected and stored as entry_file"
    )


# ---------------------------------------------------------------------------
# C10 — First run with detectable default auto-entry succeeds
# ---------------------------------------------------------------------------

def test_C10_default_auto_entry_main_py_detected(tmp_path, monkeypatch):
    """C10: Default auto-entry 'main.py' is found and used on first run."""
    (tmp_path / "main.py").write_text("def main(): pass\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "parallel_agents": False,
        "propagate_changes": False,
        # No entry_file — auto-detection should find main.py
    })

    assert stats["checked"] >= 1 or stats.get("reused", 0) >= 1


# ---------------------------------------------------------------------------
# C11 — No --entry and no auto-entry processes all files
# ---------------------------------------------------------------------------

def test_C11_no_entry_no_auto_candidate_processes_all(tmp_path, monkeypatch):
    """C11: Without --entry and no auto-entry match, all supported files are processed."""
    (tmp_path / "helper.py").write_text("x = 1\n")
    (tmp_path / "util.py").write_text("y = 2\n")

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline

    # Remove all default auto_entry_candidates so none match
    stats = run_pipeline(tmp_path, {
        "auto_entry_candidates": [],  # no candidates at all
        "parallel_agents": False,
        "propagate_changes": False,
    })

    # All .py files should be processed (entry_file = None → all files)
    total = stats["checked"] + stats.get("reused", 0)
    assert total >= 2, (
        f"All files must be processed when no entry is detected, got stats={stats}"
    )


# ---------------------------------------------------------------------------
# C12 — Provider prefixes affect provider selection
# ---------------------------------------------------------------------------

def test_C12_provider_prefixes_auto_detect_anthropic(tmp_path):
    """C12a: A model starting with 'claude' is auto-detected as anthropic."""
    from codedoc.llm.factory import _resolve_api_provider
    from codedoc.core.loader import DEFAULTS

    prefixes = DEFAULTS["provider_prefixes"]
    assert _resolve_api_provider("auto", "claude-haiku-4-5", prefixes) == "anthropic"


def test_C12_provider_prefixes_auto_detect_gemini(tmp_path):
    """C12b: A model starting with 'gemini' is auto-detected as gemini."""
    from codedoc.llm.factory import _resolve_api_provider
    from codedoc.core.loader import DEFAULTS

    prefixes = DEFAULTS["provider_prefixes"]
    assert _resolve_api_provider("auto", "gemini-2.5-flash", prefixes) == "gemini"


def test_C12_provider_prefixes_custom_prefix(tmp_path):
    """C12c: A custom prefix in provider_prefixes is used for auto-detection."""
    from codedoc.llm.factory import _resolve_api_provider

    custom_prefixes = {
        "anthropic": ["claude"],
        "gemini":    ["gemini"],
        "openai":    ["gpt-", "o1", "o3", "text-", "mycompany-"],
    }
    assert _resolve_api_provider("auto", "mycompany-model-v2", custom_prefixes) == "openai"


def test_C12_remove_prefix_changes_detection(tmp_path):
    """C12d: Removing a prefix via provider_prefixes_remove changes auto-detection."""
    cfg = _load(tmp_path, entry_file="main.py",
                provider_prefixes_remove={"openai": ["o1"]})

    from codedoc.llm.factory import _resolve_api_provider

    prefixes = cfg["provider_prefixes"]
    # "o1-mini" no longer has the "o1" prefix → should fall through to openai (default)
    # but not because of the "o1" prefix — only if "gpt-" etc. match
    assert "o1" not in prefixes["openai"], "o1 prefix must be removed"
    # "o1-mini" still falls through to openai as the default
    assert _resolve_api_provider("auto", "o1-mini", prefixes) == "openai"


# ---------------------------------------------------------------------------
# C13 — _provider_api_key uses the same resolved prefixes
# ---------------------------------------------------------------------------

def test_C13_api_key_lookup_consistent_with_detection(tmp_path, monkeypatch):
    """C13: _provider_api_key uses the same prefixes as _resolve_api_provider."""
    from codedoc.llm.factory import _provider_api_key, _resolve_api_provider

    custom_prefixes = {
        "anthropic": ["myclaude"],  # custom prefix
        "gemini":    ["gemini"],
        "openai":    ["gpt-"],
    }

    # Simulate env: anthropic key is set
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")

    provider = _resolve_api_provider("auto", "myclaude-v3", custom_prefixes)
    assert provider == "anthropic"

    key = _provider_api_key("auto", "myclaude-v3", custom_prefixes)
    assert key == "sk-test-anthropic", (
        "_provider_api_key must use the same prefixes and return the anthropic key"
    )


# ---------------------------------------------------------------------------
# C14-C16 — CLI flags
# ---------------------------------------------------------------------------

def test_C14_cli_skip_dirs_replaces_defaults():
    """C14: --skip-dirs replaces the default list."""
    from codedoc.cli.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--skip-dirs", "a", "b", "--entry", "main.py"])
    assert args.skip_dirs == ["a", "b"]
    assert args.add_skip_dirs == []
    assert args.remove_skip_dirs == []


def test_C15_cli_add_skip_dir_is_repeatable():
    """C15: --add-skip-dir can be repeated to build a list."""
    from codedoc.cli.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "--add-skip-dir", "generated",
        "--add-skip-dir", "vendor",
        "--entry", "main.py",
    ])
    assert args.add_skip_dirs == ["generated", "vendor"]


def test_C16_cli_remove_skip_dir_is_repeatable():
    """C16: --remove-skip-dir can be repeated to build a list."""
    from codedoc.cli.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "--remove-skip-dir", "codedoc",
        "--remove-skip-dir", "dist",
        "--entry", "main.py",
    ])
    assert args.remove_skip_dirs == ["codedoc", "dist"]


def test_C16_cli_remove_skip_dir_wired_to_overrides(tmp_path, monkeypatch, capsys):
    """C16b: --remove-skip-dir codedoc removes 'codedoc' from resolved skip_dirs."""
    (tmp_path / "main.py").write_text("x=1\n")

    # Capture the resolved skip_dirs by intercepting run_pipeline
    captured = {}

    def fake_run(root, config_overrides=None, **_kwargs):
        from codedoc.core.loader import load_config
        cfg = load_config(root, config_overrides)
        captured["skip_dirs"] = cfg["skip_dirs"]
        return {"checked": 0, "failed": 0, "skipped": 0, "reused": 0,
                "output_dir": str(root), "output_files": [],
                "rate_limit_warnings": [], "issues_recorded": 0,
                "error_log": None, "live_backup_path": None}

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run)
    monkeypatch.setattr("codedoc.cli.cli.sys.argv", [
        "codedoc", "run", str(tmp_path),
        "--entry", "main.py",
        "--remove-skip-dir", "codedoc",
    ])

    from codedoc.cli.cli import main
    try:
        main(["run", str(tmp_path), "--entry", "main.py", "--remove-skip-dir", "codedoc"])
    except SystemExit:
        pass

    assert "codedoc" not in captured.get("skip_dirs", ["codedoc"]), (
        "'codedoc' must be removed from skip_dirs via --remove-skip-dir"
    )


# ---------------------------------------------------------------------------
# supported_extensions derived from extension_language_map
# ---------------------------------------------------------------------------

def test_supported_extensions_derived_from_map(tmp_path):
    """supported_extensions is always derived from extension_language_map after load."""
    cfg = _load(tmp_path, entry_file="main.py",
                extension_language_map_add={".svelte": "svelte"})
    assert ".svelte" in cfg["supported_extensions"], (
        "supported_extensions must include extensions added via extension_language_map_add"
    )
    assert ".py" in cfg["supported_extensions"]


def test_supported_extensions_reflects_remove(tmp_path):
    """supported_extensions is updated when an extension is removed from the map."""
    cfg = _load(tmp_path, entry_file="main.py",
                extension_language_map_remove=[".htm"])
    assert ".htm" not in cfg["supported_extensions"]
    assert ".html" in cfg["supported_extensions"]


# ---------------------------------------------------------------------------
# Regression: P2 — scan_files positional list caller
# ---------------------------------------------------------------------------

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
# Regression: P3 — explicit supported_extensions override is honoured
# ---------------------------------------------------------------------------

def test_P3_explicit_supported_extensions_restricts_map(tmp_path):
    """P3 regression: explicit supported_extensions=['.py'] restricts scanning to .py.

    Pre-0.8.1 configs that set supported_extensions must not silently expand
    to the full default extension list after the 0.8.1 upgrade.
    """
    cfg = _load(tmp_path, entry_file="main.py",
                supported_extensions=[".py"])
    # Only .py should be in the resolved map
    assert list(cfg["extension_language_map"].keys()) == [".py"], (
        "extension_language_map must be filtered to the explicitly listed extensions"
    )
    assert cfg["supported_extensions"] == [".py"], (
        "supported_extensions must reflect the user's explicit restriction"
    )
    # .ts must NOT be included
    assert ".ts" not in cfg["extension_language_map"]


def test_P3_explicit_supported_extensions_subset(tmp_path):
    """P3: supported_extensions=['.py', '.ts'] keeps only those two extensions."""
    cfg = _load(tmp_path, entry_file="main.py",
                supported_extensions=[".py", ".ts"])
    assert sorted(cfg["extension_language_map"].keys()) == [".py", ".ts"]
    assert ".dart" not in cfg["extension_language_map"]


def test_P3_default_supported_extensions_not_treated_as_override(tmp_path):
    """P3: When supported_extensions equals the DEFAULTS value, no filtering is applied.

    Passing the exact default list must not accidentally restrict the map —
    extension_language_map_add entries must still be included.
    """
    from codedoc.core.loader import DEFAULTS

    cfg = _load(tmp_path, entry_file="main.py",
                supported_extensions=list(DEFAULTS["supported_extensions"]),
                extension_language_map_add={".svelte": "svelte"})
    # extension_language_map_add must still work
    assert ".svelte" in cfg["extension_language_map"], (
        "extension_language_map_add must work when supported_extensions equals defaults"
    )


def test_P3_legacy_config_json_with_supported_extensions(tmp_path):
    """P3: A codedoc.config.json with supported_extensions restricts scanning."""
    config_json = tmp_path / "codedoc.config.json"
    config_json.write_text(
        '{"supported_extensions": [".ts", ".tsx"]}',
        encoding="utf-8",
    )
    from codedoc.core.loader import load_config
    cfg = load_config(tmp_path)
    assert ".ts" in cfg["extension_language_map"]
    assert ".tsx" in cfg["extension_language_map"]
    assert ".py" not in cfg["extension_language_map"], (
        "Pre-0.8.1 config with supported_extensions must restrict the map"
    )
