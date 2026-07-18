"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.configuration_cases import _load

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

def test_C6_remove_skip_dir_codedoc_removes_from_list(tmp_path):
    """C6: After skip_dirs_remove=['codedoc'], 'codedoc' is not in skip_dirs."""
    cfg = _load(tmp_path, entry_file="main.py", skip_dirs_remove=["codedoc"])
    assert "codedoc" not in cfg["skip_dirs"], (
        "'codedoc' must be absent from skip_dirs after skip_dirs_remove"
    )

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
