"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError
import json
import os
from tests.support.pipeline_usage import write_py
from codedoc.cli.cli import build_parser
from codedoc.core.discovery import _select_files
from codedoc.pipeline import run_pipeline
from tests.support.selection_projects import _project
from tests.support.selection_projects import _graph_and_map
import math

def test_zero_max_file_size_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"max_file_size_kb": 0})

def test_negative_max_file_size_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"max_file_size_kb": -10})

def test_boolean_max_file_size_rejected(tmp_path):
    # True would otherwise coerce to 1; reject it explicitly.
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"max_file_size_kb": True})

def test_valid_max_file_size_unchanged(tmp_path):
    config = load_config(tmp_path, {"max_file_size_kb": 250})
    assert config["max_file_size_kb"] == 250

def test_negative_retry_after_cap_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"retry_after_cap_s": -1})

def test_boolean_retry_after_cap_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"retry_after_cap_s": True})

def test_zero_retry_after_cap_valid(tmp_path):
    config = load_config(tmp_path, {"retry_after_cap_s": 0})
    assert config["retry_after_cap_s"] == 0

def test_positive_retry_after_cap_valid(tmp_path):
    config = load_config(tmp_path, {"retry_after_cap_s": 45})
    assert config["retry_after_cap_s"] == 45

def test_follow_symlinks_defaults_to_false(tmp_path):
    config = load_config(tmp_path)
    assert config["follow_symlinks"] is False

def test_follow_symlinks_accepts_bool(tmp_path):
    assert load_config(tmp_path, {"follow_symlinks": True})["follow_symlinks"] is True

@pytest.mark.parametrize("value,expected", [
    ("true", True),
    ("1", True),
    ("yes", True),
    ("false", False),
    ("0", False),
    ("no", False),
])
def test_follow_symlinks_accepts_documented_strings(tmp_path, value, expected):
    config = load_config(tmp_path, {"follow_symlinks": value})
    assert config["follow_symlinks"] is expected

def test_follow_symlinks_rejects_unrecognized_string(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"follow_symlinks": "sometimes"})

def test_format_both_with_named_file_raises_config_error(tmp_path):
    """--format both combined with a named output file must raise ConfigError,
    not silently downgrade to a single format."""
    from codedoc.utils.errors import ConfigError
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    try:
        run_pipeline(
            tmp_path,
            {
                "output_dir": "docs/report.md",   # named file
                "output_format": "both",           # conflicts
                "entry_file": "main.py",
            },
        )
        assert False, "Expected ConfigError was not raised"
    except ConfigError as exc:
        assert "both" in str(exc).lower()
        assert "directory" in str(exc).lower()

def test_A7_unsupported_extension_raises_error(tmp_path):
    """A7: --output report.txt → ConfigError (unsupported extension)."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "docs/report.txt"})
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "txt" in str(e).lower() or "unsupported" in str(e).lower()

def test_A9_format_both_named_file_raises_error(tmp_path):
    """A9: --format both --output report.md → ConfigError (not a directory)."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"entry_file": "main.py",
                                 "output_dir": "docs/report.md", "output_format": "both"})
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "both" in str(e).lower()

def test_A10_format_both_named_json_raises_error(tmp_path):
    """A10: --format both --output report.json → ConfigError (not a directory)."""
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError
    (tmp_path / "main.py").write_text("x=1\n")
    try:
        run_pipeline(tmp_path, {"entry_file": "main.py",
                                 "output_dir": "docs/report.json", "output_format": "both"})
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "both" in str(e).lower()

def test_config_defaults_env_and_validation(tmp_path, monkeypatch):
    from codedoc.core.loader import load_config
    from codedoc.utils.errors import ConfigError

    monkeypatch.setenv("CODEDOC_DRY_RUN", "yes")
    monkeypatch.setenv("CODEDOC_MAX_FILES", "7")
    monkeypatch.setenv("CODEDOC_MAX_PLANNED_CALLS", "9")
    monkeypatch.setenv("CODEDOC_FORCE_FILES", "a.py; src/b.py ;")
    monkeypatch.setenv("CODEDOC_ALLOW_PARTIAL", "true")
    config = load_config(tmp_path)

    assert config["dry_run"] is True
    assert config["max_files"] == 7
    assert config["max_planned_calls"] == 9
    assert config["force_files"] == ["a.py", "src/b.py"]
    assert config["allow_partial"] is True

    for invalid in (-1, True, 1.5, "1.5", "+-5", "-+5", object()):
        with pytest.raises(ConfigError):
            load_config(tmp_path, {"max_files": invalid})
        with pytest.raises(ConfigError):
            load_config(tmp_path, {"max_planned_calls": invalid})
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"force_files": [""]})

def test_force_path_normalization_and_root_escape(tmp_path):
    from codedoc.core.planning import normalize_force_files
    from codedoc.utils.errors import ConfigError

    source = tmp_path / "src" / "a.py"
    write_py(source)
    normalized = normalize_force_files(
        ["src/a.py", str(source.resolve()), "src/./a.py", r"src\a.py"],
        tmp_path,
    )

    # The expected result depends on whether the backslash is a path separator
    # on the running platform.  ``normalize_force_files`` is correct per
    # platform; only the assertion must be platform-aware.
    backslash_is_separator = "\\" in (os.sep, os.altsep)
    if backslash_is_separator:
        # ``src\a.py`` collapses into the same entry as ``src/a.py``.
        assert normalized == ["src/a.py"]
    else:
        # The backslash is a legal filename character, so ``src\a.py`` is a
        # distinct relative path and is preserved as its own entry.
        assert normalized == ["src/a.py", r"src\a.py"]

    with pytest.raises(ConfigError):
        normalize_force_files(["../outside.py"], tmp_path)

def test_safe_mode_override_is_rejected_with_migration_guidance(tmp_path):
    from codedoc.cli.cli import build_parser
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import ConfigError

    help_text = build_parser().format_help()
    assert "--safe-mode" not in help_text
    write_py(tmp_path / "main.py")
    with pytest.raises(ConfigError, match="safe_mode") as excinfo:
        run_pipeline(
            tmp_path,
            {"entry_file": "main.py", "safe_mode": True, "dry_run": True},
        )
    assert "always active" in str(excinfo.value)

def test_select_files_defaults_to_entry_and_rejects_invalid_direct_value(tmp_path):
    _project(tmp_path)
    graph, file_map = _graph_and_map(tmp_path)
    reachable, documented, _ = _select_files(
        tmp_path, {"entry_file": "main.py"}, graph, file_map
    )
    assert documented == reachable == {"main.py", "helper.py"}

    with pytest.raises(ConfigError, match="documentation_scope"):
        _select_files(
            tmp_path,
            {"entry_file": "main.py", "documentation_scope": "wide"},
            graph,
            file_map,
        )

def test_loader_and_cli_validate_documentation_scope(tmp_path):
    assert load_config(tmp_path, {})["documentation_scope"] == "entry"
    assert load_config(tmp_path, {"documentation_scope": "all"})[
        "documentation_scope"
    ] == "all"
    with pytest.raises(ConfigError, match="documentation_scope"):
        load_config(tmp_path, {"documentation_scope": "wide"})

    parser = build_parser()
    assert parser.parse_args(["--documentation-scope", "all"]).documentation_scope == "all"
    assert parser.parse_args([]).documentation_scope is None

def test_unknown_keys_are_rejected_from_file_and_overrides(tmp_path):
    path = tmp_path / "codedoc.config.json"
    path.write_text('{"output_fomat":"md","dryrun":true}', encoding="utf-8")
    with pytest.raises(ConfigError, match=r"(?s)dryrun.*output_fomat"):
        load_config(tmp_path)
    path.unlink()
    with pytest.raises(ConfigError, match="max_file"):
        load_config(tmp_path, {"max_file": 1})

@pytest.mark.parametrize(
    "key",
    [
        "parallel_agents",
        "follow_symlinks",
        "propagate_changes",
        "rate_limit_adaptive",
        "respect_retry_after",
        "dry_run",
        "allow_partial",
    ],
)
@pytest.mark.parametrize("value", ["garbage", 0, 1, 2, None, [], {}])
def test_all_boolean_settings_are_strict(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: value})

@pytest.mark.parametrize(
    "key",
    [
        "max_file_size_kb",
        "max_parallel_files",
        "file_retry_attempts",
        "max_consecutive_failures",
        "retry_after_cap_s",
        "max_content_chars",
        "max_files",
        "max_planned_calls",
    ],
)
@pytest.mark.parametrize("value", [True, 1.0])
def test_integer_settings_reject_booleans_and_floats(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: value})

@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_numbers_are_rejected(tmp_path, token):
    (tmp_path / "codedoc.config.json").write_text(
        f'{{"truncation_head_ratio":{token}}}', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="non-finite"):
        load_config(tmp_path)

@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_in_memory_numbers_are_rejected(tmp_path, value):
    with pytest.raises(ConfigError, match="non-finite"):
        load_config(tmp_path, {"rate_limit_backoff_s": value})

@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("skip_dirs", [1]),
        ("ignore_paths", [""]),
        ("extension_language_map", {"py": "python"}),
        ("extension_language_map_add", {".x": ""}),
        ("provider_prefixes", {"openai": "gpt-"}),
        ("provider_prefixes_add", {"openai": [1]}),
        ("rate_limit_signals_add", [None]),
    ],
)
def test_nested_collection_members_are_validated(tmp_path, key, value):
    with pytest.raises(ConfigError, match=key):
        load_config(tmp_path, {key: value})

def test_default_resolves_to_single(tmp_path):
    config = load_config(tmp_path)
    assert config["analysis_mode"] == "single"

def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path, overrides={"analysis_mode": "quadruple"})


@pytest.mark.parametrize("value", ["truncate", "split"])
def test_large_file_strategy_accepts_only_exact_public_values(tmp_path, value):
    overrides = {"large_file_strategy": value}
    if value == "split":
        overrides["dry_run"] = True
    assert load_config(tmp_path, overrides)["large_file_strategy"] == value


@pytest.mark.parametrize(
    "value",
    ["Split", " split", "split ", "fallback", "", True, None],
)
def test_large_file_strategy_rejects_aliases_and_coercions(tmp_path, value):
    with pytest.raises(ConfigError, match="large_file_strategy"):
        load_config(tmp_path, {"large_file_strategy": value})

def test_json_config_value_used(tmp_path):
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps({"analysis_mode": "triple"}), encoding="utf-8"
    )
    config = load_config(tmp_path)
    assert config["analysis_mode"] == "triple"

@pytest.mark.parametrize("alias", ["--single-call", "--three-call"])
def test_unsupported_aliases_rejected(alias):
    from codedoc.cli.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([alias])

def test_removed_risky_override_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="prompt_customization_allow_risky"):
        run_pipeline(tmp_path, {"prompt_customization_allow_risky": True, "dry_run": True})
