"""0.9.5 — generated-artifact path-collision validation.

``validate_distinct_artifact_paths`` rejects two distinct logical artifacts that
would target the same normalized path, while accepting the intentional
final-JSON / live-backup phase alias (submitted once under one logical name).
"""
from __future__ import annotations

import pytest

import codedoc.core.output as output_mod
from codedoc.core.output import validate_distinct_artifact_paths
from codedoc.utils.errors import ConfigError


def test_distinct_paths_pass(tmp_path):
    validate_distinct_artifact_paths(
        {
            "json_live_backup": tmp_path / "codedoc.json",
            "markdown": tmp_path / "codedoc.md",
            "error_log": tmp_path / "error.log",
        }
    )


def test_none_values_are_ignored(tmp_path):
    validate_distinct_artifact_paths(
        {
            "markdown": tmp_path / "codedoc.md",
            "json_live_backup": None,
            "error_log": None,
        }
    )


def test_intentional_json_live_backup_alias_is_not_a_collision(tmp_path):
    # The final JSON and its live-backup phase share one path; representing it as
    # a single logical artifact must never trip the collision check.
    validate_distinct_artifact_paths(
        {
            "json_live_backup": tmp_path / "codedoc.json",
            "markdown": tmp_path / "codedoc.md",
            "error_log": tmp_path / "error.log",
        }
    )


def test_json_versus_markdown_collision(tmp_path):
    with pytest.raises(ConfigError) as excinfo:
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "out.txt",
                "markdown": tmp_path / "out.txt",
            }
        )
    message = str(excinfo.value)
    assert "json_live_backup" in message
    assert "markdown" in message


def test_output_versus_error_log_collision(tmp_path):
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "error.log",
                "error_log": tmp_path / "error.log",
            }
        )


def test_markdown_versus_json_backup_collision(tmp_path):
    # Markdown-only mode keeps the live backup as a JSON sibling; if a user names
    # the Markdown file identically to the backup it must be rejected.
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "markdown": tmp_path / "report.json",
                "live_backup": tmp_path / "report.json",
            }
        )


def test_relative_segments_are_normalized(tmp_path):
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "codedoc.json",
                "markdown": tmp_path / "sub" / ".." / "codedoc.json",
            }
        )


def test_case_equivalent_paths_collide_on_case_insensitive_fs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        output_mod,
        "_filesystem_is_case_insensitive",
        lambda _path: True,
    )
    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths(
            {
                "json_live_backup": tmp_path / "CodeDoc.json",
                "markdown": tmp_path / "codedoc.json",
            }
        )


def test_case_equivalent_paths_are_distinct_on_case_sensitive_fs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        output_mod,
        "_filesystem_is_case_insensitive",
        lambda _path: False,
    )
    validate_distinct_artifact_paths(
        {
            "json_live_backup": tmp_path / "CodeDoc.json",
            "markdown": tmp_path / "codedoc.json",
        }
    )
