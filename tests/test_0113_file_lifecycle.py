"""Focused 0.11.3 tests for exact output and recovery lifecycle."""

from pathlib import Path

import pytest

from codedoc.core.resume import (
    RECOVERY_FILENAME,
    _load_existing_file_docs,
    build_recovery_identity,
    load_recovery_records_if_compatible,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.utils.errors import ConfigError


def _identity(tmp_path: Path, **changes):
    values = {
        "project_root": tmp_path,
        "json_target": tmp_path / "docs" / "codedoc.json",
        "md_target": None,
        "entry_file": "main.py",
        "documentation_scope": "entry",
        "analysis_mode": "single",
        "analysis_revision": "file-doc-v2",
        "profile_digests_by_language": {"python": "none"},
    }
    values.update(changes)
    return build_recovery_identity(**values)


def test_fixed_recovery_name_and_identity_mismatch_blocks(tmp_path):
    path = tmp_path / "docs" / RECOVERY_FILENAME
    identity = _identity(tmp_path)
    writer = SafeWriter(path, "json", "main.py", {}, identity)
    writer.initialize_empty()
    assert path.name == "crash_recovery.json"
    assert load_recovery_records_if_compatible(path, identity) == {}
    with pytest.raises(ConfigError, match="analysis_mode expected.*triple.*single"):
        load_recovery_records_if_compatible(
            path, _identity(tmp_path, analysis_mode="triple")
        )
    assert path.exists()


def test_missing_target_rejects_foreign_opposite_format_sibling(tmp_path):
    json_target = tmp_path / "codedoc.json"
    md_sibling = tmp_path / "codedoc.md"
    md_sibling.write_text("foreign sibling", encoding="utf-8")
    with pytest.raises(ConfigError, match="conversion sibling"):
        _load_existing_file_docs(json_target, md_sibling, "json")


def test_completed_run_uses_only_fixed_recovery_filename(tmp_path, monkeypatch):
    from tests.test_110_prompt_profile_cli import SmartFake
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    run_pipeline(tmp_path, {"entry_file": "main.py", "propagate_changes": False})
    output = tmp_path / "codedoc"
    assert (output / "codedoc.json").exists()
    assert not (output / RECOVERY_FILENAME).exists()
    assert not list(output.glob("crash_recovery_*.json"))


def test_both_mode_cross_document_hash_conflict_blocks(tmp_path, monkeypatch):
    """A both-mode JSON/Markdown pair whose per-file content hash disagrees is a
    deterministic pre-provider conflict, not a silently-picked-by-mtime target."""
    import json as json_mod

    from tests.test_110_prompt_profile_cli import SmartFake
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "output_format": "both", "propagate_changes": False},
    )
    output = tmp_path / "codedoc"
    json_target = output / "codedoc.json"
    md_target = output / "codedoc.md"
    assert json_target.exists() and md_target.exists()

    doc = json_mod.loads(json_target.read_text(encoding="utf-8"))
    doc["files"][0]["hash"] = "0" * 64
    json_target.write_text(json_mod.dumps(doc), encoding="utf-8")

    with pytest.raises(ConfigError, match="content hash"):
        _load_existing_file_docs(json_target, md_target, "both")


def test_both_mode_cross_document_entry_conflict_blocks(tmp_path, monkeypatch):
    """A both-mode pair whose recorded entry file disagrees is a deterministic
    pre-provider conflict."""
    import json as json_mod

    from tests.test_110_prompt_profile_cli import SmartFake
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "output_format": "both", "propagate_changes": False},
    )
    output = tmp_path / "codedoc"
    json_target = output / "codedoc.json"
    md_target = output / "codedoc.md"

    doc = json_mod.loads(json_target.read_text(encoding="utf-8"))
    doc["_codedoc"]["entry_file"] = "other.py"
    json_target.write_text(json_mod.dumps(doc), encoding="utf-8")

    with pytest.raises(ConfigError, match="entry file"):
        _load_existing_file_docs(json_target, md_target, "both")
