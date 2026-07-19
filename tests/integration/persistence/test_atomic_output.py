"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
import codedoc.core.block_manager as block_manager
import codedoc.core.output as output_mod
from codedoc.core.block_manager import atomic_write_text
from codedoc.utils.errors import OutputError
from codedoc.core.document import read_codedoc_document
from codedoc.core.output import PROJECT_MARKDOWN, _is_codedoc_owned, write_summary

def _record(path="a.py"):
    return {
        "hash": "h",
        "file_path": path,
        "language": "python",
        "documentation": {
            "file_path": path,
            "language": "python",
            "description": "desc",
        },
    }

def _write(tmp_path, records, fmt="both"):
    return output_mod.write_project_outputs(
        records,
        {"checked": len(records)},
        tmp_path,
        "",
        fmt,
        None,
        None,
    )

def test_writes_utf8_content(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_text(target, "café ☕ ✓")
    assert target.read_text(encoding="utf-8") == "café ☕ ✓"

def test_writes_lf_line_endings(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_text(target, "a\nb\n")
    assert target.read_bytes() == b"a\nb\n"

def test_no_temp_file_left_after_success(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_text(target, "data")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.json"]
    assert leftovers == []

def test_existing_target_preserved_on_write_failure(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    target.write_text("ORIGINAL", encoding="utf-8")

    def boom(_fd):
        raise OSError("disk full")

    # fsync runs after the bytes are written but before the rename; a failure
    # there must leave the existing target intact and remove the temp file.
    monkeypatch.setattr(block_manager.os, "fsync", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "NEW")

    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.json"]
    assert leftovers == []

def test_oserror_propagates_with_cause(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    cause = OSError("flush failed")

    def boom(_fd):
        raise cause

    monkeypatch.setattr(block_manager.os, "fsync", boom)
    with pytest.raises(OSError) as excinfo:
        atomic_write_text(target, "x")
    assert excinfo.value is cause

def test_encoding_failure_is_cleaned_and_exposed_as_oserror(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("ORIGINAL", encoding="utf-8")

    with pytest.raises(OSError) as excinfo:
        atomic_write_text(target, "\ud800")

    assert isinstance(excinfo.value.__cause__, UnicodeEncodeError)
    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "out.json"]
    assert leftovers == []

def test_stale_temp_sibling_does_not_interfere(tmp_path):
    # A leftover temp file from a previous crashed run must not block a new write
    # and must not be removed by this operation (we only clean our own temp).
    stale = tmp_path / ".out.json.stale.tmp"
    stale.write_text("garbage", encoding="utf-8")
    target = tmp_path / "out.json"
    atomic_write_text(target, "fresh")
    assert target.read_text(encoding="utf-8") == "fresh"
    assert stale.exists()

def test_both_mode_writes_json_and_markdown(tmp_path):
    _write(tmp_path, [_record()])
    assert (tmp_path / "codedoc.json").exists()
    assert (tmp_path / "codedoc.md").exists()

def test_json_only_finalization(tmp_path):
    _write(tmp_path, [_record()], fmt="json")
    assert (tmp_path / "codedoc.json").exists()
    assert not (tmp_path / "codedoc.md").exists()

def test_md_only_finalization(tmp_path):
    _write(tmp_path, [_record()], fmt="md")
    assert (tmp_path / "codedoc.md").exists()

def test_markdown_failure_leaves_prior_json_backup_intact(tmp_path, monkeypatch):
    # First write a clean baseline of both artifacts.
    _write(tmp_path, [_record("a.py")])
    json_before = (tmp_path / "codedoc.json").read_text(encoding="utf-8")
    md_before = (tmp_path / "codedoc.md").read_text(encoding="utf-8")

    real = output_mod.atomic_write_text

    def fail_markdown(path, text):
        if str(path).endswith(".md"):
            raise OSError("md write failed")
        return real(path, text)

    monkeypatch.setattr(output_mod, "atomic_write_text", fail_markdown)

    with pytest.raises(OutputError):
        _write(tmp_path, [_record("b.py")])

    # Markdown is replaced first, so when it fails the JSON live backup is never
    # touched — both artifacts retain their prior complete contents.
    assert (tmp_path / "codedoc.json").read_text(encoding="utf-8") == json_before
    assert (tmp_path / "codedoc.md").read_text(encoding="utf-8") == md_before

def test_json_failure_after_markdown_success(tmp_path, monkeypatch):
    _write(tmp_path, [_record("a.py")])
    json_before = (tmp_path / "codedoc.json").read_text(encoding="utf-8")
    md_before = (tmp_path / "codedoc.md").read_text(encoding="utf-8")

    real = output_mod.atomic_write_text

    def fail_json(path, text):
        if str(path).endswith(".json"):
            raise OSError("json write failed")
        return real(path, text)

    monkeypatch.setattr(output_mod, "atomic_write_text", fail_json)

    with pytest.raises(OutputError):
        _write(tmp_path, [_record("b.py")])

    # JSON is replaced last; Markdown already succeeded with the new document
    # while JSON remains the prior complete live backup.  Neither is truncated.
    assert (tmp_path / "codedoc.md").read_text(encoding="utf-8") != md_before
    assert (tmp_path / "codedoc.json").read_text(encoding="utf-8") == json_before

def test_no_temp_files_left_after_both_mode_write(tmp_path):
    _write(tmp_path, [_record()])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["codedoc.json", "codedoc.md"]
    assert not any(name.endswith(".tmp") for name in names)

def _stats():
    return {"checked": 3, "failed": 0, "skipped": 1, "reused": 2}

def test_write_summary_writes_expected_content(tmp_path):
    path = write_summary(_stats(), tmp_path, error_summary="boom")
    text = path.read_text(encoding="utf-8")
    assert path.name == PROJECT_MARKDOWN
    assert "Files documented by LLM: 3" in text
    assert "Files reused (unchanged): 1" in text
    assert "Files reused (identical content): 2" in text
    assert "boom" in text
    assert _is_codedoc_owned(path)
    doc = read_codedoc_document(path)
    assert doc.view["last_run"]["files_selected"] == 6
    assert doc.view["last_run"]["files_reused_identical_content"] == 2

def test_write_summary_uses_lf_line_endings(tmp_path):
    path = write_summary(_stats(), tmp_path)
    data = path.read_bytes()
    assert b"\r\n" not in data
    assert b"\n" in data

def test_write_summary_preserves_prior_target_on_failure(tmp_path, monkeypatch):
    target = tmp_path / PROJECT_MARKDOWN
    target.write_text("ORIGINAL", encoding="utf-8")

    def boom(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(block_manager.os, "fsync", boom)

    with pytest.raises(OSError):
        write_summary(_stats(), tmp_path)

    # The pre-existing summary is untouched, not truncated in place.
    assert target.read_text(encoding="utf-8") == "ORIGINAL"

def test_write_summary_leaves_no_temp_sibling(tmp_path):
    write_summary(_stats(), tmp_path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != PROJECT_MARKDOWN]
    assert leftovers == []

def test_write_summary_temp_sibling_cleaned_on_failure(tmp_path, monkeypatch):
    target = tmp_path / PROJECT_MARKDOWN
    target.write_text("ORIGINAL", encoding="utf-8")

    monkeypatch.setattr(
        block_manager.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("x"))
    )
    with pytest.raises(OSError):
        write_summary(_stats(), tmp_path)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != PROJECT_MARKDOWN]
    assert leftovers == []
