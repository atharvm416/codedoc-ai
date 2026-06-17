"""0.9.6 — the legacy ``write_summary()`` helper is now atomic.

``write_summary`` previously truncated its ``codedoc.md`` target in place via
``Path.write_text``.  It now routes through the canonical ``atomic_write_text``
helper, so a failed write leaves the prior file intact and a unique temporary
sibling is used and cleaned up.
"""

from __future__ import annotations

import pytest

import codedoc.core.block_manager as block_manager
from codedoc.core.output import PROJECT_MARKDOWN, write_summary


def _stats():
    return {"checked": 3, "failed": 0, "skipped": 1, "reused": 2}


def test_write_summary_writes_expected_content(tmp_path):
    path = write_summary(_stats(), tmp_path, error_summary="boom")
    text = path.read_text(encoding="utf-8")
    assert path.name == PROJECT_MARKDOWN
    assert "Files checked: 3" in text
    assert "boom" in text


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
