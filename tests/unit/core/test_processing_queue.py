"""Tests organized by feature ownership."""

from __future__ import annotations

from codedoc.core.queue import (
    ProcessingQueue,
    STATUS_CHECKED,
    STATUS_FAILED,
    STATUS_SKIPPED_INSUFFICIENT_SOURCE,
    STATUS_UNCHECKED,
)
import json

def make_descriptor(rel_path: str) -> dict:
    return {"rel_path": rel_path, "path": None, "language": "python", "extension": ".py"}

class TestProcessingQueue:
    def test_add_and_next(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        item = q.next()
        assert item["rel_path"] == "a.py"

    def test_no_duplicate_adds(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.add(make_descriptor("a.py"))
        q.next()
        assert q.next() is None

    def test_mark_checked(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.next()
        q.mark_checked("a.py")
        assert q.status_of("a.py") == STATUS_CHECKED

    def test_mark_failed(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.next()
        q.mark_failed("a.py", "boom")
        assert q.status_of("a.py") == STATUS_FAILED

    def test_mark_skipped_insufficient_source_is_terminal(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.next()
        q.mark_skipped_insufficient_source("a.py", "empty_or_whitespace_only")
        assert q.status_of("a.py") == STATUS_SKIPPED_INSUFFICIENT_SOURCE
        assert q.all_checked()
        assert not q.has_pending()
        assert q.stats()[STATUS_SKIPPED_INSUFFICIENT_SOURCE] == 1

    def test_all_checked(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.add(make_descriptor("b.py"))
        q.next()
        q.mark_checked("a.py")
        q.next()
        q.mark_checked("b.py")
        assert q.all_checked()

    def test_has_pending(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        assert q.has_pending()
        q.next()
        q.mark_checked("a.py")
        assert not q.has_pending()

    def test_stats(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.add(make_descriptor("b.py"))
        q.next()
        q.mark_checked("a.py")
        stats = q.stats()
        assert stats["checked"] == 1
        assert stats["unchecked"] == 1

    def test_snapshot(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        snap = q.snapshot()
        assert "a.py" in snap
        assert snap["a.py"] == STATUS_UNCHECKED

    def test_iter_checked(self):
        q = ProcessingQueue()
        q.add(make_descriptor("a.py"))
        q.add(make_descriptor("b.py"))
        q.next()
        q.mark_checked("a.py")
        q.next()
        q.mark_failed("b.py", "err")
        checked = list(q.iter_checked())
        assert len(checked) == 1
        assert checked[0]["rel_path"] == "a.py"

    def test_empty_queue_returns_none(self):
        q = ProcessingQueue()
        assert q.next() is None

def test_4b_queue_order_with_no_set_queue_order_falls_back_to_alpha(tmp_path):
    """Test 4b: without set_queue_order, fallback is alphabetical."""
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    backup = out / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.record("c.py", {"language": "python"}, file_hash="HC")
    sw.record("a.py", {"language": "python"}, file_hash="HA")

    data = json.loads(backup.read_text(encoding="utf-8"))
    paths = [f["path"] for f in data["files"]]
    assert paths == ["a.py", "c.py"], f"Expected alpha fallback, got {paths}"
