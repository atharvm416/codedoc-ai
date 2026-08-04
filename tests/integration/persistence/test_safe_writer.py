"""Tests organized by feature ownership."""

from __future__ import annotations

import json
from pathlib import Path

def _codedoc_json(path: Path, records: list[dict], status: str | None = None) -> None:
    meta: dict = {
        "entry_file": "main.py",
        "schema_version": "1.4",
        "generated_at": "2026-05-30T00:00:00+00:00",
    }
    if status:
        meta["status"] = status
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_codedoc": meta, "files": records}), encoding="utf-8")

def test_K1_safewriter_raises_on_malformed_target(tmp_path):
    """K1 (Scenario R): SafeWriter refuses to overwrite a malformed target file."""
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ConfigError

    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "codedoc.json").write_text("{ not valid json", encoding="utf-8")

    sw = SafeWriter(out / "codedoc.json", "json", "main.py", {})
    try:
        sw.load()
        assert False, "load() should have raised on a malformed target file"
    except ConfigError:
        pass

def test_K2_safewriter_raises_on_foreign_target(tmp_path):
    """K2 (Scenario P): SafeWriter refuses a valid-JSON file with no _codedoc block."""
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ConfigError

    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "codedoc.json").write_text(json.dumps({"name": "not-codedoc"}), encoding="utf-8")

    sw = SafeWriter(out / "codedoc.json", "json", "main.py", {})
    try:
        sw.load()
        assert False, "load() should have raised on a foreign file"
    except ConfigError:
        pass

def test_K3_safewriter_preloads_completed_records(tmp_path):
    """K3 (Scenario Q): a completed codedoc.json is preserved across a safe-mode interrupt.

    After preloading, the first record() flush must keep the prior records, not
    overwrite the file with only the newly processed one.
    """
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    _codedoc_json(out / "codedoc.json", [
        {"path": "a.py", "hash": "HA"},
        {"path": "b.py", "hash": "HB"},
    ])  # completed output — no in_progress status

    sw = SafeWriter(out / "codedoc.json", "json", "main.py", {})
    sw.load()
    sw.record("c.py", {"language": "python"}, file_hash="HC")  # then "interrupt"

    after = json.loads((out / "codedoc.json").read_text(encoding="utf-8"))
    paths = sorted(f["path"] for f in after["files"])
    assert paths == ["a.py", "b.py", "c.py"], paths

def test_5_parallel_worker_records_before_main_collects(tmp_path):
    """Test 5: a worker records into the live backup before returning to main thread."""
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.pipeline import _process_and_record

    out = tmp_path / "codedoc"
    backup = out / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.set_queue_order(["main.py"])
    sw.initialize_empty()

    recorded: list[str] = []
    original_record = sw.record

    def patched_record(rel_path, result, file_hash=""):
        original_record(rel_path, result, file_hash)
        recorded.append(rel_path)

    sw.record = patched_record

    from tests.support.execution_requests import make_execution_request

    request = make_execution_request(tmp_path, "main.py", "x=1\n")

    class FakeOrchestrator:
        class llm:
            provider_name = "fake"
        @staticmethod
        def process(request):
            return {"language": "python", "description": "test"}

    _process_and_record(request, FakeOrchestrator(), sw, split_execution_mode="recovery")

    assert "main.py" in recorded, "record() must be called inside worker"
    data = json.loads(backup.read_text(encoding="utf-8"))
    paths = [f["path"] for f in data["files"]]
    assert "main.py" in paths

def test_5b_has_record_returns_true_after_worker_records(tmp_path):
    """Test 5b: has_record() returns True immediately after record() in worker."""
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    assert not sw.has_record("main.py")
    sw.record("main.py", {"language": "python"}, file_hash="H1")
    assert sw.has_record("main.py")
    assert not sw.has_record("utils.py")
