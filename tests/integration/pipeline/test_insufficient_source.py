"""Tests organized by feature ownership."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import codedoc.core.execution as execution
from codedoc.utils.errors import InsufficientSourceError
import json
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ParseError
from tests.support.execution_requests import make_execution_request

pytestmark = pytest.mark.future_split_execution

@pytest.mark.parametrize("content", ["", " \n\t"])
def test_process_one_file_skips_before_orchestrator(
    tmp_path, monkeypatch, content
):
    request = make_execution_request(tmp_path, "empty.py", content)
    calls = {"process": 0}

    class FakeOrchestrator:
        llm = SimpleNamespace(provider_name="fake")

        def process(self, request):
            calls["process"] += 1
            return {}

    with pytest.raises(InsufficientSourceError):
        execution._process_one_file(request, FakeOrchestrator())
    assert calls == {"process": 0}

def test_process_one_file_allows_minimal_non_whitespace_source(tmp_path, monkeypatch):
    request = make_execution_request(tmp_path, "minimal.py", "x")
    calls = {"process": 0}

    class FakeOrchestrator:
        llm = SimpleNamespace(provider_name="fake")

        def process(self, request):
            calls["process"] += 1
            return {"description": "minimal"}

    result = execution._process_one_file(request, FakeOrchestrator())
    assert result == {"description": "minimal"}
    assert calls == {"process": 1}

class _DocFake:
    provider_name = "fake"

    def __init__(self):
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        self.doc_calls += 1
        if "Analyse the imports" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "dependency_refs": [],
                        "catalog_updates": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        return json.dumps(
            {
                "description": "Documented file.",
                "role_in_system": "test fixture",
                "functions": [],
                "classes": [],
                "exports": [],
                "key_concepts": [],
                "usage_example": "",
            }
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

def _run(monkeypatch, root, config, fake=None):
    fake = fake or _DocFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    return run_pipeline(root, config), fake

def _document(root):
    return json.loads((root / "codedoc" / "codedoc.json").read_text(encoding="utf-8"))

@pytest.mark.parametrize("analysis_mode", ["single", "triple"])
def test_empty_file_skips_without_agent_calls_and_reconciles_last_run(
    tmp_path, monkeypatch, caplog, analysis_mode
):
    (tmp_path / "empty.py").write_text("\ufeff \n\t", encoding="utf-8")

    stats, fake = _run(
        monkeypatch,
        tmp_path,
        {
            "entry_file": "empty.py",
            "analysis_mode": analysis_mode,
            "max_parallel_files": 1,
        },
    )

    assert fake.doc_calls == 0
    assert stats["checked"] == 0
    assert stats["failed"] == 0
    assert stats["skipped_insufficient_source"] == 1
    assert stats["documentation_calls_attempted"] == 0
    assert stats["documentation_calls_planned"] == 0
    assert stats["total_calls_planned"] == 0
    assert stats["attempted_logical_calls"] == 0
    assert stats["planned_calls_not_attempted"] == 0
    assert stats["unattempted_files"] == 0
    assert "insufficient source: empty_or_whitespace_only" in caplog.text

    document = _document(tmp_path)
    assert document.get("files", []) == []
    last_run = document["last_run"]
    assert last_run["files_skipped_insufficient_source"] == 1
    assert last_run["files_failed"] == 0
    assert last_run["files_unattempted"] == 0
    assert last_run["files_selected"] == sum(
        last_run[key]
        for key in (
            "files_documented_by_llm",
            "files_failed",
            "files_reused_unchanged",
            "files_reused_identical_content",
            "files_unattempted",
            "files_skipped_insufficient_source",
        )
    )


def test_oversized_whitespace_split_is_skipped_in_dry_run_and_real(
    tmp_path, monkeypatch
):
    (tmp_path / "empty.py").write_text(" " * 300_000, encoding="utf-8")
    config = {
        "entry_file": "empty.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_parallel_files": 1,
    }

    dry_stats = run_pipeline(tmp_path, {**config, "dry_run": True})

    assert dry_stats["would_skip_insufficient_source"] == 1
    assert dry_stats["would_call_llm_for"] == 0
    assert dry_stats["split_blocked_files"] == 0
    assert dry_stats["split_blocked_by_reason"] == {}

    stats, fake = _run(monkeypatch, tmp_path, config)

    assert fake.doc_calls == 0
    assert stats["skipped_insufficient_source"] == 1
    assert stats["split_blocked_files"] == 0
    assert stats["split_blocked_by_reason"] == {}
    assert _document(tmp_path).get("files", []) == []


def test_parallel_batch_routes_empty_and_normal_files_independently(
    tmp_path, monkeypatch
):
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    (tmp_path / "normal.py").write_text("VALUE = 1\n", encoding="utf-8")

    stats, fake = _run(
        monkeypatch,
        tmp_path,
        {
            "documentation_scope": "all",
            "max_parallel_files": 2,
            "analysis_mode": "single",
        },
    )

    assert fake.doc_calls == 1
    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert stats["skipped_insufficient_source"] == 1
    assert {record["path"] for record in _document(tmp_path)["files"]} == {
        "normal.py"
    }

def test_mixed_batch_counts_skip_success_and_failure_independently(
    tmp_path, monkeypatch
):
    for name in ("empty.py", "normal.py", "failing.py"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")

    def process_one(request, _orchestrator):
        rel_path = request.rel_path
        if rel_path == "empty.py":
            raise InsufficientSourceError(rel_path, "empty_or_whitespace_only")
        if rel_path == "failing.py":
            raise ParseError(rel_path, "fixture parse failure")
        return {
            "file_path": rel_path,
            "language": "python",
            "description": "Documented file.",
        }

    monkeypatch.setattr(execution, "_process_one_file", process_one)
    stats, fake = _run(
        monkeypatch,
        tmp_path,
        {
            "documentation_scope": "all",
            "max_parallel_files": 3,
            "file_retry_attempts": 0,
        },
    )

    assert fake.doc_calls == 0
    assert stats["checked"] == 1
    assert stats["failed"] == 1
    assert stats["skipped_insufficient_source"] == 1
    assert stats["unattempted_files"] == 0
    assert {record["path"] for record in _document(tmp_path)["files"]} == {
        "normal.py"
    }

def test_partition_reconciles_when_a_run_stops_before_every_file_is_attempted(
    tmp_path, monkeypatch
):
    """A provider-free rejection is already outside ``agent_rels``, so it must
    not be subtracted from that population again when deriving the unattempted
    count.  The consecutive-failure health check stops the run with files still
    unattempted; the selected-file completion partition must still be exact."""
    (tmp_path / "empty.py").write_text("   \n\t\n", encoding="utf-8")
    for index in range(6):
        (tmp_path / f"f{index}.py").write_text(f"VALUE_{index} = {index}\n", encoding="utf-8")

    class _AlwaysMalformed:
        provider_name = "fake"
        doc_calls = 0

        def complete_json(self, prompt, system=""):
            type(self).doc_calls += 1
            return "not json at all"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    stats, _fake = _run(
        monkeypatch,
        tmp_path,
        {
            "documentation_scope": "all",
            "propagate_changes": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 0,
            "max_consecutive_failures": 2,
            "allow_partial": True,
        },
        fake=_AlwaysMalformed(),
    )

    assert stats["skipped_insufficient_source"] == 1
    assert stats["planned_files"] == 6
    assert stats["checked"] == 0
    assert stats["failed"] == 2
    # Four provider-bound files were never attempted; the whitespace file was
    # never a provider-bound file at all.
    assert stats["unattempted_files"] == 4

    last_run = _document(tmp_path)["last_run"]
    assert last_run["files_unattempted"] == 4
    assert last_run["files_skipped_insufficient_source"] == 1
    assert last_run["files_selected"] == 7
    assert last_run["files_selected"] == sum(
        last_run[key]
        for key in (
            "files_documented_by_llm",
            "files_failed",
            "files_reused_unchanged",
            "files_reused_identical_content",
            "files_unattempted",
            "files_skipped_insufficient_source",
        )
    )

def test_stale_documentation_is_removed_when_source_becomes_whitespace(
    tmp_path, monkeypatch
):
    source = tmp_path / "main.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")

    first, first_fake = _run(
        monkeypatch,
        tmp_path,
        {"entry_file": "main.py", "max_parallel_files": 1},
    )
    assert first["checked"] == 1
    assert first_fake.doc_calls == 1
    assert [record["path"] for record in _document(tmp_path)["files"]] == ["main.py"]

    source.write_text(" \n\t", encoding="utf-8")
    second, second_fake = _run(
        monkeypatch,
        tmp_path,
        {"entry_file": "main.py", "max_parallel_files": 1},
    )
    assert second_fake.doc_calls == 0
    assert second["skipped_insufficient_source"] == 1
    assert _document(tmp_path).get("files", []) == []
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()


def test_oversized_whitespace_split_removes_stale_documentation(
    tmp_path, monkeypatch
):
    source = tmp_path / "main.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    config = {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "max_parallel_files": 1,
    }

    first, first_fake = _run(monkeypatch, tmp_path, config)

    assert first["checked"] == 1
    assert first_fake.doc_calls == 1
    assert [record["path"] for record in _document(tmp_path)["files"]] == ["main.py"]

    source.write_text(" " * 300_000, encoding="utf-8")
    second, second_fake = _run(monkeypatch, tmp_path, config)

    assert second_fake.doc_calls == 0
    assert second["skipped_insufficient_source"] == 1
    assert second["split_blocked_files"] == 0
    assert _document(tmp_path).get("files", []) == []
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()
