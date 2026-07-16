"""0.12.2 end-to-end insufficient-source routing and accounting."""

import json
from pathlib import Path

import pytest

import codedoc.core.execution as execution
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import InsufficientSourceError, ParseError


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

    def process_one(descriptor, _orchestrator):
        rel_path = descriptor["rel_path"]
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


@pytest.mark.parametrize(
    ("analysis_mode", "per_file"),
    [("single", 1), ("triple", 3)],
)
def test_dry_run_subtracts_empty_file_calls_but_preserves_candidate_cap(
    tmp_path, analysis_mode, per_file
):
    (tmp_path / "empty.py").write_text("\n", encoding="utf-8")
    (tmp_path / "normal.py").write_text("VALUE = 1\n", encoding="utf-8")

    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "documentation_scope": "all",
            "analysis_mode": analysis_mode,
            "max_files": 1,
        },
    )

    assert stats["would_skip_insufficient_source"] == 1
    assert stats["would_call_llm_for"] == 1
    assert stats["estimated_calls"] == per_file
    assert stats["documentation_calls_planned"] == per_file
    assert stats["response_correction_calls_possible_max"] == 0
    assert stats["max_files_candidate_files"] == 2
    assert stats["max_files_exceeded"] is True

    (tmp_path / "empty.py").write_text("OTHER = 2\n", encoding="utf-8")
    without_skip = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "documentation_scope": "all",
            "analysis_mode": analysis_mode,
        },
    )
    assert without_skip["estimated_input_tokens"] > stats["estimated_input_tokens"]


def test_dry_run_read_failure_remains_a_documentation_candidate(
    tmp_path, monkeypatch
):
    bad = tmp_path / "bad.py"
    bad.write_text("VALUE = 1\n", encoding="utf-8")
    original = Path.read_text

    def failing_read(self, *args, **kwargs):
        if self == bad:
            raise OSError("fixture read failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", failing_read)
    stats = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "bad.py",
        },
    )

    assert stats["would_skip_insufficient_source"] == 0
    assert stats["would_call_llm_for"] == 1
    assert stats["estimated_calls"] == 1
