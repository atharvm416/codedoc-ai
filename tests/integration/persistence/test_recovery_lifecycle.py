"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.record_metadata_cases import private_key  # noqa: F401, F811

import json
from pathlib import Path
import pytest
from codedoc.core.record_meta import ANALYSIS_REVISION
from tests.support.recovery_rate_limit_runs import _make_fake_provider
from tests.support.recovery_rate_limit_runs import _patch_provider
from tests.support.recovery_rate_limit_runs import _run as recovery_rate_limit_run
from codedoc.agents.response_cleaning import clean_combined_response
from codedoc.core.document import read_codedoc_document
from codedoc.core.safe_writer import SafeWriter, _CRASH_SAFETY_BANNER
from codedoc.utils.errors import ConfigError
from tests.support.versionless_documents import _assert_versionless
from codedoc.core.document import records_by_path
from codedoc.core.resume import (
    RECOVERY_FILENAME,
    RecoveryState,
    _recovery_remedy,
    build_recovery_identity,
)
from codedoc.pipeline import run_pipeline
from tests.support.providers import SmartFake
from tests.support.cross_format_runs import _config
from tests.support.cross_format_runs import _first_run
from tests.support.cross_format_runs import _forbid_provider
from tests.support.cross_format_runs import _write_compatible_md_recovery
from tests.support.recovery_runs import _fake_provider
from tests.support.recovery_runs import _run as recovery_output_run
from tests.support.recovery_runs import _write_codedoc_json
from tests.support.recovery_cache_cases import _Fake
from codedoc.core.resume import (
    load_recovery_records_if_compatible,
)
from tests.support.execution_requests import make_execution_request


def test_crash_recovery_banner_explains_reuse_and_recovery_boundary():
    assert "compatible completed ordinary and split records may be reused" in _CRASH_SAFETY_BANNER
    assert "compatible current schema-3 split node checkpoints may resume" in _CRASH_SAFETY_BANNER
    assert "deliberately re-documented" not in _CRASH_SAFETY_BANNER
    assert "re-run the same command to resume" not in _CRASH_SAFETY_BANNER


def test_generic_recovery_remedy_explains_reuse_and_recovery_boundary(tmp_path):
    message = _recovery_remedy(tmp_path / RECOVERY_FILENAME)

    assert "completed ordinary and split records may then be reused" in message
    assert "compatible current schema-3 split node checkpoints may resume" in message
    assert "deliberately re-documented" not in message
    assert "resume that work" not in message


class _RateLimitOrch:
    class _LLM:
        provider_name = "fake"
    llm = _LLM()

    def process(self, request):
        raise RuntimeError("429 rate limit exceeded")

def test_A4_recorded_this_run_recovers_real_record(tmp_path):
    """A4: a file actually recorded THIS run whose future surfaces a rate-limit
    error is treated as done and recovers the REAL record (not {})."""
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.core.queue import ProcessingQueue
    from codedoc.utils.errors import ErrorReporter

    src = tmp_path / "x.py"
    src.write_text("import os\n", encoding="utf-8")
    descriptor = {"rel_path": "x.py", "path": src, "language": "python", "extension": ".py"}
    request = make_execution_request(tmp_path, "x.py", "import os\n", write=False)

    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "x.py", {"x.py": descriptor})
    # A worker recorded it during THIS run.
    recorder.record("x.py", {"description": "Fresh desc", "language": "python",
                             "role_in_system": "role"}, "hash123")
    assert recorder.recorded_this_run("x.py")

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0}
    reporter = ErrorReporter()

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [request], _RateLimitOrch(), queue, stats, reporter,
        max_workers=2, recorder=recorder, profile=None,
        split_execution_mode="recovery",
    )

    assert succeeded.get("x.py", {}).get("description") == "Fresh desc"
    assert retry_rate_limited == []

def test_A4_preloaded_stale_record_is_retried_not_restored(tmp_path):
    """A4 (root cause): a changed file whose only record was PRELOADED from a
    prior run (stale) and which then rate-limits must be RETRIED, never silently
    restored from the old documentation or counted as checked."""
    import json as _json
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.core.queue import ProcessingQueue
    from codedoc.utils.errors import ErrorReporter

    src = tmp_path / "x.py"
    src.write_text("import os  # changed\n", encoding="utf-8")
    descriptor = {"rel_path": "x.py", "path": src, "language": "python", "extension": ".py"}
    request = make_execution_request(tmp_path, "x.py", "import os  # changed\n", write=False)

    backup = tmp_path / "codedoc.json"
    backup.write_text(_json.dumps({
        "_codedoc": {"status": "in_progress", "schema_version": "1.4"},
        "files": [{"path": "x.py", "language": "python",
                   "description": "STALE", "hash": "old"}],
    }), encoding="utf-8")

    recorder = SafeWriter(backup, "json", "x.py", {"x.py": descriptor})
    recorder.load()  # preloads the stale record but NOT as recorded-this-run
    assert recorder.has_record("x.py")
    assert not recorder.recorded_this_run("x.py")

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0}
    reporter = ErrorReporter()

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [request], _RateLimitOrch(), queue, stats, reporter,
        max_workers=2, recorder=recorder, profile=None,
        split_execution_mode="recovery",
    )

    retried_paths = [r.rel_path for r, _exc in retry_rate_limited]
    assert "x.py" in retried_paths, "stale preloaded file must be retried"
    assert "x.py" not in succeeded
    assert stats["checked"] == 0

def test_1_live_backup_created_by_default(tmp_path, monkeypatch):
    """Test 1: running without --safe-mode creates codedoc/codedoc.json."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = recovery_rate_limit_run(tmp_path, monkeypatch, entry_file="main.py")

    backup = tmp_path / "codedoc" / "codedoc.json"
    assert backup.exists(), "live backup must be created"
    data = json.loads(backup.read_text(encoding="utf-8"))
    # Clean run: no crash banner
    assert "_crash_safety" not in data
    # Clean run: no in_progress status
    assert data.get("_codedoc", {}).get("status") != "in_progress"
    assert len(data.get("files", [])) == 1
    assert stats["checked"] == 1

def test_1b_initialize_empty_creates_banner_before_ai(tmp_path):
    """Test 1b: SafeWriter.initialize_empty() creates the crash-safety banner before any AI call."""
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc" / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.set_queue_order(["main.py"])

    # Before initialize_empty: file does not exist
    assert not backup.exists(), "backup must not exist before initialize_empty"

    sw.initialize_empty()

    # After initialize_empty: file exists with crash banner and empty files
    assert backup.exists(), "backup must exist after initialize_empty"
    data = json.loads(backup.read_text(encoding="utf-8"))
    assert data.get("_crash_safety"), "crash banner must be present after initialize_empty"
    assert data.get("_codedoc", {}).get("status") == "in_progress"
    assert data.get("files") == [], "files must be empty before any record() call"

def test_2_md_run_backup_removed_on_clean_success(tmp_path, monkeypatch):
    """Test 2: --format md → codedoc.md created, codedoc.json removed on clean finish."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = recovery_rate_limit_run(tmp_path, monkeypatch, entry_file="main.py", output_format="md")

    md = tmp_path / "codedoc" / "codedoc.md"
    json_backup = tmp_path / "codedoc" / "codedoc.json"
    assert md.exists(), "codedoc.md must exist"
    assert not json_backup.exists(), "live JSON backup must be removed after clean MD write"
    assert "codedoc.md" in stats.get("output_files", [str(tmp_path / "codedoc" / "codedoc.md")])[0]

def test_2b_md_backup_survives_interrupted_run(tmp_path):
    """Test 2b: an interrupted MD run leaves crash_recovery.json with the banner."""
    from codedoc.core.resume import RECOVERY_FILENAME
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    backup_path = out / RECOVERY_FILENAME

    sw = SafeWriter(backup_path, "md", "main.py", {})
    sw.set_queue_order(["main.py", "utils.py"])
    sw.initialize_empty()
    sw.record("main.py", {"language": "python"}, file_hash="H1")

    # "Interrupt" — do NOT call delete()
    assert backup_path.exists()
    data = json.loads(backup_path.read_text(encoding="utf-8"))
    assert data.get("_crash_safety"), "crash banner must be present after interrupt"
    assert data["_codedoc"]["status"] == "in_progress"
    assert len(data["files"]) == 1

def test_3b_named_md_sibling_persists_on_interrupt(tmp_path):
    """Test 3b: interrupted named-MD run keeps the single fixed recovery file with
    the crash banner (``crash_recovery.json`` in the output directory)."""
    from codedoc.core.resume import RECOVERY_FILENAME
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "docs"
    backup_path = out / RECOVERY_FILENAME
    assert backup_path.name == "crash_recovery.json"

    sw = SafeWriter(backup_path, "md", "main.py", {})
    sw.initialize_empty()
    sw.record("main.py", {"language": "python"}, file_hash="H1")

    assert backup_path.exists()
    data = json.loads(backup_path.read_text(encoding="utf-8"))
    assert "_crash_safety" in data
    assert data["_codedoc"]["status"] == "in_progress"

def test_14_completed_default_run_removes_recovery(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x=1\n")
    stats = recovery_rate_limit_run(tmp_path, monkeypatch, entry_file="main.py")

    assert stats["live_backup_path"] is None
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()

def test_14_completed_named_md_run_removes_recovery(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x=1\n")
    stats = recovery_rate_limit_run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        output_dir="docs/report.md",
    )

    assert stats["live_backup_path"] is None
    assert not (tmp_path / "docs" / "crash_recovery.json").exists()

def test_16_no_supported_files_no_backup(tmp_path, monkeypatch):
    """Test 16: with no supported files and NO explicit entry, the run returns
    early and no JSON is created.

    (The explicit-entry-with-zero-files case is now a hard ConfigError — A2 —
    covered by test_A2_explicit_entry_with_zero_scanned_files_raises.)
    """
    # Write only an unsupported file
    (tmp_path / "README.txt").write_text("hello\n")

    _patch_provider(monkeypatch, _make_fake_provider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "supported_extensions": [".py"],
    })

    assert stats["checked"] == 0
    assert stats.get("live_backup_path") is None
    # No JSON backup created
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()

@pytest.mark.parametrize("mode", ["single", "triple"])
def test_matching_live_recovery_is_reused_in_both_modes(tmp_path, monkeypatch, mode):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.resume import RECOVERY_FILENAME, build_recovery_identity
    from codedoc.pipeline import run_pipeline

    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    output = tmp_path / "codedoc"
    output.mkdir()
    # A compatible in-progress recovery file uses the single fixed name and carries
    # a versioned recovery identity that matches this run.
    identity = build_recovery_identity(
        project_root=tmp_path,
        json_target=output / "codedoc.json",
        md_target=None,
        entry_file="main.py",
        documentation_scope="entry",
        analysis_mode=mode,
        analysis_revision=ANALYSIS_REVISION,
    )
    (output / RECOVERY_FILENAME).write_text(
        json.dumps(
            {
                "_crash_safety": "INCOMPLETE RUN - test",
                "_codedoc": {
                    "entry_file": "main.py",
                    "schema_version": "1.4",
                    "status": "in_progress",
                    "recovery_identity": identity,
                },
                "schema_version": "1.4",
                "files": [
                    {
                        "path": "main.py",
                        "hash": compute_file_hash(target),
                        "language": "python",
                        "description": "Recovered.",
                        "_analysis_revision": ANALYSIS_REVISION,
                        "_analysis_mode": mode,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: pytest.fail("matching recovery created a provider"),
    )
    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "analysis_mode": mode,
            "propagate_changes": False,
        },
    )
    assert stats["checked"] == 0
    assert stats["skipped"] == 1
    final = json.loads((output / "codedoc.json").read_text(encoding="utf-8"))
    assert final["files"][0]["description"] == "Recovered."
    assert final["files"][0]["_analysis_mode"] == mode

def test_live_backup_uses_created_and_updated_at(tmp_path):
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.set_queue_order([])
    sw.initialize_empty()

    meta = json.loads(backup.read_text(encoding="utf-8"))["_codedoc"]
    assert "created_at" in meta
    assert "updated_at" in meta
    assert "generated_at" not in meta

def test_live_backup_load_preserves_legacy_creation_time(tmp_path):
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    legacy_created = "2026-01-01T00:00:00+00:00"
    backup.write_text(
        json.dumps({
            "_crash_safety": "INCOMPLETE",
            "_codedoc": {
                "entry_file": "main.py",
                "schema_version": "1.4",
                "generated_at": legacy_created,
                "status": "in_progress",
            },
            "files": [],
        }),
        encoding="utf-8",
    )
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.load()
    sw.initialize_empty()
    meta = json.loads(backup.read_text(encoding="utf-8"))["_codedoc"]
    assert meta["created_at"] == legacy_created
    assert "generated_at" not in meta

def test_recovery_and_cleaned_provider_response_are_versionless(tmp_path):
    recovery = tmp_path / "crash_recovery.json"
    writer = SafeWriter(recovery, "json", "main.py", {})
    writer.initialize_empty()
    recovery_data = json.loads(recovery.read_text(encoding="utf-8"))
    _assert_versionless(recovery_data)
    assert read_codedoc_document(recovery).in_progress is True

    cleaned = clean_combined_response(
        {"description": "Useful.", "schema_version": "unsolicited"}, "main.py"
    )
    assert cleaned == {"description": "Useful."}

def test_resume_reconstruction_preserves_private_key(private_key):
    from codedoc.pipeline import _public_record_to_doc

    public_record = {"path": "main.py", "language": "python", "_secret": "v"}
    doc = _public_record_to_doc(public_record)
    assert doc["_secret"] == "v"

def test_live_backup_preserves_private_key(tmp_path, private_key):
    from codedoc.core.safe_writer import SafeWriter

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {"main.py": {"path": tmp_path / "main.py"}})
    sw.set_queue_order(["main.py"])
    sw.record("main.py", {"language": "python", "description": "d", "_secret": "v"}, "h")

    data = json.loads(backup.read_text(encoding="utf-8"))
    assert data["files"][0]["_secret"] == "v"

def test_compatible_recovery_overlays_fallback_and_is_deleted_after_conversion(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    recovery = _write_compatible_md_recovery(tmp_path, "newer recovery record")
    _forbid_provider(monkeypatch)

    stats = run_pipeline(tmp_path, _config("md"))
    final = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert stats["checked"] == 0
    assert stats["resumed"] == 1
    assert final["description"] == "newer recovery record"
    assert not recovery.exists()

def test_incompatible_recovery_blocks_even_with_complete_fallback(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    recovery = _write_compatible_md_recovery(tmp_path, "newer recovery record")
    raw = recovery.read_text(encoding="utf-8")
    recovery.write_text(
        raw.replace('"analysis_mode": "single"', '"analysis_mode": "triple"'),
        encoding="utf-8",
    )
    _forbid_provider(monkeypatch)

    with pytest.raises(ConfigError, match="analysis_mode"):
        run_pipeline(tmp_path, _config("md"))
    assert recovery.exists()
    assert not (tmp_path / "docs" / "codedoc.md").exists()

def test_json_run_writes_recovery_file_and_never_mutates_stable_json(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    stable = out / "codedoc.json"
    recovery = out / "crash_recovery.json"

    # A pre-existing clean stable output (for an unrelated file) must be left
    # byte-identical until the run completes cleanly.
    _write_codedoc_json(stable, [{"path": "old.py", "hash": "OLD", "language": "python"}])
    stable_before = stable.read_bytes()

    seen = {"stable_untouched_midrun": None, "recovery_existed_midrun": False}

    def observe():
        seen["stable_untouched_midrun"] = stable.read_bytes() == stable_before
        if recovery.exists():
            seen["recovery_existed_midrun"] = True

    recovery_output_run(tmp_path, monkeypatch, provider=_fake_provider(observe), entry_file="main.py")

    # Mid-run: the in-progress records went to the recovery file and the stable
    # output was untouched.
    assert seen["recovery_existed_midrun"] is True
    assert seen["stable_untouched_midrun"] is True

    # After clean completion: stable output written, recovery file removed.
    assert stable.exists()
    assert stable.read_bytes() != stable_before  # now finalized with main.py
    data = json.loads(stable.read_text(encoding="utf-8"))
    assert "_crash_safety" not in data
    assert data.get("_codedoc", {}).get("status") != "in_progress"
    assert not recovery.exists()

def test_named_json_output_uses_fixed_recovery_name(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    recovery = tmp_path / "docs" / "crash_recovery.json"
    stable = tmp_path / "docs" / "report.json"

    seen = {"recovery_midrun": False}

    def observe():
        if recovery.exists():
            seen["recovery_midrun"] = True

    recovery_output_run(
        tmp_path, monkeypatch, provider=_fake_provider(observe),
        entry_file="main.py", output_dir="docs/report.json",
    )

    assert seen["recovery_midrun"] is True
    assert stable.exists()
    assert not recovery.exists()

def test_both_run_preserves_both_stable_artifacts_until_completion(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    stable_json = out / "codedoc.json"
    stable_md = out / "codedoc.md"
    recovery = out / "crash_recovery.json"

    seen = {"json_absent_midrun": None, "md_absent_midrun": None, "recovery_midrun": False}

    def observe():
        seen["json_absent_midrun"] = not stable_json.exists()
        seen["md_absent_midrun"] = not stable_md.exists()
        if recovery.exists():
            seen["recovery_midrun"] = True

    recovery_output_run(
        tmp_path, monkeypatch, provider=_fake_provider(observe),
        entry_file="main.py", output_format="both",
    )

    # Mid-run neither stable artifact existed; the recovery file did.
    assert seen["recovery_midrun"] is True
    assert seen["json_absent_midrun"] is True
    assert seen["md_absent_midrun"] is True

    # After completion both stable artifacts exist and the recovery file is gone.
    assert stable_json.exists()
    assert stable_md.exists()
    assert not recovery.exists()

def test_md_run_md_run_uses_fixed_recovery_name(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    recovery = tmp_path / "docs" / "crash_recovery.json"
    stable_md = tmp_path / "docs" / "report.md"

    seen = {"recovery_midrun": False}

    def observe():
        if recovery.exists():
            seen["recovery_midrun"] = True

    recovery_output_run(
        tmp_path, monkeypatch, provider=_fake_provider(observe),
        entry_file="main.py", output_dir="docs/report.md",
    )

    assert seen["recovery_midrun"] is True
    assert stable_md.exists()
    assert not recovery.exists()
    # No stray JSON sibling left behind.
    assert not (tmp_path / "docs" / "report.json").exists()

def test_foreign_file_at_recovery_name_blocks(tmp_path, monkeypatch):
    """A foreign file at the exact recovery name blocks the run (no candidate walk)."""
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    recovery = out / "crash_recovery.json"
    foreign_bytes = b"not a codedoc file"
    recovery.write_bytes(foreign_bytes)

    import codedoc.pipeline as pipeline_mod
    monkeypatch.setattr(
        pipeline_mod, "create_provider",
        lambda _cfg: pytest.fail("provider must not be created before the block"),
    )
    from codedoc.pipeline import run_pipeline
    with pytest.raises(ConfigError) as excinfo:
        run_pipeline(tmp_path, {"parallel_agents": False, "propagate_changes": False})

    assert "crash_recovery.json" in str(excinfo.value)
    assert recovery.read_bytes() == foreign_bytes  # never renamed or deleted

def test_crash_recovery_stores_no_correction_internal(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    recovery_seen = {}

    class Recorder(_Fake):
        def complete_json(self, prompt, system=""):
            rec = tmp_path / "codedoc" / "crash_recovery.json"
            if rec.exists():
                recovery_seen["text"] = rec.read_text(encoding="utf-8")
            return super().complete_json(prompt, system)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: Recorder())
    pipeline.run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "response_correction_enabled": True,
         "propagate_changes": False},
    )
    if "text" in recovery_seen:
        assert "response_contract_final" not in recovery_seen["text"]
        assert "response_contract_diagnostic" not in recovery_seen["text"]

def _identity(tmp_path: Path, **changes):
    values = {
        "project_root": tmp_path,
        "json_target": tmp_path / "docs" / "codedoc.json",
        "md_target": None,
        "entry_file": "main.py",
        "documentation_scope": "entry",
        "analysis_mode": "single",
        "analysis_revision": "file-doc-v3",
    }
    values.update(changes)
    return build_recovery_identity(**values)

def test_fixed_recovery_name_and_identity_mismatch_blocks(tmp_path):
    path = tmp_path / "docs" / RECOVERY_FILENAME
    identity = _identity(tmp_path)
    writer = SafeWriter(path, "json", "main.py", {}, identity)
    writer.initialize_empty()
    assert path.name == "crash_recovery.json"
    assert load_recovery_records_if_compatible(path, identity) == RecoveryState()
    with pytest.raises(ConfigError, match="analysis_mode expected.*triple.*single"):
        load_recovery_records_if_compatible(
            path, _identity(tmp_path, analysis_mode="triple")
        )
    assert path.exists()

def test_completed_run_uses_only_fixed_recovery_filename(tmp_path, monkeypatch):
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: SmartFake())
    run_pipeline(tmp_path, {"entry_file": "main.py", "propagate_changes": False})
    output = tmp_path / "codedoc"
    assert (output / "codedoc.json").exists()
    assert not (output / RECOVERY_FILENAME).exists()
    assert not list(output.glob("crash_recovery_*.json"))
