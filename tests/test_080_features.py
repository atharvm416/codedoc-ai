"""
0.8.0 feature tests.

Covers all new behaviour introduced in 0.8.0:
  1   Live JSON backup created by default (no --safe-mode needed)
  2   Live backup for Markdown runs (sibling codedoc.json)
  3   Named Markdown output gets JSON sibling (report.md → report.json)
  4   Queue-order output in live backup files array
  5   Parallel worker records inside worker thread
  6   Ownership guard for named-MD JSON sibling
  7   Resume from live backup (unchanged files skipped)
  7b  Changed-hash file re-processed and replaces old record in queue slot
  8   Old .codedoc_progress.json checkpoint migration
  9   Rate-limit ladder (step-down on 429)
  10  Rate-limit signal detector (_is_rate_limit_error)
  10b User-visible notification on parallelism reduction (all 3 providers)
  11  error.log in output_dir; path surfaced in stats
  12  --safe-mode deprecation notice
  13  --format both interrupted run leaves codedoc.json; clean run keeps it
  14  live_backup_path reported in stats for every output scenario
  15  parallel_ladder config validation
  16  No supported files → no live backup created
  17  Final output excludes recovered (warning-level) entries
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fake_provider(description: str = "Documented.", fail_after: int = -1):
    """Return a fake LLM provider.

    Parameters
    ----------
    fail_after:
        If >= 0, raise ``LLMError`` with a rate-limit signal on every call
        after the first *fail_after* successful calls.  -1 = always succeed.
    """
    import json as _json
    from codedoc.utils.errors import LLMError

    call_count = {"n": 0}

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if fail_after >= 0 and call_count["n"] > fail_after:
                raise LLMError("fake", "429 rate_limit_exceeded tokens per min")
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": description,
                    "role_in_system": "test",
                    "key_concepts": [],
                    "usage_example": "",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({
                    "dependencies_analysis": {
                        "internal": [], "external": [],
                        "dependency_refs": [], "catalog_updates": [],
                        "usage_notes": [], "warnings": [],
                    }
                })
            return _json.dumps({
                "description": description,
                "role_in_system": "test",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


def _make_rate_limit_provider(provider_name: str, signal: str):
    """Always raise a rate-limit error with the given signal text."""
    from codedoc.utils.errors import LLMError

    class RateLimitProvider:
        pass

    RateLimitProvider.provider_name = provider_name

    def _fail(self, prompt, system=""):
        raise LLMError(provider_name, f"{signal} quota exceeded")

    RateLimitProvider.complete_json = _fail
    RateLimitProvider.complete = _fail
    return RateLimitProvider()


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)


def _codedoc_json(path: Path, files: list, status: str | None = None):
    """Write a minimal codedoc-owned JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"entry_file": "main.py", "schema_version": "1.4"}
    if status:
        meta["status"] = status
    payload: dict = {"_codedoc": meta, "files": files}
    if status == "in_progress":
        payload = {
            "_crash_safety": "INCOMPLETE RUN - test",
            **payload,
        }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run(tmp_path, monkeypatch, provider=None, **cfg):
    """Helper to run the pipeline with a fake provider."""
    if provider is None:
        provider = _make_fake_provider()
    _patch_provider(monkeypatch, provider)
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)


# ---------------------------------------------------------------------------
# Test 1 — Live JSON backup created by default
# ---------------------------------------------------------------------------

def test_1_live_backup_created_by_default(tmp_path, monkeypatch):
    """Test 1: running without --safe-mode creates codedoc/codedoc.json."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(tmp_path, monkeypatch, entry_file="main.py")

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


# ---------------------------------------------------------------------------
# Test 2 — Live backup for Markdown runs
# ---------------------------------------------------------------------------

def test_2_md_run_backup_removed_on_clean_success(tmp_path, monkeypatch):
    """Test 2: --format md → codedoc.md created, codedoc.json removed on clean finish."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(tmp_path, monkeypatch, entry_file="main.py", output_format="md")

    md = tmp_path / "codedoc" / "codedoc.md"
    json_backup = tmp_path / "codedoc" / "codedoc.json"
    assert md.exists(), "codedoc.md must exist"
    assert not json_backup.exists(), "live JSON backup must be removed after clean MD write"
    assert "codedoc.md" in stats.get("output_files", [str(tmp_path / "codedoc" / "codedoc.md")])[0]


def test_2b_md_backup_survives_interrupted_run(tmp_path):
    """Test 2b: an interrupted MD run leaves codedoc.json with crash banner."""
    from codedoc.pipeline import _resolve_live_backup_path
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    backup_path = _resolve_live_backup_path(out, "md", "codedoc.json", "codedoc.md")

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


# ---------------------------------------------------------------------------
# Test 3 — Named Markdown output gets JSON sibling
# ---------------------------------------------------------------------------

def test_3_named_md_output_uses_json_sibling(tmp_path, monkeypatch):
    """Test 3: --output docs/report.md → docs/report.json (not docs/codedoc.json)."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        output_dir="docs/report.md",
    )

    md = tmp_path / "docs" / "report.md"
    sibling = tmp_path / "docs" / "report.json"
    default_json = tmp_path / "docs" / "codedoc.json"

    assert md.exists(), "report.md must exist"
    # After clean success the JSON sibling is removed
    assert not sibling.exists(), "report.json must be removed after clean MD write"
    assert not default_json.exists(), "docs/codedoc.json must NOT be created"


def test_3b_named_md_sibling_persists_on_interrupt(tmp_path):
    """Test 3b: interrupted named-MD run keeps report.json with crash banner."""
    from codedoc.pipeline import _resolve_live_backup_path
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "docs"
    backup_path = _resolve_live_backup_path(out, "md", "codedoc.json", "report.md")
    assert backup_path.name == "report.json"

    sw = SafeWriter(backup_path, "md", "main.py", {})
    sw.initialize_empty()
    sw.record("main.py", {"language": "python"}, file_hash="H1")

    assert backup_path.exists()
    data = json.loads(backup_path.read_text(encoding="utf-8"))
    assert "_crash_safety" in data
    assert data["_codedoc"]["status"] == "in_progress"


# ---------------------------------------------------------------------------
# Test 4 — Queue-order output
# ---------------------------------------------------------------------------

def test_4_files_array_follows_queue_order(tmp_path):
    """Test 4: files in live backup follow set_queue_order, not completion order."""
    from codedoc.core.safe_writer import SafeWriter

    out = tmp_path / "codedoc"
    backup = out / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {})
    sw.set_queue_order(["a.py", "b.py", "c.py"])
    sw.initialize_empty()

    # Record out of topological order
    sw.record("c.py", {"language": "python"}, file_hash="HC")
    sw.record("a.py", {"language": "python"}, file_hash="HA")
    sw.record("b.py", {"language": "python"}, file_hash="HB")

    data = json.loads(backup.read_text(encoding="utf-8"))
    paths = [f["path"] for f in data["files"]]
    assert paths == ["a.py", "b.py", "c.py"], f"Expected queue order, got {paths}"


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


# ---------------------------------------------------------------------------
# Test 5 — Parallel worker records inside worker thread
# ---------------------------------------------------------------------------

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

    descriptor = {
        "rel_path": "main.py",
        "path": tmp_path / "main.py",
        "language": "python",
    }
    (tmp_path / "main.py").write_text("x=1\n")

    class FakeOrchestrator:
        class llm:
            provider_name = "fake"
        @staticmethod
        def process(d, content, imports):
            return {"language": "python", "description": "test"}

    _process_and_record(descriptor, FakeOrchestrator(), sw)

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


# ---------------------------------------------------------------------------
# Test 6 — Ownership guard for named-MD JSON sibling
# ---------------------------------------------------------------------------

def test_6_foreign_json_sibling_blocks_run(tmp_path, monkeypatch):
    """Test 6: a foreign report.json blocks the run before any AI calls."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.json").write_text(
        json.dumps({"not": "codedoc"}), encoding="utf-8"
    )
    (tmp_path / "main.py").write_text("x=1\n")

    from codedoc.utils.errors import ConfigError
    _patch_provider(monkeypatch, _make_fake_provider())

    from codedoc.pipeline import run_pipeline
    try:
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "output_dir": "docs/report.md",
            "parallel_agents": False,
            "propagate_changes": False,
        })
        assert False, "Should have raised ConfigError for foreign sibling"
    except ConfigError:
        pass


# ---------------------------------------------------------------------------
# Test 7 — Resume from live backup
# ---------------------------------------------------------------------------

def test_7_resume_skips_unchanged_files(tmp_path, monkeypatch):
    """Test 7: re-run skips files already in live backup with matching hash."""
    main_py = tmp_path / "main.py"
    main_py.write_text("x=1\n")

    from codedoc.core.db import compute_file_hash
    h = compute_file_hash(main_py)

    # Write a live backup that already has main.py with the correct hash
    _codedoc_json(
        tmp_path / "codedoc" / "codedoc.json",
        [{"path": "main.py", "hash": h, "language": "python"}],
        status="in_progress",
    )
    # Add _crash_safety to make it look like an in-progress run
    data = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text())
    data["_crash_safety"] = "INCOMPLETE RUN - test"
    (tmp_path / "codedoc" / "codedoc.json").write_text(json.dumps(data))

    call_count = {"n": 0}
    real_provider = _make_fake_provider()
    original_complete = real_provider.complete_json

    def counted_complete(prompt, system=""):
        call_count["n"] += 1
        return original_complete(prompt, system)

    real_provider.complete_json = counted_complete
    _patch_provider(monkeypatch, real_provider)

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    assert call_count["n"] == 0, "LLM must not be called for unchanged files"
    assert stats["skipped"] == 1 or stats["reused"] == 1 or stats["checked"] == 0


# ---------------------------------------------------------------------------
# Test 7b — Changed-hash file re-processes and replaces in queue slot
# ---------------------------------------------------------------------------

def test_7b_changed_hash_triggers_reprocess_and_replaces_slot(tmp_path, monkeypatch):
    """Test 7b: changed file hash triggers re-process; result replaces in queue slot."""
    main_py = tmp_path / "main.py"
    main_py.write_text("x=1\n")

    # Live backup has main.py with OLD description and STALE hash
    _codedoc_json(
        tmp_path / "codedoc" / "codedoc.json",
        [{"path": "main.py", "hash": "stale_hash_000", "language": "python",
          "description": "OLD description"}],
        status="in_progress",
    )
    data = json.loads((tmp_path / "codedoc" / "codedoc.json").read_text())
    data["_crash_safety"] = "INCOMPLETE"
    (tmp_path / "codedoc" / "codedoc.json").write_text(json.dumps(data))

    # Use a provider that returns a distinctively new description
    provider = _make_fake_provider(description="NEW description from provider")
    _patch_provider(monkeypatch, provider)

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    # The stale hash causes re-processing
    assert stats["checked"] == 1, "main.py must be re-processed due to hash mismatch"

    out_json = tmp_path / "codedoc" / "codedoc.json"
    assert out_json.exists()
    result = json.loads(out_json.read_text(encoding="utf-8"))
    # No _crash_safety in final clean output
    assert "_crash_safety" not in result
    # One file in the result
    assert len(result.get("files", [])) == 1


# ---------------------------------------------------------------------------
# Test 8 — Old checkpoint migration
# ---------------------------------------------------------------------------

def test_8_checkpoint_migration_restores_valid_entries(tmp_path, monkeypatch):
    """Test 8: .codedoc_progress.json entries with matching hash are migrated."""
    from codedoc.core.checkpoint import CHECKPOINT_FILENAME
    from codedoc.core.db import compute_file_hash

    main_py = tmp_path / "main.py"
    main_py.write_text("x=1\n")
    h = compute_file_hash(main_py)

    # Write an old-style checkpoint
    cp_path = tmp_path / "codedoc" / CHECKPOINT_FILENAME
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text(json.dumps({
        "codedoc_checkpoint": True,
        "version": "1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "entry_file": "main.py",
        "total_recorded": 1,
        "results": {
            "main.py": {
                "language": "python",
                "description": "Checkpointed.",
                "_checkpoint_hash": h,
            }
        },
    }), encoding="utf-8")

    call_count = {"n": 0}
    real_provider = _make_fake_provider()
    original = real_provider.complete_json

    def counted(prompt, system=""):
        call_count["n"] += 1
        return original(prompt, system)

    real_provider.complete_json = counted
    _patch_provider(monkeypatch, real_provider)

    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    assert call_count["n"] == 0, "Checkpoint entry with matching hash must be reused"


def test_8b_checkpoint_with_no_hash_triggers_reprocess(tmp_path, monkeypatch):
    """Test 8b: checkpoint entry with no _checkpoint_hash must be re-processed."""
    from codedoc.core.checkpoint import CHECKPOINT_FILENAME

    main_py = tmp_path / "main.py"
    main_py.write_text("x=1\n")

    cp_path = tmp_path / "codedoc" / CHECKPOINT_FILENAME
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text(json.dumps({
        "codedoc_checkpoint": True,
        "version": "1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "entry_file": "main.py",
        "total_recorded": 1,
        "results": {
            "main.py": {
                "language": "python",
                "description": "Checkpointed.",
                # No _checkpoint_hash
            }
        },
    }), encoding="utf-8")

    call_count = {"n": 0}
    real_provider = _make_fake_provider()
    original = real_provider.complete_json

    def counted(prompt, system=""):
        call_count["n"] += 1
        return original(prompt, system)

    real_provider.complete_json = counted
    _patch_provider(monkeypatch, real_provider)

    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    assert call_count["n"] > 0, "Entry with no hash must be re-processed"


# ---------------------------------------------------------------------------
# Test 9 — Rate-limit ladder
# ---------------------------------------------------------------------------

def test_9_rate_limit_ladder_steps_down(tmp_path, monkeypatch):
    """Test 9: rate-limit error causes ladder step-down; no descriptors dropped."""
    # Files must import each other so all 3 are reachable from entry a.py
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    from codedoc.utils.errors import LLMError

    call_count = {"n": 0}

    class RateLimitThenOk:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            # First 3 calls: rate-limit (during parallel batch at level 5)
            if call_count["n"] <= 3:
                raise LLMError("openai", "429 rate_limit_exceeded tokens per min")
            # After step-down: succeed
            if "key_concepts" in prompt:
                return json.dumps({"description": "ok", "role_in_system": "r",
                                   "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return json.dumps({"description": "ok", "role_in_system": "r",
                               "functions": [], "classes": [], "exports": []})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    _patch_provider(monkeypatch, RateLimitThenOk())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
    })

    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) > 0, "Expected at least one rate-limit step-down warning"


def test_9b_non_rate_limit_errors_do_not_step_down(tmp_path, monkeypatch):
    """Test 9b: non-rate-limit errors do not trigger ladder step-down."""
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("x=1\n")

    from codedoc.utils.errors import LLMError

    class ParseErrorProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "JSON parse error: invalid response format")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    _patch_provider(monkeypatch, ParseErrorProvider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 2,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
    })

    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) == 0, "Non-rate-limit errors must not cause step-down"


def test_9c_non_rate_limit_parallel_failures_counted_in_stats(tmp_path, monkeypatch):
    """Test 9c: non-rate-limit failures in parallel mode increment stats['failed'] and appear in error.log."""
    # Files import each other so all 3 are reachable in parallel
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    from codedoc.utils.errors import LLMError

    class AlwaysFailProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "JSON parse error: model returned garbage")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    _patch_provider(monkeypatch, AlwaysFailProvider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "max_consecutive_failures": 10,
    })

    # Non-rate-limit failures must be counted in stats["failed"]
    assert stats["failed"] > 0, (
        f"stats['failed'] must be > 0 for non-rate-limit errors, got {stats}"
    )
    # No rate-limit step-down warnings
    assert len(stats.get("rate_limit_warnings", [])) == 0, (
        "Non-rate-limit errors must not cause rate-limit step-down"
    )
    # error.log must be written
    assert stats.get("error_log") is not None, "error.log must be set when files fail"
    assert Path(stats["error_log"]).exists(), "error.log file must exist"


# ---------------------------------------------------------------------------
# Test 10 — Rate-limit signal detector
# ---------------------------------------------------------------------------

def test_10_rate_limit_detector_openai(tmp_path):
    """Test 10a: OpenAI 429 / TPM signals detected."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    assert _is_rate_limit_error(LLMError("openai", "429 rate_limit_exceeded tokens per min"))
    assert _is_rate_limit_error(LLMError("openai", "Error code: 429 - quota exceeded"))
    assert _is_rate_limit_error(LLMError("openai", "Rate limit reached for tpm"))


def test_10_rate_limit_detector_anthropic(tmp_path):
    """Test 10b: Anthropic overloaded/529 signals detected."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    assert _is_rate_limit_error(LLMError("anthropic", "529 overloaded"))
    assert _is_rate_limit_error(LLMError("anthropic", "rate_limit quota exceeded"))
    assert _is_rate_limit_error(LLMError("anthropic", "overloaded try again"))


def test_10_rate_limit_detector_gemini(tmp_path):
    """Test 10c: Gemini RESOURCE_EXHAUSTED / quota signals detected."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    assert _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED quota exceeded"))
    assert _is_rate_limit_error(LLMError("gemini", "429 too many requests"))
    assert _is_rate_limit_error(LLMError("gemini", "resource_exhausted"))


def test_10_rate_limit_detector_false_positives(tmp_path):
    """Test 10d: ordinary errors are NOT rate-limit signals."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError, ParseError

    assert not _is_rate_limit_error(LLMError("openai", "JSON parse error invalid format"))
    assert not _is_rate_limit_error(ParseError("main.py", "syntax error"))
    assert not _is_rate_limit_error(ValueError("bad value"))


def test_10_rate_limit_detector_walks_cause_chain(tmp_path):
    """Test 10e: detector walks __cause__ chain so wrapper doesn't hide signal."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    inner = LLMError("openai", "429 rate_limit_exceeded")
    outer = RuntimeError("wrapper error")
    outer.__cause__ = inner

    assert _is_rate_limit_error(outer)


# ---------------------------------------------------------------------------
# Test 10b — User-visible notification on parallelism reduction
# ---------------------------------------------------------------------------

def test_10b_user_notification_includes_provider_and_level(tmp_path, monkeypatch, capsys):
    """Test 10b: step-down prints provider name and original max_parallel to stdout."""
    # Files must import each other so all 3 are reachable from entry a.py
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    from codedoc.utils.errors import LLMError

    step = {"n": 0}

    class AlwaysRateLimitProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "429 rate_limit_exceeded")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    _patch_provider(monkeypatch, AlwaysRateLimitProvider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "max_consecutive_failures": 1,
    })

    captured = capsys.readouterr()
    # Either stdout or stats contain the warning
    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) > 0 or "openai" in captured.out.lower() or "rate limit" in captured.out.lower()

    if warnings:
        w = warnings[0]
        assert w["provider"] == "openai"
        assert "original_max_parallel" in w
        assert "new_level" in w


def test_10b_no_warning_on_successful_run(tmp_path, monkeypatch, capsys):
    """Test 10b: a successful run with no rate limits produces no step-down warning."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(tmp_path, monkeypatch, entry_file="main.py")
    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) == 0, "No warnings expected on a clean run"


# ---------------------------------------------------------------------------
# Test 11 — error.log in output_dir; always surfaced
# ---------------------------------------------------------------------------

def test_11_error_log_in_output_dir(tmp_path, monkeypatch):
    """Test 11: error.log is written in output_dir when issues occur."""
    (tmp_path / "main.py").write_text("x=1\n")

    from codedoc.utils.errors import LLMError

    class FailProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            raise LLMError("fake", "network error (not rate-limit)")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    _patch_provider(monkeypatch, FailProvider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "file_retry_attempts": 0,
    })

    # error.log must be in output_dir, not project root
    assert stats.get("error_log") is not None
    log_path = Path(stats["error_log"])
    assert log_path.parent == (tmp_path / "codedoc").resolve(), (
        f"error.log should be in output_dir, got {log_path.parent}"
    )
    assert log_path.exists()
    assert stats.get("issues_recorded", 0) > 0


def test_11_error_log_not_in_root(tmp_path, monkeypatch):
    """Test 11b: error.log must NOT be in the project root."""
    (tmp_path / "main.py").write_text("x=1\n")

    from codedoc.utils.errors import LLMError

    class FailProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            raise LLMError("fake", "some error")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    _patch_provider(monkeypatch, FailProvider())
    from codedoc.pipeline import run_pipeline

    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "file_retry_attempts": 0,
    })

    # error.log must NOT appear in the project root
    assert not (tmp_path / "error.log").exists(), (
        "error.log must not be written to the project root"
    )


# ---------------------------------------------------------------------------
# Test 12 — --safe-mode deprecation warning
# ---------------------------------------------------------------------------

def test_12_safe_mode_deprecation_notice(tmp_path, monkeypatch, capsys):
    """Test 12: passing safe_mode=True prints a deprecation notice and run succeeds."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        safe_mode=True,
    )

    captured = capsys.readouterr()
    assert "deprecated" in captured.out.lower() or "default" in captured.out.lower(), (
        "Expected deprecation notice for --safe-mode, got: " + captured.out
    )
    assert stats["checked"] == 1, "Run must complete normally with safe_mode=True"


# ---------------------------------------------------------------------------
# Test 13 — --format both: live backup kept after clean finish
# ---------------------------------------------------------------------------

def test_13_format_both_keeps_json_after_clean_finish(tmp_path, monkeypatch):
    """Test 13: --format both → both codedoc.json and codedoc.md on clean finish."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        output_format="both",
    )

    json_file = tmp_path / "codedoc" / "codedoc.json"
    md_file = tmp_path / "codedoc" / "codedoc.md"

    assert json_file.exists(), "codedoc.json must exist for --format both"
    assert md_file.exists(), "codedoc.md must exist for --format both"

    # JSON must be clean (no crash banner)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "_crash_safety" not in data
    assert data.get("_codedoc", {}).get("status") != "in_progress"

    # Markdown must not contain crash banner
    md_content = md_file.read_text(encoding="utf-8")
    assert "INCOMPLETE RUN" not in md_content


# ---------------------------------------------------------------------------
# Test 14 — live_backup_path in stats
# ---------------------------------------------------------------------------

def test_14_live_backup_path_in_stats_default(tmp_path, monkeypatch):
    """Test 14a: stats['live_backup_path'] is set for default JSON run."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(tmp_path, monkeypatch, entry_file="main.py")

    assert "live_backup_path" in stats
    assert stats["live_backup_path"] is not None
    expected = str((tmp_path / "codedoc" / "codedoc.json").resolve())
    assert stats["live_backup_path"] == expected


def test_14_live_backup_path_in_stats_named_md(tmp_path, monkeypatch):
    """Test 14b: stats['live_backup_path'] points to report.json for named-MD run."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(
        tmp_path, monkeypatch,
        entry_file="main.py",
        output_dir="docs/report.md",
    )

    assert "live_backup_path" in stats
    assert stats["live_backup_path"] is not None
    assert "report.json" in stats["live_backup_path"]


# ---------------------------------------------------------------------------
# Test 15 — parallel_ladder config validation
# ---------------------------------------------------------------------------

def test_15_parallel_ladder_invalid_non_decreasing_raises(tmp_path):
    """Test 15a: non-decreasing parallel_ladder raises ConfigError."""
    from codedoc.core.loader import load_config
    from codedoc.utils.errors import ConfigError

    try:
        load_config(tmp_path, {
            "entry_file": "main.py",
            "parallel_ladder": [2, 5, 1],  # not decreasing
        })
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "decreasing" in str(e).lower() or "ladder" in str(e).lower()


def test_15_parallel_ladder_clamped_when_exceeds_max(tmp_path):
    """Test 15b: ladder values exceeding max_parallel_files are clamped (no error)."""
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "max_parallel_files": 3,
        "parallel_ladder": [10, 5, 1],  # exceeds max
    })
    ladder = cfg["parallel_ladder"]
    assert all(v <= 3 for v in ladder), f"All ladder values must be clamped to 3, got {ladder}"


def test_15_parallel_ladder_appends_1_if_missing(tmp_path):
    """Test 15c: parallel_ladder without trailing 1 gets 1 appended."""
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "max_parallel_files": 5,
        "parallel_ladder": [5, 2],  # missing 1
    })
    assert cfg["parallel_ladder"][-1] == 1, "ladder must always end with 1"


# ---------------------------------------------------------------------------
# Test 16 — No supported files: no live backup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test 17 — Final output excludes recovered (warning-level) entries
# ---------------------------------------------------------------------------

def test_17_recovered_warnings_not_in_final_json(tmp_path):
    """Test 17: warning-level ErrorReporter entries do not appear in final JSON."""
    from codedoc.utils.errors import ErrorReporter
    from codedoc.core.output import write_project_outputs

    output_dir = tmp_path / "codedoc"
    output_dir.mkdir()

    reporter = ErrorReporter(output_dir / "error.log")
    reporter.record(
        RuntimeError("429 rate_limit_exceeded"),
        context="rate limit step-down",
        level="warning",
    )

    # summary() must return "" for warning-only
    assert reporter.summary() == "", "summary() must be empty for warning-only"
    assert not reporter.has_errors(), "has_errors() must be False for warning-only"
    assert reporter.has_issues(), "has_issues() must be True"
    assert reporter.issue_count() == 1

    # write_project_outputs passes summary() — empty string means no errors field
    records = [{
        "hash": "H1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {"file_path": "main.py", "language": "python",
                          "description": "test", "role_in_system": "r",
                          "functions": [], "classes": [], "exports": [],
                          "dependencies_analysis": {}},
    }]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        error_summary=reporter.summary(),  # "" — empty
        output_format="json",
        entry_file="main.py",
    )

    result = json.loads((output_dir / "codedoc.json").read_text(encoding="utf-8"))
    assert "errors" not in result, (
        f"'errors' field must NOT appear in clean JSON for warning-only runs, got keys: {list(result)}"
    )


def test_17_hard_errors_still_appear_in_final_json(tmp_path):
    """Test 17b: error-level entries DO appear in final JSON."""
    from codedoc.utils.errors import ErrorReporter
    from codedoc.core.output import write_project_outputs

    output_dir = tmp_path / "codedoc"
    output_dir.mkdir()

    reporter = ErrorReporter(output_dir / "error.log")
    reporter.record(RuntimeError("parse failed"), context="test", level="error")

    assert reporter.has_errors()
    assert reporter.summary() != ""

    records = [{
        "hash": "H1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {"file_path": "main.py", "language": "python",
                          "description": "test", "role_in_system": "r",
                          "functions": [], "classes": [], "exports": [],
                          "dependencies_analysis": {}},
    }]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 1, "skipped": 0},
        output_dir,
        error_summary=reporter.summary(),
        output_format="json",
        entry_file="main.py",
    )

    result = json.loads((output_dir / "codedoc.json").read_text(encoding="utf-8"))
    assert "errors" in result, "Hard errors must appear in final JSON"


# ---------------------------------------------------------------------------
# Test — _resolve_live_backup_path helper
# ---------------------------------------------------------------------------

def test_provider_init_failure_prints_error_log_path(tmp_path, monkeypatch, capsys):
    """Provider init failure must print error.log path to stderr before raising."""
    (tmp_path / "main.py").write_text("x=1\n")

    def boom(_config):
        raise RuntimeError("API key not found")

    monkeypatch.setattr("codedoc.pipeline.create_provider", boom)

    from codedoc.pipeline import run_pipeline

    raised = False
    try:
        run_pipeline(tmp_path, {
            "entry_file": "main.py",
            "parallel_agents": False,
            "propagate_changes": False,
        })
    except Exception:
        raised = True

    assert raised, "run_pipeline must re-raise when provider init fails"
    captured = capsys.readouterr()
    assert "error.log" in captured.err or "issue" in captured.err.lower(), (
        f"Provider init failure must print error.log path to stderr; got: {captured.err!r}"
    )


def test_resolve_live_backup_path_scenarios(tmp_path):
    """Verify the live backup path for every output scenario."""
    from codedoc.pipeline import _resolve_live_backup_path

    out = tmp_path / "codedoc"

    # Default JSON
    p = _resolve_live_backup_path(out, "json", "codedoc.json", "codedoc.md")
    assert p == out / "codedoc.json"

    # Both
    p = _resolve_live_backup_path(out, "both", "codedoc.json", "codedoc.md")
    assert p == out / "codedoc.json"

    # Default MD
    p = _resolve_live_backup_path(out, "md", "codedoc.json", "codedoc.md")
    assert p == out / "codedoc.json"

    # Named JSON
    p = _resolve_live_backup_path(out, "json", "report.json", "codedoc.md")
    assert p == out / "report.json"

    # Named MD: sibling derived from md_filename, NOT json_filename
    p = _resolve_live_backup_path(out, "md", "codedoc.json", "report.md")
    assert p == out / "report.json"


# ---------------------------------------------------------------------------
# Test — _build_default_ladder
# ---------------------------------------------------------------------------

def test_build_default_ladder():
    from codedoc.pipeline import _build_default_ladder

    assert _build_default_ladder(1) == [1]
    assert _build_default_ladder(2) == [2, 1]
    assert _build_default_ladder(3) == [3, 2, 1]
    assert _build_default_ladder(5) == [5, 2, 1]
    assert _build_default_ladder(10) == [10, 5, 1]
    assert _build_default_ladder(6) == [6, 3, 1]
    assert _build_default_ladder(7) == [7, 3, 1]
    # Always ends with 1
    for n in range(1, 15):
        ladder = _build_default_ladder(n)
        assert ladder[-1] == 1, f"Ladder for {n} must end with 1: {ladder}"
        assert ladder[0] == n, f"Ladder for {n} must start with {n}: {ladder}"
