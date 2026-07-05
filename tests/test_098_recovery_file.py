"""
0.9.8 — Gate A: dedicated crash-recovery file.

Verifies that in-progress (crash-recovery) records are staged in a distinct
``crash_recovery.json`` file, that the stable completed output is never
mutated before clean completion, that the recovery file is deleted only after a
successful stable write, and that the artifact-path collision check treats the
recovery file as its own artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoc.utils.errors import OutputError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_provider(observer=None):
    """A fake provider that returns valid agent JSON.

    ``observer`` (optional) is invoked with no arguments on every
    ``complete_json`` call, letting a test inspect on-disk state *mid-run*.
    """
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if observer is not None:
                observer()
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": "Documented.",
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
                "description": "Documented.",
                "role_in_system": "test",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


def _run(tmp_path, monkeypatch, provider=None, **cfg):
    if provider is None:
        provider = _fake_provider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)


def _write_codedoc_json(path: Path, files: list, status: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {"entry_file": "main.py", "schema_version": "1.4"}
    if status:
        meta["status"] = status
    payload: dict = {"_codedoc": meta, "schema_version": "1.4", "files": files}
    if status == "in_progress":
        payload = {"_crash_safety": "INCOMPLETE RUN - test", **payload}
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# A run writes records to crash_recovery.json, not the stable JSON
# ---------------------------------------------------------------------------

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

    _run(tmp_path, monkeypatch, provider=_fake_provider(observe), entry_file="main.py")

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

    _run(
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

    _run(
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

    _run(
        tmp_path, monkeypatch, provider=_fake_provider(observe),
        entry_file="main.py", output_dir="docs/report.md",
    )

    assert seen["recovery_midrun"] is True
    assert stable_md.exists()
    assert not recovery.exists()
    # No stray JSON sibling left behind.
    assert not (tmp_path / "docs" / "report.json").exists()


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_forced_stable_write_failure_preserves_recovery_and_prior_stable(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    stable = out / "codedoc.json"
    recovery = out / "crash_recovery.json"

    _write_codedoc_json(stable, [{"path": "old.py", "hash": "OLD", "language": "python"}])
    stable_before = stable.read_bytes()

    import codedoc.pipeline as pipeline_mod

    def boom(*a, **k):
        raise OutputError(str(stable), "forced stable-write failure")

    monkeypatch.setattr(pipeline_mod, "write_project_outputs", boom)

    with pytest.raises(OutputError):
        _run(tmp_path, monkeypatch, entry_file="main.py")

    # The stable output is left exactly as it was; the recovery file survives.
    assert stable.read_bytes() == stable_before
    assert recovery.exists()
    rec = json.loads(recovery.read_text(encoding="utf-8"))
    assert rec.get("_codedoc", {}).get("status") == "in_progress"


def test_forced_recovery_deletion_oserror_raises_outputerror_and_keeps_both(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    stable = out / "codedoc.json"
    recovery = out / "crash_recovery.json"

    real_unlink = Path.unlink

    def guarded_unlink(self, *args, **kwargs):
        if self.name == "crash_recovery.json":
            raise OSError("forced recovery deletion failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", guarded_unlink)

    with pytest.raises(OutputError) as excinfo:
        _run(tmp_path, monkeypatch, entry_file="main.py")

    # The completed stable output remains; the recovery file is preserved; the
    # error names the recovery path.
    assert stable.exists()
    completed = json.loads(stable.read_text(encoding="utf-8"))
    assert "_crash_safety" not in completed
    assert recovery.exists()
    assert "crash_recovery.json" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Artifact-path collision
# ---------------------------------------------------------------------------

def test_validate_distinct_rejects_recovery_equal_to_json(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths({
            "json": tmp_path / "codedoc.json",
            "live_backup": tmp_path / "codedoc.json",
        })


def test_validate_distinct_rejects_recovery_equal_to_markdown(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths({
            "markdown": tmp_path / "codedoc.md",
            "live_backup": tmp_path / "codedoc.md",
        })


def test_validate_distinct_rejects_recovery_equal_to_log(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths
    from codedoc.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        validate_distinct_artifact_paths({
            "error_log": tmp_path / "error.log",
            "live_backup": tmp_path / "error.log",
        })


def test_validate_distinct_accepts_separate_recovery(tmp_path):
    from codedoc.core.output import validate_distinct_artifact_paths

    # The normal 0.9.8 layout: all four artifacts are distinct.
    validate_distinct_artifact_paths({
        "error_log": tmp_path / "error.log",
        "json": tmp_path / "codedoc.json",
        "markdown": tmp_path / "codedoc.md",
        "live_backup": tmp_path / "crash_recovery.json",
    })
