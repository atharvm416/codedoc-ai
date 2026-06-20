"""
0.9.8 — Gate B: resume, malformed-file suffix increment, legacy migration.

Verifies the candidate-walk selection of the active recovery file, the fixed
three-source overlay precedence (stable baseline < legacy in-progress < active
recovery), bounded suffix exhaustion, and automatic migration of a pre-0.9.8
in-progress stable output.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoc.core.db import compute_file_hash
from codedoc.core.resume import (
    _load_existing_file_docs,
    select_active_recovery_path,
)
from codedoc.utils.errors import OutputError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fake_provider():
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": "Documented.", "role_in_system": "test",
                    "key_concepts": [], "usage_example": "",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": [],
                }})
            return _json.dumps({
                "description": "Documented.", "role_in_system": "test",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


def _run(tmp_path, monkeypatch, **cfg):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: _fake_provider())
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)


def _in_progress(path: Path, files: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_crash_safety": "INCOMPLETE RUN - test",
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4", "status": "in_progress"},
        "schema_version": "1.4",
        "files": files,
    }), encoding="utf-8")


def _completed(path: Path, files: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_codedoc": {"entry_file": "main.py", "schema_version": "1.4"},
        "schema_version": "1.4",
        "files": files,
    }), encoding="utf-8")


def _foreign(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid codedoc json", encoding="utf-8")


# ---------------------------------------------------------------------------
# select_active_recovery_path — candidate-walk rules
# ---------------------------------------------------------------------------

def test_absent_base_is_fresh(tmp_path):
    base = tmp_path / "crash_recovery_codedoc.json"
    active, resume = select_active_recovery_path(base)
    assert active == base
    assert resume is False


def test_valid_in_progress_base_is_resumed(tmp_path):
    base = tmp_path / "crash_recovery_codedoc.json"
    _in_progress(base, [{"path": "a.py", "hash": "H"}])
    active, resume = select_active_recovery_path(base)
    assert active == base
    assert resume is True


def test_malformed_base_advances_to_suffix_2(tmp_path):
    base = tmp_path / "crash_recovery_codedoc.json"
    before = b"{ this is not valid codedoc json"
    base.write_bytes(before)
    active, resume = select_active_recovery_path(base)
    assert active == tmp_path / "crash_recovery_codedoc(2).json"
    assert resume is False
    # The malformed base is left byte-identical (preserved, never overwritten).
    assert base.read_bytes() == before


def test_malformed_base_and_two_advances_to_three(tmp_path):
    base = tmp_path / "crash_recovery_codedoc.json"
    _foreign(base)
    _foreign(tmp_path / "crash_recovery_codedoc(2).json")
    active, resume = select_active_recovery_path(base)
    assert active == tmp_path / "crash_recovery_codedoc(3).json"
    assert resume is False


def test_valid_two_with_malformed_base_is_resumed(tmp_path):
    base = tmp_path / "crash_recovery_codedoc.json"
    _foreign(base)
    _in_progress(tmp_path / "crash_recovery_codedoc(2).json", [{"path": "a.py", "hash": "H"}])
    active, resume = select_active_recovery_path(base)
    assert active == tmp_path / "crash_recovery_codedoc(2).json"
    assert resume is True


def test_completed_clean_recovery_name_is_preserved_not_resumed(tmp_path):
    base = tmp_path / "crash_recovery_codedoc.json"
    _completed(base, [{"path": "a.py", "hash": "H"}])
    active, resume = select_active_recovery_path(base)
    # A clean completed CodeDoc JSON at a reserved name is not a live source.
    assert active == tmp_path / "crash_recovery_codedoc(2).json"
    assert resume is False


def test_suffix_exhaustion_raises_outputerror(tmp_path, monkeypatch):
    import codedoc.core.resume as resume_mod
    monkeypatch.setattr(resume_mod, "_MAX_RECOVERY_CANDIDATES", 3)

    base = tmp_path / "crash_recovery_codedoc.json"
    _foreign(base)
    _foreign(tmp_path / "crash_recovery_codedoc(2).json")
    _foreign(tmp_path / "crash_recovery_codedoc(3).json")

    with pytest.raises(OutputError) as excinfo:
        select_active_recovery_path(base)
    assert str(tmp_path) in str(excinfo.value)


# ---------------------------------------------------------------------------
# _load_existing_file_docs — overlay precedence
# ---------------------------------------------------------------------------

def test_resume_merges_stable_baseline_with_recovery_overlay(tmp_path):
    out = tmp_path / "codedoc"
    _completed(out / "codedoc.json", [
        {"path": "a.py", "hash": "BASE", "description": "BASE_A"},
    ])
    _in_progress(out / "crash_recovery_codedoc.json", [
        {"path": "a.py", "hash": "REC", "description": "RECOVERY_A"},
        {"path": "b.py", "hash": "REC", "description": "RECOVERY_B"},
    ])

    docs = _load_existing_file_docs(
        out, "codedoc.json", "codedoc.md",
        out / "crash_recovery_codedoc.json",
        output_format="json",
        legacy_recovery_path=out / "codedoc.json",
    )
    # Active recovery overlays the stable baseline (whole-record replacement).
    assert docs["a.py"]["description"] == "RECOVERY_A"
    assert docs["b.py"]["description"] == "RECOVERY_B"


def test_three_source_overlay_precedence_md_mode(tmp_path, monkeypatch):
    """Clean md baseline < legacy in-progress json sibling < active recovery."""
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    # Produce a real, valid report.md baseline (a.py + b.py, "Documented.").
    _run(tmp_path, monkeypatch, output_dir="codedoc/report.md")
    out = tmp_path / "codedoc"
    assert (out / "report.md").exists()

    # Legacy in-progress json sibling (pre-0.9.8 location) overlays a.py.
    _in_progress(out / "report.json", [
        {"path": "a.py", "hash": "LEG", "description": "LEGACY_A"},
    ])
    # Active recovery overlays a.py again (newest).
    _in_progress(out / "crash_recovery_report.json", [
        {"path": "a.py", "hash": "REC", "description": "RECOVERY_A"},
    ])

    docs = _load_existing_file_docs(
        out, "codedoc.json", "report.md",
        out / "crash_recovery_report.json",
        output_format="md",
        legacy_recovery_path=out / "report.json",
    )
    assert docs["a.py"]["description"] == "RECOVERY_A"   # recovery wins
    assert "b.py" in docs                                # from md baseline
    assert docs["b.py"]["description"] == "Documented."


# ---------------------------------------------------------------------------
# Pipeline-level resume / increment / migration
# ---------------------------------------------------------------------------

def test_pipeline_resumes_from_valid_recovery_file(tmp_path, monkeypatch):
    main_py = tmp_path / "main.py"
    other_py = tmp_path / "other.py"
    # main imports other so the inferred entry ('main.py', read from the recovery
    # doc) reaches both files and both are selected.
    main_py.write_text("from other import y\n")
    other_py.write_text("y = 2\n")

    out = tmp_path / "codedoc"
    _in_progress(out / "crash_recovery_codedoc.json", [
        {"path": "main.py", "hash": compute_file_hash(main_py),
         "language": "python", "description": "done"},
    ])

    stats = _run(tmp_path, monkeypatch)

    # main.py was already recorded with a matching hash → skipped; only other.py
    # is sent to the provider.
    assert stats["checked"] == 1
    assert stats["skipped"] >= 1

    final = json.loads((out / "codedoc.json").read_text(encoding="utf-8"))
    paths = {f["path"] for f in final["files"]}
    assert paths == {"main.py", "other.py"}
    assert not (out / "crash_recovery_codedoc.json").exists()


def test_pipeline_skips_malformed_base_and_uses_suffix_two(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    base = out / "crash_recovery_codedoc.json"
    foreign_bytes = b"{ not codedoc json at all"
    out.mkdir(parents=True, exist_ok=True)
    base.write_bytes(foreign_bytes)

    seen = {"two_midrun": False}

    def observe():
        if (out / "crash_recovery_codedoc(2).json").exists():
            seen["two_midrun"] = True

    import codedoc.pipeline as pipeline_mod

    class _Obs:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            observe()
            return _fake_provider().complete_json(prompt, system)

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr(pipeline_mod, "create_provider", lambda _cfg: _Obs())
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"parallel_agents": False, "propagate_changes": False})

    # The malformed base is preserved byte-identical; the run used the (2) suffix.
    assert base.read_bytes() == foreign_bytes
    assert seen["two_midrun"] is True
    assert (out / "codedoc.json").exists()
    # The active (2) recovery file was cleaned up on success.
    assert not (out / "crash_recovery_codedoc(2).json").exists()


def test_legacy_in_progress_stable_output_is_migrated(tmp_path, monkeypatch):
    main_py = tmp_path / "main.py"
    other_py = tmp_path / "other.py"
    main_py.write_text("from other import y\n")
    other_py.write_text("y = 2\n")

    out = tmp_path / "codedoc"
    # Pre-0.9.8 layout: the stable codedoc.json is itself an in-progress
    # _crash_safety document (the old scheme wrote the banner into the final JSON).
    _in_progress(out / "codedoc.json", [
        {"path": "main.py", "hash": compute_file_hash(main_py),
         "language": "python", "description": "pre-migration"},
    ])

    stats = _run(tmp_path, monkeypatch)

    # main.py resumed from the legacy in-progress source; only other.py processed.
    assert stats["checked"] == 1

    # The stable output is now a clean completed document with both files; no
    # recovery file is left behind and the user deleted nothing by hand.
    final_text = (out / "codedoc.json").read_text(encoding="utf-8")
    final = json.loads(final_text)
    assert "_crash_safety" not in final
    assert final.get("_codedoc", {}).get("status") != "in_progress"
    paths = {f["path"] for f in final["files"]}
    assert paths == {"main.py", "other.py"}
    assert not (out / "crash_recovery_codedoc.json").exists()
