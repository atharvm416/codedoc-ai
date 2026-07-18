"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import pytest
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import (
    RECOVERY_FILENAME,
    build_recovery_identity,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from tests.support.providers import SmartFake

def _config(output_format: str, output_dir: str = "docs") -> dict:
    return {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "output_dir": output_dir,
        "output_format": output_format,
        "parallel_agents": False,
        "propagate_changes": False,
    }

def _first_run(tmp_path: Path, monkeypatch, output_format: str, output_dir="docs"):
    fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: fake)
    stats = run_pipeline(tmp_path, _config(output_format, output_dir))
    return fake, stats

def _forbid_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: pytest.fail("provider must not be constructed for conversion"),
    )

def _write_compatible_md_recovery(tmp_path: Path, replacement: str) -> Path:
    out = tmp_path / "docs"
    stable = out / "codedoc.json"
    record = deepcopy(records_by_path(read_codedoc_document(stable))["main.py"])
    record["description"] = replacement
    identity = build_recovery_identity(
        project_root=tmp_path,
        json_target=None,
        md_target=out / "codedoc.md",
        entry_file="main.py",
        documentation_scope="all",
        analysis_mode="single",
        analysis_revision=ANALYSIS_REVISION,
    )
    recovery = out / RECOVERY_FILENAME
    writer = SafeWriter(recovery, "md", "main.py", {}, identity)
    writer.set_queue_order(["main.py"])
    writer.load(preloaded={"main.py": record})
    writer.initialize_empty()
    return recovery
