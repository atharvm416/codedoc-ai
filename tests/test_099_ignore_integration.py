from __future__ import annotations

import json
import os

import pytest

from codedoc.cli.cli import build_parser
from codedoc.core.block_manager import BlockError
from codedoc.core.ignore_manager import END_MARKER, START_MARKER
from codedoc.core.loader import load_config
from codedoc.pipeline import _finalize_output_gitignore, run_pipeline
from codedoc.utils.errors import ConfigError


def _can_symlink(tmp_path) -> bool:
    target = tmp_path / "_probe_target"
    link = tmp_path / "_probe_link"
    target.write_text("x\n", encoding="utf-8")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError, AttributeError):
        return False
    finally:
        if link.is_symlink() or link.exists():
            link.unlink()
        target.unlink()
    return True


class _Provider:
    provider_name = "fake"

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)

    def complete_json(self, prompt, system=""):
        if "key_concepts" in prompt:
            return json.dumps(
                {
                    "description": "Documented file.",
                    "role_in_system": "test",
                    "key_concepts": [],
                    "usage_example": "",
                }
            )
        if "dependencies_analysis" in prompt:
            return json.dumps({"dependencies_analysis": {"internal": [], "external": []}})
        return json.dumps({"functions": [], "classes": [], "exports": []})


def test_manage_output_gitignore_is_inert_by_default_and_cli_is_tristate(tmp_path):
    parser = build_parser()
    assert parser.parse_args([]).manage_output_gitignore is None
    assert parser.parse_args(["--manage-output-gitignore"]).manage_output_gitignore is True
    assert parser.parse_args(["--no-manage-output-gitignore"]).manage_output_gitignore is False

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "dry_run": True})
    assert stats["output_gitignore_enabled"] is False
    assert not (tmp_path / "codedoc" / ".gitignore").exists()


@pytest.mark.parametrize(
    "filename", ["", ".", "..", "../ignore", "a/b", "CON", "bad.", "bad "]
)
def test_managed_ignore_filename_validation(tmp_path, filename):
    with pytest.raises(ConfigError, match="output_gitignore_filename"):
        load_config(tmp_path, {"output_gitignore_filename": filename})


def test_pipeline_updates_only_final_artifacts(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: _Provider())

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "manage_output_gitignore": True,
            "parallel_agents": False,
        },
    )

    target = tmp_path / "codedoc" / ".gitignore"
    text = target.read_text(encoding="utf-8")
    assert stats["output_gitignore_updated"] is True
    assert "/codedoc.json" in text
    assert "crash_recovery" not in text


def test_managed_ignore_target_colliding_with_artifact_is_rejected(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "manage_output_gitignore": True,
                "output_gitignore_filename": "codedoc.json",
                "dry_run": True,
            },
        )


def test_managed_ignore_symlink_target_is_rejected(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    output_dir = tmp_path / "codedoc"
    output_dir.mkdir()
    real = tmp_path / "real_ignore"
    real.write_text("user\n", encoding="utf-8")
    os.symlink(real, output_dir / ".gitignore")
    with pytest.raises(ConfigError):
        run_pipeline(
            tmp_path,
            {"entry_file": "main.py", "manage_output_gitignore": True, "dry_run": True},
        )


def test_managed_ignore_dry_run_reports_enabled_without_update(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    stats = run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "manage_output_gitignore": True, "dry_run": True},
    )
    assert stats["output_gitignore_enabled"] is True
    assert stats["output_gitignore_updated"] is False
    assert stats["output_gitignore_path"]
    assert not (tmp_path / "codedoc" / ".gitignore").exists()


def test_managed_ignore_failure_is_warning_only_and_preserves_content(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: _Provider())
    output_dir = tmp_path / "codedoc"
    output_dir.mkdir()
    # Pre-existing malformed ownership (duplicate start markers) — never overwritten.
    malformed = f"{START_MARKER}\n{START_MARKER}\n{END_MARKER}\n"
    (output_dir / ".gitignore").write_text(malformed, encoding="utf-8")

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "manage_output_gitignore": True,
            "parallel_agents": False,
        },
    )

    # Documentation still succeeds.
    assert stats["checked"] == 1
    assert (output_dir / "codedoc.json").exists()
    # The ignore update fails as an auxiliary, non-fatal warning.
    assert stats["output_gitignore_enabled"] is True
    assert stats["output_gitignore_updated"] is False
    assert stats["output_gitignore_warning"]
    # Malformed ownership is left byte-identical.
    assert (output_dir / ".gitignore").read_text(encoding="utf-8") == malformed


def test_managed_ignore_uses_renamed_output_and_existing_diagnostic(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    output_dir = tmp_path / "docs"
    output_dir.mkdir()
    (output_dir / "error.log").write_text("existing diagnostic\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: _Provider())

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "output_dir": "docs/report.json",
            "manage_output_gitignore": True,
            "parallel_agents": False,
        },
    )

    text = (output_dir / ".gitignore").read_text(encoding="utf-8")
    assert stats["output_gitignore_updated"] is True
    assert "/report.json" in text
    assert "/error.log" in text
    assert "codedoc.json" not in text
    assert "crash_recovery" not in text


def test_managed_ignore_finalization_follows_required_outputs_and_cleanup(
    tmp_path, monkeypatch
):
    import codedoc.pipeline as pipeline_module
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.utils.errors import ErrorReporter

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: _Provider())
    events = []

    real_write = pipeline_module.write_project_outputs
    real_flush = ErrorReporter.flush
    real_delete = SafeWriter.delete
    real_cleanup = pipeline_module._cleanup_legacy_recovery
    real_finalize = pipeline_module._finalize_output_gitignore

    def write_outputs(*args, **kwargs):
        result = real_write(*args, **kwargs)
        events.append("stable outputs")
        return result

    def flush(reporter):
        result = real_flush(reporter)
        events.append("diagnostics")
        return result

    def delete(writer):
        result = real_delete(writer)
        events.append("recovery delete")
        return result

    def cleanup(*args, **kwargs):
        result = real_cleanup(*args, **kwargs)
        events.append("recovery cleanup")
        return result

    def finalize(*args, **kwargs):
        events.append("managed ignore")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "write_project_outputs", write_outputs)
    monkeypatch.setattr(ErrorReporter, "flush", flush)
    monkeypatch.setattr(SafeWriter, "delete", delete)
    monkeypatch.setattr(pipeline_module, "_cleanup_legacy_recovery", cleanup)
    monkeypatch.setattr(pipeline_module, "_finalize_output_gitignore", finalize)

    run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "manage_output_gitignore": True,
            "parallel_agents": False,
        },
    )
    assert events == [
        "stable outputs",
        "diagnostics",
        "recovery delete",
        "recovery cleanup",
        "managed ignore",
    ]


def test_auxiliary_warning_flush_failure_remains_nonfatal(tmp_path, monkeypatch):
    output = tmp_path / "codedoc.json"
    output.write_text("{}\n", encoding="utf-8")

    class FailingReporter:
        log_path = tmp_path / "error.log"

        def record(self, *_args, **_kwargs):
            return None

        def flush(self):
            raise OSError("diagnostic unavailable")

    def fail_update(*_args, **_kwargs):
        raise BlockError("malformed owned block")

    monkeypatch.setattr("codedoc.pipeline.update_output_gitignore", fail_update)
    stats = {
        "output_gitignore_enabled": True,
        "output_gitignore_updated": False,
        "output_gitignore_path": str(tmp_path / ".gitignore"),
        "output_gitignore_warning": None,
    }
    _finalize_output_gitignore(
        stats,
        {"manage_output_gitignore": True, "output_gitignore_filename": ".gitignore"},
        tmp_path,
        tmp_path / ".gitignore",
        (output, None),
        FailingReporter(),
    )
    assert stats["output_gitignore_updated"] is False
    assert "auxiliary warning log could not be persisted" in stats[
        "output_gitignore_warning"
    ]
