"""
0.9.8 — Gate C: ``--output`` reserved-name guard and CLI interrupt messaging.

A user ``--output`` whose own filename stem begins with ``crash_recovery_`` is
rejected with ``ConfigError`` before any scan or mutation; the default and
derived recovery names are never rejected; and the CLI names the dedicated
recovery file on interrupt (or truthfully reports that none was confirmed).
"""
from __future__ import annotations

import json

import pytest

from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError


# ---------------------------------------------------------------------------
# --output reserved-name guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reserved", [
    "crash_recovery_codedoc.json",
    "crash_recovery_codedoc(2).json",
    "docs/crash_recovery_report.json",
    "crash_recovery_report.md",
])
def test_reserved_output_names_rejected(tmp_path, reserved):
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path, {"output_dir": reserved})
    assert "crash_recovery_" in str(excinfo.value)


def test_reserved_output_name_rejected_before_scan_or_mutation(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n")

    import codedoc.pipeline as pipeline_mod
    monkeypatch.setattr(
        pipeline_mod, "scan_files",
        lambda *a, **k: pytest.fail("scan must not run for a reserved --output"),
    )
    monkeypatch.setattr(
        pipeline_mod, "create_provider",
        lambda *a, **k: pytest.fail("provider must not be created"),
    )

    from codedoc.pipeline import run_pipeline
    with pytest.raises(ConfigError):
        run_pipeline(tmp_path, {"output_dir": "crash_recovery_codedoc.json"})

    # Nothing was written.
    assert not (tmp_path / "crash_recovery_codedoc.json").exists()


def test_ordinary_md_run_not_rejected(tmp_path):
    # The default (unused) output_json_filename in md mode must never trip the
    # guard — only the user-supplied filename is checked.
    cfg = load_config(tmp_path, {"output_dir": "report.md"})
    assert cfg["output_format"] == "md"
    assert cfg["output_md_filename"] == "report.md"


def test_non_reserved_named_output_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {"output_dir": "docs/report.json"})
    assert cfg["output_format"] == "json"
    assert cfg["output_json_filename"] == "report.json"


def test_default_run_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {})
    assert cfg["output_json_filename"] == "codedoc.json"


def test_foreign_file_at_base_recovery_name_is_preserved(tmp_path, monkeypatch):
    """A foreign file at the base recovery name is preserved; the run uses (2)."""
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    base = out / "crash_recovery_codedoc.json"
    foreign_bytes = b"not a codedoc file"
    base.write_bytes(foreign_bytes)

    import codedoc.pipeline as pipeline_mod

    class _Provider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return json.dumps({"description": "d", "role_in_system": "t",
                                   "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return json.dumps({"description": "d", "role_in_system": "t",
                               "functions": [], "classes": [], "exports": []})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr(pipeline_mod, "create_provider", lambda _cfg: _Provider())
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"parallel_agents": False, "propagate_changes": False})

    assert base.read_bytes() == foreign_bytes      # foreign file untouched
    assert (out / "codedoc.json").exists()          # stable output written


# ---------------------------------------------------------------------------
# CLI interrupt messaging
# ---------------------------------------------------------------------------

def test_interrupt_after_recovery_init_propagates_selected_path(tmp_path, monkeypatch):
    """An interrupt mid-processing attaches the exact selected recovery path.

    With a malformed base recovery file present, the active recovery file is the
    ``(2)`` suffix; the propagated KeyboardInterrupt must name exactly that path.
    """
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"
    out.mkdir()
    (out / "crash_recovery_codedoc.json").write_bytes(b"malformed")

    import codedoc.pipeline as pipeline_mod

    def raise_interrupt(_context):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_mod, "execute_agent_files", raise_interrupt)
    monkeypatch.setattr(
        pipeline_mod, "create_provider",
        lambda _cfg: type("P", (), {"provider_name": "fake"})(),
    )

    from codedoc.pipeline import run_pipeline
    with pytest.raises(KeyboardInterrupt) as excinfo:
        run_pipeline(tmp_path, {"parallel_agents": False, "propagate_changes": False})

    expected = out / "crash_recovery_codedoc(2).json"
    assert getattr(excinfo.value, "recovery_path", None) == str(expected)
    assert expected.exists()


def test_cli_prints_recovery_path_when_attached(tmp_path, monkeypatch, capsys):
    import codedoc.pipeline as pipeline_mod

    def raise_with_path(*a, **k):
        exc = KeyboardInterrupt()
        exc.recovery_path = str(tmp_path / "codedoc" / "crash_recovery_codedoc(2).json")
        raise exc

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_with_path)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "crash_recovery_codedoc(2).json" in err
    assert "left untouched" in err


def test_cli_generic_message_when_no_recovery_path(tmp_path, monkeypatch, capsys):
    import codedoc.pipeline as pipeline_mod

    def raise_plain(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_plain)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "crash-recovery file was created or confirmed" in err
    # The CLI does not fabricate a recovery file while reporting.
    assert not list(tmp_path.glob("**/crash_recovery_*.json"))
