"""
``--output`` reserved-name guard and CLI interrupt messaging (0.11.3).

A user ``--output`` whose filename is exactly the reserved ``crash_recovery.json``
(case-insensitively) is rejected with ``ConfigError`` before any scan or mutation;
every other name — including the former ``crash_recovery_<stem>`` prefix names — is
allowed; and the CLI names the single recovery file on interrupt (or truthfully
reports that none was confirmed).
"""
from __future__ import annotations

import pytest

from codedoc.core.loader import load_config
from codedoc.utils.errors import ConfigError


# ---------------------------------------------------------------------------
# --output reserved-name guard (exact name only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reserved", [
    "crash_recovery.json",
    "docs/crash_recovery.json",
    "CRASH_RECOVERY.JSON",
])
def test_reserved_output_name_rejected(tmp_path, reserved):
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path, {"output_dir": reserved})
    assert "crash-recovery" in str(excinfo.value).lower()


@pytest.mark.parametrize("allowed", [
    "crash_recovery_codedoc.json",
    "docs/crash_recovery_report.json",
    "crash_recovery_report.md",
    "crash_recovery.md",  # only the exact .json name is reserved
])
def test_former_prefix_names_are_no_longer_reserved(tmp_path, allowed):
    cfg = load_config(tmp_path, {"output_dir": allowed})
    assert cfg["output_format"] in ("json", "md")


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
        run_pipeline(tmp_path, {"output_dir": "crash_recovery.json"})

    assert not (tmp_path / "crash_recovery.json").exists()


def test_ordinary_md_run_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {"output_dir": "report.md"})
    assert cfg["output_format"] == "md"
    assert cfg["output_md_filename"] == "report.md"
    assert cfg["output_json_filename"] == "report.json"


def test_non_reserved_named_output_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {"output_dir": "docs/report.json"})
    assert cfg["output_format"] == "json"
    assert cfg["output_json_filename"] == "report.json"
    assert cfg["output_md_filename"] == "report.md"


def test_default_run_not_rejected(tmp_path):
    cfg = load_config(tmp_path, {})
    assert cfg["output_json_filename"] == "codedoc.json"


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


# ---------------------------------------------------------------------------
# CLI interrupt messaging
# ---------------------------------------------------------------------------

def test_interrupt_after_recovery_init_propagates_recovery_path(tmp_path, monkeypatch):
    """An interrupt mid-processing attaches the exact recovery path."""
    (tmp_path / "main.py").write_text("x = 1\n")
    out = tmp_path / "codedoc"

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

    expected = out / "crash_recovery.json"
    assert getattr(excinfo.value, "recovery_path", None) == str(expected)
    assert expected.exists()


def test_cli_prints_recovery_path_when_attached(tmp_path, monkeypatch, capsys):
    import codedoc.pipeline as pipeline_mod

    def raise_with_path(*a, **k):
        exc = KeyboardInterrupt()
        exc.recovery_path = str(tmp_path / "codedoc" / "crash_recovery.json")
        raise exc

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_with_path)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "crash_recovery.json" in err
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
    assert not list(tmp_path.glob("**/crash_recovery.json"))
