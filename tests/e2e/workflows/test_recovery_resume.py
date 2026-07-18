"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest

def test_A5_interrupt_prints_clean_message_and_exits_130(tmp_path, monkeypatch, capsys):
    """A5: KeyboardInterrupt with no recovery_path attached prints the truthful
    generic message (no recovery file confirmed) and exits 130.

    0.9.8: when the interrupt carries no ``recovery_path`` (interrupted before a
    recovery file was initialized), the CLI must not claim a recovery file
    exists, and must affirm the stable output was left untouched.
    """
    import pytest

    import codedoc.pipeline as pipeline_mod

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_interrupt)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "interrupted" in err.lower()
    # Truthful generic wording: no recovery file was created/confirmed, and the
    # stable output was left untouched.  Never the old "Progress has been saved".
    assert "crash-recovery file was created or confirmed" in err
    assert "left untouched" in err
    assert "Progress has been saved" not in err

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
