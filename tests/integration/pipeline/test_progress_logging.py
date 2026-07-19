"""Tests organized by feature ownership."""

from __future__ import annotations

import logging
from tests.support.logging_runs import patch_provider
from tests.support.logging_runs import write_py
from tests.support.clocks import capture_sleeps

class TestCleanLogs:
    """G1.1: Third-party logger silencing."""

    def test_httpx_silenced_after_configure(self):
        from codedoc.utils.logger import _configure
        _configure()
        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level >= logging.WARNING

    def test_httpcore_silenced_after_configure(self):
        from codedoc.utils.logger import _configure
        _configure()
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_all_noisy_loggers_silenced(self):
        from codedoc.utils.logger import _configure, _NOISY_LOGGERS
        _configure()
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING, (
                f"Logger '{name}' should be at WARNING or above after _configure()"
            )

    def test_debug_level_lowers_third_party_loggers(self):
        from codedoc.utils.logger import set_level, _NOISY_LOGGERS
        set_level("DEBUG")
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG, (
                f"Logger '{name}' should be at DEBUG when root level is DEBUG"
            )
        # Restore to INFO for other tests
        set_level("INFO")

    def test_info_level_keeps_third_party_loggers_at_warning(self):
        from codedoc.utils.logger import set_level, _NOISY_LOGGERS
        set_level("DEBUG")  # first lower
        set_level("INFO")   # then raise back
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING

class TestAgentProgressLogs:
    """G1.2: Per-agent progress lines appear in orchestrator."""

    def test_progress_lines_emitted(self, tmp_path, monkeypatch, caplog):
        """structure ok, dependencies ok, documentation ok lines appear per file.

        These per-agent progress lines belong to triple mode (the three-agent
        path); single mode logs a single 'combined ok' line instead.
        """
        patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        write_py(src)

        with caplog.at_level(logging.INFO, logger="codedoc.agents.orchestrator"):
            from codedoc.pipeline import run_pipeline
            run_pipeline(
                tmp_path,
                {"entry_file": "src.py", "output_dir": "out", "analysis_mode": "triple"},
            )

        messages = caplog.text
        assert "structure ok" in messages
        assert "dependencies ok" in messages
        assert "documentation ok" in messages

    def test_fallback_emits_warning(self, tmp_path, monkeypatch, caplog):
        """When _safe_run returns an error dict, WARNING with 'fallback' is emitted."""

        class FailingProvider:
            provider_name = "failing"

            def complete_json(self, prompt, system=""):
                raise RuntimeError("simulated agent failure")

            def complete(self, prompt, system="", temperature=0.1):
                raise RuntimeError("simulated agent failure")

        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: FailingProvider(),
        )

        src = tmp_path / "src.py"
        write_py(src)

        with caplog.at_level(logging.WARNING, logger="codedoc.agents.orchestrator"):
            from codedoc.pipeline import run_pipeline
            try:
                run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out"})
            except Exception:
                pass  # pipeline may raise AgentError — we only care about the log

        assert "fallback" in caplog.text

def test_D13_cli_summary_shows_compact_line_when_events(tmp_path, monkeypatch, capsys):
    """D13: CLI prints compact rate-limit line only when events occurred."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}

    class RLProvider:
        provider_name = "anthropic"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise LLMError("anthropic", "529 overloaded")
            import json as _j
            if "key_concepts" in prompt:
                return _j.dumps({"description": "ok", "role_in_system": "r",
                                  "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _j.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _j.dumps({"description": "ok", "role_in_system": "r",
                              "functions": [], "classes": [], "exports": []})
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: RLProvider())
    capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

    from codedoc.cli.cli import main
    try:
        main([str(tmp_path), "--entry", "a.py",
              "--max-parallel-files", "2",
              "--no-parallel"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    # The compact summary must mention "rate" and show event count
    assert "rate limit" in captured.out.lower() or "rate-limit" in captured.out.lower(), (
        f"CLI must print rate-limit summary when events occurred.\nOutput: {captured.out!r}"
    )

def test_D13_cli_summary_absent_on_clean_run(tmp_path, monkeypatch, capsys):
    """D13b: CLI must NOT print rate-limit summary when no events occurred."""
    (tmp_path / "main.py").write_text("x=1\n")

    import json as _j

    class OkProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _j.dumps({"description": "ok", "role_in_system": "r",
                                  "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _j.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _j.dumps({"description": "ok", "role_in_system": "r",
                              "functions": [], "classes": [], "exports": []})
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: OkProvider())

    from codedoc.cli.cli import main
    try:
        main([str(tmp_path), "--entry", "main.py", "--no-parallel"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "rate limit" not in captured.out.lower(), (
        f"No rate-limit summary expected on clean run.\nOutput: {captured.out!r}"
    )
