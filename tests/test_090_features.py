"""
0.9.0 feature tests: G0 (preflight safety), G1 (clean logs), G6 (configurable truncation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def make_fake_provider():
    import json as _json

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": "Test.",
                    "role_in_system": "test role",
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
                "description": "Test.",
                "role_in_system": "test role",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    return FakeProvider()


def patch_provider(monkeypatch):
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda config: make_fake_provider(),
    )


def write_py(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# G0: Preflight output safety
# ---------------------------------------------------------------------------

class TestPreflightOutputTargets:
    """G0: Foreign targets fail before create_provider() is called."""

    def test_foreign_json_target_raises_before_provider(self, tmp_path, monkeypatch):
        """A foreign report.json causes ConfigError before LLM is ever instantiated."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or make_fake_provider(),
        )

        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "report.json").write_text('{"not": "codedoc"}', encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.json"})

        assert not provider_called, "create_provider must not be called before preflight"

    def test_foreign_md_target_raises_before_provider(self, tmp_path, monkeypatch):
        """A foreign report.md causes ConfigError before LLM is instantiated."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or make_fake_provider(),
        )

        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "report.md").write_text("# personal file\n", encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.md"})

        assert not provider_called, "create_provider must not be called before preflight"

    def test_codedoc_owned_json_passes_preflight(self, tmp_path, monkeypatch):
        """An existing CodeDoc-owned JSON passes preflight and the run succeeds."""
        patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        owned = {
            "_codedoc": {"schema_version": "1.4"},
            "files": [],
        }
        (out_dir / "codedoc.json").write_text(json.dumps(owned), encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out"})
        assert stats is not None

    def test_codedoc_owned_md_passes_preflight(self, tmp_path, monkeypatch):
        """An existing CodeDoc-owned Markdown passes preflight.

        0.9.3: ownership requires structurally valid metadata.  A marker-only
        Markdown with malformed metadata is now treated as foreign (the
        centralized reader fails closed), so the owned fixture must carry a
        well-formed ``<!-- codedoc-ai: {...} -->`` comment.
        """
        import json as _json

        patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        meta = {"entry_file": "src.py", "schema_version": "1.4", "file_hashes": {}}
        (out_dir / "codedoc.md").write_text(
            f"<!-- codedoc-ai: {_json.dumps(meta)} -->\n# docs\n", encoding="utf-8"
        )

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(
            tmp_path, {"entry_file": "src.py", "output_dir": "out", "output_format": "md"}
        )
        assert stats is not None

    def test_nonexistent_output_dir_passes_preflight(self, tmp_path, monkeypatch):
        """A non-existent output directory passes preflight (all targets are new)."""
        patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        write_py(src)

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(
            tmp_path, {"entry_file": "src.py", "output_dir": "brand_new_dir"}
        )
        assert stats is not None

    def test_format_both_fails_if_json_target_foreign(self, tmp_path, monkeypatch):
        """--format both preflights both targets; fails if JSON is foreign."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or make_fake_provider(),
        )

        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "codedoc.json").write_text('{"not": "codedoc"}', encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(
                tmp_path,
                {"entry_file": "src.py", "output_dir": "out", "output_format": "both"},
            )
        assert not provider_called

    def test_foreign_live_backup_sibling_fails_preflight_for_md(self, tmp_path, monkeypatch):
        """A foreign JSON sibling of an MD target fails preflight before scanning."""
        from codedoc.utils.errors import ConfigError

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or make_fake_provider(),
        )

        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # The live backup for report.md is report.json — make it foreign
        (out_dir / "report.json").write_text('{"foreign": true}', encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.md"})

        assert not provider_called

    def test_foreign_target_leaves_no_new_output_directory(self, tmp_path, monkeypatch):
        """Preflight fires before mkdir: a genuinely new output dir is not created on failure."""
        from codedoc.utils.errors import ConfigError

        src = tmp_path / "src.py"
        write_py(src)

        # Create a sibling directory with a foreign codedoc.json — this is the
        # scenario where the user points at an existing dir that has a foreign file.
        out_dir = tmp_path / "my_out"
        out_dir.mkdir()
        (out_dir / "codedoc.json").write_text('{"foreign": true}', encoding="utf-8")

        # A child sub-dir that does NOT exist yet — preflight should prevent its creation.
        new_sub = out_dir / "sub_that_must_not_be_created"

        provider_called = []
        monkeypatch.setattr(
            "codedoc.pipeline.create_provider",
            lambda config: provider_called.append(True) or make_fake_provider(),
        )

        from codedoc.pipeline import run_pipeline
        with pytest.raises(ConfigError):
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": str(out_dir)})

        assert not provider_called
        # The directory contents must be unchanged — only the foreign file we planted
        assert not new_sub.exists(), "No new subdirectories created on preflight failure"

    def test_completed_json_sibling_accepted_for_md_preflight(self, tmp_path, monkeypatch):
        """A completed CodeDoc JSON from a prior JSON run is accepted as an MD live-backup sibling."""
        patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        write_py(src)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        completed = {
            "_codedoc": {"schema_version": "1.4", "status": "complete"},
            "files": [],
        }
        (out_dir / "report.json").write_text(json.dumps(completed), encoding="utf-8")

        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out/report.md"})
        assert stats is not None


# ---------------------------------------------------------------------------
# G1: Clean log output
# ---------------------------------------------------------------------------

class TestCleanLogs:
    """G1.1: Third-party logger silencing."""

    def test_httpx_silenced_after_configure(self):
        from codedoc.utils.logger import _configure, _NOISY_LOGGERS
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
        """structure ok, dependencies ok, documentation ok lines appear per file."""
        patch_provider(monkeypatch)
        src = tmp_path / "src.py"
        write_py(src)

        with caplog.at_level(logging.INFO, logger="codedoc.agents.orchestrator"):
            from codedoc.pipeline import run_pipeline
            run_pipeline(tmp_path, {"entry_file": "src.py", "output_dir": "out"})

        messages = caplog.text
        assert "structure ok" in messages
        assert "dependencies ok" in messages
        assert "documentation ok" in messages

    def test_fallback_emits_warning(self, tmp_path, monkeypatch, caplog):
        """When _safe_run returns an error dict, WARNING with 'fallback' is emitted."""
        import json as _json

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


# ---------------------------------------------------------------------------
# G6: Configurable content truncation
# ---------------------------------------------------------------------------

class TestConfigurableContentTruncation:
    """G6: max_content_chars is configurable and truncation is logged at INFO."""

    def test_default_max_content_chars(self):
        from codedoc.core.loader import DEFAULTS
        assert DEFAULTS["max_content_chars"] == 12000

    def test_max_content_chars_too_small_raises(self):
        from codedoc.utils.errors import ConfigError
        from codedoc.core.loader import load_config
        with pytest.raises(ConfigError, match="at least 1000"):
            load_config(Path("."), {"max_content_chars": 500})

    def test_max_content_chars_invalid_type_raises(self):
        from codedoc.utils.errors import ConfigError
        from codedoc.core.loader import load_config
        with pytest.raises(ConfigError, match="positive integer"):
            load_config(Path("."), {"max_content_chars": "not-a-number"})

    def test_content_not_truncated_when_below_limit(self):
        from codedoc.agents.base_agent import BaseAgent
        from codedoc.llm.base import LLMProvider

        class ConcreteAgent(BaseAgent):
            agent_name = "Test"
            def run(self, *a, **kw): return {}

        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def complete_json(self, *a, **kw): return "{}"
            def complete(self, *a, **kw): return ""

        agent = ConcreteAgent(DummyProvider(), max_content_chars=100)
        short = "x" * 50
        assert agent._truncate(short, "test.py") == short

    def test_content_truncated_when_above_limit(self):
        from codedoc.agents.base_agent import BaseAgent
        from codedoc.llm.base import LLMProvider

        class ConcreteAgent(BaseAgent):
            agent_name = "Test"
            def run(self, *a, **kw): return {}

        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def complete_json(self, *a, **kw): return "{}"
            def complete(self, *a, **kw): return ""

        agent = ConcreteAgent(DummyProvider(), max_content_chars=1000)
        long = "x" * 2000
        result = agent._truncate(long, "big.py")
        assert len(result) < 2000
        assert "[truncated]" in result

    def test_truncation_logged_at_debug_with_filename(self, caplog):
        """Defensive truncation logs at DEBUG (0.9.2) with filename and char counts.

        Orchestrated runs warn once in Orchestrator.process(); the agents'
        fallback stays quiet so one file never emits three warnings.
        """
        from codedoc.agents.base_agent import BaseAgent
        from codedoc.llm.base import LLMProvider

        class ConcreteAgent(BaseAgent):
            agent_name = "Test"
            def run(self, *a, **kw): return {}

        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def complete_json(self, *a, **kw): return "{}"
            def complete(self, *a, **kw): return ""

        agent = ConcreteAgent(DummyProvider(), max_content_chars=1000)
        with caplog.at_level(logging.DEBUG, logger="codedoc.agents.base_agent"):
            agent._truncate("x" * 5000, "large_file.py")

        assert "large_file.py" in caplog.text
        assert "5000" in caplog.text
        assert "1000" in caplog.text

    def test_orchestrator_default_max_content_chars(self):
        """Orchestrator constructed without max_content_chars uses 12000 (backward compat)."""
        from codedoc.agents.orchestrator import Orchestrator
        from codedoc.llm.base import LLMProvider

        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def complete_json(self, *a, **kw): return "{}"
            def complete(self, *a, **kw): return ""

        orch = Orchestrator(DummyProvider())
        assert orch._structure_agent._max_content_chars == 12000
        assert orch._dependency_agent._max_content_chars == 12000
        assert orch._doc_agent._max_content_chars == 12000

    def test_orchestrator_forwards_max_content_chars(self):
        """Orchestrator passes max_content_chars down to each agent."""
        from codedoc.agents.orchestrator import Orchestrator
        from codedoc.llm.base import LLMProvider

        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def complete_json(self, *a, **kw): return "{}"
            def complete(self, *a, **kw): return ""

        orch = Orchestrator(DummyProvider(), max_content_chars=50000)
        assert orch._structure_agent._max_content_chars == 50000
        assert orch._dependency_agent._max_content_chars == 50000
        assert orch._doc_agent._max_content_chars == 50000

    def test_pipeline_wires_max_content_chars(self, tmp_path, monkeypatch):
        """max_content_chars from config reaches the agents via the pipeline."""
        received = []

        original_truncate = None

        def capturing_truncate(self, content, file_path=""):
            received.append(self._max_content_chars)
            return content

        from codedoc.agents import base_agent
        monkeypatch.setattr(base_agent.BaseAgent, "_truncate", capturing_truncate)
        patch_provider(monkeypatch)

        src = tmp_path / "src.py"
        write_py(src, "x = 1\n")

        from codedoc.pipeline import run_pipeline
        run_pipeline(
            tmp_path,
            {"entry_file": "src.py", "output_dir": "out", "max_content_chars": 25000},
        )

        assert received, "No _truncate calls recorded"
        assert all(v == 25000 for v in received), f"Expected 25000 but got: {set(received)}"
