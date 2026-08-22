"""Tests organized by feature ownership."""

from __future__ import annotations

import logging
from tests.support.logging_runs import patch_provider
from tests.support.logging_runs import write_py
from tests.support.clocks import capture_sleeps
from codedoc.core.execution import _process_divided_file
from codedoc.core.file_division import build_division_plan, build_reduction_tree
from codedoc.core.safe_writer import SafeWriter
from tests.support.execution_requests import make_execution_request
from tests.support.logging_sentinels import (
    SENTINEL_PROMPT_FRAGMENT,
    SENTINEL_REQUEST_BODY,
    SENTINEL_SOURCE_LINE,
    assert_no_sentinels_leaked,
)

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

    def test_debug_level_never_lowers_third_party_loggers(self):
        """Section 3A fail-closed fix: CodeDoc DEBUG must never opt any
        third-party provider/transport logger into DEBUG. The floor is
        WARNING-or-stricter in both normal and verbose execution, and
        set_level() never touches the root logger (pytest's own logging
        capture may already own root, so this asserts root is unchanged by
        set_level() rather than pinned to a specific absolute value)."""
        from codedoc.utils.logger import set_level, _NOISY_LOGGERS

        root_level_before = logging.getLogger().level
        set_level("DEBUG")
        assert logging.getLogger("codedoc").level == logging.DEBUG
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING, (
                f"Logger '{name}' must stay at WARNING-or-stricter even when "
                "codedoc is at DEBUG"
            )
        assert logging.getLogger().level == root_level_before
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
    from tests.support.provider_failures import provider_failure_error

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}

    class RLProvider:
        provider_name = "anthropic"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise provider_failure_error(
                    "anthropic", "provider-rate-limited", status=529, limit_type="overloaded"
                )
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


class TestBoundedSplitDiagnostics:
    """Section 3A/19: verbose split execution emits one bounded completion
    event per leaf/reducer/final node, in order, with no raw payload."""

    class _StubOrchestrator:
        def __init__(self):
            self.leaf_calls = 0
            self.reduction_calls = 0
            self.synthesis_calls = 0

        def process_leaf_chunk(self, request):
            self.leaf_calls += 1
            return {
                "description": f"leaf {request.chunk_id} sentinel={SENTINEL_SOURCE_LINE}",
            }

        def process_reduction_node(self, request):
            self.reduction_calls += 1
            return {"narrative": f"reduced sentinel={SENTINEL_PROMPT_FRAGMENT}"}

        def synthesize_divided_file(self, request, digest, manifest_json):
            self.synthesis_calls += 1
            return {"description": f"final sentinel={SENTINEL_REQUEST_BODY}"}

    def _build(self, tmp_path, *, max_content_chars=2000):
        source = "\n".join(f"value_{i} = {i}" for i in range(220)) + "\n"
        request = make_execution_request(
            tmp_path, "src/large.py", source, max_content_chars=max_content_chars
        )
        plan = build_division_plan(
            rel_path=request.rel_path,
            language=request.language,
            content=source,
            source_budget_chars=max_content_chars,
        )
        tree = build_reduction_tree(plan, max_content_chars=max_content_chars)
        assert len(plan.chunks) >= 2, "fixture must produce more than one leaf"
        return request, plan, tree

    def test_bounded_event_per_node_in_order_with_no_raw_payload(self, tmp_path, caplog):
        from codedoc.utils.logger import set_level

        request, plan, tree = self._build(tmp_path)
        writer = SafeWriter(
            tmp_path / "docs" / "crash_recovery.json",
            "json",
            None,
            {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
        )
        orchestrator = self._StubOrchestrator()

        set_level("DEBUG")
        try:
            with caplog.at_level(logging.DEBUG, logger="codedoc.core.execution"):
                result = _process_divided_file(
                    request, plan, tree, "test-provider", orchestrator, writer
                )
        finally:
            set_level("INFO")

        events = [r.getMessage() for r in caplog.records]
        leaf_events = [e for e in events if "category=split-leaf" in e]
        reduction_events = [e for e in events if "category=split-reduction" in e]
        final_events = [e for e in events if "category=split-final" in e]

        assert len(leaf_events) == len(plan.chunks)
        assert len(final_events) == 1
        assert len(reduction_events) >= 1

        # In order: leaf ordinals strictly increasing, one per chunk.
        leaf_ordinals = []
        for event in leaf_events:
            ordinal_part = event.split("ordinal=")[1].split()[0]
            leaf_ordinals.append(int(ordinal_part.split("/")[0]))
        assert leaf_ordinals == sorted(leaf_ordinals)
        assert len(set(leaf_ordinals)) == len(plan.chunks)

        # Every event names "computed" (nothing was reused on a fresh run).
        assert all("computed" in e for e in leaf_events + reduction_events + final_events)

        # No raw chunk payload, prompt, or response text reached the log
        # stream. (`result` itself legitimately contains the stub's
        # sentinel-bearing description -- that is the published
        # *documentation*, not a log leak, so it is deliberately not scanned.)
        assert_no_sentinels_leaked(caplog.text)
        assert result is not None

    def test_reused_node_events_say_reused_not_computed(self, tmp_path, caplog):
        from codedoc.utils.logger import set_level

        request, plan, tree = self._build(tmp_path)
        writer = SafeWriter(
            tmp_path / "docs" / "crash_recovery.json",
            "json",
            None,
            {"src/large.py": {"path": tmp_path / "src" / "large.py"}},
        )
        first_pass = self._StubOrchestrator()
        set_level("DEBUG")
        try:
            _process_divided_file(request, plan, tree, "test-provider", first_pass, writer)

            second_pass = self._StubOrchestrator()
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger="codedoc.core.execution"):
                _process_divided_file(
                    request, plan, tree, "test-provider", second_pass, writer
                )
        finally:
            set_level("INFO")

        # The second pass is fully recovered: it must log every node as
        # reused and must never call the (stub) provider again.
        assert second_pass.leaf_calls == 0
        assert second_pass.reduction_calls == 0
        assert second_pass.synthesis_calls == 0
        events = [r.getMessage() for r in caplog.records]
        split_events = [e for e in events if e.startswith("Split node ")]
        assert split_events, "expected at least one bounded split-node event"
        assert all("reused" in e for e in split_events)
