"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.agent_fakes import mock_llm  # noqa: F401, F811

import logging
from pathlib import Path
import pytest
from tests.support.pipeline_usage import FakeProvider
from codedoc.agents.base_agent import (
    TRUNCATION_HEAD_FRACTION,
    TRUNCATION_MARKER,
    truncate_for_llm,
)
from codedoc.llm.base import LLMProvider
from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.db import source_char_count
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    MAX_CONTEXT_REVISION,
    expected_max_context_revision,
)

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
        with pytest.raises(ConfigError, match="integer"):
            load_config(Path("."), {"max_content_chars": "not-a-number"})
    def test_content_not_truncated_when_below_limit(self):
        from codedoc.agents.base_agent import BaseAgent

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
        from codedoc.llm.base import LLMProvider

        class DummyProvider(LLMProvider):
            provider_name = "dummy"
            def complete_json(self, *a, **kw): return "{}"
            def complete(self, *a, **kw): return ""

        orch = Orchestrator(DummyProvider(), max_content_chars=50000)
        assert orch._structure_agent._max_content_chars == 50000
        assert orch._dependency_agent._max_content_chars == 50000
        assert orch._doc_agent._max_content_chars == 50000

def test_orchestrator_truncates_once_and_all_agents_receive_same_text(caplog):

    # triple mode is the path that runs all three agents on the same text.
    orchestrator = Orchestrator(
        FakeProvider(), parallel=False, max_content_chars=1000, analysis_mode="triple"
    )
    seen: list[str] = []

    def structure(file_path, content, imports, language):
        seen.append(content)
        return {}

    def dependency(file_path, content, imports, language):
        seen.append(content)
        return {}

    def documentation(file_path, content, imports, language, structure, dependencies):
        seen.append(content)
        return {}

    orchestrator._structure_agent._safe_run = structure
    orchestrator._dependency_agent._safe_run = dependency
    orchestrator._doc_agent._safe_run_with_context = documentation

    with caplog.at_level(logging.WARNING, logger="codedoc.agents.orchestrator"):
        orchestrator.process(
            {"rel_path": "large.py", "language": "python", "extension": ".py"},
            "x" * 2000,
            [],
        )

    assert len(seen) == 3
    assert seen[0] == seen[1] == seen[2]
    # 0.10.1 (Workstream H): head-plus-tail truncation keeps a leading and a
    # trailing slice with the marker between them, all within the ceiling.
    assert len(seen[0]) == 1000
    assert "[truncated]" in seen[0]
    assert seen[0].startswith("x") and seen[0].endswith("x")
    assert caplog.text.count("Content truncated") == 1

class TestTruncateForLlm:
    def test_A1_default_ratio_matches_hardcoded(self):
        """Default ratio produces identical output to the 0.10.1 constant."""
        content = "A" * 500 + "B" * 500
        result_default = truncate_for_llm(content, 400)
        result_explicit = truncate_for_llm(content, 400, head_fraction=TRUNCATION_HEAD_FRACTION)
        assert result_default == result_explicit

    def test_A1_default_ratio_is_70(self):
        """The module constant is 0.70."""
        assert TRUNCATION_HEAD_FRACTION == 0.70

    def test_A2_half_ratio_equal_split(self):
        """0.50 ratio gives equal head/tail (within ±1 due to integer rounding)."""
        content = "X" * 10_000
        max_chars = 1000
        result = truncate_for_llm(content, max_chars, head_fraction=0.50)
        assert len(result) <= max_chars
        marker = TRUNCATION_MARKER
        assert marker in result
        before, after = result.split(marker, 1)
        assert abs(len(before) - len(after)) <= 1

    def test_A3_ninety_ratio_large_head(self):
        """0.90 ratio gives ~90/10 split."""
        content = "X" * 10_000
        max_chars = 1000
        result = truncate_for_llm(content, max_chars, head_fraction=0.90)
        assert len(result) <= max_chars
        marker = TRUNCATION_MARKER
        assert marker in result
        before, after = result.split(marker, 1)
        # head should be roughly 9x tail
        assert len(before) > len(after) * 5

    def test_A4_short_content_unchanged(self):
        """Content shorter than max_chars is returned unchanged."""
        content = "Hello world"
        for ratio in (0.1, 0.5, 0.7, 0.9):
            assert truncate_for_llm(content, 1000, head_fraction=ratio) == content

    def test_A5_degenerate_ceiling(self):
        """When max_chars ≤ marker length, a prefix of the marker is returned."""
        content = "X" * 1000
        marker = TRUNCATION_MARKER
        # ceiling smaller than marker
        result = truncate_for_llm(content, len(marker) - 1)
        assert len(result) <= len(marker) - 1
        assert marker.startswith(result)

    def test_A4_exact_ceiling_unchanged(self):
        """Content exactly at max_chars is returned unchanged."""
        content = "A" * 500
        assert truncate_for_llm(content, 500) == content

    def test_A4_one_over_ceiling_truncates(self):
        """Content one char over ceiling is truncated."""
        content = "A" * 501
        result = truncate_for_llm(content, 500)
        assert len(result) <= 500

class TestValidateRatio:
    def _load_with(self, overrides: dict):
        from pathlib import Path
        import tempfile
        from codedoc.core.loader import load_config
        with tempfile.TemporaryDirectory() as d:
            return load_config(Path(d), overrides)

    def test_A6_reject_zero(self):
        from codedoc.utils.errors import ConfigError
        with pytest.raises(ConfigError, match="strictly between"):
            self._load_with({"truncation_head_ratio": 0.0})

    def test_A6_reject_one(self):
        from codedoc.utils.errors import ConfigError
        with pytest.raises(ConfigError, match="strictly between"):
            self._load_with({"truncation_head_ratio": 1.0})

    def test_A6_reject_negative(self):
        from codedoc.utils.errors import ConfigError
        with pytest.raises(ConfigError, match="strictly between"):
            self._load_with({"truncation_head_ratio": -0.1})

    def test_A6_reject_greater_than_one(self):
        from codedoc.utils.errors import ConfigError
        with pytest.raises(ConfigError, match="strictly between"):
            self._load_with({"truncation_head_ratio": 1.5})

    def test_A6_reject_boolean(self):
        from codedoc.utils.errors import ConfigError
        with pytest.raises(ConfigError, match="boolean"):
            self._load_with({"truncation_head_ratio": True})

    def test_A6_reject_non_numeric_string(self):
        from codedoc.utils.errors import ConfigError
        with pytest.raises(ConfigError):
            self._load_with({"truncation_head_ratio": "fast"})

    def test_A7_accept_valid_floats(self):
        for val in (0.1, 0.5, 0.7, 0.9):
            cfg = self._load_with({"truncation_head_ratio": val})
            assert abs(cfg["truncation_head_ratio"] - val) < 1e-9

    def test_A7_accept_string_coercible(self):
        cfg = self._load_with({"truncation_head_ratio": "0.65"})
        assert abs(cfg["truncation_head_ratio"] - 0.65) < 1e-9

    def test_A7_default_is_seventy(self):
        cfg = self._load_with({})
        assert abs(cfg["truncation_head_ratio"] - 0.70) < 1e-9

class TestOrchestratorRatio:
    def _make_orchestrator(self, ratio=0.70):
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.provider_name = "fake"
        return Orchestrator(llm, max_content_chars=100, truncation_head_ratio=ratio)

    def test_A8_orchestrator_stores_ratio(self):
        orc = self._make_orchestrator(0.30)
        assert orc.truncation_head_ratio == 0.30

    def test_A8_orchestrator_content_bounded_by_max(self):
        """After truncation the content passed to the agent is ≤ max_content_chars."""
        from unittest.mock import MagicMock

        received = {}

        def fake_safe_run(fp, content, imports, language):
            received["content"] = content
            return {
                "description": "x", "role_in_system": "y",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {}, "key_concepts": [],
                "usage_example": "",
            }

        llm = MagicMock()
        llm.provider_name = "fake"
        orc = Orchestrator(llm, max_content_chars=200, truncation_head_ratio=0.80)
        orc._file_agent._safe_run = fake_safe_run

        descriptor = {"rel_path": "a.py", "language": "python"}
        orc.process(descriptor, "X" * 10_000, [])
        assert len(received["content"]) <= 200

class TestBaseAgent:
    def test_truncates_long_content(self, mock_llm):
        from codedoc.agents.structure_agent import StructureAgent
        agent = StructureAgent(mock_llm)
        long_content = "x = 1\n" * 5000
        truncated = agent._truncate(long_content)
        assert len(truncated) <= 12_100  # slight buffer for the suffix

def test_short_content_is_unchanged():
    text = "def f():\n    return 1\n"
    assert truncate_for_llm(text, 12000) == text

def test_oversized_content_stays_within_ceiling_including_marker():
    text = "x" * 5000
    result = truncate_for_llm(text, 1000)
    assert len(result) == 1000
    assert TRUNCATION_MARKER in result

def test_head_and_tail_present_and_middle_absent():
    head = "HEADMARKER\n" + "a" * 2000
    middle = "MIDDLEUNIQUE\n" + "b" * 2000
    tail = "c" * 2000 + "\nclass LateClass:\n    pass\n"
    content = head + middle + tail
    result = truncate_for_llm(content, 1500)
    assert len(result) <= 1500
    assert "HEADMARKER" in result          # leading slice retained
    assert "LateClass" in result           # late definition now visible
    assert "MIDDLEUNIQUE" not in result    # omitted middle
    assert TRUNCATION_MARKER in result

def test_unicode_characters_are_not_split():
    # A run of multibyte characters around the cut must never raise or split a
    # code point (Python slices by code point).
    content = "é" * 5000
    result = truncate_for_llm(content, 1000)
    assert len(result) == 1000
    assert "�" not in result  # no replacement char from a broken cut

def test_degenerate_ceiling_smaller_than_marker():
    result = truncate_for_llm("x" * 100, len(TRUNCATION_MARKER) - 1)
    assert len(result) == len(TRUNCATION_MARKER) - 1

def test_truncation_is_identical_for_every_caller():
    # The single shared helper guarantees both modes and dry-run estimation see
    # exactly the same bounded text for the same input.
    content = "y" * 9000
    first = truncate_for_llm(content, 4000)
    second = truncate_for_llm(content, 4000)
    assert first == second
    assert len(first) == 4000

def test_revision_is_none_when_file_fits_ceiling():
    assert expected_max_context_revision(1000, max_chars=1000, head_ratio=0.70) is None
    assert expected_max_context_revision(999, max_chars=1000, head_ratio=0.70) is None

def test_revision_encodes_ceiling_and_head_ratio_when_truncated():
    assert (
        expected_max_context_revision(1001, max_chars=1000, head_ratio=0.70)
        == "truncate-v1:max=1000:head=0.7000"
    )
    # A different ceiling and ratio produce a different identity.
    assert (
        expected_max_context_revision(9999, max_chars=8000, head_ratio=0.85)
        == "truncate-v1:max=8000:head=0.8500"
    )

def test_revision_token_and_registry():
    assert MAX_CONTEXT_REVISION == "truncate-v1"
    assert "_max_context_revision" in CACHE_IDENTITY_KEYS

def test_source_char_count_short_circuits_small_files(tmp_path):
    p = tmp_path / "small.py"
    p.write_text("x = 1\n", encoding="utf-8")
    # Byte size (6) <= ceiling: returns a value <= ceiling, never above it.
    assert source_char_count(p, ceiling=1000) <= 1000

def test_source_char_count_exact_for_large_files(tmp_path):
    p = tmp_path / "big.py"
    p.write_text("x" * 2000, encoding="utf-8")
    assert source_char_count(p, ceiling=1000) == 2000
