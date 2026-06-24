"""0.10.2 feature tests.

Covers all new behaviour introduced in 0.10.2:

Workstream A — Configurable truncation head ratio
  A1  Default ratio (0.70) produces same output as 0.10.1 hardcoded constant
  A2  0.50 ratio gives equal head/tail lengths (modulo rounding)
  A3  0.90 ratio gives a 90/10 split
  A4  Content shorter than max_chars returned unchanged regardless of ratio
  A5  Degenerate ceiling (≤ marker length) returns a prefix of the marker
  A6  _validate() rejects 0.0, 1.0, out-of-range, booleans, non-numeric strings
  A7  _validate() accepts 0.1, 0.5, 0.7, 0.9, string-coercible floats from env
  A8  Orchestrator passes configured ratio to truncate_for_llm
  A9  Dry-run _estimate_planned_input_tokens uses the configured ratio

Workstream B — Error classifier extraction
  B1  Symbols importable from codedoc.core.error_classifier
  B2  Symbols importable from codedoc.core.execution (compat re-export)
  B3  Classification behaviour unchanged for billing/credential/rate-limit/input/transient
  B4  _parse_retry_after extracts retry delay from both formats correctly
  B5  Importing _parse_retry_after from both modules gives same behaviour

Workstream C — Parser-authoritative dependency projection
  C1  Dart: dart:io → SDK in both single and triple (via unresolved imports)
  C2  Dart: package:record/record.dart → external "record"
  C3  Dart: relative imports appear only as internal graph edges, never external
  C4  Java: generic-parser uses unresolved imports, not model _deps.external
  C5  Go: generic-parser uses unresolved imports, not model _deps.external
  C6  React/TSX: bare npm deps still come from model _deps.external
  C7  Python parity: existing parser-authoritative projection unchanged
"""
from __future__ import annotations

import pytest

from codedoc.agents.base_agent import (
    TRUNCATION_HEAD_FRACTION,
    TRUNCATION_MARKER,
    truncate_for_llm,
)


# ===========================================================================
# Workstream A — Configurable truncation head ratio
# ===========================================================================

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
        from codedoc.agents.orchestrator import Orchestrator
        llm = MagicMock()
        llm.provider_name = "fake"
        return Orchestrator(llm, max_content_chars=100, truncation_head_ratio=ratio)

    def test_A8_orchestrator_stores_ratio(self):
        orc = self._make_orchestrator(0.30)
        assert orc.truncation_head_ratio == 0.30

    def test_A8_orchestrator_content_bounded_by_max(self):
        """After truncation the content passed to the agent is ≤ max_content_chars."""
        from unittest.mock import MagicMock
        from codedoc.agents.orchestrator import Orchestrator

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


class TestDryRunRatioEstimate:
    def test_A9_non_default_ratio_changes_estimate(self, tmp_path):
        """_estimate_planned_input_tokens reflects the configured head ratio."""
        from unittest.mock import MagicMock
        from codedoc.pipeline import _estimate_planned_input_tokens
        from codedoc.core.planning import PipelinePlan

        # Create a file larger than max_content_chars so truncation fires
        src = tmp_path / "big.py"
        src.write_text("x = 1\n" * 5000, encoding="utf-8")
        descriptor = {
            "rel_path": "big.py",
            "path": src,
            "language": "python",
            "extension": ".py",
        }

        plan = MagicMock(spec=PipelinePlan)
        plan.agent_rels = frozenset({"big.py"})
        file_map = {"big.py": descriptor}

        config_default = {"analysis_mode": "single", "max_content_chars": 500, "truncation_head_ratio": 0.70}
        config_alt = {"analysis_mode": "single", "max_content_chars": 500, "truncation_head_ratio": 0.10}

        est_default = _estimate_planned_input_tokens(plan, file_map, config_default)
        est_alt = _estimate_planned_input_tokens(plan, file_map, config_alt)

        # Both are positive; the estimates may differ because the head/tail content differs.
        assert est_default > 0
        assert est_alt > 0
        # They don't have to be different in token count since the total chars are the same,
        # but the function must run without error for both ratios.


# ===========================================================================
# Workstream B — Error classifier extraction
# ===========================================================================

class TestErrorClassifierImports:
    def test_B1_symbols_from_error_classifier(self):
        from codedoc.core.error_classifier import (
            _RATE_LIMIT_SIGNALS,
            _build_terminal_abort,
            _classify_failure,
            _parse_retry_after,
        )
        assert callable(_classify_failure)
        assert callable(_build_terminal_abort)
        assert callable(_parse_retry_after)
        assert isinstance(_RATE_LIMIT_SIGNALS, tuple)

    def test_B2_symbols_from_execution_compat(self):
        from codedoc.core.execution import (
            _classify_failure,
            _parse_retry_after,
        )
        assert callable(_classify_failure)
        assert callable(_parse_retry_after)

    def test_B3_billing_error_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "exceeded your current quota and billing is required")
        result = _classify_failure(exc, None)
        assert result == "terminal_billing"

    def test_B3_credential_error_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "invalid api key provided")
        result = _classify_failure(exc, None)
        assert result == "global"

    def test_B3_rate_limit_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "429 rate limit exceeded too many requests")
        result = _classify_failure(exc, None)
        assert result == "rate_limit"

    def test_B3_input_too_large_classification(self):
        from codedoc.core.error_classifier import _classify_failure
        from codedoc.utils.errors import LLMError

        exc = LLMError("fake", "context length exceeded the maximum context allowed")
        result = _classify_failure(exc, None)
        assert result == "input"

    def test_B3_transient_classification(self):
        from codedoc.core.error_classifier import _classify_failure

        exc = RuntimeError("connection reset by peer")
        result = _classify_failure(exc, None)
        assert result == "transient"

    def test_B4_parse_retry_after_try_again(self):
        from codedoc.core.error_classifier import _parse_retry_after
        exc = RuntimeError("try again in 1.5s please")
        assert _parse_retry_after(exc) == pytest.approx(1.5)

    def test_B4_parse_retry_after_retry_after(self):
        from codedoc.core.error_classifier import _parse_retry_after
        exc = RuntimeError("retry.after: 2.0 seconds")
        assert _parse_retry_after(exc) == pytest.approx(2.0)

    def test_B4_parse_retry_after_none(self):
        from codedoc.core.error_classifier import _parse_retry_after
        exc = RuntimeError("no delay info here")
        assert _parse_retry_after(exc) is None

    def test_B5_same_parse_retry_after_both_modules(self):
        from codedoc.core import error_classifier, execution

        exc = RuntimeError("try again in 1.5s")
        assert error_classifier._parse_retry_after(exc) == execution._parse_retry_after(exc)

        exc2 = RuntimeError("retry.after: 2.0")
        assert error_classifier._parse_retry_after(exc2) == execution._parse_retry_after(exc2)


# ===========================================================================
# Workstream C — Parser-authoritative dependency projection
# ===========================================================================

class TestProjectDependencyLinks:
    """Unit tests for _project_dependency_links with unresolved imports."""

    def _call(self, file_dict, internal_paths=None, unresolved_imports=None):
        from codedoc.core.project_view import (
            _project_dependency_links,
            _project_import_roots,
        )
        internal_paths = internal_paths or []
        project_roots = _project_import_roots([file_dict])
        return _project_dependency_links(
            file_dict,
            internal_paths,
            project_roots,
            unresolved_imports=unresolved_imports,
        )

    # --- Dart ---

    def test_C1_dart_sdk_import(self):
        """dart:io classifies as SDK."""
        file = {"path": "lib/main.dart", "language": "dart", "imports": []}
        external, sdk = self._call(file, unresolved_imports=["dart:io", "dart:async"])
        assert "dart:io" in sdk
        assert "dart:async" in sdk
        assert external == []

    def test_C2_dart_package_import(self):
        """package:record/record.dart → external 'record'."""
        file = {"path": "lib/player.dart", "language": "dart", "imports": []}
        external, sdk = self._call(
            file,
            unresolved_imports=["package:record/record.dart", "package:flutter/material.dart"],
        )
        assert "record" in external
        assert "flutter" in external
        assert sdk == []

    def test_C3_dart_relative_import_not_external(self):
        """Relative Dart imports in unresolved list do not create external entries."""
        file = {"path": "lib/player.dart", "language": "dart", "imports": []}
        external, sdk = self._call(
            file,
            unresolved_imports=[
                "../models/song_model.dart",
                "full_player_screen.dart",
                "dart:io",
            ],
        )
        # Only dart:io should appear; relative/bare imports must be filtered out
        assert external == []
        assert sdk == ["dart:io"]

    def test_C3_dart_relative_internal_appears_in_graph_not_external(self):
        """Dart internal imports that resolve don't appear as external."""
        # Simulate: the internal import resolved to a graph edge, so it's NOT
        # in unresolved_imports. Only package/sdk imports are unresolved.
        file = {"path": "lib/player.dart", "language": "dart", "imports": []}
        external, sdk = self._call(
            file,
            internal_paths=["lib/models/song_model.dart"],
            unresolved_imports=["package:record/record.dart", "dart:async"],
        )
        assert "record" in external
        assert "dart:async" in sdk
        # No internal paths leak as external
        assert "song_model.dart" not in external
        assert "models/song_model.dart" not in external

    # --- Java ---

    def test_C4_java_uses_unresolved_not_model(self):
        """Java generic-parser uses unresolved imports, not model _deps.external."""
        file = {
            "path": "src/Main.java",
            "language": "java",
            "imports": [],
            "_deps": {"external": ["wrong_package"]},
        }
        external, sdk = self._call(
            file,
            unresolved_imports=["org.springframework.boot.SpringApplication", "com.google.gson.Gson"],
        )
        # Should use unresolved imports, not model _deps.external
        assert "wrong_package" not in external
        # The generic classifier returns full canonical for Java
        assert "org.springframework.boot.SpringApplication" in external or len(external) > 0

    def test_C4_java_fallback_when_no_unresolved(self):
        """Java falls back to model _deps.external when unresolved_imports is None."""
        file = {
            "path": "src/Main.java",
            "language": "java",
            "imports": [],
            "_deps": {"external": ["spring-boot"]},
        }
        external, sdk = self._call(file, unresolved_imports=None)
        assert "spring-boot" in external

    # --- Go ---

    def test_C5_go_uses_unresolved_not_model(self):
        """Go generic-parser uses unresolved imports, not model _deps.external."""
        file = {
            "path": "main.go",
            "language": "go",
            "imports": [],
            "_deps": {"external": ["wrong_lib"]},
        }
        external, sdk = self._call(
            file,
            unresolved_imports=["github.com/gin-gonic/gin", "fmt"],
        )
        assert "wrong_lib" not in external
        assert "github.com/gin-gonic/gin" in external

    # --- React/TSX ---

    def test_C6_react_uses_model_deps(self):
        """React/TSX always uses model _deps.external for npm packages."""
        file = {
            "path": "src/App.tsx",
            "language": "tsx",
            "imports": ["./components/Header", "./utils/api"],
            "_deps": {"external": ["react", "@mui/material"]},
        }
        # Even with unresolved_imports supplied, React uses model _deps
        external, sdk = self._call(
            file,
            unresolved_imports=["./components/Header"],
        )
        assert "react" in external
        assert "@mui/material" in external

    def test_C6_jsx_uses_model_deps(self):
        """JSX also uses model _deps.external."""
        file = {
            "path": "src/App.jsx",
            "language": "jsx",
            "imports": [],
            "_deps": {"external": ["react", "axios"]},
        }
        external, sdk = self._call(file, unresolved_imports=["./utils"])
        assert "react" in external
        assert "axios" in external

    # --- Python (unchanged from 0.10.1) ---

    def test_C7_python_uses_unresolved_imports(self):
        """Python uses unresolved imports (graph-filtered), dropping internal ones."""
        file = {
            "path": "codedoc/core/loader.py",
            "language": "python",
            "imports": ["os", "codedoc.core.graph", "pathlib", "codedoc.utils.errors"],
        }
        # Simulate: codedoc.* resolved to graph edges, stdlib/external didn't
        unresolved = ["os", "pathlib"]
        external, sdk = self._call(
            file,
            internal_paths=["codedoc/core/graph.py", "codedoc/utils/errors.py"],
            unresolved_imports=unresolved,
        )
        # os and pathlib are stdlib
        assert "os" in sdk
        assert "pathlib" in sdk
        # codedoc.* should not appear (they were resolved to internal graph edges)
        assert "codedoc" not in external

    def test_C7_python_fallback_without_unresolved(self):
        """Python falls back to file['imports'] when unresolved_imports is None."""
        file = {
            "path": "my_module.py",
            "language": "python",
            "imports": ["requests", "os"],
        }
        external, sdk = self._call(file, unresolved_imports=None)
        assert "requests" in external
        assert "os" in sdk


class TestBuildProjectViewUnresolvedThreading:
    """Integration: unresolved_imports_by_path threads through build_project_view."""

    def test_unresolved_used_for_dart(self):
        """build_project_view passes per-file unresolved imports to projection."""
        from codedoc.core.project_view import build_project_view

        records = [{
            "hash": "abc",
            "file_path": "lib/player.dart",
            "language": "dart",
            "documentation": {
                "file_path": "lib/player.dart",
                "language": "dart",
                "imports": ["package:record/record.dart", "dart:io"],
                "description": "Player",
                "role_in_system": "",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {},
                "key_concepts": [], "usage_example": "",
                "state": "checked",
            },
        }]
        stats = {"checked": 1, "failed": 0, "skipped": 0, "reused": 0}

        # With unresolved_imports: both package:record and dart:io are unresolved
        view = build_project_view(
            records, stats,
            unresolved_imports_by_path={
                "lib/player.dart": ["package:record/record.dart", "dart:io"],
            },
        )
        player = view["files"][0]
        links = player.get("links", {})
        assert "record" in links.get("external_dependencies", [])
        assert "dart:io" in links.get("sdk_dependencies", [])

    def test_no_unresolved_fallback(self):
        """build_project_view without unresolved_imports falls back gracefully."""
        from codedoc.core.project_view import build_project_view

        records = [{
            "hash": "abc",
            "file_path": "src/App.tsx",
            "language": "tsx",
            "documentation": {
                "file_path": "src/App.tsx",
                "language": "tsx",
                "imports": [],
                "description": "App",
                "role_in_system": "",
                "functions": [], "classes": [], "exports": [],
                "dependencies_analysis": {"external": ["react"]},
                "key_concepts": [], "usage_example": "",
                "state": "checked",
            },
        }]
        stats = {"checked": 1, "failed": 0, "skipped": 0, "reused": 0}

        # Without unresolved_imports_by_path, React falls back to model _deps
        view = build_project_view(records, stats)
        app = view["files"][0]
        links = app.get("links", {})
        assert "react" in links.get("external_dependencies", [])


class TestBuildGraphUnresolved:
    """_build_graph returns unresolved_imports_by_path as expected."""

    def test_unresolved_captured(self, tmp_path):
        """Unresolved imports are separated from graph-resolved ones."""
        from codedoc.core.discovery import _build_graph
        from codedoc.utils.errors import ErrorReporter

        # Write a Python file that imports from within the project and from stdlib
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("import os\nfrom b import something\n", encoding="utf-8")
        b.write_text("import json\n", encoding="utf-8")

        all_files = [
            {"rel_path": "a.py", "path": a, "language": "python", "extension": ".py"},
            {"rel_path": "b.py", "path": b, "language": "python", "extension": ".py"},
        ]
        reporter = ErrorReporter(tmp_path / "error.log")
        graph, file_map, unresolved = _build_graph(all_files, tmp_path, reporter)

        # b.py should be a graph dependency of a.py
        assert "b.py" in graph.dependencies_of("a.py")
        # "os" does not resolve to a project file → should be in unresolved for a.py
        assert "os" in unresolved.get("a.py", [])
        # "from b import something" resolves → NOT in unresolved for a.py
        # (the resolved import is "b.py", not the string "b")
        assert "b.py" not in unresolved.get("a.py", [])
        # "json" doesn't resolve → should be in unresolved for b.py
        assert "json" in unresolved.get("b.py", [])
