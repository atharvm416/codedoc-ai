"""0.12.2 deterministic source pre-check and classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import codedoc.core.execution as execution
from codedoc.core.error_classifier import (
    _classify_failure,
    _find_insufficient_source_error,
)
from codedoc.core.source_precheck import insufficient_source
from codedoc.utils.errors import InsufficientSourceError


@pytest.mark.parametrize(
    "content",
    ["", "   ", "\n\t\r\n", "\ufeff"],
)
def test_empty_or_whitespace_source_is_insufficient(content):
    assert insufficient_source(content) == (True, "empty_or_whitespace_only")


@pytest.mark.parametrize("content", ["x", "\ufffd"])
def test_non_whitespace_source_is_not_insufficient(content):
    assert insufficient_source(content) == (False, "")


def test_find_insufficient_source_error_walks_bare_cause_and_context():
    typed = InsufficientSourceError("m.py", "empty_or_whitespace_only")
    assert _find_insufficient_source_error(typed) is typed

    cause_wrapper = RuntimeError("outer")
    cause_wrapper.__cause__ = typed
    assert _find_insufficient_source_error(cause_wrapper) is typed

    context_wrapper = RuntimeError("outer")
    context_wrapper.__context__ = typed
    assert _find_insufficient_source_error(context_wrapper) is typed
    assert _find_insufficient_source_error(ValueError("unrelated")) is None


def test_insufficient_source_precedes_rate_limit_path_signals():
    exc = InsufficientSourceError("tpm/quota.py", "empty_or_whitespace_only")
    assert _classify_failure(exc, None) == "insufficient_source"


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_process_one_file_skips_before_parser_and_orchestrator(
    tmp_path, monkeypatch, content
):
    path = tmp_path / "empty.py"
    path.write_text(content, encoding="utf-8")
    calls = {"parse": 0, "process": 0}

    def fake_parse(_descriptor):
        calls["parse"] += 1
        return []

    class FakeOrchestrator:
        llm = SimpleNamespace(provider_name="fake")

        def process(self, descriptor, source, imports):
            calls["process"] += 1
            return {}

    monkeypatch.setattr(execution, "parse_file", fake_parse)
    with pytest.raises(InsufficientSourceError):
        execution._process_one_file(
            {
                "path": path,
                "rel_path": "empty.py",
                "language": "python",
                "extension": ".py",
            },
            FakeOrchestrator(),
        )
    assert calls == {"parse": 0, "process": 0}


def test_process_one_file_allows_minimal_non_whitespace_source(tmp_path, monkeypatch):
    path = tmp_path / "minimal.py"
    path.write_text("x", encoding="utf-8")
    calls = {"parse": 0, "process": 0}

    def fake_parse(_descriptor):
        calls["parse"] += 1
        return []

    class FakeOrchestrator:
        llm = SimpleNamespace(provider_name="fake")

        def process(self, descriptor, source, imports):
            calls["process"] += 1
            return {"description": "minimal"}

    monkeypatch.setattr(execution, "parse_file", fake_parse)
    result = execution._process_one_file(
        {
            "path": path,
            "rel_path": "minimal.py",
            "language": "python",
            "extension": ".py",
        },
        FakeOrchestrator(),
    )
    assert result == {"description": "minimal"}
    assert calls == {"parse": 1, "process": 1}
