"""Section 5.4: ``BaseAgent._safe_run``'s generic exception log boundary.

Matches the same two-tier rule ``_agent_error_result`` already applies to
the recorded result dict: a :class:`~codedoc.utils.errors.CodeDocError` is
already bounded by construction and renders unchanged via ``str(exc)``; only
a genuinely foreign exception is reduced through
:func:`~codedoc.utils.errors.bounded_exception_summary`. Calling the summary
unconditionally on a bare ``LLMError`` raised outside ``_call_llm_counted``
would discard its canonical reason and misreport it as ``"unknown-error"``.
"""

from __future__ import annotations

import logging

from codedoc.agents.base_agent import BaseAgent
from tests.support.provider_failures import provider_failure_error


class _RaisingAgent(BaseAgent):
    """A minimal BaseAgent subclass whose run() raises directly -- never
    through _call_llm_counted -- so _safe_run's generic except branch is
    genuinely exercised rather than the typed AgentError branch."""

    agent_name = "RaisingAgent"

    def __init__(self, llm, exc: Exception) -> None:
        super().__init__(llm)
        self._exc = exc

    def run(self, file_path, content, imports, language, requested_shape=None, **kwargs):
        raise self._exc


def test_bare_llm_error_generic_branch_shows_canonical_reason_not_unknown_error(caplog):
    """A bare LLMError (a CodeDocError) raised outside _call_llm_counted
    must render its canonical reason unchanged in both the recorded result
    and the log line -- never reduced to "unknown-error"."""
    envelope_exc = provider_failure_error("openai", "provider-quota-exhausted", status=429)
    agent = _RaisingAgent(llm=None, exc=envelope_exc)

    with caplog.at_level(logging.WARNING, logger="codedoc.agents.base_agent"):
        result = agent._safe_run("f.py", "content", [], "python")

    assert result["error"] == "LLMError [openai]: provider-quota-exhausted (429)"
    assert "provider-quota-exhausted (429)" in result["error"]
    assert "unknown-error" not in result["error"]

    log_text = "\n".join(r.message for r in caplog.records)
    assert "provider-quota-exhausted (429)" in log_text
    assert "unknown-error" not in log_text


def test_foreign_exception_generic_branch_is_still_reduced(caplog):
    """A genuinely foreign exception (not a CodeDocError) is unchanged by
    this fix -- it is still reduced through bounded_exception_summary,
    never rendered with its raw text."""
    agent = _RaisingAgent(llm=None, exc=RuntimeError("raw provider text with a secret"))

    with caplog.at_level(logging.WARNING, logger="codedoc.agents.base_agent"):
        result = agent._safe_run("f.py", "content", [], "python")

    assert result["error"] == "unknown-error"
    assert "raw provider text with a secret" not in result["error"]

    log_text = "\n".join(r.message for r in caplog.records)
    assert "unknown-error" in log_text
    assert "raw provider text with a secret" not in log_text
