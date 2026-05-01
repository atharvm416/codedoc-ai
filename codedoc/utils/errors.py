"""
Custom exceptions and error reporter for codedoc.
All modules raise these instead of bare exceptions so the pipeline
can catch, log, and report them uniformly.
"""

from __future__ import annotations

import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CodeDocError(Exception):
    """Base class for all codedoc errors."""


class ParseError(CodeDocError):
    """Raised when a file cannot be parsed for imports."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"ParseError in '{file_path}': {reason}")


class LLMError(CodeDocError):
    """Raised when an LLM call fails or returns invalid output."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"LLMError [{provider}]: {reason}")


class ConfigError(CodeDocError):
    """Raised when the config file is missing, invalid, or incomplete."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"ConfigError: {reason}")


class AgentError(CodeDocError):
    """Raised when an agent fails to produce valid output."""

    def __init__(self, agent_name: str, file_path: str, reason: str):
        self.agent_name = agent_name
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"AgentError [{agent_name}] on '{file_path}': {reason}")


class OutputError(CodeDocError):
    """Raised when writing output files fails."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"OutputError for '{file_path}': {reason}")


# ---------------------------------------------------------------------------
# Error Reporter
# ---------------------------------------------------------------------------

class ErrorReporter:
    """
    Collects errors during a pipeline run and writes a summary
    error.log in the project root when the run finishes.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._errors: list[dict] = []

    def record(self, error: Exception, context: str = "") -> None:
        """Record an error without stopping execution."""
        entry = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
        }
        self._errors.append(entry)

    def has_errors(self) -> bool:
        return len(self._errors) > 0

    def error_count(self) -> int:
        return len(self._errors)

    def flush(self) -> None:
        """Write all recorded errors to error.log."""
        if not self._errors:
            return
        lines = [f"codedoc error log — {len(self._errors)} error(s)\n", "=" * 60 + "\n"]
        for i, e in enumerate(self._errors, 1):
            lines.append(f"\n[{i}] {e['type']}: {e['message']}\n")
            if e["context"]:
                lines.append(f"    Context: {e['context']}\n")
            lines.append(f"    Traceback:\n{e['traceback']}\n")
            lines.append("-" * 60 + "\n")
        self.log_path.write_text("".join(lines), encoding="utf-8")

    def summary(self) -> str:
        if not self._errors:
            return "No errors."
        return "\n".join(f"  - [{e['type']}] {e['message']}" for e in self._errors)