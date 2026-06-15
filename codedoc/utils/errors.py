"""
Custom exceptions and error reporter for codedoc.
All modules raise these instead of bare exceptions so the pipeline
can catch, log, and report them uniformly.

0.8.0 changes
-------------
- ``ErrorReporter.record()`` now accepts a ``level`` parameter (``"error"`` or
  ``"warning"``).  Warning-level entries appear in ``error.log`` but are
  excluded from ``summary()`` so they never leak into the final ``codedoc.json``
  ``errors`` field or the Markdown ``## Errors`` section.
- ``has_errors()`` / ``error_count()`` count only ``"error"``-level entries.
- ``has_issues()`` / ``issue_count()`` count all entries.
- ``flush()`` header changed from ``error(s)`` to ``issue(s)`` to avoid alarm
  on warning-only logs.
- ``summary()`` returns ``""`` (empty string) when there are no error-level
  entries; callers that used to check ``!= "No errors."`` treat ``""`` as
  falsy and skip the errors field.
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


class ProviderInitError(ConfigError):
    """Raised when an LLM provider cannot be constructed (import, auth
    configuration, or SDK initialization failure).

    Subclasses :class:`ConfigError` so existing setup-error handling — and the
    CLI's exit-code 2 contract — applies without special-casing.
    """


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


class LiveBackupWriteError(OutputError):
    """Raised when the live crash-safety backup cannot be persisted.

    This is a fatal output failure, not a recoverable agent or rate-limit
    failure: once the live backup cannot be written, codedoc's crash-recovery
    guarantee no longer holds, so execution must stop scheduling new work rather
    than continue under a false guarantee.  Carries the target path only — never
    source, prompt, or credential data — and retains the original SDK/OS
    exception as ``__cause__``.
    """


# ---------------------------------------------------------------------------
# Error Reporter
# ---------------------------------------------------------------------------

class ErrorReporter:
    """
    Collects issues during a pipeline run and writes a summary to
    ``error.log`` in the output directory when the run finishes.

    Severity levels
    ---------------
    ``"error"``
        Hard failure — shown in ``summary()`` and therefore included in the
        final ``codedoc.json`` ``errors`` field and Markdown ``## Errors``
        section.  Also counted by ``has_errors()`` and ``error_count()``.
    ``"warning"``
        Recovered issue (e.g. a rate-limit that was retried successfully).
        Written to ``error.log`` for diagnostics but excluded from
        ``summary()`` so the clean final output is not alarmed.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._entries: list[dict] = []

    def record(self, error: Exception, context: str = "", level: str = "error") -> None:
        """Record an issue without stopping execution.

        Parameters
        ----------
        error:
            The exception to record.
        context:
            Human-readable description of where in the pipeline this occurred.
        level:
            ``"error"`` (default) for hard failures, ``"warning"`` for
            recovered issues that should not alarm the final output.
        """
        entry = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "level": level,
            "traceback": traceback.format_exc(),
        }
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # Counting helpers
    # ------------------------------------------------------------------

    def has_errors(self) -> bool:
        """True if any error-level (hard failure) entry was recorded."""
        return any(e["level"] == "error" for e in self._entries)

    def has_issues(self) -> bool:
        """True if any entry (any severity) was recorded."""
        return len(self._entries) > 0

    def error_count(self) -> int:
        """Number of error-level entries."""
        return sum(1 for e in self._entries if e["level"] == "error")

    def issue_count(self) -> int:
        """Total number of entries across all severity levels."""
        return len(self._entries)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all recorded issues to the log file.

        Creates the log file's parent directory if needed (output_dir may
        not exist yet when flush is called on an early-exit code path).
        Does nothing when no issues have been recorded.
        """
        if not self._entries:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"codedoc issue log — {len(self._entries)} issue(s)\n",
            "=" * 60 + "\n",
        ]
        for i, e in enumerate(self._entries, 1):
            level_label = e.get("level", "error").upper()
            lines.append(f"\n[{i}] [{level_label}] {e['type']}: {e['message']}\n")
            if e["context"]:
                lines.append(f"    Context: {e['context']}\n")
            if e.get("level", "error") == "error":
                lines.append(f"    Root cause:\n{e['traceback']}\n")
            lines.append("-" * 60 + "\n")
        self.log_path.write_text("".join(lines), encoding="utf-8")

    def summary(self) -> str:
        """Return a summary string for error-level entries only.

        This string is passed to ``write_project_outputs()`` and embedded in
        the final ``codedoc.json`` / Markdown output.  Returns ``""`` (empty
        string) when there are no hard errors so warning-only runs do not
        produce a scary ``## Errors`` section.
        """
        error_entries = [e for e in self._entries if e.get("level", "error") == "error"]
        if not error_entries:
            return ""
        return "\n".join(f"  - [{e['type']}] {e['message']}" for e in error_entries)
