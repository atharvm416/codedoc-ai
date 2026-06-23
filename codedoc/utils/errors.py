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
# error.log ownership markers (0.10.1)
# ---------------------------------------------------------------------------
# A CodeDoc-owned log starts with one of these ASCII markers on its first line.
# The new marker is written by ``ErrorReporter.flush()``; the legacy prefix is
# recognized for safe cleanup of logs created before 0.10.1.  A file whose first
# line matches neither is foreign and is never deleted, truncated, or replaced.
LOG_OWNERSHIP_MARKER = "# codedoc-ai issue log"
_LEGACY_LOG_MARKER_PREFIX = "codedoc issue log"


def is_codedoc_owned_log(path: Path) -> bool:
    """Return True only if *path* is a recognized CodeDoc-owned ``error.log``.

    Reads just the first line.  A missing or unreadable file is treated as
    not-owned (the caller must never delete it).  Both the new ``# codedoc-ai
    issue log`` marker and the legacy ``codedoc issue log`` header are accepted.
    """
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            first_line = handle.readline()
    except (OSError, UnicodeDecodeError):
        return False
    stripped = first_line.lstrip("﻿").strip()
    return stripped.startswith(LOG_OWNERSHIP_MARKER) or stripped.startswith(
        _LEGACY_LOG_MARKER_PREFIX
    )


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


class UnrecoverableProviderError(LLMError):
    """Raised at the execution layer when a provider error cannot recover by
    retrying, so the run is stopped immediately to save tokens, money, and time.

    0.9.7 — this is the only error type that aborts the per-file loop on a
    *provider* fault.  It is raised exclusively by ``codedoc.core.execution`` so
    that it is distinguishable from an ordinary ``AgentError`` / ``LLMError`` that
    may legitimately appear in an exception chain.  Every stop it represents is
    *safe*: the live JSON backup is left intact and resumable; no stop path
    deletes the backup or overwrites it with a "complete" final output.

    Parameters
    ----------
    provider:
        The provider name (e.g. ``"openai"``), forwarded to :class:`LLMError`.
    reason:
        A human-readable message naming the likely cause class (billing/credit,
        credentials, model name, access, or persistent rate limit).  It must not
        invent specifics that are absent from the original error.
    category:
        One of two stable values so the CLI can pick the exit code without
        re-parsing the message:

        ``"terminal"``
            A confirmed billing/credit, credentials, unknown-model, or
            forbidden/permission abort (Workstream B).  Setup/credentials class
            → CLI exit code 2.
        ``"rate_limit_exhausted"``
            The bounded zero-progress rate-limit / quota stop (Workstream C).  A
            transient "retry later" condition, not a credentials fault → CLI
            exit code 1.

    The original SDK/agent exception is retained as ``__cause__`` (callers raise
    ``... from exc``) so diagnostics are preserved.
    """

    def __init__(self, provider: str, reason: str, category: str):
        if category not in {"terminal", "rate_limit_exhausted"}:
            raise ValueError(f"Unsupported unrecoverable-provider category: {category!r}")
        self.category = category
        super().__init__(provider, reason)


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
        self._last_flush_persisted: bool = False

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

    def flush(self) -> bool:
        """Atomically write all recorded issues to the log file.

        Creates the log file's parent directory if needed (output_dir may
        not exist yet when flush is called on an early-exit code path).
        Returns ``True`` when a CodeDoc-owned log was actually persisted, and
        ``False`` when there were no issues or a foreign file occupied the log
        path.

        0.10.1: the log begins with a stable ASCII ownership marker so a later
        successful run can recognize and clear it.  When a *foreign* file already
        occupies the path it is never replaced — its bytes are left intact and a
        warning is logged instead of raising.  Writing is routed through the
        canonical atomic-write helper so a stale owned log is replaced in one
        atomic rename rather than appended to or truncated in place.
        """
        self._last_flush_persisted = False
        if not self._entries:
            return False
        if self.log_path.exists() and not is_codedoc_owned_log(self.log_path):
            # A foreign file sits at the log path: never overwrite it.
            from codedoc.utils.logger import get_logger

            get_logger(__name__).warning(
                "A foreign file occupies the issue-log path '%s'; codedoc will "
                "not overwrite it. %d issue(s) were not persisted to disk.",
                self.log_path,
                len(self._entries),
            )
            return False
        lines = [
            f"{LOG_OWNERSHIP_MARKER} — {len(self._entries)} issue(s)\n",
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
        # Function-local import avoids a module-load cycle (block_manager imports
        # io_diagnostics which imports nothing from this module, but routing the
        # canonical writer lazily keeps errors.py free of core imports at load).
        from codedoc.core.block_manager import atomic_write_text

        atomic_write_text(self.log_path, "".join(lines))
        self._last_flush_persisted = True
        return True

    def has_persisted_log(self) -> bool:
        """True when the last successful ``flush()`` wrote a CodeDoc-owned log."""
        return self._last_flush_persisted and self.log_path.exists() and is_codedoc_owned_log(
            self.log_path
        )

    def clear_stale_owned_log(self) -> str | None:
        """Remove a stale CodeDoc-owned ``error.log`` after a clean, issue-free run.

        Returns ``None`` when nothing needed removal or removal succeeded, or a
        short warning string when an owned log could not be removed.  A foreign
        file is left byte-identical and reported as ``None`` (not our file).  A
        missing file is a no-op.  Never raises — a stale-log removal failure is an
        auxiliary warning, not a failure of otherwise valid documentation output.
        """
        if self._entries:
            return None
        if not self.log_path.exists():
            return None
        if not is_codedoc_owned_log(self.log_path):
            return None
        try:
            self.log_path.unlink()
        except OSError as exc:
            return (
                f"a stale codedoc issue log '{self.log_path.name}' could not be "
                f"removed: {exc}"
            )
        return None

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
