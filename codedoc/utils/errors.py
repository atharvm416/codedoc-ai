"""Shared exceptions and bounded in-memory issue reporting.

Issues are collected in memory only — CodeDoc never writes a persistent
``error.log``.  Bounded diagnostics are printed to the terminal and embedded in
the final output (and preserved in ``crash_recovery.json`` while a run is
interrupted).  Error counters include only errors; issue counters include both.
"""

from __future__ import annotations

import traceback

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

    This is the only error type that aborts the per-file loop on a
    *provider* fault.  It is raised exclusively by ``codedoc.core.execution`` so
    that it is distinguishable from an ordinary ``AgentError`` / ``LLMError`` that
    may legitimately appear in an exception chain.  Every stop it represents is
    *safe*: crash_recovery.json is left intact and resumable; no stop path
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
            forbidden/permission abort.  Setup/credentials class
            → CLI exit code 2.
        ``"rate_limit_exhausted"``
            The bounded zero-progress rate-limit / quota stop.  A
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


class PromptCustomizationValidationError(ConfigError):
    """Raised when a prompt-customization profile is rejected.

    Every outcome is non-overridable and stops the run:

    - a deterministic schema/type/bound/rendering failure;
    - a ``TOO_RISKY`` standards/safety verdict (there is no bypass);
    - a fail-closed malformed / empty / ambiguous / transport-failed / unknown
      verdict, or a batch-contract mismatch.

    Subclasses :class:`ConfigError` so the CLI maps it to exit code 2 (a
    setup-class problem the user must correct) without special-casing.  It is
    raised before any source file is processed and before any crash-recovery,
    crash-recovery, or output artifact is written, so the run leaves nothing behind.
    """

    def __init__(self, message: str, *, stats: dict | None = None) -> None:
        super().__init__(message)
        self.stats = dict(stats or {})


class AgentError(CodeDocError):
    """Raised when an agent fails to produce valid output."""

    def __init__(self, agent_name: str, file_path: str, reason: str):
        self.agent_name = agent_name
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"AgentError [{agent_name}] on '{file_path}': {reason}")


class ResponseContractError(AgentError):
    """Raised when a provider response fails the deterministic response contract.

    A ``ResponseContractError`` is a specialized :class:`AgentError` carrying a
    bounded :class:`~codedoc.agents.response_diagnostics.ResponseDiagnostic` and a
    flag recording whether a targeted correction call was attempted.  Every
    ``ResponseContractError`` is *non-retryable*: a deterministic
    response-contract rejection never silently becomes a duplicate whole-file
    provider call, whether correction is disabled or has already been attempted.

    ``diagnostic`` is bounded and carries no source, prompt, credential, or
    full-response text.  Existing generic error reporting still receives the
    concise :class:`AgentError` message string produced by ``super().__init__``.
    """

    def __init__(
        self,
        agent_name: str,
        file_path: str,
        reason: str,
        *,
        diagnostic,
        correction_attempted: bool,
    ):
        self.diagnostic = diagnostic
        self.correction_attempted = correction_attempted
        super().__init__(agent_name, file_path, reason)


class OutputError(CodeDocError):
    """Raised when writing output files fails."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"OutputError for '{file_path}': {reason}")


class LiveBackupWriteError(OutputError):
    """Raised when the live crash-safety backup cannot be persisted.

    This is a fatal output failure, not a recoverable agent or rate-limit
    failure: once crash_recovery.json cannot be written, codedoc's recovery
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
    Collects issues during a pipeline run in memory only.

    CodeDoc never writes a persistent ``error.log``.  Bounded diagnostics are
    printed to the terminal and hard-error summaries are embedded in the final
    output; while a run is interrupted, completed work and context live in
    ``crash_recovery.json``.

    Severity levels
    ---------------
    ``"error"``
        Hard failure — shown in ``summary()`` and therefore included in the
        final ``codedoc.json`` ``errors`` field and Markdown ``## Errors``
        section.  Also counted by ``has_errors()`` and ``error_count()``.
    ``"warning"``
        Recovered issue (e.g. a rate-limit that was retried successfully).
        Excluded from ``summary()`` so the clean final output is not alarmed.
    """

    # Bound the in-memory entry list so a pathological run cannot grow it without
    # limit; the summary/display is derived from these bounded entries.
    _MAX_ENTRIES = 1000

    def __init__(self) -> None:
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
        if len(self._entries) >= self._MAX_ENTRIES:
            return
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
