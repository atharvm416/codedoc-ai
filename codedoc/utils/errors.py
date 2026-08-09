"""Shared exceptions and bounded in-memory issue reporting.

Issues are collected in memory only — CodeDoc never writes a persistent
``error.log``.  Bounded diagnostics are printed to the terminal and embedded in
the final output (and preserved in ``crash_recovery.json`` while a run is
interrupted).  Error counters include only errors; issue counters include both.

This module is also the canonical bounded error-projection owner
(:func:`bounded_exception_summary`).  Two tiers govern every rendering
boundary in this codebase:

1. **CodeDoc-owned exceptions are bounded by construction, not by rendering.**
   Every :class:`CodeDocError` subclass has a message CodeDoc composed itself
   (any foreign text was already passed through :func:`bounded_exception_summary`
   at its construction site).  A rendering boundary emits ``str(exc)``
   **unchanged** for these types.
2. **Everything else is reduced** through :func:`bounded_exception_summary`.

The rule is decidable from the type alone: never call
:func:`bounded_exception_summary` on a :class:`CodeDocError` instance, and
never skip it for a foreign exception being wrapped into a new CodeDoc error.
"""

from __future__ import annotations

from concurrent.futures import CancelledError

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


class InsufficientSourceError(CodeDocError):
    """Raised when a source file contains no non-whitespace content."""

    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"InsufficientSourceError in '{file_path}': {reason}")


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
    *safe*: crash_recovery.json is left intact. Compatible completed ordinary
    and split records may be reused, and compatible current schema-4 split node
    checkpoints may resume; forced, stale, identity-mismatched, legacy,
    foreign, or unsupported state is rerun or preserved and blocked according
    to the documented remedy. No stop path deletes the backup or overwrites it
    with a "complete" final output.

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
# Bounded exception rendering (section 3A)
# ---------------------------------------------------------------------------
# Frozen contract: bounded_exception_summary(exc) returns exactly
# "<reason_code>" or "<reason_code> (<detail>)" -- never an exception class
# name, module path, repr(), or chained-cause text.  Callers decide the tier:
# a CodeDocError's own str(exc) is never passed through this helper.

MAX_BOUNDED_SUMMARY_CHARS = 200

# Closed reason-code enum. No free text is ever emitted by
# bounded_exception_summary outside this set.
BOUNDED_EXCEPTION_REASON_CODES: frozenset[str] = frozenset(
    {
        "provider-request-failed",
        "provider-initialization-failed",
        "provider-authentication-rejected",
        "provider-rate-limited",
        "provider-quota-exhausted",
        "provider-model-unavailable",
        "provider-timeout",
        "provider-connection-failed",
        "provider-response-malformed",
        "filesystem-error",
        "cancelled",
        "unknown-error",
    }
)

# Frozen (module, class_name) -> reason-code table, resolved by walking the
# exception's MRO from most- to least-derived and taking the first match.
# Matching is by fully-qualified type name only: no provider SDK is imported
# here, so this table has zero import-time coupling to optional or
# not-yet-installed provider packages.  OpenAI and Anthropic mirror the same
# exception hierarchy (module == package name); google-genai only
# distinguishes client/server faults, so both map to the general
# request-failed code and rely on the numeric ``.code`` detail below to carry
# the HTTP status.
_PROVIDER_EXCEPTION_REASON_CODES: dict[tuple[str, str], str] = {
    ("openai", "AuthenticationError"): "provider-authentication-rejected",
    ("openai", "PermissionDeniedError"): "provider-authentication-rejected",
    ("openai", "RateLimitError"): "provider-rate-limited",
    ("openai", "APITimeoutError"): "provider-timeout",
    ("openai", "APIConnectionError"): "provider-connection-failed",
    ("openai", "NotFoundError"): "provider-model-unavailable",
    ("openai", "UnprocessableEntityError"): "provider-response-malformed",
    ("openai", "APIStatusError"): "provider-request-failed",
    ("openai", "APIError"): "provider-request-failed",
    ("anthropic", "AuthenticationError"): "provider-authentication-rejected",
    ("anthropic", "PermissionDeniedError"): "provider-authentication-rejected",
    ("anthropic", "RateLimitError"): "provider-rate-limited",
    ("anthropic", "APITimeoutError"): "provider-timeout",
    ("anthropic", "APIConnectionError"): "provider-connection-failed",
    ("anthropic", "NotFoundError"): "provider-model-unavailable",
    ("anthropic", "UnprocessableEntityError"): "provider-response-malformed",
    ("anthropic", "APIStatusError"): "provider-request-failed",
    ("anthropic", "APIError"): "provider-request-failed",
    ("google.genai.errors", "ClientError"): "provider-request-failed",
    ("google.genai.errors", "ServerError"): "provider-request-failed",
    ("google.genai.errors", "APIError"): "provider-request-failed",
}


def _reason_code_for(exc: BaseException) -> str:
    """Resolve the closed reason code for *exc*, by type alone."""
    if isinstance(exc, OSError):
        return "filesystem-error"
    if isinstance(exc, (KeyboardInterrupt, SystemExit, CancelledError)):
        return "cancelled"
    for cls in type(exc).__mro__:
        code = _PROVIDER_EXCEPTION_REASON_CODES.get((cls.__module__, cls.__name__))
        if code is not None:
            return code
    return "unknown-error"


def _bounded_detail(exc: BaseException) -> int | str | None:
    """Resolve the one permitted ``detail`` for *exc*, from a closed source
    list only.  Never calls ``str()``/``repr()`` on the exception or any of
    its attributes, and never traverses ``__cause__``/``__context__``."""
    if isinstance(exc, OSError):
        from codedoc.core.io_diagnostics import category_reason, classify_os_error

        return category_reason(classify_os_error(exc))
    for attr in ("status_code", "code", "retry_after"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def bounded_exception_summary(exc: BaseException) -> str:
    """Render *exc* as a stable, non-sensitive, bounded summary.

    Returns exactly ``"<reason_code>"`` or ``"<reason_code> (<detail>)"`` --
    never an exception class name, module path, ``repr()``, or chained-cause
    text.  *exc* must not be a :class:`CodeDocError`: those are bounded at
    construction, not at rendering, and every rendering boundary in this
    codebase emits their text unchanged instead of calling this helper.

    An exception whose type matches no row in the frozen mapping returns
    exactly ``"unknown-error"`` with no detail -- there is no fallback to
    ``str(exc)``, no partial rendering, and no best-effort path.
    """
    reason_code = _reason_code_for(exc)
    detail = (
        _bounded_detail(exc)
        if reason_code == "filesystem-error" or reason_code.startswith("provider-")
        else None
    )
    text = reason_code if detail is None else f"{reason_code} ({detail})"
    return text[:MAX_BOUNDED_SUMMARY_CHARS]


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
        message = (
            str(error)
            if isinstance(error, CodeDocError)
            else bounded_exception_summary(error)
        )
        entry = {
            "type": type(error).__name__,
            "message": message,
            "context": context,
            "level": level,
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
