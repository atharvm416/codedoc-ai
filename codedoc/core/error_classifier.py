"""Provider-error classification helpers.

Extracted from ``codedoc.core.execution`` as part of a structural decomposition.
This module
owns all signal-constant tuples, exception-chain walking, and the fixed failure
precedence classifier.  It has no dependency on execution state, the orchestrator,
or file-processing logic.

Import note
-----------
``_is_rate_limit_error`` is included here (not kept in execution.py) because
``_classify_failure`` depends on it and keeping both in the same module avoids
a circular import.  ``execution.py`` imports this entire module and re-exports
every symbol as a backward-compatibility shim.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from codedoc.utils.errors import (
    AgentError,
    InsufficientSourceError,
    LLMError,
    ResponseContractError,
    UnrecoverableProviderError,
)
from codedoc.utils.logger import get_logger

if TYPE_CHECKING:
    from codedoc.llm.rate_limit_profile import RateLimitProfile

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit signals (backward-compat fallback when no RateLimitProfile given)
# ---------------------------------------------------------------------------

_RATE_LIMIT_SIGNALS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "tokens per min",
    "tpm",
    "quota",
    "resource_exhausted",   # Gemini / Google AI
    "overloaded",           # Anthropic
    "529",                  # Anthropic overloaded
    "503",
)

# ---------------------------------------------------------------------------
# Terminal-billing / credential / access / model signals
# ---------------------------------------------------------------------------

_TERMINAL_BILLING_SIGNALS = (
    "insufficient_quota",
    "exceeded your current quota",
    "credit balance is too low",
    "billing is required",
    "billing not active",
    "billing is not active",
    "billing disabled",
    "payment required",
    "hard limit",
    "spending limit",
)

_CREDENTIAL_SIGNALS = (
    "invalid api key",
    "incorrect api key",
    "invalid_api_key",
    "invalid x-api-key",
    "authentication_error",
    "authentication error",
    "authentication failed",
    "authentication failure",
    "failed to authenticate",
    "could not authenticate",
    "unauthenticated",
    "unauthorized",
)

_ACCESS_SIGNALS = (
    "permission denied",
    "permission_denied",
    "permission_error",
    "permission error",
    "forbidden",
    "access denied",
)

_MODEL_SIGNALS = (
    "model not found",
    "model_not_found",
    "not found for api version",
    "unknown model",
    "no such model",
)

_GLOBAL_PERMANENT_SIGNALS = (
    _CREDENTIAL_SIGNALS + _ACCESS_SIGNALS + _MODEL_SIGNALS
)

_INPUT_PERMANENT_SIGNALS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "prompt is too long",
    "prompt too long",
    "input is too long",
    "input too long",
    "string is too long",
    "string too long",
    "request too large",
    "request_too_large",
    "payload too large",
)

# Pre-compiled patterns for _detect_limit_type.
_DETECT_TPM_RE = re.compile(r"\btpm\b", re.IGNORECASE)
_DETECT_RPM_RE = re.compile(r"\brpm\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Exception-chain walker
# ---------------------------------------------------------------------------

def _walk_chain(exc: BaseException):
    """Yield *exc* and every ``__cause__`` / ``__context__`` node, with a
    visited-id guard so a cyclic chain cannot loop forever."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _find_insufficient_source_error(
    exc: BaseException,
) -> InsufficientSourceError | None:
    """Return the first insufficient-source error in *exc*'s chain."""
    for node in _walk_chain(exc):
        if isinstance(node, InsufficientSourceError):
            return node
    return None


def _has_provider_or_agent_error(exc: BaseException) -> bool:
    """Return whether *exc* came through the provider/agent boundary.

    Permanent-error phrases such as ``permission denied`` also occur in local
    filesystem and parser failures.  Restricting permanent classification to
    an ``LLMError`` or ``AgentError`` prevents those from being mistaken for
    invalid credentials or access.
    """
    return any(isinstance(node, (LLMError, AgentError)) for node in _walk_chain(exc))


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

def _is_rate_limit_error(
    exc: BaseException,
    profile: "RateLimitProfile | None" = None,
) -> bool:
    """Return True if *exc* or any cause in its chain is a rate-limit signal.

    Parameters
    ----------
    exc:
        The exception to classify.
    profile:
        When supplied, only ``profile.signals`` are used for detection,
        giving provider-specific accuracy.  When ``None`` (backward-compat),
        the module-level ``_RATE_LIMIT_SIGNALS`` tuple is used so that existing
        callers without a profile continue to work unchanged.
    """
    signals = profile.signals if profile is not None else _RATE_LIMIT_SIGNALS
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current).lower()
        if any(sig in msg for sig in signals):
            return True
        current = current.__cause__ or current.__context__
    return False


# ---------------------------------------------------------------------------
# Terminal-billing classifier
# ---------------------------------------------------------------------------

def _is_terminal_billing_error(exc: BaseException) -> bool:
    """True for an unambiguous billing/credit/quota-exhaustion signal.

    Conservative by design: matches only the specific phrases in
    ``_TERMINAL_BILLING_SIGNALS``, never a bare ``quota`` / ``429`` / ``402``.
    """
    for node in _walk_chain(exc):
        msg = str(node).lower()
        signals = _TERMINAL_BILLING_SIGNALS
        if any(sig in msg for sig in _INPUT_PERMANENT_SIGNALS):
            signals = tuple(sig for sig in signals if sig != "hard limit")
        if any(sig in msg for sig in signals):
            return True
    return False


# ---------------------------------------------------------------------------
# Global / input permanent-error classifier
# ---------------------------------------------------------------------------

def _classify_permanent_error(exc: BaseException) -> str | None:
    """Return ``"global"``, ``"input"``, or ``None``.

    ``"global"`` — invalid credentials, unknown model, forbidden access.
    ``"input"`` — request/context too large for this file.
    ``None``    — not classifiable as permanent; treat as retryable.
    """
    saw_global = False
    saw_input = False
    for node in _walk_chain(exc):
        msg = str(node).lower()
        if any(sig in msg for sig in _INPUT_PERMANENT_SIGNALS):
            saw_input = True
        if any(sig in msg for sig in _GLOBAL_PERMANENT_SIGNALS):
            saw_global = True
        if "model" in msg and "does not exist" in msg:
            saw_global = True
    if saw_input:
        return "input"
    if saw_global:
        return "global"
    return None


# ---------------------------------------------------------------------------
# Fixed failure-precedence classifier
# ---------------------------------------------------------------------------

def _classify_failure(
    exc: BaseException,
    profile: "RateLimitProfile | None",
) -> str:
    """Apply the fixed failure precedence to *exc* and return a verdict.

    Returns one of ``"insufficient_source"``, ``"response_contract_final"``,
    ``"terminal_billing"``, ``"rate_limit"``, ``"global"``, ``"input"``, or
    ``"transient"``.  Precedence:

    0. insufficient-source (a deterministic local condition whose path may contain
       provider-like rate-limit signal text);
    1. response-contract-final (a deterministic response-contract
       rejection is non-retryable, and a correction call that failed on a
       rate-limit/transport fault is wrapped in a ``ResponseContractError`` whose
       wrapped cause would otherwise match the rate-limit branch below);
    2. terminal-billing (co-occurs with quota/429 signals);
    3. rate-limit;
    4. global-permanent (abort);
    5. input-permanent (fail this file, continue run);
    6. transient (existing retry/sleep behavior).

    ``repair`` never wraps a correction fault that classifies as
    ``"terminal_billing"`` or ``"global"`` (it re-raises those unchanged), so no
    genuine run-level abort reaches this early check.
    """
    if _find_insufficient_source_error(exc) is not None:
        return "insufficient_source"
    if any(isinstance(node, ResponseContractError) for node in _walk_chain(exc)):
        return "response_contract_final"
    provider_or_agent_error = _has_provider_or_agent_error(exc)
    if provider_or_agent_error and _is_terminal_billing_error(exc):
        return "terminal_billing"
    if _is_rate_limit_error(exc, profile):
        return "rate_limit"
    permanent = _classify_permanent_error(exc) if provider_or_agent_error else None
    if permanent == "global":
        return "global"
    if permanent == "input":
        return "input"
    return "transient"


# ---------------------------------------------------------------------------
# Abort builders
# ---------------------------------------------------------------------------

def _build_terminal_abort(
    exc: BaseException,
    provider_name: str,
    verdict: str,
) -> UnrecoverableProviderError:
    """Build an ``UnrecoverableProviderError(category="terminal")`` for a
    confirmed billing/credentials/model/access fault."""
    if verdict == "terminal_billing":
        cause = (
            "billing/credit exhausted — the account is out of funds or credit, "
            "or has hit a hard spend limit"
        )
    else:
        messages = "\n".join(str(node).lower() for node in _walk_chain(exc))
        if any(signal in messages for signal in _CREDENTIAL_SIGNALS):
            cause = "invalid credentials or authentication failure"
        elif any(signal in messages for signal in _MODEL_SIGNALS) or (
            "model" in messages and "does not exist" in messages
        ):
            cause = "unknown or unavailable model name"
        else:
            cause = "forbidden or permission-denied access"
    reason = (
        f"Provider error that cannot recover by retrying ({cause}). "
        "Completed file-level results were saved to crash_recovery.json in the "
        "output directory. Re-running the same command resumes compatible "
        "completed ordinary files; completed fresh-split files are deliberately "
        "re-documented from scratch."
    )
    err = UnrecoverableProviderError(provider_name, reason, category="terminal")
    err.__cause__ = exc
    err.__suppress_context__ = True
    return err


def _build_rate_limit_exhausted_abort(
    provider_name: str,
) -> UnrecoverableProviderError:
    """Build the bounded zero-progress rate-limit stop abort."""
    reason = (
        "Provider is persistently rate-limited or out of quota: no file made "
        "progress after stepping down to the lowest concurrency, so retrying was "
        "stopped to avoid sleeping through the backoff schedule for nothing. "
        "Completed file-level results were saved to crash_recovery.json in the "
        "output directory. Re-running the same command resumes compatible "
        "completed ordinary files; completed fresh-split files are deliberately "
        "re-documented from scratch."
    )
    return UnrecoverableProviderError(
        provider_name, reason, category="rate_limit_exhausted"
    )


def _raise_rate_limit_exhausted(
    provider_name: str,
    error_reporter,  # ErrorReporter — typed loosely to avoid pulling in utils.errors here
) -> None:
    """Emit a warning for the bounded zero-progress stop, then raise the abort."""
    abort = _build_rate_limit_exhausted_abort(provider_name)
    warn_msg = (
        f"[{provider_name}] Persistent rate limit / quota: no file made progress "
        "at the lowest concurrency. Stopping the run; completed files are saved "
        "in crash_recovery.json. Re-running resumes compatible completed ordinary "
        "files; completed fresh-split files are deliberately re-documented from "
        "scratch."
    )
    print(warn_msg, flush=True)
    logger.warning(warn_msg)
    error_reporter.record(
        RuntimeError(warn_msg),
        context="rate limit bound — zero-progress stop",
        level="warning",
    )
    raise abort


# ---------------------------------------------------------------------------
# Limit-type detector
# ---------------------------------------------------------------------------

def _detect_limit_type(error_msg: str) -> str | None:
    """Classify the kind of rate limit from an error message string.

    Returns ``"tpm"``, ``"rpm"``, ``"quota"``, ``"overloaded"``, or ``None``.
    """
    msg = error_msg.lower()
    if "tokens per min" in msg or _DETECT_TPM_RE.search(error_msg):
        return "tpm"
    if "requests per min" in msg or _DETECT_RPM_RE.search(error_msg):
        return "rpm"
    if "daily" in msg or "quota" in msg or "resource_exhausted" in msg:
        return "quota"
    if "overloaded" in msg or "529" in msg:
        return "overloaded"
    return None


# ---------------------------------------------------------------------------
# Retry-After parser
# ---------------------------------------------------------------------------

def _parse_retry_after(exc: BaseException) -> float | None:
    """Extract a Retry-After delay (seconds) from the exception message chain."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current)
        m = re.search(r"try again in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"retry.after[:\s]+([\d.]+)", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        current = current.__cause__ or current.__context__
    return None
