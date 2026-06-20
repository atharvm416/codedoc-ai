"""Agent-file execution: rate-limit handling, retries, and parallelism.

0.9.4 — extracted from ``codedoc.pipeline`` as part of the internal
decomposition.  This module owns:

- rate-limit and retry-after classification;
- the adaptive-parallelism step-down ladder;
- sequential and parallel descriptor processing with per-file retries and
  worker-thread recording;
- progress logging, result-error inspection, and pending-work cancellation.

The public entry point is :func:`execute_agent_files`, which takes a fully
constructed :class:`ExecutionContext`.  The provider-specific
:class:`~codedoc.llm.rate_limit_profile.RateLimitProfile` and the
:class:`ExecutionOptions` are built by the pipeline from resolved
configuration and passed in; this module never receives the full
configuration dictionary nor recomputes configuration policy.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.db import compute_file_hash
from codedoc.core.queue import ProcessingQueue
from codedoc.core.resume import _public_record_to_doc
from codedoc.core.safe_writer import SafeWriter
from codedoc.llm.rate_limit_profile import RateLimitProfile
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import (
    AgentError,
    ErrorReporter,
    LiveBackupWriteError,
    LLMError,
    OutputError,
    ParseError,
    UnrecoverableProviderError,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit detection helpers (Work Item 3)
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


def _is_rate_limit_error(
    exc: BaseException,
    profile: RateLimitProfile | None = None,
) -> bool:
    """Return True if *exc* or any cause in its chain is a rate-limit signal.

    Inspects ``str(exc)`` and walks ``__cause__`` / ``__context__`` so that
    provider signals are not hidden by wrapper exceptions.

    Parameters
    ----------
    exc:
        The exception to classify.
    profile:
        0.8.1 — when supplied, only ``profile.signals`` are used for detection,
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
# Unrecoverable-error classification (0.9.7)
# ---------------------------------------------------------------------------
#
# These two classifiers sit next to ``_is_rate_limit_error`` because they are
# the same kind of conservative, network-free, chain-walking message classifier:
# they inspect only the text already present in the raised exception chain
# (``str(exc)`` plus every ``__cause__`` / ``__context__`` node), match
# lowercased substrings against narrow signal sets, and never make a provider
# call.  They are deliberately conservative: when in doubt they return the
# retryable default and let the Workstream C time bound stop a doomed run.  A
# false *abort* wrongly kills a healthy run, which is worse than bounded
# retrying, so bare numeric HTTP codes are never matched on their own.
#
# Provider message text reaches this layer by two routes and both are covered by
# walking the chain *and* inspecting each node's own ``str()`` (exactly as
# ``_is_rate_limit_error`` does): the common agent path folds ``str(exc)`` into
# an ``AgentError`` message raised *without* ``from exc`` (phrase lives in the
# ``AgentError``'s own string), while the validation/parse path raises
# ``AgentError(...) from exc`` (original in ``__cause__``).

# Unambiguous billing/credit/quota-exhaustion phrases.  These are *specific
# phrases*, never the bare word ``quota``: an account out of funds/credit or at a
# hard spend limit cannot recover by waiting.  Some co-occur with
# ``quota``/``429`` (also rate-limit signals), so terminal-billing is always
# checked *before* ``_is_rate_limit_error`` at every call site.  The textual
# ``payment required`` phrase also covers the HTTP 402 case without matching a
# bare ``402``.
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

# Permanent signals that affect *every* file the same way: invalid credentials,
# authentication failure, forbidden/permission-denied access, and
# unknown/not-found model.  Phrases only — a bare ``401`` / ``403`` / ``404`` is
# never matched on its own; the model phrases ("model not found", "does not
# exist", ...) already require corroborating text, so a naked ``404`` in a
# request id does not trigger an abort.
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

# Permanent signals that affect *only this file's input* (the request/context is
# too large).  Re-sending the identical oversized prompt is guaranteed to fail
# again, so this file is recorded as failed without retrying; the rest of the run
# proceeds.  Phrases only — a bare ``413`` is never matched on its own.
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


def _walk_chain(exc: BaseException):
    """Yield *exc* and every ``__cause__`` / ``__context__`` node, with a
    visited-id guard so a cyclic chain cannot loop forever."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _has_provider_or_agent_error(exc: BaseException) -> bool:
    """Return whether *exc* came through the provider/agent boundary.

    Permanent-error phrases such as ``permission denied`` also occur in local
    filesystem and parser failures.  Restricting permanent classification to
    an ``LLMError`` or ``AgentError`` in the chain prevents those local errors
    from being mistaken for invalid provider credentials or access.
    """
    return any(isinstance(node, (LLMError, AgentError)) for node in _walk_chain(exc))


def _is_terminal_billing_error(exc: BaseException) -> bool:
    """True for an unambiguous billing/credit/quota-exhaustion signal.

    These cannot recover by waiting: the account is out of funds/credit or has
    hit a hard spend limit.  Takes priority over rate-limit classification, so
    it is checked before ``_is_rate_limit_error`` at every call site.

    Conservative by design: matches only the specific phrases in
    ``_TERMINAL_BILLING_SIGNALS``, never a bare ``quota`` / ``429`` / ``402``.
    """
    for node in _walk_chain(exc):
        msg = str(node).lower()
        # "hard limit" is used for both account spending limits and per-request
        # context limits.  Input-size evidence is narrower and must keep the
        # failure scoped to one file instead of aborting the entire run.
        signals = _TERMINAL_BILLING_SIGNALS
        if any(sig in msg for sig in _INPUT_PERMANENT_SIGNALS):
            signals = tuple(sig for sig in signals if sig != "hard limit")
        if any(sig in msg for sig in signals):
            return True
    return False


def _classify_permanent_error(exc: BaseException) -> str | None:
    """Return ``"global"``, ``"input"``, or ``None``.

    ``"global"``
        Affects every file the same way (invalid credentials, unknown model,
        forbidden/permission-denied access).
    ``"input"``
        Affects only this file's input (request/context too large).
    ``None``
        Not classifiable as permanent; treat as retryable.

    Conservative by design: matches only the specific phrases in
    ``_GLOBAL_PERMANENT_SIGNALS`` / ``_INPUT_PERMANENT_SIGNALS`` and never a bare
    numeric HTTP code.  When a message matches both a global and an input signal,
    ``"input"`` (the narrower, run-continuing verdict) is preferred.
    """
    saw_global = False
    saw_input = False
    for node in _walk_chain(exc):
        msg = str(node).lower()
        if any(sig in msg for sig in _INPUT_PERMANENT_SIGNALS):
            saw_input = True
        if any(sig in msg for sig in _GLOBAL_PERMANENT_SIGNALS):
            saw_global = True
        # Provider wording commonly inserts a model id between "model" and
        # "does not exist" (for example, "The model `x` does not exist").
        # Require both concepts so unrelated missing resources do not abort the
        # whole run.
        if "model" in msg and "does not exist" in msg:
            saw_global = True
    if saw_input:
        return "input"
    if saw_global:
        return "global"
    return None


def _classify_failure(
    exc: BaseException,
    profile: RateLimitProfile | None,
) -> str:
    """Apply the fixed 0.9.7 failure precedence to *exc* and return a verdict.

    Returns one of ``"terminal_billing"``, ``"rate_limit"``, ``"global"``,
    ``"input"``, or ``"transient"``.  Precedence (identical at every call site):

    1. terminal-billing — checked first because billing phrases co-occur with
       ``quota`` / ``429`` rate-limit signals;
    2. rate-limit — existing handling, bounded by Workstream C;
    3. global-permanent — abort;
    4. input-permanent — do not retry this file;
    5. transient — existing retry/sleep behavior.
    """
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


def _build_terminal_abort(
    exc: BaseException,
    provider_name: str,
    verdict: str,
) -> UnrecoverableProviderError:
    """Build an ``UnrecoverableProviderError(category="terminal")`` for a
    confirmed billing/credentials/model/access fault.

    The message names the likely cause class and the provider without inventing
    specifics absent from *exc*, and tells the operator that completed work is
    saved and re-running resumes.  The original error is retained as
    ``__cause__`` so diagnostics are preserved (equivalent to ``raise ... from
    exc`` regardless of which site raises it).
    """
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
        "Completed files were saved to the live JSON backup in the output "
        "directory; re-running the same command resumes the unfinished files."
    )
    err = UnrecoverableProviderError(provider_name, reason, category="terminal")
    err.__cause__ = exc
    err.__suppress_context__ = True
    return err


def _build_rate_limit_exhausted_abort(
    provider_name: str,
) -> UnrecoverableProviderError:
    """Build the bounded zero-progress rate-limit stop (Workstream C).

    Carries ``category="rate_limit_exhausted"`` so the CLI exits ``1`` (a
    transient "retry later" condition, not a credentials fault).  The message
    states the provider is persistently rate-limited / out of quota, that partial
    results were saved to the live backup, and that re-running resumes.
    """
    reason = (
        "Provider is persistently rate-limited or out of quota: no file made "
        "progress after stepping down to the lowest concurrency, so retrying was "
        "stopped to avoid sleeping through the backoff schedule for nothing. "
        "Partial results were saved to the live JSON backup in the output "
        "directory; re-running the same command resumes the unfinished files."
    )
    return UnrecoverableProviderError(
        provider_name, reason, category="rate_limit_exhausted"
    )


def _raise_rate_limit_exhausted(
    provider_name: str,
    error_reporter: ErrorReporter,
) -> None:
    """Emit a final warning describing the bounded zero-progress stop, then raise
    the ``category="rate_limit_exhausted"`` abort.  Does not sleep."""
    abort = _build_rate_limit_exhausted_abort(provider_name)
    warn_msg = (
        f"[{provider_name}] Persistent rate limit / quota: no file made progress "
        "at the lowest concurrency. Stopping the run; completed files are saved "
        "in the live JSON backup — re-run the same command to resume."
    )
    print(warn_msg, flush=True)
    logger.warning(warn_msg)
    error_reporter.record(
        RuntimeError(warn_msg),
        context="rate limit bound — zero-progress stop",
        level="warning",
    )
    raise abort


@dataclass(frozen=True)
class _SequentialOutcome:
    """Minimal progress signal threaded back from a sequential pass so the
    Workstream C zero-progress bound can be evaluated without parsing logs.

    ``succeeded_any``
        True if at least one file was recorded successfully during the pass.
    ``failures``
        Number of files the pass marked failed.
    ``all_failures_rate_limited``
        True when every failure the pass saw was rate-limit-classified (and there
        was at least one failure).  False if any non-rate-limit failure occurred.
    """

    succeeded_any: bool
    failures: int
    all_failures_rate_limited: bool


def _is_zero_progress_pass(outcome: _SequentialOutcome) -> bool:
    """True when a lowest-concurrency sequential pass made no progress and every
    failure it saw was rate-limit-classified — the Workstream C stop condition."""
    return (
        not outcome.succeeded_any
        and outcome.failures > 0
        and outcome.all_failures_rate_limited
    )


# Pre-compiled patterns for _detect_limit_type.  Word boundaries ensure that
# "tpm" inside "uptime" does not match, and parenthesised forms like "(TPM)"
# do match because ( and ) are non-word characters.
_DETECT_TPM_RE = re.compile(r"\btpm\b", re.IGNORECASE)
_DETECT_RPM_RE = re.compile(r"\brpm\b", re.IGNORECASE)


def _detect_limit_type(error_msg: str) -> str | None:
    """Classify the kind of rate limit from an error message string.

    Returns one of ``"tpm"``, ``"rpm"``, ``"quota"``, ``"overloaded"``, or
    ``None`` when the type cannot be determined.  Patterns are checked
    case-insensitively in priority order.

    Examples that must classify correctly:
    - ``"429 tokens per min exceeded"`` → ``"tpm"``
    - ``"limit exceeded (TPM)"``        → ``"tpm"``
    - ``"requests per min exceeded"``   → ``"rpm"``
    - ``"daily quota exhausted"``       → ``"quota"``
    - ``"529 overloaded"``              → ``"overloaded"``
    - ``"429 too many requests"``       → ``None``
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


def _build_default_ladder(max_p: int) -> list[int]:
    """Build the default parallelism step-down ladder for *max_p* workers."""
    if max_p <= 1:
        return [1]
    if max_p == 2:
        return [2, 1]
    mid = max(2, max_p // 2)
    ladder: list[int] = [max_p]
    if mid < max_p:
        ladder.append(mid)
    if 1 not in ladder:
        ladder.append(1)
    return ladder


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


# ---------------------------------------------------------------------------
# Execution boundary (0.9.4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionOptions:
    """Resolved execution policy, derived from configuration by the pipeline."""

    max_workers: int
    retry_attempts: int
    max_consecutive_failures: int
    rate_limit_adaptive: bool
    parallel_ladder: tuple[int, ...] | None
    respect_retry_after: bool
    retry_after_cap_s: int


@dataclass
class ExecutionContext:
    """All collaborators required to execute the agent-file queue.

    The pipeline constructs this after provider creation and live-backup
    initialization, so execution never touches configuration loading,
    provider creation, or output writing.
    """

    orchestrator: Orchestrator
    queue: ProcessingQueue
    recorder: SafeWriter
    error_reporter: ErrorReporter
    rate_limit_profile: RateLimitProfile
    stats: dict
    new_results: dict[str, dict]
    options: ExecutionOptions


# ---------------------------------------------------------------------------
# Worker wrapper (Work Item 2)
# ---------------------------------------------------------------------------

def _process_and_record(
    descriptor: dict,
    orchestrator: Orchestrator,
    recorder: SafeWriter,
) -> dict:
    """Process one file and record it in the live backup from the worker thread.

    Recording happens here — inside the worker — so a Ctrl-C or crash that
    interrupts the main ``as_completed`` collection loop never discards a
    file whose AI work already completed.

    If ``_process_one_file`` raises for any reason (rate-limit, parse failure,
    model error), the exception propagates out of this function unchanged and
    ``recorder.record()`` is NOT called.  The batch-level code catches the
    future's exception and classifies it as rate-limit or non-rate-limit.
    """
    result = _process_one_file(descriptor, orchestrator)
    recorder.record(
        descriptor["rel_path"],
        result,
        _safe_file_hash(descriptor.get("path")),
    )
    return result


# ---------------------------------------------------------------------------
# Parallel / sequential file processing (Work Items 2 & 3)
# ---------------------------------------------------------------------------

def execute_agent_files(context: ExecutionContext) -> None:
    """Process the agent-file queue, applying retries and the rate-limit ladder.

    Replaces the pre-0.9.4 ``_process_agent_files`` with a context-driven
    facade; behavior is unchanged.
    """
    queue = context.queue
    orchestrator = context.orchestrator
    stats = context.stats
    error_reporter = context.error_reporter
    new_results = context.new_results
    recorder = context.recorder
    profile = context.rate_limit_profile
    options = context.options

    max_workers = options.max_workers
    retry_attempts = options.retry_attempts
    max_consecutive_failures = options.max_consecutive_failures
    respect_retry_after = options.respect_retry_after
    retry_after_cap = options.retry_after_cap_s

    descriptors: list[dict] = []
    while True:
        descriptor = queue.next()
        if descriptor is None:
            break
        descriptors.append(descriptor)

    if max_workers <= 1 or len(descriptors) <= 1:
        # This single sequential pass IS the lowest-concurrency pass, so the
        # Workstream C zero-progress bound applies directly to it.
        outcome = _process_files_sequentially(
            descriptors,
            orchestrator,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
            new_results,
            recorder,
            respect_retry_after,
            retry_after_cap,
            profile,
        )
        if _is_zero_progress_pass(outcome):
            _raise_rate_limit_exhausted(
                orchestrator.llm.provider_name, error_reporter
            )
        return

    # Build the parallelism ladder.
    rate_limit_adaptive = options.rate_limit_adaptive
    custom_ladder = options.parallel_ladder
    if custom_ladder:
        ladder = list(custom_ladder)
    else:
        ladder = _build_default_ladder(max_workers)

    provider_name = orchestrator.llm.provider_name
    original_max_workers = max_workers
    remaining = list(descriptors)
    event_number = 0  # cumulative step-down event count across all rungs

    for level_index, level in enumerate(ladder):
        if not remaining:
            break

        level = min(level, len(remaining)) or 1
        succeeded, retry_rate_limited, failed_non_rate_limited = _process_descriptor_batch(
            remaining,
            orchestrator,
            queue,
            stats,
            error_reporter,
            max_workers=level,
            max_consecutive_failures=max_consecutive_failures,
            recorder=recorder,
            profile=profile,
        )
        new_results.update(succeeded)

        # Non-rate-limit failures are retried sequentially immediately so errors
        # are clearly diagnosed and stats["failed"] is correctly incremented.
        if failed_non_rate_limited:
            logger.info(
                "Retrying %d non-rate-limit failed file(s) sequentially for clearer diagnostics.",
                len(failed_non_rate_limited),
            )
            _process_files_sequentially(
                failed_non_rate_limited,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                respect_retry_after,
                retry_after_cap,
                profile,
            )

        if not retry_rate_limited or not rate_limit_adaptive:
            # No rate-limited files remain, or adaptive mode is off.
            if retry_rate_limited and not rate_limit_adaptive:
                # Treat remaining rate-limited files as sequential retry.
                remaining_descs = [d for d, _e in retry_rate_limited]
                outcome = _process_files_sequentially(
                    remaining_descs,
                    orchestrator,
                    queue,
                    stats,
                    error_reporter,
                    retry_attempts,
                    max_consecutive_failures,
                    new_results,
                    recorder,
                    respect_retry_after,
                    retry_after_cap,
                    profile,
                )
                if _is_zero_progress_pass(outcome):
                    _raise_rate_limit_exhausted(provider_name, error_reporter)
            break

        # --- Step down to next ladder rung ---
        next_level = ladder[level_index + 1] if level_index + 1 < len(ladder) else 1

        # 0.8.1: unpack descriptors and exceptions from the rate-limited list.
        remaining_descs = [d for d, _e in retry_rate_limited]
        exceptions = [e for _d, e in retry_rate_limited]

        # 0.8.1: compute inter-rung sleep duration.
        retry_afters = [
            ra for ra in (_parse_retry_after(e) for e in exceptions)
            if ra is not None
        ]
        retry_after_s = max(retry_afters) if retry_afters else None

        if respect_retry_after and retry_after_s is not None:
            sleep_s = min(retry_after_s, retry_after_cap)
        elif profile.min_backoff_s > 0:
            sleep_s = min(
                profile.min_backoff_s * (profile.backoff_scale ** level_index),
                retry_after_cap,
            )
        else:
            sleep_s = 0.0

        # 0.8.1: derive error sample and limit type from the first exception.
        error_sample = str(exceptions[0])[:200] if exceptions else ""
        limit_type = _detect_limit_type(error_sample) if error_sample else None

        event_number += 1
        warning = {
            "provider": provider_name,
            "original_max_parallel": original_max_workers,
            "current_level": level,
            "new_level": next_level,
            "retried_count": len(remaining_descs),
            "retry_after_s": retry_after_s,
            "sleep_s": sleep_s,
            "error_sample": error_sample,
            "limit_type": limit_type,
            "event_number": event_number,
            "rung_index": level_index,
        }
        stats.setdefault("rate_limit_warnings", []).append(warning)

        warn_msg = (
            f"[{provider_name}] Rate limit detected - your configured "
            f"max_parallel_files ({original_max_workers}) has been reduced to "
            f"{next_level}. Retrying {len(remaining_descs)} remaining file(s) "
            f"at lower concurrency."
        )
        if sleep_s > 0:
            warn_msg += f" Sleeping {sleep_s:.1f}s before retry."
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        error_reporter.record(RuntimeError(warn_msg), context="rate limit step-down", level="warning")

        # 0.8.1: apply inter-rung backoff sleep before the next ladder level.
        if sleep_s > 0:
            logger.info(
                "Rate-limit backoff: sleeping %.1fs before level %d (rung %d, event %d)",
                sleep_s, next_level, level_index, event_number,
            )
            time.sleep(sleep_s)

        remaining = remaining_descs

        # If we have exhausted the ladder, fall through to sequential.
        if level_index + 1 >= len(ladder):
            if remaining:
                still_limited_msg = (
                    f"[{provider_name}] Still rate-limited at max_parallel_files=1. "
                    f"Processing sequentially. You may want to lower your default "
                    f"max_parallel_files in config."
                )
                print(still_limited_msg, flush=True)
                logger.warning(still_limited_msg)
            # The ladder has been fully traversed; this sequential fall-through is
            # the single lowest-concurrency pass.  Apply the Workstream C
            # zero-progress bound to it.  The inter-rung sleep above already
            # happened; if this pass makes no progress we stop without sleeping
            # again.
            outcome = _process_files_sequentially(
                remaining,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                respect_retry_after,
                retry_after_cap,
                profile,
            )
            if _is_zero_progress_pass(outcome):
                _raise_rate_limit_exhausted(provider_name, error_reporter)
            break


def _process_descriptor_batch(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    recorder: SafeWriter,
    max_consecutive_failures: int = 5,
    profile: RateLimitProfile | None = None,
) -> tuple[dict[str, dict], list[tuple[dict, Exception]], list[dict]]:
    """Process a batch of descriptors in parallel at *max_workers* concurrency.

    Returns
    -------
    succeeded : dict[str, dict]
        rel_path → result for files that completed without error.  These have
        already been recorded in the live backup by the worker thread.
    retry_rate_limited : list[tuple[dict, Exception]]
        (descriptor, causing_exception) pairs for files that hit a rate-limit
        signal.  The exception is preserved so the caller can parse
        ``Retry-After`` hints and compute appropriate inter-rung backoff.
    failed_non_rate_limited : list[dict]
        Descriptors that failed for non-rate-limit reasons.
    """
    succeeded: dict[str, dict] = {}
    retry_rate_limited: list[tuple[dict, Exception]] = []
    failed_non_rate_limited: list[dict] = []

    consecutive_failures = 0
    health_reported = False
    total = len(descriptors)
    completed = 0
    fatal_error: LiveBackupWriteError | None = None
    abort_error: UnrecoverableProviderError | None = None

    logger.info(
        "Parallel batch: %d file(s) at max_workers=%d, provider=%s",
        total,
        max_workers,
        orchestrator.llm.provider_name,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map: dict[concurrent.futures.Future, dict] = {
            pool.submit(_process_and_record, descriptor, orchestrator, recorder): descriptor
            for descriptor in descriptors
        }

        for future in concurrent.futures.as_completed(future_map):
            descriptor = future_map[future]
            rel_path = descriptor["rel_path"]
            try:
                result = future.result()
                # Already recorded in the worker — just update new_results and stats.
                succeeded[rel_path] = result
                queue.mark_checked(rel_path)
                stats["checked"] += 1
                consecutive_failures = 0
                completed += 1
                _log_file_progress("OK", rel_path, completed, total)
            except LiveBackupWriteError as exc:
                # Fatal output failure raised inside a worker.  Identify it
                # before any rate-limit / ordinary-failure classification: cancel
                # work not yet started, let already-running workers finish or
                # fail without scheduling retries, and re-raise the original
                # error after the executor shuts down.  Never enter the retry or
                # rate-limit lists.
                fatal_error = exc
                _cancel_pending(future_map)
                break
            except Exception as exc:
                completed += 1
                # 0.9.7: evaluate terminal-billing and global-permanent BEFORE the
                # rate-limit branch and the consecutive-failure health check.  A
                # confirmed unrecoverable provider fault is handled exactly like
                # the fatal LiveBackupWriteError path: cancel work not yet started,
                # stop scheduling retries, and re-raise the abort after the
                # executor shuts down.  It never enters any retry list.
                verdict = _classify_failure(exc, profile)
                if verdict in ("terminal_billing", "global"):
                    abort_error = _build_terminal_abort(
                        exc, orchestrator.llm.provider_name, verdict
                    )
                    _cancel_pending(future_map)
                    break
                if verdict == "rate_limit":
                    # Only treat as done if a worker recorded it THIS run (it may
                    # have succeeded before the future was cancelled).  A file
                    # that is only *preloaded* from a prior output (stale) must be
                    # retried, never restored from old documentation (A4).
                    if not recorder.recorded_this_run(rel_path):
                        # 0.8.1: preserve the causing exception alongside the
                        # descriptor so the caller can parse Retry-After hints.
                        retry_rate_limited.append((descriptor, exc))
                        _log_file_progress("RATE-LIMIT", rel_path, completed, total, str(exc))
                    else:
                        # Already recorded — treat as succeeded.  Recover the
                        # real persisted record (A4) so the final output is not
                        # overwritten with an empty placeholder.
                        recovered = recorder.get_record(rel_path)
                        succeeded[rel_path] = (
                            _public_record_to_doc(recovered) if recovered else {}
                        )
                        queue.mark_checked(rel_path)
                        stats["checked"] += 1
                        _log_file_progress("OK(late)", rel_path, completed, total)
                elif verdict == "input":
                    # The identical oversized input cannot succeed on a second
                    # processing path.  Record it here and keep processing other
                    # files without placing it on the sequential retry list.
                    error_reporter.record(exc, context=rel_path)
                    queue.mark_failed(rel_path, str(exc))
                    stats["failed"] += 1
                    consecutive_failures += 1
                    _log_file_progress("FAIL", rel_path, completed, total, str(exc))
                else:
                    # Transient non-rate-limit failures retain the existing
                    # sequential retry path for clearer diagnostics.
                    failed_non_rate_limited.append(descriptor)
                    consecutive_failures += 1
                    _log_file_progress("RETRY", rel_path, completed, total, str(exc))

                if (
                    consecutive_failures >= max_consecutive_failures
                    and not health_reported
                ):
                    health_reported = True
                    _cancel_pending(future_map)
                    error_reporter.record(
                        RuntimeError(
                            f"Parallel processing saw {consecutive_failures} consecutive "
                            "non-rate-limit file failures. "
                            "Failed files will be retried sequentially."
                        ),
                        context="parallel processing health check",
                        level="warning",
                    )

    # The executor's context manager has now shut down (wait=True), so all
    # running workers have completed or failed and no new work was scheduled.
    # Propagate the fatal persistence failure as the original error.
    if fatal_error is not None:
        raise fatal_error
    # 0.9.7: propagate a confirmed unrecoverable provider abort so it leaves
    # execution and the pipeline records + re-raises it.  Raised only after the
    # executor shut down, so no further descriptors run.
    if abort_error is not None:
        raise abort_error

    return succeeded, retry_rate_limited, failed_non_rate_limited


def _process_files_sequentially(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
    recorder: SafeWriter,
    respect_retry_after: bool = True,
    retry_after_cap: int = 30,
    profile: RateLimitProfile | None = None,
) -> _SequentialOutcome:
    """Process *descriptors* one at a time with per-file retries.

    0.9.7 — returns a :class:`_SequentialOutcome` so that, when this is the
    lowest-concurrency pass, ``execute_agent_files`` can apply the zero-progress
    rate-limit bound (Workstream C).  Existing callers ignore the return value;
    behavior is otherwise unchanged.
    """
    consecutive_failures = 0
    total = len(descriptors)
    succeeded_any = False
    failures = 0
    all_failures_rate_limited = True

    logger.info(
        "Starting sequential documentation: %d file(s), provider=%s",
        total,
        orchestrator.llm.provider_name,
    )

    for index, descriptor in enumerate(descriptors, start=1):
        rel_path = descriptor["rel_path"]
        try:
            result = _process_one_file_with_retries(
                descriptor,
                orchestrator,
                retry_attempts,
                respect_retry_after=respect_retry_after,
                retry_after_cap=retry_after_cap,
                profile=profile,
            )
            new_results[rel_path] = result
            recorder.record(rel_path, result, _safe_file_hash(descriptor.get("path")))
            queue.mark_checked(rel_path)
            stats["checked"] += 1
            consecutive_failures = 0
            succeeded_any = True
            _log_file_progress("OK", rel_path, index, total)
        except LiveBackupWriteError:
            # Fatal: the crash-safety backup could not be persisted, so the
            # resume guarantee no longer holds.  Do not retry, do not mark the
            # file failed — stop scheduling work and propagate immediately.
            # (LiveBackupWriteError subclasses OutputError, so this clause must
            # precede the recoverable OutputError handler below.)
            raise
        except UnrecoverableProviderError:
            # 0.9.7: a terminal billing/credentials/model/access abort raised by
            # the per-file retry routing must propagate out of execution so the
            # pipeline records it and stops while the live backup stays resumable.
            # Mirrors the LiveBackupWriteError re-raise; must precede the
            # recoverable AgentError/OutputError and generic handlers below
            # (UnrecoverableProviderError is an LLMError, not one of those).
            raise
        except (ParseError, OutputError, AgentError) as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, str(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            failures += 1
            if not _is_rate_limit_error(exc, profile):
                all_failures_rate_limited = False
            _log_file_progress("FAIL", rel_path, index, total, str(exc))
        except Exception as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, str(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            failures += 1
            if not _is_rate_limit_error(exc, profile):
                all_failures_rate_limited = False
            _log_file_progress("FAIL", rel_path, index, total, str(exc))

        if consecutive_failures >= max_consecutive_failures:
            error_reporter.record(
                RuntimeError(
                    "Stopping sequential processing after "
                    f"{consecutive_failures} consecutive file failures. "
                    "Check API credentials, provider availability, model name, "
                    "rate limits, and network connectivity."
                ),
                context="sequential processing health check",
            )
            break

    return _SequentialOutcome(
        succeeded_any=succeeded_any,
        failures=failures,
        all_failures_rate_limited=all_failures_rate_limited and failures > 0,
    )


def _process_one_file_with_retries(
    descriptor: dict,
    orchestrator: Orchestrator,
    retry_attempts: int,
    respect_retry_after: bool = True,
    retry_after_cap: int = 30,
    profile: RateLimitProfile | None = None,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retry_attempts + 1):
        try:
            return _process_one_file(descriptor, orchestrator)
        except Exception as exc:
            last_error = exc
            # 0.9.7: apply the fixed failure precedence before consuming the next
            # attempt.  Abort cases raise immediately (no remaining attempts);
            # an input-too-large error re-raises immediately so it is recorded as
            # a normal failed file without a guaranteed-to-fail retry; transient
            # keeps the existing retry/sleep behavior.
            verdict = _classify_failure(exc, profile)
            if verdict in ("terminal_billing", "global"):
                raise _build_terminal_abort(
                    exc, orchestrator.llm.provider_name, verdict
                )
            if verdict == "input":
                raise
            if attempt < retry_attempts:
                # Apply Retry-After sleep for rate-limit errors in sequential mode.
                if respect_retry_after and verdict == "rate_limit":
                    delay = _parse_retry_after(exc)
                    if delay is not None:
                        sleep_s = min(delay, retry_after_cap)
                        logger.info(
                            "Retry-After: sleeping %.1fs before retrying %s",
                            sleep_s,
                            descriptor["rel_path"],
                        )
                        time.sleep(sleep_s)
                logger.info(
                    "Retrying %s (%d/%d): %s",
                    descriptor["rel_path"],
                    attempt + 1,
                    retry_attempts,
                    exc,
                )
    raise last_error or RuntimeError("Unknown processing failure")


def _log_file_progress(
    status: str,
    rel_path: str,
    completed: int,
    total: int,
    detail: str | None = None,
) -> None:
    percent = int((completed / total) * 100) if total else 100
    remaining = max(total - completed, 0)
    message = "[%s] %s | %d/%d complete (%d%%), %d remaining"
    if detail:
        logger.warning(
            message + " | %s",
            status, rel_path, completed, total, percent, remaining, detail,
        )
    else:
        logger.info(message, status, rel_path, completed, total, percent, remaining)


def _process_one_file(descriptor: dict, orchestrator: Orchestrator) -> dict:
    rel_path = descriptor["rel_path"]
    logger.info("[START] %s | provider=%s", rel_path, orchestrator.llm.provider_name)
    file_path: Path = descriptor["path"]
    content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    imports = parse_file(descriptor)
    result = orchestrator.process(descriptor, content, imports)
    errors = _agent_errors(result)
    if errors:
        raise AgentError(orchestrator.__class__.__name__, descriptor["rel_path"], "; ".join(errors))
    return result


def _agent_errors(result: dict) -> list[str]:
    errors = []
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if isinstance(value, dict) and value.get("error"):
            agent = value.get("agent", key)
            errors.append(f"{agent}: {value['error']}")
    return errors


def _safe_file_hash(file_path: Path | None) -> str:
    if not file_path:
        return ""
    try:
        return compute_file_hash(file_path)
    except Exception:
        return ""


def _cancel_pending(future_map: dict) -> None:
    for future in future_map:
        if not future.done():
            future.cancel()
