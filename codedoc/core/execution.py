"""Agent-file execution: rate-limit handling, retries, and parallelism.

0.9.4 — extracted from ``codedoc.pipeline`` as part of the internal
decomposition.  This module owns:

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

0.10.2 — error classification logic moved to :mod:`codedoc.core.error_classifier`.
Compat re-exports below preserve all ``codedoc.core.execution._name`` imports
for one release.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codedoc.agents.orchestrator import Orchestrator
from codedoc.core.db import compute_file_hash
from codedoc.core.error_classifier import (
    _classify_failure,
    _detect_limit_type,
    _is_rate_limit_error,
    _parse_retry_after,
    _raise_rate_limit_exhausted,
)
from codedoc.core.queue import ProcessingQueue
from codedoc.core.resume import _public_record_to_doc
from codedoc.core.safe_writer import SafeWriter
from codedoc.llm.rate_limit_profile import RateLimitProfile
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import (
    AgentError,
    ErrorReporter,
    LiveBackupWriteError,
    OutputError,
    ParseError,
    UnrecoverableProviderError,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Compat re-exports (0.10.2 — deprecated; import from error_classifier instead)
# ---------------------------------------------------------------------------
from codedoc.core.error_classifier import (  # noqa: F401, E402  (compat re-export)
    _ACCESS_SIGNALS,
    _CREDENTIAL_SIGNALS,
    _DETECT_RPM_RE,
    _DETECT_TPM_RE,
    _GLOBAL_PERMANENT_SIGNALS,
    _INPUT_PERMANENT_SIGNALS,
    _MODEL_SIGNALS,
    _RATE_LIMIT_SIGNALS,
    _TERMINAL_BILLING_SIGNALS,
    _build_rate_limit_exhausted_abort,
    _build_terminal_abort,
    _classify_permanent_error,
    _has_provider_or_agent_error,
    _is_terminal_billing_error,
    _walk_chain,
)


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
