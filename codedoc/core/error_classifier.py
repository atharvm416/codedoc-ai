"""Provider-error classification helpers.

Extracted from ``codedoc.core.execution`` as part of a structural decomposition.
This module owns exception-chain walking and the fixed failure precedence
classifier, which resolves a verdict from the closed provider-failure-envelope
reason-code table (section 5.3) rather than free-text substring matching. It
has no dependency on execution state, the orchestrator, or file-processing
logic.

Import note
-----------
``execution.py`` imports this entire module and re-exports every symbol as a
backward-compatibility shim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codedoc.utils.errors import (
    AgentError,
    InsufficientSourceError,
    LLMError,
    PROVIDER_FAILURE_REASON_CODES,
    ResponseContractError,
    UnrecoverableProviderError,
    detect_limit_type,
    find_provider_failure,
)
from codedoc.utils.logger import get_logger

# Compatibility alias: detect_limit_type's body and backing regexes now live
# in codedoc.utils.errors (section 11.5's "relocated" name class); this
# module keeps the historical private name bound to the same function.
_detect_limit_type = detect_limit_type

if TYPE_CHECKING:
    from codedoc.llm.rate_limit_profile import RateLimitProfile

logger = get_logger(__name__)

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
# Closed provider-failure-envelope verdict table (section 5.3)
# ---------------------------------------------------------------------------

_REASON_CODE_VERDICTS: dict[str, str] = {
    "provider-quota-exhausted": "terminal_billing",
    "provider-authentication-rejected": "global",
    "provider-model-unavailable": "global",
    "provider-rate-limited": "rate_limit",
    "provider-input-rejected": "input",
    "provider-timeout": "transient",
    "provider-connection-failed": "transient",
    "provider-response-malformed": "transient",
    "provider-request-failed": "transient",
}
assert set(_REASON_CODE_VERDICTS) == PROVIDER_FAILURE_REASON_CODES, (
    "_REASON_CODE_VERDICTS must cover exactly the nine closed reason codes"
)


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
    ``"transient"``.  Precedence (section 5.3):

    0. insufficient-source (a deterministic local condition);
    1. response-contract-final (a deterministic response-contract
       rejection is non-retryable, and a correction call that failed on a
       rate-limit/transport fault is wrapped in a ``ResponseContractError``
       whose wrapped cause would otherwise match the rate-limit branch below);
    2. envelope verdict, resolved from the closed
       :data:`~codedoc.utils.errors.ProviderFailureEnvelope` reason-code table
       via :func:`~codedoc.utils.errors.find_provider_failure` -- never from
       rendered text, path, agent, endpoint, or credential;
    3. transient, for an envelope-less provider-like message (no fallback to
       text matching).

    *profile* is accepted for call-site compatibility but is no longer
    consulted here: configured rate-limit text signals now apply only at the
    adapter's own exception boundary (section 5.3), promoting an otherwise
    ``provider-request-failed`` envelope before it ever reaches this
    classifier.

    ``repair`` never wraps a correction fault that classifies as
    ``"terminal_billing"`` or ``"global"`` (it re-raises those unchanged), so no
    genuine run-level abort reaches this early check.
    """
    if _find_insufficient_source_error(exc) is not None:
        return "insufficient_source"
    if any(isinstance(node, ResponseContractError) for node in _walk_chain(exc)):
        return "response_contract_final"
    failure = find_provider_failure(exc)
    if failure is None:
        return "transient"
    if failure.reason_code not in _REASON_CODE_VERDICTS:
        raise ValueError(
            f"No verdict mapping for provider failure reason_code: "
            f"{failure.reason_code!r}"
        )
    return _REASON_CODE_VERDICTS[failure.reason_code]


# ---------------------------------------------------------------------------
# Abort builders
# ---------------------------------------------------------------------------

def _build_terminal_abort(
    exc: BaseException,
    provider_name: str,
    verdict: str,
) -> UnrecoverableProviderError:
    """Build an ``UnrecoverableProviderError(category="terminal")`` for a
    confirmed billing/credentials/model/access fault.

    Wording is selected structurally from the envelope (section 5.4), never
    from rendered provider text: quota -> billing wording; model unavailable
    -> model wording; authentication status 403 -> forbidden/permission
    wording; every other authentication status -> invalid-credentials
    wording. Requires an envelope-bearing exception -- there is no longer a
    text-substitution fallback for a caller that bypasses
    :func:`_classify_failure`.
    """
    failure = find_provider_failure(exc)
    if failure is None:
        raise ValueError(
            "_build_terminal_abort requires an envelope-bearing exception "
            f"(verdict={verdict!r}); none was found in the exception chain."
        )
    if failure.reason_code == "provider-quota-exhausted":
        cause = (
            "billing/credit exhausted — the account is out of funds or credit, "
            "or has hit a hard spend limit"
        )
    elif failure.reason_code == "provider-model-unavailable":
        cause = "unknown or unavailable model name"
    elif failure.status == 403:
        cause = "forbidden or permission-denied access"
    else:
        cause = "invalid credentials or authentication failure"
    reason = (
        f"Provider error that cannot recover by retrying ({cause}). "
        "Completed file-level results were saved to crash_recovery.json in the "
        "output directory. Re-running the same command reuses compatible "
        "completed ordinary and split records and resumes compatible current "
        "schema-4 split node checkpoints; forced, stale, identity-mismatched, "
        "legacy, foreign, or unsupported state is rerun or preserved and "
        "blocked according to the documented remedy."
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
        "output directory. Re-running the same command reuses compatible "
        "completed ordinary and split records and resumes compatible current "
        "schema-4 split node checkpoints; forced, stale, identity-mismatched, "
        "legacy, foreign, or unsupported state is rerun or preserved and "
        "blocked according to the documented remedy."
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
        "in crash_recovery.json. Re-running reuses compatible completed ordinary "
        "and split records and resumes compatible current schema-4 split node "
        "checkpoints; forced, stale, identity-mismatched, legacy, foreign, or "
        "unsupported state is rerun or preserved and blocked according to the "
        "documented remedy."
    )
    print(warn_msg, flush=True)
    logger.warning(warn_msg)
    error_reporter.record(
        RuntimeError(warn_msg),
        context="rate limit bound — zero-progress stop",
        level="warning",
    )
    raise abort
