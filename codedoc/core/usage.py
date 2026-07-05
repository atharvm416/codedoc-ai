"""Approximate LLM usage accounting for codedoc.

Everything in this module is an *estimate*.  ``estimate_tokens`` is a
character heuristic (~4 characters per token), not a provider tokenizer.
No monetary cost is computed anywhere because model pricing is provider-,
model-, date-, cache-, and tier-dependent.
"""

from __future__ import annotations

import threading


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for *text*.

    This is a character heuristic (about 4 characters per token), not a
    tokenizer.  Non-empty text always counts as at least 1 token.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class UsageAccumulator:
    """Thread-safe accumulator for approximate per-run LLM usage.

    One accumulator is created per real pipeline run and the same instance is
    shared by the orchestrator and all three agents.  All counters are
    estimates (see :func:`estimate_tokens`) except the call counters, which
    count actual provider attempts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._estimated_input_tokens = 0
        self._estimated_output_tokens = 0
        self._successful_calls = 0
        self._failed_calls = 0

    def record_input(self, *texts: str) -> None:
        """Add the estimated token count of each prompt part (system + user).

        Called immediately before every provider attempt, so retries count as
        additional input.
        """
        added = sum(estimate_tokens(t) for t in texts)
        with self._lock:
            self._estimated_input_tokens += added

    def record_success(self, output_text: str = "") -> None:
        """Count one successful provider call and estimate its output tokens."""
        added = estimate_tokens(output_text)
        with self._lock:
            self._successful_calls += 1
            self._estimated_output_tokens += added

    def record_failure(self) -> None:
        """Count one failed provider call (the provider call itself raised)."""
        with self._lock:
            self._failed_calls += 1

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def estimated_input_tokens(self) -> int:
        with self._lock:
            return self._estimated_input_tokens

    @property
    def estimated_output_tokens(self) -> int:
        with self._lock:
            return self._estimated_output_tokens

    @property
    def successful_calls(self) -> int:
        with self._lock:
            return self._successful_calls

    @property
    def failed_calls(self) -> int:
        with self._lock:
            return self._failed_calls

    @property
    def attempted_calls(self) -> int:
        """Total provider attempts so far (successful + failed)."""
        with self._lock:
            return self._successful_calls + self._failed_calls

    def snapshot(self) -> dict:
        """Return a consistent snapshot of all counters."""
        with self._lock:
            return {
                "estimated_input_tokens": self._estimated_input_tokens,
                "estimated_output_tokens": self._estimated_output_tokens,
                "successful_calls": self._successful_calls,
                "failed_calls": self._failed_calls,
                "attempted_calls": self._successful_calls + self._failed_calls,
            }
