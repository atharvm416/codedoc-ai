"""Deterministic-clock support for retry, backoff, and rate-limit tests.

Generalizes the recorder pattern used across the suite: replace a named sleep
callable with one that records the requested duration and never waits.
"""
from __future__ import annotations

import pytest


def capture_sleeps(monkeypatch: pytest.MonkeyPatch, target: str) -> list[float]:
    """Replace the dotted callable named by *target* with a recorder.

    Returns the mutable list of requested durations, appended to in call order
    with no coercion. The replacement never sleeps.
    """
    sleeps: list[float] = []
    monkeypatch.setattr(target, lambda seconds: sleeps.append(seconds))
    return sleeps
