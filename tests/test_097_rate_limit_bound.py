"""0.9.7 — Workstream C: bounded rate-limit retrying (time safety).

All via injected fake providers; ``time.sleep`` is stubbed so the tests assert
the *bound* on sleeping without actually waiting.

- A persistent ambiguous rate limit (every file rate-limited, none succeed)
  stops after one full ladder traversal plus one lowest-concurrency pass via
  ``UnrecoverableProviderError(category="rate_limit_exhausted")``, with bounded
  cumulative sleep and the live backup intact and resumable.
- A transient rate limit where files keep succeeding between step-downs is
  unaffected and completes normally.
"""

from __future__ import annotations

import json

import pytest

import codedoc.core.execution as ex
from codedoc.utils.errors import LLMError, UnrecoverableProviderError


@pytest.fixture
def captured_sleeps(monkeypatch):
    """Stub execution-layer ``time.sleep`` and record every requested duration."""
    sleeps: list[float] = []
    monkeypatch.setattr(ex.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)


# ---------------------------------------------------------------------------
# Persistent ambiguous rate limit → bounded stop
# ---------------------------------------------------------------------------

class _AlwaysRateLimit:
    provider_name = "openai"

    def complete_json(self, prompt, system=""):
        raise LLMError("openai", "429 rate_limit_exceeded tokens per min")

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt)


def test_persistent_rate_limit_stops_after_bounded_passes(
    tmp_path, monkeypatch, captured_sleeps
):
    (tmp_path / "a.py").write_text("from b import x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from c import y\nx = 1\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("y = 1\n", encoding="utf-8")

    _patch_provider(monkeypatch, _AlwaysRateLimit())
    from codedoc.pipeline import run_pipeline

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "a.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "propagate_changes": False,
                "max_parallel_files": 3,          # ladder [3, 2, 1]
                "rate_limit_adaptive": True,
                "file_retry_attempts": 0,         # no per-file retry sleeps
                "retry_after_cap_s": 5,
            },
        )

    # A bounded rate-limit / quota stop — NOT a credentials fault.
    assert excinfo.value.category == "rate_limit_exhausted"

    # Honest bound: at most one inter-rung sleep per ladder step-down (the
    # ladder [3, 2, 1] has at most 3 step-downs), and no per-file retry sleeps
    # were added (file_retry_attempts=0).  The run did NOT grind through an
    # unbounded backoff schedule.
    assert len(captured_sleeps) <= 3
    assert all(s <= 5 for s in captured_sleeps)

    # Every stop is safe: the dedicated recovery file is intact and resumable,
    # never deleted or overwritten with a "complete" final output.  0.9.8: the
    # stable output (docs/codedoc.json) is never created on this path.
    assert not (tmp_path / "docs" / "codedoc.json").exists()
    backup = tmp_path / "docs" / "crash_recovery_codedoc.json"
    assert backup.exists()
    data = json.loads(backup.read_text(encoding="utf-8"))
    assert "_crash_safety" in data or data.get("_codedoc", {}).get("status") == "in_progress"


def test_initial_sequential_persistent_rate_limit_stops(
    tmp_path, monkeypatch, captured_sleeps
):
    """The initial-sequential path (max_parallel_files=1) is bounded too."""
    (tmp_path / "a.py").write_text("from b import x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")

    _patch_provider(monkeypatch, _AlwaysRateLimit())
    from codedoc.pipeline import run_pipeline

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "a.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "propagate_changes": False,
                "max_parallel_files": 1,   # straight to the sequential path
                "file_retry_attempts": 0,
            },
        )

    assert excinfo.value.category == "rate_limit_exhausted"
    backup = tmp_path / "docs" / "crash_recovery_codedoc.json"
    assert backup.exists()


def test_non_adaptive_persistent_rate_limit_stops(
    tmp_path, monkeypatch, captured_sleeps
):
    """Disabling adaptive step-down still bounds the sequential fallback."""
    (tmp_path / "a.py").write_text("from b import x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")

    _patch_provider(monkeypatch, _AlwaysRateLimit())
    from codedoc.pipeline import run_pipeline

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "a.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "propagate_changes": False,
                "max_parallel_files": 2,
                "rate_limit_adaptive": False,
                "file_retry_attempts": 0,
            },
        )

    assert excinfo.value.category == "rate_limit_exhausted"
    assert (tmp_path / "docs" / "crash_recovery_codedoc.json").exists()


def test_initial_sequential_uses_custom_rate_limit_signal(
    tmp_path, monkeypatch, captured_sleeps
):
    """Configured signals must reach retries and the zero-progress bound."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")

    class _CustomRateLimit:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "custom throttle sentinel")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    _patch_provider(monkeypatch, _CustomRateLimit())
    from codedoc.pipeline import run_pipeline

    with pytest.raises(UnrecoverableProviderError) as excinfo:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "a.py",
                "output_dir": "docs",
                "parallel_agents": False,
                "max_parallel_files": 1,
                "file_retry_attempts": 0,
                "rate_limit_signals_add": ["custom throttle sentinel"],
            },
        )

    assert excinfo.value.category == "rate_limit_exhausted"


# ---------------------------------------------------------------------------
# Transient rate limit → unaffected, completes normally
# ---------------------------------------------------------------------------

class _RateLimitOnceThenOk:
    """Rate-limits only the very first provider call, then always succeeds, so
    one file is rate-limited at the top concurrency and succeeds after a
    step-down while the other file succeeds immediately."""

    provider_name = "openai"

    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt, system=""):
        self.calls += 1
        if self.calls == 1:
            raise LLMError("openai", "429 rate_limit_exceeded")
        if "key_concepts" in prompt:
            return json.dumps({
                "description": "ok", "role_in_system": "r",
                "key_concepts": [], "usage_example": "",
            })
        if "dependencies_analysis" in prompt:
            return json.dumps({"dependencies_analysis": {
                "internal": [], "external": [], "dependency_refs": [],
                "catalog_updates": [], "usage_notes": [], "warnings": [],
            }})
        return json.dumps({
            "description": "ok", "role_in_system": "r",
            "functions": [], "classes": [], "exports": [],
        })

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt)


def test_transient_rate_limit_completes_normally(tmp_path, monkeypatch, captured_sleeps):
    (tmp_path / "a.py").write_text("from b import x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")

    _patch_provider(monkeypatch, _RateLimitOnceThenOk())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "a.py",
            "output_dir": "docs",
            "parallel_agents": False,
            "propagate_changes": False,
            "max_parallel_files": 2,      # ladder [2, 1]
            "rate_limit_adaptive": True,
            "file_retry_attempts": 1,
        },
    )

    # Both files documented; progress between step-downs resets the bound, so no
    # abort.  A clean MD/JSON run deletes nothing it should keep.
    assert stats["checked"] == 2
    assert stats["failed"] == 0
