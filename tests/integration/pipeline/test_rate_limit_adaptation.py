"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import pytest
from codedoc.utils.errors import LLMError, UnrecoverableProviderError
from tests.support.clocks import capture_sleeps
from tests.support.recovery_rate_limit_runs import _patch_provider as recovery_rate_limit_patch_provider
from tests.support.recovery_rate_limit_runs import _run

@pytest.fixture
def captured_sleeps(monkeypatch):
    """Stub execution-layer ``time.sleep`` and record every requested duration."""
    return capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: provider)

class _AlwaysRateLimit:
    provider_name = "openai"

    def complete_json(self, prompt, system=""):
        raise LLMError("openai", "429 rate_limit_exceeded tokens per min")

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt)

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
    backup = tmp_path / "docs" / "crash_recovery.json"
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
    backup = tmp_path / "docs" / "crash_recovery.json"
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
    assert (tmp_path / "docs" / "crash_recovery.json").exists()

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

def test_9_rate_limit_ladder_steps_down(tmp_path, monkeypatch):
    """Test 9: rate-limit error causes ladder step-down; no descriptors dropped."""
    sleeps = capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")
    # Files must import each other so all 3 are reachable from entry a.py
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    from codedoc.utils.errors import LLMError

    call_count = {"n": 0}

    class RateLimitThenOk:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            # First 3 calls: rate-limit (during parallel batch at level 5)
            if call_count["n"] <= 3:
                raise LLMError("openai", "429 rate_limit_exceeded tokens per min")
            # After step-down: succeed
            if "key_concepts" in prompt:
                return json.dumps({"description": "ok", "role_in_system": "r",
                                   "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return json.dumps({"description": "ok", "role_in_system": "r",
                               "functions": [], "classes": [], "exports": []})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

    recovery_rate_limit_patch_provider(monkeypatch, RateLimitThenOk())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
    })

    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) > 0, "Expected at least one rate-limit step-down warning"
    assert sleeps == [5.0]

def test_9b_non_rate_limit_errors_do_not_step_down(tmp_path, monkeypatch):
    """Test 9b: non-rate-limit errors do not trigger ladder step-down."""
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("x=1\n")

    from codedoc.utils.errors import LLMError

    class ParseErrorProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "JSON parse error: invalid response format")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    recovery_rate_limit_patch_provider(monkeypatch, ParseErrorProvider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 2,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
    })

    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) == 0, "Non-rate-limit errors must not cause step-down"

def test_9c_non_rate_limit_parallel_failures_counted_in_stats(tmp_path, monkeypatch):
    """Non-rate-limit failures are counted without a persistent issue log."""
    # Files import each other so all 3 are reachable in parallel
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    from codedoc.utils.errors import LLMError

    class AlwaysFailProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "JSON parse error: model returned garbage")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    recovery_rate_limit_patch_provider(monkeypatch, AlwaysFailProvider())
    from codedoc.pipeline import run_pipeline

    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "max_consecutive_failures": 10,
    })

    # Non-rate-limit failures must be counted in stats["failed"]
    assert stats["failed"] > 0, (
        f"stats['failed'] must be > 0 for non-rate-limit errors, got {stats}"
    )
    # No rate-limit step-down warnings
    assert len(stats.get("rate_limit_warnings", [])) == 0, (
        "Non-rate-limit errors must not cause rate-limit step-down"
    )
    assert stats["issues_recorded"] == stats["failed"]
    assert "error_log" not in stats
    assert not (tmp_path / "codedoc" / "error.log").exists()

def test_10b_user_notification_includes_provider_and_level(tmp_path, monkeypatch, capsys):
    """Test 10b: step-down prints provider name and original max_parallel to stdout."""
    sleeps = capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")
    # Files must import each other so all 3 are reachable from entry a.py
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    from codedoc.utils.errors import LLMError

    class AlwaysRateLimitProvider:
        provider_name = "openai"

        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "429 rate_limit_exceeded")

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    recovery_rate_limit_patch_provider(monkeypatch, AlwaysRateLimitProvider())
    from codedoc.pipeline import run_pipeline
    from codedoc.utils.errors import UnrecoverableProviderError

    # 0.9.7: a persistent rate limit where no file ever succeeds now stops after
    # the bounded zero-progress pass with the transient-class abort, instead of
    # grinding to the consecutive-failure breaker.  The adaptive step-down
    # notification is still printed before the stop and names the provider and
    # the reduced concurrency — which is what this test verifies.
    with pytest.raises(UnrecoverableProviderError) as excinfo:
        run_pipeline(tmp_path, {
            "entry_file": "a.py",
            "parallel_agents": False,
            "propagate_changes": False,
            "max_parallel_files": 3,
            "rate_limit_adaptive": True,
            "file_retry_attempts": 0,
            "max_consecutive_failures": 1,
        })

    assert excinfo.value.category == "rate_limit_exhausted"

    captured = capsys.readouterr()
    out = captured.out.lower()
    assert "openai" in out
    assert "rate limit" in out
    # The step-down message reports the original max_parallel and the reduced level.
    assert "max_parallel_files" in captured.out
    assert "reduced to" in out
    assert sleeps == [5.0, 7.5, 11.25]

def test_10b_no_warning_on_successful_run(tmp_path, monkeypatch, capsys):
    """Test 10b: a successful run with no rate limits produces no step-down warning."""
    (tmp_path / "main.py").write_text("x=1\n")
    stats = _run(tmp_path, monkeypatch, entry_file="main.py")
    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) == 0, "No warnings expected on a clean run"

def _make_fake_provider_rate_limit_then_ok(
    fail_calls: int,
    provider_name: str = "openai",
    signal: str = "429 rate_limit_exceeded tokens per min",
):
    """Fail with a rate-limit signal for the first *fail_calls*, then succeed."""
    import json as _json
    from codedoc.utils.errors import LLMError

    call_count = {"n": 0}

    class Provider:
        pass

    Provider.provider_name = provider_name

    def _complete(self, prompt, system=""):
        call_count["n"] += 1
        if call_count["n"] <= fail_calls:
            raise LLMError(provider_name, signal)
        if "key_concepts" in prompt:
            return _json.dumps({"description": "ok", "role_in_system": "r",
                                 "key_concepts": [], "usage_example": ""})
        if "dependencies_analysis" in prompt:
            return _json.dumps({"dependencies_analysis": {
                "internal": [], "external": [], "dependency_refs": [],
                "catalog_updates": [], "usage_notes": [], "warnings": []}})
        return _json.dumps({"description": "ok", "role_in_system": "r",
                             "functions": [], "classes": [], "exports": []})

    Provider.complete_json = _complete
    Provider.complete = _complete
    return Provider()

def test_D9_inter_rung_sleep_uses_retry_after(tmp_path, monkeypatch):
    """D9: When Retry-After hint is present, sleep uses that value."""
    from codedoc.utils.errors import LLMError
    # Files must be >= 2 so we go parallel
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}

    class RetryAfterProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Include Retry-After hint in the error message
                raise LLMError("openai", "429 rate_limit. try again in 7.5s")
            import json as _j
            if "key_concepts" in prompt:
                return _j.dumps({"description": "ok", "role_in_system": "r",
                                  "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _j.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _j.dumps({"description": "ok", "role_in_system": "r",
                              "functions": [], "classes": [], "exports": []})
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: RetryAfterProvider())
    sleep_calls = capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 2,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "respect_retry_after": True,
        "rate_limit_backoff_s": 1.0,  # low default so retry-after wins when present
    })

    # A sleep must have been called (either Retry-After or profile backoff)
    assert len(sleep_calls) > 0, "Must sleep between ladder rungs when rate-limited"
    # The Retry-After hint was 7.5s — at least one sleep should be close to that
    # (or capped by retry_after_cap_s which defaults to 30)
    assert any(abs(s - 7.5) < 0.5 for s in sleep_calls), (
        f"Expected a sleep close to 7.5s (Retry-After hint), got sleeps={sleep_calls}"
    )

def test_D10_inter_rung_sleep_uses_profile_backoff(tmp_path, monkeypatch):
    """D10: When no Retry-After hint, sleep uses profile.min_backoff_s × scale^rung."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}

    class NoRetryAfterProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise LLMError("openai", "429 rate_limit_exceeded")  # no Retry-After hint
            import json as _j
            if "key_concepts" in prompt:
                return _j.dumps({"description": "ok", "role_in_system": "r",
                                  "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _j.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _j.dumps({"description": "ok", "role_in_system": "r",
                              "functions": [], "classes": [], "exports": []})
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: NoRetryAfterProvider())
    sleep_calls = capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 2,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "rate_limit_backoff_s": 4.0,   # predictable test value
        "rate_limit_backoff_scale": 1.0,  # no exponential growth for simplicity
    })

    # At rung 0, sleep = min(4.0 × 1.0^0, 30) = 4.0
    assert len(sleep_calls) > 0, "Profile backoff must trigger a sleep"
    assert any(abs(s - 4.0) < 0.5 for s in sleep_calls), (
        f"Expected sleep ~4.0s from profile backoff, got sleeps={sleep_calls}"
    )

def test_D11_backoff_s_zero_disables_sleep(tmp_path, monkeypatch):
    """D11: rate_limit_backoff_s=0 must not trigger any time.sleep call."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}

    class RLProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise LLMError("openai", "429 rate_limit")
            import json as _j
            if "key_concepts" in prompt:
                return _j.dumps({"description": "ok", "role_in_system": "r",
                                  "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _j.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _j.dumps({"description": "ok", "role_in_system": "r",
                              "functions": [], "classes": [], "exports": []})
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: RLProvider())
    sleep_calls = capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 2,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "rate_limit_backoff_s": 0,       # disable sleep
        "respect_retry_after": False,    # also disable Retry-After
    })

    positive_sleeps = [s for s in sleep_calls if s > 0]
    assert len(positive_sleeps) == 0, (
        f"rate_limit_backoff_s=0 must produce no positive sleep calls, got {sleep_calls}"
    )

def test_D12_warning_dict_has_all_required_fields(tmp_path, monkeypatch):
    """D12: Rate-limit warning dict includes error_sample, limit_type,
    retry_after_s, sleep_s, event_number, rung_index."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    call_count = {"n": 0}

    class RLProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 3:
                raise LLMError("openai", "429 rate_limit_exceeded tokens per min")
            import json as _j
            if "key_concepts" in prompt:
                return _j.dumps({"description": "ok", "role_in_system": "r",
                                  "key_concepts": [], "usage_example": ""})
            if "dependencies_analysis" in prompt:
                return _j.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": []}})
            return _j.dumps({"description": "ok", "role_in_system": "r",
                              "functions": [], "classes": [], "exports": []})
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: RLProvider())
    capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
        "rate_limit_backoff_s": 0.1,
    })

    warnings = stats.get("rate_limit_warnings", [])
    assert len(warnings) > 0, "Expected at least one rate-limit warning"

    w = warnings[0]
    required_fields = {
        "provider", "original_max_parallel", "current_level", "new_level",
        "retried_count", "retry_after_s", "sleep_s", "error_sample",
        "limit_type", "event_number", "rung_index",
    }
    missing = required_fields - w.keys()
    assert not missing, f"Warning dict is missing fields: {missing}\nGot: {w}"

    # Content checks
    assert w["event_number"] == 1, "First event must be numbered 1"
    assert w["rung_index"] == 0, "First step-down is at rung index 0"
    assert w["error_sample"] != "", "error_sample must be non-empty"
    assert w["limit_type"] == "tpm", f"'tokens per min' must classify as 'tpm', got {w['limit_type']}"
    assert w["sleep_s"] >= 0

def test_D14_no_files_dropped_after_step_down(tmp_path, monkeypatch):
    """D14: All files are documented after a ladder step-down."""
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    provider = _make_fake_provider_rate_limit_then_ok(
        fail_calls=3, provider_name="openai",
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: provider)
    capture_sleeps(monkeypatch, "codedoc.core.execution.time.sleep")

    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {
        "entry_file": "a.py",
        "parallel_agents": False,
        "propagate_changes": False,
        "max_parallel_files": 3,
        "rate_limit_adaptive": True,
        "file_retry_attempts": 0,
    })

    out = tmp_path / "codedoc" / "codedoc.json"
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    documented = {f["path"] for f in result.get("files", [])}

    assert "a.py" in documented
    assert "b.py" in documented
    assert "c.py" in documented
    assert len(documented) == 3, f"All 3 files must be documented, got {documented}"
    assert stats.get("checked", 0) + stats.get("reused", 0) == 3, (
        f"Total processed must be 3, got stats={stats}"
    )
