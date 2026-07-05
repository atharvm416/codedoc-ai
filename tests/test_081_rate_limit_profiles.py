"""
0.8.1 — Provider-aware rate-limit profile tests (Workstream D).

Plan acceptance criteria:
  D1   OpenAI, Anthropic, Gemini, and default profiles classify expected signals.
  D2   Non-rate-limit errors remain non-rate-limit.
  D3   rate_limit_signals_add appends extra signals to the active provider profile.
  D4   rate_limit_signals_remove drops signals from the active provider profile.
  D5   Combined add + remove resolves correctly.
  D6   rate_limit_backoff_s and rate_limit_backoff_scale override profile values.
  D7   _detect_limit_type classifies error messages correctly for each category.
  D8   _process_descriptor_batch() preserves causing exceptions for rate-limited files.
  D9   Parallel ladder sleeps between rungs when a Retry-After hint is present.
  D10  Parallel ladder sleeps using profile backoff when no Retry-After hint exists.
  D11  rate_limit_backoff_s = 0 disables computed inter-rung sleep.
  D12  Warning dict includes error_sample, limit_type, retry_after_s, sleep_s,
       event_number, and rung_index.
  D13  CLI summary prints a compact rate-limit line only when events occurred.
  D14  No files are dropped or double-counted after step-down.
  D15  get_rate_limit_profile() falls back to "default" for unknown providers.
  D16  _is_rate_limit_error() without a profile still works (backward compat).
"""
from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# D1 — Profile signal classification
# ---------------------------------------------------------------------------

def test_D1_openai_profile_classifies_429():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    profile = get_rate_limit_profile("openai")
    assert _is_rate_limit_error(LLMError("openai", "429 rate_limit_exceeded"), profile)
    assert _is_rate_limit_error(LLMError("openai", "tokens per min exceeded"), profile)
    assert _is_rate_limit_error(LLMError("openai", "quota exceeded"), profile)


def test_D1_anthropic_profile_classifies_529():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    profile = get_rate_limit_profile("anthropic")
    assert _is_rate_limit_error(LLMError("anthropic", "529 overloaded"), profile)
    assert _is_rate_limit_error(LLMError("anthropic", "overloaded try again"), profile)
    assert _is_rate_limit_error(LLMError("anthropic", "rate_limit exceeded"), profile)


def test_D1_gemini_profile_classifies_resource_exhausted():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    profile = get_rate_limit_profile("gemini")
    assert _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED"), profile)
    assert _is_rate_limit_error(LLMError("gemini", "quota exceeded"), profile)
    assert _is_rate_limit_error(LLMError("gemini", "503 service unavailable"), profile)


def test_D1_default_profile_classifies_all_provider_signals():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    profile = get_rate_limit_profile("unknown_gateway")
    assert _is_rate_limit_error(LLMError("x", "429"), profile)
    assert _is_rate_limit_error(LLMError("x", "529 overloaded"), profile)
    assert _is_rate_limit_error(LLMError("x", "resource_exhausted"), profile)
    assert _is_rate_limit_error(LLMError("x", "tokens per min exceeded"), profile)


# ---------------------------------------------------------------------------
# D2 — Non-rate-limit errors are not classified
# ---------------------------------------------------------------------------

def test_D2_json_parse_error_not_rate_limit():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError, ParseError

    profile = get_rate_limit_profile("openai")
    assert not _is_rate_limit_error(LLMError("openai", "JSON parse error"), profile)
    assert not _is_rate_limit_error(ParseError("main.py", "syntax error"), profile)
    assert not _is_rate_limit_error(ValueError("bad value"), profile)


def test_D2_anthropic_profile_does_not_match_openai_only_signals():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    # Anthropic profile does NOT include "tokens per min" or "too many requests"
    profile = get_rate_limit_profile("anthropic")
    assert not _is_rate_limit_error(LLMError("anthropic", "tokens per min exceeded"), profile)
    assert not _is_rate_limit_error(LLMError("anthropic", "too many requests"), profile)


# ---------------------------------------------------------------------------
# D3 — rate_limit_signals_add appends extra signals
# ---------------------------------------------------------------------------

def test_D3_signals_add_appended_to_profile(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_add": ["capacity exceeded", "throttled"],
    })
    profile = get_rate_limit_profile("openai", cfg)
    assert _is_rate_limit_error(LLMError("openai", "capacity exceeded"), profile)
    assert _is_rate_limit_error(LLMError("openai", "throttled"), profile)
    # Original signals still present
    assert _is_rate_limit_error(LLMError("openai", "429"), profile)


def test_D3_signals_add_no_duplicates(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_add": ["429", "new_signal"],  # "429" already in openai profile
    })
    profile = get_rate_limit_profile("openai", cfg)
    assert profile.signals.count("429") == 1, "No duplicate '429' in signals"
    assert "new_signal" in profile.signals


# ---------------------------------------------------------------------------
# D4 — rate_limit_signals_remove drops signals
# ---------------------------------------------------------------------------

def test_D4_signals_remove_drops_from_profile(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_remove": ["503"],
    })
    profile = get_rate_limit_profile("gemini", cfg)
    assert "503" not in profile.signals
    # Other gemini signals still present
    assert _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED"), profile)


def test_D4_module_defaults_not_mutated(tmp_path):
    """D4: Removing a signal via config must not mutate PROVIDER_PROFILES."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile, PROVIDER_PROFILES
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_remove": ["429"],
    })
    _ = get_rate_limit_profile("openai", cfg)
    # Module default must still have "429"
    assert "429" in PROVIDER_PROFILES["openai"].signals


# ---------------------------------------------------------------------------
# D5 — Combined add + remove resolves correctly
# ---------------------------------------------------------------------------

def test_D5_combined_add_and_remove(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_add": ["custom_signal"],
        "rate_limit_signals_remove": ["quota"],  # remove from openai profile
    })
    profile = get_rate_limit_profile("openai", cfg)
    assert "custom_signal" in profile.signals
    assert "quota" not in profile.signals
    assert "429" in profile.signals  # unaffected


# ---------------------------------------------------------------------------
# D6 — rate_limit_backoff_s and rate_limit_backoff_scale override profile
# ---------------------------------------------------------------------------

def test_D6_backoff_s_overrides_min_backoff(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {"entry_file": "main.py", "rate_limit_backoff_s": 3.0})
    profile = get_rate_limit_profile("anthropic", cfg)
    assert profile.min_backoff_s == 3.0  # was 10.0 by default


def test_D6_backoff_scale_overrides(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {"entry_file": "main.py", "rate_limit_backoff_scale": 3.0})
    profile = get_rate_limit_profile("openai", cfg)
    assert profile.backoff_scale == 3.0  # was 1.5 by default


def test_D6_backoff_s_zero_means_no_sleep(tmp_path):
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {"entry_file": "main.py", "rate_limit_backoff_s": 0})
    profile = get_rate_limit_profile("openai", cfg)
    assert profile.min_backoff_s == 0.0


# ---------------------------------------------------------------------------
# D7 — _detect_limit_type classifies error messages
# ---------------------------------------------------------------------------

def test_D7_detect_limit_type_tpm():
    from codedoc.pipeline import _detect_limit_type
    assert _detect_limit_type("rate limit exceeded tokens per min") == "tpm"
    assert _detect_limit_type("TPM limit reached for this model") == "tpm"


def test_D7_detect_limit_type_rpm():
    from codedoc.pipeline import _detect_limit_type
    assert _detect_limit_type("requests per min limit exceeded") == "rpm"
    assert _detect_limit_type("RPM quota hit") == "rpm"


def test_D7_detect_limit_type_quota():
    from codedoc.pipeline import _detect_limit_type
    assert _detect_limit_type("daily quota exhausted") == "quota"
    assert _detect_limit_type("RESOURCE_EXHAUSTED quota exceeded") == "quota"
    assert _detect_limit_type("quota limit reached") == "quota"


def test_D7_detect_limit_type_overloaded():
    from codedoc.pipeline import _detect_limit_type
    assert _detect_limit_type("529 overloaded") == "overloaded"
    assert _detect_limit_type("server overloaded, please retry") == "overloaded"


def test_D7_detect_limit_type_unknown_returns_none():
    from codedoc.pipeline import _detect_limit_type
    assert _detect_limit_type("429 too many requests") is None
    assert _detect_limit_type("rate limit exceeded") is None
    assert _detect_limit_type("") is None


# ---------------------------------------------------------------------------
# D8 — _process_descriptor_batch preserves exceptions for rate-limited files
# ---------------------------------------------------------------------------

def test_D8_process_batch_returns_exception_with_descriptor(tmp_path, monkeypatch):
    """D8: retry_rate_limited contains (descriptor, exception) tuples."""
    from codedoc.utils.errors import LLMError
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.queue import ProcessingQueue
    from codedoc.core.safe_writer import SafeWriter

    (tmp_path / "main.py").write_text("x=1\n")
    descriptor = {
        "rel_path": "main.py",
        "path": tmp_path / "main.py",
        "language": "python",
    }

    class RateLimitProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "429 rate_limit_exceeded tpm")
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    from codedoc.agents.orchestrator import Orchestrator
    orch = Orchestrator(RateLimitProvider(), parallel=False)

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0, "failed": 0}
    from codedoc.utils.errors import ErrorReporter
    error_reporter = ErrorReporter()

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {"main.py": descriptor})
    sw.set_queue_order(["main.py"])

    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    profile = get_rate_limit_profile("openai")

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [descriptor], orch, queue, stats, error_reporter,
        max_workers=1, recorder=sw, profile=profile,
    )

    assert len(retry_rate_limited) == 1, "Rate-limited file must be in retry list"
    desc_back, exc_back = retry_rate_limited[0]
    assert desc_back is descriptor, "Original descriptor must be returned"
    # The exception may be AgentError wrapping LLMError — verify it carries
    # the rate-limit signal so _is_rate_limit_error can detect it via chain walk.
    assert exc_back is not None, "Exception must be preserved (not None)"
    assert "429" in str(exc_back), (
        f"Rate-limit signal must be somewhere in the exception string: {exc_back!r}"
    )


# ---------------------------------------------------------------------------
# D9 — Parallel ladder sleeps using Retry-After when available
# ---------------------------------------------------------------------------

def test_D9_inter_rung_sleep_uses_retry_after(tmp_path, monkeypatch):
    """D9: When Retry-After hint is present, sleep uses that value."""
    from codedoc.utils.errors import LLMError
    # Files must be >= 2 so we go parallel
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}
    sleep_calls: list[float] = []

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
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda s: sleep_calls.append(s))

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


# ---------------------------------------------------------------------------
# D10 — Profile backoff when no Retry-After hint
# ---------------------------------------------------------------------------

def test_D10_inter_rung_sleep_uses_profile_backoff(tmp_path, monkeypatch):
    """D10: When no Retry-After hint, sleep uses profile.min_backoff_s × scale^rung."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}
    sleep_calls: list[float] = []

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
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda s: sleep_calls.append(s))

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


# ---------------------------------------------------------------------------
# D11 — rate_limit_backoff_s = 0 disables inter-rung sleep
# ---------------------------------------------------------------------------

def test_D11_backoff_s_zero_disables_sleep(tmp_path, monkeypatch):
    """D11: rate_limit_backoff_s=0 must not trigger any time.sleep call."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}
    sleep_calls: list[float] = []

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
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda s: sleep_calls.append(s))

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


# ---------------------------------------------------------------------------
# D12 — Warning dict includes all required fields
# ---------------------------------------------------------------------------

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
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda _: None)

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


# ---------------------------------------------------------------------------
# D13 — CLI compact rate-limit summary
# ---------------------------------------------------------------------------

def test_D13_cli_summary_shows_compact_line_when_events(tmp_path, monkeypatch, capsys):
    """D13: CLI prints compact rate-limit line only when events occurred."""
    from codedoc.utils.errors import LLMError

    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("x=1\n")

    call_count = {"n": 0}

    class RLProvider:
        provider_name = "anthropic"
        def complete_json(self, prompt, system=""):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise LLMError("anthropic", "529 overloaded")
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
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda _: None)

    from codedoc.cli.cli import main
    try:
        main([str(tmp_path), "--entry", "a.py",
              "--max-parallel-files", "2",
              "--no-parallel"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    # The compact summary must mention "rate" and show event count
    assert "rate limit" in captured.out.lower() or "rate-limit" in captured.out.lower(), (
        f"CLI must print rate-limit summary when events occurred.\nOutput: {captured.out!r}"
    )


def test_D13_cli_summary_absent_on_clean_run(tmp_path, monkeypatch, capsys):
    """D13b: CLI must NOT print rate-limit summary when no events occurred."""
    (tmp_path / "main.py").write_text("x=1\n")

    import json as _j

    class OkProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
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

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: OkProvider())

    from codedoc.cli.cli import main
    try:
        main([str(tmp_path), "--entry", "main.py", "--no-parallel"])
    except SystemExit:
        pass

    captured = capsys.readouterr()
    assert "rate limit" not in captured.out.lower(), (
        f"No rate-limit summary expected on clean run.\nOutput: {captured.out!r}"
    )


# ---------------------------------------------------------------------------
# D14 — No files dropped or double-counted after step-down
# ---------------------------------------------------------------------------

def test_D14_no_files_dropped_after_step_down(tmp_path, monkeypatch):
    """D14: All files are documented after a ladder step-down."""
    (tmp_path / "a.py").write_text("from b import x\n")
    (tmp_path / "b.py").write_text("from c import y\nx=1\n")
    (tmp_path / "c.py").write_text("y=1\n")

    provider = _make_fake_provider_rate_limit_then_ok(
        fail_calls=3, provider_name="openai",
    )
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: provider)
    monkeypatch.setattr("codedoc.core.execution.time.sleep", lambda _: None)

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


# ---------------------------------------------------------------------------
# D15 — get_rate_limit_profile falls back to "default" for unknown providers
# ---------------------------------------------------------------------------

def test_D15_unknown_provider_falls_back_to_default():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile, PROVIDER_PROFILES

    profile = get_rate_limit_profile("my_custom_gateway")
    default = PROVIDER_PROFILES["default"]
    assert profile.signals == default.signals
    assert profile.min_backoff_s == default.min_backoff_s
    assert profile.backoff_scale == default.backoff_scale


def test_D15_none_provider_falls_back_to_default():
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile, PROVIDER_PROFILES

    profile = get_rate_limit_profile("")
    assert profile.signals == PROVIDER_PROFILES["default"].signals


# ---------------------------------------------------------------------------
# D16 — _is_rate_limit_error without profile (backward compat)
# ---------------------------------------------------------------------------

def test_D16_is_rate_limit_error_without_profile_backward_compat():
    """D16: _is_rate_limit_error(exc) without a profile still uses _RATE_LIMIT_SIGNALS."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    # All the original signals from _RATE_LIMIT_SIGNALS must still work
    assert _is_rate_limit_error(LLMError("openai", "429 rate_limit_exceeded tokens per min"))
    assert _is_rate_limit_error(LLMError("anthropic", "529 overloaded"))
    assert _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED quota exceeded"))
    assert not _is_rate_limit_error(LLMError("openai", "JSON parse error"))


# ---------------------------------------------------------------------------
# Regression: P2 — custom signals are case-insensitive
# ---------------------------------------------------------------------------

def test_P2_signals_add_uppercase_detected(tmp_path):
    """P2 regression: rate_limit_signals_add=['Throttled'] must match 'THROTTLED' in error."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_add": ["Throttled"],  # mixed case user input
    })
    profile = get_rate_limit_profile("openai", cfg)

    # Error message has DIFFERENT case than the user-supplied signal
    assert _is_rate_limit_error(LLMError("openai", "THROTTLED by gateway"), profile), (
        "Custom signal 'Throttled' must match 'THROTTLED' in error message"
    )
    assert _is_rate_limit_error(LLMError("openai", "throttled"), profile), (
        "Custom signal 'Throttled' must match 'throttled' in error message"
    )


def test_P2_signals_remove_uppercase_removes_lowercase_default(tmp_path):
    """P2 regression: rate_limit_signals_remove=['RESOURCE_EXHAUSTED'] removes default 'resource_exhausted'."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_remove": ["RESOURCE_EXHAUSTED"],  # uppercase remove entry
    })
    profile = get_rate_limit_profile("gemini", cfg)

    assert "resource_exhausted" not in profile.signals, (
        "Uppercase 'RESOURCE_EXHAUSTED' in remove list must strip lowercase default"
    )
    # The error must no longer be detected via resource_exhausted
    assert not _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED"), profile), (
        "After removing 'resource_exhausted', errors with that signal must not match"
    )


def test_P2_signals_add_stored_lowercase_in_profile():
    """P2 regression: user-added signals are normalized to lowercase in the profile."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile

    profile = get_rate_limit_profile("openai", {
        "rate_limit_signals_add": ["BackPressure", "CAPACITY_EXCEEDED"],
    })
    assert "backpressure" in profile.signals, "Added signal must be lowercased"
    assert "capacity_exceeded" in profile.signals, "Added signal must be lowercased"
    assert "BackPressure" not in profile.signals  # original case must not be stored


# ---------------------------------------------------------------------------
# Regression: P3 — _detect_limit_type word-boundary tpm/rpm
# ---------------------------------------------------------------------------

def test_P3_detect_limit_type_tpm_in_parentheses():
    """P3 regression: '(TPM)' and 'limit exceeded (tpm)' must classify as 'tpm'."""
    from codedoc.pipeline import _detect_limit_type

    assert _detect_limit_type("limit exceeded (TPM)") == "tpm", (
        "Parenthesised '(TPM)' must be detected via word-boundary regex"
    )
    assert _detect_limit_type("rate limit exceeded (tpm) for this model") == "tpm"
    assert _detect_limit_type("TPM limit reached") == "tpm"


def test_P3_detect_limit_type_rpm_in_parentheses():
    """P3 regression: '(RPM)' must classify as 'rpm'."""
    from codedoc.pipeline import _detect_limit_type

    assert _detect_limit_type("requests limit (RPM) exceeded") == "rpm"
    assert _detect_limit_type("RPM quota hit") == "rpm"


def test_P3_detect_limit_type_uptime_not_tpm():
    """P3 regression: 'uptime' must NOT be classified as tpm (false positive guard)."""
    from codedoc.pipeline import _detect_limit_type

    # 'uptime' contains 'tpm'? No — \btpm\b requires word boundary.
    # But 'uptmpe' is contrived; test real guard: 'uptime' has no word-boundary 'tpm'.
    result = _detect_limit_type("server uptime exceeded SLA")
    assert result is None, (
        f"'uptime' must not trigger tpm classification, got {result!r}"
    )
