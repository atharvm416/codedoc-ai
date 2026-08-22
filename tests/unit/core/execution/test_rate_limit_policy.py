"""Tests organized by feature ownership."""

from __future__ import annotations

from codedoc.utils.errors import ProviderFailureEnvelope

def test_15_parallel_ladder_invalid_non_decreasing_raises(tmp_path):
    """Test 15a: non-decreasing parallel_ladder raises ConfigError."""
    from codedoc.core.loader import load_config
    from codedoc.utils.errors import ConfigError

    try:
        load_config(tmp_path, {
            "entry_file": "main.py",
            "parallel_ladder": [2, 5, 1],  # not decreasing
        })
        assert False, "Should have raised ConfigError"
    except ConfigError as e:
        assert "decreasing" in str(e).lower() or "ladder" in str(e).lower()

def test_15_parallel_ladder_clamped_when_exceeds_max(tmp_path):
    """Test 15b: ladder values exceeding max_parallel_files are clamped (no error)."""
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "max_parallel_files": 3,
        "parallel_ladder": [10, 5, 1],  # exceeds max
    })
    ladder = cfg["parallel_ladder"]
    assert all(v <= 3 for v in ladder), f"All ladder values must be clamped to 3, got {ladder}"

def test_15_parallel_ladder_appends_1_if_missing(tmp_path):
    """Test 15c: parallel_ladder without trailing 1 gets 1 appended."""
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "max_parallel_files": 5,
        "parallel_ladder": [5, 2],  # missing 1
    })
    assert cfg["parallel_ladder"][-1] == 1, "ladder must always end with 1"

def test_build_default_ladder():
    from codedoc.pipeline import _build_default_ladder

    assert _build_default_ladder(1) == [1]
    assert _build_default_ladder(2) == [2, 1]
    assert _build_default_ladder(3) == [3, 2, 1]
    assert _build_default_ladder(5) == [5, 2, 1]
    assert _build_default_ladder(10) == [10, 5, 1]
    assert _build_default_ladder(6) == [6, 3, 1]
    assert _build_default_ladder(7) == [7, 3, 1]
    # Always ends with 1
    for n in range(1, 15):
        ladder = _build_default_ladder(n)
        assert ladder[-1] == 1, f"Ladder for {n} must end with 1: {ladder}"
        assert ladder[0] == n, f"Ladder for {n} must start with {n}: {ladder}"

def test_D3_signals_add_appended_to_profile(tmp_path):
    """D3, rewritten against section 5.3's adapter-boundary signal evaluation:
    a configured added signal reaches the promotion helper the real adapter
    calls at its own exception boundary, and the profile's original signals
    still work alongside it."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.llm.api_provider import _promote_via_configured_signals
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_add": ["capacity exceeded", "throttled"],
    })
    profile = get_rate_limit_profile("openai", cfg)
    unmapped = ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-request-failed")
    signals = tuple(profile.signals)

    assert _promote_via_configured_signals(
        RuntimeError("capacity exceeded"), unmapped, signals
    ).reason_code == "provider-rate-limited"
    assert _promote_via_configured_signals(
        RuntimeError("throttled"), unmapped, signals
    ).reason_code == "provider-rate-limited"
    # Original signals still present
    assert _promote_via_configured_signals(
        RuntimeError("429"), unmapped, signals
    ).reason_code == "provider-rate-limited"

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

def test_D4_signals_remove_drops_from_profile(tmp_path):
    """D4, rewritten against section 5.3's adapter-boundary signal evaluation:
    a removed signal no longer promotes at the adapter boundary, while other
    signals from the same profile are unaffected."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.llm.api_provider import _promote_via_configured_signals
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_remove": ["503"],
    })
    profile = get_rate_limit_profile("gemini", cfg)
    assert "503" not in profile.signals
    unmapped = ProviderFailureEnvelope(provider_kind="gemini", reason_code="provider-request-failed")
    signals = tuple(profile.signals)

    # Other gemini signals still present
    assert _promote_via_configured_signals(
        RuntimeError("RESOURCE_EXHAUSTED"), unmapped, signals
    ).reason_code == "provider-rate-limited"
    assert _promote_via_configured_signals(
        RuntimeError("503 service unavailable"), unmapped, signals
    ).reason_code == "provider-request-failed"

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

def test_P2_signals_add_uppercase_detected(tmp_path):
    """P2 regression, rewritten against the adapter-boundary promotion:
    rate_limit_signals_add=['Throttled'] must promote regardless of the
    matched message's case."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.llm.api_provider import _promote_via_configured_signals
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_add": ["Throttled"],  # mixed case user input
    })
    profile = get_rate_limit_profile("openai", cfg)
    unmapped = ProviderFailureEnvelope(provider_kind="openai", reason_code="provider-request-failed")
    signals = tuple(profile.signals)

    # Error message has DIFFERENT case than the user-supplied signal
    assert _promote_via_configured_signals(
        RuntimeError("THROTTLED by gateway"), unmapped, signals
    ).reason_code == "provider-rate-limited", (
        "Custom signal 'Throttled' must match 'THROTTLED' in error message"
    )
    assert _promote_via_configured_signals(
        RuntimeError("throttled"), unmapped, signals
    ).reason_code == "provider-rate-limited", (
        "Custom signal 'Throttled' must match 'throttled' in error message"
    )

def test_P2_signals_remove_uppercase_removes_lowercase_default(tmp_path):
    """P2 regression, rewritten against the adapter-boundary promotion:
    rate_limit_signals_remove=['RESOURCE_EXHAUSTED'] removes default
    'resource_exhausted' so it no longer promotes."""
    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    from codedoc.llm.api_provider import _promote_via_configured_signals
    from codedoc.core.loader import load_config

    cfg = load_config(tmp_path, {
        "entry_file": "main.py",
        "rate_limit_signals_remove": ["RESOURCE_EXHAUSTED"],  # uppercase remove entry
    })
    profile = get_rate_limit_profile("gemini", cfg)

    assert "resource_exhausted" not in profile.signals, (
        "Uppercase 'RESOURCE_EXHAUSTED' in remove list must strip lowercase default"
    )
    unmapped = ProviderFailureEnvelope(provider_kind="gemini", reason_code="provider-request-failed")
    # The error must no longer promote via resource_exhausted
    assert _promote_via_configured_signals(
        RuntimeError("RESOURCE_EXHAUSTED"), unmapped, tuple(profile.signals)
    ).reason_code == "provider-request-failed", (
        "After removing 'resource_exhausted', errors with that signal must not promote"
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
