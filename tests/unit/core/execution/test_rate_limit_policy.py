"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
from codedoc.core.execution import _is_rate_limit_error
from codedoc.utils.errors import LLMError
from tests.support.providers import _install_anthropic, _install_gemini, _install_openai
from tests.support.provider_contract_cases import _make

def test_10_rate_limit_detector_openai(tmp_path):
    """Test 10a: OpenAI 429 / TPM signals detected."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    assert _is_rate_limit_error(LLMError("openai", "429 rate_limit_exceeded tokens per min"))
    assert _is_rate_limit_error(LLMError("openai", "Error code: 429 - quota exceeded"))
    assert _is_rate_limit_error(LLMError("openai", "Rate limit reached for tpm"))

def test_10_rate_limit_detector_anthropic(tmp_path):
    """Test 10b: Anthropic overloaded/529 signals detected."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    assert _is_rate_limit_error(LLMError("anthropic", "529 overloaded"))
    assert _is_rate_limit_error(LLMError("anthropic", "rate_limit quota exceeded"))
    assert _is_rate_limit_error(LLMError("anthropic", "overloaded try again"))

def test_10_rate_limit_detector_gemini(tmp_path):
    """Test 10c: Gemini RESOURCE_EXHAUSTED / quota signals detected."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    assert _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED quota exceeded"))
    assert _is_rate_limit_error(LLMError("gemini", "429 too many requests"))
    assert _is_rate_limit_error(LLMError("gemini", "resource_exhausted"))

def test_10_rate_limit_detector_false_positives(tmp_path):
    """Test 10d: ordinary errors are NOT rate-limit signals."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError, ParseError

    assert not _is_rate_limit_error(LLMError("openai", "JSON parse error invalid format"))
    assert not _is_rate_limit_error(ParseError("main.py", "syntax error"))
    assert not _is_rate_limit_error(ValueError("bad value"))

def test_10_rate_limit_detector_walks_cause_chain(tmp_path):
    """Test 10e: detector walks __cause__ chain so wrapper doesn't hide signal."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    inner = LLMError("openai", "429 rate_limit_exceeded")
    outer = RuntimeError("wrapper error")
    outer.__cause__ = inner

    assert _is_rate_limit_error(outer)

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

def test_D16_is_rate_limit_error_without_profile_backward_compat():
    """D16: _is_rate_limit_error(exc) without a profile still uses _RATE_LIMIT_SIGNALS."""
    from codedoc.pipeline import _is_rate_limit_error
    from codedoc.utils.errors import LLMError

    # All the original signals from _RATE_LIMIT_SIGNALS must still work
    assert _is_rate_limit_error(LLMError("openai", "429 rate_limit_exceeded tokens per min"))
    assert _is_rate_limit_error(LLMError("anthropic", "529 overloaded"))
    assert _is_rate_limit_error(LLMError("gemini", "RESOURCE_EXHAUSTED quota exceeded"))
    assert not _is_rate_limit_error(LLMError("openai", "JSON parse error"))

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

@pytest.mark.parametrize(
    "installer, cls",
    [
        (_install_openai, "OpenAIProvider"),
        (_install_anthropic, "AnthropicProvider"),
        (_install_gemini, "GeminiProvider"),
    ],
)
def test_rate_limit_error_is_classifiable_after_wrapping(monkeypatch, installer, cls):
    rec = {}
    installer(monkeypatch, rec, error=RuntimeError("429 rate limit exceeded"))
    provider = _make(cls)(api_key="k")
    with pytest.raises(LLMError) as excinfo:
        provider.complete_json("p", "s")
    # The shared classifier must still see the rate-limit signal through the
    # LLMError wrapper (message text and/or cause chain).
    assert _is_rate_limit_error(excinfo.value)
