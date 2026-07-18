"""Tests organized by feature ownership."""

from __future__ import annotations


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
