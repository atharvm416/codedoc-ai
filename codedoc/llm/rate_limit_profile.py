"""
Provider-aware rate-limit profiles for codedoc.

0.8.1: Extracts provider-specific rate-limit signal lists and backoff
parameters from the hardcoded ``_RATE_LIMIT_SIGNALS`` tuple in
``pipeline.py`` into configurable per-provider profiles.

The pipeline can now:
- Use the correct signals for each provider (tighter false-positive
  avoidance than the global union).
- Apply exponential backoff between parallel ladder rungs that scales
  with the provider's typical recovery time.
- Allow users to extend or restrict signals via config without touching
  source code (``rate_limit_signals_add`` / ``rate_limit_signals_remove``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RateLimitProfile:
    """Rate-limit classification and backoff policy for one provider.

    Attributes
    ----------
    provider:
        Normalized provider name (``"openai"``, ``"anthropic"``, ``"gemini"``,
        or ``"default"``).
    signals:
        Case-insensitive substrings used to classify an exception as a
        rate-limit event.  Each signal is checked with ``signal in msg.lower()``.
    min_backoff_s:
        Minimum inter-rung sleep in seconds.  Used as the base for the
        exponential formula: ``min(min_backoff_s * backoff_scale ** rung, cap)``.
        Set to ``0`` to disable computed backoff entirely.
    backoff_scale:
        Exponential multiplier.  A value of ``1.5`` grows the sleep by 50 %
        per step-down rung.
    """

    provider: str
    signals: list[str] = field(default_factory=list)
    min_backoff_s: float = 5.0
    backoff_scale: float = 1.5


# ---------------------------------------------------------------------------
# Default provider profiles
# ---------------------------------------------------------------------------

PROVIDER_PROFILES: dict[str, RateLimitProfile] = {
    "openai": RateLimitProfile(
        provider="openai",
        signals=[
            "429",
            "rate limit",
            "rate_limit",
            "too many requests",
            "tokens per min",
            "tpm",
            "quota",
        ],
        min_backoff_s=5.0,
        backoff_scale=1.5,
    ),
    "anthropic": RateLimitProfile(
        provider="anthropic",
        signals=[
            "529",
            "overloaded",
            "rate_limit",
            "429",
        ],
        min_backoff_s=10.0,
        backoff_scale=2.0,
    ),
    "gemini": RateLimitProfile(
        provider="gemini",
        signals=[
            "resource_exhausted",
            "quota",
            "429",
            "503",
        ],
        min_backoff_s=8.0,
        backoff_scale=1.5,
    ),
}

# The "default" profile is the union of all provider signals, so it correctly
# classifies rate-limit events from any provider or proxy server.
_ALL_SIGNALS: list[str] = []
_seen: set[str] = set()
for _p in ("openai", "anthropic", "gemini"):
    for _s in PROVIDER_PROFILES[_p].signals:
        if _s not in _seen:
            _ALL_SIGNALS.append(_s)
            _seen.add(_s)

PROVIDER_PROFILES["default"] = RateLimitProfile(
    provider="default",
    signals=_ALL_SIGNALS,
    min_backoff_s=5.0,
    backoff_scale=1.5,
)

del _seen, _ALL_SIGNALS, _p, _s  # clean up module namespace


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------

def get_rate_limit_profile(
    provider_name: str,
    config: dict | None = None,
) -> RateLimitProfile:
    """Return the rate-limit profile for *provider_name*, with config overrides applied.

    The function never mutates ``PROVIDER_PROFILES``.  A fresh
    :class:`RateLimitProfile` is returned whenever a config override is present.

    Parameters
    ----------
    provider_name:
        Provider identifier, e.g. ``"openai"``, ``"anthropic"``, ``"gemini"``,
        or any custom gateway name.  Unknown names fall back to ``"default"``.
    config:
        Resolved config dict from :func:`codedoc.core.loader.load_config`.
        Recognised override keys:

        - ``rate_limit_backoff_s`` (``float | None``): overrides ``min_backoff_s``
          globally.  Set to ``0`` to disable computed inter-rung sleep entirely.
        - ``rate_limit_backoff_scale`` (``float | None``): overrides
          ``backoff_scale`` globally.
        - ``rate_limit_signals_add`` (``list[str]``): extra signal strings
          appended to the profile's signal list.
        - ``rate_limit_signals_remove`` (``list[str]``): signal strings to
          remove from the profile's signal list.
    """
    config = config or {}
    normalized = (provider_name or "default").strip().lower()
    base = PROVIDER_PROFILES.get(normalized, PROVIDER_PROFILES["default"])

    # Start with a copy of the base signals list so the module default is never
    # mutated even when add / remove overrides are applied.
    signals = list(base.signals)

    # Normalize all signals to lower-case so comparisons in _is_rate_limit_error
    # are always case-insensitive.  The pipeline already lowercases the message,
    # so the signals must also be lower-case to match correctly.
    add = [s.lower() for s in (config.get("rate_limit_signals_add") or [])]
    seen = set(signals)
    for sig in add:
        if sig not in seen:
            signals.append(sig)
            seen.add(sig)

    # Normalize remove entries to lower-case so "RESOURCE_EXHAUSTED" removes
    # the default "resource_exhausted" as expected.
    remove_set = {s.lower() for s in (config.get("rate_limit_signals_remove") or [])}
    if remove_set:
        signals = [s for s in signals if s not in remove_set]

    min_backoff_s = base.min_backoff_s
    backoff_scale = base.backoff_scale

    override_backoff = config.get("rate_limit_backoff_s")
    if override_backoff is not None:
        min_backoff_s = float(override_backoff)

    override_scale = config.get("rate_limit_backoff_scale")
    if override_scale is not None:
        backoff_scale = float(override_scale)

    # Return a new instance only when something actually changed so callers
    # that compare by identity to PROVIDER_PROFILES still work in tests.
    if (
        signals == base.signals
        and min_backoff_s == base.min_backoff_s
        and backoff_scale == base.backoff_scale
    ):
        return base

    return RateLimitProfile(
        provider=normalized,
        signals=signals,
        min_backoff_s=min_backoff_s,
        backoff_scale=backoff_scale,
    )
