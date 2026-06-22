"""Private per-file record metadata registry (0.9.3).

CodeDoc file records may carry a small set of *private* keys that are
persisted through JSON, Markdown (embedded view), live backups, and resume
reconstruction, but are never rendered into the visible Markdown prose.

Only keys explicitly listed in :data:`PRIVATE_RECORD_KEYS` are preserved.
Arbitrary underscore-prefixed model output is *not* carried — this prevents a
weak model from smuggling unbounded private-looking fields into the output.

As of 0.10.0 the registry carries the per-file cache-identity keys
``_analysis_revision`` and ``_analysis_mode`` (see :data:`CACHE_IDENTITY_KEYS`).
Focused tests may monkeypatch the module-level ``PRIVATE_RECORD_KEYS`` with a
synthetic key to exercise the carry behaviour.
"""

from __future__ import annotations

# 0.10.0 cache identity.  Bump ``ANALYSIS_REVISION`` whenever the generation
# strategy changes in a way that should invalidate previously cached records.
ANALYSIS_REVISION = "file-doc-v1"

# Cache-identity keys: private keys that, together with the content hash, decide
# whether a stored record may be reused.  This is a *narrower* set than
# ``PRIVATE_RECORD_KEYS`` — private metadata is persisted, but only these keys
# gate reuse.  Every reuse source must compare all of these.
CACHE_IDENTITY_KEYS: frozenset[str] = frozenset({"_analysis_revision", "_analysis_mode"})

# Registered private record keys: persisted through JSON / Markdown / live
# backups / resume, never rendered into visible prose.  Must include every
# cache-identity key so the carrier preserves them.
PRIVATE_RECORD_KEYS: frozenset[str] = frozenset(CACHE_IDENTITY_KEYS)


def expected_analysis_identity(analysis_mode: str) -> dict[str, str]:
    """Return the cache-identity keys a freshly generated record must carry."""
    return {"_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": analysis_mode}


def carry_private_keys(source: dict, target: dict) -> None:
    """Copy every registered private key present in *source* into *target*.

    Rules:

    - only keys in :data:`PRIVATE_RECORD_KEYS` are considered;
    - absent keys are skipped;
    - present values are copied exactly — including falsey values such as
      ``None``, ``""``, ``False``, ``0``, ``[]`` and ``{}``;
    - *source* is never mutated;
    - repeated calls are idempotent.

    Arbitrary underscore-prefixed keys are never preserved — only the
    explicitly registered ones.
    """
    if not isinstance(source, dict) or not isinstance(target, dict):
        return
    for key in PRIVATE_RECORD_KEYS:
        if key in source:
            target[key] = source[key]
