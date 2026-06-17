"""Private per-file record metadata registry (0.9.3).

CodeDoc file records may carry a small set of *private* keys that are
persisted through JSON, Markdown (embedded view), live backups, and resume
reconstruction, but are never rendered into the visible Markdown prose.

Only keys explicitly listed in :data:`PRIVATE_RECORD_KEYS` are preserved.
Arbitrary underscore-prefixed model output is *not* carried — this prevents a
weak model from smuggling unbounded private-looking fields into the output.

The production registry is intentionally **empty** in this release; it exists
as plumbing for later features.  Focused tests may monkeypatch the module-level
``PRIVATE_RECORD_KEYS`` with a synthetic key to exercise the carry behaviour.
"""

from __future__ import annotations

# Registered private record keys.  Empty in production for 0.9.3.
PRIVATE_RECORD_KEYS: frozenset[str] = frozenset()


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
