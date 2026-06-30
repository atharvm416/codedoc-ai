"""Private per-file record metadata registry.

CodeDoc file records may carry a small set of *private* keys that are
persisted through JSON, Markdown (embedded view), live backups, and resume
reconstruction, but are never rendered into the visible Markdown prose.

Only keys explicitly listed in :data:`PRIVATE_RECORD_KEYS` are preserved.
Arbitrary underscore-prefixed model output is *not* carried — this prevents a
weak model from smuggling unbounded private-looking fields into the output.

The registry carries the per-file cache-identity keys
``_analysis_revision`` and ``_analysis_mode`` (see :data:`CACHE_IDENTITY_KEYS`).
Focused tests may monkeypatch the module-level ``PRIVATE_RECORD_KEYS`` with a
synthetic key to exercise the carry behaviour.
"""

from __future__ import annotations

from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST

# Cache identity.  Bump ``ANALYSIS_REVISION`` whenever the generation strategy
# changes in a way that should invalidate previously cached records.
#
# The current revision is ``file-doc-v2``: prompt semantics (precise local-symbol
# / export / usage-example definitions, the head-plus-tail truncation marker) and
# response cleaning define the contract for both ``single`` and ``triple`` modes.
# Older ``file-doc-v1`` records remain readable but are reprocessed exactly once
# under the current contract before reuse.
ANALYSIS_REVISION = "file-doc-v2"

# Per-file truncation identity token.  The head-plus-tail truncation of an
# oversized file depends on the effective ``max_content_chars`` ceiling and the
# ``truncation_head_ratio``.  Both participate in cache identity: otherwise
# changing either would change the truncated prompt without invalidating cached
# records, so an incremental re-run would silently reuse stale documentation (the
# remedy the truncation warning recommends, "raise max_content_chars", would have
# no effect on a cached run).  ``_max_context_revision`` encodes both for any file
# large enough to be truncated; a file that fits within the ceiling carries no
# value and stays reusable across ceiling/ratio changes.  Bump the token below if
# the truncation algorithm itself changes.
MAX_CONTEXT_REVISION = "truncate-v1"

# Cache-identity keys: private keys that, together with the content hash, decide
# whether a stored record may be reused.  This is a *narrower* set than
# ``PRIVATE_RECORD_KEYS`` — private metadata is persisted, but only these keys
# gate reuse.  Every reuse source must compare all of these.
#
# ``_prompt_profile_digest`` is part of the set so an active prompt-customization
# profile precisely invalidates the files it affects.  It is omitted from a record
# when no profile is active (developer-standard); the absent-default mapping below
# makes an omitted key and an explicit ``NO_PROMPT_PROFILE_DIGEST`` compare equal,
# so legacy and no-profile records stay reusable.
CACHE_IDENTITY_KEYS: frozenset[str] = frozenset(
    {
        "_analysis_revision",
        "_analysis_mode",
        "_max_context_revision",
        "_prompt_profile_digest",
    }
)

# Absent-default mapping for cache-identity comparison.  A key listed
# here compares as its default value when absent from a record, so an omitted key
# and an explicitly stored default are equivalent.  Keys not listed here default
# to ``None`` when absent (the historical behaviour for the other identity keys).
_CACHE_KEY_ABSENT_DEFAULTS: dict[str, str] = {
    "_prompt_profile_digest": NO_PROMPT_PROFILE_DIGEST,
}

# Registered private record keys: persisted through JSON / Markdown / live
# backups / resume, never rendered into visible prose.  Must include every
# cache-identity key so the carrier preserves them.
PRIVATE_RECORD_KEYS: frozenset[str] = frozenset(CACHE_IDENTITY_KEYS)


def normalized_identity_value(key: str, source: dict) -> object:
    """Return ``source[key]`` normalized through :data:`_CACHE_KEY_ABSENT_DEFAULTS`.

    An absent key resolves to its registered absent-default (or ``None`` when the
    key has no mapping), so the centralized reuse predicate can normalize both the
    stored and the expected side identically.
    """
    if key in source:
        return source[key]
    return _CACHE_KEY_ABSENT_DEFAULTS.get(key)


def expected_analysis_identity(analysis_mode: str) -> dict[str, str]:
    """Return the run-level cache-identity keys a freshly generated record carries.

    This is the part shared by every file in a run.  The per-file truncation part
    (:func:`expected_max_context_revision`) is merged in separately because it
    depends on each file's source length.
    """
    return {"_analysis_revision": ANALYSIS_REVISION, "_analysis_mode": analysis_mode}


def expected_max_context_revision(
    source_chars: int,
    *,
    max_chars: int,
    head_ratio: float,
) -> str | None:
    """Return the per-file truncation cache-identity value, or ``None``.

    ``None`` when the file fits within *max_chars* (``source_chars <= max_chars``):
    it is sent whole, so neither the ceiling nor the head ratio affects its prompt
    and the record stays reusable across ceiling/ratio changes.  For a file large
    enough to be truncated (``source_chars > max_chars``), a stable string
    encoding the effective ceiling and head ratio, e.g.
    ``"truncate-v1:max=12000:head=0.7000"``.  The head ratio is rendered with a
    fixed 4-place decimal so byte-identical configuration yields byte-identical
    identities.

    The value never encodes *source_chars* itself, so two truncated files under
    the same configuration share an identity — only whether a file is truncated,
    plus the ceiling and ratio, matter.
    """
    if source_chars > max_chars:
        return f"{MAX_CONTEXT_REVISION}:max={int(max_chars)}:head={float(head_ratio):.4f}"
    return None


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
