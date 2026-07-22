"""Private per-file record metadata registry.

CodeDoc file records may carry a small set of *private* keys that are
persisted through JSON, Markdown (embedded view), crash recovery, and resume
reconstruction, but are never rendered into the visible Markdown prose.

Only keys explicitly listed in :data:`PRIVATE_RECORD_KEYS` are preserved.
Arbitrary underscore-prefixed model output is *not* carried — this prevents a
weak model from smuggling unbounded private-looking fields into the output.

Two module-level names own two distinct responsibilities and must not be
conflated:

- :data:`PRIVATE_KEY_ORDER` owns the canonical *ordering* of the production
  private keys.  Iterating a ``set`` of strings depends on the process hash
  seed, so a set-ordered carrier would insert the same keys into otherwise
  identical records in different orders across processes — and neither
  serializer sorts keys, so insertion order is output order.
- :data:`PRIVATE_RECORD_KEYS` owns *membership*: which keys are carried at all.
  It is derived from :data:`PRIVATE_KEY_ORDER` so ordering and membership
  cannot drift apart.

Ordering never widens membership.  :func:`carry_private_keys` resolves
``PRIVATE_RECORD_KEYS`` at call time, so focused tests may monkeypatch the
module-level name with a synthetic ``frozenset`` to exercise the carry
behaviour; an unregistered production key is then not carried.

The registry carries the per-file cache-identity keys
``_analysis_revision`` and ``_analysis_mode`` (see :data:`CACHE_IDENTITY_KEYS`).
"""

from __future__ import annotations

from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST

# Cache identity.  Bump ``ANALYSIS_REVISION`` whenever the generation strategy
# changes in a way that should invalidate previously cached records.
#
# The current revision is ``file-doc-v3``: the strengthened exact-JSON response
# rules shared across all four prompts, plus stricter response acceptance
# (registry-required-field validation and rejection of a response that retains
# none of its requested fields), change generation semantics for both ``single``
# and ``triple`` modes even though the rendered requested-shape block — and the
# ``_prompt_profile_digest`` computed from it — are unchanged.  Older
# ``file-doc-v2`` (and ``file-doc-v1``) records remain readable but are
# reprocessed exactly once under the current contract before reuse.
ANALYSIS_REVISION = "file-doc-v3"

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

# Canonical insertion order for the production private keys.  This is the order
# in which a freshly generated record already acquires them — the orchestrator
# stamps ``expected_analysis_identity()`` (``_analysis_revision`` then
# ``_analysis_mode``) first, then ``_max_context_revision``, then
# ``_prompt_profile_digest`` — so it preserves the record's own construction
# order and keeps run-level identity ahead of per-file identity.  Plain
# ``sorted()`` would also be deterministic but would gratuitously reorder every
# freshly generated record; do not "simplify" this to alphabetical order.
PRIVATE_KEY_ORDER: tuple[str, ...] = (
    "_analysis_revision",
    "_analysis_mode",
    "_max_context_revision",
    "_prompt_profile_digest",
)

# Registered private record keys: persisted through JSON / Markdown / live
# backups / resume, never rendered into visible prose.  Derived from
# ``PRIVATE_KEY_ORDER`` so ordering and membership cannot drift.  Must include
# every cache-identity key so the carrier preserves them; the relationship is
# asserted by a focused test rather than inverted here, because deriving
# ``CACHE_IDENTITY_KEYS`` from this set would silently promote any future
# persistence-only private key into a cache-invalidating one.
PRIVATE_RECORD_KEYS: frozenset[str] = frozenset(PRIVATE_KEY_ORDER)


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


def _ordered_private_keys(registered: frozenset[str]) -> list[str]:
    """Return *registered* in canonical order.

    Known production keys come first, in :data:`PRIVATE_KEY_ORDER` order and
    filtered to those actually registered; any remaining registered key (a
    synthetic test key, or a future extension) follows in sorted order.  The
    result is therefore deterministic for any registry, and ordering never adds
    a key that membership did not authorize.
    """
    known = [key for key in PRIVATE_KEY_ORDER if key in registered]
    return known + sorted(registered.difference(PRIVATE_KEY_ORDER))


def carry_private_keys(source: dict, target: dict) -> None:
    """Copy every registered private key present in *source* into *target*.

    Rules:

    - only keys in :data:`PRIVATE_RECORD_KEYS` are considered, resolved at call
      time so the module-level name stays monkeypatchable;
    - keys are inserted in the canonical order defined by
      :func:`_ordered_private_keys`, independent of ``source``'s own key order
      and of the process hash seed;
    - absent keys are skipped;
    - present values are copied exactly — including falsey values such as
      ``None``, ``""``, ``False``, ``0``, ``[]`` and ``{}``;
    - *source* is never mutated;
    - repeated calls are idempotent, and re-copying an already-present key
      updates its value without moving its established position.

    Arbitrary underscore-prefixed keys are never preserved — only the
    explicitly registered ones.

    The registry is iterated, never *source*: testing ``key in source`` keeps
    the O(1) dict lookup and, critically, keeps output order independent of the
    caller's dictionary order.
    """
    if not isinstance(source, dict) or not isinstance(target, dict):
        return
    for key in _ordered_private_keys(PRIVATE_RECORD_KEYS):
        if key in source:
            target[key] = source[key]
