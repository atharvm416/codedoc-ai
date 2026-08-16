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

import hashlib
import json

from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST
from codedoc.parser.source_structure import normalize_rel_path
from codedoc.core.file_division import (
    FINAL_SYNTHESIS_REVISION,
    LEAF_CAPSULE_SCHEMA_REVISION,
    LEDGER_SCHEMA_REVISION,
    MAX_ATOMS_PER_FILE,
    MAX_CHUNKS_PER_FILE,
    MAX_KNOWN_SYMBOLS_PER_CHUNK,
    MAX_LEAF_CAPSULE_CANONICAL_CHARS,
    MAX_LEAF_PROMPT_METADATA_CHARS,
    MAX_LEDGER_SYNOPSIS_CHARS,
    MAX_REDUCTION_CAPSULE_CANONICAL_CHARS,
    MAX_REDUCTION_NARRATIVE_CHARS,
    MAX_REDUCTION_TREE_DEPTH,
    MAX_SYMBOLS_PER_FILE,
    MAX_UNITS_PER_FILE,
    PACKER_SCHEMA_REVISION,
    REDUCER_PROMPT_REVISION,
    REDUCTION_CAPSULE_SCHEMA_REVISION,
    REDUCTION_ENVELOPE_OVERHEAD_CHARS,
    REDUCTION_PACKING_REVISION,
    STRUCTURE_SCHEMA_REVISION,
    UNIT_SCHEMA_REVISION,
)
from codedoc.parser.tree_sitter_structure import PARSER_PACKAGE_VERSION

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

# Rejected predecessor value from the 0.14.1 fresh-only split contract. Current
# production code never stamps it, but the key remains registered so predecessor
# records survive every public-format round trip and mismatch the current absent
# expected value without inferring policy from the package version.
FRESH_SPLIT_REUSE_CONTRACT = "fresh-only-v1"

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
        "_large_file_identity",
        "_split_reuse_contract",
        "_ordinary_path_identity",
    }
)

# Absent-default mapping for cache-identity comparison.  A key listed
# here compares as its default value when absent from a record, so an omitted key
# and an explicitly stored default are equivalent.  Keys not listed here default
# to ``None`` when absent (the historical behaviour for the other identity keys).
# ``_ordinary_path_identity`` is deliberately unregistered here: every pre-0.14.4
# ordinary/truncate-path record lacks it, so an absent key must normalize to
# ``None`` and compare unequal to any expected (non-``None``) value -- leaving
# every such legacy record invalid until it is regenerated under the current
# path-bound contract (see ``_record_is_reusable``'s ``rel_path`` keyword).
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
# ``_ordinary_path_identity`` is appended last so every existing record's key
# order is preserved.
PRIVATE_KEY_ORDER: tuple[str, ...] = (
    "_analysis_revision",
    "_analysis_mode",
    "_max_context_revision",
    "_prompt_profile_digest",
    "_large_file_identity",
    "_split_reuse_contract",
    "_ordinary_path_identity",
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
        ratio = float(head_ratio)
        rendered = f"{ratio:.4f}"
        identity = f"{MAX_CONTEXT_REVISION}:max={int(max_chars)}:head={rendered}"
        # The 4-place rendering aliases ratios that differ beyond the fourth
        # decimal (e.g. 0.7000501 and 0.7001499 both render "0.7001") even
        # though they produce different head/tail splits and therefore
        # different prompts. Append the exact round-trippable value only when
        # rounding actually lost information, so every ratio representable at
        # four decimals — including the 0.70 default — keeps its existing
        # identity bytes and stays reusable.
        if float(rendered) != ratio:
            identity = f"{identity}:exact={ratio!r}"
        return identity
    return None


def expected_large_file_identity(
    *,
    source_chars: int,
    max_chars: int,
    rel_path: str,
    division_plan_digest: str,
    reduction_tree_digest: str,
    structural_mode: str,
    imports_digest: str,
) -> str | None:
    """Return the effective-split cache-identity value, or ``None``.

    ``None`` for a record whose source fits `max_chars` (never split) or that
    was never a completed effective split.  There is no capacity-fallback or
    effective-truncate identity case: a completed record carrying this key
    always describes a complete effective split (D8/D11/section 13).  The
    revision prefix (``large-file-v3``) is distinct from every predecessor's
    so an earlier split identity (including the dormant ``large-file-v2``)
    can never satisfy this comparison.  Binds the deterministic imports
    digest (section 6) so an equal-length import change invalidates the
    completed record.
    """
    if source_chars <= max_chars:
        return None
    payload = {
        "revision": "large-file-identity-v3",
        "requested_strategy": "split",
        "effective_strategy": "split",
        "source_budget": int(max_chars),
        "path": rel_path,
        "division_plan_digest": division_plan_digest,
        "reduction_tree_digest": reduction_tree_digest,
        "structural_mode": structural_mode,
        "imports_digest": imports_digest,
        "parser_package_version": PARSER_PACKAGE_VERSION,
        "grammar_availability_mode": (
            "bundled-grammar-or-complete-lexical-fallback-v1"
        ),
        "bounds": {
            "atoms": MAX_ATOMS_PER_FILE,
            "symbols": MAX_SYMBOLS_PER_FILE,
            "units": MAX_UNITS_PER_FILE,
            "chunks": MAX_CHUNKS_PER_FILE,
            "known_symbols_per_chunk": MAX_KNOWN_SYMBOLS_PER_CHUNK,
            "leaf_prompt_metadata_chars": MAX_LEAF_PROMPT_METADATA_CHARS,
        },
        "reduction_bounds": {
            "leaf_capsule_chars": MAX_LEAF_CAPSULE_CANONICAL_CHARS,
            "reduction_capsule_chars": MAX_REDUCTION_CAPSULE_CANONICAL_CHARS,
            "reduction_envelope_overhead": REDUCTION_ENVELOPE_OVERHEAD_CHARS,
            "final_narrative_chars": MAX_REDUCTION_NARRATIVE_CHARS,
            "final_ledger_synopsis_chars": MAX_LEDGER_SYNOPSIS_CHARS,
            "final_envelope": "exact-worst-case-v2",
            "max_tree_depth": MAX_REDUCTION_TREE_DEPTH,
        },
        "revisions": {
            "structure": STRUCTURE_SCHEMA_REVISION,
            "units": UNIT_SCHEMA_REVISION,
            "packer": PACKER_SCHEMA_REVISION,
            "leaf_capsule": LEAF_CAPSULE_SCHEMA_REVISION,
            "ledger": LEDGER_SCHEMA_REVISION,
            "reduction_capsule": REDUCTION_CAPSULE_SCHEMA_REVISION,
            "reduction_packing": REDUCTION_PACKING_REVISION,
            "reducer_prompt": REDUCER_PROMPT_REVISION,
            "final_synthesis": FINAL_SYNTHESIS_REVISION,
        },
    }
    return "large-file-v3:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def expected_ordinary_path_identity(rel_path: str) -> str:
    """Return the ordinary-only path-bound cache identity for *rel_path*.

    Ordinary identical-content reuse (same content hash, same cache identity)
    must never copy model-authored, path-specific documentation from one
    relative path to another: only *this exact path*'s prior record may ever
    satisfy it.  ``_record_is_reusable``'s ``rel_path`` keyword compares this
    value against the record's own stored ``_ordinary_path_identity``, so a
    record whose stored ``path`` and stamped identity do not both match the
    destination path is never reusable there.

    Deliberately serializes with ``ensure_ascii=True`` -- unlike
    :func:`codedoc.core.file_division.canonical_json`, which uses
    ``ensure_ascii=False`` -- so a non-ASCII path always escapes to the same
    bytes regardless of platform or Python version; do not switch this to the
    shared ``canonical_json`` helper.
    """
    payload = {"revision": "ordinary-path-v1", "path": normalize_rel_path(rel_path)}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return "ordinary-path-v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
