"""Deterministic large-file division, hierarchical reduction, and recovery
contracts for the ``single + split`` large-file workflow.

This module owns every provider-free planning concern for split:

- semantic-first division of one canonical decoded source snapshot into
  bounded, ordered leaf chunks (:func:`build_division_plan`);
- the deterministic hierarchical reduction tree that bounds any leaf-chunk
  count before a provider is ever created (:func:`build_reduction_tree`);
- the lossless internal structured fact ledger merged from cleaned leaf
  capsules (:func:`build_fact_ledger` / :func:`merge_leaf_capsules`);
- the bounded final-synthesis manifest (:func:`final_synthesis_input`);
- node-keyed split-partial recovery state (schema version 3) and its
  dependency-validated reuse predicate; and
- the provider-free provider/model/effective-endpoint execution identity
  shared by every recoverable node.

Terminology used throughout this module (design decisions D4/D6):

- a **semantic unit** is one function/method/class/type declaration, one
  top-level declaration, one import/header region, or (lexical fallback) one
  physical source line — in this implementation, one :class:`~codedoc.parser.
  source_structure.Atom` plus its associated symbol (if any);
- a **chunk** is the leaf AI-call boundary: either several complete adjacent
  semantic units packed together, or exactly one continuation fragment of a
  single oversized semantic unit;
- every parser atom retains its own :class:`SemanticUnitIdentity`, independently
  of leaf-call packing. A fitting chunk may reference several complete units;
  only continuation chunks of the same oversized unit share a reduction group
  and are consolidated before their narrative is mixed with any other chunk's.

No public output (JSON, Markdown, embedded view) is produced from this
module.  Only the final synthesized file-level record — assembled elsewhere
from the outputs this module coordinates — is ever published.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from codedoc.parser.source_structure import (
    Atom,
    SourceRange,
    StructureResult,
    SymbolFact,
    normalize_rel_path,
)
from codedoc.parser.tree_sitter_structure import (
    PARSER_PACKAGE_VERSION,
    SourceIndex,
    extract_structure,
    validate_structure,
)
from codedoc.core.result_assembly import flat_combined_result

# ---------------------------------------------------------------------------
# Versioned internal revisions
# ---------------------------------------------------------------------------
# Every revision below participates in `_large_file_identity` (record_meta.py)
# and/or node execution identity.  Bump the specific revision whose contract
# changed; do not bump the global `ANALYSIS_REVISION` for split-only changes.

STRUCTURE_SCHEMA_REVISION = "source-structure-v2"
UNIT_SCHEMA_REVISION = "semantic-unit-v3"
PACKER_SCHEMA_REVISION = "division-packer-v5"
# Advanced from v4: signature-aware matching became load-bearing after
# leaf-capsule-v4 was already active, so a predecessor signatureless leaf
# cannot validate as a current checkpoint.
LEAF_CAPSULE_SCHEMA_REVISION = "leaf-capsule-v5"
# Advanced from v5. Bound into final-node execution identity, the final-node
# exact input digest, and the completed split identity; a ledger revision
# change alone reruns final synthesis but preserves compatible leaves and
# reducers (they consume narratives, not the final structured ledger).
LEDGER_SCHEMA_REVISION = "fact-ledger-v6"
REDUCTION_CAPSULE_SCHEMA_REVISION = "reduction-capsule-v1"
REDUCTION_PACKING_REVISION = "reduction-packing-v4"
REDUCER_PROMPT_REVISION = "file-reduction-v1"
FINAL_SYNTHESIS_REVISION = "file-synthesis-v3"
# Advanced from v5 to division-execution-v6 alongside the leaf/ledger bumps
# above.
EXECUTION_IDENTITY_SCHEMA_REVISION = "division-execution-v6"

# ---------------------------------------------------------------------------
# Deterministic safety bounds
# ---------------------------------------------------------------------------
# All bounds below are internal, versioned, developer-owned constants (D3,
# D6).  Focused tests freeze boundary behavior by monkeypatching the module
# attribute, not by constructing pathologically large fixtures.

MAX_ATOMS_PER_FILE = 4096
MAX_SYMBOLS_PER_FILE = 8192
MAX_UNITS_PER_FILE = 4096
MAX_CHUNKS_PER_FILE = 256
MAX_STRUCTURE_DIAGNOSTICS = 32

# Fixed internal leaf-capsule schema bounds. Symbol and export fields use the
# corresponding established whole-file limits; the leaf description is capped
# at the 300-character child-narrative ceiling because it can feed a reducer
# directly. Valid visible facts must not be made unusably short merely because
# they were discovered through the split path. The fixed-capsule cleaner
# rejects over-limit fields/items through the correction contract instead of
# truncating or silently publishing an incomplete fact ledger.
MAX_LEAF_DESCRIPTION_CHARS = 300
MAX_LEAF_SYMBOL_NAME_CHARS = 128
MAX_LEAF_SYMBOL_DESCRIPTION_CHARS = 300
MAX_LEAF_SYMBOL_ITEMS_PER_KIND = 12
MAX_LEAF_EXPORT_ITEMS = 32
MAX_LEAF_EXPORT_ITEM_CHARS = 256
# An optional bounded signature is part of the fixed leaf capsule because the
# ledger's semantic key uses it to keep same-named overloads distinct (D7/section 8).
# Without it the cleaner would strip the only field that separates
# `run(int)` from `run(str)`, and the ledger would silently publish one fact.
MAX_LEAF_SYMBOL_SIGNATURE_CHARS = 256

# Parser-derived names are optional prompt grounding, not accepted response
# facts. Bound that hint independently so declaration-heavy semantic units
# cannot make prompt metadata grow without limit. Source coverage and fixed
# leaf response bounds are enforced independently.
MAX_KNOWN_SYMBOLS_PER_CHUNK = 32
MAX_LEAF_PROMPT_METADATA_CHARS = 32 * 1024

# Worst-case canonical (cleaned, compact-JSON) sizes. Leaf capsule size remains
# part of metadata and final-synthesis planning, but reducer fan-in is based on
# the exact rendered child-narrative manifest because reducers never receive
# leaf JSON capsules.
_MAXIMUM_JSON_ESCAPED_CHAR = "\x00"
_MAX_LEAF_CAPSULE = {
    "description": _MAXIMUM_JSON_ESCAPED_CHAR * MAX_LEAF_DESCRIPTION_CHARS,
    "functions": [
        {
            "name": _MAXIMUM_JSON_ESCAPED_CHAR * MAX_LEAF_SYMBOL_NAME_CHARS,
            "description": (
                _MAXIMUM_JSON_ESCAPED_CHAR
                * MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
            ),
            "signature": (
                _MAXIMUM_JSON_ESCAPED_CHAR
                * MAX_LEAF_SYMBOL_SIGNATURE_CHARS
            ),
        }
        for _ in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
    ],
    "classes": [
        {
            "name": _MAXIMUM_JSON_ESCAPED_CHAR * MAX_LEAF_SYMBOL_NAME_CHARS,
            "description": (
                _MAXIMUM_JSON_ESCAPED_CHAR
                * MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
            ),
            "signature": (
                _MAXIMUM_JSON_ESCAPED_CHAR
                * MAX_LEAF_SYMBOL_SIGNATURE_CHARS
            ),
        }
        for _ in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
    ],
    "exports": [
        _MAXIMUM_JSON_ESCAPED_CHAR * MAX_LEAF_EXPORT_ITEM_CHARS
        for _ in range(MAX_LEAF_EXPORT_ITEMS)
    ],
}
MAX_LEAF_CAPSULE_CANONICAL_CHARS = len(
    json.dumps(
        _MAX_LEAF_CAPSULE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
)
del _MAX_LEAF_CAPSULE
MAX_REDUCTION_NARRATIVE_CHARS = 300
MAX_REDUCTION_CAPSULE_CANONICAL_CHARS = len(
    json.dumps(
        {
            "narrative": (
                _MAXIMUM_JSON_ESCAPED_CHAR
                * MAX_REDUCTION_NARRATIVE_CHARS
            )
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
)

# A reducer child renders as "- " plus one narrative; adjacent children add
# one newline. Canonical reduction-envelope metadata is separately reserved
# from `max_content_chars`. Final synthesis is intentionally not sized through
# a fixed overhead: planning constructs its exact deterministic fields plus
# maximally populated bounded response-derived fields.
REDUCTION_MANIFEST_ITEM_PREFIX = "- "
REDUCTION_MANIFEST_ITEM_SEPARATOR = "\n"
REDUCTION_ENVELOPE_OVERHEAD_CHARS = 80
MAX_LEDGER_SYNOPSIS_CHARS = 3000
MAX_REDUCTION_TREE_DEPTH = 6

# Dry-run complexity advisory thresholds (D6a).  Both advisories fire only
# when the measured value is strictly greater than the threshold.
SPLIT_COMPLEXITY_ADVISORY_CHUNKS = 24
SPLIT_COMPLEXITY_ADVISORY_REDUCTION_DEPTH = 2

# Node-keyed split-partial container schema version.  Distinct from the
# enclosing run-level `_RECOVERY_IDENTITY_VERSION` (resume.py), which stays 1
# because the recovery-container contract is unchanged (D11/D12).  Advanced
# from 2: the per-node `reduction_tree_digest` gate is removed (a topology-only
# change must not invalidate an otherwise-compatible leaf), a per-node exact
# stage-local input digest is added, and a bounded quarantine map is added.
SPLIT_PARTIAL_SCHEMA_VERSION = 3
# Schema 2 (node-keyed, per-node tree-digest gated) remains recognizable for
# preserve-first guidance but is dormant: never executable and never
# automatically migrated (section 16).
DORMANT_SPLIT_PARTIAL_SCHEMA_VERSION = 2
# The predecessor ordered-prefix partial's schema version, recognized only so
# it can be rejected with the D11 preserve-or-move-aside remedy.
LEGACY_SPLIT_PARTIAL_SCHEMA_VERSION = 1

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
BlockedReason = Literal[
    "atom-cap",
    "symbol-cap",
    "unit-cap",
    "chunk-cap",
    "reduction-envelope-cap",
    "reduction-fan-in-cap",
    "reduction-depth-cap",
    "final-synthesis-envelope-cap",
]

# The frozen structural-planning evaluation order (D3/D6 "Capacity failures").
# `split_blocked_by_reason` sums exactly to `split_blocked_files` because every
# blocked file is classified under exactly the first reason it violates.
BLOCKED_REASON_ORDER: tuple[BlockedReason, ...] = (
    "atom-cap",
    "symbol-cap",
    "unit-cap",
    "chunk-cap",
    "reduction-envelope-cap",
    "reduction-fan-in-cap",
    "reduction-depth-cap",
    "final-synthesis-envelope-cap",
)


class DuplicateCanonicalKeyError(ValueError):
    pass


class DivisionInternalDefect(ValueError):
    """A non-capacity CodeDoc invariant failure in deterministic division.

    Never a capacity outcome: aborts the whole dry or real run provider-free,
    before writer/provider creation, through the internal/unexpected CLI exit
    path (see D8).
    """


class SplitCapacityBlocked(Exception):
    """One requested-split file cannot be completely planned, provider-free.

    Carries the single first-failing named reason under the frozen evaluation
    order.  Never a truncate fallback: a real run must raise `ConfigError`
    listing every blocked `(rel_path, reason)` pair before any provider or
    writer is created; a dry run reports the same pairs without mutation.
    """

    def __init__(self, rel_path: str, reason: BlockedReason) -> None:
        self.rel_path = normalize_rel_path(rel_path)
        self.reason = reason
        super().__init__(f"{self.rel_path}: blocked by capacity reason {reason!r}")


class SplitRecoveryStateError(ValueError):
    """A current schema-3 recovery container cannot be safely executed.

    Planning translates this into preserve-first ``ConfigError`` guidance
    before writer initialization or provider construction.  It is distinct
    from a division invariant defect: the source plan may be sound while the
    retained recovery state is foreign, unplanned, or exceeds its hard bound.
    """


# ---------------------------------------------------------------------------
# Canonical JSON / digest primitives
# ---------------------------------------------------------------------------


def canonical_json(value: object) -> str:
    """Canonical finite JSON text used for identities and recovery payloads."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateCanonicalKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_canonical_json_object(
    text: str, *, require_canonical: bool = True
) -> dict:
    """Load one JSON object, rejecting duplicates and non-finite values.

    Node ``result_json`` uses the canonical default.  The pretty-printed
    recovery envelope calls this same duplicate-aware owner with
    ``require_canonical=False``; it must not grow a second JSON parser merely
    because its whitespace is intentionally human-readable.
    """
    if not isinstance(text, str):
        raise ValueError("canonical JSON must be text.")

    def _non_finite(value: str):
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_non_finite,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid canonical JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("canonical recovery JSON must contain one object.")
    if require_canonical and canonical_json(value) != text:
        raise ValueError("recovery JSON is not in canonical form.")
    return value


def _digest(prefix: str, value: object) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _domain_id(prefix: str, value: object) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


DETERMINISTIC_IMPORTS_REVISION = "deterministic-imports-v1"


def deterministic_imports_digest(imports: Sequence[str]) -> str:
    """Pure digest over the exact ordered parser-derived imports (section 6).

    Preserves parser order and duplicates -- never sorted or case-folded, so
    a reordering (with no net addition/removal) or a duplicate change is
    still visible. Bound into the final-node execution identity, the
    final-node exact stage-local input digest, and the completed
    `_large_file_identity`; never into leaf or reducer identities, so an
    equal-length import change reruns only final synthesis while leaves and
    reducers stay reusable. Persists only this digest in identity metadata --
    never the raw import list.
    """
    payload = {
        "domain": "deterministic-imports",
        "revision": DETERMINISTIC_IMPORTS_REVISION,
        "imports": list(imports),
    }
    return _digest("deterministic-imports", payload)


def _real_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _digest_text(value: object, name: str, *, prefix: str | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text.")
    candidate = value if prefix is None else value.removeprefix(prefix + ":")
    if (prefix is not None and candidate == value) or not _HEX_64_RE.fullmatch(
        candidate
    ):
        raise ValueError(f"{name} is not a valid SHA-256 digest.")
    return value


def _id_text(value: object, name: str, prefix: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix + "_"):
        raise ValueError(f"{name} must use the {prefix} domain.")
    if not _HEX_64_RE.fullmatch(value[len(prefix) + 1 :]):
        raise ValueError(f"{name} must contain a full SHA-256 digest.")
    return value


def _bounded_text(value: object, name: str, *, maximum: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text.")
    if required and not value:
        raise ValueError(f"{name} is required.")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters.")
    return value


# ---------------------------------------------------------------------------
# Semantic units and chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticUnitIdentity:
    """Stable identity for one semantic unit (D4).

    `unit_id` and `source_range` are the final authority: two declarations
    with the same short name in different owning scopes always carry
    different `atom_ids`/ranges and therefore different `unit_id`s, so they
    never consolidate together regardless of `qualified_name` collisions.
    """

    unit_id: str
    kind: str
    qualified_name: str
    signature: str
    atom_ids: tuple[str, ...]
    source_range: SourceRange

    def __post_init__(self) -> None:
        _id_text(self.unit_id, "unit_id", "unit")
        _bounded_text(self.kind, "unit kind", maximum=160)
        _bounded_text(self.qualified_name, "unit qualified_name", maximum=240)
        _bounded_text(self.signature, "unit signature", maximum=600)
        atom_ids = tuple(self.atom_ids)
        if not atom_ids:
            raise ValueError("unit must own at least one atom.")
        if len(atom_ids) != len(set(atom_ids)):
            raise ValueError("unit has duplicate atom IDs.")
        if not isinstance(self.source_range, SourceRange):
            raise ValueError("unit source_range must be a SourceRange.")
        object.__setattr__(self, "atom_ids", atom_ids)

    @property
    def is_continuation_unit(self) -> bool:
        return len(self.atom_ids) == 1


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One immutable leaf AI-call boundary."""

    chunk_id: str
    semantic_units: tuple[SemanticUnitIdentity, ...]
    group_unit_id: str
    unit_chunk_index: int
    unit_chunk_count: int
    global_index: int
    global_count: int
    owning_ranges: tuple[SourceRange, ...]
    continuation_before: bool
    continuation_after: bool
    known_symbols: tuple[str, ...]
    payload: str
    payload_chars: int

    def __post_init__(self) -> None:
        _id_text(self.chunk_id, "chunk_id", "chunk")
        _real_int(self.unit_chunk_index, "unit_chunk_index")
        _real_int(self.unit_chunk_count, "unit_chunk_count", minimum=1)
        _real_int(self.global_index, "global_index")
        _real_int(self.global_count, "global_count", minimum=1)
        if self.unit_chunk_index >= self.unit_chunk_count:
            raise ValueError("unit_chunk_index must be within unit_chunk_count.")
        if self.global_index >= self.global_count:
            raise ValueError("global_index must be within global_count.")
        owning_ranges = tuple(self.owning_ranges)
        if not owning_ranges:
            raise ValueError("chunk must own at least one range.")
        if any(not isinstance(item, SourceRange) for item in owning_ranges):
            raise ValueError("chunk ranges must be SourceRange values.")
        if not isinstance(self.payload, str) or not self.payload:
            raise ValueError("chunk payload must be non-empty text.")
        _real_int(self.payload_chars, "payload_chars", minimum=1)
        if self.payload_chars != len(self.payload):
            raise ValueError("payload_chars must equal len(payload).")
        known_symbols = tuple(self.known_symbols)
        if any(not isinstance(item, str) for item in known_symbols):
            raise ValueError("known_symbols must contain text.")
        semantic_units = tuple(self.semantic_units)
        if not semantic_units or any(
            not isinstance(item, SemanticUnitIdentity) for item in semantic_units
        ):
            raise ValueError("chunk semantic_units must contain semantic units.")
        if len({item.unit_id for item in semantic_units}) != len(semantic_units):
            raise ValueError("chunk semantic_units contain duplicate unit IDs.")
        _id_text(self.group_unit_id, "group_unit_id", "unit")
        if self.unit_chunk_count > 1 and len(semantic_units) != 1:
            raise ValueError(
                "only one oversized semantic unit may own continuation chunks."
            )
        object.__setattr__(self, "owning_ranges", owning_ranges)
        object.__setattr__(self, "known_symbols", known_symbols)
        object.__setattr__(self, "semantic_units", semantic_units)

    @property
    def unit_id(self) -> str:
        """Internal call-group identity used by continuation consolidation."""
        return self.group_unit_id

    @property
    def unit_kind(self) -> str:
        if len(self.semantic_units) == 1:
            return self.semantic_units[0].kind
        return "packed-group"

    @property
    def unit_qualified_name(self) -> str:
        if len(self.semantic_units) == 1:
            return self.semantic_units[0].qualified_name
        names = [
            unit.qualified_name
            for unit in self.semantic_units
            if unit.qualified_name
        ]
        return "; ".join(names)[:240]

    @property
    def unit_signature(self) -> str:
        if len(self.semantic_units) == 1:
            return self.semantic_units[0].signature
        return ""


def _leaf_prompt_unit_value(unit: SemanticUnitIdentity) -> dict:
    return {
        "unit_id": unit.unit_id,
        "kind": unit.kind,
        "qualified_name": unit.qualified_name,
        "signature": unit.signature,
        "source_range": unit.source_range.to_public(),
    }


def _leaf_prompt_metadata_value(
    *,
    group_unit_id: str,
    semantic_units: Sequence[SemanticUnitIdentity],
    unit_indexes: Sequence[int],
    unit_count: int,
    owning_ranges: Sequence[SourceRange],
) -> dict:
    units = tuple(semantic_units)
    indexes = tuple(unit_indexes)
    ranges = tuple(owning_ranges)
    if not units or len(units) != len(indexes):
        raise ValueError("leaf prompt metadata units and indexes must align.")
    if not ranges or len(ranges) != len(units):
        raise ValueError("leaf prompt metadata ranges must align with units.")
    return {
        "leaf_call_group_id": group_unit_id,
        "semantic_unit_count": unit_count,
        "semantic_unit_indexes": [index + 1 for index in indexes],
        "semantic_units": [
            _leaf_prompt_unit_value(unit) for unit in units
        ],
        "owning_ranges": [
            source_range.to_public() for source_range in ranges
        ],
    }


def render_leaf_prompt_metadata(
    *,
    group_unit_id: str,
    semantic_units: Sequence[SemanticUnitIdentity],
    unit_indexes: Sequence[int],
    unit_count: int,
    owning_ranges: Sequence[SourceRange],
) -> str:
    return canonical_json(
        _leaf_prompt_metadata_value(
            group_unit_id=group_unit_id,
            semantic_units=semantic_units,
            unit_indexes=unit_indexes,
            unit_count=unit_count,
            owning_ranges=owning_ranges,
        )
    )


def leaf_prompt_metadata_chars(
    *,
    group_unit_id: str,
    semantic_units: Sequence[SemanticUnitIdentity],
    unit_indexes: Sequence[int],
    unit_count: int,
    owning_ranges: Sequence[SourceRange],
) -> int:
    _leaf_prompt_metadata_value(
        group_unit_id=group_unit_id,
        semantic_units=semantic_units,
        unit_indexes=unit_indexes,
        unit_count=unit_count,
        owning_ranges=owning_ranges,
    )
    units = tuple(semantic_units)
    indexes = tuple(unit_indexes)
    ranges = tuple(owning_ranges)
    return (
        _leaf_prompt_metadata_base_chars(group_unit_id, unit_count)
        + sum(
            _leaf_prompt_metadata_entry_chars(unit, index, source_range)
            for unit, index, source_range in zip(units, indexes, ranges)
        )
        + 3 * (len(units) - 1)
    )


def _leaf_prompt_metadata_base_chars(
    group_unit_id: str,
    unit_count: int,
) -> int:
    return len(
        canonical_json(
            {
                "leaf_call_group_id": group_unit_id,
                "semantic_unit_count": unit_count,
                "semantic_unit_indexes": [],
                "semantic_units": [],
                "owning_ranges": [],
            }
        )
    )


def _leaf_prompt_metadata_entry_chars(
    unit: SemanticUnitIdentity,
    unit_index: int,
    source_range: SourceRange,
) -> int:
    return (
        len(canonical_json(_leaf_prompt_unit_value(unit)))
        + len(canonical_json(unit_index + 1))
        + len(canonical_json(source_range.to_public()))
    )


def distinct_units(chunks: Sequence[ChunkPlan]) -> tuple[SemanticUnitIdentity, ...]:
    """Ordered, de-duplicated source semantic units (first-seen order)."""
    seen: set[str] = set()
    out: list[SemanticUnitIdentity] = []
    for chunk in chunks:
        for unit in chunk.semantic_units:
            if unit.unit_id not in seen:
                seen.add(unit.unit_id)
                out.append(unit)
    return tuple(out)


# ---------------------------------------------------------------------------
# Division plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DivisionPlan:
    """One complete deterministic split plan.

    A `DivisionPlan` only ever describes a complete, gap-free, non-overlapping
    division of the canonical decoded source into ordered chunks.  There is no
    "capacity fallback" plan shape: `build_division_plan` either returns a
    complete plan or raises `SplitCapacityBlocked` / `DivisionInternalDefect`.
    """

    rel_path: str
    structural_mode: Literal["syntax", "lexical"]
    source_chars: int
    source_bytes: int
    source_budget_chars: int
    atoms: tuple[Atom, ...]
    symbols: tuple[SymbolFact, ...]
    chunks: tuple[ChunkPlan, ...]
    diagnostics: tuple[str, ...]
    plan_digest: str

    def __post_init__(self) -> None:
        rel_path = normalize_rel_path(self.rel_path)
        atoms = tuple(self.atoms)
        symbols = tuple(self.symbols)
        chunks = tuple(self.chunks)
        diagnostics = tuple(self.diagnostics)
        for name in ("source_chars", "source_bytes"):
            _real_int(getattr(self, name), name)
        _real_int(self.source_budget_chars, "source_budget_chars", minimum=1)
        _digest_text(self.plan_digest, "plan_digest", prefix="division-plan")
        if self.structural_mode not in ("syntax", "lexical"):
            raise ValueError("division plan has an invalid structural mode.")
        if len(diagnostics) > MAX_STRUCTURE_DIAGNOSTICS:
            raise ValueError("too many division diagnostics.")
        if any(not isinstance(item, str) or len(item) > 400 for item in diagnostics):
            raise ValueError("division diagnostic is invalid or too long.")
        if not atoms or not chunks:
            raise ValueError("complete split plan state is incomplete.")
        if (
            len(atoms) > MAX_ATOMS_PER_FILE
            or len(symbols) > MAX_SYMBOLS_PER_FILE
            or len(chunks) > MAX_CHUNKS_PER_FILE
        ):
            raise ValueError("complete split plan exceeds a substrate cap.")
        _validate_plan_partitions(
            rel_path,
            self.structural_mode,
            self.source_chars,
            self.source_bytes,
            self.source_budget_chars,
            atoms,
            symbols,
            chunks,
        )
        object.__setattr__(self, "rel_path", rel_path)
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def units(self) -> tuple[SemanticUnitIdentity, ...]:
        return distinct_units(self.chunks)

    def unit_positions(self, chunk: ChunkPlan) -> tuple[int, ...]:
        units = self.units
        positions = {unit.unit_id: index for index, unit in enumerate(units)}
        try:
            return tuple(positions[unit.unit_id] for unit in chunk.semantic_units)
        except KeyError as exc:
            raise ValueError(
                "chunk's semantic units are not part of this division plan."
            ) from exc

    def unit_position(self, chunk: ChunkPlan) -> tuple[int, int]:
        """Return `(unit_index, unit_count)`: this chunk's unit's 0-based
        ordinal position among every distinct semantic unit in the file, and
        the total distinct-unit count (section 7's fragment-prompt grounding)."""
        positions = self.unit_positions(chunk)
        if len(positions) != 1:
            raise ValueError("packed chunks have more than one semantic-unit position.")
        return positions[0], len(self.units)


def _validate_plan_partitions(
    rel_path: str,
    structural_mode: str,
    source_chars: int,
    source_bytes: int,
    source_budget_chars: int,
    atoms: tuple[Atom, ...],
    symbols: tuple[SymbolFact, ...],
    chunks: tuple[ChunkPlan, ...],
) -> None:
    content = "".join(atom.source for atom in atoms)
    if len(content) != source_chars or len(content.encode("utf-8")) != source_bytes:
        raise ValueError("plan source counts do not match its atoms.")
    validate_structure(content, StructureResult(structural_mode, atoms, symbols, ()))
    if any(atom.rel_path != rel_path for atom in atoms):
        raise ValueError("plan atoms use a different path.")

    data = content.encode("utf-8")
    offset = 0
    unit_chunk_seen: dict[str, int] = {}
    units = distinct_units(chunks)
    unit_positions = {
        unit.unit_id: index for index, unit in enumerate(units)
    }
    for chunk in chunks:
        expected_index = unit_chunk_seen.get(chunk.unit_id, 0)
        if chunk.unit_chunk_index != expected_index:
            raise ValueError("chunk indexes are not consecutive within a unit.")
        unit_chunk_seen[chunk.unit_id] = expected_index + 1
        owning_source_parts = []
        for source_range in chunk.owning_ranges:
            if source_range.start_byte != offset:
                raise ValueError("chunks are not an ordered gap-free partition.")
            owning_source_parts.append(
                data[source_range.start_byte : source_range.end_byte].decode("utf-8")
            )
            offset = source_range.end_byte
        owning_source = "".join(owning_source_parts)
        if chunk.payload != owning_source:
            raise ValueError("chunk payload does not equal its owning source.")
        if chunk.payload_chars > source_budget_chars:
            raise ValueError("chunk payload exceeds the source character budget.")
        metadata = render_leaf_prompt_metadata(
            group_unit_id=chunk.unit_id,
            semantic_units=chunk.semantic_units,
            unit_indexes=tuple(
                unit_positions[unit.unit_id]
                for unit in chunk.semantic_units
            ),
            unit_count=len(units),
            owning_ranges=chunk.owning_ranges,
        )
        metadata_chars = leaf_prompt_metadata_chars(
            group_unit_id=chunk.unit_id,
            semantic_units=chunk.semantic_units,
            unit_indexes=tuple(
                unit_positions[unit.unit_id]
                for unit in chunk.semantic_units
            ),
            unit_count=len(units),
            owning_ranges=chunk.owning_ranges,
        )
        if metadata_chars != len(metadata):
            raise ValueError("leaf prompt metadata accounting drifted.")
        if metadata_chars > MAX_LEAF_PROMPT_METADATA_CHARS:
            raise ValueError("chunk exceeds the leaf prompt metadata bound.")
    if offset != source_bytes:
        raise ValueError("chunks do not cover the complete source.")
    for unit_id, count in unit_chunk_seen.items():
        expected_counts = {c.unit_chunk_count for c in chunks if c.unit_id == unit_id}
        if expected_counts != {count}:
            raise ValueError("unit_chunk_count does not match its owned chunk count.")
    if source_chars > source_budget_chars and len(chunks) < 2:
        raise ValueError("an oversized split source must contain at least two chunks.")


# ---------------------------------------------------------------------------
# Packing: atoms -> semantic units -> chunks
# ---------------------------------------------------------------------------


def _semantic_unit_identity(
    rel_path: str,
    atom: Atom,
    symbol: SymbolFact | None,
) -> SemanticUnitIdentity:
    """Return the authoritative one-atom source semantic-unit identity."""
    unit_id = _domain_id(
        "unit",
        {
            "revision": UNIT_SCHEMA_REVISION,
            "path": rel_path,
            "atom": atom.atom_id,
        },
    )
    if symbol is not None:
        qualified_name = symbol.qualified_name
        signature = symbol.signature
        kind = symbol.kind
    else:
        qualified_name = atom.name
        signature = ""
        kind = atom.kind
    return SemanticUnitIdentity(
        unit_id=unit_id,
        kind=kind,
        qualified_name=qualified_name,
        signature=signature,
        atom_ids=(atom.atom_id,),
        source_range=atom.range,
    )


def _chunk_group_unit_id(
    rel_path: str,
    units: Sequence[SemanticUnitIdentity],
) -> str:
    """Return the internal leaf/reduction group identity for *units*."""
    if len(units) == 1:
        return units[0].unit_id
    return _domain_id(
        "unit",
        {
            "revision": UNIT_SCHEMA_REVISION,
            "path": rel_path,
            "kind": "packed-call-group",
            "units": [unit.unit_id for unit in units],
        },
    )


def _source_segments(
    source_index: SourceIndex,
    start_byte: int,
    end_byte: int,
    budget: int,
) -> Iterable[tuple[SourceRange, str]]:
    """Yield budget-filled, code-point-safe owning pieces for one byte range.

    Complete adjacent physical lines are accumulated until adding the next
    line would cross *budget*. Only an individual line that exceeds the budget
    is split by code-point count. This preserves preferred lexical boundaries
    without turning every physical line into its own paid chunk.
    """
    source = source_index.slice(start_byte, end_byte)
    offset = start_byte
    physical_lines = source.splitlines(keepends=True) or ([source] if source else [])
    pending: list[str] = []
    pending_chars = 0

    def emit(piece: str) -> tuple[SourceRange, str]:
        nonlocal offset
        piece_end = offset + len(piece.encode("utf-8"))
        result = source_index.range(offset, piece_end), piece
        offset = piece_end
        return result

    for line in physical_lines:
        if len(line) > budget:
            if pending:
                yield emit("".join(pending))
                pending = []
                pending_chars = 0
            for index in range(0, len(line), budget):
                yield emit(line[index : index + budget])
            continue
        if pending and pending_chars + len(line) > budget:
            yield emit("".join(pending))
            pending = []
            pending_chars = 0
        pending.append(line)
        pending_chars += len(line)
    if pending:
        yield emit("".join(pending))
    if offset != end_byte:
        raise DivisionInternalDefect("chunk segmentation lost source bytes.")


def _known_symbols_for(
    symbols: tuple[SymbolFact, ...], ranges: list[SourceRange]
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol.qualified_name in seen:
            continue
        if any(
            r.start_byte <= symbol.range.start_byte < r.end_byte for r in ranges
        ):
            seen.add(symbol.qualified_name)
            names.append(symbol.qualified_name)
            if len(names) == MAX_KNOWN_SYMBOLS_PER_CHUNK:
                return tuple(names)
    return tuple(names)


def pack_chunks(
    rel_path: str,
    atoms: tuple[Atom, ...],
    symbols: tuple[SymbolFact, ...],
    source_budget_chars: int,
) -> tuple[ChunkPlan, ...]:
    """Pack ordered atoms into bounded leaf chunks without duplicating coverage.

    Adjacent small semantic units are packed together until the character
    budget would be exceeded (D4). A single semantic unit whose own source
    exceeds the budget is divided into deterministic, budget-filled
    continuation chunks that prefer complete-line boundaries; every fragment
    shares that unit's stable identity.
    """
    _real_int(source_budget_chars, "source_budget_chars", minimum=1)
    rel_path = normalize_rel_path(rel_path)
    content = "".join(atom.source for atom in atoms)
    source_index = SourceIndex(content)
    primary_symbol_by_atom: dict[str, SymbolFact] = {}
    for symbol in symbols:
        primary_symbol_by_atom.setdefault(symbol.atom_id, symbol)

    units_by_atom = {
        atom.atom_id: _semantic_unit_identity(
            rel_path,
            atom,
            primary_symbol_by_atom.get(atom.atom_id),
        )
        for atom in atoms
    }
    unit_index_by_atom = {
        atom.atom_id: index for index, atom in enumerate(atoms)
    }
    unit_count = len(atoms)
    metadata_base_chars = _leaf_prompt_metadata_base_chars(
        "unit_" + "0" * 64,
        unit_count,
    )
    metadata_entry_chars_by_atom = {
        atom.atom_id: _leaf_prompt_metadata_entry_chars(
            units_by_atom[atom.atom_id],
            unit_index_by_atom[atom.atom_id],
            atom.range,
        )
        for atom in atoms
    }

    # Each spec: (source units, call-group ID, ranges, payload,
    # continuation_before, continuation_after, group_chunk_index,
    # group_chunk_count).
    specs: list[
        tuple[
            tuple[SemanticUnitIdentity, ...],
            str,
            list[SourceRange],
            str,
            bool,
            bool,
            int,
            int,
        ]
    ] = []
    pending: list[Atom] = []
    pending_chars = 0
    pending_metadata_entry_chars = 0

    def flush_pending() -> None:
        nonlocal pending, pending_chars, pending_metadata_entry_chars
        if not pending:
            return
        units = tuple(units_by_atom[atom.atom_id] for atom in pending)
        group_unit_id = _chunk_group_unit_id(rel_path, units)
        ranges = [atom.range for atom in pending]
        payload = "".join(atom.source for atom in pending)
        specs.append((units, group_unit_id, ranges, payload, False, False, 0, 1))
        pending = []
        pending_chars = 0
        pending_metadata_entry_chars = 0

    for atom in atoms:
        atom_chars = len(atom.source)
        if atom_chars > source_budget_chars:
            flush_pending()
            unit = units_by_atom[atom.atom_id]
            pieces = list(
                _source_segments(
                    source_index, atom.range.start_byte, atom.range.end_byte, source_budget_chars
                )
            )
            count = len(pieces)
            for piece_index, (source_range, piece) in enumerate(pieces):
                metadata_chars = (
                    metadata_base_chars
                    + _leaf_prompt_metadata_entry_chars(
                        unit,
                        unit_index_by_atom[atom.atom_id],
                        source_range,
                    )
                )
                if metadata_chars > MAX_LEAF_PROMPT_METADATA_CHARS:
                    raise DivisionInternalDefect(
                        "one semantic unit exceeds the leaf prompt metadata bound."
                    )
                specs.append(
                    (
                        (unit,),
                        unit.unit_id,
                        [source_range],
                        piece,
                        piece_index > 0,
                        piece_index < count - 1,
                        piece_index,
                        count,
                    )
                )
            continue
        candidate_metadata_chars = (
            metadata_base_chars
            + pending_metadata_entry_chars
            + metadata_entry_chars_by_atom[atom.atom_id]
            + 3 * len(pending)
        )
        if pending and (
            pending_chars + atom_chars > source_budget_chars
            or candidate_metadata_chars > MAX_LEAF_PROMPT_METADATA_CHARS
        ):
            flush_pending()
            candidate_metadata_chars = (
                metadata_base_chars
                + metadata_entry_chars_by_atom[atom.atom_id]
            )
        if candidate_metadata_chars > MAX_LEAF_PROMPT_METADATA_CHARS:
            raise DivisionInternalDefect(
                "one semantic unit exceeds the leaf prompt metadata bound."
            )
        pending.append(atom)
        pending_chars += atom_chars
        pending_metadata_entry_chars += metadata_entry_chars_by_atom[
            atom.atom_id
        ]
    flush_pending()

    total = len(specs)
    chunks: list[ChunkPlan] = []
    for global_index, (
        semantic_units,
        group_unit_id,
        ranges,
        payload,
        cont_before,
        cont_after,
        uidx,
        ucount,
    ) in enumerate(specs):
        known_symbols = _known_symbols_for(symbols, ranges)
        chunk_id = _domain_id(
            "chunk",
            {
                "revision": PACKER_SCHEMA_REVISION,
                "group_unit": group_unit_id,
                "semantic_units": [unit.unit_id for unit in semantic_units],
                "unit_index": uidx,
                "ranges": [r.to_public() for r in ranges],
            },
        )
        chunks.append(
            ChunkPlan(
                chunk_id=chunk_id,
                semantic_units=semantic_units,
                group_unit_id=group_unit_id,
                unit_chunk_index=uidx,
                unit_chunk_count=ucount,
                global_index=global_index,
                global_count=total,
                owning_ranges=tuple(ranges),
                continuation_before=cont_before,
                continuation_after=cont_after,
                known_symbols=known_symbols,
                payload=payload,
                payload_chars=len(payload),
            )
        )
    return tuple(chunks)


def build_division_plan(
    *,
    rel_path: str,
    language: str,
    content: str,
    source_budget_chars: int,
) -> DivisionPlan:
    """Build one complete deterministic split plan.

    Raises `SplitCapacityBlocked` with the first-failing named reason under
    the frozen evaluation order (`BLOCKED_REASON_ORDER`), or
    `DivisionInternalDefect` for a CodeDoc invariant failure.  Never returns a
    partial or capacity-fallback plan.
    """
    rel_path = normalize_rel_path(rel_path)
    if not isinstance(language, str) or not language:
        raise DivisionInternalDefect("division language is required.")
    if not isinstance(content, str) or not content:
        raise DivisionInternalDefect("division requires non-empty decoded source.")
    _real_int(source_budget_chars, "source_budget_chars", minimum=1)

    try:
        # No `max_atom_chars`: atoms come back at their full natural size (one
        # physical line in lexical mode, one declaration/gap in syntax mode) so
        # `pack_chunks` below is the single place that recognizes an oversized
        # semantic unit and produces its deterministic continuation chunks.
        # Pre-splitting here would fragment one oversized line into several
        # independent lexical atoms before packing ever sees it, destroying the
        # single stable unit identity a continuation group must share.
        structure = extract_structure(rel_path, language, content)
        validate_structure(content, structure)
    except DivisionInternalDefect:
        raise
    except Exception as exc:
        raise DivisionInternalDefect(
            f"deterministic structure extraction failed for {rel_path}: {exc}"
        ) from exc

    if len(structure.atoms) > MAX_ATOMS_PER_FILE:
        raise SplitCapacityBlocked(rel_path, "atom-cap")
    if len(structure.symbols) > MAX_SYMBOLS_PER_FILE:
        raise SplitCapacityBlocked(rel_path, "symbol-cap")

    try:
        chunks = pack_chunks(rel_path, structure.atoms, structure.symbols, source_budget_chars)
    except (SplitCapacityBlocked, DivisionInternalDefect):
        raise
    except Exception as exc:
        raise DivisionInternalDefect(
            f"deterministic packing failed for {rel_path}: {exc}"
        ) from exc

    # Unit capacity applies to authoritative source semantic units, never to
    # the packed leaf-call groups used only for execution efficiency.
    if len(distinct_units(chunks)) > MAX_UNITS_PER_FILE:
        raise SplitCapacityBlocked(rel_path, "unit-cap")
    if len(chunks) > MAX_CHUNKS_PER_FILE:
        raise SplitCapacityBlocked(rel_path, "chunk-cap")

    digest_payload = _plan_payload(
        rel_path, language, content, source_budget_chars, structure, chunks
    )
    try:
        return DivisionPlan(
            rel_path=rel_path,
            structural_mode=structure.structural_mode,
            source_chars=len(content),
            source_bytes=len(content.encode("utf-8")),
            source_budget_chars=source_budget_chars,
            atoms=structure.atoms,
            symbols=structure.symbols,
            chunks=chunks,
            diagnostics=structure.diagnostics[:MAX_STRUCTURE_DIAGNOSTICS],
            plan_digest=_digest("division-plan", digest_payload),
        )
    except (SplitCapacityBlocked, DivisionInternalDefect):
        raise
    except Exception as exc:
        raise DivisionInternalDefect(
            f"deterministic division invariant failed for {rel_path}: {exc}"
        ) from exc


def _plan_payload(
    rel_path: str,
    language: str,
    content: str,
    budget: int,
    structure: StructureResult,
    chunks: tuple[ChunkPlan, ...],
) -> dict:
    return {
        "structure_revision": STRUCTURE_SCHEMA_REVISION,
        "unit_revision": UNIT_SCHEMA_REVISION,
        "packer_revision": PACKER_SCHEMA_REVISION,
        "parser_package_version": PARSER_PACKAGE_VERSION,
        "path": rel_path,
        # The effective language selects the grammar and is rendered into every
        # leaf/final prompt, so remapping an extension to a different language
        # must invalidate this plan and every identity derived from it —
        # otherwise stale work is silently reused under new prompt semantics.
        "language": language,
        "source_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "budget": budget,
        "bounds": {
            "atoms": MAX_ATOMS_PER_FILE,
            "symbols": MAX_SYMBOLS_PER_FILE,
            "units": MAX_UNITS_PER_FILE,
            "chunks": MAX_CHUNKS_PER_FILE,
            "known_symbols_per_chunk": MAX_KNOWN_SYMBOLS_PER_CHUNK,
            "leaf_prompt_metadata_chars": MAX_LEAF_PROMPT_METADATA_CHARS,
        },
        "structural_mode": structure.structural_mode,
        "chunks": [
            {
                "id": chunk.chunk_id,
                "group_unit": chunk.unit_id,
                "semantic_units": [unit.unit_id for unit in chunk.semantic_units],
                "unit_kind": chunk.unit_kind,
                "unit_name": chunk.unit_qualified_name,
                "unit_index": chunk.unit_chunk_index,
                "unit_count": chunk.unit_chunk_count,
                "global_index": chunk.global_index,
                "owning": [item.to_public() for item in chunk.owning_ranges],
                "continuation_before": chunk.continuation_before,
                "continuation_after": chunk.continuation_after,
                "payload_chars": chunk.payload_chars,
            }
            for chunk in chunks
        ],
    }
# ---------------------------------------------------------------------------
# Deterministic hierarchical reduction tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReductionNodePlan:
    """One planned reduction (or final) node.

    `leaf_ids` is the exact, locally computed union of descendant leaf chunk
    IDs — the coverage this node is responsible for once executed.
    """

    node_id: str
    phase: Literal["unit-consolidation", "general", "final"]
    node_type: Literal["intermediate", "final"]
    unit_id: str | None
    level: int
    ordinal: int
    child_ids: tuple[str, ...]
    leaf_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _id_text(self.node_id, "node_id", "node")
        if self.phase not in ("unit-consolidation", "general", "final"):
            raise ValueError("invalid reduction node phase.")
        if self.node_type not in ("intermediate", "final"):
            raise ValueError("invalid reduction node type.")
        _real_int(self.level, "level")
        _real_int(self.ordinal, "ordinal", minimum=1)
        child_ids = tuple(self.child_ids)
        leaf_ids = tuple(self.leaf_ids)
        if len(child_ids) < 2 and self.node_type == "intermediate":
            raise ValueError("an intermediate reduction node needs at least two children.")
        if not child_ids:
            raise ValueError("a reduction node must have at least one child.")
        if not leaf_ids:
            raise ValueError("a reduction node must cover at least one leaf.")
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("reduction node coverage has duplicate leaves.")
        object.__setattr__(self, "child_ids", child_ids)
        object.__setattr__(self, "leaf_ids", leaf_ids)


@dataclass(frozen=True, slots=True)
class ReductionTreePlan:
    """The complete deterministic reduction tree for one effective split file."""

    rel_path: str
    division_plan_digest: str
    max_fan_in: int
    unit_consolidation_nodes: tuple[ReductionNodePlan, ...]
    general_nodes: tuple[ReductionNodePlan, ...]
    final_node: ReductionNodePlan
    tree_digest: str
    packing_revision: str

    def __post_init__(self) -> None:
        rel_path = normalize_rel_path(self.rel_path)
        _digest_text(self.division_plan_digest, "division_plan_digest", prefix="division-plan")
        _real_int(self.max_fan_in, "max_fan_in", minimum=2)
        _digest_text(self.tree_digest, "tree_digest", prefix="reduction-tree")
        if self.final_node.phase != "final" or self.final_node.node_type != "final":
            raise ValueError("reduction tree final_node must be phase/type 'final'.")
        object.__setattr__(self, "rel_path", rel_path)
        object.__setattr__(
            self, "unit_consolidation_nodes", tuple(self.unit_consolidation_nodes)
        )
        object.__setattr__(self, "general_nodes", tuple(self.general_nodes))

    @property
    def all_intermediate_nodes(self) -> tuple[ReductionNodePlan, ...]:
        return self.unit_consolidation_nodes + self.general_nodes

    @property
    def all_nodes(self) -> tuple[ReductionNodePlan, ...]:
        return self.all_intermediate_nodes + (self.final_node,)


def _reduction_node_id(
    rel_path: str,
    division_plan_digest: str,
    phase: str,
    unit_id: str | None,
    level: int,
    ordinal: int,
    child_ids: Sequence[str],
) -> str:
    return _domain_id(
        "node",
        {
            "revision": REDUCTION_PACKING_REVISION,
            "path": rel_path,
            "division_plan_digest": division_plan_digest,
            "phase": phase,
            "unit_id": unit_id,
            "level": level,
            "ordinal": ordinal,
            "children": list(child_ids),
        },
    )


def _pack_level(
    rel_path: str,
    division_plan_digest: str,
    phase: Literal["unit-consolidation", "general"],
    unit_id: str | None,
    level: int,
    ordered_ids: list[str],
    ordered_leaf_sets: list[tuple[str, ...]],
    max_fan_in: int,
) -> tuple[list[ReductionNodePlan], list[str], list[tuple[str, ...]]]:
    """Group one ordered sequence into (at most) `max_fan_in`-wide nodes.

    A trailing group of exactly one child is promoted unchanged rather than
    wrapped in a unary reducer (D6/section 9's frozen singleton-remainder rule).
    """
    nodes: list[ReductionNodePlan] = []
    next_ids: list[str] = []
    next_leaf_sets: list[tuple[str, ...]] = []
    ordinal = 0
    index = 0
    while index < len(ordered_ids):
        group_ids = ordered_ids[index : index + max_fan_in]
        group_leaf_ids = tuple(
            leaf for leaves in ordered_leaf_sets[index : index + max_fan_in] for leaf in leaves
        )
        if len(group_ids) == 1:
            next_ids.append(group_ids[0])
            next_leaf_sets.append(group_leaf_ids)
            index += 1
            continue
        ordinal += 1
        node_id = _reduction_node_id(
            rel_path, division_plan_digest, phase, unit_id, level, ordinal, group_ids
        )
        nodes.append(
            ReductionNodePlan(
                node_id=node_id,
                phase=phase,
                node_type="intermediate",
                unit_id=unit_id,
                level=level,
                ordinal=ordinal,
                child_ids=tuple(group_ids),
                leaf_ids=group_leaf_ids,
            )
        )
        next_ids.append(node_id)
        next_leaf_sets.append(group_leaf_ids)
        index += len(group_ids)
    return nodes, next_ids, next_leaf_sets


def render_reduction_child_manifest(
    child_narratives: Sequence[str],
) -> str:
    """Render reducer children exactly as the fixed reducer prompt does."""
    narratives = tuple(child_narratives)
    if any(not isinstance(item, str) for item in narratives):
        raise DivisionInternalDefect(
            "reduction child narratives must contain text."
        )
    return REDUCTION_MANIFEST_ITEM_SEPARATOR.join(
        f"{REDUCTION_MANIFEST_ITEM_PREFIX}{text}" for text in narratives
    )


def worst_case_reduction_manifest_chars(child_count: int) -> int:
    """Exact rendered length for *child_count* maximum-length narratives."""
    _real_int(child_count, "reduction child count", minimum=1)
    return (
        child_count
        * (
            len(REDUCTION_MANIFEST_ITEM_PREFIX)
            + MAX_REDUCTION_NARRATIVE_CHARS
        )
        + (child_count - 1) * len(REDUCTION_MANIFEST_ITEM_SEPARATOR)
    )


def build_reduction_tree(
    plan: DivisionPlan,
    *,
    max_content_chars: int,
    language: str = "",
    imports: Sequence[str] = (),
) -> ReductionTreePlan:
    """Build the complete deterministic reduction tree for *plan*.

    Raises `SplitCapacityBlocked` with the appropriate reduction-phase reason
    when no complete tree can be planned provider-free.  Topology, node IDs,
    and exact call counts are final before any provider is created (D6).
    """
    rel_path = plan.rel_path
    _real_int(max_content_chars, "max_content_chars", minimum=1)

    if REDUCTION_ENVELOPE_OVERHEAD_CHARS >= max_content_chars:
        raise SplitCapacityBlocked(rel_path, "reduction-envelope-cap")
    available_manifest_chars = (
        max_content_chars - REDUCTION_ENVELOPE_OVERHEAD_CHARS
    )
    per_child_chars = (
        len(REDUCTION_MANIFEST_ITEM_PREFIX)
        + MAX_REDUCTION_NARRATIVE_CHARS
        + len(REDUCTION_MANIFEST_ITEM_SEPARATOR)
    )
    # The final child has no trailing separator, so restore that one character
    # before floor division. This is algebraically equivalent to requiring
    # `worst_case_reduction_manifest_chars(max_fan_in)` to fit exactly.
    max_fan_in = (
        available_manifest_chars + len(REDUCTION_MANIFEST_ITEM_SEPARATOR)
    ) // per_child_chars
    if max_fan_in < 2:
        raise SplitCapacityBlocked(rel_path, "reduction-fan-in-cap")
    if (
        worst_case_reduction_manifest_chars(max_fan_in)
        > available_manifest_chars
    ):
        raise DivisionInternalDefect(
            "reduction fan-in sizing exceeded the child-manifest ceiling."
        )

    if not isinstance(language, str):
        raise DivisionInternalDefect("reduction-tree language must be text.")
    normalized_imports = tuple(imports)
    if any(not isinstance(item, str) for item in normalized_imports):
        raise DivisionInternalDefect("reduction-tree imports must contain text.")

    units_order: list[str] = []
    chunks_by_unit: dict[str, list[ChunkPlan]] = {}
    for chunk in plan.chunks:
        uid = chunk.unit_id
        if uid not in chunks_by_unit:
            chunks_by_unit[uid] = []
            units_order.append(uid)
        chunks_by_unit[uid].append(chunk)

    unit_consolidation_nodes: list[ReductionNodePlan] = []
    unit_root_ids: dict[str, str] = {}
    unit_root_leaves: dict[str, tuple[str, ...]] = {}
    node_depths = {chunk.chunk_id: 0 for chunk in plan.chunks}

    for uid in units_order:
        unit_chunks = chunks_by_unit[uid]
        if len(unit_chunks) == 1:
            unit_root_ids[uid] = unit_chunks[0].chunk_id
            unit_root_leaves[uid] = (unit_chunks[0].chunk_id,)
            continue
        current_ids = [chunk.chunk_id for chunk in unit_chunks]
        current_leaves = [(chunk.chunk_id,) for chunk in unit_chunks]
        level = 0
        while len(current_ids) > 1:
            level += 1
            if level > MAX_REDUCTION_TREE_DEPTH:
                raise SplitCapacityBlocked(rel_path, "reduction-depth-cap")
            nodes, current_ids, current_leaves = _pack_level(
                rel_path,
                plan.plan_digest,
                "unit-consolidation",
                uid,
                level,
                current_ids,
                current_leaves,
                max_fan_in,
            )
            for node in nodes:
                depth = 1 + max(node_depths[child_id] for child_id in node.child_ids)
                if depth > MAX_REDUCTION_TREE_DEPTH:
                    raise SplitCapacityBlocked(rel_path, "reduction-depth-cap")
                node_depths[node.node_id] = depth
            unit_consolidation_nodes.extend(nodes)
        unit_root_ids[uid] = current_ids[0]
        unit_root_leaves[uid] = current_leaves[0]

    current_ids = [unit_root_ids[uid] for uid in units_order]
    current_leaves = [unit_root_leaves[uid] for uid in units_order]
    general_nodes: list[ReductionNodePlan] = []
    level = 0
    while (
        worst_case_final_synthesis_chars(
            rel_path=rel_path,
            language=language,
            imports=normalized_imports,
            root_count=len(current_ids),
            leaf_count=len(plan.chunks),
            max_chars=max_content_chars,
        )
        > max_content_chars
        and len(current_ids) > 1
    ):
        level += 1
        if level > MAX_REDUCTION_TREE_DEPTH:
            raise SplitCapacityBlocked(rel_path, "reduction-depth-cap")
        nodes, current_ids, current_leaves = _pack_level(
            rel_path, plan.plan_digest, "general", None, level, current_ids, current_leaves, max_fan_in
        )
        for node in nodes:
            depth = 1 + max(node_depths[child_id] for child_id in node.child_ids)
            if depth > MAX_REDUCTION_TREE_DEPTH:
                raise SplitCapacityBlocked(rel_path, "reduction-depth-cap")
            node_depths[node.node_id] = depth
        general_nodes.extend(nodes)

    if (
        worst_case_final_synthesis_chars(
            rel_path=rel_path,
            language=language,
            imports=normalized_imports,
            root_count=len(current_ids),
            leaf_count=len(plan.chunks),
            max_chars=max_content_chars,
        )
        > max_content_chars
    ):
        raise SplitCapacityBlocked(rel_path, "final-synthesis-envelope-cap")

    final_leaf_ids = tuple(leaf for leaves in current_leaves for leaf in leaves)
    final_node_id = _reduction_node_id(
        rel_path, plan.plan_digest, "final", None, 0, 1, current_ids
    )
    final_node = ReductionNodePlan(
        node_id=final_node_id,
        phase="final",
        node_type="final",
        unit_id=None,
        level=0,
        ordinal=1,
        child_ids=tuple(current_ids),
        leaf_ids=final_leaf_ids,
    )

    expected_leaves = tuple(sorted(chunk.chunk_id for chunk in plan.chunks))
    if tuple(sorted(final_leaf_ids)) != expected_leaves or len(final_leaf_ids) != len(
        set(final_leaf_ids)
    ):
        raise DivisionInternalDefect(
            f"reduction tree for {rel_path} does not cover every leaf exactly once."
        )

    tree_digest = _digest(
        "reduction-tree",
        {
            "revision": REDUCTION_PACKING_REVISION,
            "division_plan_digest": plan.plan_digest,
            "max_fan_in": max_fan_in,
            "unit_consolidation": [
                (node.node_id, node.phase, node.unit_id, node.level, node.ordinal, node.child_ids)
                for node in unit_consolidation_nodes
            ],
            "general": [
                (node.node_id, node.level, node.ordinal, node.child_ids)
                for node in general_nodes
            ],
            "final": (final_node.node_id, final_node.child_ids),
        },
    )
    return ReductionTreePlan(
        rel_path=rel_path,
        division_plan_digest=plan.plan_digest,
        max_fan_in=max_fan_in,
        unit_consolidation_nodes=tuple(unit_consolidation_nodes),
        general_nodes=tuple(general_nodes),
        final_node=final_node,
        tree_digest=tree_digest,
        packing_revision=REDUCTION_PACKING_REVISION,
    )


def reduction_depth(tree: ReductionTreePlan) -> int:
    """Maximum number of reduction calls on any leaf-to-root path (D6a).

    Excludes the final synthesis call itself.
    """
    nodes_by_id = {node.node_id: node for node in tree.all_intermediate_nodes}
    memo: dict[str, int] = {}

    def depth_of(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        node = nodes_by_id.get(node_id)
        if node is None:
            return 0
        value = 1 + max((depth_of(child) for child in node.child_ids), default=0)
        memo[node_id] = value
        return value

    return max((depth_of(child) for child in tree.final_node.child_ids), default=0)


# ---------------------------------------------------------------------------
# Structured fact ledger (section 8)
# ---------------------------------------------------------------------------

_CASE_INSENSITIVE_LANGUAGES = frozenset({"sql", "pascal", "basic", "fortran"})


def _normalize_symbol_key(name: str, language: str) -> str:
    # NFC first (D7/section 8): canonically equivalent spellings of the same
    # identifier — e.g. precomposed "é" vs "e" + combining acute — must
    # produce one ledger entry rather than two.
    normalized = " ".join(unicodedata.normalize("NFC", name).strip().split())
    if language.lower() in _CASE_INSENSITIVE_LANGUAGES:
        normalized = normalized.lower()
    return normalized


def _semantic_key(
    item: Mapping[str, object],
    kind: str,
    language: str,
    *,
    scope_id: str = "",
) -> str:
    """Deterministic dedup identity for one structured fact.

    Signature and locally supplied semantic-unit scope participate so
    overloads and same-named declarations in different owning scopes remain
    distinct.
    """
    name = item.get("name")
    raw_signature = item.get("signature")
    signature = (
        _normalize_symbol_key(raw_signature, language)
        if isinstance(raw_signature, str)
        else ""
    )
    if isinstance(name, str) and name.strip():
        return canonical_json(
            {
                "kind": kind,
                "name": _normalize_symbol_key(name, language),
                "scope_id": scope_id,
                "signature": signature,
            }
        )
    return canonical_json(
        {
            "kind": kind,
            "scope_id": scope_id,
            "value": item,
        }
    )


def _fact_occurrence_key(
    item: Mapping[str, object], kind: str, language: str
) -> str:
    """Return a signature-independent authoritative allocation counter key."""

    name = item.get("name")
    return canonical_json(
        {
            "kind": kind,
            "name": (
                _normalize_symbol_key(name, language)
                if isinstance(name, str)
                else ""
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class FactLedger:
    """Lossless internal structured fact ledger for one file (D7/section 8)."""

    functions: tuple[Mapping[str, object], ...] = ()
    classes: tuple[Mapping[str, object], ...] = ()
    exports: tuple[str, ...] = ()
    export_provenance: tuple[tuple[Mapping[str, object], ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "functions", tuple(self.functions))
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "exports", tuple(self.exports))
        export_provenance = tuple(
            tuple(origins) for origins in self.export_provenance
        )
        if export_provenance and len(export_provenance) != len(self.exports):
            raise ValueError("export provenance must align with exports.")
        object.__setattr__(self, "export_provenance", export_provenance)


def _merge_symbol_descriptions(existing: dict, candidate: Mapping[str, object]) -> dict:
    """Deterministically refine a compatible description; first source wins."""
    result = dict(existing)
    existing_desc = existing.get("description", "")
    candidate_desc = candidate.get("description", "") if isinstance(candidate, Mapping) else ""
    if not existing_desc and candidate_desc:
        result["description"] = candidate_desc
    existing_signature = existing.get("signature", "")
    candidate_signature = (
        candidate.get("signature", "") if isinstance(candidate, Mapping) else ""
    )
    if not existing_signature and candidate_signature:
        # A continuation may omit the declaration signature. When its signed
        # report is allocated to the same authoritative scope, retain that
        # bounded matching metadata instead of letting response order erase it.
        result["signature"] = candidate_signature
    origins: list[Mapping[str, object]] = []
    seen_origins: set[str] = set()
    for source in (existing, candidate):
        raw_origins = source.get("_provenance", ())
        if not isinstance(raw_origins, (list, tuple)):
            continue
        for origin in raw_origins:
            if not isinstance(origin, Mapping):
                continue
            key = canonical_json(origin)
            if key not in seen_origins:
                seen_origins.add(key)
                origins.append(dict(origin))
    if origins:
        result["_provenance"] = origins
    return result


def _structural_name_parts(name: str, language: str) -> tuple[str, str]:
    qualified = _normalize_symbol_key(name, language)
    parts = re.split(r"(?:::|[./#$\\])", name)
    short = _normalize_symbol_key(parts[-1], language) if parts else qualified
    return qualified, short


def _structural_kind_matches(source_kind: str, fact_kind: str) -> bool:
    kind = source_kind.lower()
    class_like = any(
        token in kind
        for token in (
            "class",
            "enum",
            "interface",
            "object",
            "record",
            "struct",
            "trait",
            "type_declaration",
        )
    )
    return class_like if fact_kind == "classes" else not class_like


def _matching_structural_candidates(
    item: Mapping[str, object],
    kind: str,
    language: str,
    candidates: Sequence[SemanticUnitIdentity | SymbolFact],
) -> tuple[SemanticUnitIdentity | SymbolFact, ...]:
    raw_name = item.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return ()
    name = _normalize_symbol_key(raw_name, language)
    compatible = tuple(
        candidate
        for candidate in candidates
        if _structural_kind_matches(candidate.kind, kind)
    )
    exact = tuple(
        candidate
        for candidate in compatible
        if _structural_name_parts(candidate.qualified_name, language)[0] == name
    )
    named = exact or tuple(
        candidate
        for candidate in compatible
        if _structural_name_parts(candidate.qualified_name, language)[1] == name
    )
    raw_signature = item.get("signature")
    if not named or not isinstance(raw_signature, str) or not raw_signature.strip():
        return named
    signature = re.sub(
        r"\s+",
        "",
        _normalize_symbol_key(raw_signature, language),
    )
    signature_matches = tuple(
        candidate
        for candidate in named
        if (
            candidate_signature := re.sub(
                r"\s+",
                "",
                _normalize_symbol_key(candidate.signature, language),
            )
        )
        and (
            signature in candidate_signature
            or candidate_signature in signature
        )
    )
    return signature_matches or named


def _matching_fact_units(
    item: Mapping[str, object],
    kind: str,
    language: str,
    chunk: ChunkPlan,
) -> tuple[SemanticUnitIdentity, ...]:
    return tuple(
        candidate
        for candidate in _matching_structural_candidates(
            item,
            kind,
            language,
            chunk.semantic_units,
        )
        if isinstance(candidate, SemanticUnitIdentity)
    )


def _matching_fact_symbols(
    item: Mapping[str, object],
    kind: str,
    language: str,
    chunk: ChunkPlan,
    symbols: Sequence[SymbolFact],
) -> tuple[SymbolFact, ...]:
    candidates = tuple(
        candidate
        for candidate in _matching_structural_candidates(
            item,
            kind,
            language,
            symbols,
        )
        if isinstance(candidate, SymbolFact)
    )
    visible = tuple(
        symbol
        for symbol in candidates
        if any(
            source_range.start_byte
            <= symbol.range.start_byte
            < source_range.end_byte
            for source_range in chunk.owning_ranges
        )
    )
    return visible or candidates


def _fact_scope(
    item: Mapping[str, object],
    kind: str,
    language: str,
    chunk: ChunkPlan,
    occurrence: int,
    symbols: Sequence[SymbolFact],
    reserved_scope_ids: set[str] | None = None,
    fallback_unit_signatures: dict[str, str] | None = None,
) -> tuple[
    str,
    tuple[SemanticUnitIdentity, ...],
    SymbolFact | None,
]:
    symbol_candidates = _matching_fact_symbols(
        item,
        kind,
        language,
        chunk,
        symbols,
    )
    reserved = reserved_scope_ids if reserved_scope_ids is not None else set()
    available_symbols = tuple(
        symbol for symbol in symbol_candidates if symbol.symbol_id not in reserved
    )
    if available_symbols or len(symbol_candidates) == 1:
        # A sole already-reserved exact match is a duplicate report for the
        # same declaration and intentionally consolidates. Otherwise consume
        # the first still-unassigned parser symbol.
        symbol = (
            available_symbols[0]
            if available_symbols
            else symbol_candidates[0]
        )
        reserved.add(symbol.symbol_id)
        owner_units = tuple(
            unit
            for unit in chunk.semantic_units
            if symbol.atom_id in unit.atom_ids
        )
        return (
            symbol.symbol_id,
            owner_units or chunk.semantic_units,
            symbol,
        )
    if symbol_candidates:
        # Parser symbols are authoritative.  When multiple matching
        # declarations have all been consumed, an additional ambiguous model
        # fact must not fall through to their still-unreserved semantic-unit
        # IDs and acquire false declaration ownership.
        return (
            _domain_id(
                "fact-scope",
                {
                    "revision": LEDGER_SCHEMA_REVISION,
                    "group_unit_id": chunk.unit_id,
                    "fact_key": _semantic_key(item, kind, language),
                    "occurrence": occurrence,
                },
            ),
            chunk.semantic_units,
            None,
        )
    candidates = _matching_fact_units(item, kind, language, chunk)
    available_units = tuple(
        unit for unit in candidates if unit.unit_id not in reserved
    )
    if available_units or len(candidates) == 1:
        unit = available_units[0] if available_units else candidates[0]
        reserved.add(unit.unit_id)
        return unit.unit_id, (unit,), None
    if len(chunk.semantic_units) == 1:
        unit = chunk.semantic_units[0]
        raw_signature = item.get("signature")
        signature = (
            _normalize_symbol_key(raw_signature, language)
            if isinstance(raw_signature, str) and raw_signature.strip()
            else ""
        )
        assigned_signatures = (
            fallback_unit_signatures
            if fallback_unit_signatures is not None
            else {}
        )
        assigned_signature = assigned_signatures.get(unit.unit_id)
        if unit.unit_id not in reserved:
            reserved.add(unit.unit_id)
            assigned_signatures[unit.unit_id] = signature
            return unit.unit_id, (unit,), None
        if not signature or signature == assigned_signature:
            # An unsigned continuation or exact duplicate remains attached to
            # the one authoritative unit. A second explicit, different
            # signature cannot safely inherit that declaration identity.
            return unit.unit_id, (unit,), None
    return (
        _domain_id(
            "fact-scope",
            {
                "revision": LEDGER_SCHEMA_REVISION,
                "group_unit_id": chunk.unit_id,
                "fact_key": _semantic_key(item, kind, language),
                "occurrence": occurrence,
            },
        ),
        chunk.semantic_units,
        None,
    )


def _fact_origin(
    chunk: ChunkPlan,
    semantic_units: Sequence[SemanticUnitIdentity],
    symbol: SymbolFact | None,
) -> dict:
    origin = {
        "chunk_id": chunk.chunk_id,
        "source_order": chunk.global_index,
        "semantic_unit_ids": [unit.unit_id for unit in semantic_units],
        "semantic_units": [
            {
                "unit_id": unit.unit_id,
                "kind": unit.kind,
                "qualified_name": unit.qualified_name,
                "signature": unit.signature,
                "source_range": unit.source_range.to_public(),
            }
            for unit in semantic_units
        ],
        "semantic_unit_ranges": [
            unit.source_range.to_public() for unit in semantic_units
        ],
        "owning_ranges": [
            source_range.to_public() for source_range in chunk.owning_ranges
        ],
    }
    if symbol is not None:
        origin["symbol_id"] = symbol.symbol_id
        origin["symbol"] = {
            "symbol_id": symbol.symbol_id,
            "atom_id": symbol.atom_id,
            "kind": symbol.kind,
            "qualified_name": symbol.qualified_name,
            "signature": symbol.signature,
            "source_range": symbol.range.to_public(),
        }
    return origin


def build_fact_ledger(
    cleaned_leaf_capsules: Sequence[Mapping[str, object]],
    *,
    language: str = "",
    chunks: Sequence[ChunkPlan] | None = None,
    symbols: Sequence[SymbolFact] | None = None,
) -> FactLedger:
    """Deterministically merge cleaned leaf capsules into one whole-file ledger.

    Without aligned chunks, functions/classes are deduplicated by normalized
    name, kind, and reported signature. With aligned chunks, parser-owned symbol
    or semantic-unit scope is authoritative and reported signatures are only
    matching hints. Order is deterministic first-source order.
    """
    authoritative_chunks = tuple(chunks) if chunks is not None else None
    authoritative_symbols = tuple(symbols) if symbols is not None else ()
    if symbols is not None and authoritative_chunks is None:
        raise ValueError("authoritative symbols require aligned planned chunks.")
    if any(
        not isinstance(symbol, SymbolFact)
        for symbol in authoritative_symbols
    ):
        raise ValueError("authoritative symbols must be SymbolFact values.")
    if (
        authoritative_chunks is not None
        and len(authoritative_chunks) != len(cleaned_leaf_capsules)
    ):
        raise ValueError("leaf capsules must align one-for-one with planned chunks.")
    functions: dict[str, dict] = {}
    classes: dict[str, dict] = {}
    exports: dict[str, tuple[str, list[Mapping[str, object]]]] = {}
    for capsule_index, capsule in enumerate(cleaned_leaf_capsules):
        if not isinstance(capsule, Mapping):
            continue
        chunk = (
            authoritative_chunks[capsule_index]
            if authoritative_chunks is not None
            else None
        )
        chunk_symbols: tuple[SymbolFact, ...] = ()
        if chunk is not None and authoritative_symbols:
            atom_ids = {
                atom_id
                for unit in chunk.semantic_units
                for atom_id in unit.atom_ids
            }
            chunk_symbols = tuple(
                symbol
                for symbol in authoritative_symbols
                if symbol.atom_id in atom_ids
            )
        for kind, bucket in (("functions", functions), ("classes", classes)):
            raw_items = tuple(capsule.get(kind, []) or [])
            valid_items = tuple(
                (index, item)
                for index, item in enumerate(raw_items)
                if isinstance(item, Mapping)
                and isinstance(item.get("name"), str)
            )
            # Allocate explicit signatures before unsigned facts regardless of
            # model response order. Signed facts reserve their authoritative
            # parser match; unsigned facts then consume the first unassigned
            # same-name symbol/unit. Canonical item ordering makes ambiguous
            # unsigned allocation independent of response list order.
            allocation_order = sorted(
                valid_items,
                key=lambda pair: (
                    0
                    if isinstance(pair[1].get("signature"), str)
                    and bool(str(pair[1].get("signature", "")).strip())
                    else 1,
                    canonical_json(pair[1]),
                ),
            )
            occurrences: dict[str, int] = {}
            reserved_scope_ids: set[str] = set()
            fallback_unit_signatures: dict[str, str] = {}
            allocated_scopes: dict[
                int,
                tuple[str, tuple[SemanticUnitIdentity, ...], SymbolFact | None],
            ] = {}
            for item_index, item in allocation_order:
                # Signed and unsigned reports of the same declaration share
                # one authoritative occurrence namespace.  Signature text is
                # matching metadata only; it cannot choose a separate counter.
                base_key = _fact_occurrence_key(item, kind, language)
                occurrence = occurrences.get(base_key, 0)
                occurrences[base_key] = occurrence + 1
                if chunk is not None:
                    allocated_scopes[item_index] = _fact_scope(
                        item,
                        kind,
                        language,
                        chunk,
                        occurrence,
                        chunk_symbols,
                        reserved_scope_ids,
                        fallback_unit_signatures,
                    )

            output_items = valid_items
            if chunk is not None:
                def _allocated_source_key(
                    pair: tuple[int, Mapping[str, object]],
                ) -> tuple[int, str]:
                    scope_id, semantic_units, symbol = allocated_scopes[pair[0]]
                    if symbol is not None:
                        start_byte = symbol.range.start_byte
                    elif semantic_units:
                        start_byte = min(
                            unit.source_range.start_byte for unit in semantic_units
                        )
                    elif chunk.owning_ranges:
                        start_byte = min(
                            source_range.start_byte
                            for source_range in chunk.owning_ranges
                        )
                    else:
                        start_byte = chunk.source_range.start_byte
                    return start_byte, canonical_json(
                        {"scope_id": scope_id, "fact": pair[1]}
                    )

                output_items = tuple(sorted(valid_items, key=_allocated_source_key))

            for item_index, item in output_items:
                scope_id = ""
                local_origin: dict | None = None
                if chunk is not None:
                    scope_id, semantic_units, symbol = allocated_scopes[item_index]
                    local_origin = _fact_origin(
                        chunk,
                        semantic_units,
                        symbol,
                    )
                if chunk is not None:
                    # The parser/plan-owned scope is declaration identity.
                    # This consolidates signed/unsigned continuation reports
                    # while distinct overloads retain distinct symbol/unit IDs.
                    key = canonical_json(
                        {
                            "kind": kind,
                            "name": _normalize_symbol_key(
                                str(item.get("name", "")), language
                            ),
                            "scope_id": scope_id,
                        }
                    )
                else:
                    key = _semantic_key(item, kind, language)
                candidate = {
                    field: value
                    for field, value in item.items()
                    if field != "_provenance"
                }
                if chunk is not None and symbol is not None:
                    if symbol.signature:
                        candidate["signature"] = symbol.signature
                    else:
                        candidate.pop("signature", None)
                if local_origin is not None:
                    candidate["_provenance"] = [local_origin]
                if key in bucket:
                    bucket[key] = _merge_symbol_descriptions(bucket[key], candidate)
                else:
                    bucket[key] = candidate
        for export in capsule.get("exports", []) or []:
            if not isinstance(export, str) or not export.strip():
                continue
            key = f"export:{_normalize_symbol_key(export, language)}"
            origin = (
                _fact_origin(chunk, chunk.semantic_units, None)
                if chunk is not None
                else None
            )
            if key not in exports:
                exports[key] = (export, [origin] if origin is not None else [])
            elif origin is not None:
                value, origins = exports[key]
                origin_key = canonical_json(origin)
                if all(canonical_json(existing) != origin_key for existing in origins):
                    origins.append(origin)
    return FactLedger(
        functions=tuple(functions.values()),
        classes=tuple(classes.values()),
        exports=tuple(value for value, _origins in exports.values()),
        export_provenance=tuple(
            tuple(origins) for _value, origins in exports.values()
        ),
    )


def merge_leaf_capsules(
    cleaned_leaf_capsules: Sequence[Mapping[str, object]],
    *,
    language: str = "",
    chunks: Sequence[ChunkPlan] | None = None,
    symbols: Sequence[SymbolFact] | None = None,
) -> FactLedger:
    """Whole-file entry point: merge every cleaned leaf capsule into the ledger."""
    return build_fact_ledger(
        cleaned_leaf_capsules,
        language=language,
        chunks=chunks,
        symbols=symbols,
    )


def refine_narrative_inputs(narratives: Sequence[str]) -> tuple[str, ...]:
    """Deterministic, order-preserving de-duplication of child narrative text.

    Applied at each reduction node to its direct children's narrative text so
    repeated boilerplate (e.g. many "comment-only fragment" leaves) does not
    grow the prompt at every level.
    """
    seen: set[str] = set()
    out: list[str] = []
    for text in narratives:
        if isinstance(text, str) and text.strip() and text not in seen:
            seen.add(text)
            out.append(text)
    return tuple(out)


def _ledger_synopsis(ledger: FactLedger) -> dict:
    def _names(items: Sequence[Mapping[str, object]]) -> list[str]:
        names: list[str] = []
        for item in items:
            name = item.get("name") if isinstance(item, Mapping) else None
            if isinstance(name, str) and name:
                names.append(name)
        return names

    synopsis = {
        "functions": _names(ledger.functions),
        "classes": _names(ledger.classes),
        "exports": list(ledger.exports),
    }
    _trim_synopsis_lists(
        synopsis,
        lambda: len(canonical_json(synopsis))
        <= MAX_LEDGER_SYNOPSIS_CHARS,
    )
    return synopsis


def _trim_synopsis_lists(synopsis: dict, fits) -> None:
    """Apply the frozen synopsis trim order in logarithmic serializations.

    The prior item-at-a-time loop was exact but quadratic for a conservative
    maximum ledger. Each list is an ordered prefix after trimming, so binary
    search produces the identical retained prefix and priority behavior.
    """
    if fits():
        return
    for key in ("functions", "classes", "exports"):
        original = synopsis[key]
        if not original:
            continue
        synopsis[key] = []
        if not fits():
            continue
        low = 0
        high = len(original)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            synopsis[key] = original[:middle]
            if fits():
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        synopsis[key] = original[:best]
        return


def final_synthesis_input(
    *,
    rel_path: str,
    language: str,
    imports: Sequence[str],
    root_narratives: Sequence[str],
    root_coverage_leaf_ids: Sequence[str],
    ledger: FactLedger,
    max_chars: int | None = None,
) -> str:
    """Bounded canonical JSON manifest for the final synthesis prompt (D10).

    Grounds the final call with deterministic path/language/imports, the
    refined root narrative(s), exact root coverage, and a bounded
    fact-ledger synopsis.  Never includes raw source, internal IDs, or
    ranges.

    ``root_narratives`` is plural: the final node's direct children are not
    always reduced to exactly one — general reduction stops once the
    remaining count already fits the final synthesis envelope (D6), so the
    final call may itself combine a handful of already-refined root-level
    narratives.  Callers should already have de-duplicated them (see
    :func:`refine_narrative_inputs`).
    """
    synopsis = _ledger_synopsis(ledger)
    payload = {
        "file_path": normalize_rel_path(rel_path),
        "language": language,
        "imports": list(imports),
        "root_narratives": list(root_narratives),
        "root_coverage_leaf_count": len(set(root_coverage_leaf_ids)),
        "fact_ledger_synopsis": synopsis,
    }
    if max_chars is not None:
        _real_int(max_chars, "max_chars", minimum=1)
        # The full manifest, not merely the ledger field, owns the ceiling.
        # Shrink only the already-lossy synopsis; path, language, imports,
        # narratives, and exact coverage count remain authoritative.
        _trim_synopsis_lists(
            synopsis,
            lambda: len(canonical_json(payload)) <= max_chars,
        )
    return canonical_json(payload)


_MAXIMUM_JSON_ESCAPE_ALPHABET = tuple(
    chr(code)
    for code in (*range(0, 8), *range(14, 28))
)


def _maximum_json_escaped_text(length: int, ordinal: int) -> str:
    _real_int(length, "maximum JSON-escaped text length", minimum=1)
    _real_int(ordinal, "maximum JSON-escaped text ordinal")
    base = len(_MAXIMUM_JSON_ESCAPE_ALPHABET)
    digits: list[str] = []
    value = ordinal
    while True:
        digits.append(_MAXIMUM_JSON_ESCAPE_ALPHABET[value % base])
        value //= base
        if value == 0:
            break
    if len(digits) > length:
        raise DivisionInternalDefect(
            "maximum JSON-escaped placeholder ordinal exceeds its field bound."
        )
    return (
        _MAXIMUM_JSON_ESCAPED_CHAR * (length - len(digits))
        + "".join(reversed(digits))
    )


def maximally_populated_fact_ledger(
    leaf_count: int,
    *,
    language: str = "",
) -> FactLedger:
    """Construct the largest schema-valid ledger reachable from *leaf_count*.

    Each leaf capsule may contribute the fixed maximum number of functions,
    classes, and exports. Names are distinct and maximum-length so deterministic
    synopsis truncation sees the true worst case instead of an empty or
    accidentally de-duplicated placeholder ledger.
    """
    _real_int(leaf_count, "leaf_count", minimum=1)

    def _name(ordinal: int, maximum: int) -> str:
        return _maximum_json_escaped_text(maximum, ordinal)

    capsules: list[dict] = []
    for leaf_index in range(leaf_count):
        base = leaf_index * MAX_LEAF_SYMBOL_ITEMS_PER_KIND
        functions = [
            {
                "name": _name(
                    base + item_index,
                    MAX_LEAF_SYMBOL_NAME_CHARS,
                ),
                "description": (
                    _MAXIMUM_JSON_ESCAPED_CHAR
                    * MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
                ),
                "signature": (
                    _MAXIMUM_JSON_ESCAPED_CHAR
                    * MAX_LEAF_SYMBOL_SIGNATURE_CHARS
                ),
            }
            for item_index in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
        ]
        classes = [
            {
                "name": _name(
                    base + item_index,
                    MAX_LEAF_SYMBOL_NAME_CHARS,
                ),
                "description": (
                    _MAXIMUM_JSON_ESCAPED_CHAR
                    * MAX_LEAF_SYMBOL_DESCRIPTION_CHARS
                ),
                "signature": (
                    _MAXIMUM_JSON_ESCAPED_CHAR
                    * MAX_LEAF_SYMBOL_SIGNATURE_CHARS
                ),
            }
            for item_index in range(MAX_LEAF_SYMBOL_ITEMS_PER_KIND)
        ]
        exports = [
            _name(
                leaf_index * MAX_LEAF_EXPORT_ITEMS + item_index,
                MAX_LEAF_EXPORT_ITEM_CHARS,
            )
            for item_index in range(MAX_LEAF_EXPORT_ITEMS)
        ]
        capsules.append(
            {
                "description": (
                    _MAXIMUM_JSON_ESCAPED_CHAR
                    * MAX_LEAF_DESCRIPTION_CHARS
                ),
                "functions": functions,
                "classes": classes,
                "exports": exports,
            }
        )
    return build_fact_ledger(capsules, language=language)


def maximum_distinct_narratives(count: int) -> tuple[str, ...]:
    """Return *count* distinct maximum-length valid reduction narratives."""
    _real_int(count, "narrative count", minimum=1)
    return tuple(
        _maximum_json_escaped_text(
            MAX_REDUCTION_NARRATIVE_CHARS,
            ordinal,
        )
        for ordinal in range(count)
    )


def worst_case_final_synthesis_chars(
    *,
    rel_path: str,
    language: str,
    imports: Sequence[str],
    root_count: int,
    leaf_count: int,
    max_chars: int | None = None,
) -> int:
    """Return the conservative final-manifest character bound used by planning.

    Deterministic fields use their real values. Response-derived fields use
    distinct maximum-size narratives and reserve the complete bounded ledger
    synopsis. Therefore any live cleaned final manifest for the same plan is
    no larger than this provider-free character count.
    """
    _real_int(root_count, "root_count", minimum=1)
    _real_int(leaf_count, "leaf_count", minimum=1)
    empty_synopsis = {
        "functions": [],
        "classes": [],
        "exports": [],
    }
    minimum_manifest = final_synthesis_input(
        rel_path=rel_path,
        language=language,
        imports=imports,
        root_narratives=maximum_distinct_narratives(root_count),
        root_coverage_leaf_ids=tuple(
            f"leaf-{ordinal}" for ordinal in range(leaf_count)
        ),
        ledger=FactLedger(),
    )
    minimum_chars = len(minimum_manifest)
    full_bound = (
        minimum_chars
        - len(canonical_json(empty_synopsis))
        + MAX_LEDGER_SYNOPSIS_CHARS
    )
    if max_chars is None:
        return full_bound
    _real_int(max_chars, "max_chars", minimum=1)
    if minimum_chars > max_chars:
        return minimum_chars
    return min(full_bound, max_chars)


# ---------------------------------------------------------------------------
# Provider/model/effective-endpoint execution identity (D12/D10)
# ---------------------------------------------------------------------------


def provider_execution_identity(config: Mapping[str, object]) -> str:
    """Provider-free digest of the resolved provider/model/effective-endpoint.

    Derived from the same pure selection logic used by provider creation
    (`describe_provider_selection`), so recovery identity and the provider
    actually constructed for execution can never silently diverge.  Never
    persists the literal `auto` selector, API keys, or credentials.
    """
    from codedoc.llm.factory import describe_provider_execution

    descriptor = describe_provider_execution(dict(config))
    payload = {
        "revision": "provider-execution-identity-v1",
        "provider": descriptor.provider_kind,
        "model": descriptor.model,
        "endpoint": descriptor.endpoint_identity,
    }
    return _digest("provider-execution", payload)


def verify_provider_execution_identity(
    provider: object,
    config: Mapping[str, object],
    planned_identity: str,
) -> None:
    """Confirm a constructed provider matches the provider-free plan.

    Production providers carry a factory-authored descriptor derived from the
    concrete constructor branch and arguments. Alternate factories and test
    doubles must attach the same explicit descriptor; a missing or malformed
    attestation fails closed.
    """
    from codedoc.llm.factory import (
        constructed_provider_execution,
    )
    from codedoc.utils.errors import ConfigError

    _digest_text(
        planned_identity,
        "planned provider identity",
        prefix="provider-execution",
    )
    descriptor = constructed_provider_execution(provider)
    if descriptor is None:
        raise ConfigError(
            "Constructed LLM provider is missing a valid concrete execution "
            "attestation. Provider factories and injected providers must attach "
            "their provider, model, and effective-endpoint descriptor before "
            "execution."
        )
    actual_identity = _digest(
        "provider-execution",
        {
            "revision": "provider-execution-identity-v1",
            "provider": descriptor.provider_kind,
            "model": descriptor.model,
            "endpoint": descriptor.endpoint_identity,
        },
    )
    if actual_identity != planned_identity:
        raise ConfigError(
            "Constructed LLM provider identity does not match the provider-free "
            "plan (provider, model, or effective endpoint changed). Re-run "
            "planning with one stable resolved configuration."
        )


def leaf_execution_identity(
    *,
    rel_path: str,
    content_hash: str,
    division_plan_digest: str,
    provider_identity: str,
    chunk: ChunkPlan,
) -> str:
    """Execution identity for one leaf chunk node.

    Never includes the final prompt-profile digest: the fixed leaf prompt
    never varies with a customized final shape (D10/D12).
    """
    _digest_text(content_hash, "content_hash")
    return _digest(
        "division-execution",
        {
            "revision": EXECUTION_IDENTITY_SCHEMA_REVISION,
            "node_kind": "leaf",
            "path": normalize_rel_path(rel_path),
            "content_hash": content_hash,
            "division_plan_digest": division_plan_digest,
            "provider_identity": provider_identity,
            "node_id": chunk.chunk_id,
            "node_type": "leaf",
            "group_unit_id": chunk.unit_id,
            "semantic_unit_ids": [
                unit.unit_id for unit in chunk.semantic_units
            ],
            "group_chunk_index": chunk.unit_chunk_index,
            "group_chunk_count": chunk.unit_chunk_count,
            "owning_ranges": [
                source_range.to_public()
                for source_range in chunk.owning_ranges
            ],
            "coverage_leaf_ids": [chunk.chunk_id],
            "leaf_capsule_revision": LEAF_CAPSULE_SCHEMA_REVISION,
        },
    )


def reduction_execution_identity(
    *,
    rel_path: str,
    content_hash: str,
    division_plan_digest: str,
    reduction_tree_digest: str,
    provider_identity: str,
    node: ReductionNodePlan,
) -> str:
    """Execution identity for one unit-consolidation or general reduction node."""
    _digest_text(content_hash, "content_hash")
    return _digest(
        "division-execution",
        {
            "revision": EXECUTION_IDENTITY_SCHEMA_REVISION,
            "node_kind": "reduction",
            "path": normalize_rel_path(rel_path),
            "content_hash": content_hash,
            "division_plan_digest": division_plan_digest,
            "reduction_tree_digest": reduction_tree_digest,
            "provider_identity": provider_identity,
            "node_id": node.node_id,
            "node_type": node.phase,
            "unit_id": node.unit_id,
            "level": node.level,
            "ordinal": node.ordinal,
            "child_ids": list(node.child_ids),
            "coverage_leaf_ids": list(node.leaf_ids),
            "reduction_capsule_revision": REDUCTION_CAPSULE_SCHEMA_REVISION,
            "reducer_prompt_revision": REDUCER_PROMPT_REVISION,
        },
    )


def final_execution_identity(
    *,
    rel_path: str,
    content_hash: str,
    division_plan_digest: str,
    reduction_tree_digest: str,
    provider_identity: str,
    prompt_profile_digest: str,
    imports_digest: str,
    node: ReductionNodePlan,
) -> str:
    """Execution identity for the final synthesis node.

    Additionally includes the resolved final-shape digest, so changing a
    final prompt profile reuses compatible leaves/reducers but reruns final
    synthesis only.  Also includes the deterministic imports digest
    (section 6): leaf and reducer identities deliberately exclude it, so an
    equal-length import change reruns only final synthesis.
    """
    _digest_text(content_hash, "content_hash")
    return _digest(
        "division-execution",
        {
            "revision": EXECUTION_IDENTITY_SCHEMA_REVISION,
            "node_kind": "final",
            "path": normalize_rel_path(rel_path),
            "content_hash": content_hash,
            "division_plan_digest": division_plan_digest,
            "reduction_tree_digest": reduction_tree_digest,
            "provider_identity": provider_identity,
            "prompt_profile_digest": prompt_profile_digest,
            "imports_digest": imports_digest,
            "node_id": node.node_id,
            "node_type": node.phase,
            "level": node.level,
            "ordinal": node.ordinal,
            "child_ids": list(node.child_ids),
            "coverage_leaf_ids": list(node.leaf_ids),
            "ledger_revision": LEDGER_SCHEMA_REVISION,
            "final_synthesis_revision": FINAL_SYNTHESIS_REVISION,
        },
    )


# ---------------------------------------------------------------------------
# Stage-local exact input digests (section 10)
# ---------------------------------------------------------------------------
# Domain-separated from execution identity: execution identity binds *which*
# planned node this is and *which* provider serviced it; an input digest binds
# only the exact stage-local inputs a node's prompt was built from, so a
# recovered reducer or final result can never be silently paired with
# different child results that happen to share planned IDs (section 11).

LEAF_INPUT_DIGEST_REVISION = "leaf-input-v1"
REDUCTION_INPUT_DIGEST_REVISION = "reduction-input-v1"
FINAL_INPUT_DIGEST_REVISION = "final-input-v1"


def leaf_input_digest(
    *,
    rel_path: str,
    language: str,
    chunk: ChunkPlan,
    unit_indexes: Sequence[int],
    unit_count: int,
) -> str:
    """Exact stage-local input digest for one leaf chunk.

    Provider-agnostic by design (never includes a provider identity -- that
    is ``leaf_execution_identity``'s job): binds only the fixed prompt
    inputs -- effective language, chunk payload, metadata, ranges, and unit
    identities -- plus the leaf capsule revision.
    """
    return _digest(
        "leaf-input",
        {
            "revision": LEAF_INPUT_DIGEST_REVISION,
            "path": normalize_rel_path(rel_path),
            "language": language,
            "fragment_index": chunk.unit_chunk_index + 1,
            "fragment_count": chunk.unit_chunk_count,
            "global_index": chunk.global_index + 1,
            "global_count": chunk.global_count,
            "fragment_metadata": render_leaf_prompt_metadata(
                group_unit_id=chunk.group_unit_id,
                semantic_units=chunk.semantic_units,
                unit_indexes=unit_indexes,
                unit_count=unit_count,
                owning_ranges=chunk.owning_ranges,
            ),
            "unit_chunk_index": chunk.unit_chunk_index,
            "unit_chunk_count": chunk.unit_chunk_count,
            "continuation_before": chunk.continuation_before,
            "continuation_after": chunk.continuation_after,
            "known_symbols": list(chunk.known_symbols),
            "payload": chunk.payload,
            "leaf_capsule_revision": LEAF_CAPSULE_SCHEMA_REVISION,
        },
    )


def reduction_input_digest(
    *,
    rel_path: str,
    phase: str,
    level: int,
    unit_id: str | None,
    child_count: int,
    ordered_child_narratives: Sequence[str],
) -> str:
    """Exact stage-local input digest for one unit-consolidation or general
    reduction node.

    Binds the exact rendered ordered child narratives a reducer actually
    consumed (after the same order-preserving de-duplication the reducer
    prompt itself applies -- see :func:`refine_narrative_inputs`), plus
    phase, level, unit, and the reducer prompt revision.
    """
    return _digest(
        "reduction-input",
        {
            "revision": REDUCTION_INPUT_DIGEST_REVISION,
            "path": normalize_rel_path(rel_path),
            "phase": phase,
            "level": level,
            "unit_id": unit_id,
            "child_count": child_count,
            "child_narratives": list(ordered_child_narratives),
            "reducer_prompt_revision": REDUCER_PROMPT_REVISION,
        },
    )


def final_input_digest(
    *,
    imports_digest: str,
    resolved_shape_digest: str,
    manifest_json: str,
) -> str:
    """Exact stage-local input digest for the final synthesis node.

    ``manifest_json`` is the exact bounded canonical JSON manifest string
    already built by :func:`final_synthesis_input` for the final prompt --
    reused directly rather than re-derived, so this digest is guaranteed to
    bind precisely what the final call actually received.
    """
    return _digest(
        "final-input",
        {
            "revision": FINAL_INPUT_DIGEST_REVISION,
            "imports_digest": imports_digest,
            "resolved_shape_digest": resolved_shape_digest,
            "manifest_json": manifest_json,
            "ledger_revision": LEDGER_SCHEMA_REVISION,
            "final_synthesis_revision": FINAL_SYNTHESIS_REVISION,
        },
    )


# ---------------------------------------------------------------------------
# Node-keyed split-partial recovery (schema version 3) — section 9/section 12
# ---------------------------------------------------------------------------


QUARANTINE_REASONS: frozenset[str] = frozenset(
    {
        "unknown-node",
        "stage-mismatch",
        "stale-identity",
        "stale-revision",
        "child-order-mismatch",
        "coverage-mismatch",
        "input-digest-mismatch",
        "live-schema-mismatch",
    }
)
MAX_QUARANTINE_ENTRIES_PER_FILE = 32
MAX_QUARANTINE_ENTRY_CHARS = 4096


@dataclass(frozen=True, slots=True)
class ReductionNodeState:
    """One recoverable, dependency-validated checkpoint: leaf, unit-
    consolidation, general reduction, or final synthesis.

    ``reduction_tree_digest`` is deliberately NOT a field here (schema 3):
    the writer-time tree digest is container-level provenance only, never a
    per-node reuse gate, so a topology-only change (compatible chunk payload,
    changed reduction shape) does not invalidate a compatible leaf.
    """

    node_id: str
    node_type: Literal["leaf", "unit-consolidation", "general", "final"]
    rel_path: str
    content_hash: str
    division_plan_digest: str
    execution_identity_digest: str
    input_digest: str
    unit_id: str | None
    child_ids: tuple[str, ...]
    coverage_leaf_ids: tuple[str, ...]
    result_json: str

    def __post_init__(self) -> None:
        rel_path = normalize_rel_path(self.rel_path)
        if self.node_type not in ("leaf", "unit-consolidation", "general", "final"):
            raise ValueError("invalid reduction node state type.")
        _digest_text(self.content_hash, "content_hash")
        _digest_text(self.division_plan_digest, "division_plan_digest", prefix="division-plan")
        _digest_text(
            self.execution_identity_digest,
            "execution_identity_digest",
            prefix="division-execution",
        )
        if not isinstance(self.input_digest, str) or ":" not in self.input_digest:
            raise ValueError("input_digest must be a domain-prefixed digest.")
        child_ids = tuple(self.child_ids)
        coverage_leaf_ids = tuple(self.coverage_leaf_ids)
        if not coverage_leaf_ids:
            raise ValueError("node state must cover at least one leaf.")
        if len(coverage_leaf_ids) != len(set(coverage_leaf_ids)):
            raise ValueError("node state coverage has duplicate leaves.")
        load_canonical_json_object(self.result_json)
        object.__setattr__(self, "rel_path", rel_path)
        object.__setattr__(self, "child_ids", child_ids)
        object.__setattr__(self, "coverage_leaf_ids", coverage_leaf_ids)


@dataclass(frozen=True, slots=True)
class QuarantineEntry:
    """One bounded, non-executable record of a rejected node (section 9/14).

    Preserved until a valid replacement with the same ``node_id`` is
    transactionally checkpointed, or the file's completed record is
    transactionally recorded. Never enters public output, prompts, logs, or
    recovered-work counts.
    """

    node_id: str
    reason: str
    raw_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id:
            raise ValueError("quarantine entry node_id must be non-empty text.")
        if self.reason not in QUARANTINE_REASONS:
            raise ValueError(f"unsupported quarantine reason: {self.reason!r}")
        if not isinstance(self.raw_json, str):
            raise ValueError("quarantine entry raw_json must be text.")
        if len(self.raw_json) > MAX_QUARANTINE_ENTRY_CHARS:
            object.__setattr__(self, "raw_json", self.raw_json[:MAX_QUARANTINE_ENTRY_CHARS])


@dataclass(frozen=True, slots=True)
class SplitTreeState:
    """Node-keyed split-partial container, schema version 3.

    Distinct from — and never interchangeable with — the predecessor
    ordered-prefix payload (schema version 1) or the dormant per-node
    tree-digest-gated payload (schema version 2); see D11 and section 9.
    """

    schema_version: Literal[3]
    owner: Literal["codedoc-ai"]
    rel_path: str
    content_hash: str
    division_plan_digest: str
    # Writer-time provenance only, never a per-node reuse gate (section 9).
    reduction_tree_digest: str
    nodes: tuple[ReductionNodeState, ...]
    quarantine: tuple[QuarantineEntry, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_PARTIAL_SCHEMA_VERSION or self.owner != "codedoc-ai":
            raise ValueError("unsupported split-partial tree state.")
        rel_path = normalize_rel_path(self.rel_path)
        _digest_text(self.content_hash, "content_hash")
        _digest_text(
            self.division_plan_digest,
            "division_plan_digest",
            prefix="division-plan",
        )
        _digest_text(
            self.reduction_tree_digest,
            "reduction_tree_digest",
            prefix="reduction-tree",
        )
        nodes = tuple(self.nodes)
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("split-partial tree state has duplicate node IDs.")
        if any(
            node.rel_path != rel_path
            or node.content_hash != self.content_hash
            or node.division_plan_digest != self.division_plan_digest
            for node in nodes
        ):
            raise ValueError("split-partial tree state node identity mismatch.")
        quarantine = tuple(self.quarantine)
        if len(quarantine) > MAX_QUARANTINE_ENTRIES_PER_FILE:
            raise ValueError("quarantine map exceeds the bounded entry count.")
        quarantine_ids = [entry.node_id for entry in quarantine]
        if len(quarantine_ids) != len(set(quarantine_ids)):
            raise ValueError("split-partial quarantine has duplicate node IDs.")
        if set(node_ids) & set(quarantine_ids):
            raise ValueError("a split-partial node cannot also be quarantined.")
        object.__setattr__(self, "rel_path", rel_path)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "quarantine", quarantine)

    def by_id(self) -> dict[str, ReductionNodeState]:
        return {node.node_id: node for node in self.nodes}

    def quarantine_by_id(self) -> dict[str, QuarantineEntry]:
        return {entry.node_id: entry for entry in self.quarantine}


def tree_node_state(
    *,
    node_id: str,
    node_type: Literal["leaf", "unit-consolidation", "general", "final"],
    rel_path: str,
    content_hash: str,
    division_plan_digest: str,
    execution_identity_digest: str,
    input_digest: str,
    unit_id: str | None,
    child_ids: Sequence[str],
    coverage_leaf_ids: Sequence[str],
    result: Mapping[str, object],
) -> ReductionNodeState:
    """Build one checkpointed node state from an already-cleaned result."""
    return ReductionNodeState(
        node_id=node_id,
        node_type=node_type,
        rel_path=rel_path,
        content_hash=content_hash,
        division_plan_digest=division_plan_digest,
        execution_identity_digest=execution_identity_digest,
        input_digest=input_digest,
        unit_id=unit_id,
        child_ids=tuple(child_ids),
        coverage_leaf_ids=tuple(coverage_leaf_ids),
        result_json=canonical_json(dict(result)),
    )


def node_state_public_json(node: ReductionNodeState) -> dict:
    """The exact per-node public wire shape (section 9), shared by the
    container serializer (``safe_writer._tree_state_to_json``) and quarantine
    raw-JSON capture so the two never drift apart."""
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "rel_path": node.rel_path,
        "content_hash": node.content_hash,
        "division_plan_digest": node.division_plan_digest,
        "execution_identity_digest": node.execution_identity_digest,
        "input_digest": node.input_digest,
        "unit_id": node.unit_id,
        "child_ids": list(node.child_ids),
        "coverage_leaf_ids": list(node.coverage_leaf_ids),
        "result_json": node.result_json,
    }


def _expected_node_shapes(
    plan: DivisionPlan, tree: ReductionTreePlan
) -> dict[
    str,
    tuple[
        Literal["leaf", "unit-consolidation", "general", "final"],
        tuple[str, ...],
        tuple[str, ...],
        str | None,
    ],
]:
    """Map every valid node_id (leaf chunk or reduction/final node) to its
    expected `(node_type, child_ids, leaf_ids, unit_id)` shape."""
    shapes: dict[
        str,
        tuple[
            Literal["leaf", "unit-consolidation", "general", "final"],
            tuple[str, ...],
            tuple[str, ...],
            str | None,
        ],
    ] = {}
    for chunk in plan.chunks:
        shapes[chunk.chunk_id] = ("leaf", (), (chunk.chunk_id,), None)
    for node in tree.all_nodes:
        shapes[node.node_id] = (
            node.phase,
            node.child_ids,
            node.leaf_ids,
            node.unit_id,
        )
    return shapes


def _node_result_matches_live_schema(
    node: ReductionNodeState,
    expected_type: Literal["leaf", "unit-consolidation", "general", "final"],
    *,
    expected_chunk: ChunkPlan | None,
    resolved_shape: object | None,
    language: str,
    imports: Sequence[str],
) -> bool:
    """Reapply the same cleaner/required-field contract used for live work."""
    try:
        from codedoc.agents.response_cleaning import (
            clean_combined_report,
            clean_leaf_capsule_report,
            clean_reduction_capsule_report,
        )
        from codedoc.agents.response_diagnostics import (
            process_fixed_capsule_response,
            process_response,
        )

        stored = load_canonical_json_object(node.result_json)
        if expected_type == "leaf":
            if expected_chunk is None:
                return False
            if (
                stored.get("chunk_id") != expected_chunk.chunk_id
                or stored.get("unit_id") != expected_chunk.unit_id
            ):
                return False
            capsule = {
                key: value
                for key, value in stored.items()
                if key not in ("chunk_id", "unit_id")
            }
            cleaned = process_fixed_capsule_response(
                canonical_json(capsule),
                label="split-leaf",
                agent="leaf",
                file_path=node.rel_path,
                clean_reporter=clean_leaf_capsule_report,
                requested_paths=(
                    "description",
                    "functions",
                    "classes",
                    "exports",
                ),
                required_paths=("description",),
            )
            expected_result = dict(cleaned)
            expected_result["chunk_id"] = expected_chunk.chunk_id
            expected_result["unit_id"] = expected_chunk.unit_id
        elif expected_type in ("unit-consolidation", "general"):
            cleaned = process_fixed_capsule_response(
                node.result_json,
                label="split-reduction",
                agent="reduction",
                file_path=node.rel_path,
                clean_reporter=clean_reduction_capsule_report,
                requested_paths=("narrative",),
                required_paths=("narrative",),
            )
            expected_result = cleaned
        else:
            if resolved_shape is None:
                return False
            model_fields = {
                key: stored[key]
                for key in (
                    "description",
                    "role_in_system",
                    "functions",
                    "classes",
                    "exports",
                    "dependencies_analysis",
                    "key_concepts",
                    "usage_example",
                )
                if key in stored
            }
            cleaned = process_response(
                canonical_json(model_fields),
                mode="single",
                agent="combined",
                file_path=node.rel_path,
                clean_reporter=clean_combined_report,
                resolved_shape=resolved_shape,
            )
            expected_result = flat_combined_result(
                node.rel_path,
                language,
                list(imports),
                cleaned,
            )
    except Exception:
        return False
    # A live checkpoint stores the post-cleaner object. Requiring byte-canonical
    # equivalence rejects unknown fields, over-limit values, and empty values
    # that a live response would have removed.
    return canonical_json(expected_result) == node.result_json


def validate_node_for_tree(
    node: ReductionNodeState,
    *,
    plan: DivisionPlan,
    tree: ReductionTreePlan,
    content_hash: str,
    expected_identity: str,
    expected_input_digest: str | None = None,
    resolved_shape: object | None = None,
    language: str = "",
    imports: Sequence[str] = (),
) -> bool:
    """Whether *node* is an exact, current, dependency-valid checkpoint.

    Rejects a foreign/malformed node, a node from another content revision, a
    node with incorrect child order or coverage, a node produced under a
    different provider/model/effective-endpoint execution identity or a
    stale final-shape digest, and (when *expected_input_digest* is supplied)
    a node whose stage-local input digest no longer matches -- the
    recompute-from-retained-children check for reducers/final (section 11).

    Never compares a per-node ``reduction_tree_digest`` (schema 3, section
    9): the container-level digest is writer-time provenance only, so a
    topology-only change never invalidates a compatible leaf through this
    function.  Coverage and child order are compared by ordered-tuple
    equality, never a sorted set, so a coverage permutation is rejected.
    """
    if (
        node.rel_path != plan.rel_path
        or node.content_hash != content_hash
        or node.division_plan_digest != plan.plan_digest
        or node.execution_identity_digest != expected_identity
    ):
        return False
    if expected_input_digest is not None and node.input_digest != expected_input_digest:
        return False
    shapes = _expected_node_shapes(plan, tree)
    expected = shapes.get(node.node_id)
    if expected is None:
        return False
    expected_type, expected_children, expected_leaves, expected_unit = expected
    if (
        node.node_type != expected_type
        or node.child_ids != expected_children
        or node.coverage_leaf_ids != expected_leaves
        or node.unit_id != expected_unit
    ):
        return False
    return _node_result_matches_live_schema(
        node,
        expected_type,
        expected_chunk=(
            next(
                (
                    chunk
                    for chunk in plan.chunks
                    if chunk.chunk_id == node.node_id
                ),
                None,
            )
            if expected_type == "leaf"
            else None
        ),
        resolved_shape=resolved_shape,
        language=language,
        imports=imports,
    )


def dependency_closed_nodes(
    nodes: Sequence[ReductionNodeState],
    *,
    plan: DivisionPlan,
    tree: ReductionTreePlan,
) -> tuple[ReductionNodeState, ...]:
    """Retain only a topologically complete recovered dependency subgraph.

    Leaves are independently reusable. An intermediate or final node is
    retained only when every ordered direct child has already been retained;
    pruning one invalid/missing descendant therefore deterministically prunes
    every affected ancestor and schedules those calls again.
    """
    candidates = {node.node_id: node for node in nodes}
    retained: list[ReductionNodeState] = []
    retained_ids: set[str] = set()

    for chunk in plan.chunks:
        candidate = candidates.get(chunk.chunk_id)
        if candidate is not None:
            retained.append(candidate)
            retained_ids.add(candidate.node_id)

    for planned_node in tree.all_nodes:
        candidate = candidates.get(planned_node.node_id)
        if candidate is None:
            continue
        if all(child_id in retained_ids for child_id in planned_node.child_ids):
            retained.append(candidate)
            retained_ids.add(candidate.node_id)

    return tuple(retained)


def final_node_covers_every_leaf(
    plan: DivisionPlan, node: ReductionNodeState
) -> bool:
    """Ordered-tuple equality against the planned chunk-ID sequence (section
    11): a coverage permutation must not validate."""
    return node.coverage_leaf_ids == tuple(chunk.chunk_id for chunk in plan.chunks)


def _quarantine_reason_for(
    node: ReductionNodeState,
    *,
    plan: DivisionPlan,
    tree: ReductionTreePlan,
    content_hash: str,
    expected_identity: str,
    expected_input_digest: str,
) -> str:
    """Classify why *node* failed :func:`validate_node_for_tree`, as one of
    the closed quarantine reason codes (section 9/14).

    Only called for a node reached through a known planned ID (a leaf chunk
    ID or a tree node ID), so the returned reason is always attributable to a
    node the wire format's quarantine-entry ``node_id`` constraint allows;
    a genuinely foreign node ID is never classified or quarantined by this
    function -- section 9's "unknown-node" and "stale-revision" reasons are
    reserved for round-trip preservation of an entry written by a different
    mechanism and are never produced here.  Mirrors
    ``validate_node_for_tree``'s own check order, so the reported reason is
    the first check that would actually fail.
    """
    if (
        node.rel_path != plan.rel_path
        or node.content_hash != content_hash
        or node.division_plan_digest != plan.plan_digest
        or node.execution_identity_digest != expected_identity
    ):
        return "stale-identity"
    if node.input_digest != expected_input_digest:
        return "input-digest-mismatch"
    expected_type, expected_children, expected_leaves, expected_unit = (
        _expected_node_shapes(plan, tree)[node.node_id]
    )
    if node.node_type != expected_type or node.unit_id != expected_unit:
        return "stage-mismatch"
    if node.child_ids != expected_children:
        return "child-order-mismatch"
    if node.coverage_leaf_ids != expected_leaves:
        return "coverage-mismatch"
    return "live-schema-mismatch"


def validate_recovered_tree(
    nodes: Sequence[ReductionNodeState],
    *,
    plan: DivisionPlan,
    tree: ReductionTreePlan,
    content_hash: str,
    provider_identity: str,
    prompt_profile_digest: str,
    imports_digest: str,
    imports: Sequence[str] = (),
    language: str = "",
    resolved_shape: object | None = None,
    max_content_chars: int | None = None,
    existing_quarantine: Sequence[QuarantineEntry] = (),
) -> tuple[tuple[ReductionNodeState, ...], tuple[QuarantineEntry, ...]]:
    """Full section-11 topological validation of one file's recovered nodes.

    A node's own execution identity never binds its children's actual result
    content (only structure/provenance), so an individually-valid, dependency-
    closed reducer or final node can still be stale if a sibling beneath it
    was replaced. This walks the plan in strict topological order -- leaves,
    then each reducer only once every ordered child is already retained, then
    final only once its complete dependency graph is retained -- and for each
    reducer/final recomputes the expected stage-local input digest from the
    *exact retained child results* (mirroring the live executor's own
    narrative extraction and final-manifest assembly) rather than trusting
    the checkpoint's own claim. A node whose recomputed input digest no
    longer matches is rejected, which prunes every ancestor in turn.

    Returns ``(retained_nodes, quarantine_entries)``. A node reached through
    a known planned ID that fails validation is quarantined (bounded,
    non-executable, canonical-plan-ordered) rather than silently discarded
    (section 14). Exceeding ``MAX_QUARANTINE_ENTRIES_PER_FILE`` raises
    :class:`SplitRecoveryStateError`: the original recovery remains untouched
    and planning makes zero provider calls.
    """
    raw_by_id = {node.node_id: node for node in nodes}
    canonical_order = tuple(chunk.chunk_id for chunk in plan.chunks) + tuple(
        node.node_id for node in tree.all_nodes
    )
    planned_ids = frozenset(canonical_order)
    unplanned_nodes = sorted(set(raw_by_id) - planned_ids)
    if unplanned_nodes:
        raise SplitRecoveryStateError(
            "schema-3 recovery contains an unplanned split node ID."
        )
    existing_by_id = {entry.node_id: entry for entry in existing_quarantine}
    if set(existing_by_id) - planned_ids:
        raise SplitRecoveryStateError(
            "schema-3 recovery quarantine contains an unplanned split node ID."
        )
    retained: dict[str, ReductionNodeState] = {}
    retained_results: dict[str, dict] = {}
    quarantine_by_id: dict[str, QuarantineEntry] = dict(existing_by_id)

    def _quarantine(node: ReductionNodeState, reason: str) -> None:
        quarantine_by_id[node.node_id] = QuarantineEntry(
            node_id=node.node_id,
            reason=reason,
            raw_json=canonical_json(node_state_public_json(node)),
        )
        if len(quarantine_by_id) > MAX_QUARANTINE_ENTRIES_PER_FILE:
            raise SplitRecoveryStateError(
                "schema-3 recovery quarantine exceeds its bounded entry count."
            )

    def _keep(
        node: ReductionNodeState, expected_identity: str, expected_input_digest: str
    ) -> None:
        if validate_node_for_tree(
            node,
            plan=plan,
            tree=tree,
            content_hash=content_hash,
            expected_identity=expected_identity,
            expected_input_digest=expected_input_digest,
            resolved_shape=resolved_shape,
            language=language,
            imports=imports,
        ):
            retained[node.node_id] = node
            retained_results[node.node_id] = load_canonical_json_object(node.result_json)
            quarantine_by_id.pop(node.node_id, None)
            return
        _quarantine(
            node,
            _quarantine_reason_for(
                node,
                plan=plan,
                tree=tree,
                content_hash=content_hash,
                expected_identity=expected_identity,
                expected_input_digest=expected_input_digest,
            )
        )

    for chunk in plan.chunks:
        node = raw_by_id.get(chunk.chunk_id)
        if node is None:
            continue
        _keep(
            node,
            leaf_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                provider_identity=provider_identity,
                chunk=chunk,
            ),
            leaf_input_digest(
                rel_path=plan.rel_path,
                language=language,
                chunk=chunk,
                unit_indexes=plan.unit_positions(chunk),
                unit_count=len(plan.units),
            ),
        )

    for planned in tree.all_intermediate_nodes:
        if not all(child_id in retained for child_id in planned.child_ids):
            continue
        node = raw_by_id.get(planned.node_id)
        if node is None:
            continue
        raw_narratives = tuple(
            capsule.get("narrative", capsule.get("description", ""))
            if isinstance(capsule, dict)
            else ""
            for capsule in (retained_results[cid] for cid in planned.child_ids)
        )
        _keep(
            node,
            reduction_execution_identity(
                rel_path=plan.rel_path,
                content_hash=content_hash,
                division_plan_digest=plan.plan_digest,
                reduction_tree_digest=tree.tree_digest,
                provider_identity=provider_identity,
                node=planned,
            ),
            reduction_input_digest(
                rel_path=plan.rel_path,
                phase=planned.phase,
                level=planned.level,
                unit_id=planned.unit_id,
                child_count=len(planned.child_ids),
                ordered_child_narratives=refine_narrative_inputs(raw_narratives),
            ),
        )

    final_node = tree.final_node
    all_leaves_retained = all(chunk.chunk_id in retained for chunk in plan.chunks)
    if all_leaves_retained and all(
        child_id in retained for child_id in final_node.child_ids
    ):
        node = raw_by_id.get(final_node.node_id)
        if node is not None:
            leaf_capsules_ordered = [
                retained_results[chunk.chunk_id] for chunk in plan.chunks
            ]
            ledger = build_fact_ledger(
                leaf_capsules_ordered,
                language=language,
                chunks=plan.chunks,
                symbols=plan.symbols,
            )
            child_results = [retained_results[cid] for cid in final_node.child_ids]
            raw_narratives = tuple(
                child.get("narrative", child.get("description", ""))
                for child in child_results
            )
            manifest_json = final_synthesis_input(
                rel_path=plan.rel_path,
                language=language,
                imports=imports,
                root_narratives=refine_narrative_inputs(raw_narratives),
                root_coverage_leaf_ids=final_node.leaf_ids,
                ledger=ledger,
                max_chars=max_content_chars,
            )
            _keep(
                node,
                final_execution_identity(
                    rel_path=plan.rel_path,
                    content_hash=content_hash,
                    division_plan_digest=plan.plan_digest,
                    reduction_tree_digest=tree.tree_digest,
                    provider_identity=provider_identity,
                    prompt_profile_digest=prompt_profile_digest,
                    imports_digest=imports_digest,
                    node=final_node,
                ),
                final_input_digest(
                    imports_digest=imports_digest,
                    resolved_shape_digest=prompt_profile_digest,
                    manifest_json=manifest_json,
                ),
            )

    closed = dependency_closed_nodes(tuple(retained.values()), plan=plan, tree=tree)
    closed_ids = {node.node_id for node in closed}
    # A checkpointed ancestor whose dependency was rejected is non-executable
    # too. Preserve its bounded raw state until a valid replacement lands.
    for node_id in canonical_order:
        node = raw_by_id.get(node_id)
        if node is not None and node_id not in closed_ids and node_id not in quarantine_by_id:
            _quarantine(node, "input-digest-mismatch")
    quarantine = tuple(
        quarantine_by_id[node_id]
        for node_id in canonical_order
        if node_id in quarantine_by_id
    )
    return closed, quarantine


def is_legacy_split_partial(value: object) -> bool:
    """Whether *value* structurally matches the predecessor (schema version 1)
    ordered-prefix split partial, which is migration-readable but never
    resumed or executed by the node-keyed recovery path (D11)."""
    if not isinstance(value, dict):
        return False
    schema_version = value.get("schema_version", LEGACY_SPLIT_PARTIAL_SCHEMA_VERSION)
    return schema_version == LEGACY_SPLIT_PARTIAL_SCHEMA_VERSION and (
        "completed_chunks" in value
    )
