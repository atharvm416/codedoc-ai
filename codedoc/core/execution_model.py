"""Immutable execution model: frozen per-file requests and the call manifest.

This module owns the frozen, provider-neutral types that carry already-resolved
planning decisions into execution:

- :class:`AgentCallContext` / :class:`FileExecutionRequest` — one immutable,
  self-contained request per provider-bound file. ``codedoc.core.planning``
  constructs these on the provider-free path (see
  :func:`codedoc.core.db.read_source_snapshot`); workers only read them.
- :class:`UnitChunkExecutionRequest` — one immutable bounded leaf-chunk
  request for split large-file documentation.
- :class:`FileReductionExecutionRequest` — one immutable unit-consolidation or
  general reduction node request.
- :class:`PlannedCall` / :class:`CallManifest` — the canonical, ordered set of
  initially planned logical calls shared by dry-run and real execution.

No provider, agent, or worker constructs these types itself; that would be a
second planner. Retries and corrections are actual usage, not initially
planned calls, and never consume or add a manifest entry.
"""

from __future__ import annotations

import hashlib
import re
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

from codedoc.core.file_division import (
    DivisionPlan,
    MAX_LEAF_PROMPT_METADATA_CHARS,
    ReductionTreePlan,
    SemanticUnitIdentity,
    SplitTreeState,
    render_leaf_prompt_metadata,
)
from codedoc.parser.source_structure import SourceRange
from codedoc.core.prompt_profiles import ResolvedFileShapeBundle, ReviewBatch

_ID_SEP = "\x1f"
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class AgentCallContext:
    """Immutable per-call context.

    ``analysis_mode``, ``max_content_chars``, and ``truncation_head_ratio`` are
    identical across every context built within one run; only
    ``resolved_shape_bundle`` varies per file (extension scope).
    """

    analysis_mode: str
    max_content_chars: int
    truncation_head_ratio: float
    resolved_shape_bundle: ResolvedFileShapeBundle

    def __post_init__(self) -> None:
        if self.analysis_mode not in DOC_AGENTS_BY_MODE:
            raise ValueError(f"unsupported analysis mode: {self.analysis_mode!r}")
        if self.max_content_chars <= 0:
            raise ValueError("max_content_chars must be greater than zero.")
        if not 0 < self.truncation_head_ratio < 1:
            raise ValueError("truncation_head_ratio must be between zero and one.")
        bundle = self.resolved_shape_bundle
        if bundle.mode != self.analysis_mode:
            raise ValueError("resolved shape bundle mode does not match analysis mode.")
        immutable_bundle = ResolvedFileShapeBundle(
            mode=bundle.mode,
            scope=bundle.scope,
            selections=MappingProxyType(dict(bundle.selections)),
            digest=bundle.digest,
        )
        object.__setattr__(self, "resolved_shape_bundle", immutable_bundle)


@dataclass(frozen=True, slots=True)
class FileExecutionRequest:
    """One frozen, self-contained request for a provider-bound file.

    ``content_hash`` is the SHA-256 of the raw file bytes; ``content`` is those
    same bytes decoded with the canonical ``utf-8-sig`` / ``errors="replace"``
    policy — see :func:`codedoc.core.db.read_source_snapshot`, which produces
    both from one read. ``imports`` is derived from that identical decoded
    snapshot via :func:`codedoc.parser.factory.parse_source`.

    Every field here is data execution actually consumes. The request carries no
    source path: planning has already read, decoded, hashed, and parsed the file
    by the time it builds one, so a worker consumes this object only — it cannot
    reopen the source file, and remains valid after that file is edited or
    deleted.
    """

    rel_path: str
    language: str
    imports: tuple[str, ...]
    content: str
    content_hash: str
    context: AgentCallContext

    def __post_init__(self) -> None:
        normalized = _normalize_rel_path(self.rel_path)
        object.__setattr__(self, "rel_path", normalized)
        object.__setattr__(self, "imports", tuple(self.imports))


@dataclass(frozen=True, slots=True)
class UnitChunkExecutionRequest:
    """One immutable bounded leaf-chunk request for split large-file documentation.

    Carries complete fragment metadata (D4/section 7) so the fixed fragment prompt
    builder never needs to recompute topology or reopen source. There is no
    separately parsed whole-file ``imports`` field: an import statement
    physically present in ``payload`` is ordinary visible source, but
    parser-derived file imports reach only final synthesis (D5/section 7).
    """

    rel_path: str
    language: str
    full_content_hash: str
    division_plan_digest: str
    chunk_id: str
    unit_id: str
    semantic_units: tuple[SemanticUnitIdentity, ...]
    unit_indexes: tuple[int, ...]
    unit_count: int
    unit_chunk_index: int
    unit_chunk_count: int
    global_index: int
    global_count: int
    owning_ranges: tuple[SourceRange, ...]
    continuation_before: bool
    continuation_after: bool
    known_symbols: tuple[str, ...]
    payload: str
    context: AgentCallContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "rel_path", _normalize_rel_path(self.rel_path))
        if not self.language or not isinstance(self.language, str):
            raise ValueError("chunk requests require a language tag.")
        if not _HEX_64_RE.fullmatch(self.full_content_hash):
            raise ValueError("chunk request content hash must be a SHA-256 digest.")
        if (
            not self.division_plan_digest.startswith("division-plan:")
            or not _HEX_64_RE.fullmatch(
                self.division_plan_digest.removeprefix("division-plan:")
            )
        ):
            raise ValueError("chunk requests require a division-plan digest.")
        if not self.unit_id.startswith("unit_") or not _HEX_64_RE.fullmatch(
            self.unit_id.removeprefix("unit_")
        ):
            raise ValueError("chunk requests require a unit ID.")
        if not self.chunk_id.startswith("chunk_") or not _HEX_64_RE.fullmatch(
            self.chunk_id.removeprefix("chunk_")
        ):
            raise ValueError("chunk requests require a chunk ID.")
        semantic_units = tuple(self.semantic_units)
        if not semantic_units or any(
            not isinstance(unit, SemanticUnitIdentity)
            for unit in semantic_units
        ):
            raise ValueError("chunk requests require ordered semantic units.")
        if len({unit.unit_id for unit in semantic_units}) != len(semantic_units):
            raise ValueError("chunk request semantic units must be distinct.")
        unit_indexes = tuple(self.unit_indexes)
        if len(unit_indexes) != len(semantic_units):
            raise ValueError("chunk request unit indexes must align with semantic units.")
        for name in ("unit_count", "unit_chunk_count", "global_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be one or greater.")
        for name in ("unit_chunk_index", "global_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be zero or greater.")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.unit_count
            for index in unit_indexes
        ):
            raise ValueError("chunk request unit indexes must be within unit_count.")
        if tuple(sorted(set(unit_indexes))) != unit_indexes:
            raise ValueError("chunk request unit indexes must be ordered and distinct.")
        if self.unit_chunk_index >= self.unit_chunk_count:
            raise ValueError("unit_chunk_index must be within unit_chunk_count.")
        if self.global_index >= self.global_count:
            raise ValueError("global_index must be within global_count.")
        if self.unit_chunk_count > 1 and len(semantic_units) != 1:
            raise ValueError("continuation chunks must own exactly one semantic unit.")
        if len(semantic_units) == 1 and self.unit_id != semantic_units[0].unit_id:
            raise ValueError("single-unit chunk request has a mismatched unit ID.")
        owning_ranges = tuple(self.owning_ranges)
        if not owning_ranges or any(
            not isinstance(source_range, SourceRange)
            for source_range in owning_ranges
        ):
            raise ValueError("chunk requests require ordered owning source ranges.")
        if any(
            current.end_byte > following.start_byte
            for current, following in zip(owning_ranges, owning_ranges[1:])
        ):
            raise ValueError("chunk request owning ranges must be ordered and non-overlapping.")
        known_symbols = tuple(self.known_symbols)
        if any(not isinstance(item, str) for item in known_symbols):
            raise ValueError("known_symbols must contain text.")
        object.__setattr__(self, "semantic_units", semantic_units)
        object.__setattr__(self, "unit_indexes", unit_indexes)
        object.__setattr__(self, "owning_ranges", owning_ranges)
        object.__setattr__(self, "known_symbols", known_symbols)
        if not isinstance(self.payload, str) or not self.payload:
            raise ValueError("chunk requests require a non-empty payload.")
        if len(self.payload) > self.context.max_content_chars:
            raise ValueError("chunk payload exceeds max_content_chars.")
        if len(self.payload.encode("utf-8")) != sum(
            source_range.end_byte - source_range.start_byte
            for source_range in owning_ranges
        ):
            raise ValueError("chunk payload bytes must match its owning source ranges.")
        metadata = render_leaf_prompt_metadata(
            group_unit_id=self.unit_id,
            semantic_units=semantic_units,
            unit_indexes=unit_indexes,
            unit_count=self.unit_count,
            owning_ranges=owning_ranges,
        )
        if len(metadata) > MAX_LEAF_PROMPT_METADATA_CHARS:
            raise ValueError("chunk request metadata exceeds its fixed bound.")


@dataclass(frozen=True, slots=True)
class FileReductionExecutionRequest:
    """One immutable unit-consolidation or general file-reduction node request.

    Carries the ordered child result capsules directly (D6/section 9): execution
    never re-reads source or recomputes reduction-tree topology to build a
    reducer prompt.
    """

    rel_path: str
    division_plan_digest: str
    reduction_tree_digest: str
    node_id: str
    phase: Literal["unit-consolidation", "general"]
    unit_id: str | None
    level: int
    ordinal: int
    child_ids: tuple[str, ...]
    child_capsules: tuple[Mapping[str, object], ...]
    reducer_revision: str
    context: AgentCallContext

    def __post_init__(self) -> None:
        object.__setattr__(self, "rel_path", _normalize_rel_path(self.rel_path))
        if self.phase not in ("unit-consolidation", "general"):
            raise ValueError("reduction request phase must be unit-consolidation or general.")
        if not self.node_id.startswith("node_") or not _HEX_64_RE.fullmatch(
            self.node_id.removeprefix("node_")
        ):
            raise ValueError("reduction requests require a node ID.")
        child_ids = tuple(self.child_ids)
        child_capsules = tuple(self.child_capsules)
        if len(child_ids) < 2:
            raise ValueError("a reduction node needs at least two children.")
        if len(child_ids) != len(child_capsules):
            raise ValueError("reduction request child IDs and capsules must align.")
        object.__setattr__(self, "child_ids", child_ids)
        object.__setattr__(self, "child_capsules", child_capsules)


PlannedCallCategory = Literal[
    "prompt-review",
    "file-documentation",
    "unit-documentation",
    "file-reduction",
    "file-synthesis",
]


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One initially planned logical call.

    ``call_id`` identifies an initially planned logical call, not retries or
    corrections. ``owner`` groups calls that share a consecutive ordinal
    sequence — a file's relative path for documentation calls, a chunk ID for
    a leaf call, a node ID for a reduction call, or the fixed review-owner
    sentinel for prompt-review calls.
    """

    call_id: str
    category: PlannedCallCategory
    owner: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class CallManifest:
    """The canonical, ordered set of initially planned calls for one run."""

    calls: tuple[PlannedCall, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls", tuple(self.calls))


def _normalize_rel_path(rel_path: str) -> str:
    """Return one normalized project-relative POSIX path.

    Backslashes are accepted at construction boundaries for Windows callers,
    but absolute paths, empty paths, and parent traversal are rejected.  Every
    stored request and every documentation call ID therefore uses the same
    canonical path spelling.
    """
    raw = str(rel_path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"rel_path must be a project-relative path: {rel_path!r}")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"rel_path must name a file: {rel_path!r}")
    return normalized


# ---------------------------------------------------------------------------
# Domain-separated call identifiers
# ---------------------------------------------------------------------------


def _domain_call_id(category: str, *parts: str) -> str:
    """Domain-separated SHA-256 identifying one initially planned logical call.

    Fields are joined with an ASCII unit-separator so that, e.g.,
    ``("ab", "c")`` and ``("a", "bc")`` never collide.
    """
    payload = _ID_SEP.join((category, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def documentation_call_id(
    rel_path: str, analysis_mode: str, agent: str, ordinal: int
) -> str:
    """The call id for one initial per-file documentation call."""
    return _domain_call_id(
        "file-documentation",
        _normalize_rel_path(rel_path),
        analysis_mode,
        agent,
        str(ordinal),
    )


def unit_documentation_call_id(
    rel_path: str,
    unit_id: str,
    chunk_id: str,
    unit_chunk_index: int,
    division_plan_digest: str,
    analysis_mode: str,
    agent: str,
    ordinal: int,
) -> str:
    """The call id for one initial bounded leaf-chunk documentation call."""
    return _domain_call_id(
        "unit-documentation",
        _normalize_rel_path(rel_path),
        unit_id,
        chunk_id,
        str(unit_chunk_index),
        division_plan_digest,
        analysis_mode,
        agent,
        str(ordinal),
    )


def file_reduction_call_id(
    rel_path: str,
    node_id: str,
    reduction_tree_digest: str,
    reducer_revision: str,
    ordinal: int,
) -> str:
    """The call id for one unit-consolidation or general reduction node call."""
    return _domain_call_id(
        "file-reduction",
        _normalize_rel_path(rel_path),
        node_id,
        reduction_tree_digest,
        reducer_revision,
        str(ordinal),
    )


def file_synthesis_call_id(
    rel_path: str,
    division_plan_digest: str,
    prompt_revision: str,
    ordinal: int,
) -> str:
    """The call id for one final divided-file synthesis call."""
    return _domain_call_id(
        "file-synthesis",
        _normalize_rel_path(rel_path),
        division_plan_digest,
        prompt_revision,
        str(ordinal),
    )


def review_call_id(
    stream_digest: str, component_ids: Sequence[str], ordinal: int
) -> str:
    """The call id for one prompt-customization review batch."""
    return _domain_call_id("prompt-review", stream_digest, *component_ids, str(ordinal))


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

# The fixed per-owner sentinel for every prompt-review call. Review batches
# have no natural per-file owner; they share one consecutive ordinal sequence
# for the whole run.
REVIEW_OWNER = "prompt-review"

# The canonical, fixed per-mode agent order, identical to
# ``codedoc.core.prompt_profiles.VALID_AGENTS_BY_MODE`` — documentation-call
# agent naming and ordinal order never drift from prompt-profile component
# naming because both are defined from the same registry shape.
DOC_AGENTS_BY_MODE: dict[str, tuple[str, ...]] = {
    "single": ("combined",),
    "triple": ("structure", "dependency", "documentation"),
}

# Split is valid only in single-mode (D2); every split call therefore uses
# exactly this one-agent sequence regardless of the run's own analysis_mode
# value (which loader/planning already guarantee is "single" whenever any
# division plan exists).
_SPLIT_AGENTS: tuple[str, ...] = DOC_AGENTS_BY_MODE["single"]


def _synthesis_prompt_revision() -> str:
    from codedoc.core.file_division import FINAL_SYNTHESIS_REVISION

    return FINAL_SYNTHESIS_REVISION


def _reducer_prompt_revision() -> str:
    from codedoc.core.file_division import REDUCER_PROMPT_REVISION

    return REDUCER_PROMPT_REVISION


def build_call_manifest(
    review_batches: Sequence[ReviewBatch],
    agent_rels: Sequence[str],
    analysis_mode: str,
    division_plans: Mapping[str, DivisionPlan] | None = None,
    reduction_trees: Mapping[str, ReductionTreePlan] | None = None,
    tree_states: Mapping[str, SplitTreeState] | None = None,
) -> CallManifest:
    """Build and validate the one canonical call manifest for a run.

    Review calls come first, in canonical review-batch order; documentation
    calls follow in global phase passes: all ordinary-file/leaf calls in
    canonical path/source order, then every unit-consolidation node, every
    general-reduction node, and every final synthesis. Within each reduction
    phase paths and the tree's canonical level/source order are stable. Each
    restored (already-completed and validated) node is excluded. Every
    initially planned logical call maps to exactly one manifest entry; retries
    and corrections are excluded and never consume or add a manifest slot.
    """
    agents = DOC_AGENTS_BY_MODE[analysis_mode]
    divided = dict(division_plans or {})
    trees = dict(reduction_trees or {})
    states = dict(tree_states or {})
    agent_rel_set = set(agent_rels)
    if set(divided) - agent_rel_set:
        raise ValueError("division-plan paths must be a subset of agent_rels.")
    if set(trees) != set(divided):
        raise ValueError("every division plan requires exactly one reduction tree.")
    if set(states) - set(divided):
        raise ValueError("tree-state paths must be a subset of effective split plans.")

    calls: list[PlannedCall] = []

    for batch in review_batches:
        component_ids = tuple(component.component for component in batch.components)
        calls.append(
            PlannedCall(
                call_id=review_call_id(batch.stream_digest, component_ids, batch.ordinal),
                category="prompt-review",
                owner=REVIEW_OWNER,
                ordinal=batch.ordinal,
            )
        )

    reducer_revision = _reducer_prompt_revision()
    synthesis_revision = _synthesis_prompt_revision()

    completed_by_path = {
        rel_path: (
            frozenset(states[rel_path].by_id())
            if rel_path in states
            else frozenset()
        )
        for rel_path in divided
    }

    # Phase 1: ordinary file documentation and split leaves.
    for rel_path in sorted(agent_rels):
        plan = divided.get(rel_path)
        if plan is not None:
            if plan.rel_path != rel_path:
                raise ValueError("division-plan key does not match its relative path.")
            completed_ids = completed_by_path[rel_path]

            for chunk in plan.chunks:
                if chunk.chunk_id in completed_ids:
                    continue
                calls.append(
                    PlannedCall(
                        call_id=unit_documentation_call_id(
                            rel_path,
                            chunk.unit_id,
                            chunk.chunk_id,
                            chunk.unit_chunk_index,
                            plan.plan_digest,
                            "single",
                            _SPLIT_AGENTS[0],
                            1,
                        ),
                        category="unit-documentation",
                        owner=chunk.chunk_id,
                        ordinal=1,
                    )
                )
            continue

        for ordinal, agent in enumerate(agents, start=1):
            calls.append(
                PlannedCall(
                    call_id=documentation_call_id(rel_path, analysis_mode, agent, ordinal),
                    category="file-documentation",
                    owner=rel_path,
                    ordinal=ordinal,
                )
            )

    # Phase 2: unit-continuation consolidation across every file.
    for rel_path in sorted(divided):
        tree = trees[rel_path]
        completed_ids = completed_by_path[rel_path]
        for node in tree.unit_consolidation_nodes:
            if node.node_id in completed_ids:
                continue
            calls.append(
                PlannedCall(
                    call_id=file_reduction_call_id(
                        rel_path, node.node_id, tree.tree_digest, reducer_revision, 1
                    ),
                    category="file-reduction",
                    owner=node.node_id,
                    ordinal=1,
                )
            )

    # Phase 3: general reductions across every file.
    for rel_path in sorted(divided):
        tree = trees[rel_path]
        completed_ids = completed_by_path[rel_path]
        for node in tree.general_nodes:
            if node.node_id in completed_ids:
                continue
            calls.append(
                PlannedCall(
                    call_id=file_reduction_call_id(
                        rel_path, node.node_id, tree.tree_digest, reducer_revision, 1
                    ),
                    category="file-reduction",
                    owner=node.node_id,
                    ordinal=1,
                )
            )

    # Phase 4: final synthesis across every file.
    for rel_path in sorted(divided):
        plan = divided[rel_path]
        tree = trees[rel_path]
        completed_ids = completed_by_path[rel_path]
        if tree.final_node.node_id not in completed_ids:
            calls.append(
                PlannedCall(
                    call_id=file_synthesis_call_id(
                        rel_path, plan.plan_digest, synthesis_revision, 1
                    ),
                    category="file-synthesis",
                    owner=rel_path,
                    ordinal=1,
                )
            )

    ordinary_rels = [rel for rel in agent_rels if rel not in divided]
    expected_ordinary = len(ordinary_rels) * len(agents)
    expected_split = 0
    for rel_path, plan in divided.items():
        if rel_path not in agent_rel_set:
            continue
        tree = trees[rel_path]
        state = states.get(rel_path)
        completed_ids = frozenset(state.by_id()) if state is not None else frozenset()
        expected_split += sum(1 for c in plan.chunks if c.chunk_id not in completed_ids)
        expected_split += sum(
            1
            for node in tree.unit_consolidation_nodes + tree.general_nodes
            if node.node_id not in completed_ids
        )
        if tree.final_node.node_id not in completed_ids:
            expected_split += 1
    expected_total = len(review_batches) + expected_ordinary + expected_split
    _validate_manifest(calls, expected_total=expected_total)

    digest = hashlib.sha256(
        "\n".join(call.call_id for call in calls).encode("utf-8")
    ).hexdigest()
    return CallManifest(calls=tuple(calls), digest=digest)


def _validate_manifest(calls: list[PlannedCall], *, expected_total: int) -> None:
    """Internal defect check: unique ids, known categories, consecutive
    per-owner ordinals starting at 1, and the expected total length."""
    if len(calls) != expected_total:
        raise ValueError(
            f"call manifest has {len(calls)} entries; expected {expected_total}."
        )

    seen_ids: set[str] = set()
    per_owner_ordinals: dict[tuple[str, str], list[int]] = {}
    for call in calls:
        if call.call_id in seen_ids:
            raise ValueError(f"duplicate planned call id: {call.call_id}")
        seen_ids.add(call.call_id)
        if call.category not in (
            "prompt-review",
            "file-documentation",
            "unit-documentation",
            "file-reduction",
            "file-synthesis",
        ):
            raise ValueError(f"unknown planned-call category: {call.category!r}")
        per_owner_ordinals.setdefault((call.category, call.owner), []).append(
            call.ordinal
        )

    for (category, owner), ordinals in per_owner_ordinals.items():
        expected = list(range(1, len(ordinals) + 1))
        if ordinals != expected:
            raise ValueError(
                f"planned calls for {category} owner {owner!r} are not consecutive ordinals "
                f"starting at 1: {ordinals}"
            )


class CallManifestTracker:
    """Thread-safe consumption state for one immutable call manifest.

    Initial provider attempts consume a planned logical ID exactly once.
    Retries and corrections must attach to an ID that has already been
    consumed; they never consume another manifest entry.  The tracker stores no
    prompts, source, responses, credentials, or token/accounting data.
    """

    def __init__(self, manifest: CallManifest) -> None:
        self._calls = {call.call_id: call for call in manifest.calls}
        if len(self._calls) != len(manifest.calls):
            raise ValueError("call manifest contains duplicate call IDs.")
        self._attempted: set[str] = set()
        self._stop_event: threading.Event | None = None
        self._lock = threading.Lock()

    def bind_stop_event(self, stop_event: threading.Event | None) -> None:
        with self._lock:
            self._stop_event = stop_event

    def raise_if_cancelled(self) -> None:
        with self._lock:
            stop_event = self._stop_event
        if stop_event is not None and stop_event.is_set():
            raise CancelledError()

    def signal_stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()

    def authorize(self, call: PlannedCall, *, additional_attempt: bool) -> None:
        """Authorize one provider attempt for *call*.

        ``additional_attempt=False`` consumes the planned logical call.
        ``True`` attaches a retry/correction to an already-consumed call.
        Unknown IDs, metadata mismatches, double initial consumption, and an
        additional attempt without an originating initial attempt are internal
        defects and fail before the provider is called.
        """
        with self._lock:
            expected = self._calls.get(call.call_id)
            if expected is None:
                raise RuntimeError(
                    f"provider attempt referenced unknown planned call ID {call.call_id}."
                )
            if expected != call:
                raise RuntimeError(
                    "provider attempt metadata does not match its planned call ID."
                )
            already_attempted = call.call_id in self._attempted
            if additional_attempt:
                if not already_attempted:
                    raise RuntimeError(
                        "additional provider attempt has no attempted logical call."
                    )
                return
            if already_attempted:
                raise RuntimeError(
                    f"planned call ID {call.call_id} was consumed more than once."
                )
            self._attempted.add(call.call_id)

    def owner_was_attempted(self, category: PlannedCallCategory, owner: str) -> bool:
        """Whether any planned logical call for ``(category, owner)`` ran."""
        with self._lock:
            return any(
                call_id in self._attempted
                and call.category == category
                and call.owner == owner
                for call_id, call in self._calls.items()
            )

    def call_was_attempted(self, call_id: str) -> bool:
        """Whether the exact initially planned logical call has been attempted."""
        with self._lock:
            if call_id not in self._calls:
                raise RuntimeError(
                    f"attempt lookup referenced unknown planned call ID {call_id}."
                )
            return call_id in self._attempted

    def snapshot(self) -> dict[str, int]:
        """Return provider-free planned/attempted reconciliation counts."""
        with self._lock:
            attempted = len(self._attempted)
            total = len(self._calls)
        return {
            "attempted_logical_calls": attempted,
            "planned_calls_not_attempted": total - attempted,
        }
