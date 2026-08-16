"""Shared pipeline planning for codedoc.

The planning helper computes every routing decision — selection, forcing,
propagation, unchanged skipping, identical-content reuse, and the paid-file cap
— into one immutable :class:`PipelinePlan` that both ``--dry-run`` and real
execution consume.  It may read source contents and hashes, but it never writes,
never creates a provider, and never initializes ``SafeWriter``.

For a split-configured run (single mode only — see D2), this module also
derives the complete provider-free division plan and reduction tree for every
oversized selected file, collects every capacity-blocked ``(rel_path, reason)``
pair under the frozen evaluation order, and lets a genuine
``DivisionInternalDefect`` propagate immediately and uncaught — it is never a
per-file failure, only a whole-run provider-free abort (D8).

Format detection and ownership inspection live in ``codedoc.core.output``;
this module only consumes their results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from codedoc.core.db import compute_file_hash, read_source_snapshot
from codedoc.core.file_division import (
    FINAL_SYNTHESIS_REVISION,
    REDUCER_PROMPT_REVISION,
    BlockedReason,
    DivisionPlan,
    ReductionTreePlan,
    SplitCapacityBlocked,
    SplitRecoveryStateError,
    SplitTreeState,
    build_division_plan,
    build_reduction_tree,
    deterministic_imports_digest,
    provider_execution_identity,
    validate_recovered_tree,
)
from codedoc.core.execution_model import (
    DOC_AGENTS_BY_MODE,
    REVIEW_OWNER,
    AgentCallContext,
    CallManifest,
    FileExecutionRequest,
    documentation_call_id,
    file_reduction_call_id,
    file_synthesis_call_id,
    unit_documentation_call_id,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    resolved_synthesis_shape,
)
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    expected_analysis_identity,
    expected_large_file_identity,
    expected_max_context_revision,
    expected_ordinary_path_identity,
    normalized_identity_value,
)
from codedoc.core.release_policy import current_split_release_policy
from codedoc.core.source_precheck import insufficient_source
from codedoc.parser.factory import parse_source
from codedoc.utils.errors import ConfigError, ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def _identity_matches(stored: dict, expected: dict) -> bool:
    """Compare every key in :data:`CACHE_IDENTITY_KEYS`.

    Both the stored and the expected side are normalized through the shared
    absent-default mapping (:func:`normalized_identity_value`), so an absent key
    and an explicitly stored default (e.g. ``_prompt_profile_digest`` ==
    ``NO_PROMPT_PROFILE_DIGEST``) compare equal; a present-but-mismatched key
    blocks reuse.
    """
    if not isinstance(stored, dict):
        return False
    for key in CACHE_IDENTITY_KEYS:
        if normalized_identity_value(key, stored) != normalized_identity_value(key, expected):
            return False
    return True


def _record_is_reusable(
    stored: dict | None,
    content_hash: str,
    expected: dict,
    expected_language: str,
    *,
    rel_path: str,
) -> bool:
    """The single centralized reuse predicate.

    A stored record may be reused only when its content hash matches, every
    cache-identity key and its stored language match the current effective
    language and expected revision/mode, and the record's own stored ``path``
    and stamped ``_ordinary_path_identity`` both match *rel_path* exactly.
    Language is a compatibility guard, not persisted identity material, so
    this check does not change ordinary cache-identity bytes.

    The path and ``_ordinary_path_identity`` checks are explicit here in
    addition to ``_ordinary_path_identity`` being a member of
    ``CACHE_IDENTITY_KEYS`` (compared generically below by
    ``_identity_matches``): refusing ordinary cross-path reuse is a closed
    security boundary (a stored record must never document a different path
    than the one it would be reused into), so its enforcement does not rely
    solely on set membership that an unrelated future change could alter.  A
    record whose stored ``_ordinary_path_identity`` is absent (every
    pre-0.14.4 ordinary/truncate-path record) normalizes to ``None`` and
    compares unequal to the expected non-``None`` value, so it is refused
    until regenerated; a split (non-ordinary) record and expectation both
    normalize to ``None`` here and rely on ``_large_file_identity`` instead.
    """
    if not isinstance(stored, dict):
        return False
    if stored.get("hash", "") != content_hash:
        return False
    if stored.get("language") != expected_language:
        return False
    if stored.get("path") != rel_path:
        return False
    if normalized_identity_value(
        "_ordinary_path_identity", stored
    ) != normalized_identity_value("_ordinary_path_identity", expected):
        return False
    return _identity_matches(stored, expected)


@dataclass(frozen=True)
class PipelinePlan:
    """Immutable description of all routing decisions for one pipeline run."""

    scanned_rels: frozenset[str]
    documented_rels: frozenset[str]
    changed_rels: frozenset[str]
    forced_rels: frozenset[str]
    process_rels: frozenset[str]
    unchanged_rels: frozenset[str]
    identical_reuse_rels: frozenset[str]
    agent_rels: frozenset[str]
    entry_rel: str | None
    max_files: int
    max_files_exceeded: bool
    # Selected files with at least one initially planned provider action that
    # is not already covered by validated recovery. Insufficient-source,
    # capacity-blocked, and fully restored files are excluded.
    unpaid_action_rels: frozenset[str] = frozenset()
    # Effective oversized split files that still require paid execution after
    # completed-record reuse and partial recovery have been evaluated. This is
    # separate from changed_rels so execution policy alone never propagates work
    # to unchanged dependents.
    split_execution_rels: frozenset[str] = frozenset()
    division_plan_rels: frozenset[str] = frozenset()
    # Same-path completed split records accepted by the canonical reuse
    # predicate. Cross-path split reuse remains unavailable because the
    # completed identity is path-bound.
    completed_split_reuse_rels: frozenset[str] = frozenset()
    # rel_path -> first-failing capacity-blocked reason under the frozen
    # evaluation order (D3/D6). A blocked file is never in agent_rels or
    # division_plan_rels and never counts toward max_files (D8).
    division_blocked: Mapping[str, str] = field(default_factory=dict)
    # Attached by with_call_manifest() once the canonical call manifest exists;
    # zero/empty defaults keep every existing direct PipelinePlan(...) caller
    # (e.g. focused planning tests) source-compatible.
    review_calls_planned: int = 0
    file_documentation_calls_planned: int = 0
    unit_documentation_calls_planned: int = 0
    file_reduction_calls_planned: int = 0
    synthesis_calls_planned: int = 0
    documentation_calls_planned: int = 0
    total_calls_planned: int = 0
    max_planned_calls: int = 0
    max_planned_calls_exceeded: bool = False
    call_manifest_digest: str = ""

    @property
    def selected_rels(self) -> frozenset[str]:
        """Read-only compatibility alias for :attr:`documented_rels`.

        ``documented_rels`` is canonical; this non-settable delegating property
        preserves compatibility for existing callers.
        """
        return self.documented_rels

    def with_call_manifest(
        self,
        manifest: CallManifest,
        max_planned_calls: int,
        division_plans: Mapping[str, DivisionPlan] | None = None,
        reduction_trees: Mapping[str, ReductionTreePlan] | None = None,
        tree_states: Mapping[str, SplitTreeState] | None = None,
    ) -> "PipelinePlan":
        """Return a copy of this plan with the canonical call manifest's counts
        and cap attached.
        """
        review_count = sum(1 for c in manifest.calls if c.category == "prompt-review")
        file_documentation_count = sum(
            1 for c in manifest.calls if c.category == "file-documentation"
        )
        unit_documentation_count = sum(
            1 for c in manifest.calls if c.category == "unit-documentation"
        )
        file_reduction_count = sum(1 for c in manifest.calls if c.category == "file-reduction")
        synthesis_count = sum(1 for c in manifest.calls if c.category == "file-synthesis")
        documentation_count = (
            file_documentation_count
            + unit_documentation_count
            + file_reduction_count
            + synthesis_count
        )
        total = len(manifest.calls)
        if review_count + documentation_count != total:
            raise ValueError("call manifest contains a call outside the known categories.")
        review_calls = [c for c in manifest.calls if c.category == "prompt-review"]
        if any(call.owner != REVIEW_OWNER for call in review_calls):
            raise ValueError("prompt-review call has a non-canonical owner.")

        doc_calls = [c for c in manifest.calls if c.category == "file-documentation"]
        doc_owners = {call.owner for call in doc_calls}
        ordinary_rels = set(self.agent_rels) - set(self.division_plan_rels)
        if doc_owners != ordinary_rels:
            raise ValueError(
                "documentation-call owners do not exactly match agent_rels."
            )

        for owner in sorted(ordinary_rels):
            owner_calls = [call for call in doc_calls if call.owner == owner]
            possible_modes: set[str] = set()
            for mode, agents in DOC_AGENTS_BY_MODE.items():
                expected = [
                    (documentation_call_id(owner, mode, agent, ordinal), ordinal)
                    for ordinal, agent in enumerate(agents, start=1)
                ]
                actual = [(call.call_id, call.ordinal) for call in owner_calls]
                if actual == expected:
                    possible_modes.add(mode)
            if not possible_modes:
                raise ValueError(
                    f"documentation calls for {owner!r} do not match a canonical mode."
                )

        plans = dict(division_plans or {})
        trees = dict(reduction_trees or {})
        states = dict(tree_states or {})
        if not set(self.division_plan_rels) <= set(plans):
            raise ValueError("split plan paths are missing division plans.")
        if set(plans) != set(trees):
            raise ValueError("every division plan requires exactly one reduction tree.")
        if set(states) - set(self.division_plan_rels):
            raise ValueError("tree-state paths are not effective split paths.")

        known_leaf_owners: set[str] = set()
        known_node_owners: set[str] = set()
        for rel_path in sorted(self.division_plan_rels):
            division_plan = plans[rel_path]
            reduction_tree = trees[rel_path]
            if division_plan.rel_path != rel_path:
                raise ValueError(
                    f"division plan for {rel_path!r} does not match its relative path."
                )
            file_leaf_owners = {chunk.chunk_id for chunk in division_plan.chunks}
            file_node_owners = {
                node.node_id
                for node in reduction_tree.unit_consolidation_nodes + reduction_tree.general_nodes
            }
            known_leaf_owners.update(file_leaf_owners)
            known_node_owners.update(file_node_owners)
            known_node_owners.add(reduction_tree.final_node.node_id)
            state = states.get(rel_path)
            completed_ids = frozenset(state.by_id()) if state is not None else frozenset()

            expected_unit_calls = {
                unit_documentation_call_id(
                    rel_path,
                    chunk.unit_id,
                    chunk.chunk_id,
                    chunk.unit_chunk_index,
                    division_plan.plan_digest,
                    "single",
                    "combined",
                    1,
                )
                for chunk in division_plan.chunks
                if chunk.chunk_id not in completed_ids
            }
            actual_unit_calls = {
                call.call_id
                for call in manifest.calls
                if call.category == "unit-documentation" and call.owner in file_leaf_owners
            }
            if actual_unit_calls != expected_unit_calls:
                raise ValueError(f"unit-documentation calls for {rel_path!r} are not canonical.")

            expected_reduction_calls = {
                file_reduction_call_id(
                    rel_path, node.node_id, reduction_tree.tree_digest, REDUCER_PROMPT_REVISION, 1
                )
                for node in reduction_tree.unit_consolidation_nodes + reduction_tree.general_nodes
                if node.node_id not in completed_ids
            }
            actual_reduction_calls = {
                call.call_id
                for call in manifest.calls
                if call.category == "file-reduction" and call.owner in file_node_owners
            }
            if actual_reduction_calls != expected_reduction_calls:
                raise ValueError(f"file-reduction calls for {rel_path!r} are not canonical.")

            expected_synthesis = (
                []
                if reduction_tree.final_node.node_id in completed_ids
                else [
                    (
                        file_synthesis_call_id(
                            rel_path, division_plan.plan_digest, FINAL_SYNTHESIS_REVISION, 1
                        ),
                        1,
                    )
                ]
            )
            synthesis_calls = [
                call
                for call in manifest.calls
                if call.category == "file-synthesis" and call.owner == rel_path
            ]
            if [(call.call_id, call.ordinal) for call in synthesis_calls] != expected_synthesis:
                raise ValueError(f"synthesis calls for {rel_path!r} are not canonical.")

        if any(
            call.owner not in known_leaf_owners
            for call in manifest.calls
            if call.category == "unit-documentation"
        ):
            raise ValueError("unit-documentation call has an unknown chunk owner.")
        if any(
            call.owner not in known_node_owners
            for call in manifest.calls
            if call.category == "file-reduction"
        ):
            raise ValueError("file-reduction call has an unknown node owner.")
        if any(
            call.owner not in self.division_plan_rels
            for call in manifest.calls
            if call.category == "file-synthesis"
        ):
            raise ValueError("file-synthesis call has an unknown split-file owner.")

        # Any single mode/triple mode is possible for ordinary files; only the
        # exact per-owner reconciliation above (which already ran) enforces
        # correctness, so this final aggregate check is a defense-in-depth
        # cross total against whichever uniform mode the manifest actually used.
        possible_totals = {
            len(ordinary_rels) * len(agents) for agents in DOC_AGENTS_BY_MODE.values()
        }
        if file_documentation_count not in possible_totals:
            raise ValueError("documentation-call count does not match agent_rels and mode.")

        expected_digest = hashlib.sha256(
            "\n".join(call.call_id for call in manifest.calls).encode("utf-8")
        ).hexdigest()
        if manifest.digest != expected_digest:
            raise ValueError("call manifest digest does not match its ordered call IDs.")

        return replace(
            self,
            review_calls_planned=review_count,
            file_documentation_calls_planned=file_documentation_count,
            unit_documentation_calls_planned=unit_documentation_count,
            file_reduction_calls_planned=file_reduction_count,
            synthesis_calls_planned=synthesis_count,
            documentation_calls_planned=documentation_count,
            total_calls_planned=total,
            max_planned_calls=max_planned_calls,
            max_planned_calls_exceeded=(
                max_planned_calls > 0 and total > max_planned_calls
            ),
            call_manifest_digest=manifest.digest,
        )


@dataclass(frozen=True)
class PlanMaterials:
    """Auxiliary planning data execution needs to materialize the plan.

    These are derived from the same inputs as the plan, so execution never
    recomputes routing decisions — it only looks up the records the plan
    already chose.
    """

    # rel_path -> existing doc record reused via identical content.
    identical_reuse_docs: Mapping[str, dict] = field(default_factory=dict)
    # rel_path -> SHA-256 from the one frozen planning snapshot. Final output
    # assembly consumes this map instead of reopening source files after paid
    # work, so documentation and its persisted cache hash always describe the
    # same source revision.
    content_hashes: Mapping[str, str] = field(default_factory=dict)
    # rel_path -> the frozen execution request, keyed by plan.agent_rels.
    # Internal — never serialized into public documentation or recovery.
    execution_requests: Mapping[str, FileExecutionRequest] = field(default_factory=dict)
    # Files rejected by the provider-free source precheck. They have no unpaid
    # provider action, so they receive no execution request or manifest entry
    # and are excluded from `unpaid_action_rels` — and therefore from
    # `max_files` candidate counting — in every mode (D13's exact-unpaid-plan
    # rule, not a split-only behavior).
    insufficient_source_reasons: Mapping[str, str] = field(default_factory=dict)
    # rel_path -> deterministic complete split division plan (always effective
    # split; a blocked file never has one — see division_blocked).
    division_plans: Mapping[str, DivisionPlan] = field(default_factory=dict)
    # rel_path -> the matching provider-free reduction tree for every division
    # plan above.
    reduction_trees: Mapping[str, ReductionTreePlan] = field(default_factory=dict)
    # rel_path -> exact, dependency-validated retained split-tree checkpoint
    # state (only nodes that passed validate_recovered_tree survive here).
    tree_states: Mapping[str, SplitTreeState] = field(default_factory=dict)
    # rel_path -> a forced split file's structurally valid recovered
    # container, carried forward untouched (section 9): never validated,
    # never consulted for scheduling, never counted as restored work.
    carry_states: Mapping[str, SplitTreeState] = field(default_factory=dict)
    # rel_path -> first-failing capacity-blocked reason (D3/D6/D8).
    division_blocked: Mapping[str, str] = field(default_factory=dict)
    # The provider-free provider/model/effective-endpoint execution identity
    # for this run (D12), shared by every recoverable split node.
    provider_identity: str = ""
    # Provider-free recovery observability (section 19). These count planned
    # nodes and conflict files, never provider attempts.
    reexecuted_nodes: int = 0
    recovery_conflict_files: int = 0

    def __post_init__(self) -> None:
        for name in (
            "identical_reuse_docs",
            "content_hashes",
            "execution_requests",
            "insufficient_source_reasons",
            "division_plans",
            "reduction_trees",
            "tree_states",
            "carry_states",
            "division_blocked",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(getattr(self, name))),
            )
        for name in ("reexecuted_nodes", "recovery_conflict_files"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True)
class PlanSourceInputs:
    """The source-dependent planning inputs a stale-revision rebuild refreshes.

    Returned by the ``rebuild_source_inputs`` callback ``codedoc.pipeline``
    supplies to :func:`build_pipeline_plan`.  The dependency graph is parsed
    before routing hashes are taken, so a file that changed since then may have
    been routed from a revision that is already stale; the pipeline owns
    re-running scanning, parsing, graph construction, and entry selection, and
    hands the fresh results back here.
    """

    file_map: dict[str, dict]
    graph: DependencyGraph
    selected_rels: set[str]
    entry_rel: str | None


class _StaleSourceSnapshotError(Exception):
    """Internal signal: a file changed between the routing hash and the
    request snapshot. Caught only by :func:`build_pipeline_plan`, which
    rebuilds the complete source-dependent planning inputs once before
    reporting a concurrent-source-change error."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        super().__init__(rel_path)


def _build_execution_request(
    rel_path: str,
    descriptor: dict,
    expected_hash: str,
    resolved_profile: ResolvedProfile,
    analysis_mode: str,
    max_content_chars: int,
    head_ratio: float,
    snapshot: tuple[str, str] | None = None,
    parsed_imports: tuple[str, ...] | None = None,
) -> FileExecutionRequest:
    """Build one frozen :class:`FileExecutionRequest` from a fresh snapshot.

    Reads raw bytes once (:func:`read_source_snapshot`) and verifies the
    resulting hash against *expected_hash* (the hash that already decided this
    file's routing) before using the snapshot for anything. A mismatch means
    the file changed concurrently; it is reported to the caller as
    :class:`_StaleSourceSnapshotError` rather than silently mixing content,
    imports, and a hash from different file revisions.

    A deterministic parser failure (e.g. invalid Python syntax) no longer has
    a per-file execution attempt to fail during, since planning now derives
    imports once, up front, for every agent-routed file in one pass. Rather
    than aborting the whole run for one unparseable file, this falls back to
    an empty import list and logs a warning; the file still reaches its normal
    documentation call from its real content.
    """
    snapshot_hash, content = (
        snapshot
        if snapshot is not None
        else read_source_snapshot(descriptor["path"])
    )
    if snapshot_hash != expected_hash:
        raise _StaleSourceSnapshotError(rel_path)
    if parsed_imports is None:
        try:
            imports = tuple(parse_source(descriptor, content))
        except ParseError as exc:
            logger.warning(
                "Could not parse imports for %s: %s. Proceeding with an empty "
                "import list for this file.",
                rel_path,
                exc,
            )
            imports = ()
    else:
        imports = tuple(parsed_imports)
    bundle = resolved_profile.resolve_bundle(resolved_profile.scope_for(descriptor))
    return FileExecutionRequest(
        rel_path=rel_path,
        language=descriptor.get("language", "generic"),
        imports=imports,
        content=content,
        content_hash=snapshot_hash,
        context=AgentCallContext(
            analysis_mode=analysis_mode,
            max_content_chars=max_content_chars,
            truncation_head_ratio=head_ratio,
            resolved_shape_bundle=bundle,
        ),
    )


def normalize_force_files(
    force_files: list[str],
    root: Path,
) -> list[str]:
    """Normalize forced paths against the resolved project root.

    Accepts project-relative paths and absolute paths that resolve inside the
    project root; resolves ``.`` and ``..`` without allowing root escape;
    returns de-duplicated project-relative POSIX paths in input order.
    Raises :class:`ConfigError` for a path outside the project root.
    """
    root_resolved = root.resolve()
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in force_files:
        p = Path(str(raw))
        candidate = p if p.is_absolute() else root_resolved / p
        resolved = candidate.resolve()
        try:
            rel = resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            raise ConfigError(
                f"force_files path '{raw}' resolves outside the project root "
                f"'{root_resolved}'. Provide a path inside the project."
            )
        if rel not in seen:
            seen.add(rel)
            normalized.append(rel)
    return normalized


def build_pipeline_plan(
    file_map: dict[str, dict],
    graph: DependencyGraph,
    selected_rels: set[str],
    entry_rel: str | None,
    existing_docs: dict[str, dict],
    forced_paths: list[str],
    config: dict,
    resolved_profile: ResolvedProfile | None = None,
    *,
    rebuild_source_inputs: Callable[[], PlanSourceInputs] | None = None,
    recovered_partials: Mapping[str, SplitTreeState] | None = None,
) -> tuple[PipelinePlan, PlanMaterials]:
    """Compute the full routing plan for one pipeline run, read-only.

    Parameters
    ----------
    file_map:
        Scanned descriptors keyed by project-relative path.
    graph:
        The parsed dependency graph for all scanned files.
    selected_rels:
        Files selected by entry reachability (or all scanned files).
    entry_rel:
        The resolved entry path, or ``None``.
    existing_docs:
        Per-file records loaded read-only from the selected output target(s), or
        the exact validated opposite-format sibling when a single-format target
        is missing, overlaid with compatible ``crash_recovery.json`` records.
    forced_paths:
        Normalized project-relative forced paths (see
        :func:`normalize_force_files`).
    config:
        The resolved configuration dict.
    rebuild_source_inputs:
        Optional callback supplied by ``codedoc.pipeline`` that re-runs
        scanning, parsing, dependency-graph construction, and entry selection
        and returns the fresh :class:`PlanSourceInputs`.  It is invoked exactly
        once, only when a stale revision is detected.  A caller that owns no
        graph (e.g. a focused planning test) may omit it, in which case the one
        retry re-reads hashes and snapshots against the graph it supplied.

    A genuine :class:`~codedoc.core.file_division.DivisionInternalDefect`
    propagates uncaught from this function: it is a CodeDoc invariant failure,
    never a per-file failure, and must abort the whole dry or real run
    provider-free (D8).

    Every planned agent file's :class:`~codedoc.core.execution_model.FileExecutionRequest`
    is built here from one canonical source snapshot (see
    :func:`codedoc.core.db.read_source_snapshot`).  If a file's content changes
    between the routing hash and that snapshot, the complete source-dependent
    planning input construction is repeated once from fresh hashes and
    snapshots — including the caller's scan/parse/graph/entry-selection inputs
    when *rebuild_source_inputs* is supplied, because dependency routing may
    itself have observed the stale revision.  A second such change raises a
    deterministic :class:`~codedoc.utils.errors.ConfigError` before any provider
    is created.
    """
    args = (
        file_map, graph, selected_rels, entry_rel, existing_docs, forced_paths, config,
        resolved_profile, recovered_partials,
    )
    try:
        return _build_pipeline_plan_once(*args)
    except _StaleSourceSnapshotError as first:
        if rebuild_source_inputs is not None:
            logger.warning(
                "Source file '%s' changed while codedoc was planning this run. "
                "Rescanning, reparsing, and rebuilding the dependency graph and "
                "selection once so content, imports, hashes, and routing all "
                "describe one file revision.",
                first.rel_path,
            )
            fresh = rebuild_source_inputs()
            args = (
                fresh.file_map, fresh.graph, fresh.selected_rels, fresh.entry_rel,
                existing_docs, forced_paths, config, resolved_profile,
                recovered_partials,
            )
        try:
            return _build_pipeline_plan_once(*args)
        except _StaleSourceSnapshotError as exc:
            raise ConfigError(
                f"Source file '{exc.rel_path}' changed while codedoc was scanning "
                "and planning this run, and changed again on the retry. This is a "
                "concurrent-modification error, not a codedoc defect — re-run "
                "codedoc once the working tree is stable."
            ) from exc


def _build_pipeline_plan_once(
    file_map: dict[str, dict],
    graph: DependencyGraph,
    selected_rels: set[str],
    entry_rel: str | None,
    existing_docs: dict[str, dict],
    forced_paths: list[str],
    config: dict,
    resolved_profile: ResolvedProfile | None,
    recovered_partials: Mapping[str, SplitTreeState] | None,
) -> tuple[PipelinePlan, PlanMaterials]:
    """One provider-free planning pass. See :func:`build_pipeline_plan`."""
    scanned_rels = frozenset(file_map)

    # Forced-path scanning/selection filters — warn once per excluded path.
    effective_forced: set[str] = set()
    for rel in forced_paths:
        if rel not in scanned_rels:
            logger.warning(
                "force_files: '%s' is not in the scanned file set and will be "
                "ignored. Check the path and your skip_dirs / ignore_paths / "
                "extension settings.",
                rel,
            )
        elif rel not in selected_rels:
            logger.warning(
                "force_files: '%s' was scanned but is outside the current "
                "entry-based selection and will be ignored. Run without "
                "--entry, or force a file reachable from the entry.",
                rel,
            )
        else:
            effective_forced.add(rel)

    # The run-level part of the expected cache identity (revision + mode),
    # shared by every file.
    analysis_mode = config.get("analysis_mode", "single")
    base_identity = expected_analysis_identity(analysis_mode)
    # Every AgentCallContext built below shares this identical mode/ceiling/ratio
    # and a resolved-profile instance; only resolved_shape_bundle varies per file.
    effective_resolved_profile = (
        resolved_profile
        if resolved_profile is not None
        else ResolvedProfile(analysis_mode, None)
    )

    # Capture the routing hash, then one canonical raw-byte/decoded snapshot for
    # every selected file. Comparing the two catches a concurrent edit between
    # routing and request construction even for a file that would otherwise be
    # classified unchanged. The outer build_pipeline_plan() retries the entire
    # provider-free planning pass once; workers never perform either read.
    routing_hashes = {
        rel: compute_file_hash(file_map[rel]["path"])
        for rel in selected_rels
    }
    source_snapshots: dict[str, tuple[str, str]] = {}
    for rel in selected_rels:
        snapshot = read_source_snapshot(file_map[rel]["path"])
        if snapshot[0] != routing_hashes[rel]:
            raise _StaleSourceSnapshotError(rel)
        source_snapshots[rel] = snapshot
    insufficient_source_reasons = {
        rel: reason
        for rel, (_, content) in source_snapshots.items()
        for is_insufficient, reason in (insufficient_source(content),)
        if is_insufficient
    }

    # Per-file truncation identity. Source snapshots were already read once for
    # every selected file above because frozen execution requests and optional
    # split planning both consume the exact decoded text. This memoized helper
    # computes the same character count the orchestrator consumes.
    max_content_chars = int(config.get("max_content_chars", 12000) or 12000)
    head_ratio = float(config.get("truncation_head_ratio", 0.70) or 0.70)
    _mcr_cache: dict[str, str | None] = {}
    # rel -> (DivisionPlan, ReductionTreePlan) | None. `None` means: not a
    # split-eligible oversized file, OR blocked (see division_blocked below).
    _split_cache: dict[str, tuple[DivisionPlan, ReductionTreePlan] | None] = {}
    _imports_cache: dict[str, tuple[str, ...]] = {}
    division_blocked: dict[str, BlockedReason] = {}
    provider_identity = provider_execution_identity(config)
    split_release = current_split_release_policy()

    def _imports_for(rel: str) -> tuple[str, ...]:
        if rel not in _imports_cache:
            try:
                _imports_cache[rel] = tuple(
                    parse_source(file_map[rel], source_snapshots[rel][1])
                )
            except ParseError as exc:
                logger.warning(
                    "Could not parse imports for %s: %s. Proceeding with an "
                    "empty import list for this file.",
                    rel,
                    exc,
                )
                _imports_cache[rel] = ()
        return _imports_cache[rel]

    def _split_outcome_for(rel: str) -> tuple[DivisionPlan, ReductionTreePlan] | None:
        """Provider-free split plan/tree for *rel*, or `None`.

        `None` covers three distinct cases the caller must not conflate:
        split is not requested/eligible for this file, the file fits within
        `max_content_chars` (D1), or the file is capacity-blocked (its
        reason is recorded in `division_blocked`).  A genuine
        `DivisionInternalDefect` propagates uncaught (D8).
        """
        if rel in _split_cache:
            return _split_cache[rel]
        if rel in insufficient_source_reasons:
            _split_cache[rel] = None
            return None
        # D2 defense-in-depth: split is valid only in single mode, even if
        # validated configuration were somehow bypassed.
        if (
            config.get("large_file_strategy", "truncate") != "split"
            or analysis_mode != "single"
            or len(source_snapshots[rel][1]) <= max_content_chars
        ):
            _split_cache[rel] = None
            return None
        try:
            plan = build_division_plan(
                rel_path=rel,
                language=file_map[rel].get("language", "generic"),
                content=source_snapshots[rel][1],
                source_budget_chars=max_content_chars,
            )
            tree = build_reduction_tree(
                plan,
                max_content_chars=max_content_chars,
                language=file_map[rel].get("language", "generic"),
                imports=_imports_for(rel),
            )
        except SplitCapacityBlocked as exc:
            division_blocked[rel] = exc.reason
            _split_cache[rel] = None
            return None
        _split_cache[rel] = (plan, tree)
        return (plan, tree)

    def _expected_identity_for(rel: str) -> dict[str, str]:
        if rel not in _mcr_cache:
            _mcr_cache[rel] = expected_max_context_revision(
                len(source_snapshots[rel][1]),
                max_chars=max_content_chars,
                head_ratio=head_ratio,
            )
        mcr = _mcr_cache[rel]
        identity = dict(base_identity)
        # A capacity/tree feasibility check always runs before reuse is
        # authorized (D8): calling _split_outcome_for here — even for an
        # otherwise "unchanged" file — means a newly blocked plan is never
        # hidden behind stale reuse.
        # D2 explicit local gate: visible here, not only via delegation to
        # _split_outcome_for's own internal check.
        effective_split_requested = (
            config.get("large_file_strategy", "truncate") == "split"
            and analysis_mode == "single"
        )
        outcome = _split_outcome_for(rel) if effective_split_requested else None
        if outcome is not None:
            division_plan, reduction_tree = outcome
            large_identity = expected_large_file_identity(
                source_chars=len(source_snapshots[rel][1]),
                max_chars=max_content_chars,
                rel_path=rel,
                division_plan_digest=division_plan.plan_digest,
                reduction_tree_digest=reduction_tree.tree_digest,
                structural_mode=division_plan.structural_mode,
                imports_digest=deterministic_imports_digest(_imports_for(rel)),
            )
            if large_identity is not None:
                identity["_large_file_identity"] = large_identity
        else:
            # No split outcome: an ordinary record (source fits max_content_chars)
            # or an oversized truncate-path record, covered identically here.
            # `_large_file_identity` already binds the path for a split record
            # (D12/section 13), so `_ordinary_path_identity` is stamped only on
            # this branch — never alongside `_large_file_identity` above.
            identity["_ordinary_path_identity"] = expected_ordinary_path_identity(rel)
            if mcr is not None:
                # Effective split records never carry `_max_context_revision`
                # (D12/section 13): only an ordinary oversized truncate-path record does.
                identity["_max_context_revision"] = mcr
        # An active prompt-customization profile contributes a per-file digest
        # keyed on the file's basename (extension scope).  The basename is derived
        # from the ``rel`` key itself — no descriptor field is needed.  Omitted
        # when no profile is active for that scope, so the absent-default
        # normalization keeps no-profile records reusable.
        if resolved_profile is not None:
            basename = PurePosixPath(rel).name.lower()
            digest = resolved_profile.file_digest(basename)
            if digest != NO_PROMPT_PROFILE_DIGEST:
                identity["_prompt_profile_digest"] = digest
        return identity

    # Index reusable candidates by content hash, retaining *all* records
    # with the same hash (was a single-record-per-hash last-writer-wins map).
    # Two records with identical content can carry different cache identities, so
    # the per-file loop must be free to pick a candidate that passes the
    # centralized predicate for the destination file.
    docs_by_hash: dict[str, list[dict]] = {}
    for doc in existing_docs.values():
        doc_hash = doc.get("hash")
        if doc_hash:
            docs_by_hash.setdefault(doc_hash, []).append(doc)

    # Changed = the same-path existing record is not reusable (hash differs, or
    # the hash matches but the cache identity is missing/stale).  Routing the
    # same-path "unchanged" determination through the predicate ensures a record
    # whose revision/mode no longer matches is reprocessed instead of silently
    # reused.  Forced paths are added before dependency propagation, so
    # dependents of a forced file are included exactly as for a hash change.
    changed_rels = {
        rel for rel in selected_rels
        if not _record_is_reusable(
            existing_docs.get(rel),
            routing_hashes[rel],
            _expected_identity_for(rel),
            file_map[rel].get("language", "generic"),
            rel_path=rel,
        )
    }
    changed_rels |= effective_forced
    changed_rels |= set(insufficient_source_reasons)

    # Track oversized targets that were not accepted for same-path completed reuse.
    # Keep this policy set out of changed_rels: an unchanged split file can still
    # require paid work, but that is not a source/identity change and must not
    # falsely propagate to its dependents.
    split_execution_rels = {
        rel
        for rel in selected_rels
        if not split_release.completed_reuse
        and _split_outcome_for(rel) is not None
    }

    if config.get("propagate_changes", True):
        process_rels = graph.affected_by_changes(changed_rels) & selected_rels
    else:
        process_rels = set(changed_rels)
    process_rels |= split_execution_rels

    unchanged_rels = selected_rels - process_rels
    completed_split_reuse_rels = {
        rel for rel in unchanged_rels if _split_outcome_for(rel) is not None
    }

    identical_reuse_docs: dict[str, dict] = {}
    execution_requests: dict[str, FileExecutionRequest] = {}
    division_plans: dict[str, DivisionPlan] = {}
    reduction_trees: dict[str, ReductionTreePlan] = {}
    tree_states: dict[str, SplitTreeState] = {}
    # A forced split file's structurally valid recovered container, carried
    # forward untouched (section 9): force bypasses reuse and recovery for
    # THIS run's scheduling, but the prior checkpoint is preserved so a later
    # non-forced run may still resume it if forced execution fails or is
    # interrupted. Never validated, never consulted for scheduling, never
    # counted as restored work.
    carry_states: dict[str, SplitTreeState] = {}
    reexecuted_nodes = 0
    recovery_conflict_paths: set[str] = set()
    recovered_by_path = dict(recovered_partials or {})
    identical_reuse: set[str] = set()
    agent_rels: set[str] = set()
    agent_candidate_rels: set[str] = set()

    def _route_execution_request(
        rel_path: str,
        descriptor: dict,
        content_hash: str,
    ) -> None:
        nonlocal reexecuted_nodes
        if rel_path in insufficient_source_reasons:
            return
        request = _build_execution_request(
            rel_path,
            descriptor,
            content_hash,
            effective_resolved_profile,
            analysis_mode,
            max_content_chars,
            head_ratio,
            source_snapshots[rel_path],
            _imports_for(rel_path),
        )
        split_requested_and_oversized = (
            config.get("large_file_strategy", "truncate") == "split"
            and len(request.content) > max_content_chars
        )
        if split_requested_and_oversized:
            # D2 explicit local gate: visible here, not only via delegation
            # to _split_outcome_for's own internal check. An invalid mode
            # combination excludes the file rather than silently falling
            # back to ordinary/truncate-style treatment — split never falls
            # back to truncation, even defensively, and this state should
            # already be unreachable via loader._validate() (D2/D8).
            if analysis_mode != "single":
                return
            outcome = _split_outcome_for(rel_path)
            if outcome is None:
                # Blocked (or, defensively, ineligible): no provider action,
                # excluded from max_files candidates (D8).
                return
            division_plan, reduction_tree = outcome
            split_content_hash = content_hash
            division_plans[rel_path] = division_plan
            reduction_trees[rel_path] = reduction_tree
            agent_rels.add(rel_path)
            execution_requests[rel_path] = request
            recovered = (
                recovered_by_path.get(rel_path)
                if split_release.partial_recovery
                else None
            )
            if rel_path in effective_forced:
                # Force bypasses execution, not preservation (section 9): the
                # recovered container is carried forward untouched rather than
                # validated or scheduled from, so a later non-forced run may
                # still resume it if this forced run fails or is interrupted.
                if recovered is not None:
                    carry_states[rel_path] = recovered
            elif recovered is not None:
                if recovered.content_hash != split_content_hash:
                    # A source revision invalidates every old node. Preserve
                    # the old container until a completed replacement succeeds;
                    # SafeWriter suppresses new checkpoints while carry state
                    # for this path exists, so a failed run cannot overwrite it.
                    carry_states[rel_path] = recovered
                    recovery_conflict_paths.add(rel_path)
                    current_node_ids = {
                        *(chunk.chunk_id for chunk in division_plan.chunks),
                        *(node.node_id for node in reduction_tree.all_nodes),
                    }
                    recovered_paid_ids = {
                        node.node_id for node in recovered.nodes
                    } | {entry.node_id for entry in recovered.quarantine}
                    reexecuted_nodes += len(recovered_paid_ids & current_node_ids)
                else:
                    try:
                        retained_nodes, quarantine_entries = validate_recovered_tree(
                            recovered.nodes,
                            plan=division_plan,
                            tree=reduction_tree,
                            content_hash=split_content_hash,
                            provider_identity=provider_identity,
                            prompt_profile_digest=(
                                request.context.resolved_shape_bundle.digest
                            ),
                            imports_digest=deterministic_imports_digest(
                                request.imports
                            ),
                            imports=request.imports,
                            language=request.language,
                            resolved_shape=resolved_synthesis_shape(
                                request.context.resolved_shape_bundle
                            ),
                            max_content_chars=request.context.max_content_chars,
                            existing_quarantine=recovered.quarantine,
                        )
                    except SplitRecoveryStateError as exc:
                        raise ConfigError(
                            f"Recovery for {rel_path!r} cannot be safely bounded "
                            "under the current schema-4 plan. The recovery file "
                            "and stable output were left untouched. Resume with "
                            "the CodeDoc version that created it, or move the "
                            "recovery file aside before starting fresh; delete "
                            "it only as a deliberate discard."
                        ) from exc
                    retained_ids = {node.node_id for node in retained_nodes}
                    previously_paid_ids = {
                        node.node_id for node in recovered.nodes
                    } | {entry.node_id for entry in recovered.quarantine}
                    reexecuted_nodes += len(previously_paid_ids - retained_ids)
                    if quarantine_entries:
                        recovery_conflict_paths.add(rel_path)
                    if retained_nodes or quarantine_entries:
                        tree_states[rel_path] = SplitTreeState(
                            schema_version=recovered.schema_version,
                            owner=recovered.owner,
                            rel_path=rel_path,
                            content_hash=split_content_hash,
                            division_plan_digest=division_plan.plan_digest,
                            reduction_tree_digest=reduction_tree.tree_digest,
                            nodes=retained_nodes,
                            quarantine=quarantine_entries,
                        )
            completed_ids = frozenset(
                tree_states[rel_path].by_id()
                if rel_path in tree_states
                else ()
            )
            planned_ids = {
                *(chunk.chunk_id for chunk in division_plan.chunks),
                *(node.node_id for node in reduction_tree.all_nodes),
            }
            if planned_ids - completed_ids:
                agent_candidate_rels.add(rel_path)
            return
        agent_candidate_rels.add(rel_path)
        agent_rels.add(rel_path)
        execution_requests[rel_path] = request

    for rel_path in graph.topological_order():
        if rel_path not in process_rels:
            continue
        descriptor = file_map[rel_path]
        content_hash = routing_hashes[rel_path]

        if rel_path in insufficient_source_reasons:
            _route_execution_request(rel_path, descriptor, content_hash)
            continue

        if rel_path in effective_forced:
            # Forcing bypasses identical-content reuse for the explicitly forced
            # file.  Propagated dependents keep normal reuse behaviour below.
            _route_execution_request(rel_path, descriptor, content_hash)
            continue

        if rel_path in split_execution_rels:
            _route_execution_request(rel_path, descriptor, content_hash)
            continue

        # Identical-content reuse: only a candidate that passes the centralized
        # predicate (hash + every cache-identity key) for this destination file
        # is eligible.  A candidate matching content but carrying a stale/missing
        # revision or mode is skipped.
        candidate = None
        if content_hash in docs_by_hash:
            candidate = next(
                (
                    doc
                    for doc in docs_by_hash[content_hash]
                    if _record_is_reusable(
                        doc,
                        content_hash,
                        _expected_identity_for(rel_path),
                        descriptor.get("language", "generic"),
                        rel_path=rel_path,
                    )
                ),
                None,
            )
        if candidate is not None:
            identical_reuse.add(rel_path)
            identical_reuse_docs[rel_path] = candidate
            continue

        _route_execution_request(rel_path, descriptor, content_hash)

    max_files = int(config.get("max_files", 0) or 0)
    max_files_exceeded = max_files > 0 and len(agent_candidate_rels) > max_files

    plan = PipelinePlan(
        scanned_rels=scanned_rels,
        documented_rels=frozenset(selected_rels),
        changed_rels=frozenset(changed_rels),
        forced_rels=frozenset(effective_forced),
        process_rels=frozenset(process_rels),
        unchanged_rels=frozenset(unchanged_rels),
        identical_reuse_rels=frozenset(identical_reuse),
        agent_rels=frozenset(agent_rels),
        division_plan_rels=frozenset(division_plans),
        completed_split_reuse_rels=frozenset(completed_split_reuse_rels),
        division_blocked=dict(division_blocked),
        entry_rel=entry_rel,
        max_files=max_files,
        max_files_exceeded=max_files_exceeded,
        unpaid_action_rels=frozenset(agent_candidate_rels),
        split_execution_rels=frozenset(split_execution_rels),
    )
    materials = PlanMaterials(
        identical_reuse_docs=identical_reuse_docs,
        content_hashes=routing_hashes,
        execution_requests=execution_requests,
        insufficient_source_reasons=insufficient_source_reasons,
        division_plans=division_plans,
        reduction_trees=reduction_trees,
        tree_states=tree_states,
        carry_states=carry_states,
        division_blocked=dict(division_blocked),
        provider_identity=provider_identity,
        reexecuted_nodes=reexecuted_nodes,
        recovery_conflict_files=len(recovery_conflict_paths),
    )
    return plan, materials
