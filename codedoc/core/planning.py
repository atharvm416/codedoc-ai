"""Shared pipeline planning for codedoc.

The planning helper computes every routing decision — selection, forcing,
propagation, unchanged skipping, identical-content reuse, and the paid-file cap
— into one immutable :class:`PipelinePlan` that both ``--dry-run`` and real
execution consume.  It may read source contents and hashes, but it never writes,
never creates a provider, and never initializes ``SafeWriter``.

Format detection and ownership inspection live in ``codedoc.core.output``;
this module only consumes their results.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

from codedoc.core.db import compute_file_hash, read_source_snapshot
from codedoc.core.execution_model import (
    DOC_AGENTS_BY_MODE,
    REVIEW_OWNER,
    AgentCallContext,
    CallManifest,
    FileExecutionRequest,
    documentation_call_id,
)
from codedoc.core.graph import DependencyGraph
from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST, ResolvedProfile
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    expected_analysis_identity,
    expected_max_context_revision,
    normalized_identity_value,
)
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


def _record_is_reusable(stored: dict | None, content_hash: str, expected: dict) -> bool:
    """The single centralized reuse predicate.

    A stored record may be reused only when its content hash matches *and* every
    cache-identity key matches the expected revision/mode.  A record that matches
    the content hash while ignoring a registered cache-identity key is a defect.
    """
    if not isinstance(stored, dict):
        return False
    if stored.get("hash", "") != content_hash:
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
    # Attached by with_call_manifest() once the canonical call manifest exists;
    # zero/empty defaults keep every existing direct PipelinePlan(...) caller
    # (e.g. focused planning tests) source-compatible.
    review_calls_planned: int = 0
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
        self, manifest: CallManifest, max_planned_calls: int
    ) -> "PipelinePlan":
        """Return a copy of this plan with the canonical call manifest's counts
        and cap attached.

        The per-file multiplier (1 for single mode, 3 for triple) is guaranteed
        by :func:`~codedoc.core.execution_model.build_call_manifest`'s own
        construction, not re-derived here; this defensively verifies the
        manifest's category counts are internally consistent with
        ``agent_rels`` before attaching them.
        """
        review_count = sum(1 for c in manifest.calls if c.category == "prompt-review")
        documentation_count = sum(
            1 for c in manifest.calls if c.category == "file-documentation"
        )
        total = len(manifest.calls)
        if review_count + documentation_count != total:
            raise ValueError("call manifest contains a call outside the known categories.")
        review_calls = [c for c in manifest.calls if c.category == "prompt-review"]
        if any(call.owner != REVIEW_OWNER for call in review_calls):
            raise ValueError("prompt-review call has a non-canonical owner.")

        doc_calls = [c for c in manifest.calls if c.category == "file-documentation"]
        doc_owners = {call.owner for call in doc_calls}
        if doc_owners != set(self.agent_rels):
            raise ValueError(
                "documentation-call owners do not exactly match agent_rels."
            )

        matched_modes: set[str] = set()
        for owner in sorted(self.agent_rels):
            owner_calls = [call for call in doc_calls if call.owner == owner]
            owner_matches: list[str] = []
            for mode, agents in DOC_AGENTS_BY_MODE.items():
                expected = [
                    (
                        documentation_call_id(owner, mode, agent, ordinal),
                        ordinal,
                    )
                    for ordinal, agent in enumerate(agents, start=1)
                ]
                actual = [(call.call_id, call.ordinal) for call in owner_calls]
                if actual == expected:
                    owner_matches.append(mode)
            if len(owner_matches) != 1:
                raise ValueError(
                    f"documentation calls for {owner!r} do not match a canonical mode."
                )
            matched_modes.add(owner_matches[0])
        if len(matched_modes) > 1:
            raise ValueError("call manifest mixes documentation analysis modes.")

        expected_documentation = (
            len(self.agent_rels)
            * (len(DOC_AGENTS_BY_MODE[next(iter(matched_modes))]) if matched_modes else 0)
        )
        if documentation_count != expected_documentation:
            raise ValueError("documentation-call count does not match agent_rels and mode.")

        expected_digest = hashlib.sha256(
            "\n".join(call.call_id for call in manifest.calls).encode("utf-8")
        ).hexdigest()
        if manifest.digest != expected_digest:
            raise ValueError("call manifest digest does not match its ordered call IDs.")

        return replace(
            self,
            review_calls_planned=review_count,
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

    # rel_path -> content hash for every file in process_rels.
    content_hashes: dict[str, str] = field(default_factory=dict)
    # rel_path -> existing doc record reused via identical content.
    identical_reuse_docs: dict[str, dict] = field(default_factory=dict)
    # rel_path -> the frozen execution request, keyed by plan.agent_rels.
    # Internal — never serialized into public documentation or recovery.
    execution_requests: dict[str, FileExecutionRequest] = field(default_factory=dict)
    # Files that reached the paid-file candidate set but were rejected by the
    # provider-free source precheck. They retain max_files candidate semantics,
    # but receive neither an execution request nor a manifest entry.
    insufficient_source_reasons: dict[str, str] = field(default_factory=dict)


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
    bundle = resolved_profile.resolve_bundle(resolved_profile.scope_for(descriptor))
    return FileExecutionRequest(
        rel_path=rel_path,
        absolute_path=descriptor["path"],
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
        resolved_profile,
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

    # The per-file part — the truncation revision for a file large enough
    # to be truncated under the current ceiling / head ratio.  Read-only and
    # memoized; a file whose byte size is within the ceiling never reads its text
    # (it cannot be truncated).  The char count is computed exactly as the
    # orchestrator computes ``len(content)``, so the expected and stored values
    # agree for every file.
    max_content_chars = int(config.get("max_content_chars", 12000) or 12000)
    head_ratio = float(config.get("truncation_head_ratio", 0.70) or 0.70)
    _mcr_cache: dict[str, str | None] = {}

    def _expected_identity_for(rel: str) -> dict[str, str]:
        if rel not in _mcr_cache:
            _mcr_cache[rel] = expected_max_context_revision(
                len(source_snapshots[rel][1]),
                max_chars=max_content_chars,
                head_ratio=head_ratio,
            )
        mcr = _mcr_cache[rel]
        identity = dict(base_identity)
        if mcr is not None:
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
        )
    }
    changed_rels |= effective_forced

    if config.get("propagate_changes", True):
        process_rels = graph.affected_by_changes(changed_rels) & selected_rels
    else:
        process_rels = set(changed_rels)

    unchanged_rels = selected_rels - process_rels

    materials = PlanMaterials()
    identical_reuse: set[str] = set()
    agent_rels: set[str] = set()
    agent_candidate_rels: set[str] = set()

    def _route_execution_request(
        rel_path: str,
        descriptor: dict,
        content_hash: str,
    ) -> None:
        agent_candidate_rels.add(rel_path)
        request = _build_execution_request(
            rel_path,
            descriptor,
            content_hash,
            effective_resolved_profile,
            analysis_mode,
            max_content_chars,
            head_ratio,
            source_snapshots[rel_path],
        )
        is_insufficient, reason = insufficient_source(request.content)
        if is_insufficient:
            materials.insufficient_source_reasons[rel_path] = reason
            return
        agent_rels.add(rel_path)
        materials.execution_requests[rel_path] = request

    for rel_path in graph.topological_order():
        if rel_path not in process_rels:
            continue
        descriptor = file_map[rel_path]
        content_hash = routing_hashes[rel_path]
        materials.content_hashes[rel_path] = content_hash

        if rel_path in effective_forced:
            # Forcing bypasses identical-content reuse for the explicitly forced
            # file.  Propagated dependents keep normal reuse behaviour below.
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
                    if _record_is_reusable(doc, content_hash, _expected_identity_for(rel_path))
                ),
                None,
            )
        if candidate is not None:
            identical_reuse.add(rel_path)
            materials.identical_reuse_docs[rel_path] = candidate
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
        entry_rel=entry_rel,
        max_files=max_files,
        max_files_exceeded=max_files_exceeded,
    )
    return plan, materials
