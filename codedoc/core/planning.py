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

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from codedoc.core.db import compute_file_hash, source_char_count
from codedoc.core.graph import DependencyGraph
from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST, ResolvedProfile
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    expected_analysis_identity,
    expected_max_context_revision,
    normalized_identity_value,
)
from codedoc.utils.errors import ConfigError
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

    @property
    def selected_rels(self) -> frozenset[str]:
        """Read-only compatibility alias for :attr:`documented_rels`.

        ``documented_rels`` is canonical; this non-settable delegating property
        preserves compatibility for existing callers.
        """
        return self.documented_rels


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
    """
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
    base_identity = expected_analysis_identity(
        config.get("analysis_mode", "single")
    )

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
                source_char_count(file_map[rel]["path"], ceiling=max_content_chars),
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
            compute_file_hash(file_map[rel]["path"]),
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

    for rel_path in graph.topological_order():
        if rel_path not in process_rels:
            continue
        descriptor = file_map[rel_path]
        content_hash = compute_file_hash(descriptor["path"])
        materials.content_hashes[rel_path] = content_hash

        if rel_path in effective_forced:
            # Forcing bypasses identical-content reuse for the explicitly forced
            # file.  Propagated dependents keep normal reuse behaviour below.
            agent_rels.add(rel_path)
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

        agent_rels.add(rel_path)

    max_files = int(config.get("max_files", 0) or 0)
    max_files_exceeded = max_files > 0 and len(agent_rels) > max_files

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
