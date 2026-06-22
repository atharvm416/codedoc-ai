"""Shared pipeline planning for codedoc (0.9.2).

The planning helper computes every routing decision — selection, forcing,
propagation, unchanged skipping, identical-content reuse, legacy checkpoint
reuse, and the paid-file cap — into one immutable :class:`PipelinePlan` that
both ``--dry-run`` and real execution consume.  It may read source contents
and hashes, but it never writes, never creates a provider, and never
initializes ``SafeWriter``.

Format detection and ownership inspection live in ``codedoc.core.output``;
this module only consumes their results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codedoc.core.db import compute_file_hash
from codedoc.core.graph import DependencyGraph
from codedoc.core.record_meta import CACHE_IDENTITY_KEYS, expected_analysis_identity
from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def _identity_matches(stored: dict, expected: dict) -> bool:
    """Compare every key in :data:`CACHE_IDENTITY_KEYS`.

    An absent expected key and absent stored key compare equal; a
    present-but-mismatched key blocks reuse.
    """
    if not isinstance(stored, dict):
        return False
    for key in CACHE_IDENTITY_KEYS:
        if stored.get(key) != expected.get(key):
            return False
    return True


def _record_is_reusable(stored: dict | None, content_hash: str, expected: dict) -> bool:
    """The single centralized reuse predicate (0.10.0).

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
    checkpoint_reuse_rels: frozenset[str]
    agent_rels: frozenset[str]
    entry_rel: str | None
    max_files: int
    max_files_exceeded: bool

    @property
    def selected_rels(self) -> frozenset[str]:
        """Read-only compatibility alias for :attr:`documented_rels` (0.10.0).

        The canonical field is now ``documented_rels``; ``selected_rels`` is
        retained as a non-settable delegating property so existing callers do
        not break.  Do not remove it in this release.
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
    # rel_path -> legacy checkpoint result (hash key already stripped).
    checkpoint_reuse_docs: dict[str, dict] = field(default_factory=dict)


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
    checkpoint_records: dict[str, dict],
    forced_paths: list[str],
    config: dict,
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
        Per-file records loaded read-only from existing output files.
    checkpoint_records:
        Eligible legacy ``.codedoc_progress.json`` records.  Must be empty
        when the live backup already contains records (``SafeWriter.size == 0``
        equivalence is the caller's responsibility).
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

    # 0.10.0: the expected cache identity for this run (revision + resolved mode).
    expected_identity = expected_analysis_identity(
        config.get("analysis_mode", "single")
    )

    # 0.10.0: index reusable candidates by content hash, retaining *all* records
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
            expected_identity,
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
    checkpoint_reuse: set[str] = set()
    agent_rels: set[str] = set()

    for rel_path in graph.topological_order():
        if rel_path not in process_rels:
            continue
        descriptor = file_map[rel_path]
        content_hash = compute_file_hash(descriptor["path"])
        materials.content_hashes[rel_path] = content_hash

        if rel_path in effective_forced:
            # Forcing bypasses identical-content reuse and checkpoint reuse
            # for the explicitly forced file.  Propagated dependents keep
            # normal reuse behaviour below.
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
                    if _record_is_reusable(doc, content_hash, expected_identity)
                ),
                None,
            )
        if candidate is not None:
            identical_reuse.add(rel_path)
            materials.identical_reuse_docs[rel_path] = candidate
            continue

        # Legacy checkpoint reuse: eligible only when the checkpoint hash matches
        # and the checkpoint carries a matching cache identity.  A checkpoint
        # without the analysis revision/mode is reprocessed once.
        if rel_path in checkpoint_records:
            checkpoint_entry = checkpoint_records[rel_path]
            stored_hash = checkpoint_entry.get("_checkpoint_hash", "")
            checkpoint_candidate = {**checkpoint_entry, "hash": stored_hash}
            if not stored_hash:
                logger.info(
                    "Checkpoint entry for '%s' has no hash — reprocessing.", rel_path
                )
                agent_rels.add(rel_path)
            elif not _record_is_reusable(
                checkpoint_candidate, content_hash, expected_identity
            ):
                if content_hash == stored_hash:
                    logger.info(
                        "Checkpoint entry for '%s' predates the current analysis "
                        "revision/mode — reprocessing.",
                        rel_path,
                    )
                else:
                    logger.info(
                        "File '%s' was modified after it was checkpointed — reprocessing.",
                        rel_path,
                    )
                agent_rels.add(rel_path)
            else:
                checkpoint_reuse.add(rel_path)
                materials.checkpoint_reuse_docs[rel_path] = {
                    k: v for k, v in checkpoint_entry.items() if k != "_checkpoint_hash"
                }
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
        checkpoint_reuse_rels=frozenset(checkpoint_reuse),
        agent_rels=frozenset(agent_rels),
        entry_rel=entry_rel,
        max_files=max_files,
        max_files_exceeded=max_files_exceeded,
    )
    return plan, materials
