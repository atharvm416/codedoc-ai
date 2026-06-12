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
from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelinePlan:
    """Immutable description of all routing decisions for one pipeline run."""

    scanned_rels: frozenset[str]
    selected_rels: frozenset[str]
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

    docs_by_hash: dict[str, dict] = {
        doc["hash"]: doc for doc in existing_docs.values() if doc.get("hash")
    }

    # Changed = hash differs from existing docs; forced paths are added before
    # dependency propagation, so dependents of a forced file are included
    # exactly as they would be for a hash change.
    changed_rels = {
        rel for rel in selected_rels
        if compute_file_hash(file_map[rel]["path"]) != existing_docs.get(rel, {}).get("hash", "")
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
        elif content_hash in docs_by_hash:
            identical_reuse.add(rel_path)
            materials.identical_reuse_docs[rel_path] = docs_by_hash[content_hash]
        elif rel_path in checkpoint_records:
            checkpoint_entry = checkpoint_records[rel_path]
            stored_hash = checkpoint_entry.get("_checkpoint_hash", "")
            if not stored_hash:
                logger.info(
                    "Checkpoint entry for '%s' has no hash — reprocessing.", rel_path
                )
                agent_rels.add(rel_path)
            elif content_hash != stored_hash:
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
        else:
            agent_rels.add(rel_path)

    max_files = int(config.get("max_files", 0) or 0)
    max_files_exceeded = max_files > 0 and len(agent_rels) > max_files

    plan = PipelinePlan(
        scanned_rels=scanned_rels,
        selected_rels=frozenset(selected_rels),
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
