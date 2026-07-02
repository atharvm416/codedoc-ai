"""File discovery, graph construction, and entry-based selection.

Extracted from ``codedoc.pipeline`` as part of the internal module decomposition.
This module owns:

- entry recovery from existing CodeDoc metadata (``_resolve_entry_and_docs``);
- dependency-graph construction from scanned descriptors (``_build_graph``);
- entry validation and the current entry-reachability selection
  (``_select_files``);
- graph-edge serialization (``_graph_edges``).

The selection logic is moved unchanged — it keeps the current
selected-file terminology and behavior so the extraction does not become
mixed with a feature change.  This module may depend on scanner/parser/graph
helpers, configuration errors, and the error reporter.  It must not write
output or create providers.
"""

from __future__ import annotations

from pathlib import Path

from codedoc.core.graph import DependencyGraph, resolve_import
from codedoc.core.project_view import read_codedoc_meta
from codedoc.core.scanner import detect_entry_file
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import ConfigError, ErrorReporter, ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_entry_and_docs(root: Path, config: dict) -> None:
    """Auto-discover the entry point from the exact selected CodeDoc output only.

    Inspects only the exact selected final target(s) for the current format — the
    exact JSON target (json/both) first, then the exact Markdown target (md/both) —
    and reads entry metadata from the first that exists and yields an entry.  No
    sibling, recovery, or opposite-format file is probed: a format switch never
    consults the other format's document for entry metadata (consistent with the
    Workstream G incremental-reuse rule).  If no selected target supplies an entry,
    ``entry_file`` is left unset so ``_select_files`` runs ordinary
    ``detect_entry_file()`` auto-detection.
    """
    if config.get("entry_file"):
        return

    raw_output = config.get("output_dir", "codedoc")
    p = Path(raw_output)

    if p.suffix.lower() in (".json", ".md"):
        # A fully-qualified file output path is itself the single exact target.
        candidates: list[Path] = [(root / p) if not p.is_absolute() else p]
    else:
        out_dir = (root / p) if not p.is_absolute() else p
        json_filename = config.get("output_json_filename", "codedoc.json")
        md_filename = config.get("output_md_filename", "codedoc.md")
        output_format = config.get("output_format", "json")
        candidates = []
        if output_format in ("json", "both"):
            candidates.append(out_dir / json_filename)
        if output_format in ("md", "both"):
            candidates.append(out_dir / md_filename)

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            meta = read_codedoc_meta(candidate)
        except ConfigError:
            continue
        entry = meta.get("entry_file")
        if entry:
            logger.info("Resuming: entry '%s' read from '%s'", entry, candidate.name)
            config["entry_file"] = entry
            return

    # No exact selected output supplied an entry and no --entry was provided.
    # Leave config["entry_file"] unset so _select_files() calls
    # detect_entry_file() with the configured auto-entry candidates.
    logger.info(
        "No entry found in the exact selected output and no --entry provided. "
        "Auto-detection via detect_entry_file() will be attempted."
    )


def _build_graph(
    all_files: list[dict],
    root: Path,
    error_reporter: ErrorReporter,
) -> tuple[DependencyGraph, dict[str, dict], dict[str, list[str]]]:
    """Build the dependency graph from scanned file descriptors.

    Returns ``(graph, file_map, unresolved_imports_by_path)``.

    ``unresolved_imports_by_path`` maps each file's ``rel_path`` to the list of
    raw import strings that the parser emitted but that did not resolve to any
    internal project file via ``resolve_import()``.  These are the candidates for
    external / SDK dependency projection in ``_project_dependency_links()``.
    """
    graph = DependencyGraph()
    all_rel_paths = {d["rel_path"] for d in all_files}
    file_map = {d["rel_path"]: d for d in all_files}
    unresolved_imports_by_path: dict[str, list[str]] = {}

    for descriptor in all_files:
        rel_path = descriptor["rel_path"]
        graph.add_file(rel_path)
        unresolved: list[str] = []
        try:
            imports = parse_file(descriptor)
            for imp in imports:
                resolved = resolve_import(imp, rel_path, all_rel_paths, root)
                if resolved:
                    graph.add_dependency(rel_path, resolved)
                else:
                    unresolved.append(imp)
        except ParseError as exc:
            error_reporter.record(exc, context=rel_path)
        unresolved_imports_by_path[rel_path] = unresolved

    return graph, file_map, unresolved_imports_by_path


def _select_files(
    root: Path,
    config: dict,
    graph: DependencyGraph,
    file_map: dict[str, dict],
) -> tuple[set[str], set[str], str | None]:
    all_rel_paths = set(file_map)
    scope = config.get("documentation_scope", "entry")
    if scope not in {"entry", "all"}:
        raise ConfigError("documentation_scope must be 'entry' or 'all'")
    # A2: distinguish an *explicitly specified* entry (via --entry or read from
    # existing docs) from auto-detection.  An explicit entry that cannot be
    # resolved or is absent from the scanned set must be a hard error — silently
    # falling back to documenting the whole repo turns a typo into an expensive
    # full-repo LLM run.  Auto-detection finding nothing keeps the original
    # "document all files" behaviour.
    explicit_entry = config.get("entry_file")
    entry = detect_entry_file(
        root,
        explicit_entry,
        config.get("auto_entry_candidates"),
    )
    if not entry:
        if explicit_entry:
            raise ConfigError(
                f"Entry file '{explicit_entry}' was not found in '{root}'. "
                "Provide an entry path inside the project root, or remove the "
                "entry setting to document all files."
            )
        return all_rel_paths, all_rel_paths, None

    # An explicit entry may resolve to a path outside the project root (e.g. an
    # absolute path or one with '..').  relative_to() would raise ValueError;
    # surface it as an actionable ConfigError instead.
    try:
        entry_rel = entry.relative_to(root).as_posix()
    except ValueError:
        if explicit_entry:
            raise ConfigError(
                f"Entry file '{explicit_entry}' resolves outside the project "
                f"root '{root}'. Provide an entry path inside the project."
            )
        return all_rel_paths, all_rel_paths, None

    if entry_rel not in file_map:
        if explicit_entry:
            raise ConfigError(
                f"Entry file '{entry_rel}' exists but was not in the scanned file "
                "set. It may be excluded by skip_dirs, ignore_paths, an "
                "unsupported extension, or max_file_size_kb. Adjust your "
                "configuration, or remove the entry setting to document all files."
            )
        logger.warning(
            "Entry file '%s' was not found in the scanned file set — "
            "documenting all %d file(s) instead.",
            entry_rel,
            len(all_rel_paths),
        )
        return all_rel_paths, all_rel_paths, None

    reachable = graph.reachable_dependencies(entry_rel) | {entry_rel}
    documented = all_rel_paths if scope == "all" else reachable

    # Entry scope excludes files not transitively imported from the entry. Warn
    # with a sample so that omission remains visible.
    excluded = all_rel_paths - reachable
    if excluded:
        sample = ", ".join(sorted(excluded)[:10])
        more = "" if len(excluded) <= 10 else f" (+{len(excluded) - 10} more)"
        logger.warning(
            "Entry reachability: %d of %d scanned file(s) are disconnected from "
            "entry '%s': %s%s. documentation_scope='%s' %s disconnected files.%s",
            len(excluded),
            len(all_rel_paths),
            entry_rel,
            sample,
            more,
            scope,
            "includes" if scope == "all" else "excludes",
            " Use --documentation-scope all for complete coverage."
            if scope == "entry"
            else "",
        )
    else:
        logger.info(
            "Entry file: %s (%d reachable project file(s); all scanned files reachable)",
            entry_rel,
            len(reachable),
        )
    return reachable, documented, entry_rel


def _graph_edges(graph: DependencyGraph, selected_rels: set[str]) -> list[dict]:
    edges: list[dict] = []
    for rel_path in sorted(selected_rels):
        for dependency in sorted(graph.dependencies_of(rel_path)):
            if dependency in selected_rels:
                edges.append({"from": rel_path, "to": dependency, "type": "internal_import"})
    return edges
