"""File discovery, graph construction, and entry-based selection.

0.9.4 — extracted verbatim from ``codedoc.pipeline`` as part of the internal
decomposition.  This module owns:

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
from codedoc.core.resume import _resolve_live_backup_path
from codedoc.core.scanner import detect_entry_file
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import ConfigError, ErrorReporter, ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_entry_and_docs(root: Path, config: dict) -> None:
    """Auto-discover the entry point from a previously generated CodeDoc file."""
    if config.get("entry_file"):
        return

    raw_output = config.get("output_dir", "codedoc")
    p = Path(raw_output)

    if p.suffix.lower() in (".json", ".md"):
        candidate = (root / p) if not p.is_absolute() else p
        candidates: list[Path] = [candidate]
    else:
        out_dir = (root / p) if not p.is_absolute() else p
        json_filename = config.get("output_json_filename", "codedoc.json")
        md_filename = config.get("output_md_filename", "codedoc.md")
        output_format = config.get("output_format", "json")

        json_cand = out_dir / json_filename
        # 0.8.0: also probe the live backup sibling for MD-only format.
        live_backup_cand = _resolve_live_backup_path(out_dir, output_format, json_filename, md_filename)
        md_cand = out_dir / md_filename
        candidates = [json_cand]
        if live_backup_cand.resolve() != json_cand.resolve():
            candidates.append(live_backup_cand)
        if output_format in ("md", "both"):
            candidates.append(md_cand)

    for candidate in candidates:
        if candidate.exists():
            try:
                meta = read_codedoc_meta(candidate)
            except ConfigError:
                continue
            entry = meta.get("entry_file")
            if entry:
                logger.info(
                    "Resuming: entry '%s' read from '%s'", entry, candidate.name
                )
                config["entry_file"] = entry
            return

        if candidate.suffix.lower() == ".json":
            md_sibling = candidate.with_suffix(".md")
            if md_sibling.exists():
                try:
                    meta = read_codedoc_meta(md_sibling)
                    entry = meta.get("entry_file")
                    if entry:
                        logger.info(
                            "Cross-format resume: entry '%s' read from '%s'",
                            entry,
                            md_sibling.name,
                        )
                        config["entry_file"] = entry
                    return
                except ConfigError:
                    pass

    # No existing output was found and no --entry was provided.
    # Leave config["entry_file"] unset so _select_files() can call
    # detect_entry_file() with the configured auto-entry candidates.
    # If auto-detection also finds nothing, the existing "process all
    # supported files" behaviour from detect_entry_file() applies.
    logger.info(
        "No existing CodeDoc output found and no --entry provided. "
        "Auto-detection via detect_entry_file() will be attempted."
    )


def _build_graph(
    all_files: list[dict],
    root: Path,
    error_reporter: ErrorReporter,
) -> tuple[DependencyGraph, dict[str, dict]]:
    graph = DependencyGraph()
    all_rel_paths = {d["rel_path"] for d in all_files}
    file_map = {d["rel_path"]: d for d in all_files}

    for descriptor in all_files:
        graph.add_file(descriptor["rel_path"])
        try:
            imports = parse_file(descriptor)
            for imp in imports:
                resolved = resolve_import(imp, descriptor["rel_path"], all_rel_paths, root)
                if resolved:
                    graph.add_dependency(descriptor["rel_path"], resolved)
        except ParseError as exc:
            error_reporter.record(exc, context=descriptor["rel_path"])

    return graph, file_map


def _select_files(
    root: Path,
    config: dict,
    graph: DependencyGraph,
    file_map: dict[str, dict],
) -> tuple[set[str], str | None]:
    all_rel_paths = set(file_map)
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
        return all_rel_paths, None

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
        return all_rel_paths, None

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
        return all_rel_paths, None

    reachable = graph.reachable_dependencies(entry_rel) or {entry_rel}

    # Visibility for the known entry-reachability limitation (A1): files that are
    # not transitively imported from the entry are NOT documented.  Previously
    # this exclusion was silent.  We now warn loudly and list a sample so the
    # omission is never invisible.  The structural fix (how selection should
    # behave) is deferred to 0.10.0; this only surfaces the current behaviour.
    excluded = all_rel_paths - reachable
    if excluded:
        sample = ", ".join(sorted(excluded)[:10])
        more = "" if len(excluded) <= 10 else f" (+{len(excluded) - 10} more)"
        logger.warning(
            "Entry-based selection: %d of %d scanned file(s) are NOT reachable "
            "from entry '%s' and will NOT be documented: %s%s. "
            "Files not imported (transitively) from the entry are skipped — this "
            "is a known limitation. To document every scanned file, run without "
            "--entry. Full reachability handling is planned for 0.10.0.",
            len(excluded),
            len(all_rel_paths),
            entry_rel,
            sample,
            more,
        )
    else:
        logger.info(
            "Entry file: %s (%d reachable project file(s); all scanned files reachable)",
            entry_rel,
            len(reachable),
        )
    return reachable, entry_rel


def _graph_edges(graph: DependencyGraph, selected_rels: set[str]) -> list[dict]:
    edges: list[dict] = []
    for rel_path in sorted(selected_rels):
        for dependency in sorted(graph.dependencies_of(rel_path)):
            if dependency in selected_rels:
                edges.append({"from": rel_path, "to": dependency, "type": "internal_import"})
    return edges
