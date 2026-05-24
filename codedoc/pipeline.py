"""Main documentation pipeline for codedoc."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

from codedoc.agents.orchestrator import Orchestrator
from codedoc.bootstrap import ensure_codedoc_installed
from codedoc.core.db import compute_file_hash
from codedoc.core.graph import DependencyGraph, resolve_import
from codedoc.core.loader import load_config
from codedoc.core.output import write_project_outputs
from codedoc.core.project_view import markdown_to_view, read_codedoc_meta
from codedoc.core.queue import ProcessingQueue
from codedoc.core.scanner import detect_entry_file, scan_files
from codedoc.llm.factory import create_provider
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import AgentError, ConfigError, ErrorReporter, OutputError, ParseError
from codedoc.utils.logger import get_logger, set_level

logger = get_logger(__name__)


def _load_existing_file_docs(
    output_dir: Path,
    json_filename: str,
    md_filename: str = "codedoc.md",
) -> dict[str, dict]:
    """Load per-file documentation from an existing public output file.

    Tries JSON first (full fidelity, includes ``_deps``).  When JSON is absent
    — e.g. the project has only ever been run with ``--format md`` — falls back
    to the MD file so that incremental hash checks still work.  The MD fallback
    checks two candidates in order:

    1. Same-stem sibling of ``json_filename`` (``claude.json`` → ``claude.md``)
    2. The explicitly configured ``md_filename`` (default ``codedoc.md``)

    Returns a dict mapping rel_path → file record dict.
    """
    json_path = output_dir / json_filename
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return {
                f["path"]: f
                for f in data.get("files", [])
                if isinstance(f, dict) and f.get("path")
            }
        except Exception:
            return {}

    # JSON not found — try MD fallback (MD-only users and cross-format resume).
    stem_sibling = output_dir / (Path(json_filename).stem + ".md")
    configured_md = output_dir / md_filename
    md_candidates: list[Path] = [stem_sibling]
    if configured_md != stem_sibling:
        md_candidates.append(configured_md)

    for md_path in md_candidates:
        if md_path.exists():
            try:
                return _load_existing_file_docs_from_md(md_path)
            except Exception:
                pass

    return {}


def _load_existing_file_docs_from_md(md_path: Path) -> dict[str, dict]:
    """Parse an existing MD output file into a file-record dict.

    Hashes are read from the ``file_hashes`` map embedded in the
    ``<!-- codedoc-ai: ... -->`` metadata comment (written since 0.7.0).
    Older MD files without that field produce records with empty hashes,
    causing all files to re-process once — after which the new MD will
    carry hashes and incremental will work normally.

    Note: ``_deps`` is absent (MD does not store raw dependency analysis),
    so catalog_updates for unchanged files are not reconstructed.  External
    dependencies are preserved via the ``links`` block.
    """
    content = md_path.read_text(encoding="utf-8-sig", errors="replace")

    # Extract file_hashes from the embedded metadata comment.
    file_hashes: dict[str, str] = {}
    try:
        meta = read_codedoc_meta(md_path)
        file_hashes = meta.get("file_hashes") or {}
    except ConfigError:
        pass  # Old MD without file_hashes — all files will re-process this run.

    # Parse file records from the MD content.
    view = markdown_to_view(content)
    result: dict[str, dict] = {}
    for file_record in view.get("files", []):
        path = file_record.get("path")
        if path:
            result[path] = {**file_record, "hash": file_hashes.get(path, "")}
    return result


def _public_record_to_doc(file_record: dict) -> dict:
    """Convert a public JSON file record back to a documentation dict for pipeline use."""
    links = file_record.get("links", {})
    deps = file_record.get("_deps") or {
        "external": links.get("external_dependencies", []),
    }
    doc = {
        "file_path": file_record.get("path", ""),
        "language": file_record.get("language", ""),
        "imports": file_record.get("imports", []),
        "description": file_record.get("description", ""),
        "role_in_system": file_record.get("role_in_system", ""),
        "key_concepts": file_record.get("key_concepts", []),
        "functions": file_record.get("functions", []),
        "classes": file_record.get("classes", []),
        "exports": file_record.get("exports", []),
        "usage_example": file_record.get("usage_example", ""),
        "dependencies_analysis": deps,
    }
    return {k: v for k, v in doc.items() if v not in (None, "", [], {}, {"external": [], "internal": []})}


def _build_documentation_records(
    rel_paths: set,
    file_map: dict,
    ordered_paths: list,
    existing_docs: dict,
    new_results: dict,
) -> list[dict]:
    """Build documentation records for write_project_outputs.

    For files processed this run: use new_results (raw LLM dict or reused public record).
    For unchanged files: use existing_docs (from public JSON).
    """
    records = []
    for rel_path in ordered_paths:
        if rel_path not in rel_paths:
            continue

        if rel_path in new_results:
            result = new_results[rel_path]
            if isinstance(result, dict) and result.get("path") and not result.get("file_path"):
                # Public JSON file record (from reuse)
                doc = _public_record_to_doc(result)
            else:
                # Raw LLM result
                doc = dict(result)
        elif rel_path in existing_docs:
            doc = _public_record_to_doc(existing_docs[rel_path])
        else:
            continue

        # Hash: compute fresh for processed files; use stored hash for unchanged files
        if rel_path in new_results:
            descriptor = file_map.get(rel_path, {})
            try:
                file_hash = compute_file_hash(descriptor["path"]) if descriptor.get("path") else ""
            except Exception:
                file_hash = existing_docs.get(rel_path, {}).get("hash", "")
        else:
            file_hash = existing_docs.get(rel_path, {}).get("hash", "")

        records.append({
            "hash": file_hash,
            "file_path": rel_path,
            "language": doc.get("language", ""),
            "documentation": doc,
        })
    return records


def _remove_legacy_db(output_dir: Path) -> None:
    """Remove codedoc_db.json left over from earlier versions."""
    legacy = output_dir / "codedoc_db.json"
    if legacy.exists():
        try:
            legacy.unlink()
            logger.info("Removed legacy codedoc_db.json (no longer used since 0.6.4)")
        except Exception:
            pass


def run_pipeline(
    project_root: str | Path | dict | None = ".",
    config_overrides: dict | None = None,
) -> dict:
    """Run the full documentation pipeline on a project.

    ``project_root`` defaults to the current working directory. For convenience,
    callers may pass the config dict as the first argument:

        run_pipeline({"output_format": "md"})
    """
    if isinstance(project_root, dict) and config_overrides is None:
        config_overrides = project_root
        project_root = "."
    if project_root is None:
        project_root = "."

    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    if not ensure_codedoc_installed(root):
        raise RuntimeError("codedoc is not importable in the current Python environment.")

    if config_overrides is None:
        config_overrides = {}

    config = load_config(root, config_overrides)

    # Resolve entry point from previous docs *after* loading config so that
    # entry_file set in codedoc.config.json is also visible.
    _resolve_entry_and_docs(root, config)

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)
    output_format = config.get("output_format", "json")
    logger.info("Output format: %s", output_format)

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_db(output_dir)
    error_reporter = ErrorReporter(root / "error.log")

    json_filename = config.get("output_json_filename", "codedoc.json")
    md_filename = config.get("output_md_filename", "codedoc.md")
    existing_docs = _load_existing_file_docs(output_dir, json_filename, md_filename)
    docs_by_hash: dict[str, dict] = {
        doc["hash"]: doc for doc in existing_docs.values() if doc.get("hash")
    }

    all_files = scan_files(
        root,
        supported_extensions=config["supported_extensions"],
        max_file_size_kb=config["max_file_size_kb"],
        skip_dirs=config.get("skip_dirs"),
        ignore_paths=config.get("ignore_paths"),
    )
    if not all_files:
        logger.warning("No supported files found in %s. Done.", root)
        return {"checked": 0, "failed": 0, "skipped": 0, "output_dir": str(output_dir)}

    graph, file_map = _build_graph(all_files, root, error_reporter)
    selected_rels, entry_rel = _select_files(root, config, graph, file_map)

    changed_rels = {
        rel for rel in selected_rels
        if compute_file_hash(file_map[rel]["path"]) != existing_docs.get(rel, {}).get("hash", "")
    }
    if config.get("propagate_changes", True):
        process_rels = graph.affected_by_changes(changed_rels) & selected_rels
    else:
        process_rels = changed_rels

    skipped = len(selected_rels) - len(process_rels)
    if skipped > 0:
        logger.info("Incremental mode: skipping %d unchanged file(s)", skipped)

    new_results: dict[str, dict] = {}
    reused = 0
    agent_rels: set[str] = set()
    for rel_path in graph.topological_order():
        if rel_path not in process_rels:
            continue
        descriptor = file_map[rel_path]
        content_hash = compute_file_hash(descriptor["path"])
        if content_hash in docs_by_hash:
            source_doc = docs_by_hash[content_hash]
            new_results[rel_path] = source_doc
            logger.info(
                "Reusing cached documentation for %s from identical content in %s",
                rel_path,
                source_doc.get("path", "unknown"),
            )
            reused += 1
        else:
            agent_rels.add(rel_path)

    if not agent_rels:
        logger.info("All selected files are up-to-date or reused from cached content.")
        stats = {
            "checked": 0,
            "failed": 0,
            "skipped": skipped,
            "reused": reused,
            "output_dir": str(output_dir),
        }
        output_files = write_project_outputs(
            _build_documentation_records(
                selected_rels,
                file_map,
                graph.topological_order(),
                existing_docs,
                new_results,
            ),
            stats,
            output_dir,
            error_reporter.summary(),
            output_format,
            entry_rel,
            _graph_edges(graph, selected_rels),
            json_filename=config.get("output_json_filename", "codedoc.json"),
            md_filename=config.get("output_md_filename", "codedoc.md"),
        )
        stats["output_files"] = [str(path) for path in output_files if path]
        error_reporter.flush()
        return stats

    queue = ProcessingQueue()
    for rel_path in graph.topological_order():
        if rel_path in agent_rels:
            queue.add(file_map[rel_path])

    try:
        llm = create_provider(config)
        logger.info("LLM provider: %s", llm.provider_name)
    except Exception as exc:
        error_reporter.record(exc, context="LLM provider init")
        error_reporter.flush()
        raise

    orchestrator = Orchestrator(llm, parallel=config.get("parallel_agents", True))
    stats = {"checked": 0, "failed": 0, "skipped": skipped, "reused": reused}
    max_workers = min(config.get("max_parallel_files", 5), len(agent_rels)) or 1
    retry_attempts = config.get("file_retry_attempts", 1)
    max_consecutive_failures = config.get("max_consecutive_failures", 5)
    logger.info(
        "File processing plan: %d file(s), up to %d file(s) in parallel, "
        "provider=%s, retry_attempts=%d, max_consecutive_failures=%d",
        len(agent_rels),
        max_workers,
        llm.provider_name,
        retry_attempts,
        max_consecutive_failures,
    )

    _process_agent_files(
        queue=queue,
        orchestrator=orchestrator,
        stats=stats,
        error_reporter=error_reporter,
        max_workers=max_workers,
        retry_attempts=retry_attempts,
        max_consecutive_failures=max_consecutive_failures,
        new_results=new_results,
    )

    stats["output_dir"] = str(output_dir)
    output_files = write_project_outputs(
        _build_documentation_records(
            selected_rels,
            file_map,
            graph.topological_order(),
            existing_docs,
            new_results,
        ),
        stats,
        output_dir,
        error_reporter.summary(),
        output_format,
        entry_rel,
        _graph_edges(graph, selected_rels),
        json_filename=config.get("output_json_filename", "codedoc.json"),
        md_filename=config.get("output_md_filename", "codedoc.md"),
    )
    stats["output_files"] = [str(path) for path in output_files if path]
    error_reporter.flush()

    logger.info(
        "Done. checked=%d failed=%d skipped=%d output=%s",
        stats["checked"],
        stats["failed"],
        stats["skipped"],
        output_dir,
    )

    if error_reporter.has_errors():
        logger.warning(
            "%d error(s) occurred. See error.log for details.",
            error_reporter.error_count(),
        )

    return stats


def _process_agent_files(
    queue: ProcessingQueue,
    orchestrator: Orchestrator,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
) -> None:
    descriptors = []
    while True:
        descriptor = queue.next()
        if descriptor is None:
            break
        descriptors.append(descriptor)

    if max_workers <= 1 or len(descriptors) <= 1:
        _process_files_sequentially(
            descriptors,
            orchestrator,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
            new_results,
        )
        return

    consecutive_failures = 0
    failed_descriptors = []
    health_reported = False

    total = len(descriptors)
    completed = 0
    logger.info(
        "Starting parallel documentation: %d file(s), max_parallel_files=%d, provider=%s",
        total,
        max_workers,
        orchestrator.llm.provider_name,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_process_one_file, descriptor, orchestrator): descriptor
            for descriptor in descriptors
        }

        for future in concurrent.futures.as_completed(future_map):
            descriptor = future_map[future]
            rel_path = descriptor["rel_path"]
            try:
                result = future.result()
                new_results[rel_path] = result
                queue.mark_checked(rel_path)
                stats["checked"] += 1
                consecutive_failures = 0
                completed += 1
                _log_file_progress("OK", rel_path, completed, total)
            except Exception as exc:
                failed_descriptors.append(descriptor)
                consecutive_failures += 1
                completed += 1
                _log_file_progress("RETRY", rel_path, completed, total, str(exc))
                if consecutive_failures >= max_consecutive_failures and not health_reported:
                    health_reported = True
                    _cancel_pending(future_map)
                    error_reporter.record(
                        RuntimeError(
                            "Parallel processing saw "
                            f"{consecutive_failures} consecutive file failures. "
                            "The API/provider may be unavailable or rate-limited; "
                            "failed files will be retried sequentially for clearer diagnostics."
                        ),
                        context="parallel processing health check",
                    )

    if failed_descriptors:
        logger.info(
            "Retrying %d failed file(s) sequentially for clearer errors.",
            len(failed_descriptors),
        )
        _process_files_sequentially(
            failed_descriptors,
            orchestrator,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
            new_results,
        )


def _process_files_sequentially(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
) -> None:
    consecutive_failures = 0

    total = len(descriptors)
    logger.info(
        "Starting sequential documentation: %d file(s), provider=%s",
        total,
        orchestrator.llm.provider_name,
    )

    for index, descriptor in enumerate(descriptors, start=1):
        rel_path = descriptor["rel_path"]
        try:
            result = _process_one_file_with_retries(descriptor, orchestrator, retry_attempts)
            new_results[rel_path] = result
            queue.mark_checked(rel_path)
            stats["checked"] += 1
            consecutive_failures = 0
            _log_file_progress("OK", rel_path, index, total)
        except (ParseError, OutputError, AgentError) as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, str(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            _log_file_progress("FAIL", rel_path, index, total, str(exc))
        except Exception as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, str(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            _log_file_progress("FAIL", rel_path, index, total, str(exc))

        if consecutive_failures >= max_consecutive_failures:
            error_reporter.record(
                RuntimeError(
                    "Stopping sequential processing after "
                    f"{consecutive_failures} consecutive file failures. "
                    "Check API credentials, provider availability, model name, "
                    "rate limits, and network connectivity."
                ),
                context="sequential processing health check",
            )
            break


def _process_one_file_with_retries(
    descriptor: dict,
    orchestrator: Orchestrator,
    retry_attempts: int,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retry_attempts + 1):
        try:
            return _process_one_file(descriptor, orchestrator)
        except Exception as exc:
            last_error = exc
            if attempt < retry_attempts:
                logger.info(
                    "Retrying %s (%d/%d): %s",
                    descriptor["rel_path"],
                    attempt + 1,
                    retry_attempts,
                    exc,
                )
    raise last_error or RuntimeError("Unknown processing failure")


def _log_file_progress(
    status: str,
    rel_path: str,
    completed: int,
    total: int,
    detail: str | None = None,
) -> None:
    percent = int((completed / total) * 100) if total else 100
    remaining = max(total - completed, 0)
    message = "[%s] %s | %d/%d complete (%d%%), %d remaining"
    if detail:
        logger.warning(
            message + " | %s",
            status,
            rel_path,
            completed,
            total,
            percent,
            remaining,
            detail,
        )
    else:
        logger.info(message, status, rel_path, completed, total, percent, remaining)


def _process_one_file(descriptor: dict, orchestrator: Orchestrator) -> dict:
    rel_path = descriptor["rel_path"]
    logger.info("[START] %s | provider=%s", rel_path, orchestrator.llm.provider_name)
    file_path: Path = descriptor["path"]
    content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    imports = parse_file(descriptor)
    result = orchestrator.process(descriptor, content, imports)
    errors = _agent_errors(result)
    if errors:
        raise AgentError(orchestrator.__class__.__name__, descriptor["rel_path"], "; ".join(errors))
    return result


def _agent_errors(result: dict) -> list[str]:
    errors = []
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if isinstance(value, dict) and value.get("error"):
            agent = value.get("agent", key)
            errors.append(f"{agent}: {value['error']}")
    return errors


def _cancel_pending(future_map: dict) -> None:
    for future in future_map:
        if not future.done():
            future.cancel()


def _resolve_entry_and_docs(root: Path, config: dict) -> None:
    """
    Auto-discover the entry point from a previously generated CodeDoc file.

    Called *after* load_config so that ``entry_file`` from any source
    (CLI flag, config JSON, environment variable) is already present.
    Mutates ``config`` in-place.

    Rules
    -----
    1. If entry_file is already set in config, return immediately — nothing to resolve.
    2. Determine which docs file to examine:
       - If ``output_dir`` in config contains a file extension (.json / .md), treat it
         as a direct path to the docs file.
       - Otherwise look for ``codedoc.json`` or ``codedoc.md`` inside the output
         directory (default: ``<root>/codedoc/``).
    3. If a docs file is found:
       - Parse its metadata via ``read_codedoc_meta`` (raises ConfigError if invalid).
       - Populate ``config["entry_file"]`` from the metadata.
    4. If *no* docs file is found and no entry_file is configured, raise a
       ConfigError explaining that the entry point is mandatory.
    """
    # Already have an explicit entry from any config source — nothing to resolve.
    if config.get("entry_file"):
        return

    raw_output = config.get("output_dir", "codedoc")
    p = Path(raw_output)

    if p.suffix.lower() in (".json", ".md"):
        # User gave a specific file path like "docs/report.json"
        candidate = (root / p) if not p.is_absolute() else p
        candidates: list[Path] = [candidate]
    else:
        # Directory — probe for the configured filenames in order.
        out_dir = (root / p) if not p.is_absolute() else p
        candidates = []
        if config.get("output_format") in ("json", "both"):
            candidates.append(out_dir / config.get("output_json_filename", "codedoc.json"))
        if config.get("output_format") in ("md", "both"):
            candidates.append(out_dir / config.get("output_md_filename", "codedoc.md"))

    for candidate in candidates:
        if candidate.exists():
            # read_codedoc_meta raises ConfigError when metadata is absent/corrupt
            meta = read_codedoc_meta(candidate)
            entry = meta.get("entry_file")
            if entry:
                logger.info(
                    "Resuming: entry '%s' read from '%s'", entry, candidate.name
                )
                config["entry_file"] = entry
            return  # docs file found (even if entry was empty — handled by validate)

        # Cross-format resume: JSON candidate missing → check same-stem MD sibling.
        # e.g. --output codedoc/claude.json after a previous --format md run that
        # wrote codedoc/claude.md.
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
                    return  # sibling found — existing docs loaded separately via _load_existing_file_docs
                except ConfigError:
                    pass  # sibling exists but is not a valid CodeDoc file — keep searching

    raise ConfigError(
        "No entry point specified and no existing CodeDoc documentation was found.\n\n"
        "For a first run, provide the entry file:\n"
        "  codedoc run --entry src/main.py\n\n"
        "For subsequent runs, point to your previously generated file:\n"
        "  codedoc run --output path/to/codedoc.json\n"
        "or run from the same directory so the default codedoc/ folder can be "
        "checked automatically."
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
    entry = detect_entry_file(root, config.get("entry_file"))
    if not entry:
        return all_rel_paths, None

    entry_rel = entry.relative_to(root).as_posix()
    if entry_rel not in file_map:
        logger.warning(
            "Entry file '%s' was not found in the scanned file set — "
            "documenting all %d file(s) instead. "
            "Check --ignore flags, skip_dirs config, and supported_extensions.",
            entry_rel,
            len(all_rel_paths),
        )
        return all_rel_paths, None

    reachable = graph.reachable_dependencies(entry_rel) or {entry_rel}
    logger.info(
        "Entry file: %s (%d reachable project file(s))",
        entry_rel,
        len(reachable),
    )
    return reachable, entry_rel


def _graph_edges(graph: DependencyGraph, selected_rels: set[str]) -> list[dict]:
    edges: list[dict] = []
    for rel_path in sorted(selected_rels):
        for dependency in sorted(graph.dependencies_of(rel_path)):
            if dependency in selected_rels:
                edges.append(
                    {
                        "from": rel_path,
                        "to": dependency,
                        "type": "internal_import",
                    }
                )
    return edges
