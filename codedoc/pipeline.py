"""Main documentation pipeline for codedoc."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from codedoc.agents.orchestrator import Orchestrator
from codedoc.bootstrap import ensure_codedoc_installed
from codedoc.core.db import CodeDocDB
from codedoc.core.graph import DependencyGraph, resolve_import
from codedoc.core.loader import load_config
from codedoc.core.output import write_project_outputs
from codedoc.core.queue import ProcessingQueue
from codedoc.core.scanner import detect_entry_file, scan_files
from codedoc.llm.factory import create_provider
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import AgentError, ErrorReporter, OutputError, ParseError
from codedoc.utils.logger import get_logger, set_level

logger = get_logger(__name__)


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

    config = load_config(root, config_overrides)

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)
    output_format = config.get("output_format", "json")
    logger.info("Output format: %s", output_format)

    output_dir = root / config["output_dir"]
    error_reporter = ErrorReporter(root / "error.log")

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

    db = CodeDocDB(root)
    changed_rels = {
        rel for rel in selected_rels if db.needs_processing(rel, file_map[rel]["path"])
    }
    if config.get("propagate_changes", True):
        process_rels = graph.affected_by_changes(changed_rels) & selected_rels
    else:
        process_rels = changed_rels

    skipped = len(selected_rels) - len(process_rels)
    if skipped > 0:
        logger.info("Incremental mode: skipping %d unchanged file(s)", skipped)

    reused = 0
    agent_rels: set[str] = set()
    for rel_path in graph.topological_order():
        if rel_path not in process_rels:
            continue
        if db.reuse_by_content_hash(file_map[rel_path]):
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
            db.documentation_records(
                selected_rels,
                file_map,
                graph.topological_order(),
            ),
            stats,
            output_dir,
            error_reporter.summary(),
            output_format,
            entry_rel,
            _graph_edges(graph, selected_rels),
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
        "File concurrency: max_parallel_files=%d retry_attempts=%d max_consecutive_failures=%d",
        max_workers,
        retry_attempts,
        max_consecutive_failures,
    )

    _process_agent_files(
        queue=queue,
        orchestrator=orchestrator,
        db=db,
        stats=stats,
        error_reporter=error_reporter,
        max_workers=max_workers,
        retry_attempts=retry_attempts,
        max_consecutive_failures=max_consecutive_failures,
    )

    stats["output_dir"] = str(output_dir)
    output_files = write_project_outputs(
        db.documentation_records(
            selected_rels,
            file_map,
            graph.topological_order(),
        ),
        stats,
        output_dir,
        error_reporter.summary(),
        output_format,
        entry_rel,
        _graph_edges(graph, selected_rels),
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
    db: CodeDocDB,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    retry_attempts: int,
    max_consecutive_failures: int,
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
            db,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
        )
        return

    consecutive_failures = 0
    failed_descriptors = []
    health_reported = False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Documenting files...", total=len(descriptors))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(_process_one_file, descriptor, orchestrator): descriptor
                for descriptor in descriptors
            }

            for future in concurrent.futures.as_completed(future_map):
                descriptor = future_map[future]
                rel_path = descriptor["rel_path"]
                progress.update(task, description=f"[cyan]{rel_path}[/cyan]")
                try:
                    result = future.result()
                    db.mark_processed(rel_path, descriptor["path"], result)
                    queue.mark_checked(rel_path)
                    stats["checked"] += 1
                    consecutive_failures = 0
                    logger.info("[OK] %s", rel_path)
                except Exception as exc:
                    failed_descriptors.append(descriptor)
                    consecutive_failures += 1
                    logger.warning("[RETRY] %s failed in parallel: %s", rel_path, exc)
                    if (
                        consecutive_failures >= max_consecutive_failures
                        and not health_reported
                    ):
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
                finally:
                    progress.advance(task)

    if failed_descriptors:
        logger.info(
            "Retrying %d failed file(s) sequentially for clearer errors.",
            len(failed_descriptors),
        )
        _process_files_sequentially(
            failed_descriptors,
            orchestrator,
            db,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
        )


def _process_files_sequentially(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    db: CodeDocDB,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    retry_attempts: int,
    max_consecutive_failures: int,
) -> None:
    consecutive_failures = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Documenting files...", total=len(descriptors))
        for descriptor in descriptors:
            rel_path = descriptor["rel_path"]
            progress.update(task, description=f"[cyan]{rel_path}[/cyan]")
            try:
                result = _process_one_file_with_retries(descriptor, orchestrator, retry_attempts)
                db.mark_processed(rel_path, descriptor["path"], result)
                queue.mark_checked(rel_path)
                stats["checked"] += 1
                consecutive_failures = 0
                logger.info("[OK] %s", rel_path)
            except (ParseError, OutputError, AgentError) as exc:
                error_reporter.record(exc, context=rel_path)
                queue.mark_failed(rel_path, str(exc))
                stats["failed"] += 1
                consecutive_failures += 1
            except Exception as exc:
                error_reporter.record(exc, context=rel_path)
                queue.mark_failed(rel_path, str(exc))
                stats["failed"] += 1
                consecutive_failures += 1
                logger.warning("[FAIL] %s: %s", rel_path, exc)
            finally:
                progress.advance(task)

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


def _process_one_file(descriptor: dict, orchestrator: Orchestrator) -> dict:
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
