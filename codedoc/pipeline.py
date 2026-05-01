"""Main documentation pipeline for codedoc."""

from __future__ import annotations

from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from codedoc.agents.orchestrator import Orchestrator
from codedoc.bootstrap import ensure_codedoc_installed
from codedoc.core.db import CodeDocDB
from codedoc.core.graph import DependencyGraph, resolve_import
from codedoc.core.loader import load_config
from codedoc.core.output import write_outputs, write_summary
from codedoc.core.queue import ProcessingQueue
from codedoc.core.scanner import detect_entry_file, scan_files
from codedoc.llm.factory import create_provider
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import ErrorReporter, OutputError, ParseError
from codedoc.utils.logger import get_logger, set_level

logger = get_logger(__name__)


def run_pipeline(project_root: str | Path, config_overrides: dict | None = None) -> dict:
    """Run the full documentation pipeline on a project."""
    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    if not ensure_codedoc_installed(root):
        raise RuntimeError("codedoc is not importable in the current Python environment.")

    config = load_config(root, config_overrides)

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)

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
    selected_rels = _select_files(root, config, graph, file_map)

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

    if not process_rels:
        logger.info("All selected files are up-to-date. Nothing to document.")
        stats = {
            "checked": 0,
            "failed": 0,
            "skipped": len(selected_rels),
            "output_dir": str(output_dir),
        }
        write_summary(stats, output_dir, error_reporter.summary())
        error_reporter.flush()
        return stats

    queue = ProcessingQueue()
    for rel_path in graph.topological_order():
        if rel_path in process_rels:
            queue.add(file_map[rel_path])

    try:
        llm = create_provider(config)
        logger.info("LLM provider: %s", llm.provider_name)
    except Exception as exc:
        error_reporter.record(exc, context="LLM provider init")
        error_reporter.flush()
        raise

    orchestrator = Orchestrator(llm, parallel=config.get("parallel_agents", True))
    stats = {"checked": 0, "failed": 0, "skipped": skipped}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Documenting files...", total=len(process_rels))

        while True:
            descriptor = queue.next()
            if descriptor is None:
                break

            rel_path = descriptor["rel_path"]
            file_path: Path = descriptor["path"]
            progress.update(task, description=f"[cyan]{rel_path}[/cyan]")

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                imports = parse_file(descriptor)
                result = orchestrator.process(descriptor, content, imports)

                write_outputs(result, output_dir)
                db.mark_processed(rel_path, file_path, result)
                queue.mark_checked(rel_path)

                stats["checked"] += 1
                logger.info("[OK] %s", rel_path)

            except ParseError as exc:
                error_reporter.record(exc, context=rel_path)
                queue.mark_failed(rel_path, str(exc))
                stats["failed"] += 1

            except OutputError as exc:
                error_reporter.record(exc, context=rel_path)
                queue.mark_failed(rel_path, str(exc))
                stats["failed"] += 1

            except Exception as exc:
                error_reporter.record(exc, context=rel_path)
                queue.mark_failed(rel_path, str(exc))
                stats["failed"] += 1
                logger.warning("[FAIL] %s: %s", rel_path, exc)

            finally:
                progress.advance(task)

    stats["output_dir"] = str(output_dir)
    write_summary(stats, output_dir, error_reporter.summary())
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
) -> set[str]:
    all_rel_paths = set(file_map)
    entry = detect_entry_file(root, config.get("entry_file"))
    if not entry:
        return all_rel_paths

    entry_rel = entry.relative_to(root).as_posix()
    if entry_rel not in file_map:
        return all_rel_paths

    reachable = graph.reachable_dependencies(entry_rel) or {entry_rel}
    logger.info(
        "Entry file: %s (%d reachable project file(s))",
        entry_rel,
        len(reachable),
    )
    return reachable
