"""Main documentation pipeline for codedoc.

0.8.0 changes
-------------
- **Always-on live JSON backup** (Work Item 1): ``SafeWriter`` is now the
  default recorder for every run.  ``--safe-mode`` is a deprecated no-op.
  The live backup is created before AI work starts and finalized (banner
  removed) by ``write_project_outputs`` on clean completion.
- **Worker-thread recording** (Work Item 2): in parallel mode, ``record()``
  is called inside the worker via ``_process_and_record()``, so a Ctrl-C or
  crash never discards a file whose AI work already completed.
- **Live backup path helper** (Work Item 1): ``_resolve_live_backup_path()``
  centralises path logic so the MD-sibling case is handled consistently
  across writer creation, ownership checks, resume, stats, and cleanup.
- **error.log in output_dir** (Work Item 4): ``ErrorReporter`` is now
  initialised with ``output_dir / "error.log"`` instead of ``root / "error.log"``.
- **Error stats always set** (Work Item 4): ``stats["error_log"]``,
  ``stats["issues_recorded"]``, and ``stats["live_backup_path"]`` are
  populated on every return path.
- **Checkpoint migration** (Work Item 1): ``Checkpoint`` entries are loaded
  as a fallback when no live backup exists, then retired after the first
  successful run.
- **Rate-limit ladder** (Work Item 3): ``_is_rate_limit_error()``,
  ``_build_default_ladder()``, and ladder-based retry batches implemented in
  ``_process_agent_files()``.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any

from codedoc.agents.orchestrator import Orchestrator
from codedoc.bootstrap import ensure_codedoc_installed
from codedoc.core.checkpoint import Checkpoint
from codedoc.core.db import compute_file_hash
from codedoc.core.safe_writer import SafeWriter
from codedoc.core.graph import DependencyGraph, resolve_import
from codedoc.core.loader import load_config
from codedoc.core.output import BUILD_FILENAME, write_project_outputs
from codedoc.core.project_view import markdown_to_view, read_codedoc_meta
from codedoc.core.queue import ProcessingQueue
from codedoc.core.scanner import detect_entry_file, scan_files
from codedoc.llm.factory import create_provider
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import AgentError, ConfigError, ErrorReporter, LLMError, OutputError, ParseError
from codedoc.utils.logger import get_logger, set_level

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit detection helpers (Work Item 3)
# ---------------------------------------------------------------------------

_RATE_LIMIT_SIGNALS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "tokens per min",
    "tpm",
    "quota",
    "resource_exhausted",   # Gemini / Google AI
    "overloaded",           # Anthropic
    "529",                  # Anthropic overloaded
    "503",
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if *exc* or any cause in its chain is a rate-limit signal.

    Inspects ``str(exc)`` and walks ``__cause__`` / ``__context__`` so that
    provider signals are not hidden by wrapper exceptions.
    """
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current).lower()
        if any(sig in msg for sig in _RATE_LIMIT_SIGNALS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _build_default_ladder(max_p: int) -> list[int]:
    """Build the default parallelism step-down ladder for *max_p* workers."""
    if max_p <= 1:
        return [1]
    if max_p == 2:
        return [2, 1]
    mid = max(2, max_p // 2)
    ladder: list[int] = [max_p]
    if mid < max_p:
        ladder.append(mid)
    if 1 not in ladder:
        ladder.append(1)
    return ladder


def _parse_retry_after(exc: BaseException) -> float | None:
    """Extract a Retry-After delay (seconds) from the exception message chain."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current)
        m = re.search(r"try again in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"retry.after[:\s]+([\d.]+)", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        current = current.__cause__ or current.__context__
    return None


# ---------------------------------------------------------------------------
# Live backup path helper (Work Item 1)
# ---------------------------------------------------------------------------

def _resolve_live_backup_path(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
) -> Path:
    """Return the absolute live JSON backup path for the given output scenario.

    | output_format | md_filename    | json_filename | Live backup path         |
    |---------------|----------------|---------------|--------------------------|
    | json / both   | (any)          | codedoc.json  | output_dir/codedoc.json  |
    | md            | codedoc.md     | (ignored)     | output_dir/codedoc.json  |
    | md            | report.md      | (ignored)     | output_dir/report.json   |

    For MD-only runs the backup is a JSON sibling of the Markdown file,
    derived from ``md_filename`` stem (not from ``json_filename``, which
    ``_resolve_output_spec`` leaves as the default ``codedoc.json`` even when
    the user requested ``--output docs/report.md``).
    """
    if output_format == "md":
        md_stem = Path(md_filename).stem
        return output_dir / f"{md_stem}.json"
    return output_dir / json_filename


# ---------------------------------------------------------------------------
# Existing-docs loader
# ---------------------------------------------------------------------------

def _load_existing_file_docs(
    output_dir: Path,
    json_filename: str,
    md_filename: str = "codedoc.md",
    live_backup_path: Path | None = None,
) -> dict[str, dict]:
    """Load per-file documentation from existing output files.

    Priority order
    --------------
    1. **Live backup path** (e.g. ``report.json`` for ``--output report.md``)
       when it differs from the default JSON path.  This handles the named-MD
       case where ``_resolve_output_spec`` leaves ``output_json_filename`` as
       ``codedoc.json`` even though the live backup is ``report.json``.
    2. **Final JSON output** (``json_filename``).  Covers the JSON/both case
       and the default-MD case where the live backup IS ``codedoc.json``.
    3. **Stale build file** (``.codedoc_build.json``).  Migration fallback for
       0.7.x interrupted MD runs.  Overlaid only when newer than the JSON.
    4. **Markdown fallback** — same-stem sibling or configured ``md_filename``.

    Returns a dict mapping rel_path → file record dict.
    """
    existing: dict[str, dict] = {}

    # 1. Live backup when it differs from the standard JSON path (named-MD case).
    json_path = output_dir / json_filename
    if live_backup_path and live_backup_path.resolve() != json_path.resolve():
        if live_backup_path.exists():
            try:
                data = json.loads(live_backup_path.read_text(encoding="utf-8"))
                existing = {
                    f["path"]: f
                    for f in data.get("files", [])
                    if isinstance(f, dict) and f.get("path")
                }
                if existing:
                    logger.info(
                        "Loaded %d existing record(s) from live backup '%s'.",
                        len(existing),
                        live_backup_path.name,
                    )
                    return existing
            except Exception:
                pass

    # 2. Final JSON output (also the live backup for JSON/both and default-MD runs).
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            existing = {
                f["path"]: f
                for f in data.get("files", [])
                if isinstance(f, dict) and f.get("path")
            }
        except Exception:
            pass

    # 3. Stale 0.7.x build file migration fallback.
    build_path = output_dir / BUILD_FILENAME
    if build_path.exists():
        try:
            build_is_stale = (
                json_path.exists()
                and build_path.stat().st_mtime < json_path.stat().st_mtime
            )
            if build_is_stale:
                logger.info(
                    "Build file '%s' is older than '%s' — treating as stale and removing it.",
                    BUILD_FILENAME,
                    json_filename,
                )
                try:
                    build_path.unlink()
                except Exception:
                    pass
            else:
                data = json.loads(build_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("_codedoc"), dict):
                    build_files = {
                        f["path"]: f
                        for f in data.get("files", [])
                        if isinstance(f, dict) and f.get("path")
                    }
                    if build_files:
                        logger.info(
                            "Build file '%s' found (%d record(s)) — merging (migration from 0.7.x).",
                            BUILD_FILENAME,
                            len(build_files),
                        )
                        existing.update(build_files)
        except Exception:
            pass

    if existing:
        return existing

    # 4. Markdown fallback.
    stem_sibling = output_dir / (Path(json_filename).stem + ".md")
    configured_md = output_dir / md_filename
    md_candidates: list[Path] = [stem_sibling]
    if configured_md != stem_sibling:
        md_candidates.append(configured_md)
    # Also check the live backup sibling when named-MD
    if live_backup_path:
        md_sibling = live_backup_path.with_suffix(".md")
        if md_sibling not in md_candidates:
            md_candidates.insert(0, md_sibling)

    for md_path in md_candidates:
        if md_path.exists():
            try:
                return _load_existing_file_docs_from_md(md_path)
            except Exception:
                pass

    return {}


def _load_existing_file_docs_from_md(md_path: Path) -> dict[str, dict]:
    """Parse an existing MD output file into a file-record dict."""
    content = md_path.read_text(encoding="utf-8-sig", errors="replace")

    file_hashes: dict[str, str] = {}
    try:
        meta = read_codedoc_meta(md_path)
        file_hashes = meta.get("file_hashes") or {}
    except ConfigError:
        pass

    view = markdown_to_view(content)
    result: dict[str, dict] = {}
    for file_record in view.get("files", []):
        path = file_record.get("path")
        if path:
            result[path] = {**file_record, "hash": file_hashes.get(path, "")}
    return result


def _public_record_to_doc(file_record: dict) -> dict:
    """Convert a public JSON file record back to a documentation dict."""
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
    """Build documentation records for write_project_outputs."""
    records = []
    for rel_path in ordered_paths:
        if rel_path not in rel_paths:
            continue

        if rel_path in new_results:
            result = new_results[rel_path]
            if isinstance(result, dict) and result.get("path") and not result.get("file_path"):
                doc = _public_record_to_doc(result)
            else:
                doc = dict(result)
        elif rel_path in existing_docs:
            doc = _public_record_to_doc(existing_docs[rel_path])
        else:
            continue

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


# ---------------------------------------------------------------------------
# Worker wrapper (Work Item 2)
# ---------------------------------------------------------------------------

def _process_and_record(
    descriptor: dict,
    orchestrator: Orchestrator,
    recorder: SafeWriter,
) -> dict:
    """Process one file and record it in the live backup from the worker thread.

    Recording happens here — inside the worker — so a Ctrl-C or crash that
    interrupts the main ``as_completed`` collection loop never discards a
    file whose AI work already completed.

    If ``_process_one_file`` raises for any reason (rate-limit, parse failure,
    model error), the exception propagates out of this function unchanged and
    ``recorder.record()`` is NOT called.  The batch-level code catches the
    future's exception and classifies it as rate-limit or non-rate-limit.
    """
    result = _process_one_file(descriptor, orchestrator)
    recorder.record(
        descriptor["rel_path"],
        result,
        _safe_file_hash(descriptor.get("path")),
    )
    return result


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    project_root: str | Path | dict | None = ".",
    config_overrides: dict | None = None,
) -> dict:
    """Run the full documentation pipeline on a project."""
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
    _resolve_entry_and_docs(root, config)

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)
    output_format = config.get("output_format", "json")
    logger.info("Output format: %s", output_format)

    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_db(output_dir)

    # 0.8.0: error.log lives in the output directory, not the project root.
    error_reporter = ErrorReporter(output_dir / "error.log")

    json_filename = config.get("output_json_filename", "codedoc.json")
    md_filename = config.get("output_md_filename", "codedoc.md")
    live_backup_path = _resolve_live_backup_path(output_dir, output_format, json_filename, md_filename)

    existing_docs = _load_existing_file_docs(
        output_dir, json_filename, md_filename, live_backup_path
    )
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
        return {
            "checked": 0,
            "failed": 0,
            "skipped": 0,
            "output_dir": str(output_dir),
            "live_backup_path": None,
            "error_log": None,
            "issues_recorded": 0,
            "rate_limit_warnings": [],
        }

    graph, file_map = _build_graph(all_files, root, error_reporter)
    selected_rels, entry_rel = _select_files(root, config, graph, file_map)

    # Queue/topological order for the selected file set (all selected, not just agent files).
    ordered_selected = [p for p in graph.topological_order() if p in selected_rels]

    # 0.8.0: --safe-mode is deprecated; print a notice and continue normally.
    if config.get("safe_mode", False):
        print(
            "WARNING: --safe-mode is now the default behaviour in 0.8.0 and this "
            "flag has no additional effect.  It is kept for backwards compatibility "
            "and will be removed in a future release.",
            flush=True,
        )
        logger.info("--safe-mode flag noted; live backup is always on in 0.8.0.")

    # Always-on live backup writer (Work Item 1).
    recorder = SafeWriter(live_backup_path, output_format, entry_rel, file_map)
    recorder.set_queue_order(ordered_selected)

    # Ownership check + pre-load existing records (raises ConfigError for foreign files).
    recorder.load()

    # Migration: load old Checkpoint entries when no live backup exists.
    checkpoint_results: dict[str, dict] = {}
    if recorder.size == 0:
        cp = Checkpoint(output_dir, entry_file=entry_rel)
        checkpoint_results = cp.load()
        if checkpoint_results:
            logger.info(
                "Migration: loaded %d entry(s) from legacy .codedoc_progress.json "
                "— they will be migrated into the live backup on first flush.",
                len(checkpoint_results),
            )

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
    resumed = 0
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
        elif rel_path in checkpoint_results:
            # Migration: verify and restore checkpoint entry.
            checkpoint_entry = checkpoint_results[rel_path]
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
                new_results[rel_path] = {
                    k: v for k, v in checkpoint_entry.items() if k != "_checkpoint_hash"
                }
                logger.info("Migrating from checkpoint: %s", rel_path)
                resumed += 1
        else:
            agent_rels.add(rel_path)

    rate_limit_warnings: list[dict] = []

    if not agent_rels:
        logger.info("All selected files are up-to-date or reused from cached content.")
        stats: dict = {
            "checked": 0,
            "failed": 0,
            "skipped": skipped,
            "reused": reused,
            "resumed": resumed,
            "output_dir": str(output_dir),
            "rate_limit_warnings": rate_limit_warnings,
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
            json_filename=json_filename,
            md_filename=md_filename,
        )
        stats["output_files"] = [str(path) for path in output_files if path]
        error_reporter.flush()
        recorder.delete()
        _set_issue_stats(stats, error_reporter, live_backup_path)
        return stats

    # Build the agent-file queue in topological order.
    queue = ProcessingQueue()
    for rel_path in graph.topological_order():
        if rel_path in agent_rels:
            queue.add(file_map[rel_path])

    # Create LLM provider AFTER the live backup is initialised.
    # initialize_empty() must be called before provider creation so the backup
    # exists even if provider init fails.
    recorder.initialize_empty()

    try:
        llm = create_provider(config)
        logger.info("LLM provider: %s", llm.provider_name)
    except Exception as exc:
        error_reporter.record(exc, context="LLM provider init")
        error_reporter.flush()
        _set_issue_stats({}, error_reporter, live_backup_path)
        raise

    orchestrator = Orchestrator(llm, parallel=config.get("parallel_agents", True))
    stats = {
        "checked": 0,
        "failed": 0,
        "skipped": skipped,
        "reused": reused,
        "resumed": resumed,
        "rate_limit_warnings": rate_limit_warnings,
    }

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
        recorder=recorder,
        config=config,
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
        json_filename=json_filename,
        md_filename=md_filename,
    )
    stats["output_files"] = [str(path) for path in output_files if path]
    error_reporter.flush()
    # For MD-only runs, remove the live JSON backup after clean MD conversion.
    recorder.delete()
    _set_issue_stats(stats, error_reporter, live_backup_path)

    logger.info(
        "Done. checked=%d failed=%d skipped=%d output=%s",
        stats["checked"],
        stats["failed"],
        stats["skipped"],
        output_dir,
    )

    if error_reporter.has_issues():
        logger.info(
            "%d issue(s) recorded. See %s for details.",
            error_reporter.issue_count(),
            error_reporter.log_path,
        )

    return stats


def _set_issue_stats(
    stats: dict,
    error_reporter: ErrorReporter,
    live_backup_path: Path,
) -> None:
    """Populate error/issue stats keys on *stats* in-place."""
    if error_reporter.has_issues():
        stats["error_log"] = str(error_reporter.log_path.resolve())
        stats["issues_recorded"] = error_reporter.issue_count()
    else:
        stats["error_log"] = None
        stats["issues_recorded"] = 0
    if live_backup_path is not None:
        stats["live_backup_path"] = str(live_backup_path.resolve())
    else:
        stats["live_backup_path"] = None


# ---------------------------------------------------------------------------
# Parallel / sequential file processing (Work Items 2 & 3)
# ---------------------------------------------------------------------------

def _process_agent_files(
    queue: ProcessingQueue,
    orchestrator: Orchestrator,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
    recorder: SafeWriter,
    config: dict,
) -> None:
    descriptors: list[dict] = []
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
            recorder,
            config,
        )
        return

    # Build the parallelism ladder (Work Item 3).
    rate_limit_adaptive = config.get("rate_limit_adaptive", True)
    custom_ladder = config.get("parallel_ladder")
    if custom_ladder:
        ladder = list(custom_ladder)
    else:
        ladder = _build_default_ladder(max_workers)

    provider_name = orchestrator.llm.provider_name
    original_max_workers = max_workers
    remaining = list(descriptors)

    for level_index, level in enumerate(ladder):
        if not remaining:
            break

        level = min(level, len(remaining)) or 1
        succeeded, retry_rate_limited, failed_non_rate_limited = _process_descriptor_batch(
            remaining,
            orchestrator,
            queue,
            stats,
            error_reporter,
            max_workers=level,
            max_consecutive_failures=max_consecutive_failures,
            recorder=recorder,
        )
        new_results.update(succeeded)

        # Non-rate-limit failures are retried sequentially immediately so errors
        # are clearly diagnosed and stats["failed"] is correctly incremented.
        # This mirrors the sequential fallback in the old single-batch implementation.
        if failed_non_rate_limited:
            logger.info(
                "Retrying %d non-rate-limit failed file(s) sequentially for clearer diagnostics.",
                len(failed_non_rate_limited),
            )
            _process_files_sequentially(
                failed_non_rate_limited,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                config,
            )

        if not retry_rate_limited or not rate_limit_adaptive:
            # No rate-limited files remain, or adaptive mode is off.
            if retry_rate_limited and not rate_limit_adaptive:
                # Treat remaining rate-limited files as sequential retry.
                remaining = retry_rate_limited
                _process_files_sequentially(
                    remaining,
                    orchestrator,
                    queue,
                    stats,
                    error_reporter,
                    retry_attempts,
                    max_consecutive_failures,
                    new_results,
                    recorder,
                    config,
                )
            break

        # Step down to next ladder rung.
        next_level = ladder[level_index + 1] if level_index + 1 < len(ladder) else 1
        warning = {
            "provider": provider_name,
            "original_max_parallel": original_max_workers,
            "current_level": level,
            "new_level": next_level,
            "retried_count": len(retry_rate_limited),
        }
        stats.setdefault("rate_limit_warnings", []).append(warning)
        warn_msg = (
            f"[{provider_name}] Rate limit detected - your configured "
            f"max_parallel_files ({original_max_workers}) has been reduced to "
            f"{next_level}. Retrying {len(retry_rate_limited)} remaining file(s) "
            f"at lower concurrency."
        )
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        error_reporter.record(RuntimeError(warn_msg), context="rate limit step-down", level="warning")

        remaining = retry_rate_limited

        # If we have exhausted the ladder, fall through to sequential.
        if level_index + 1 >= len(ladder):
            if remaining:
                still_limited_msg = (
                    f"[{provider_name}] Still rate-limited at max_parallel_files=1. "
                    f"Processing sequentially. You may want to lower your default "
                    f"max_parallel_files in config."
                )
                print(still_limited_msg, flush=True)
                logger.warning(still_limited_msg)
            _process_files_sequentially(
                remaining,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                config,
            )
            break


def _process_descriptor_batch(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    recorder: SafeWriter,
    max_consecutive_failures: int = 5,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    """Process a batch of descriptors in parallel at *max_workers* concurrency.

    Returns
    -------
    succeeded : dict[str, dict]
        rel_path → result for files that completed without error.  These have
        already been recorded in the live backup by the worker thread.
    retry_rate_limited : list[dict]
        Descriptors that failed with a rate-limit signal.
    failed_non_rate_limited : list[dict]
        Descriptors that failed for non-rate-limit reasons.
    """
    succeeded: dict[str, dict] = {}
    retry_rate_limited: list[dict] = []
    failed_non_rate_limited: list[dict] = []

    consecutive_failures = 0
    health_reported = False
    total = len(descriptors)
    completed = 0

    logger.info(
        "Parallel batch: %d file(s) at max_workers=%d, provider=%s",
        total,
        max_workers,
        orchestrator.llm.provider_name,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map: dict[concurrent.futures.Future, dict] = {
            pool.submit(_process_and_record, descriptor, orchestrator, recorder): descriptor
            for descriptor in descriptors
        }

        for future in concurrent.futures.as_completed(future_map):
            descriptor = future_map[future]
            rel_path = descriptor["rel_path"]
            try:
                result = future.result()
                # Already recorded in the worker — just update new_results and stats.
                succeeded[rel_path] = result
                queue.mark_checked(rel_path)
                stats["checked"] += 1
                consecutive_failures = 0
                completed += 1
                _log_file_progress("OK", rel_path, completed, total)
            except Exception as exc:
                completed += 1
                if _is_rate_limit_error(exc):
                    # Only retry if not already recorded (worker may have succeeded
                    # before the future was cancelled).
                    if not recorder.has_record(rel_path):
                        retry_rate_limited.append(descriptor)
                        _log_file_progress("RATE-LIMIT", rel_path, completed, total, str(exc))
                    else:
                        # Already recorded — treat as succeeded.
                        succeeded[rel_path] = {}
                        queue.mark_checked(rel_path)
                        stats["checked"] += 1
                        _log_file_progress("OK(late)", rel_path, completed, total)
                else:
                    failed_non_rate_limited.append(descriptor)
                    consecutive_failures += 1
                    _log_file_progress("RETRY", rel_path, completed, total, str(exc))

                if (
                    consecutive_failures >= max_consecutive_failures
                    and not health_reported
                ):
                    health_reported = True
                    _cancel_pending(future_map)
                    error_reporter.record(
                        RuntimeError(
                            f"Parallel processing saw {consecutive_failures} consecutive "
                            "non-rate-limit file failures. "
                            "Failed files will be retried sequentially."
                        ),
                        context="parallel processing health check",
                        level="warning",
                    )

    return succeeded, retry_rate_limited, failed_non_rate_limited


def _process_files_sequentially(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
    recorder: SafeWriter,
    config: dict,
) -> None:
    consecutive_failures = 0
    total = len(descriptors)
    respect_retry_after = config.get("respect_retry_after", True)
    retry_after_cap = config.get("retry_after_cap_s", 30)

    logger.info(
        "Starting sequential documentation: %d file(s), provider=%s",
        total,
        orchestrator.llm.provider_name,
    )

    for index, descriptor in enumerate(descriptors, start=1):
        rel_path = descriptor["rel_path"]
        try:
            result = _process_one_file_with_retries(
                descriptor,
                orchestrator,
                retry_attempts,
                respect_retry_after=respect_retry_after,
                retry_after_cap=retry_after_cap,
            )
            new_results[rel_path] = result
            recorder.record(rel_path, result, _safe_file_hash(descriptor.get("path")))
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
    respect_retry_after: bool = True,
    retry_after_cap: int = 30,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retry_attempts + 1):
        try:
            return _process_one_file(descriptor, orchestrator)
        except Exception as exc:
            last_error = exc
            if attempt < retry_attempts:
                # Apply Retry-After sleep for rate-limit errors in sequential mode.
                if respect_retry_after and _is_rate_limit_error(exc):
                    delay = _parse_retry_after(exc)
                    if delay is not None:
                        sleep_s = min(delay, retry_after_cap)
                        logger.info(
                            "Retry-After: sleeping %.1fs before retrying %s",
                            sleep_s,
                            descriptor["rel_path"],
                        )
                        time.sleep(sleep_s)
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
            status, rel_path, completed, total, percent, remaining, detail,
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


def _safe_file_hash(file_path: Path | None) -> str:
    if not file_path:
        return ""
    try:
        return compute_file_hash(file_path)
    except Exception:
        return ""


def _cancel_pending(future_map: dict) -> None:
    for future in future_map:
        if not future.done():
            future.cancel()


# ---------------------------------------------------------------------------
# Config / graph helpers
# ---------------------------------------------------------------------------

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
            "documenting all %d file(s) instead.",
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
                edges.append({"from": rel_path, "to": dependency, "type": "internal_import"})
    return edges
