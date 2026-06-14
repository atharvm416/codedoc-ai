"""Documentation pipeline lifecycle coordinator for codedoc.

0.9.4 internal decomposition
----------------------------
This module is now a thin lifecycle coordinator.  The heavy lifting has been
moved into cohesive, single-responsibility modules:

- :mod:`codedoc.core.resume` — live-backup path resolution, existing-record
  loading, public→internal record reconstruction, final documentation-record
  construction, and stale/legacy cleanup.
- :mod:`codedoc.core.discovery` — entry recovery, dependency-graph
  construction, entry-reachability selection, and graph-edge serialization.
- :mod:`codedoc.core.execution` — rate-limit/retry classification, the
  adaptive-parallelism ladder, and sequential/parallel file processing behind
  :class:`~codedoc.core.execution.ExecutionContext` /
  :class:`~codedoc.core.execution.ExecutionOptions`.

``run_pipeline`` keeps its public signature and the required phase ordering:

1. configuration and read-only ownership inspection;
2. scan, graph, selection, and planning;
3. paid-file-cap decision;
4. filesystem mutation and live-backup initialization;
5. provider creation;
6. execution;
7. final output, logs, cleanup, and statistics.

Behavior is unchanged from 0.9.3; this release only reorganizes structure.

Compatibility note (0.9.4)
--------------------------
Private helpers that moved to the modules above are re-exported here for one
release because repository tests and documented integrations import them as
``codedoc.pipeline._name``.  These re-exports are deprecated; import from the
defining module instead.  No runtime warning is emitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

from codedoc.agents.base_agent import truncate_for_llm
from codedoc.agents.orchestrator import Orchestrator
from codedoc.bootstrap import ensure_codedoc_installed
from codedoc.core.checkpoint import Checkpoint
from codedoc.core.discovery import (
    _build_graph,
    _graph_edges,
    _resolve_entry_and_docs,
    _select_files,
)
from codedoc.core.execution import (
    ExecutionContext,
    ExecutionOptions,
    execute_agent_files,
)
from codedoc.core.loader import load_config
from codedoc.core.output import (
    inspect_output_ownership,
    read_existing_records,
    write_project_outputs,
)
from codedoc.core.planning import (
    PipelinePlan,
    build_pipeline_plan,
    normalize_force_files,
)
from codedoc.core.queue import ProcessingQueue
from codedoc.core.resume import (
    _build_documentation_records,
    _cleanup_stale_build_file,
    _load_existing_file_docs,
    _remove_legacy_db,
    _resolve_live_backup_path,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.core.scanner import scan_files
from codedoc.core.usage import UsageAccumulator, estimate_tokens
from codedoc.llm.factory import create_provider
from codedoc.llm.rate_limit_profile import get_rate_limit_profile
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import ConfigError, ErrorReporter
from codedoc.utils.logger import get_logger, set_level

# ---------------------------------------------------------------------------
# Compatibility re-exports (0.9.4 — deprecated; import from the defining module)
# ---------------------------------------------------------------------------
# These private helpers moved to resume/discovery/execution.  They are
# re-exported here for one release because repository tests and documented
# integrations still import them as ``codedoc.pipeline._name``.  Several are
# already imported above for use by ``run_pipeline``; the names below cover the
# remaining helpers that only external/test callers reference.
from codedoc.core.resume import (  # noqa: F401  (compat re-export)
    _load_existing_file_docs_from_md,
    _public_record_to_doc,
)
from codedoc.core.execution import (  # noqa: F401  (compat re-export)
    _RATE_LIMIT_SIGNALS,
    _agent_errors,
    _build_default_ladder,
    _cancel_pending,
    _detect_limit_type,
    _is_rate_limit_error,
    _log_file_progress,
    _parse_retry_after,
    _process_and_record,
    _process_descriptor_batch,
    _process_files_sequentially,
    _process_one_file,
    _process_one_file_with_retries,
    _safe_file_hash,
)

logger = get_logger(__name__)


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

    dry_run = bool(config.get("dry_run", False))
    if dry_run:
        logger.info("Dry run: planning only — no writes, no provider, no LLM calls.")

    output_dir = root / config["output_dir"]

    json_filename = config.get("output_json_filename", "codedoc.json")
    md_filename = config.get("output_md_filename", "codedoc.md")
    live_backup_path = _resolve_live_backup_path(output_dir, output_format, json_filename, md_filename)

    # Read-only ownership inspection (0.9.2).  A real run fails fast before any
    # filesystem side effect, scanning, or LLM call when a final output target
    # is foreign-owned; a dry run records the conflicts and reports them.
    ownership_conflicts = inspect_output_ownership(
        output_dir, output_format, json_filename, md_filename, live_backup_path
    )
    if ownership_conflicts and not dry_run:
        raise ConfigError(ownership_conflicts[0]["message"])

    # 0.8.0: --safe-mode is deprecated; print at most one notice when it was
    # explicitly enabled, then continue normally.
    if config.get("safe_mode", False):
        print(
            "WARNING: --safe-mode is now the default behaviour in 0.8.0 and this "
            "flag has no additional effect.  It is kept for backwards compatibility "
            "and will be removed in a future release.",
            flush=True,
        )
        logger.info("--safe-mode flag noted; live backup is always on in 0.8.0.")

    # 0.8.0: error.log lives in the output directory, not the project root.
    # Construction is in-memory only; nothing is written until flush(), which
    # dry-run never calls.
    error_reporter = ErrorReporter(output_dir / "error.log")

    existing_docs = _load_existing_file_docs(
        output_dir, json_filename, md_filename, live_backup_path, read_only=True
    )

    # Build the scanner skip_dirs list.  Start from config["skip_dirs"] (already
    # resolved by load_config with _add/_remove applied), then unconditionally
    # append the output directory name so codedoc never scans its own output
    # even when the user removes "codedoc" via --remove-skip-dir.
    _scan_skip_dirs = list(config.get("skip_dirs", []))
    _raw_output_dir = str(config.get("output_dir", "codedoc"))
    if _raw_output_dir not in (".", ""):
        _output_dir_name = Path(_raw_output_dir).name
        if _output_dir_name and _output_dir_name not in _scan_skip_dirs:
            _scan_skip_dirs.append(_output_dir_name)

    all_files = scan_files(
        root,
        extension_language_map=config["extension_language_map"],
        max_file_size_kb=config["max_file_size_kb"],
        skip_dirs=_scan_skip_dirs,
        ignore_paths=config.get("ignore_paths"),
    )
    if not all_files:
        # A2: an explicitly specified entry cannot be honoured if nothing was
        # scanned — fail loudly rather than exit successfully having documented
        # nothing.
        if config.get("entry_file"):
            raise ConfigError(
                f"Entry file '{config['entry_file']}' was requested but no "
                f"supported source files were found in '{root}'. Check the path "
                "and your skip_dirs / ignore_paths / extension settings."
            )
        logger.warning("No supported files found in %s. Done.", root)
        if dry_run:
            return {
                "dry_run": True,
                "scanned": 0,
                "selected": 0,
                "entry_excluded": 0,
                "would_process": 0,
                "would_call_llm_for": 0,
                "unchanged": 0,
                "would_reuse": 0,
                "would_resume": 0,
                "forced": 0,
                "estimated_calls": 0,
                "estimated_input_tokens": 0,
                "estimate_is_lower_bound": True,
                "max_files": int(config.get("max_files", 0) or 0),
                "max_files_exceeded": False,
                "ownership_conflicts": ownership_conflicts,
                "output_dir": str(output_dir),
                "output_files": [],
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        _remove_legacy_db(output_dir)
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

    # Count of scanned files excluded by entry-reachability selection (A1).
    # Zero when no entry is in effect (selected_rels == all scanned files).
    # Surfaced in stats so the CLI can report it at the run summary.
    entry_excluded = len(file_map) - len(selected_rels)

    # Queue/topological order for the selected file set (all selected, not just agent files).
    ordered_selected = [p for p in graph.topological_order() if p in selected_rels]

    # 0.9.2: normalize forced paths against the project root (raises
    # ConfigError for paths outside the root).
    forced_paths = normalize_force_files(config.get("force_files") or [], root)

    # Migration eligibility: legacy Checkpoint entries may be used only when
    # the live backup contains no records — read-only equivalent of the
    # ``SafeWriter.size == 0`` check.
    live_records = read_existing_records(live_backup_path)
    checkpoint_results: dict[str, dict] = {}
    if not live_records:
        cp = Checkpoint(output_dir, entry_file=entry_rel)
        checkpoint_results = cp.load()
        if checkpoint_results:
            logger.info(
                "Migration: loaded %d entry(s) from legacy .codedoc_progress.json "
                "— they will be migrated into the live backup on first flush.",
                len(checkpoint_results),
            )

    # 0.9.2: one shared plan drives both dry-run and real execution.
    plan, materials = build_pipeline_plan(
        file_map=file_map,
        graph=graph,
        selected_rels=selected_rels,
        entry_rel=entry_rel,
        existing_docs=existing_docs,
        checkpoint_records=checkpoint_results,
        forced_paths=forced_paths,
        config=config,
    )

    if dry_run:
        return _build_dry_run_stats(
            plan, file_map, config, output_dir, ownership_conflicts
        )

    # Paid-file safety cap: enforced after the complete plan exists and before
    # any mutation, writer initialization, or provider creation.
    if plan.max_files_exceeded:
        raise ConfigError(
            f"This run would send {len(plan.agent_rels)} file(s) to the LLM, "
            f"which exceeds the configured max_files limit of {plan.max_files}. "
            "Inspect the plan first with --dry-run, and raise --max-files only "
            "after reviewing it."
        )

    # ------------------------------------------------------------------
    # Mutation boundary — everything below may write to the filesystem.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_db(output_dir)
    _cleanup_stale_build_file(output_dir, json_filename)

    # Always-on live backup writer (Work Item 1).
    recorder = SafeWriter(live_backup_path, output_format, entry_rel, file_map)
    recorder.set_queue_order(ordered_selected)

    # Ownership check + pre-load existing records (raises ConfigError for foreign files).
    recorder.load()

    skipped = len(plan.unchanged_rels)
    if skipped > 0:
        logger.info("Incremental mode: skipping %d unchanged file(s)", skipped)

    # Materialize reuse/resume records exactly as the plan routed them.
    new_results: dict[str, dict] = {}
    for rel_path in plan.identical_reuse_rels:
        source_doc = materials.identical_reuse_docs[rel_path]
        new_results[rel_path] = source_doc
        logger.info(
            "Reusing cached documentation for %s from identical content in %s",
            rel_path,
            source_doc.get("path", "unknown"),
        )
    reused = len(plan.identical_reuse_rels)

    for rel_path in plan.checkpoint_reuse_rels:
        new_results[rel_path] = materials.checkpoint_reuse_docs[rel_path]
        logger.info("Migrating from checkpoint: %s", rel_path)
    resumed = len(plan.checkpoint_reuse_rels)

    agent_rels = set(plan.agent_rels)
    usage = UsageAccumulator()

    rate_limit_warnings: list[dict] = []

    if not agent_rels:
        logger.info("All selected files are up-to-date or reused from cached content.")
        stats: dict = {
            "checked": 0,
            "failed": 0,
            "skipped": skipped,
            "reused": reused,
            "resumed": resumed,
            "entry_excluded": entry_excluded,
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
        _set_usage_stats(stats, usage, plan, config)
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
        # The exception propagates to the CLI which never sees stats, so print
        # the error.log path here so the user can always find diagnostics.
        if error_reporter.has_issues():
            print(
                f"\n  1 issue recorded. See {error_reporter.log_path.resolve()} for details.",
                file=sys.stderr,
                flush=True,
            )
        raise

    orchestrator = Orchestrator(
        llm,
        parallel=config.get("parallel_agents", True),
        max_content_chars=config.get("max_content_chars", 12000),
        usage=usage,
    )
    stats = {
        "checked": 0,
        "failed": 0,
        "skipped": skipped,
        "reused": reused,
        "resumed": resumed,
        "entry_excluded": entry_excluded,
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

    # 0.9.4: build the execution context from resolved configuration.  The
    # provider-aware rate-limit profile and the execution policy are computed
    # here; execution.py never sees the configuration dictionary.
    custom_ladder = config.get("parallel_ladder")
    options = ExecutionOptions(
        max_workers=max_workers,
        retry_attempts=retry_attempts,
        max_consecutive_failures=max_consecutive_failures,
        rate_limit_adaptive=config.get("rate_limit_adaptive", True),
        parallel_ladder=tuple(custom_ladder) if custom_ladder else None,
        respect_retry_after=config.get("respect_retry_after", True),
        retry_after_cap_s=config.get("retry_after_cap_s", 30),
    )
    context = ExecutionContext(
        orchestrator=orchestrator,
        queue=queue,
        recorder=recorder,
        error_reporter=error_reporter,
        rate_limit_profile=get_rate_limit_profile(llm.provider_name, config),
        stats=stats,
        new_results=new_results,
        options=options,
    )
    execute_agent_files(context)

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
    _set_usage_stats(stats, usage, plan, config)

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


def _set_usage_stats(
    stats: dict,
    usage: UsageAccumulator,
    plan: PipelinePlan,
    config: dict,
) -> None:
    """Populate planned/actual usage keys on *stats* in-place (0.9.2).

    Token figures are character-heuristic estimates, not tokenizer counts.
    """
    stats.update(usage.snapshot())
    stats["planned_calls"] = len(plan.agent_rels) * 3
    stats["planned_files"] = len(plan.agent_rels)
    # Files the plan routed to the LLM that were neither completed nor failed
    # (e.g. a run aborted early by the consecutive-failure health check).
    stats["unattempted_files"] = max(
        0, len(plan.agent_rels) - stats.get("checked", 0) - stats.get("failed", 0)
    )
    # Resolved from config so env/config-enabled partial mode reaches the CLI.
    stats["allow_partial"] = bool(config.get("allow_partial", False))


def _build_dry_run_stats(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    config: dict,
    output_dir: Path,
    ownership_conflicts: list[dict],
) -> dict:
    """Build the read-only dry-run stats dict from the shared plan."""
    return {
        "dry_run": True,
        "scanned": len(plan.scanned_rels),
        "selected": len(plan.selected_rels),
        "entry_excluded": len(plan.scanned_rels - plan.selected_rels),
        "would_process": len(plan.process_rels),
        "would_call_llm_for": len(plan.agent_rels),
        "unchanged": len(plan.unchanged_rels),
        "would_reuse": len(plan.identical_reuse_rels),
        "would_resume": len(plan.checkpoint_reuse_rels),
        "forced": len(plan.forced_rels),
        "estimated_calls": len(plan.agent_rels) * 3,
        "estimated_input_tokens": _estimate_planned_input_tokens(plan, file_map, config),
        "estimate_is_lower_bound": True,
        "max_files": plan.max_files,
        "max_files_exceeded": plan.max_files_exceeded,
        "ownership_conflicts": ownership_conflicts,
        "output_dir": str(output_dir),
        "output_files": [],
    }


def _estimate_planned_input_tokens(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    config: dict,
) -> int:
    """Estimate input tokens for the planned LLM calls — a lower bound.

    The structure and dependency prompts are built from known inputs.  The
    documentation prompt embeds the other agents' responses, which do not
    exist yet, so it is estimated with empty analysis objects.  Uses the same
    centralized truncation helper as real execution so the estimated source
    size matches what would actually be sent.
    """
    from codedoc.agents import dependency_agent, documentation_agent, structure_agent

    max_chars = config.get("max_content_chars", 12000)
    total = 0
    for rel_path in plan.agent_rels:
        descriptor = file_map[rel_path]
        try:
            content = descriptor["path"].read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            content = ""
        try:
            imports = parse_file(descriptor)
        except Exception:
            imports = []
        language = descriptor.get("language", "generic")
        content = truncate_for_llm(content, max_chars)
        prompts = (
            structure_agent.build_prompt(rel_path, content, imports, language),
            dependency_agent.build_prompt(rel_path, content, imports, language),
            documentation_agent.build_prompt(rel_path, content, language, {}, {}),
        )
        for system, prompt in prompts:
            total += estimate_tokens(system) + estimate_tokens(prompt)
    return total
