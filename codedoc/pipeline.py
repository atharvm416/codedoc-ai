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
from codedoc.agents.orchestrator import Orchestrator, initial_calls_per_file
from codedoc.agents.prompt_customization_validation_agent import (
    PromptCustomizationValidationAgent,
)
from codedoc.agents.prompt_profile_routing_agent import PromptProfileRoutingAgent
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
from codedoc.core.block_manager import BlockError
from codedoc.core.ignore_manager import generated_ignore_entries, update_output_gitignore
from codedoc.core.output import (
    inspect_output_ownership,
    preflight_output_accessibility,
    validate_distinct_artifact_paths,
    write_project_outputs,
)
from codedoc.core.planning import (
    PipelinePlan,
    build_pipeline_plan,
    normalize_force_files,
)
from codedoc.core.prompt_profiles import (
    PROFILE_ACTION_CONVERSION_REQUIRED,
    ProfileResolution,
    ResolvedProfile,
    ReviewBatch,
    build_conversion_proposal_fragment,
    build_conversion_review_batches,
    build_resolved_profile,
    build_review_batches,
    build_routing_request,
    classify_profile_action,
    resolve_profile_source,
    routing_conversion_id,
)
from codedoc.core.queue import ProcessingQueue
from codedoc.core.resume import (
    _build_documentation_records,
    _cleanup_legacy_recovery,
    _cleanup_stale_build_file,
    _load_existing_file_docs,
    _remove_legacy_db,
    _resolve_legacy_backup_path,
    _resolve_live_backup_path,
    select_active_recovery_path,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.core.scanner import scan_files
from codedoc.core.usage import UsageAccumulator, estimate_tokens
from codedoc.llm.factory import create_provider, describe_provider_selection
from codedoc.llm.rate_limit_profile import get_rate_limit_profile
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import (
    ConfigError,
    ErrorReporter,
    LiveBackupWriteError,
    PromptCustomizationValidationError,
    UnrecoverableProviderError,
)
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

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)
    dry_run = bool(config.get("dry_run", False))

    analysis_mode = config.get("analysis_mode", "single")
    known_languages = frozenset(config["extension_language_map"].values())
    # 0.11.1 (Review Addendum 11): resolve and classify the prompt profile BEFORE
    # entry resolution, artifact/ownership inspection, existing-output reads,
    # source scanning, graph building, and planning.  Both calls are read-only and
    # provider-free.  A customized single-only structure selected in triple mode is
    # routed into the conversion-proposal workflow here so it can never silently
    # resolve to inactive built-in triple defaults.
    profile_resolution = resolve_profile_source(
        config,
        root,
        known_languages=known_languages,
        active_mode=analysis_mode,
    )
    profile_action = classify_profile_action(profile_resolution.profile, analysis_mode)
    if profile_action.action == PROFILE_ACTION_CONVERSION_REQUIRED:
        return _run_single_to_triple_conversion(
            config=config,
            profile_resolution=profile_resolution,
            profile_action=profile_action,
            analysis_mode=analysis_mode,
            dry_run=dry_run,
        )

    _resolve_entry_and_docs(root, config)
    resolved_profile = build_resolved_profile(profile_action, analysis_mode)
    no_work_profile_stats = {
        "prompt_profile_source": profile_resolution.source,
        "prompt_profile_file": (
            str(profile_resolution.source_path)
            if profile_resolution.source_path is not None
            else None
        ),
        "prompt_profile_active": False,
        "prompt_profile_affected_files": 0,
        "prompt_customization_security_review": "not-required",
        "prompt_customization_security_review_calls_planned": 0,
        "prompt_customization_security_review_calls_completed": 0,
        "prompt_customization_security_warnings": 0,
        "prompt_customization_security_blocking_reasons": 0,
        "prompt_customization_allow_risky": bool(
            config.get("prompt_customization_allow_risky", False)
        ),
        **_conversion_stats_defaults(),
    }

    output_format = config.get("output_format", "json")
    logger.info("Output format: %s", output_format)

    if dry_run:
        logger.info("Dry run: planning only — no writes, no provider, no LLM calls.")

    output_dir = root / config["output_dir"]
    ignore_target: Path | None = None
    if config.get("manage_output_gitignore", False):
        ignore_target = output_dir / config["output_gitignore_filename"]
        resolved_output = output_dir.resolve()
        resolved_ignore = ignore_target.resolve()
        try:
            resolved_ignore.relative_to(resolved_output)
        except ValueError as exc:
            raise ConfigError("output_gitignore_filename must resolve inside output_dir") from exc
        if ignore_target.is_symlink():
            raise ConfigError(f"Managed ignore target '{ignore_target}' is a symbolic link")
        if ignore_target.exists() and ignore_target.is_dir():
            raise ConfigError(f"Managed ignore target '{ignore_target}' is a directory")

    json_filename = config.get("output_json_filename", "codedoc.json")
    md_filename = config.get("output_md_filename", "codedoc.md")
    # 0.9.8: crash-recovery data is staged in its own dedicated file, never the
    # stable output.  Derive the base recovery path, then deterministically
    # select the active candidate (absent → fresh; valid in-progress → resume;
    # invalid/foreign/completed → advance to the next ``(<n>)`` suffix).  The
    # walk is read-only.  The legacy path is the pre-0.9.8 live-backup location,
    # read only as an in-progress overlay during resume/migration.
    base_recovery_path = _resolve_live_backup_path(
        output_dir, output_format, json_filename, md_filename
    )
    live_backup_path, _recovery_resume = select_active_recovery_path(base_recovery_path)
    legacy_recovery_path = _resolve_legacy_backup_path(
        output_dir, output_format, json_filename, md_filename
    )
    # Stable output targets and the active recovery file must never be removed by
    # the legacy-migration cleanup (which only prunes a distinct, migrated,
    # in-progress legacy sibling — e.g. the old md JSON sibling).
    _recovery_keep_paths: set[Path] = {live_backup_path}
    if output_format in ("json", "both"):
        _recovery_keep_paths.add(output_dir / json_filename)
    if output_format in ("md", "both"):
        _recovery_keep_paths.add(output_dir / md_filename)

    # Reject generated-artifact path collisions before any scan or mutation.
    # The dedicated recovery file is always its own ``live_backup`` artifact,
    # distinct from the final JSON (``json``), Markdown (``markdown``), and the
    # diagnostic log (``error_log``), for every format.  The *selected* recovery
    # candidate is the artifact validated here, so a configuration that points a
    # final output or the log at the recovery file is rejected rather than hidden
    # behind a suffix increment.
    artifact_paths: dict[str, Path | None] = {"error_log": output_dir / "error.log"}
    if output_format in ("json", "both"):
        artifact_paths["json"] = output_dir / json_filename
    if output_format in ("md", "both"):
        artifact_paths["markdown"] = output_dir / md_filename
    artifact_paths["live_backup"] = live_backup_path
    if ignore_target is not None:
        artifact_paths["output_gitignore"] = ignore_target
    validate_distinct_artifact_paths(artifact_paths)

    # Read-only ownership inspection (0.9.2).  A real run fails fast before any
    # filesystem side effect, scanning, or LLM call when a final output target
    # is foreign-owned; a dry run records the conflicts and reports them.  The
    # recovery file is NOT inspected here: a foreign file at a recovery name is
    # preserved and skipped by the candidate walk, not treated as a run-blocking
    # final-output conflict.
    ownership_conflicts = inspect_output_ownership(
        output_dir, output_format, json_filename, md_filename
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
        output_dir,
        json_filename,
        md_filename,
        live_backup_path,
        read_only=True,
        output_format=output_format,
        legacy_recovery_path=legacy_recovery_path,
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
        follow_symlinks=config.get("follow_symlinks", False),
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
                "analysis_mode": config.get("analysis_mode", "single"),
                "initial_calls_per_file": initial_calls_per_file(
                    config.get("analysis_mode", "single")
                ),
                "documentation_scope": config.get("documentation_scope", "entry"),
                "entry_reachable": 0,
                "entry_disconnected": 0,
                "disconnected_paid_files": 0,
                "disconnected_planned_calls": 0,
                "would_process": 0,
                "would_call_llm_for": 0,
                "unchanged": 0,
                "would_reuse": 0,
                "would_resume": 0,
                "forced": 0,
                "estimated_calls": 0,
                "estimated_input_tokens": 0,
                "estimate_is_lower_bound": config.get("analysis_mode", "single") == "triple",
                "max_files": int(config.get("max_files", 0) or 0),
                "max_files_exceeded": False,
                "ownership_conflicts": ownership_conflicts,
                "output_dir": str(output_dir),
                "output_files": [],
                **_initial_ignore_stats(config, ignore_target),
                **no_work_profile_stats,
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        _remove_legacy_db(output_dir)
        return {
            "checked": 0,
            "failed": 0,
            "skipped": 0,
            "entry_excluded": 0,
            "analysis_mode": config.get("analysis_mode", "single"),
            "initial_calls_per_file": initial_calls_per_file(
                config.get("analysis_mode", "single")
            ),
            "output_dir": str(output_dir),
            "live_backup_path": None,
            "error_log": None,
            "issues_recorded": 0,
            "rate_limit_warnings": [],
            "documentation_scope": config.get("documentation_scope", "entry"),
            "entry_reachable": 0,
            "entry_disconnected": 0,
            "disconnected_paid_files": 0,
            "disconnected_planned_calls": 0,
            **_initial_ignore_stats(config, ignore_target),
            **no_work_profile_stats,
        }

    graph, file_map, unresolved_imports_by_path = _build_graph(all_files, root, error_reporter)
    reachable_rels, documented_rels, entry_rel = _select_files(
        root, config, graph, file_map
    )

    # Count of scanned files excluded by entry-reachability selection (A1).
    # Zero when no entry is in effect (selected_rels == all scanned files).
    # Surfaced in stats so the CLI can report it at the run summary.
    entry_excluded = len(file_map) - len(documented_rels)

    # Queue/topological order for the selected file set (all selected, not just agent files).
    ordered_selected = [p for p in graph.topological_order() if p in documented_rels]

    # 0.9.2: normalize forced paths against the project root (raises
    # ConfigError for paths outside the root).
    forced_paths = normalize_force_files(config.get("force_files") or [], root)

    # Migration eligibility: legacy Checkpoint entries may be used only when no
    # existing records were recovered — read-only equivalent of the
    # ``SafeWriter.size == 0`` check.  0.9.8: the recovery file is now a separate
    # path, so eligibility is keyed on the merged reuse set (stable baseline +
    # legacy overlay + active recovery), i.e. exactly what seeds the writer.
    checkpoint_results: dict[str, dict] = {}
    if not existing_docs:
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
        selected_rels=documented_rels,
        entry_rel=entry_rel,
        existing_docs=existing_docs,
        checkpoint_records=checkpoint_results,
        forced_paths=forced_paths,
        config=config,
        resolved_profile=resolved_profile,
    )

    planned_languages = frozenset(
        file_map[rel].get("language", "generic") for rel in plan.agent_rels
    )
    review_batches = build_review_batches(resolved_profile, planned_languages)

    if dry_run:
        return _build_dry_run_stats(
            plan,
            file_map,
            config,
            output_dir,
            ownership_conflicts,
            reachable_rels,
            resolved_profile,
            profile_resolution,
            review_batches,
        )

    scope_stats = _build_scope_stats(
        config, file_map, reachable_rels, documented_rels, plan.agent_rels, entry_rel
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

    usage = UsageAccumulator()
    llm = None
    review_stats = _base_profile_stats(
        resolved_profile, profile_resolution, plan, file_map, review_batches
    )
    review_stats["prompt_customization_allow_risky"] = bool(
        config.get("prompt_customization_allow_risky", False)
    )
    if review_batches:
        # This is a probabilistic semantic standards/safety gate. TOO_RISKY
        # blocks by default and only the explicit user override may bypass that
        # well-formed verdict. Deterministic validation and strict cleaners are
        # the non-overridable structural boundary.
        review_provider, review_model = describe_provider_selection(config)
        print(
            "Prompt customization standards/safety review will make "
            f"{len(review_batches)} paid provider call(s) using "
            f"provider={review_provider}, model={review_model}. No profile content "
            "is persisted in run statistics.",
            flush=True,
        )
        llm = create_provider(config)
        reviewer = PromptCustomizationValidationAgent(llm, usage=usage)
        try:
            outcome = reviewer.review(review_batches)
        except PromptCustomizationValidationError as exc:
            # A fail-closed review (transport/binding/contradiction/malformed)
            # aborts before any documentation call.  Every batch whose provider
            # call was attempted is reflected by the shared usage accumulator, so
            # category attempts reconcile to attempted_calls.
            review_stats["prompt_customization_security_review"] = "failed-closed"
            review_stats["prompt_customization_security_review_calls_attempted"] = (
                usage.attempted_calls
            )
            review_stats.update(usage.snapshot())
            raise PromptCustomizationValidationError(str(exc), stats=review_stats) from exc
        review_stats["prompt_customization_security_review_calls_completed"] = (
            outcome.calls_completed
        )
        # On a non-raising return every batch's provider call completed, so
        # attempted equals completed for the review category.
        review_stats["prompt_customization_security_review_calls_attempted"] = (
            outcome.calls_completed
        )
        review_stats.update(usage.snapshot())
        review_stats["prompt_customization_security_warnings"] = len(outcome.warnings)
        review_stats["prompt_customization_security_blocking_reasons"] = len(
            outcome.reasons
        )
        if outcome.verdict == "SAFE":
            review_stats["prompt_customization_security_review"] = "safe"
            print("Prompt customization standards/safety review: SAFE — continuing.")
        elif outcome.verdict == "RISKY":
            review_stats["prompt_customization_security_review"] = "risky"
            print("Prompt customization standards/safety review: RISKY — proceeding with warnings:")
            for warning in outcome.warnings:
                print(f"  - {warning}")
            print("Review the flagged items; re-run with a corrected profile to clear them.")
        else:
            shown = _reviewed_shape_text(review_batches, analysis_mode)
            if config.get("prompt_customization_allow_risky", False):
                review_stats["prompt_customization_security_review"] = (
                    "too-risky-overridden"
                )
                print(
                    "Prompt customization standards/safety review: TOO RISKY — "
                    "proceeding ONLY because --allow-risky-prompt-customization is set."
                )
                for reason in outcome.reasons:
                    print(f"  - {reason}")
                print(f"Customization applied (effective requested-shape JSON for {analysis_mode}):\n{shown}")
            else:
                review_stats["prompt_customization_security_review"] = (
                    "too-risky-blocked"
                )
                raise PromptCustomizationValidationError(
                    "Prompt customization standards/safety review: TOO RISKY — "
                    "blocked and not applied.\n"
                    + "\n".join(f"- {reason}" for reason in outcome.reasons)
                    + f"\nCustomization under review (effective requested-shape JSON for {analysis_mode}):\n"
                    + shown
                    + "\nTo override the semantic review and proceed at your own risk, "
                    "re-run with --allow-risky-prompt-customization (or set "
                    "prompt_customization_allow_risky: true). Deterministic "
                    "validation and the strict cleaners cannot be overridden.",
                    stats=review_stats,
                )

    # ------------------------------------------------------------------
    # Mutation boundary — everything below may write to the filesystem.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    # 0.10.1 (Workstream C): a provider-free output accessibility probe before
    # any provider is created.  Runs for every real finalization path — including
    # all-reused runs that still rewrite stable output — but never for dry-run
    # (which returns above).  Raises a classified OutputError if the directory
    # cannot be written, so a persistent permission/space failure is caught
    # before paid work instead of after it.
    preflight_output_accessibility(output_dir)
    _remove_legacy_db(output_dir)
    _cleanup_stale_build_file(output_dir, json_filename)

    # Always-on crash-recovery writer (Work Item 1).  Targets the dedicated
    # recovery file for every format; the stable output is untouched until
    # finalization.
    recorder = SafeWriter(live_backup_path, output_format, entry_rel, file_map)
    recorder.set_queue_order(ordered_selected)

    # Seed the writer with the same merged reuse set planning consumed (stable
    # baseline + legacy in-progress overlay + active recovery overlay) so every
    # partial flush of the recovery file is a self-contained resumable snapshot.
    recorder.load(preloaded=existing_docs)

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
            **scope_stats,
            "output_dir": str(output_dir),
            "rate_limit_warnings": rate_limit_warnings,
            **_initial_ignore_stats(config, ignore_target),
            **no_work_profile_stats,
            **review_stats,
        }
        output_files = write_project_outputs(
            _build_documentation_records(
                documented_rels,
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
            _graph_edges(graph, documented_rels),
            json_filename=json_filename,
            md_filename=md_filename,
            reachable_rels=reachable_rels,
            unresolved_imports_by_path=unresolved_imports_by_path,
        )
        stats["output_files"] = [str(path) for path in output_files if path]
        error_reporter.flush()
        _cleanup_stale_error_log(stats, error_reporter)
        recorder.delete()
        _cleanup_legacy_recovery(legacy_recovery_path, _recovery_keep_paths)
        _finalize_output_gitignore(
            stats, config, output_dir, ignore_target, output_files, error_reporter
        )
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
        if llm is None:
            llm = create_provider(config)
        logger.info("LLM provider: %s", llm.provider_name)
    except Exception as exc:
        error_reporter.record(exc, context="LLM provider init")
        error_reporter.flush()
        # The exception propagates to the CLI which never sees stats, so print
        # the error.log path here so the user can always find diagnostics.
        if error_reporter.has_issues() and error_reporter.has_persisted_log():
            print(
                f"\n  1 issue recorded. See {error_reporter.log_path.resolve()} for details.",
                file=sys.stderr,
                flush=True,
            )
        raise
    except KeyboardInterrupt as exc:
        # 0.9.8: an interrupt during provider init still happens after the
        # recovery file was initialized; name it for the CLI (see below).
        if live_backup_path.exists():
            exc.recovery_path = str(live_backup_path)
        raise

    orchestrator = Orchestrator(
        llm,
        parallel=config.get("parallel_agents", True),
        max_content_chars=config.get("max_content_chars", 12000),
        usage=usage,
        analysis_mode=config.get("analysis_mode", "single"),
        truncation_head_ratio=config.get("truncation_head_ratio", 0.70),
        resolved_profile=resolved_profile,
    )
    stats = {
        "checked": 0,
        "failed": 0,
        "skipped": skipped,
        "reused": reused,
        "resumed": resumed,
        "entry_excluded": entry_excluded,
        **scope_stats,
        "rate_limit_warnings": rate_limit_warnings,
        **_initial_ignore_stats(config, ignore_target),
        **review_stats,
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
    try:
        execute_agent_files(context)
    except UnrecoverableProviderError as exc:
        # 0.9.7: a confirmed unrecoverable provider abort (terminal
        # billing/credentials/model/access, or a bounded zero-progress rate
        # limit).  Record + flush it to error.log here — where the ErrorReporter
        # lives — so the abort is logged exactly like other issues even though it
        # leaves the pipeline by exception.  Then re-raise for the CLI to present.
        # Deliberately do NOT call write_project_outputs(...) or recorder.delete()
        # on this path: the live JSON backup must stay intact and resumable.
        error_reporter.record(exc, context="provider abort")
        error_reporter.flush()
        raise
    except LiveBackupWriteError as exc:
        # 0.10.1 (Workstream D3): the dedicated recovery file could not be
        # persisted, so crash-safety no longer holds and execution stopped
        # scheduling paid work.  The last valid recovery file is preserved by the
        # atomic writer.  Record the failure (target path + classified cause +
        # traceback) to error.log on a best-effort basis, then print the recovery
        # and error-log paths so the user can act.  A log-write failure must never
        # replace or hide this primary persistence error.
        error_reporter.record(exc, context="live backup write")
        log_note = ""
        try:
            error_reporter.flush()
        except Exception as flush_exc:  # noqa: BLE001 — secondary, must not mask primary
            log_note = f" (issue log could not be written: {flush_exc})"
        recovery_note = (
            f"\n  Completed work is preserved in: {live_backup_path}"
            if live_backup_path.exists()
            else ""
        )
        error_log_note = (
            f"\n  Diagnostics: {error_reporter.log_path.resolve()}"
            if error_reporter.has_persisted_log()
            else ""
        )
        print(
            f"\nError: {exc}{recovery_note}{error_log_note}{log_note}",
            file=sys.stderr,
            flush=True,
        )
        raise
    except KeyboardInterrupt as exc:
        # 0.9.8: the run was interrupted mid-processing.  The stable output was
        # never opened; completed work is staged in the dedicated recovery file.
        # Attach the exact selected recovery path (only when it exists on disk)
        # so the CLI can name it in the interrupt message, then re-raise the same
        # exception unchanged.  No suffix walk, candidate creation, or filesystem
        # mutation happens here.
        if live_backup_path.exists():
            exc.recovery_path = str(live_backup_path)
        raise

    stats["output_dir"] = str(output_dir)
    output_files = write_project_outputs(
        _build_documentation_records(
            documented_rels,
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
        _graph_edges(graph, documented_rels),
        json_filename=json_filename,
        md_filename=md_filename,
        reachable_rels=reachable_rels,
        unresolved_imports_by_path=unresolved_imports_by_path,
    )
    stats["output_files"] = [str(path) for path in output_files if path]
    error_reporter.flush()
    # 0.10.1: a clean, issue-free run clears a stale CodeDoc-owned error.log left
    # by a prior failed run so a historical failure no longer looks current.  A
    # foreign log is left byte-identical; a removal failure is an auxiliary
    # warning surfaced in stats, never a failure of valid documentation output.
    _cleanup_stale_error_log(stats, error_reporter)
    # Clean completion: the stable output is written above; only now delete the
    # dedicated recovery file (all formats).  A deletion OSError raises
    # OutputError and leaves both the stable output and the recovery file intact.
    recorder.delete()
    # Complete the migration of a pre-0.9.8 legacy in-progress sibling (md mode):
    # its records are already in the stable output, so remove the leftover.
    _cleanup_legacy_recovery(legacy_recovery_path, _recovery_keep_paths)
    _finalize_output_gitignore(
        stats, config, output_dir, ignore_target, output_files, error_reporter
    )
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


# ---------------------------------------------------------------------------
# Single-to-triple conversion proposal (Workstream D)
# ---------------------------------------------------------------------------

def _conversion_stats_defaults() -> dict:
    """Stable conversion/attempt stat keys, zeroed (present on every return path)."""
    return {
        "documentation_calls_planned": 0,
        "documentation_calls_attempted": 0,
        "prompt_profile_conversion": "not-required",
        "prompt_profile_conversion_calls_planned": 0,
        "prompt_profile_conversion_calls_attempted": 0,
        "prompt_profile_conversion_calls_completed": 0,
        "prompt_customization_security_review_calls_attempted": 0,
    }


def _conversion_base_stats(
    profile_resolution: ProfileResolution,
    config: dict,
    analysis_mode: str,
    review_planned: int,
) -> dict:
    """Shared stat skeleton for every conversion return/raise path."""
    return {
        "analysis_mode": analysis_mode,
        "prompt_profile_source": profile_resolution.source,
        "prompt_profile_file": (
            str(profile_resolution.source_path)
            if profile_resolution.source_path is not None
            else None
        ),
        "prompt_profile_active": True,
        "prompt_profile_affected_files": 0,
        "prompt_customization_allow_risky": bool(
            config.get("prompt_customization_allow_risky", False)
        ),
        "prompt_customization_security_review": (
            "pending" if review_planned else "not-required"
        ),
        "prompt_customization_security_review_calls_planned": review_planned,
        "prompt_customization_security_review_calls_completed": 0,
        "prompt_customization_security_review_calls_attempted": 0,
        "prompt_customization_security_warnings": 0,
        "prompt_customization_security_blocking_reasons": 0,
        "documentation_calls_planned": 0,
        "documentation_calls_attempted": 0,
        "prompt_profile_conversion": "pending",
        "prompt_profile_conversion_calls_planned": 1,
        "prompt_profile_conversion_calls_attempted": 0,
        "prompt_profile_conversion_calls_completed": 0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "attempted_calls": 0,
    }


def _inline_profile_comment(config: dict) -> object | None:
    """Return a bounded inline ``$comment`` to preserve in the proposal, if any."""
    inline = config.get("prompt_profiles")
    if isinstance(inline, dict):
        comment = inline.get("$comment")
        if comment is not None:
            return comment
    return None


def _run_single_to_triple_conversion(
    *,
    config: dict,
    profile_resolution: ProfileResolution,
    profile_action,
    analysis_mode: str,
    dry_run: bool,
) -> dict:
    """Run the bounded single-to-triple conversion proposal and stop.

    Performs only its disclosed review/routing calls; never scans source, mutates
    the filesystem, initializes recovery, or makes a documentation call.  Returns
    a distinct conversion status; raises a setup-class error (with attached
    statistics) on any fail-closed condition (Workstream D, Review Addenda 3/10).
    """
    single_profile = profile_action.single_profile
    review_batches = build_conversion_review_batches(single_profile)
    review_planned = len(review_batches)
    conversion_id = routing_conversion_id(single_profile)

    base = _conversion_base_stats(profile_resolution, config, analysis_mode, review_planned)

    # Complete the bounded deterministic conversion preflight for both dry and
    # real runs.  This keeps a dry-run projection from claiming a proposal is
    # viable when the corresponding real request would be rejected locally.
    try:
        build_routing_request(single_profile, conversion_id)
    except PromptCustomizationValidationError as exc:
        raise PromptCustomizationValidationError(str(exc), stats=dict(base)) from exc

    if dry_run:
        # No provider contact: project the paid plan only.
        base["prompt_profile_conversion"] = "pending"
        return {
            "dry_run": True,
            "analysis_mode": analysis_mode,
            "prompt_profile_conversion_id": conversion_id,
            "total_paid_proposal_calls": review_planned + 1,
            **base,
        }

    review_provider, review_model = describe_provider_selection(config)
    print(
        f"Single-to-triple profile conversion will make {review_planned} paid "
        "security-review call(s) plus 1 paid routing call using "
        f"provider={review_provider}, model={review_model}. It will print a "
        "proposal and stop without generating documentation.",
        flush=True,
    )

    usage = UsageAccumulator()
    llm = create_provider(config)

    # --- Source security review (the customized single structure) ---
    review_status = "not-required"
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    if review_batches:
        reviewer = PromptCustomizationValidationAgent(llm, usage=usage)
        try:
            outcome = reviewer.review(review_batches)
        except PromptCustomizationValidationError as exc:
            stats = dict(base)
            stats["prompt_customization_security_review"] = "failed-closed"
            stats["prompt_customization_security_review_calls_attempted"] = (
                usage.attempted_calls
            )
            stats.update(usage.snapshot())
            raise PromptCustomizationValidationError(str(exc), stats=stats) from exc
        reasons = outcome.reasons
        warnings = outcome.warnings
        base["prompt_customization_security_review_calls_completed"] = outcome.calls_completed
        base["prompt_customization_security_review_calls_attempted"] = outcome.calls_completed
        base.update(usage.snapshot())
        base["prompt_customization_security_warnings"] = len(warnings)
        base["prompt_customization_security_blocking_reasons"] = len(reasons)
        if outcome.verdict == "SAFE":
            review_status = "safe"
            print("Conversion source review: SAFE — continuing to routing.", flush=True)
        elif outcome.verdict == "RISKY":
            review_status = "risky"
            print("Conversion source review: RISKY — proceeding with warnings:", flush=True)
            for warning in warnings:
                print(f"  - {warning}", flush=True)
        else:  # TOO_RISKY
            if config.get("prompt_customization_allow_risky", False):
                review_status = "too-risky-overridden"
                print(
                    "Conversion source review: TOO RISKY — proceeding ONLY because "
                    "--allow-risky-prompt-customization is set.",
                    flush=True,
                )
                for reason in reasons:
                    print(f"  - {reason}", flush=True)
            else:
                stats = dict(base)
                stats["prompt_customization_security_review"] = "too-risky-blocked"
                raise PromptCustomizationValidationError(
                    "Conversion source review: TOO RISKY — blocked and not converted.\n"
                    + "\n".join(f"- {reason}" for reason in reasons)
                    + "\nTo override the semantic review and proceed at your own risk, "
                    "re-run with --allow-risky-prompt-customization (or set "
                    "prompt_customization_allow_risky: true). Deterministic "
                    "validation and the strict cleaners cannot be overridden.",
                    stats=stats,
                )
        base["prompt_customization_security_review"] = review_status

    # --- Paid routing call ---
    routing_agent = PromptProfileRoutingAgent(llm, usage=usage)
    review_attempted = base["prompt_customization_security_review_calls_attempted"]
    try:
        routing_outcome = routing_agent.route(single_profile, conversion_id)
    except PromptCustomizationValidationError as exc:
        stats = dict(base)
        stats.update(usage.snapshot())
        stats["prompt_profile_conversion_calls_attempted"] = max(
            0, usage.attempted_calls - review_attempted
        )
        raise PromptCustomizationValidationError(str(exc), stats=stats) from exc

    # --- Success: build the canonical config-ready proposal fragment ---
    fragment = build_conversion_proposal_fragment(
        single_profile,
        routing_outcome.triple,
        comment=_inline_profile_comment(config),
    )
    import json as _json

    stats = dict(base)
    stats["prompt_profile_conversion"] = "generated-awaiting-confirmation"
    stats["prompt_profile_conversion_calls_attempted"] = routing_outcome.calls_attempted
    stats["prompt_profile_conversion_calls_completed"] = routing_outcome.calls_completed
    stats["prompt_profile_conversion_id"] = conversion_id
    stats["prompt_profile_conversion_factors"] = list(routing_outcome.factors)
    stats["prompt_profile_conversion_fragment"] = _json.dumps(
        fragment, indent=2, ensure_ascii=False
    )
    stats.update(usage.snapshot())
    # Documentation never ran; the three paid categories reconcile to the total.
    stats["documentation_calls_attempted"] = max(
        0,
        stats["attempted_calls"]
        - stats["prompt_customization_security_review_calls_attempted"]
        - stats["prompt_profile_conversion_calls_attempted"],
    )
    return stats


def _cleanup_stale_error_log(stats: dict, error_reporter: ErrorReporter) -> None:
    """Remove a stale CodeDoc-owned error log on a clean run; record any warning.

    Only acts when the current run recorded no issues (``flush`` already wrote
    the log otherwise).  A removal failure is surfaced as ``stats['stale_log_warning']``
    and logged, but never raised.
    """
    warning = error_reporter.clear_stale_owned_log()
    if warning:
        stats["stale_log_warning"] = warning
        logger.warning(warning)


def _set_issue_stats(
    stats: dict,
    error_reporter: ErrorReporter,
    live_backup_path: Path,
) -> None:
    """Populate error/issue stats keys on *stats* in-place."""
    if error_reporter.has_issues():
        stats["issues_recorded"] = error_reporter.issue_count()
        if error_reporter.has_persisted_log():
            stats["error_log"] = str(error_reporter.log_path.resolve())
        else:
            stats["error_log"] = None
            stats["issue_log_warning"] = (
                f"{error_reporter.issue_count()} issue(s) were recorded, but "
                f"'{error_reporter.log_path.name}' could not be written because "
                "a foreign file already exists at that path."
            )
    else:
        stats["error_log"] = None
        stats["issues_recorded"] = 0
    if live_backup_path is not None:
        stats["live_backup_path"] = str(live_backup_path.resolve())
    else:
        stats["live_backup_path"] = None


def _initial_ignore_stats(config: dict, ignore_target: Path | None) -> dict:
    enabled = bool(config.get("manage_output_gitignore", False))
    return {
        "output_gitignore_enabled": enabled,
        "output_gitignore_updated": False,
        "output_gitignore_path": str(ignore_target.resolve()) if enabled and ignore_target else None,
        "output_gitignore_warning": None,
    }


def _finalize_output_gitignore(
    stats: dict,
    config: dict,
    output_dir: Path,
    ignore_target: Path | None,
    output_files: tuple[Path | None, Path | None],
    error_reporter: ErrorReporter,
) -> None:
    """Best-effort auxiliary ignore update after required finalization."""
    if ignore_target is None or not config.get("manage_output_gitignore", False):
        return
    artifacts = [path.name for path in output_files if path is not None and path.exists()]
    if error_reporter.log_path.exists():
        artifacts.append(error_reporter.log_path.name)
    if not artifacts:
        return
    try:
        entries = generated_ignore_entries(artifacts)
        update_output_gitignore(
            output_dir, config["output_gitignore_filename"], entries
        )
        stats["output_gitignore_updated"] = True
    except (BlockError, OSError) as exc:
        warning = f"Documentation succeeded, but managed ignore update failed: {exc}"
        stats["output_gitignore_warning"] = warning
        logger.warning(warning)
        error_reporter.record(exc, context="managed output .gitignore", level="warning")
        try:
            error_reporter.flush()
        except OSError as flush_exc:
            warning = f"{warning}; auxiliary warning log could not be persisted: {flush_exc}"
            stats["output_gitignore_warning"] = warning
            logger.warning(warning)


def _set_usage_stats(
    stats: dict,
    usage: UsageAccumulator,
    plan: PipelinePlan,
    config: dict,
) -> None:
    """Populate planned/actual usage keys on *stats* in-place (0.9.2).

    Token figures are character-heuristic estimates, not tokenizer counts.
    """
    analysis_mode = config.get("analysis_mode", "single")
    per_file = initial_calls_per_file(analysis_mode)
    stats.update(usage.snapshot())
    stats["analysis_mode"] = analysis_mode
    stats["initial_calls_per_file"] = per_file
    stats["planned_calls"] = len(plan.agent_rels) * per_file
    stats["planned_files"] = len(plan.agent_rels)
    # 0.11.1: documentation-call category accounting.  Every provider attempt is
    # exactly one of three categories (documentation, customization review,
    # routing).  Review and routing track their own attempts; documentation is the
    # remainder so the three categories reconcile to ``attempted_calls`` exactly.
    stats["documentation_calls_planned"] = stats["planned_calls"]
    review_attempted = stats.get("prompt_customization_security_review_calls_attempted", 0)
    routing_attempted = stats.get("prompt_profile_conversion_calls_attempted", 0)
    stats["documentation_calls_attempted"] = max(
        0, stats.get("attempted_calls", 0) - review_attempted - routing_attempted
    )
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
    reachable_rels: set[str],
    resolved_profile: ResolvedProfile | None = None,
    profile_resolution: ProfileResolution | None = None,
    review_batches: list[ReviewBatch] | None = None,
) -> dict:
    """Build the read-only dry-run stats dict from the shared plan."""
    scope_stats = _build_scope_stats(
        config,
        file_map,
        reachable_rels,
        set(plan.documented_rels),
        plan.agent_rels,
        plan.entry_rel,
    )
    ignore_target = (
        output_dir / config["output_gitignore_filename"]
        if config.get("manage_output_gitignore", False)
        else None
    )
    analysis_mode = config.get("analysis_mode", "single")
    per_file = initial_calls_per_file(analysis_mode)
    # single mode embeds only known inputs, so its input estimate is exact;
    # triple mode's documentation prompt estimate is a lower bound.
    estimate_is_lower_bound = analysis_mode == "triple"
    review_batches = review_batches or []
    profile_stats = (
        _base_profile_stats(
            resolved_profile, profile_resolution, plan, file_map, review_batches
        )
        if resolved_profile is not None and profile_resolution is not None
        else {}
    )
    if review_batches:
        profile_stats["prompt_customization_security_review"] = "pending"
    if profile_stats:
        profile_stats["prompt_customization_allow_risky"] = bool(
            config.get("prompt_customization_allow_risky", False)
        )
        # Dry-run projects the documentation plan without attempting any call.
        profile_stats["documentation_calls_planned"] = len(plan.agent_rels) * per_file
    return {
        "dry_run": True,
        "scanned": len(plan.scanned_rels),
        "selected": len(plan.documented_rels),
        "entry_excluded": len(plan.scanned_rels - plan.documented_rels),
        "analysis_mode": analysis_mode,
        "initial_calls_per_file": per_file,
        **scope_stats,
        "would_process": len(plan.process_rels),
        "would_call_llm_for": len(plan.agent_rels),
        "unchanged": len(plan.unchanged_rels),
        "would_reuse": len(plan.identical_reuse_rels),
        "would_resume": len(plan.checkpoint_reuse_rels),
        "forced": len(plan.forced_rels),
        "estimated_calls": len(plan.agent_rels) * per_file,
        "estimated_input_tokens": _estimate_planned_input_tokens(
            plan, file_map, config, resolved_profile
        ),
        "estimate_is_lower_bound": estimate_is_lower_bound,
        "max_files": plan.max_files,
        "max_files_exceeded": plan.max_files_exceeded,
        "ownership_conflicts": ownership_conflicts,
        "output_dir": str(output_dir),
        "output_files": [],
        **_initial_ignore_stats(config, ignore_target),
        **profile_stats,
    }


def _base_profile_stats(
    resolved: ResolvedProfile,
    resolution: ProfileResolution,
    plan: PipelinePlan,
    file_map: dict[str, dict],
    batches: list[ReviewBatch],
) -> dict:
    affected_rels = {
        rel
        for rel in plan.documented_rels
        if resolved.is_active_for(file_map[rel].get("language", "generic"))
    }
    affected = len(affected_rels)
    return {
        "prompt_profile_source": resolution.source,
        "prompt_profile_file": (
            str(resolution.source_path) if resolution.source_path is not None else None
        ),
        "prompt_profile_active": affected > 0,
        "prompt_profile_affected_files": affected,
        "prompt_customization_security_review": (
            "pending" if batches else "not-required"
        ),
        "prompt_customization_security_review_calls_planned": len(batches),
        "prompt_customization_security_review_calls_completed": 0,
        "prompt_customization_security_warnings": 0,
        "prompt_customization_security_blocking_reasons": 0,
        "prompt_customization_allow_risky": False,
        **_conversion_stats_defaults(),
    }


def _reviewed_shape_text(batches: list[ReviewBatch], mode: str) -> str:
    seen: set[str] = set()
    blocks: list[str] = []
    for batch in batches:
        for component in batch.components:
            if component.component in seen:
                continue
            seen.add(component.component)
            blocks.append(f"[{component.component}]\n{component.block_text}")
    return "\n\n".join(blocks) or f"[{mode}: no active blocks]"


def _build_scope_stats(
    config: dict,
    file_map: dict[str, dict],
    reachable_rels: set[str] | frozenset[str],
    documented_rels: set[str] | frozenset[str],
    agent_rels: set[str] | frozenset[str],
    entry_rel: str | None,
) -> dict:
    """Return the stable scope/reachability statistics contract."""
    scope = config.get("documentation_scope", "entry")
    per_file = initial_calls_per_file(config.get("analysis_mode", "single"))
    disconnected_paid = (
        len(set(agent_rels) - set(reachable_rels))
        if scope == "all" and entry_rel is not None
        else 0
    )
    return {
        "documentation_scope": scope,
        "entry_reachable": len(reachable_rels),
        "entry_disconnected": len(file_map) - len(reachable_rels),
        "entry_excluded": len(file_map) - len(documented_rels),
        "disconnected_paid_files": disconnected_paid,
        "disconnected_planned_calls": disconnected_paid * per_file,
    }


def _estimate_planned_input_tokens(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    config: dict,
    resolved_profile: ResolvedProfile | None = None,
) -> int:
    """Estimate input tokens for the planned LLM calls — a lower bound.

    The structure and dependency prompts are built from known inputs.  The
    documentation prompt embeds the other agents' responses, which do not
    exist yet, so it is estimated with empty analysis objects.  Uses the same
    centralized truncation helper as real execution so the estimated source
    size matches what would actually be sent.
    """
    from codedoc.agents import (
        dependency_agent,
        documentation_agent,
        file_documentation_agent,
        structure_agent,
    )

    analysis_mode = config.get("analysis_mode", "single")
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
        head_fraction = config.get("truncation_head_ratio", 0.70)
        content = truncate_for_llm(content, max_chars, head_fraction=head_fraction)
        if analysis_mode == "triple":
            structure_shape = (
                resolved_profile.resolve_block("structure", language)
                if resolved_profile is not None
                else None
            )
            dependency_shape = (
                resolved_profile.resolve_block("dependency", language)
                if resolved_profile is not None
                else None
            )
            documentation_shape = (
                resolved_profile.resolve_block("documentation", language)
                if resolved_profile is not None
                else None
            )
            prompts = (
                structure_agent.build_prompt(
                    rel_path, content, imports, language, structure_shape
                ),
                dependency_agent.build_prompt(
                    rel_path, content, imports, language, dependency_shape
                ),
                documentation_agent.build_prompt(
                    rel_path, content, language, {}, {}, documentation_shape
                ),
            )
        else:
            combined_shape = (
                resolved_profile.resolve_block("combined", language)
                if resolved_profile is not None
                else None
            )
            prompts = (
                file_documentation_agent.build_prompt(
                    rel_path, content, imports, language, combined_shape
                ),
            )
        for system, prompt in prompts:
            total += estimate_tokens(system) + estimate_tokens(prompt)
    return total
