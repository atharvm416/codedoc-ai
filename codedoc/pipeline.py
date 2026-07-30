"""Documentation pipeline lifecycle coordinator for codedoc.

Internal decomposition
----------------------
This module is now a thin lifecycle coordinator.  The heavy lifting has been
moved into cohesive, single-responsibility modules:

- :mod:`codedoc.core.resume` — selected-output and strict cross-format fallback
  loading, exact crash-recovery inspection, public→internal record
  reconstruction, and final documentation-record construction.
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
4. filesystem mutation and crash-recovery initialization;
5. provider creation;
6. execution;
7. final output, logs, cleanup, and statistics.

Compatibility note
--------------------------
Private helpers that moved to the modules above are re-exported here because
repository tests and documented integrations import them as
``codedoc.pipeline._name``.  These re-exports are deprecated; import from the
defining module instead.  No runtime warning is emitted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Mapping

from codedoc.agents.base_agent import truncate_for_llm
from codedoc.agents.orchestrator import Orchestrator, initial_calls_per_file
from codedoc.agents.prompt_customization_validation_agent import (
    PromptCustomizationValidationAgent,
)
from codedoc.agents.response_diagnostics import CorrectionLedger
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
    reconcile_planned_calls,
    restore_completed_tree_result,
)
from codedoc.core.execution_model import (
    REVIEW_OWNER,
    CallManifest,
    CallManifestTracker,
    build_call_manifest,
)
from codedoc.core.feasibility import build_feasibility_notes
from codedoc.core.file_division import (
    DivisionPlan,
    ReductionTreePlan,
    verify_provider_execution_identity,
)
from codedoc.core.loader import load_config
from codedoc.core.output import (
    inspect_output_ownership,
    preflight_output_accessibility,
    validate_distinct_artifact_paths,
    write_project_outputs,
)
from codedoc.core.planning import (
    PlanMaterials,
    PipelinePlan,
    PlanSourceInputs,
    build_pipeline_plan,
    normalize_force_files,
)
from codedoc.core.prompt_profiles import (
    ProfileResolution,
    REVIEW_SYSTEM,
    ResolvedProfile,
    ReviewBatch,
    build_resolved_profile,
    build_review_batches,
    classify_profile_action,
    resolve_profile_source,
    resolved_synthesis_shape,
)
from codedoc.core.queue import ProcessingQueue, STATUS_SKIPPED_INSUFFICIENT_SOURCE
from codedoc.core.record_meta import ANALYSIS_REVISION
from codedoc.core.resume import (
    RECOVERY_FILENAME,
    RecoveryState,
    _build_documentation_records,
    _load_existing_file_docs,
    build_recovery_identity,
    load_recovery_records_if_compatible,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.core.scanner import scan_files
from codedoc.core.usage import UsageAccumulator, estimate_tokens
from codedoc.llm.factory import create_provider, describe_provider_selection
from codedoc.llm.rate_limit_profile import get_rate_limit_profile
from codedoc.utils.errors import (
    ConfigError,
    ErrorReporter,
    LiveBackupWriteError,
    PromptCustomizationValidationError,
    UnrecoverableProviderError,
)
from codedoc.utils.logger import get_logger, set_level

# ---------------------------------------------------------------------------
# Compatibility re-exports (deprecated; import from the defining module)
# ---------------------------------------------------------------------------
# These private helpers moved to resume/discovery/execution.  They are
# re-exported here for compatibility because repository tests and documented
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
    *,
    confirm_risky: Callable[[tuple[str, ...]], bool] | None = None,
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

    if config_overrides is None:
        config_overrides = {}

    config = load_config(root, config_overrides)

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)
    dry_run = bool(config.get("dry_run", False))
    split_planning_preview = (
        dry_run and config.get("large_file_strategy", "truncate") == "split"
    )

    analysis_mode = config.get("analysis_mode", "single")
    # The configured extension keys (lowercased) gate ``per_extension`` overrides.
    # The language tag no longer selects a profile scope, so the language set is no
    # longer computed here.
    known_extensions = frozenset(e.lower() for e in config["extension_language_map"])
    # Resolve and classify the prompt profile BEFORE entry resolution,
    # artifact/ownership inspection, existing-output reads, source scanning, graph
    # building, and planning.  Both calls are read-only and provider-free.  A
    # customized single-only structure selected in triple mode is executable — its
    # documentation is resolved by deterministic projection, never a paid routing
    # conversion.
    profile_resolution = resolve_profile_source(
        config,
        root,
        known_extensions=known_extensions,
        active_mode=analysis_mode,
    )
    profile_action = classify_profile_action(profile_resolution.profile, analysis_mode)

    entry_resolution_source = _resolve_entry_and_docs(root, config)
    resolved_profile = build_resolved_profile(profile_action, analysis_mode)
    # Empty/no-scanned-file paths use the same manifest helper as every other
    # path and report zero calls throughout.
    _empty_manifest = build_call_manifest([], [], analysis_mode)
    no_work_profile_stats = {
        "prompt_profile_source": profile_resolution.source,
        "prompt_profile_active": False,
        "prompt_profile_affected_files": 0,
        "prompt_customization_security_review": "not-required",
        "prompt_customization_security_review_calls_planned": 0,
        "prompt_customization_security_review_calls_completed": 0,
        "prompt_customization_security_review_calls_attempted": 0,
        "prompt_customization_security_warnings": 0,
        "prompt_customization_security_blocking_reasons": 0,
        "prompt_customization_feasibility_notes": 0,
        "prompt_customization_feasibility_advisories": (),
        "documentation_calls_planned": 0,
        "documentation_calls_attempted": 0,
        "total_calls_planned": 0,
        "max_planned_calls": int(config.get("max_planned_calls", 0) or 0),
        "max_planned_calls_exceeded": False,
        "call_manifest_digest": _empty_manifest.digest,
        "attempted_logical_calls": 0,
        "planned_calls_not_attempted": 0,
        "additional_attempts": 0,
    }

    output_format = config.get("output_format", "json")
    logger.info("Output format: %s", output_format)

    if dry_run:
        logger.info("Dry run: planning only — no writes, no provider, no LLM calls.")

    output_dir = root / config["output_dir"]

    json_filename = config.get("output_json_filename", "codedoc.json")
    md_filename = config.get("output_md_filename", "codedoc.md")
    # Read candidates always include the deterministic JSON/Markdown pair.  Write
    # targets remain limited to the selected format.  This distinction lets a
    # missing selected target reuse its validated opposite-format sibling without
    # broadening ownership checks, recovery identity, or the final write set.
    json_candidate = output_dir / json_filename
    md_candidate = output_dir / md_filename
    json_target = json_candidate if output_format in ("json", "both") else None
    md_target = md_candidate if output_format in ("md", "both") else None
    # The single fixed crash-recovery file, staged separately from the stable
    # output for every format.  There is no candidate walk, numbered suffix, or
    # legacy sibling: absent means a fresh recovery state; an owned in-progress
    # file is reused only when its versioned identity matches this run.
    recovery_path = output_dir / RECOVERY_FILENAME

    # Reject generated-artifact path collisions before any scan or mutation.  The
    # recovery file is its own ``live_backup`` artifact, always distinct from the
    # exact JSON and Markdown targets.
    artifact_paths: dict[str, Path | None] = {
        "json": json_target,
        "markdown": md_target,
        "live_backup": recovery_path,
    }
    validate_distinct_artifact_paths(artifact_paths)

    # Read-only ownership inspection of the exact final targets.  A real run fails
    # fast before any filesystem side effect, scanning, or LLM call when a target
    # is foreign-owned; a dry run records the conflicts and reports them.  The
    # recovery file is not inspected here.
    ownership_conflicts = inspect_output_ownership(
        output_dir, output_format, json_filename, md_filename
    )
    if ownership_conflicts and not dry_run:
        raise ConfigError(ownership_conflicts[0]["message"])

    # In-memory issue reporter; nothing is persisted to disk.
    error_reporter = ErrorReporter()

    # Stable records come from an existing selected target, or from its strict,
    # deterministic opposite-format sibling only when the selected target is
    # absent.  The both-mode cross-document identity check runs inside this
    # helper. Compatible recovery records are overlaid after selection/profile
    # resolution (below).
    existing_docs = _load_existing_file_docs(
        json_candidate, md_candidate, output_format
    )
    if split_planning_preview:
        existing_docs = {
            rel_path: record
            for rel_path, record in existing_docs.items()
            if not record.get("_large_file_identity")
        }

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

    def _scan_source_files() -> list[dict]:
        return scan_files(
            root,
            extension_language_map=config["extension_language_map"],
            max_file_size_kb=config["max_file_size_kb"],
            skip_dirs=_scan_skip_dirs,
            ignore_paths=config.get("ignore_paths"),
            follow_symlinks=config.get("follow_symlinks", False),
        )

    all_files = _scan_source_files()
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
                "initial_calls_per_file": _initial_calls_per_file_for_stats(
                    config.get("analysis_mode", "single"),
                    config.get("large_file_strategy", "truncate"),
                    None,
                ),
                "documentation_scope": config.get("documentation_scope", "entry"),
                "entry_reachable": 0,
                "entry_disconnected": 0,
                "disconnected_paid_files": 0,
                "disconnected_planned_calls": 0,
                "would_process": 0,
                "would_call_llm_for": 0,
                "would_skip_insufficient_source": 0,
                "unchanged": 0,
                "would_reuse": 0,
                "would_resume": 0,
                "forced": 0,
                "estimated_calls": 0,
                "estimated_input_tokens": 0,
                "estimate_is_lower_bound": config.get("analysis_mode", "single") == "triple",
                "max_files": int(config.get("max_files", 0) or 0),
                "max_files_candidate_files": 0,
                "max_files_exceeded": False,
                "ownership_conflicts": ownership_conflicts,
                "output_dir": str(output_dir),
                "output_files": [],
                **no_work_profile_stats,
                **_split_division_stats(config, None, None),
            }
        preflight_output_accessibility(output_dir)
        return {
            "checked": 0,
            "failed": 0,
            "skipped": 0,
            "skipped_insufficient_source": 0,
            "entry_excluded": 0,
            "analysis_mode": config.get("analysis_mode", "single"),
            "initial_calls_per_file": _initial_calls_per_file_for_stats(
                config.get("analysis_mode", "single"),
                config.get("large_file_strategy", "truncate"),
                None,
            ),
            "output_dir": str(output_dir),
            "live_backup_path": None,
            "issues_recorded": 0,
            "rate_limit_warnings": [],
            "documentation_scope": config.get("documentation_scope", "entry"),
            "entry_reachable": 0,
            "entry_disconnected": 0,
            "disconnected_paid_files": 0,
            "disconnected_planned_calls": 0,
            **no_work_profile_stats,
            **_split_division_stats(config, None, None),
        }

    graph, file_map, unresolved_imports_by_path = _build_graph(all_files, root, error_reporter)
    reachable_rels, documented_rels, entry_rel = _select_files(
        root, config, graph, file_map
    )

    def _rebuild_source_inputs() -> PlanSourceInputs:
        """Re-run scanning, parsing, graph construction, and entry selection.

        Planning detects a file whose content changed between its routing hash
        and its canonical snapshot, but the dependency graph was parsed earlier
        still — so routing may itself have been derived from the revision that
        is now stale.  Re-reading snapshots against that old graph would leave
        content and dependency routing describing different revisions, so the
        pipeline (which owns scanning and graph construction) rebuilds them
        here.  Invoked at most once per run, only on a detected stale revision.

        The rebuild is triggered only by a content change, so the entry/scope
        identity the recovery file was already bound to above is unaffected.
        """
        nonlocal graph, file_map, unresolved_imports_by_path
        nonlocal reachable_rels, documented_rels, entry_rel
        graph, file_map, unresolved_imports_by_path = _build_graph(
            _scan_source_files(), root, error_reporter
        )
        reachable_rels, documented_rels, entry_rel = _select_files(
            root, config, graph, file_map
        )
        return PlanSourceInputs(
            file_map=file_map,
            graph=graph,
            selected_rels=documented_rels,
            entry_rel=entry_rel,
        )

    # Normalize forced paths against the project root (raises
    # ConfigError for paths outside the root).
    forced_paths = normalize_force_files(config.get("force_files") or [], root)

    # Build the versioned recovery identity from the now-known project root,
    # selected targets, entry, scope, and analysis mode/revision.  The run-level
    # identity no longer binds a profile-wide digest: per-file
    # ``_prompt_profile_digest`` values in CACHE_IDENTITY_KEYS already re-validate
    # every recovered record individually, so narrowing this field set keeps
    # version 1 (a backward-compatible narrowing) and never discards resumable
    # work for an unrelated profile edit or a newly added file.
    # The effective large-file strategy is part of the identity because split
    # recovery carries partial chunk checkpoints a truncate run can neither read
    # nor rewrite.  Binding it makes the cross-strategy boundary fail closed
    # here — before SafeWriter initialization and provider creation — instead of
    # letting a truncate flush silently drop already-paid split checkpoints.
    large_file_strategy = config.get("large_file_strategy", "truncate")
    recovery_identity = build_recovery_identity(
        project_root=root,
        json_target=json_target,
        md_target=md_target,
        entry_file=entry_rel,
        documentation_scope=config.get("documentation_scope", "entry"),
        analysis_mode=analysis_mode,
        analysis_revision=ANALYSIS_REVISION,
        large_file_strategy=large_file_strategy,
    )
    # D2 explicit local gate: partial (split-node) recovery is enabled only
    # from the valid effective split route, not the bare requested string —
    # analysis_mode 'triple' can never legitimately reach here with
    # large_file_strategy 'split' (loader._validate rejects it first), but the
    # check stays visible here as defense-in-depth rather than relying solely
    # on that earlier gate.
    include_partial_recovery = (
        large_file_strategy == "split"
        and analysis_mode == "single"
        and not dry_run
    )
    # Inspect the single exact recovery file, read-only.  A real run blocks on an
    # incompatible / foreign / malformed / completed recovery file; a dry run
    # never mutates it and treats an incompatible file as non-resumable rather
    # than blocking planning.
    recovery_state = RecoveryState()
    try:
        recovery_state = load_recovery_records_if_compatible(
            recovery_path,
            recovery_identity,
            include_partial_files=include_partial_recovery,
        )
    except ConfigError:
        if not dry_run:
            raise
    recovery_records = recovery_state.records_by_path()
    if split_planning_preview:
        recovery_records = {}
    # Every structurally parseable partial. Planning narrows this to the subset
    # that is actually reusable; the writer is seeded from that narrower set.
    recovered_partials_by_path = (
        {}
        if split_planning_preview
        else {state.rel_path: state for state in recovery_state.partial_files}
    )
    if recovery_records:
        logger.info(
            "Resuming: overlaying %d compatible record(s) from '%s'.",
            len(recovery_records),
            recovery_path.name,
        )
    # Overlay compatible recovery records onto the stable baseline before planning.
    existing_docs = {**existing_docs, **recovery_records}

    # One shared plan drives both dry-run and real execution.  A stale source
    # revision rebuilds the complete source-dependent inputs once through
    # ``_rebuild_source_inputs`` before a second change fails deterministically.
    plan, materials = build_pipeline_plan(
        file_map=file_map,
        graph=graph,
        selected_rels=documented_rels,
        entry_rel=entry_rel,
        existing_docs=existing_docs,
        forced_paths=forced_paths,
        config=config,
        resolved_profile=resolved_profile,
        rebuild_source_inputs=_rebuild_source_inputs,
        recovered_partials=recovered_partials_by_path,
    )

    # Capacity-blocked requested-split files have no authorized provider
    # action (D8): a real run raises one deterministic ConfigError listing
    # every blocked (rel_path, reason) pair before any writer/provider
    # creation; a dry run reports the same pairs without mutation (see
    # _build_dry_run_stats / _split_division_stats) and exits 0.
    if materials.division_blocked and not dry_run:
        raise ConfigError(_blocked_split_files_message(materials.division_blocked))

    # Derived from the file set/graph/selection the plan was actually built
    # from, so a stale-revision rebuild above is reflected here rather than
    # leaving pre-rebuild counts and queue order in place.
    entry_source = _final_entry_source(entry_resolution_source, entry_rel)
    # Count of scanned files excluded by entry-reachability selection (A1).
    # Zero when no entry is in effect (selected_rels == all scanned files).
    # Surfaced in stats so the CLI can report it at the run summary.
    entry_excluded = len(file_map) - len(documented_rels)
    # Queue/topological order for the selected file set (all selected, not just agent files).
    ordered_selected = [p for p in graph.topological_order() if p in documented_rels]

    # The exact set of file scopes reachable by a planned agent file drives which
    # profile components (common and per-extension) are reviewed.  An explicitly
    # empty set means no planned agent file is reachable, so no review is built.
    planned_scopes = frozenset(
        resolved_profile.scope_for(file_map[rel])
        for rel in plan.unpaid_action_rels
    )
    review_batches = build_review_batches(resolved_profile, planned_scopes)
    feasibility_notes = build_feasibility_notes(resolved_profile, planned_scopes)

    # One canonical call manifest, shared by dry-run and real execution alike.
    # Built and attached before any cap is enforced or any provider-side effect
    # can occur.
    call_manifest = build_call_manifest(
        review_batches,
        plan.agent_rels,
        analysis_mode,
        materials.division_plans,
        materials.reduction_trees,
        materials.tree_states,
    )
    plan = plan.with_call_manifest(
        call_manifest,
        int(config.get("max_planned_calls", 0) or 0),
        materials.division_plans,
        materials.reduction_trees,
        materials.tree_states,
    )
    insufficient_source_rels = frozenset(materials.insufficient_source_reasons)
    max_files_candidate_count = len(plan.unpaid_action_rels)
    call_tracker = CallManifestTracker(call_manifest)
    # Bounded: digest and counts only — never raw source, prompts, custom
    # instructions, credentials, or provider responses.
    logger.debug(
        "Call manifest built: digest=%s total=%d review=%d documentation=%d "
        "max_planned_calls=%d",
        plan.call_manifest_digest,
        plan.total_calls_planned,
        plan.review_calls_planned,
        plan.documentation_calls_planned,
        plan.max_planned_calls,
    )
    if config.get("large_file_strategy", "truncate") == "split" and analysis_mode == "single":
        logger.debug(
            "Split call categories: file=%d leaf=%d reduction=%d synthesis=%d",
            plan.file_documentation_calls_planned,
            plan.unit_documentation_calls_planned,
            plan.file_reduction_calls_planned,
            plan.synthesis_calls_planned,
        )

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
            call_manifest=call_manifest,
            recovery_resumed_paths=frozenset(recovery_records),
            materials=materials,
            feasibility_notes=feasibility_notes,
        )

    scope_stats = _build_scope_stats(
        config,
        file_map,
        reachable_rels,
        documented_rels,
        plan.agent_rels,
        entry_rel,
        call_manifest=call_manifest,
        materials=materials,
    )

    # Paid-file safety cap: enforced after the complete plan exists and before
    # any mutation, writer initialization, or provider creation.
    if plan.max_files_exceeded:
        raise ConfigError(
            f"This run has {max_files_candidate_count} documentation-call candidate(s), "
            f"which exceeds the configured max_files limit of {plan.max_files}. "
            "Inspect the plan first with --dry-run, and raise --max-files only "
            "after reviewing it."
        )

    # Whole-run call-authorization cap: independent of max_files, checked before
    # usage accounting, provider creation, or any confirmation callback.
    if plan.max_planned_calls_exceeded:
        if config.get("large_file_strategy", "truncate") == "split" and analysis_mode == "single":
            call_breakdown = (
                f"{plan.review_calls_planned} prompt-customization review, "
                f"{plan.file_documentation_calls_planned} file documentation, "
                f"{plan.unit_documentation_calls_planned} leaf documentation, "
                f"{plan.file_reduction_calls_planned} file reduction, "
                f"{plan.synthesis_calls_planned} file synthesis"
            )
        else:
            call_breakdown = (
                f"{plan.review_calls_planned} prompt-customization review, "
                f"{plan.file_documentation_calls_planned} file documentation"
            )
        raise ConfigError(
            f"This run has {plan.total_calls_planned} initially planned LLM "
            f"call(s) ({call_breakdown}), which "
            f"exceeds the configured max_planned_calls limit of "
            f"{plan.max_planned_calls}. Inspect the plan first with --dry-run, "
            "and raise --max-planned-calls only after reviewing it."
        )

    usage = UsageAccumulator()
    # One run-level correction ledger, initialized with the resolved opt-in flag,
    # constructed beside the usage accumulator and threaded through the
    # orchestrator to the single shared correction component.
    correction_enabled = bool(config.get("response_correction_enabled", False))
    correction_ledger = CorrectionLedger(correction_enabled)
    llm = None
    review_stats = _base_profile_stats(
        resolved_profile,
        profile_resolution,
        plan,
        file_map,
        review_batches,
        feasibility_notes=feasibility_notes,
    )
    if review_batches:
        # This is a mandatory, non-overridable, probabilistic semantic
        # standards/safety gate. A well-formed TOO_RISKY verdict always blocks;
        # deterministic validation and strict cleaners are the additional
        # non-overridable structural boundary.  It is probabilistic and must not
        # be marketed as a complete security guarantee.
        review_provider, review_model = describe_provider_selection(config)
        print(
            "Prompt customization standards/safety review will make "
            f"{len(review_batches)} paid provider call(s) using "
            f"provider={review_provider}, model={review_model}. No profile content "
            "is persisted in run statistics.",
            flush=True,
        )
        logger.debug(
            "Planned call(s) for owner=%s category=prompt-review call_ids=%s",
            REVIEW_OWNER,
            [c.call_id for c in call_manifest.calls if c.category == "prompt-review"],
        )
        llm = create_provider(config)
        verify_provider_execution_identity(
            llm,
            config,
            materials.provider_identity,
        )
        reviewer = PromptCustomizationValidationAgent(
            llm, usage=usage, call_tracker=call_tracker
        )
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
            _set_manifest_reconciliation(
                review_stats, usage, plan, call_tracker
            )
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
        _set_manifest_reconciliation(review_stats, usage, plan, call_tracker)
        review_stats["prompt_customization_security_warnings"] = len(outcome.warnings)
        review_stats["prompt_customization_security_blocking_reasons"] = len(
            outcome.reasons
        )
        if outcome.verdict == "SAFE":
            review_stats["prompt_customization_security_review"] = "safe"
            print("Prompt customization standards/safety review: SAFE — continuing.")
        elif outcome.verdict == "RISKY":
            print("Prompt customization standards/safety review: RISKY — confirmation required:")
            for warning in outcome.warnings:
                print(f"  - {warning}")
            confirmed = False
            if confirm_risky is not None:
                try:
                    confirmed = confirm_risky(tuple(outcome.warnings)) is True
                except Exception:
                    confirmed = False
            if not confirmed:
                review_stats["prompt_customization_security_review"] = (
                    "risky-confirmation-blocked"
                )
                raise PromptCustomizationValidationError(
                    "Prompt customization standards/safety review: RISKY — "
                    "explicit confirmation was not provided. No documentation "
                    "call or persistent mutation was performed. Revise the "
                    "flagged instructions or confirm this run interactively.",
                    stats=review_stats,
                )
            review_stats["prompt_customization_security_review"] = "risky-confirmed"
            print("Medium-risk customization confirmed for this run — continuing.")
        else:
            # TOO_RISKY always stops before any documentation call or mutation.
            # There is no override.
            shown = _reviewed_shape_text(review_batches, analysis_mode)
            review_stats["prompt_customization_security_review"] = "too-risky-blocked"
            raise PromptCustomizationValidationError(
                "Prompt customization standards/safety review: TOO RISKY — "
                "blocked and not applied.\n"
                + "\n".join(f"- {reason}" for reason in outcome.reasons)
                + f"\nCustomization under review (effective requested-shape JSON for {analysis_mode}):\n"
                + shown
                + "\nRevise the flagged instructions and re-run. Deterministic "
                "validation, the strict cleaners, and this semantic review cannot "
                "be overridden.",
                stats=review_stats,
            )

    # ------------------------------------------------------------------
    # Mutation boundary — everything below may write to the filesystem.
    # ------------------------------------------------------------------
    # An output accessibility probe before the documentation provider is
    # created.  Runs for every real finalization path — including all-reused
    # runs that still rewrite stable output — but never for dry-run (which
    # returns above).  Raises a classified OutputError if the directory cannot
    # be written, so a persistent permission/space failure is caught before
    # paid documentation work instead of after it.
    #
    # Directory creation belongs to the probe itself: doing it here would put
    # the mkdir outside the probe's classified OSError boundary, so an absent
    # directory under an unwritable parent would escape as a raw PermissionError.
    #
    # `llm is not None` here means the prompt-customization review already ran
    # and was billed, so the diagnostic must not claim a provider-free abort.
    preflight_output_accessibility(output_dir, provider_contacted=llm is not None)

    # Always-on crash-recovery writer.  Targets the single fixed recovery file for
    # every format; the stable output is untouched until finalization.  The
    # versioned run identity is emitted inside every flush so a partially written
    # recovery file already carries it.
    recorder = SafeWriter(
        recovery_path, output_format, entry_rel, file_map, recovery_identity
    )
    recorder.set_queue_order(ordered_selected)

    # Seed the writer with the same merged reuse set planning consumed (stable
    # baseline + compatible recovery overlay) so every partial flush of the
    # recovery file is a self-contained resumable snapshot.
    # Seed the writer with the exact retention-eligible tree-state set rather
    # than letting it re-read every parseable partial.  Planning already
    # narrowed the recovered nodes to those that are current, selected, and
    # dependency-valid (validate_node_for_tree); a rejected, stale, or
    # unselected checkpoint must never keep the recovery file alive after a
    # successful finalization.
    recorder.load(
        preloaded=existing_docs,
        preloaded_partials=dict(materials.tree_states),
    )

    skipped = len(plan.unchanged_rels)
    if skipped > 0:
        logger.info("Incremental mode: skipping %d unchanged file(s)", skipped)

    # Materialize reuse records exactly as the plan routed them.
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
    resumed = len(set(recovery_records) | set(materials.tree_states))

    agent_rels = set(plan.agent_rels)
    locally_restored = 0
    for rel_path, tree_state in materials.tree_states.items():
        reduction_tree = materials.reduction_trees[rel_path]
        if reduction_tree.final_node.node_id not in tree_state.by_id():
            continue
        result = restore_completed_tree_result(
            materials.execution_requests[rel_path],
            materials.division_plans[rel_path],
            reduction_tree,
            tree_state,
        )
        recorder.record(
            rel_path,
            result,
            materials.execution_requests[rel_path].content_hash,
        )
        new_results[rel_path] = result
        agent_rels.discard(rel_path)
        locally_restored += 1
        logger.info(
            "Finalized synthesized split recovery for %s without provider work.",
            rel_path,
        )
    for rel_path, reason in materials.insufficient_source_reasons.items():
        logger.info(
            "[SKIP] %s | insufficient source: %s — not sent to provider",
            rel_path,
            reason,
        )
    rate_limit_warnings: list[dict] = []

    if not agent_rels:
        logger.info("All selected files are up-to-date or reused from cached content.")
        stats: dict = {
            "checked": locally_restored,
            "failed": 0,
            "skipped_insufficient_source": len(insufficient_source_rels),
            "skipped": skipped,
            "reused": reused,
            "resumed": resumed,
            "analysis_mode": analysis_mode,
            "entry_source": entry_source,
            "entry_excluded": entry_excluded,
            **scope_stats,
            "output_dir": str(output_dir),
            "rate_limit_warnings": rate_limit_warnings,
            **no_work_profile_stats,
            **review_stats,
            **_split_division_stats(config, materials, plan),
        }
        _set_plan_counters(
            stats,
            plan,
            provider_free_skips=len(insufficient_source_rels),
        )
        output_files = write_project_outputs(
            _build_documentation_records(
                documented_rels - insufficient_source_rels,
                dict(materials.content_hashes),
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
        if not recorder.has_partial_state():
            recorder.delete()
        _set_issue_stats(stats, error_reporter, recovery_path)
        _set_usage_stats(
            stats,
            usage,
            plan,
            config,
            correction_ledger,
            call_tracker=call_tracker,
        )
        return stats

    # Build the agent-file queue in topological order.
    queue = ProcessingQueue()
    for rel_path in graph.topological_order():
        if rel_path in agent_rels:
            queue.add(file_map[rel_path])

    # Create the LLM provider after crash recovery is initialized.
    # initialize_empty() must be called before provider creation so recovery
    # exists even if provider init fails.
    recorder.initialize_empty()

    try:
        if llm is None:
            llm = create_provider(config)
        verify_provider_execution_identity(
            llm,
            config,
            materials.provider_identity,
        )
        logger.info("LLM provider: %s", llm.provider_name)
    except Exception as exc:
        error_reporter.record(exc, context="LLM provider init")
        raise
    except KeyboardInterrupt as exc:
        # An interrupt during provider init still happens after the
        # recovery file was initialized; name it for the CLI (see below).
        if recovery_path.exists():
            exc.recovery_path = str(recovery_path)
        raise

    orchestrator = Orchestrator(
        llm,
        parallel=config.get("parallel_agents", True),
        max_content_chars=config.get("max_content_chars", 12000),
        usage=usage,
        analysis_mode=config.get("analysis_mode", "single"),
        truncation_head_ratio=config.get("truncation_head_ratio", 0.70),
        response_correction_enabled=correction_enabled,
        correction_ledger=correction_ledger,
        call_tracker=call_tracker,
    )
    stats = {
        "checked": locally_restored,
        "failed": 0,
        "skipped_insufficient_source": len(insufficient_source_rels),
        "skipped": skipped,
        "reused": reused,
        "resumed": resumed,
        "analysis_mode": analysis_mode,
        "entry_source": entry_source,
        "entry_excluded": entry_excluded,
        **scope_stats,
        "rate_limit_warnings": rate_limit_warnings,
        **review_stats,
        **_split_division_stats(config, materials, plan),
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

    # Build the execution context from resolved configuration.  The
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
        execution_requests=materials.execution_requests,
        division_plans=materials.division_plans,
        reduction_trees=materials.reduction_trees,
        provider_identity=materials.provider_identity,
    )
    try:
        execute_agent_files(context)
    except UnrecoverableProviderError as exc:
        # A confirmed unrecoverable provider abort (terminal
        # billing/credentials/model/access, or a bounded zero-progress rate
        # limit).  Record it in memory, then re-raise for the CLI to present.
        # Deliberately do NOT call write_project_outputs(...) or recorder.delete()
        # on this path: the recovery file must stay intact and resumable.
        error_reporter.record(exc, context="provider abort")
        _set_usage_stats(
            stats,
            usage,
            plan,
            config,
            correction_ledger,
            call_tracker=call_tracker,
        )
        exc.stats = dict(stats)
        raise
    except LiveBackupWriteError as exc:
        # The dedicated recovery file could not be persisted, so crash-safety no
        # longer holds and execution stopped scheduling paid work.  The last valid
        # recovery file is preserved by the atomic writer.  Record the failure in
        # memory and print the recovery path so the user can act.
        error_reporter.record(exc, context="crash-recovery write")
        recovery_note = (
            f"\n  Completed work is preserved in: {recovery_path}"
            if recovery_path.exists()
            else ""
        )
        print(
            f"\nError: {exc}{recovery_note}",
            file=sys.stderr,
            flush=True,
        )
        raise
    except KeyboardInterrupt as exc:
        # The run was interrupted mid-processing.  The stable output was
        # never opened; completed work is staged in the dedicated recovery file.
        # Attach the exact selected recovery path (only when it exists on disk)
        # so the CLI can name it in the interrupt message, then re-raise the same
        # exception unchanged.  No suffix walk, candidate creation, or filesystem
        # mutation happens here.
        if recovery_path.exists():
            exc.recovery_path = str(recovery_path)
        raise

    stats["output_dir"] = str(output_dir)
    _set_plan_counters(
        stats,
        plan,
        provider_free_skips=len(insufficient_source_rels),
    )
    skipped_rels = set(insufficient_source_rels) | {
        rel
        for rel, status in queue.snapshot().items()
        if status == STATUS_SKIPPED_INSUFFICIENT_SOURCE
    }
    output_files = write_project_outputs(
        _build_documentation_records(
            documented_rels - skipped_rels,
            dict(materials.content_hashes),
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
    # Clean completion: the stable output is written above; only now delete the
    # dedicated recovery file (all formats).  A deletion OSError raises
    # OutputError and leaves both the stable output and the recovery file intact.
    if not recorder.has_partial_state():
        recorder.delete()
    _set_issue_stats(stats, error_reporter, recovery_path)
    _set_usage_stats(
        stats, usage, plan, config, correction_ledger,
        call_tracker=call_tracker,
    )

    logger.info(
        "Done. checked=%d failed=%d skipped=%d output=%s",
        stats["checked"],
        stats["failed"],
        stats["skipped"],
        output_dir,
    )

    if error_reporter.has_issues():
        logger.info("%d issue(s) recorded.", error_reporter.issue_count())

    return stats


def _set_issue_stats(
    stats: dict,
    error_reporter: ErrorReporter,
    recovery_path: Path | None,
) -> None:
    """Populate in-memory issue/recovery stats keys on *stats* in-place."""
    stats["issues_recorded"] = error_reporter.issue_count()
    if recovery_path is not None and recovery_path.exists():
        stats["live_backup_path"] = str(recovery_path.resolve())
    else:
        stats["live_backup_path"] = None


def _final_entry_source(resolution_source: str, entry_rel: str | None) -> str:
    """Return the public entry-source value after auto-detection has run."""
    if resolution_source in {"explicit", "recovered"}:
        return resolution_source
    return "auto-detected" if entry_rel else "none"


def _set_plan_counters(
    stats: dict,
    plan: PipelinePlan,
    *,
    provider_free_skips: int = 0,
) -> None:
    """Populate provider-free plan counters needed by the public view.

    ``unattempted_files`` is a state of ``plan.agent_rels`` only.  A file
    rejected by the provider-free source precheck was removed from
    ``agent_rels`` during planning, so it never had an ``agent_rels`` state and
    must not be subtracted from that population a second time — only the
    execution-time defensive guard's skips are.  Both skip populations still
    report through the one ``skipped_insufficient_source`` statistic, so the
    selected-file completion partition reconciles exactly.  A capacity-blocked
    split file (``plan.division_blocked``) never reaches ``agent_rels`` either
    (D8), so it needs no separate accounting here.
    """
    stats["files_scanned"] = len(plan.scanned_rels)
    stats["files_selected"] = len(plan.documented_rels)
    execution_skips = max(
        0, stats.get("skipped_insufficient_source", 0) - provider_free_skips
    )
    execution_failures = stats.get("failed", 0)
    stats["unattempted_files"] = max(
        0,
        len(plan.agent_rels)
        - stats.get("checked", 0)
        - execution_failures
        - execution_skips,
    )


def _set_usage_stats(
    stats: dict,
    usage: UsageAccumulator,
    plan: PipelinePlan,
    config: dict,
    correction_ledger: CorrectionLedger | None = None,
    *,
    call_tracker: CallManifestTracker,
) -> None:
    """Populate planned/actual usage keys on *stats* in-place.

    Token figures are character-heuristic estimates, not tokenizer counts.
    """
    analysis_mode = config.get("analysis_mode", "single")
    per_file = _initial_calls_per_file_for_stats(
        analysis_mode, config.get("large_file_strategy", "truncate"), plan
    )
    stats.update(usage.snapshot())
    # Correction statistics are run-level metadata, snapshot beside the existing
    # documentation-call counter.  Every correction provider call is one paid
    # attempt already counted in ``attempted_calls`` and therefore a subset of
    # ``documentation_calls_attempted``; these keys are reported additionally, not
    # subtracted, so the call-category invariant below is preserved unchanged.
    if correction_ledger is not None:
        stats.update(correction_ledger.snapshot())
    stats["analysis_mode"] = analysis_mode
    stats["initial_calls_per_file"] = per_file
    stats["planned_calls"] = plan.documentation_calls_planned
    stats["planned_files"] = len(plan.agent_rels)
    # Documentation-call category accounting.  Every provider attempt is exactly
    # one of two categories (documentation or mandatory customization review).
    # Review tracks its own attempts; documentation is the remainder so the two
    # categories reconcile to ``attempted_calls`` exactly.
    stats["documentation_calls_planned"] = plan.documentation_calls_planned
    review_attempted = stats.get("prompt_customization_security_review_calls_attempted", 0)
    stats["documentation_calls_attempted"] = max(
        0, stats.get("attempted_calls", 0) - review_attempted
    )
    # Resolved from config so env/config-enabled partial mode reaches the CLI.
    stats["allow_partial"] = bool(config.get("allow_partial", False))

    # The canonical call manifest's own counts/cap/digest — identical fields to
    # the dry-run surface, from the same immutable plan.
    stats["total_calls_planned"] = plan.total_calls_planned
    stats["max_planned_calls"] = plan.max_planned_calls
    stats["max_planned_calls_exceeded"] = plan.max_planned_calls_exceeded
    stats["call_manifest_digest"] = plan.call_manifest_digest

    # Planned-vs-attempted-logical reconciliation. Valid for a clean completion,
    # an allow_partial completion, or (independently, via direct queue snapshot
    # inspection) a terminal abort — never requires execution to continue.
    _set_manifest_reconciliation(stats, usage, plan, call_tracker)


def _set_manifest_reconciliation(
    stats: dict,
    usage: UsageAccumulator,
    plan: PipelinePlan,
    call_tracker: CallManifestTracker,
) -> None:
    """Attach canonical planned-versus-attempted logical-call statistics.

    The counts come from manifest-ID consumption, never from queue or file
    status, so this is equally valid after a clean completion, an
    ``allow_partial`` completion, and a terminal abort.  The reconciliation
    equation itself is owned by
    :func:`~codedoc.core.execution.reconcile_planned_calls`; this only supplies
    the plan's manifest fields and the tracker's attempted count.
    """
    stats["total_calls_planned"] = plan.total_calls_planned
    stats["max_planned_calls"] = plan.max_planned_calls
    stats["max_planned_calls_exceeded"] = plan.max_planned_calls_exceeded
    stats["call_manifest_digest"] = plan.call_manifest_digest
    stats.update(
        reconcile_planned_calls(
            total_calls_planned=plan.total_calls_planned,
            attempted_logical_calls=call_tracker.snapshot()["attempted_logical_calls"],
            attempted_calls=usage.attempted_calls,
        )
    )


_SCOPE_PRECEDENCE = ("extension", "common", "built-in")


def _prompt_profile_scope_counts(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    resolved_profile: ResolvedProfile | None,
) -> dict[str, int]:
    """Count unpaid-action files by the strongest scope their bundle resolved to.

    Counted over ``plan.unpaid_action_rels`` with precedence ``extension > common >
    built-in`` (a triple-mode file is counted once, under the strongest scope any
    of its three agents resolved to), so the counts sum to the files that can
    still cause provider work.
    """
    counts = {"extension": 0, "common": 0, "built-in": 0}
    for rel in plan.unpaid_action_rels:
        if resolved_profile is None:
            counts["built-in"] += 1
            continue
        bundle = resolved_profile.resolve_bundle(
            resolved_profile.scope_for(file_map[rel])
        )
        strongest = "built-in"
        for selection in bundle.selections.values():
            if _SCOPE_PRECEDENCE.index(selection.scope) < _SCOPE_PRECEDENCE.index(
                strongest
            ):
                strongest = selection.scope
        counts[strongest] += 1
    return counts


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
    call_manifest: CallManifest | None = None,
    recovery_resumed_paths: frozenset[str] = frozenset(),
    materials=None,
    *,
    feasibility_notes: tuple[str, ...],
) -> dict:
    """Build the read-only dry-run stats dict from the shared plan."""
    scope_stats = _build_scope_stats(
        config,
        file_map,
        reachable_rels,
        set(plan.documented_rels),
        plan.agent_rels,
        plan.entry_rel,
        call_manifest=call_manifest,
        materials=materials,
    )
    analysis_mode = config.get("analysis_mode", "single")
    per_file = _initial_calls_per_file_for_stats(
        analysis_mode, config.get("large_file_strategy", "truncate"), plan
    )
    # single mode embeds only known inputs, so its input estimate is exact;
    # triple mode's documentation prompt estimate is a lower bound.
    estimate_is_lower_bound = analysis_mode == "triple"
    # Correction is opt-in.  The worst case is one extra call per planned per-agent
    # documentation call; it is 0 when correction is disabled.  The baseline
    # estimate stays the expected charge; the ceiling below is a worst case.
    correction_enabled = bool(config.get("response_correction_enabled", False))
    skip_rels: set[str] = set(
        materials.insufficient_source_reasons if materials is not None else ()
    )
    execution_requests = (
        materials.execution_requests if materials is not None else {}
    )
    frozen_skip_rels = frozenset(skip_rels)
    remaining = len(plan.unpaid_action_rels)
    # From the one canonical manifest, never re-derived here: the exact sum of
    # every documentation-related category (ordinary file-documentation plus
    # any split file's leaf/reduction/synthesis calls), already reconciled by
    # with_call_manifest().
    estimated_calls = plan.documentation_calls_planned
    correction_possible_max = estimated_calls if correction_enabled else 0
    review_batches = review_batches or []
    profile_stats = (
        _base_profile_stats(
            resolved_profile,
            profile_resolution,
            plan,
            file_map,
            review_batches,
            feasibility_notes=feasibility_notes,
        )
        if resolved_profile is not None and profile_resolution is not None
        else {}
    )
    if review_batches:
        profile_stats["prompt_customization_security_review"] = "pending"
    if profile_stats:
        # Dry-run projects the documentation plan without attempting any call.
        profile_stats["documentation_calls_planned"] = (
            plan.documentation_calls_planned
        )
    return {
        "dry_run": True,
        "scanned": len(plan.scanned_rels),
        "selected": len(plan.documented_rels),
        "entry_excluded": len(plan.scanned_rels - plan.documented_rels),
        "analysis_mode": analysis_mode,
        "initial_calls_per_file": per_file,
        **scope_stats,
        "would_process": len(plan.process_rels),
        "would_call_llm_for": remaining,
        "would_skip_insufficient_source": len(frozen_skip_rels),
        "unchanged": len(plan.unchanged_rels),
        "would_reuse": len(plan.identical_reuse_rels),
        "would_resume": len(
            set(recovery_resumed_paths)
            | set(materials.tree_states if materials is not None else ())
        ),
        "forced": len(plan.forced_rels),
        "estimated_calls": estimated_calls,
        "response_correction_enabled": correction_enabled,
        "response_correction_calls_possible_max": correction_possible_max,
        "estimated_calls_max_with_correction": estimated_calls + correction_possible_max,
        "estimated_input_tokens": _estimate_planned_input_tokens(
            plan,
            file_map,
            config,
            resolved_profile,
            skip_rels=frozen_skip_rels,
            review_batches=review_batches,
            execution_requests=execution_requests,
            division_plans=(
                materials.division_plans if materials is not None else None
            ),
            reduction_trees=(
                materials.reduction_trees if materials is not None else None
            ),
            tree_states=(
                materials.tree_states if materials is not None else None
            ),
        ),
        "estimate_is_lower_bound": estimate_is_lower_bound,
        "max_files": plan.max_files,
        # Capacity-blocked files never enter agent_rels (D8), so they are
        # already excluded here without a separate term.
        "max_files_candidate_files": len(plan.unpaid_action_rels),
        "max_files_exceeded": plan.max_files_exceeded,
        # The canonical call manifest's own counts/cap/digest. Provider-free and
        # identical whether returned here or (after attempts) from a real run.
        "total_calls_planned": plan.total_calls_planned,
        "max_planned_calls": plan.max_planned_calls,
        "max_planned_calls_exceeded": plan.max_planned_calls_exceeded,
        "call_manifest_digest": plan.call_manifest_digest,
        "documentation_calls_planned": plan.documentation_calls_planned,
        **_split_division_stats(config, materials, plan),
        "ownership_conflicts": ownership_conflicts,
        "output_dir": str(output_dir),
        "output_files": [],
        "prompt_profile_scope_counts": _prompt_profile_scope_counts(
            plan, file_map, resolved_profile
        ),
        **profile_stats,
    }


def _base_profile_stats(
    resolved: ResolvedProfile,
    resolution: ProfileResolution,
    plan: PipelinePlan,
    file_map: dict[str, dict],
    batches: list[ReviewBatch],
    *,
    feasibility_notes: tuple[str, ...],
) -> dict:
    affected_rels = {
        rel
        for rel in plan.documented_rels
        if resolved.is_active_for(resolved.scope_for(file_map[rel]).basename)
    }
    affected = len(affected_rels)
    return {
        "prompt_profile_source": resolution.source,
        "prompt_profile_active": affected > 0,
        "prompt_profile_affected_files": affected,
        "prompt_customization_security_review": (
            "pending" if batches else "not-required"
        ),
        "prompt_customization_security_review_calls_planned": len(batches),
        "prompt_customization_security_review_calls_completed": 0,
        "prompt_customization_security_review_calls_attempted": 0,
        "prompt_customization_security_warnings": 0,
        "prompt_customization_security_blocking_reasons": 0,
        "prompt_customization_feasibility_notes": len(feasibility_notes),
        "prompt_customization_feasibility_advisories": feasibility_notes,
        "documentation_calls_planned": 0,
        "documentation_calls_attempted": 0,
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
    *,
    call_manifest: CallManifest | None = None,
    materials: PlanMaterials | None = None,
) -> dict:
    """Return the stable scope/reachability statistics contract."""
    scope = config.get("documentation_scope", "entry")
    disconnected_rels = (
        set(agent_rels) - set(reachable_rels)
        if scope == "all" and entry_rel is not None
        else set()
    )
    chunk_paths = {
        chunk.chunk_id: rel_path
        for rel_path, division_plan in (
            materials.division_plans.items() if materials is not None else ()
        )
        for chunk in division_plan.chunks
    }
    node_paths = {
        node.node_id: rel_path
        for rel_path, reduction_tree in (
            materials.reduction_trees.items() if materials is not None else ()
        )
        for node in reduction_tree.unit_consolidation_nodes + reduction_tree.general_nodes
    }
    calls_by_path: dict[str, int] = {}
    for call in call_manifest.calls if call_manifest is not None else ():
        rel_path = None
        if call.category in ("file-documentation", "file-synthesis"):
            rel_path = call.owner
        elif call.category == "unit-documentation":
            rel_path = chunk_paths.get(call.owner)
        elif call.category == "file-reduction":
            rel_path = node_paths.get(call.owner)
        if rel_path is not None:
            calls_by_path[rel_path] = calls_by_path.get(rel_path, 0) + 1
    disconnected_planned_calls = sum(
        calls_by_path.get(rel_path, 0) for rel_path in disconnected_rels
    )
    disconnected_paid = sum(
        1 for rel_path in disconnected_rels if calls_by_path.get(rel_path, 0)
    )
    return {
        "documentation_scope": scope,
        "entry_reachable": len(reachable_rels),
        "entry_disconnected": len(file_map) - len(reachable_rels),
        "entry_excluded": len(file_map) - len(documented_rels),
        "disconnected_paid_files": disconnected_paid,
        "disconnected_planned_calls": disconnected_planned_calls,
    }


def _blocked_split_files_message(division_blocked: Mapping[str, str]) -> str:
    """The deterministic ConfigError message for every capacity-blocked file.

    Lists every ``(rel_path, reason)`` pair in sorted order (D3/D8): a real
    run must name each one before any writer/provider creation.
    """
    pairs = ", ".join(
        f"{rel_path} ({reason})" for rel_path, reason in sorted(division_blocked.items())
    )
    return (
        f"{len(division_blocked)} file(s) cannot be completely split-planned "
        f"provider-free: {pairs}. Split never falls back to truncation. "
        "Inspect each reason: raising max_content_chars can help chunk or "
        "reduction capacity when the provider supports a larger input, but "
        "does not change atom or symbol counts. Reduce/refactor the affected "
        "source for structural caps. If a supported language is using lexical "
        "fallback, the optional structure extra may produce coarser "
        "syntax-aware atoms. Otherwise choose "
        "large_file_strategy 'truncate' only when incomplete-source analysis "
        "is acceptable for these files. Inspect the exact reasons first with "
        "--dry-run."
    )


def _initial_calls_per_file_for_stats(
    analysis_mode: str,
    large_file_strategy: str,
    plan: PipelinePlan | None,
) -> int:
    """The per-file call count for the `initial_calls_per_file` summary field.

    Ordinary `truncate` behavior (single or triple mode) is unchanged: this is
    exactly `initial_calls_per_file(analysis_mode)`. A valid effective split
    route must never call that helper (workstream 6/D9) — including for the
    under-threshold ordinary files a split-configured run may still contain.
    For split, the per-ordinary-file count is instead derived from the one
    canonical manifest already attached to *plan*: its exact
    `file_documentation_calls_planned` total (ordinary `file-documentation`
    calls only — never a leaf/reduction/synthesis call) divided by the number
    of ordinary (non-division-plan) files currently in scope. With no plan yet
    (nothing scanned) or no ordinary file in scope, there is nothing to derive
    and the value is 0.
    """
    if large_file_strategy != "split":
        return initial_calls_per_file(analysis_mode)
    if plan is None:
        return 0
    ordinary_count = len(set(plan.agent_rels) - set(plan.division_plan_rels))
    if ordinary_count == 0:
        return 0
    return plan.file_documentation_calls_planned // ordinary_count


def _split_division_stats(
    config: dict,
    materials: PlanMaterials | None,
    plan: PipelinePlan | None,
) -> dict:
    """Return split-only observability; default truncate runs remain unchanged.

    Every field/definition here matches the frozen run-statistics contract:
    `split_ordinary_files` is the current exact-provider-plan count (`O` in
    the call formula), every `split_restored_*` count is <= its matching
    total, and no split statistic ever applies a triple-mode multiplier
    (split is valid only in single mode — D2).
    """
    # D2 explicit local gate: split statistics are emitted only for the valid
    # effective split route (large_file_strategy 'split' AND analysis_mode
    # 'single' — D2), never from the bare requested strategy string alone.
    if (
        config.get("large_file_strategy", "truncate") != "split"
        or config.get("analysis_mode", "single") != "single"
    ):
        return {}
    plans = materials.division_plans if materials is not None else {}
    trees = materials.reduction_trees if materials is not None else {}
    tree_states = materials.tree_states if materials is not None else {}
    blocked = materials.division_blocked if materials is not None else {}

    syntax = lexical = chunks = units = divided = continuation_groups = 0
    unit_consolidation_levels = general_reduction_levels = 0
    unit_consolidation_total = general_reduction_total = final_total = 0
    restored_chunks = restored_uc = restored_gr = restored_final = 0

    for rel_path, division_plan in plans.items():
        divided += 1
        chunks += len(division_plan.chunks)
        file_units = division_plan.units
        units += len(file_units)
        chunk_counts_by_unit: dict[str, int] = {}
        for chunk in division_plan.chunks:
            chunk_counts_by_unit[chunk.unit_id] = (
                chunk_counts_by_unit.get(chunk.unit_id, 0) + 1
            )
        continuation_groups += sum(1 for count in chunk_counts_by_unit.values() if count >= 2)
        if division_plan.structural_mode == "syntax":
            syntax += 1
        else:
            lexical += 1

        reduction_tree = trees.get(rel_path)
        if reduction_tree is None:
            continue
        unit_consolidation_levels = max(
            unit_consolidation_levels,
            max((node.level for node in reduction_tree.unit_consolidation_nodes), default=0),
        )
        general_reduction_levels = max(
            general_reduction_levels,
            max((node.level for node in reduction_tree.general_nodes), default=0),
        )
        unit_consolidation_total += len(reduction_tree.unit_consolidation_nodes)
        general_reduction_total += len(reduction_tree.general_nodes)
        final_total += 1

        state = tree_states.get(rel_path)
        completed_ids = frozenset(state.by_id()) if state is not None else frozenset()
        restored_chunks += sum(
            1 for chunk in division_plan.chunks if chunk.chunk_id in completed_ids
        )
        restored_uc += sum(
            1 for node in reduction_tree.unit_consolidation_nodes if node.node_id in completed_ids
        )
        restored_gr += sum(
            1 for node in reduction_tree.general_nodes if node.node_id in completed_ids
        )
        if reduction_tree.final_node.node_id in completed_ids:
            restored_final += 1

    ordinary = (
        len(set(plan.unpaid_action_rels) - set(plans))
        if plan is not None
        else 0
    )

    blocked_by_reason: dict[str, int] = {}
    for reason in blocked.values():
        blocked_by_reason[reason] = blocked_by_reason.get(reason, 0) + 1
    # Dry-run must expose the complete deterministic diagnostic, not only
    # aggregate counts (D8).  Real runs abort above before these pairs could
    # become completed-output metadata; project_view intentionally persists only
    # the aggregate split counters.
    blocked_pairs = tuple(sorted(blocked.items()))

    return {
        "large_file_strategy": "split",
        "split_ordinary_files": ordinary,
        "split_syntax_files": syntax,
        "split_lexical_files": lexical,
        "split_blocked_files": len(blocked),
        "split_blocked_by_reason": blocked_by_reason,
        "split_blocked_pairs": blocked_pairs,
        "split_divided_files": divided,
        "split_units": units,
        "split_chunks": chunks,
        "split_continuation_groups": continuation_groups,
        "split_unit_consolidation_levels": unit_consolidation_levels,
        "split_unit_consolidation_calls_planned": unit_consolidation_total,
        "split_general_reduction_levels": general_reduction_levels,
        "split_general_reduction_calls_planned": general_reduction_total,
        "split_final_synthesis_calls_planned": final_total,
        "split_restored_complete_chunks": restored_chunks,
        "split_restored_unit_consolidation_calls": restored_uc,
        "split_restored_general_reduction_calls": restored_gr,
        "split_restored_final_synthesis_calls": restored_final,
        "file_documentation_calls_planned": (
            plan.file_documentation_calls_planned if plan is not None else 0
        ),
        "unit_documentation_calls_planned": (
            plan.unit_documentation_calls_planned if plan is not None else 0
        ),
        "file_reduction_calls_planned": (
            plan.file_reduction_calls_planned if plan is not None else 0
        ),
        "synthesis_calls_planned": (
            plan.synthesis_calls_planned if plan is not None else 0
        ),
        "split_synthesis_input_estimate": "deterministic-worst-case-envelope",
        "split_complexity_advisory": _split_complexity_advisory(plans, trees),
    }


def _split_complexity_advisory(
    plans: Mapping[str, DivisionPlan], trees: Mapping[str, ReductionTreePlan]
) -> str | None:
    """The D6a non-blocking higher-capability-model advisory, or `None`.

    Derived solely from the completed provider-free plan; never creates a
    provider, ranks/selects a model, or blocks/prompts.  Fires only when a
    chunk count or reduction depth is strictly greater than its frozen
    threshold (tested at threshold-1/threshold/threshold+1).
    """
    from codedoc.core.file_division import (
        SPLIT_COMPLEXITY_ADVISORY_CHUNKS,
        SPLIT_COMPLEXITY_ADVISORY_REDUCTION_DEPTH,
        reduction_depth,
    )

    max_chunks = max((len(plan.chunks) for plan in plans.values()), default=0)
    max_depth = max((reduction_depth(tree) for tree in trees.values()), default=0)
    if max_chunks <= SPLIT_COMPLEXITY_ADVISORY_CHUNKS and max_depth <= (
        SPLIT_COMPLEXITY_ADVISORY_REDUCTION_DEPTH
    ):
        return None
    return (
        "This file requires multi-level synthesis. A higher-capability model may "
        "improve cross-chunk reasoning and final documentation quality. If the "
        "selected provider/model supports a larger context window, raising "
        "max_content_chars may reduce calls but can increase per-call cost."
    )


def _estimate_planned_input_tokens(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    config: dict,
    resolved_profile: ResolvedProfile | None = None,
    *,
    skip_rels: frozenset[str] = frozenset(),
    review_batches: list[ReviewBatch] | tuple[ReviewBatch, ...] = (),
    execution_requests: dict | None = None,
    division_plans: dict | None = None,
    reduction_trees: dict | None = None,
    tree_states: dict | None = None,
) -> int:
    """Estimate input tokens for the planned LLM calls — a lower bound for
    ordinary triple-mode prompts, exact for single-mode/leaf/reduction/final.

    The structure and dependency prompts are built from known inputs.  The
    triple-mode documentation prompt embeds the other agents' responses, which
    do not exist yet, so it is estimated with empty analysis objects.  Uses the
    same centralized truncation helper as real execution so the estimated
    source size matches what would actually be sent.

    Every unpaid leaf chunk uses its exact fixed fragment prompt.  Every
    unpaid unit-consolidation, general-reduction, and final synthesis call
    uses its deterministic *worst-case* canonical envelope (D15/section 15): restored
    nodes are excluded, and the predecessor "all chunks plus one synthesis"
    assumption is never used.
    """
    from codedoc.agents import (
        dependency_agent,
        documentation_agent,
        file_documentation_agent,
        file_synthesis_agent,
        structure_agent,
    )
    from codedoc.core.execution_model import FileReductionExecutionRequest, UnitChunkExecutionRequest
    from codedoc.core.file_division import (
        REDUCER_PROMPT_REVISION,
        maximum_distinct_narratives,
        worst_case_final_synthesis_chars,
    )

    analysis_mode = config.get("analysis_mode", "single")
    total = 0
    execution_requests = execution_requests or {}
    division_plans = division_plans or {}
    reduction_trees = reduction_trees or {}
    tree_states = tree_states or {}

    def _estimate_documentation_prompts(
        rel_path: str,
        content: str,
        imports: list[str],
        language: str,
        bundle,
    ) -> int:
        prompt_total = 0
        if analysis_mode == "triple":
            structure_shape = bundle.selections["structure"].block if bundle else None
            dependency_shape = bundle.selections["dependency"].block if bundle else None
            documentation_shape = (
                bundle.selections["documentation"].block if bundle else None
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
            combined_shape = bundle.selections["combined"].block if bundle else None
            prompts = (
                file_documentation_agent.build_prompt(
                    rel_path, content, imports, language, combined_shape
                ),
            )
        for system, prompt in prompts:
            prompt_total += estimate_tokens(system) + estimate_tokens(prompt)
        return prompt_total

    for rel_path in plan.agent_rels:
        if rel_path in skip_rels:
            continue
        request = execution_requests.get(rel_path)
        if request is None:
            continue
        imports = list(request.imports)
        language = request.language
        bundle = request.context.resolved_shape_bundle
        division_plan = division_plans.get(rel_path)
        if division_plan is not None:
            reduction_tree = reduction_trees.get(rel_path)
            state = tree_states.get(rel_path)
            completed_ids = frozenset(state.by_id()) if state is not None else frozenset()

            for chunk in division_plan.chunks:
                if chunk.chunk_id in completed_ids:
                    continue
                leaf_request = UnitChunkExecutionRequest(
                    rel_path=rel_path,
                    language=language,
                    full_content_hash=request.content_hash,
                    division_plan_digest=division_plan.plan_digest,
                    chunk_id=chunk.chunk_id,
                    unit_id=chunk.unit_id,
                    semantic_units=chunk.semantic_units,
                    unit_indexes=division_plan.unit_positions(chunk),
                    unit_count=len(division_plan.units),
                    unit_chunk_index=chunk.unit_chunk_index,
                    unit_chunk_count=chunk.unit_chunk_count,
                    global_index=chunk.global_index,
                    global_count=chunk.global_count,
                    owning_ranges=chunk.owning_ranges,
                    continuation_before=chunk.continuation_before,
                    continuation_after=chunk.continuation_after,
                    known_symbols=chunk.known_symbols,
                    payload=chunk.payload,
                    context=request.context,
                )
                system, prompt = file_documentation_agent.build_fragment_prompt(leaf_request)
                total += estimate_tokens(system) + estimate_tokens(prompt)

            if reduction_tree is None:
                continue
            for node in reduction_tree.unit_consolidation_nodes + reduction_tree.general_nodes:
                if node.node_id in completed_ids:
                    continue
                placeholder_narratives = maximum_distinct_narratives(
                    len(node.child_ids)
                )
                reduction_request = FileReductionExecutionRequest(
                    rel_path=rel_path,
                    division_plan_digest=division_plan.plan_digest,
                    reduction_tree_digest=reduction_tree.tree_digest,
                    node_id=node.node_id,
                    phase=node.phase,
                    unit_id=node.unit_id,
                    level=node.level,
                    ordinal=node.ordinal,
                    child_ids=node.child_ids,
                    child_capsules=tuple(
                        {"narrative": narrative}
                        for narrative in placeholder_narratives
                    ),
                    reducer_revision=REDUCER_PROMPT_REVISION,
                    context=request.context,
                )
                system, prompt = file_synthesis_agent.build_reduction_prompt(
                    reduction_request, placeholder_narratives
                )
                total += estimate_tokens(system) + estimate_tokens(prompt)

            if reduction_tree.final_node.node_id not in completed_ids:
                manifest_chars = worst_case_final_synthesis_chars(
                    rel_path=rel_path,
                    language=language,
                    imports=imports,
                    root_count=len(reduction_tree.final_node.child_ids),
                    leaf_count=len(reduction_tree.final_node.leaf_ids),
                    max_chars=request.context.max_content_chars,
                )
                shape = resolved_synthesis_shape(bundle)
                system, empty_prompt = file_synthesis_agent.build_prompt(
                    rel_path,
                    "",
                    shape,
                )
                prompt_chars = len(empty_prompt) + manifest_chars
                total += estimate_tokens(system) + max(
                    1,
                    (prompt_chars + 3) // 4,
                )
            continue
        content = truncate_for_llm(
            request.content,
            request.context.max_content_chars,
            head_fraction=request.context.truncation_head_ratio,
        )
        # The profile block is selected by extension scope (basename), matching
        # exactly what real execution will send for this file.
        total += _estimate_documentation_prompts(
            rel_path,
            content,
            imports,
            language,
            bundle,
        )
    for batch in review_batches:
        total += estimate_tokens(REVIEW_SYSTEM) + estimate_tokens(batch.text)
    return total
