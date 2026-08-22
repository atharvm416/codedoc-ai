"""Agent-file execution: rate-limit handling, retries, and parallelism.

Extracted from ``codedoc.pipeline`` as part of the internal module decomposition.
This module owns:

- the adaptive-parallelism step-down ladder;
- sequential and parallel descriptor processing with per-file retries and
  worker-thread recording;
- progress logging, result-error inspection, and pending-work cancellation.

The public entry point is :func:`execute_agent_files`, which takes a fully
constructed :class:`ExecutionContext`.  The provider-specific
:class:`~codedoc.llm.rate_limit_profile.RateLimitProfile` and the
:class:`ExecutionOptions` are built by the pipeline from resolved
configuration and passed in; this module never receives the full
configuration dictionary nor recomputes configuration policy.

Error classification logic lives in :mod:`codedoc.core.error_classifier`.
Compat re-exports below preserve all ``codedoc.core.execution._name`` imports
for compatibility.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codedoc.agents.orchestrator import Orchestrator, assemble_final_result
from codedoc.core.db import compute_file_hash
from codedoc.core.error_classifier import (
    _classify_failure,
    _detect_limit_type,  # noqa: F401  (re-exported: codedoc.pipeline._detect_limit_type)
    _find_insufficient_source_error,
    _raise_rate_limit_exhausted,
)
from codedoc.core.execution_model import (
    DOC_AGENTS_BY_MODE,
    FileExecutionRequest,
    FileReductionExecutionRequest,
    UnitChunkExecutionRequest,
    documentation_call_id,
)
from codedoc.core.file_division import (
    REDUCER_PROMPT_REVISION,
    DivisionPlan,
    ReductionTreePlan,
    SplitTreeState,
    build_fact_ledger,
    canonical_json,
    deterministic_imports_digest,
    final_execution_identity,
    final_input_digest,
    final_synthesis_input,
    leaf_execution_identity,
    leaf_input_digest,
    load_canonical_json_object,
    reduction_execution_identity,
    reduction_input_digest,
    refine_narrative_inputs,
    tree_node_state,
)
from codedoc.core.prompt_profiles import resolved_synthesis_shape
from codedoc.core.record_meta import expected_large_file_identity
from codedoc.core.queue import ProcessingQueue
from codedoc.core.resume import _public_record_to_doc
from codedoc.core.safe_writer import SafeWriter
from codedoc.core.source_precheck import insufficient_source
from codedoc.llm.rate_limit_profile import RateLimitProfile
from codedoc.utils.errors import (
    AgentError,
    CodeDocError,
    ErrorReporter,
    InsufficientSourceError,
    LiveBackupWriteError,
    OutputError,
    ParseError,
    ResponseContractError,
    UnrecoverableProviderError,
    bounded_exception_summary,
    find_provider_failure,
    provider_failure_from_mapping,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def _bounded_reason(exc: BaseException) -> str:
    """The rendering-boundary two-tier check for a progress/failure reason.

    A :class:`CodeDocError` (e.g. ``AgentError``, ``ParseError``,
    ``OutputError``) is already bounded by construction and renders unchanged;
    anything else is reduced through :func:`bounded_exception_summary`.
    """
    return str(exc) if isinstance(exc, CodeDocError) else bounded_exception_summary(exc)

# ---------------------------------------------------------------------------
# Compat re-exports (deprecated; import from error_classifier instead)
# ---------------------------------------------------------------------------
from codedoc.core.error_classifier import (  # noqa: F401, E402  (compat re-export)
    _build_rate_limit_exhausted_abort,
    _build_terminal_abort,
    _has_provider_or_agent_error,
    _walk_chain,
)


@dataclass(frozen=True)
class _SequentialOutcome:
    """Minimal progress signal threaded back from a sequential pass so the
    zero-progress bound can be evaluated without parsing logs.

    ``succeeded_any``
        True if at least one file was recorded successfully during the pass.
    ``failures``
        Number of files the pass marked failed.
    ``all_failures_rate_limited``
        True when every failure the pass saw was rate-limit-classified (and there
        was at least one failure).  False if any non-rate-limit failure occurred.
    """

    succeeded_any: bool
    failures: int
    all_failures_rate_limited: bool


def _is_zero_progress_pass(outcome: _SequentialOutcome) -> bool:
    """True when a lowest-concurrency sequential pass made no progress and every
    failure it saw was rate-limit-classified — the zero-progress stop condition."""
    return (
        not outcome.succeeded_any
        and outcome.failures > 0
        and outcome.all_failures_rate_limited
    )


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


# ---------------------------------------------------------------------------
# Execution boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionOptions:
    """Resolved execution policy, derived from configuration by the pipeline."""

    max_workers: int
    retry_attempts: int
    max_consecutive_failures: int
    rate_limit_adaptive: bool
    parallel_ladder: tuple[int, ...] | None
    respect_retry_after: bool
    retry_after_cap_s: int


@dataclass
class ExecutionContext:
    """All collaborators required to execute the agent-file queue.

    The pipeline constructs this after provider creation and crash-recovery
    initialization, so execution never touches configuration loading,
    provider creation, or output writing.
    """

    orchestrator: Orchestrator
    queue: ProcessingQueue
    recorder: SafeWriter
    error_reporter: ErrorReporter
    rate_limit_profile: RateLimitProfile
    stats: dict
    new_results: dict[str, dict]
    options: ExecutionOptions
    # rel_path -> the frozen request PipelinePlan already built for every
    # provider-bound file. The coordinator looks these up; it never rescans
    # source, recomputes a hash, or resolves a prompt scope itself.
    execution_requests: dict[str, FileExecutionRequest]
    division_plans: dict[str, DivisionPlan] | None = None
    # rel_path -> the matching provider-free reduction tree for every division
    # plan above. Always present together (planning builds both or neither).
    reduction_trees: dict[str, ReductionTreePlan] | None = None
    # The provider-free provider/model/effective-endpoint execution identity
    # for this run (see codedoc.core.file_division.provider_execution_identity),
    # computed once by the pipeline from resolved configuration.
    provider_identity: str = ""
    # Keyword-only and required (section 4): a shared executor or request must
    # never default silently to either mode. ``"fresh"`` performs no split node
    # read/write; ``"recovery"`` is the production reuse/recovery-aware
    # executor. Omission is a TypeError at every construction site, in tests
    # and production alike, rather than a silent behavior change in either
    # direction.
    split_execution_mode: str = field(kw_only=True)


# ---------------------------------------------------------------------------
# Worker wrapper
# ---------------------------------------------------------------------------

def _process_and_record(
    request: FileExecutionRequest,
    orchestrator: Orchestrator,
    recorder: SafeWriter,
    division_plan: DivisionPlan | None = None,
    reduction_tree: ReductionTreePlan | None = None,
    provider_identity: str = "",
    stop_event: threading.Event | None = None,
    *,
    split_execution_mode: str,
    fresh_completed_nodes: dict[str, dict] | None = None,
) -> dict:
    """Process one file and record it in crash recovery from the worker thread.

    Recording happens here — inside the worker — so a Ctrl-C or crash that
    interrupts the main ``as_completed`` collection loop never discards a
    file whose AI work already completed.

    If processing raises for any reason (rate-limit, parse failure, model
    error), the exception propagates out of this function unchanged and
    ``recorder.record()`` is NOT called.  The batch-level code catches the
    future's exception and classifies it as rate-limit or non-rate-limit.

    A `DivisionPlan` always describes a complete effective split (D8): there
    is no capacity-fallback/truncate outcome to special-case here.
    """
    try:
        _raise_if_stopping(stop_event)
        if division_plan is not None:
            if reduction_tree is None:
                raise ValueError("a division plan requires its matching reduction tree.")
            if split_execution_mode == "fresh":
                result = _process_fresh_divided_file(
                    request,
                    division_plan,
                    reduction_tree,
                    provider_identity,
                    orchestrator,
                    stop_event,
                    fresh_completed_nodes,
                )
            elif split_execution_mode == "recovery":
                result = _process_divided_file(
                    request,
                    division_plan,
                    reduction_tree,
                    provider_identity,
                    orchestrator,
                    recorder,
                    stop_event,
                )
            else:
                raise ValueError("unknown split execution mode.")
        else:
            result = _process_one_file(request, orchestrator)
    except (KeyboardInterrupt, SystemExit):
        if stop_event is not None:
            stop_event.set()
        raise
    recorder.record(request.rel_path, result, request.content_hash)
    return result


def _raise_if_stopping(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise concurrent.futures.CancelledError()


def _checkpoint_node(
    recorder: SafeWriter,
    *,
    request: FileExecutionRequest,
    division_plan: DivisionPlan,
    reduction_tree: ReductionTreePlan,
    node_id: str,
    node_type: str,
    unit_id: str | None,
    child_ids: tuple[str, ...],
    coverage_leaf_ids: tuple[str, ...],
    execution_identity: str,
    input_digest: str,
    result: dict,
) -> None:
    """Persist one dependency-validated node-keyed checkpoint (split-partial
    schema version 3; see D11/D12/section 9/section 12).

    ``reduction_tree.tree_digest`` is passed to the writer as container-level
    writer-time provenance only -- never stamped onto the node itself, so a
    topology-only change never invalidates this otherwise-compatible node.
    """
    node = tree_node_state(
        node_id=node_id,
        node_type=node_type,
        rel_path=request.rel_path,
        content_hash=request.content_hash,
        division_plan_digest=division_plan.plan_digest,
        execution_identity_digest=execution_identity,
        input_digest=input_digest,
        unit_id=unit_id,
        child_ids=child_ids,
        coverage_leaf_ids=coverage_leaf_ids,
        result=result,
    )
    recorder.record_tree_node(
        request.rel_path, node, reduction_tree_digest=reduction_tree.tree_digest
    )


def _process_divided_file(
    request: FileExecutionRequest,
    division_plan: DivisionPlan,
    reduction_tree: ReductionTreePlan,
    provider_identity: str,
    orchestrator: Orchestrator,
    recorder: SafeWriter,
    stop_event: threading.Event | None = None,
) -> dict:
    """Run (or reuse) every leaf, unit-consolidation, general-reduction, and
    final node in canonical dependency order and assemble the published
    file-level record.

    ``recorder.get_tree_state(...)`` already carries only nodes the pipeline
    validated as exact, current, and dependency-consistent for this plan/tree
    (see ``codedoc.core.file_division.validate_recovered_tree``); this never
    re-derives that judgement, only whether a given node ID is present.
    """
    return _execute_divided_file(
        request,
        division_plan,
        reduction_tree,
        provider_identity,
        orchestrator,
        recorder=recorder,
        stop_event=stop_event,
        recovery_enabled=True,
    )


def _process_fresh_divided_file(
    request: FileExecutionRequest,
    division_plan: DivisionPlan,
    reduction_tree: ReductionTreePlan,
    provider_identity: str,
    orchestrator: Orchestrator,
    stop_event: threading.Event | None = None,
    completed_nodes: dict[str, dict] | None = None,
) -> dict:
    """Execute one effective split without reading or writing node state.

    This retained policy-specific executor accepts no recovery writer, so a
    leaf/reducer/final result cannot become a resumable node checkpoint
    accidentally. The current release selects the recovery-capable executor;
    ``completed_nodes`` remains run-local retry memory for explicit fresh-mode
    callers and is never serialized. The caller may persist only the fully
    assembled file-level result after this function returns successfully.
    """

    return _execute_divided_file(
        request,
        division_plan,
        reduction_tree,
        provider_identity,
        orchestrator,
        recorder=None,
        stop_event=stop_event,
        recovery_enabled=False,
        fresh_completed_nodes=(
            completed_nodes if completed_nodes is not None else {}
        ),
    )


def _log_split_node_event(
    *,
    rel_path: str,
    category: str,
    node_id: str,
    stage: str,
    reason: str,
    ordinal: int,
    count: int,
    result: dict,
) -> None:
    """Emit one bounded verbose split-diagnostic event (section 3A/19).

    Bounded to: normalized relative owner path, category, the node's own
    content-hash-derived ID, fragment ordinal/count, stage, reason code, and a
    response character count. Never the chunk payload, complete range
    manifest, prompt, request options/body, uncleaned response, authorization
    data, or recovery payload.  A CodeDoc verbosity setting must never opt any
    dependency into request/body DEBUG logging; this event is the intended
    fragment-sequence substitute.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Split node %s: owner=%s category=%s node_id=%s stage=%s ordinal=%d/%d "
        "response_chars=%d",
        reason,
        rel_path,
        category,
        node_id,
        stage,
        ordinal,
        count,
        len(canonical_json(dict(result))),
    )


def _execute_divided_file(
    request: FileExecutionRequest,
    division_plan: DivisionPlan,
    reduction_tree: ReductionTreePlan,
    provider_identity: str,
    orchestrator: Orchestrator,
    *,
    recorder: SafeWriter | None,
    stop_event: threading.Event | None,
    recovery_enabled: bool,
    fresh_completed_nodes: dict[str, dict] | None = None,
) -> dict:
    """Shared split kernel behind fresh and recovery-aware entry points."""
    if recovery_enabled and recorder is None:
        raise ValueError("recovery-aware split execution requires a recorder.")
    rel_path = request.rel_path
    split_content_hash = request.content_hash
    allowed_paths = resolved_synthesis_shape(
        request.context.resolved_shape_bundle
    ).requested_field_paths
    recovered = recorder.get_tree_state(rel_path) if recovery_enabled else None
    completed = (
        recovered.by_id()
        if recovered is not None
        else (fresh_completed_nodes or {})
    )

    results_by_id: dict[str, dict] = {}
    leaf_capsules_ordered: list[dict] = []
    for chunk in division_plan.chunks:
        stored = completed.get(chunk.chunk_id)
        if stored is not None:
            result = (
                load_canonical_json_object(stored.result_json)
                if recovery_enabled
                else stored
            )
        else:
            _raise_if_stopping(stop_event)
            leaf_request = UnitChunkExecutionRequest(
                rel_path=rel_path,
                language=request.language,
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
            result = orchestrator.process_leaf_chunk(leaf_request)
            if recovery_enabled:
                _checkpoint_node(
                    recorder,
                    request=request,
                    division_plan=division_plan,
                    reduction_tree=reduction_tree,
                    node_id=chunk.chunk_id,
                    node_type="leaf",
                    unit_id=None,
                    child_ids=(),
                    coverage_leaf_ids=(chunk.chunk_id,),
                    execution_identity=leaf_execution_identity(
                        rel_path=rel_path,
                        content_hash=split_content_hash,
                        division_plan_digest=division_plan.plan_digest,
                        provider_identity=provider_identity,
                        chunk=chunk,
                    ),
                    input_digest=leaf_input_digest(
                        rel_path=rel_path,
                        language=request.language,
                        chunk=chunk,
                        unit_indexes=division_plan.unit_positions(chunk),
                        unit_count=len(division_plan.units),
                    ),
                    result=result,
                )
            elif fresh_completed_nodes is not None:
                fresh_completed_nodes[chunk.chunk_id] = result
        _log_split_node_event(
            rel_path=rel_path,
            category="split-leaf",
            node_id=chunk.chunk_id,
            stage="leaf",
            reason="reused" if stored is not None else "computed",
            ordinal=chunk.global_index,
            count=chunk.global_count,
            result=result,
        )
        results_by_id[chunk.chunk_id] = result
        leaf_capsules_ordered.append(result)

    ledger = build_fact_ledger(
        leaf_capsules_ordered,
        language=request.language,
        chunks=division_plan.chunks,
        symbols=division_plan.symbols,
    )

    reduction_nodes = reduction_tree.unit_consolidation_nodes + reduction_tree.general_nodes
    reduction_node_count = len(reduction_nodes)
    for reduction_ordinal, node in enumerate(reduction_nodes, start=1):
        stored = completed.get(node.node_id)
        if stored is not None:
            result = (
                load_canonical_json_object(stored.result_json)
                if recovery_enabled
                else stored
            )
        else:
            _raise_if_stopping(stop_event)
            child_capsules = tuple(results_by_id[child_id] for child_id in node.child_ids)
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
                child_capsules=child_capsules,
                reducer_revision=REDUCER_PROMPT_REVISION,
                context=request.context,
            )
            result = orchestrator.process_reduction_node(reduction_request)
            if recovery_enabled:
                # Mirrors the orchestrator's own narrative extraction exactly
                # (codedoc/agents/orchestrator.py Orchestrator.process_reduction_node)
                # so the stored input digest binds precisely what the reducer
                # prompt actually consumed.
                raw_narratives = tuple(
                    capsule.get("narrative", capsule.get("description", ""))
                    if isinstance(capsule, dict)
                    else ""
                    for capsule in child_capsules
                )
                _checkpoint_node(
                    recorder,
                    request=request,
                    division_plan=division_plan,
                    reduction_tree=reduction_tree,
                    node_id=node.node_id,
                    node_type=node.phase,
                    unit_id=node.unit_id,
                    child_ids=node.child_ids,
                    coverage_leaf_ids=node.leaf_ids,
                    execution_identity=reduction_execution_identity(
                        rel_path=rel_path,
                        content_hash=split_content_hash,
                        division_plan_digest=division_plan.plan_digest,
                        reduction_tree_digest=reduction_tree.tree_digest,
                        provider_identity=provider_identity,
                        node=node,
                    ),
                    input_digest=reduction_input_digest(
                        rel_path=rel_path,
                        phase=node.phase,
                        level=node.level,
                        unit_id=node.unit_id,
                        child_count=len(node.child_ids),
                        ordered_child_narratives=refine_narrative_inputs(raw_narratives),
                    ),
                    result=result,
                )
            elif fresh_completed_nodes is not None:
                fresh_completed_nodes[node.node_id] = result
        _log_split_node_event(
            rel_path=rel_path,
            category="split-reduction",
            node_id=node.node_id,
            stage=node.phase,
            reason="reused" if stored is not None else "computed",
            ordinal=reduction_ordinal,
            count=reduction_node_count,
            result=result,
        )
        results_by_id[node.node_id] = result

    final_node = reduction_tree.final_node
    stored = completed.get(final_node.node_id)
    if stored is not None:
        final_result = (
            load_canonical_json_object(stored.result_json)
            if recovery_enabled
            else stored
        )
    else:
        _raise_if_stopping(stop_event)
        child_results = [results_by_id[child_id] for child_id in final_node.child_ids]
        raw_narratives = tuple(
            child.get("narrative", child.get("description", "")) for child in child_results
        )
        root_narratives = refine_narrative_inputs(raw_narratives)
        manifest_json = final_synthesis_input(
            rel_path=rel_path,
            language=request.language,
            imports=request.imports,
            root_narratives=root_narratives,
            root_coverage_leaf_ids=final_node.leaf_ids,
            ledger=ledger,
            max_chars=request.context.max_content_chars,
        )
        final_result = orchestrator.synthesize_divided_file(
            request, division_plan.plan_digest, manifest_json
        )
        if recovery_enabled:
            imports_digest = deterministic_imports_digest(request.imports)
            _checkpoint_node(
                recorder,
                request=request,
                division_plan=division_plan,
                reduction_tree=reduction_tree,
                node_id=final_node.node_id,
                node_type="final",
                unit_id=None,
                child_ids=final_node.child_ids,
                coverage_leaf_ids=final_node.leaf_ids,
                execution_identity=final_execution_identity(
                    rel_path=rel_path,
                    content_hash=split_content_hash,
                    division_plan_digest=division_plan.plan_digest,
                    reduction_tree_digest=reduction_tree.tree_digest,
                    provider_identity=provider_identity,
                    prompt_profile_digest=request.context.resolved_shape_bundle.digest,
                    imports_digest=imports_digest,
                    node=final_node,
                ),
                input_digest=final_input_digest(
                    imports_digest=imports_digest,
                    resolved_shape_digest=request.context.resolved_shape_bundle.digest,
                    manifest_json=manifest_json,
                ),
                result=final_result,
            )
        elif fresh_completed_nodes is not None:
            fresh_completed_nodes[final_node.node_id] = final_result
    _log_split_node_event(
        rel_path=rel_path,
        category="split-final",
        node_id=final_node.node_id,
        stage="final",
        reason="reused" if stored is not None else "computed",
        ordinal=1,
        count=1,
        result=final_result,
    )

    large_identity = _large_identity(request, division_plan, reduction_tree)
    return assemble_final_result(request, final_result, ledger, allowed_paths, large_identity)


def _large_identity(
    request: FileExecutionRequest,
    division_plan: DivisionPlan,
    reduction_tree: ReductionTreePlan,
) -> str:
    value = expected_large_file_identity(
        source_chars=division_plan.source_chars,
        max_chars=division_plan.source_budget_chars,
        rel_path=request.rel_path,
        division_plan_digest=division_plan.plan_digest,
        reduction_tree_digest=reduction_tree.tree_digest,
        structural_mode=division_plan.structural_mode,
        imports_digest=deterministic_imports_digest(request.imports),
    )
    if value is None:
        raise ValueError("oversized split record is missing its cache identity.")
    return value


def restore_completed_tree_result(
    request: FileExecutionRequest,
    division_plan: DivisionPlan,
    reduction_tree: ReductionTreePlan,
    recovered: SplitTreeState,
) -> dict:
    """Finalize one exact, fully computed recovered tree locally, without a
    provider.  Callers must supply a tree state whose every node already
    passed ``validate_recovered_tree`` (the pipeline retains only such nodes).
    """
    completed = recovered.by_id()
    final_node = reduction_tree.final_node
    if final_node.node_id not in completed:
        raise ValueError("recovered tree state is not fully synthesized for this file.")
    allowed_paths = resolved_synthesis_shape(
        request.context.resolved_shape_bundle
    ).requested_field_paths
    leaf_capsules_ordered: list[dict] = []
    for chunk in division_plan.chunks:
        stored = completed.get(chunk.chunk_id)
        if stored is None:
            raise ValueError(f"recovered tree state is missing leaf {chunk.chunk_id!r}.")
        leaf_capsules_ordered.append(load_canonical_json_object(stored.result_json))
    ledger = build_fact_ledger(
        leaf_capsules_ordered,
        language=request.language,
        chunks=division_plan.chunks,
        symbols=division_plan.symbols,
    )
    final_result = load_canonical_json_object(completed[final_node.node_id].result_json)
    large_identity = _large_identity(request, division_plan, reduction_tree)
    return assemble_final_result(request, final_result, ledger, allowed_paths, large_identity)


# ---------------------------------------------------------------------------
# Parallel / sequential file processing
# ---------------------------------------------------------------------------

def execute_agent_files(context: ExecutionContext) -> None:
    """Process the agent-file queue, applying retries and the rate-limit ladder.

    A context-driven facade over the per-file processing loop.
    """
    queue = context.queue
    orchestrator = context.orchestrator
    stats = context.stats
    error_reporter = context.error_reporter
    new_results = context.new_results
    recorder = context.recorder
    profile = context.rate_limit_profile
    options = context.options

    max_workers = options.max_workers
    retry_attempts = options.retry_attempts
    max_consecutive_failures = options.max_consecutive_failures
    respect_retry_after = options.respect_retry_after
    retry_after_cap = options.retry_after_cap_s
    # Fresh split calls may be retried across parallel ladder rungs and the
    # sequential fallback. Keep accepted node results for this process run so a
    # retry repeats only the failed logical call. This map is never persisted or
    # exposed as recovery state.
    fresh_completed_nodes_by_path: dict[str, dict[str, dict]] | None = (
        {} if context.split_execution_mode == "fresh" else None
    )

    # The coordinator's one lookup step: every queued descriptor already has a
    # frozen request PipelinePlan built (reused/unchanged files never reach the
    # queue). Everything below consumes requests only — no worker rescans
    # source, recomputes a hash, or resolves a prompt scope.
    requests: list[FileExecutionRequest] = []
    while True:
        descriptor = queue.next()
        if descriptor is None:
            break
        requests.append(context.execution_requests[descriptor["rel_path"]])

    if max_workers <= 1 or len(requests) <= 1:
        # This single sequential pass IS the lowest-concurrency pass, so the
        # zero-progress bound applies directly to it.
        outcome = _process_files_sequentially(
            requests,
            orchestrator,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
            new_results,
            recorder,
            respect_retry_after,
            retry_after_cap,
            profile,
            context.division_plans or {},
            context.reduction_trees or {},
            context.provider_identity,
            split_execution_mode=context.split_execution_mode,
            fresh_completed_nodes_by_path=fresh_completed_nodes_by_path,
        )
        if _is_zero_progress_pass(outcome):
            _raise_rate_limit_exhausted(
                orchestrator.llm.provider_name, error_reporter
            )
        return

    # Build the parallelism ladder.
    rate_limit_adaptive = options.rate_limit_adaptive
    custom_ladder = options.parallel_ladder
    if custom_ladder:
        ladder = list(custom_ladder)
    else:
        ladder = _build_default_ladder(max_workers)

    provider_name = orchestrator.llm.provider_name
    original_max_workers = max_workers
    remaining = list(requests)
    event_number = 0  # cumulative step-down event count across all rungs

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
            division_plans=context.division_plans or {},
            reduction_trees=context.reduction_trees or {},
            provider_identity=context.provider_identity,
            split_execution_mode=context.split_execution_mode,
            fresh_completed_nodes_by_path=fresh_completed_nodes_by_path,
            profile=profile,
        )
        new_results.update(succeeded)

        # Non-rate-limit failures are retried sequentially immediately so errors
        # are clearly diagnosed and stats["failed"] is correctly incremented.
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
                respect_retry_after,
                retry_after_cap,
                profile,
                context.division_plans or {},
                context.reduction_trees or {},
                context.provider_identity,
                split_execution_mode=context.split_execution_mode,
                fresh_completed_nodes_by_path=fresh_completed_nodes_by_path,
            )

        if not retry_rate_limited or not rate_limit_adaptive:
            # No rate-limited files remain, or adaptive mode is off.
            if retry_rate_limited and not rate_limit_adaptive:
                # Treat remaining rate-limited files as sequential retry.
                remaining_requests = [d for d, _e in retry_rate_limited]
                outcome = _process_files_sequentially(
                    remaining_requests,
                    orchestrator,
                    queue,
                    stats,
                    error_reporter,
                    retry_attempts,
                    max_consecutive_failures,
                    new_results,
                    recorder,
                    respect_retry_after,
                    retry_after_cap,
                    profile,
                    context.division_plans or {},
                    context.reduction_trees or {},
                    context.provider_identity,
                    split_execution_mode=context.split_execution_mode,
                    fresh_completed_nodes_by_path=fresh_completed_nodes_by_path,
                )
                if _is_zero_progress_pass(outcome):
                    _raise_rate_limit_exhausted(provider_name, error_reporter)
            break

        # --- Step down to next ladder rung ---
        next_level = ladder[level_index + 1] if level_index + 1 < len(ladder) else 1

        # Unpack descriptors and exceptions from the rate-limited list.
        remaining_requests = [d for d, _e in retry_rate_limited]
        exceptions = [e for _d, e in retry_rate_limited]

        # Compute inter-rung sleep duration. Both sleep sites source
        # retry_after_s from the envelope (section 5.4), never from text.
        retry_afters = [
            failure.retry_after_s
            for failure in (find_provider_failure(e) for e in exceptions)
            if failure is not None and failure.retry_after_s is not None
        ]
        retry_after_s = max(retry_afters) if retry_afters else None

        if respect_retry_after and retry_after_s is not None:
            sleep_s = min(retry_after_s, retry_after_cap)
        elif profile.min_backoff_s > 0:
            sleep_s = min(
                profile.min_backoff_s * (profile.backoff_scale ** level_index),
                retry_after_cap,
            )
        else:
            sleep_s = 0.0

        # Derive error_sample and limit_type from the first exception's
        # envelope (section 5.4): error_sample is canonical compact JSON of
        # exactly reason_code, status, and retry_after_s (JSON null for
        # absent optional fields) -- never rendered provider text.
        first_failure = find_provider_failure(exceptions[0]) if exceptions else None
        error_sample = (
            json.dumps(
                {
                    "reason_code": first_failure.reason_code if first_failure else None,
                    "status": first_failure.status if first_failure else None,
                    "retry_after_s": (
                        first_failure.retry_after_s if first_failure else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if exceptions
            else ""
        )
        limit_type = first_failure.limit_type if first_failure is not None else None

        event_number += 1
        warning = {
            "provider": provider_name,
            "original_max_parallel": original_max_workers,
            "current_level": level,
            "new_level": next_level,
            "retried_count": len(remaining_requests),
            "retry_after_s": retry_after_s,
            "sleep_s": sleep_s,
            "error_sample": error_sample,
            "limit_type": limit_type,
            "event_number": event_number,
            "rung_index": level_index,
        }
        stats.setdefault("rate_limit_warnings", []).append(warning)

        warn_msg = (
            f"[{provider_name}] Rate limit detected - your configured "
            f"max_parallel_files ({original_max_workers}) has been reduced to "
            f"{next_level}. Retrying {len(remaining_requests)} remaining file(s) "
            f"at lower concurrency."
        )
        if sleep_s > 0:
            warn_msg += f" Sleeping {sleep_s:.1f}s before retry."
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        error_reporter.record(RuntimeError(warn_msg), context="rate limit step-down", level="warning")

        # Apply inter-rung backoff sleep before the next ladder level.
        if sleep_s > 0:
            logger.info(
                "Rate-limit backoff: sleeping %.1fs before level %d (rung %d, event %d)",
                sleep_s, next_level, level_index, event_number,
            )
            time.sleep(sleep_s)

        remaining = remaining_requests

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
            # The ladder has been fully traversed; this sequential fall-through is
            # the single lowest-concurrency pass.  Apply the
            # zero-progress bound to it.  The inter-rung sleep above already
            # happened; if this pass makes no progress we stop without sleeping
            # again.
            outcome = _process_files_sequentially(
                remaining,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                respect_retry_after,
                retry_after_cap,
                profile,
                context.division_plans or {},
                context.reduction_trees or {},
                context.provider_identity,
                split_execution_mode=context.split_execution_mode,
                fresh_completed_nodes_by_path=fresh_completed_nodes_by_path,
            )
            if _is_zero_progress_pass(outcome):
                _raise_rate_limit_exhausted(provider_name, error_reporter)
            break


def _process_descriptor_batch(
    requests: list[FileExecutionRequest],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    recorder: SafeWriter,
    division_plans: dict[str, DivisionPlan] | None = None,
    reduction_trees: dict[str, ReductionTreePlan] | None = None,
    provider_identity: str = "",
    max_consecutive_failures: int = 5,
    profile: RateLimitProfile | None = None,
    *,
    split_execution_mode: str,
    fresh_completed_nodes_by_path: dict[str, dict[str, dict]] | None = None,
) -> tuple[dict[str, dict], list[tuple[FileExecutionRequest, Exception]], list[FileExecutionRequest]]:
    """Process a batch of requests in parallel at *max_workers* concurrency.

    Returns
    -------
    succeeded : dict[str, dict]
        rel_path → result for files that completed without error.  These have
        already been recorded in crash recovery by the worker thread.
    retry_rate_limited : list[tuple[FileExecutionRequest, Exception]]
        (request, causing_exception) pairs for files that hit a rate-limit
        signal.  The exception is preserved so the caller can parse
        ``Retry-After`` hints and compute appropriate inter-rung backoff.
    failed_non_rate_limited : list[FileExecutionRequest]
        Requests that failed for non-rate-limit reasons.
    """
    succeeded: dict[str, dict] = {}
    retry_rate_limited: list[tuple[FileExecutionRequest, Exception]] = []
    failed_non_rate_limited: list[FileExecutionRequest] = []

    consecutive_failures = 0
    health_reported = False
    total = len(requests)
    completed = 0
    fatal_error: LiveBackupWriteError | None = None
    abort_error: UnrecoverableProviderError | None = None

    logger.info(
        "Parallel batch: %d file(s) at max_workers=%d, provider=%s",
        total,
        max_workers,
        orchestrator.llm.provider_name,
    )

    stop_event = threading.Event()
    bind_stop_event = getattr(orchestrator, "bind_stop_event", None)
    if callable(bind_stop_event):
        bind_stop_event(stop_event)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    future_map: dict[concurrent.futures.Future, FileExecutionRequest] = {}
    try:
        future_map = {
            pool.submit(
                _process_and_record,
                request,
                orchestrator,
                recorder,
                (division_plans or {}).get(request.rel_path),
                (reduction_trees or {}).get(request.rel_path),
                provider_identity,
                stop_event,
                split_execution_mode=split_execution_mode,
                fresh_completed_nodes=(
                    fresh_completed_nodes_by_path.setdefault(request.rel_path, {})
                    if fresh_completed_nodes_by_path is not None
                    else None
                ),
            ): request
            for request in requests
        }

        for future in concurrent.futures.as_completed(future_map):
            request = future_map[future]
            rel_path = request.rel_path
            try:
                result = future.result()
                # Already recorded in the worker — just update new_results and stats.
                succeeded[rel_path] = result
                queue.mark_checked(rel_path)
                stats["checked"] += 1
                consecutive_failures = 0
                completed += 1
                _log_file_progress("OK", rel_path, completed, total)
            except LiveBackupWriteError as exc:
                # Fatal output failure raised inside a worker.  Identify it
                # before any rate-limit / ordinary-failure classification: cancel
                # work not yet started, let already-running workers finish or
                # fail without scheduling retries, and re-raise the original
                # error after the executor shuts down.  Never enter the retry or
                # rate-limit lists.
                fatal_error = exc
                stop_event.set()
                _cancel_pending(future_map)
                break
            except Exception as exc:
                completed += 1
                # Evaluate terminal-billing and global-permanent BEFORE the
                # rate-limit branch and the consecutive-failure health check.  A
                # confirmed unrecoverable provider fault is handled exactly like
                # the fatal LiveBackupWriteError path: cancel work not yet started,
                # stop scheduling retries, and re-raise the abort after the
                # executor shuts down.  It never enters any retry list.
                verdict = _classify_failure(exc, profile)
                if verdict in ("terminal_billing", "global"):
                    abort_error = _build_terminal_abort(
                        exc, orchestrator.llm.provider_name, verdict
                    )
                    stop_event.set()
                    _cancel_pending(future_map)
                    break
                if verdict == "rate_limit":
                    # Only treat as done if a worker recorded it THIS run (it may
                    # have succeeded before the future was cancelled).  A file
                    # that is only *preloaded* from a prior output (stale) must be
                    # retried, never restored from old documentation (A4).
                    if not recorder.recorded_this_run(rel_path):
                        # Preserve the causing exception alongside the
                        # request so the caller can parse Retry-After hints.
                        retry_rate_limited.append((request, exc))
                        _log_file_progress(
                            "RATE-LIMIT", rel_path, completed, total, _bounded_reason(exc)
                        )
                    else:
                        # Already recorded — treat as succeeded.  Recover the
                        # real persisted record (A4) so the final output is not
                        # overwritten with an empty placeholder.
                        recovered = recorder.get_record(rel_path)
                        succeeded[rel_path] = (
                            _public_record_to_doc(recovered) if recovered else {}
                        )
                        queue.mark_checked(rel_path)
                        stats["checked"] += 1
                        _log_file_progress("OK(late)", rel_path, completed, total)
                elif verdict == "insufficient_source":
                    matched = _find_insufficient_source_error(exc)
                    reason = matched.reason if matched else ""
                    try:
                        recorder.discard(rel_path)
                    except LiveBackupWriteError as discard_exc:
                        fatal_error = discard_exc
                        stop_event.set()
                        _cancel_pending(future_map)
                        break
                    queue.mark_skipped_insufficient_source(rel_path, reason)
                    stats["skipped_insufficient_source"] += 1
                    consecutive_failures = 0
                    _log_file_progress(
                        "SKIP",
                        rel_path,
                        completed,
                        total,
                        f"insufficient source: {reason} — not sent to provider",
                    )
                elif verdict in ("input", "response_contract_final"):
                    # The identical oversized input cannot succeed on a second
                    # processing path; a deterministic response-contract rejection
                    # must never become a duplicate whole-file call.  Record here
                    # and keep processing other files without a sequential retry.
                    error_reporter.record(exc, context=rel_path)
                    queue.mark_failed(rel_path, _bounded_reason(exc))
                    stats["failed"] += 1
                    consecutive_failures += 1
                    _log_file_progress("FAIL", rel_path, completed, total, _bounded_reason(exc))
                else:
                    # Transient non-rate-limit failures retain the existing
                    # sequential retry path for clearer diagnostics.
                    failed_non_rate_limited.append(request)
                    consecutive_failures += 1
                    _log_file_progress("RETRY", rel_path, completed, total, _bounded_reason(exc))

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
    except BaseException:
        stop_event.set()
        _cancel_pending(future_map)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=stop_event.is_set())

    # Propagate the fatal persistence failure as the original error.
    if fatal_error is not None:
        raise fatal_error
    # Propagate a confirmed unrecoverable provider abort so it leaves
    # execution and the pipeline records + re-raises it.  Raised only after the
    # executor shut down, so no further descriptors run.
    if abort_error is not None:
        raise abort_error

    return succeeded, retry_rate_limited, failed_non_rate_limited


def _process_files_sequentially(
    requests: list[FileExecutionRequest],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
    recorder: SafeWriter,
    respect_retry_after: bool = True,
    retry_after_cap: int = 30,
    profile: RateLimitProfile | None = None,
    division_plans: dict[str, DivisionPlan] | None = None,
    reduction_trees: dict[str, ReductionTreePlan] | None = None,
    provider_identity: str = "",
    *,
    split_execution_mode: str,
    fresh_completed_nodes_by_path: dict[str, dict[str, dict]] | None = None,
) -> _SequentialOutcome:
    """Process *requests* one at a time with per-file retries.

    Returns a :class:`_SequentialOutcome` so that, when this is the
    lowest-concurrency pass, ``execute_agent_files`` can apply the zero-progress
    rate-limit bound.  Existing callers may ignore the return value.
    """
    consecutive_failures = 0
    total = len(requests)
    succeeded_any = False
    failures = 0
    all_failures_rate_limited = True

    logger.info(
        "Starting sequential documentation: %d file(s), provider=%s",
        total,
        orchestrator.llm.provider_name,
    )

    stop_event = threading.Event()
    bind_stop_event = getattr(orchestrator, "bind_stop_event", None)
    if callable(bind_stop_event):
        bind_stop_event(stop_event)

    for index, request in enumerate(requests, start=1):
        rel_path = request.rel_path
        try:
            result = _process_one_file_with_retries(
                request,
                orchestrator,
                retry_attempts,
                division_plan=(division_plans or {}).get(rel_path),
                reduction_tree=(reduction_trees or {}).get(rel_path),
                provider_identity=provider_identity,
                recorder=recorder,
                stop_event=stop_event,
                split_execution_mode=split_execution_mode,
                fresh_completed_nodes=(
                    fresh_completed_nodes_by_path.setdefault(rel_path, {})
                    if fresh_completed_nodes_by_path is not None
                    else None
                ),
                respect_retry_after=respect_retry_after,
                retry_after_cap=retry_after_cap,
                profile=profile,
            )
            new_results[rel_path] = result
            recorder.record(rel_path, result, request.content_hash)
            queue.mark_checked(rel_path)
            stats["checked"] += 1
            consecutive_failures = 0
            succeeded_any = True
            _log_file_progress("OK", rel_path, index, total)
        except (KeyboardInterrupt, SystemExit):
            stop_event.set()
            raise
        except LiveBackupWriteError:
            # Fatal: the crash-safety backup could not be persisted, so the
            # resume guarantee no longer holds.  Do not retry, do not mark the
            # file failed — stop scheduling work and propagate immediately.
            # (LiveBackupWriteError subclasses OutputError, so this clause must
            # precede the recoverable OutputError handler below.)
            stop_event.set()
            raise
        except UnrecoverableProviderError:
            # A terminal billing/credentials/model/access abort raised by
            # the per-file retry routing must propagate out of execution so the
            # pipeline records it and stops while crash recovery stays resumable.
            # Mirrors the LiveBackupWriteError re-raise; must precede the
            # recoverable AgentError/OutputError and generic handlers below
            # (UnrecoverableProviderError is an LLMError, not one of those).
            stop_event.set()
            raise
        except InsufficientSourceError as exc:
            recorder.discard(rel_path)
            queue.mark_skipped_insufficient_source(rel_path, exc.reason)
            stats["skipped_insufficient_source"] += 1
            consecutive_failures = 0
            _log_file_progress(
                "SKIP",
                rel_path,
                index,
                total,
                f"insufficient source: {exc.reason} — not sent to provider",
            )
        except (ParseError, OutputError, AgentError) as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, _bounded_reason(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            failures += 1
            if _classify_failure(exc, profile) != "rate_limit":
                all_failures_rate_limited = False
            _log_file_progress("FAIL", rel_path, index, total, _bounded_reason(exc))
        except Exception as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, _bounded_reason(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            failures += 1
            if _classify_failure(exc, profile) != "rate_limit":
                all_failures_rate_limited = False
            _log_file_progress("FAIL", rel_path, index, total, _bounded_reason(exc))

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

    return _SequentialOutcome(
        succeeded_any=succeeded_any,
        failures=failures,
        all_failures_rate_limited=all_failures_rate_limited and failures > 0,
    )


def _process_one_file_with_retries(
    request: FileExecutionRequest,
    orchestrator: Orchestrator,
    retry_attempts: int,
    division_plan: DivisionPlan | None = None,
    reduction_tree: ReductionTreePlan | None = None,
    provider_identity: str = "",
    recorder: SafeWriter | None = None,
    stop_event: threading.Event | None = None,
    *,
    split_execution_mode: str,
    fresh_completed_nodes: dict[str, dict] | None = None,
    respect_retry_after: bool = True,
    retry_after_cap: int = 30,
    profile: RateLimitProfile | None = None,
) -> dict:
    last_error: Exception | None = None
    if split_execution_mode == "fresh" and fresh_completed_nodes is None:
        fresh_completed_nodes = {}
    for attempt in range(retry_attempts + 1):
        try:
            _raise_if_stopping(stop_event)
            if division_plan is not None:
                if reduction_tree is None:
                    raise ValueError("a division plan requires its matching reduction tree.")
                if split_execution_mode == "fresh":
                    return _process_fresh_divided_file(
                        request,
                        division_plan,
                        reduction_tree,
                        provider_identity,
                        orchestrator,
                        stop_event,
                        fresh_completed_nodes,
                    )
                if split_execution_mode != "recovery":
                    raise ValueError("unknown split execution mode.")
                if recorder is None:
                    raise ValueError("split retries require a recovery recorder.")
                return _process_divided_file(
                    request,
                    division_plan,
                    reduction_tree,
                    provider_identity,
                    orchestrator,
                    recorder,
                    stop_event,
                )
            return _process_one_file(request, orchestrator)
        except Exception as exc:
            last_error = exc
            # Cancellation is a scheduling boundary, never a retryable file
            # failure.  Re-raise immediately so even retry bookkeeping/sleep
            # cannot occur after the shared stop signal is observed.
            if stop_event is not None and stop_event.is_set():
                raise
            # Apply the fixed failure precedence before consuming the next
            # attempt.  Abort cases raise immediately (no remaining attempts);
            # an input-too-large error re-raises immediately so it is recorded as
            # a normal failed file without a guaranteed-to-fail retry; transient
            # keeps the existing retry/sleep behavior.
            verdict = _classify_failure(exc, profile)
            if verdict in ("terminal_billing", "global"):
                if stop_event is not None:
                    stop_event.set()
                raise _build_terminal_abort(
                    exc, orchestrator.llm.provider_name, verdict
                )
            if verdict in ("insufficient_source", "input", "response_contract_final"):
                # A deterministic local skip, oversized input, or final
                # response-contract rejection re-raises immediately for the
                # outer router, without consuming a guaranteed-to-fail retry.
                raise
            if attempt < retry_attempts:
                # Apply Retry-After sleep for rate-limit errors in sequential mode.
                if respect_retry_after and verdict == "rate_limit":
                    failure = find_provider_failure(exc)
                    delay = failure.retry_after_s if failure is not None else None
                    if delay is not None:
                        sleep_s = min(delay, retry_after_cap)
                        logger.info(
                            "Retry-After: sleeping %.1fs before retrying %s",
                            sleep_s,
                            request.rel_path,
                        )
                        time.sleep(sleep_s)
                logger.info(
                    "Retrying %s (%d/%d): %s",
                    request.rel_path,
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


def _process_one_file(request: FileExecutionRequest, orchestrator: Orchestrator) -> dict:
    rel_path = request.rel_path
    logger.info("[START] %s | provider=%s", rel_path, orchestrator.llm.provider_name)
    mode = request.context.analysis_mode
    if logger.isEnabledFor(logging.DEBUG):
        # Bounded: rel_path, mode, and domain-separated call-id hashes only —
        # never raw source, prompts, custom instructions, credentials, or
        # provider responses.
        logger.debug(
            "Planned call(s) for owner=%s category=file-documentation mode=%s call_ids=%s",
            rel_path,
            mode,
            [
                documentation_call_id(rel_path, mode, agent, ordinal)
                for ordinal, agent in enumerate(DOC_AGENTS_BY_MODE[mode], start=1)
            ],
        )
    insufficient, reason = insufficient_source(request.content)
    if insufficient:
        raise InsufficientSourceError(rel_path, reason)
    result = orchestrator.process(request)
    logger.debug("Outcome for owner=%s: state=%s", rel_path, result.get("state", "unknown"))
    _raise_result_errors(result, orchestrator, rel_path)
    return result


def _raise_result_errors(
    result: dict,
    orchestrator: Orchestrator,
    rel_path: str,
) -> None:
    """Raise the canonical typed failure represented by an agent result."""
    errors = _agent_errors(result)
    if errors:
        message = "; ".join(errors)
        terminal_error = _terminal_sibling_error(result, rel_path)
        if terminal_error is not None:
            # A contract failure in one triple-mode sibling must not mask a
            # simultaneous billing/credential/model failure in another. Preserve
            # the terminal signal so the retry wrapper aborts the run.
            raise terminal_error
        marker = _response_contract_final_marker(result)
        if marker is not None:
            diagnostic, correction_attempted = marker
            # A deterministic response-contract rejection is non-retryable: raise a
            # typed error carrying the bounded diagnostic and the opt-out/exhausted
            # distinction so the classifier keeps it off every retry ladder.
            raise ResponseContractError(
                orchestrator.__class__.__name__,
                rel_path,
                message,
                diagnostic=diagnostic,
                correction_attempted=correction_attempted,
            )
        agent_error = AgentError(orchestrator.__class__.__name__, rel_path, message)
        # Section 5.4: restore the envelope this dict round-trip would
        # otherwise discard, so downstream classification (rate-limit
        # detection, retry-after sourcing) still sees it.
        agent_error.provider_failure = _first_sibling_provider_failure(result)
        raise agent_error


def _first_sibling_provider_failure(result: dict):
    """Return the first fixed-order sibling's reconstructed
    :class:`~codedoc.utils.errors.ProviderFailureEnvelope`, or ``None`` when
    that sibling carries no envelope.

    Section 12.1 C4: stops at the first sibling with an error, in fixed
    order (structure, dependencies_analysis, documentation) -- never skips
    an envelope-less first error to inherit a later sibling's envelope,
    which would misclassify the overall failure (e.g. reporting an
    unrelated local failure as rate-limited) from metadata that belongs to
    a different call entirely.
    """
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if isinstance(value, dict) and value.get("error"):
            return provider_failure_from_mapping(value.get("provider_failure"))
    return None


def _agent_errors(result: dict) -> list[str]:
    errors = []
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if isinstance(value, dict) and value.get("error"):
            agent = value.get("agent", key)
            errors.append(f"{agent}: {value['error']}")
    return errors


def _response_contract_final_marker(result: dict) -> tuple | None:
    """Return ``(diagnostic, correction_attempted)`` if any inspected sub-result
    carries the non-retryable ``response_contract_final`` marker, else ``None``."""
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if isinstance(value, dict) and value.get("response_contract_final"):
            return (
                value.get("response_contract_diagnostic"),
                bool(value.get("response_contract_correction_attempted", False)),
            )
    return None


def _terminal_sibling_error(result: dict, file_path: str) -> AgentError | None:
    """Return a non-contract terminal/global agent error, if one is present.

    Triple-mode agents return error dictionaries, which discard the original
    exception type. Reclassifies every sibling error from its reconstructed
    envelope, in fixed order (structure, dependencies_analysis,
    documentation) -- section 5.4: the first terminal-billing/global
    envelope wins even when that same sibling also carries
    ``response_contract_final``, since a genuine run-level abort is never
    masked by a contract failure on the very call that produced it. Rate-limit
    and transient siblings intentionally do not override the
    no-whole-file-retry contract.
    """
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if not isinstance(value, dict) or not value.get("error"):
            continue
        candidate = AgentError(
            str(value.get("agent", key)),
            file_path,
            str(value["error"]),
        )
        # Section 5.4: reconstruct the envelope this dict round-trip would
        # otherwise discard, so classification reads the structured reason
        # rather than always falling through to "transient".
        candidate.provider_failure = provider_failure_from_mapping(
            value.get("provider_failure")
        )
        if _classify_failure(candidate, None) in ("terminal_billing", "global"):
            return candidate
    return None


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


def reconcile_planned_calls(
    *,
    total_calls_planned: int,
    attempted_logical_calls: int,
    attempted_calls: int,
) -> dict[str, int]:
    """Reconcile planned vs. attempted logical calls into the final stats.

    ``attempted_logical_calls`` comes from canonical manifest-ID consumption,
    never from queue/file status. This remains valid after a terminal abort:
    unconsumed manifest entries are reported as planned-but-not-attempted, while
    retries and corrections increase ``additional_attempts`` without consuming
    another logical ID.
    """
    if attempted_logical_calls < 0 or attempted_logical_calls > total_calls_planned:
        raise ValueError("attempted logical calls must be within the manifest bounds.")
    return {
        "attempted_logical_calls": attempted_logical_calls,
        "planned_calls_not_attempted": total_calls_planned - attempted_logical_calls,
        "additional_attempts": max(0, attempted_calls - attempted_logical_calls),
    }
