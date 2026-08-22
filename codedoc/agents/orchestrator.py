"""
Agent Orchestrator.

Coordinates per-file analysis in one of two selectable modes:

- ``single`` (default): one combined :class:`FileDocumentationAgent` call per
  file (one provider call), then merges the cleaned response with the
  deterministic identity/import fields.
- ``triple``: the legacy path running StructureAgent, DependencyAgent, and
  DocumentationAgent (three provider calls), merging their outputs.

Both modes return the identical flat record consumed by SafeWriter and
``project_view.py``.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

from codedoc.agents.base_agent import truncate_for_llm
from codedoc.agents.file_documentation_agent import FileDocumentationAgent
from codedoc.agents.file_synthesis_agent import FileSynthesisAgent, SYNTHESIS_PROMPT_REVISION
from codedoc.agents.structure_agent import StructureAgent
from codedoc.agents.dependency_agent import DependencyAgent
from codedoc.agents.documentation_agent import DocumentationAgent
from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
from codedoc.agents.response_cleaning import clean_combined_report
from codedoc.agents.response_diagnostics import CorrectionLedger
from codedoc.core.execution_model import (
    CallManifestTracker,
    FileExecutionRequest,
    FileReductionExecutionRequest,
    PlannedCall,
    UnitChunkExecutionRequest,
    documentation_call_id,
    file_reduction_call_id,
    file_synthesis_call_id,
    unit_documentation_call_id,
)
from codedoc.core.file_division import FactLedger, refine_narrative_inputs
from codedoc.core.record_meta import (
    expected_analysis_identity,
    expected_max_context_revision,
    expected_ordinary_path_identity,
)
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    resolved_synthesis_shape,
)
from codedoc.core.result_assembly import extension_of, flat_combined_result
from codedoc.core.usage import UsageAccumulator
from codedoc.llm.base import LLMProvider
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ANALYSIS_MODES = ("single", "triple")

# Split is valid only in single mode (D2); this fixed label is the sole
# "agent" a leaf-chunk call ID or diagnostic ever names, regardless of the
# run's own analysis_mode string.
_SPLIT_LEAF_AGENT = "combined"


def initial_calls_per_file(analysis_mode: str) -> int:
    """Return the number of provider calls one file makes on its initial attempt.

    Canonical helper used by both mode statistics and the planning multiplier so
    the per-file call count is defined in exactly one place.
    """
    return 3 if analysis_mode == "triple" else 1


def _result_has_agent_error(result: dict) -> bool:
    """Whether a merged result carries an agent error on any sub-result.

    Mirrors the execution-layer ``_agent_errors`` check without importing it
    (execution imports this module, so the dependency must not be reversed).
    """
    for key in ("structure", "dependencies_analysis", "documentation"):
        value = result.get(key)
        if isinstance(value, dict) and value.get("error"):
            return True
    return False


def assemble_final_result(
    request: FileExecutionRequest,
    synthesis: dict,
    ledger: FactLedger,
    requested_field_paths: tuple[str, ...],
    large_file_identity: str,
) -> dict:
    """Locally assemble and identity-stamp the final published split record.

    Publishes only the ordinary supported file-level documentation shape
    (D9/D14): no ``division``, ``documentation_units``, internal IDs, ranges,
    capsules, or reduction content ever reach the result.
    ``functions``/``classes``/``exports`` are assembled from the locally
    maintained fact ledger — never from the model's own final response —
    whenever the resolved final shape requests them (D10/section 14); every other
    field (description, role, dependencies, concepts, usage) comes from the
    cleaned final synthesis response unchanged.

    Effective split records never carry ``_max_context_revision``: their
    `_large_file_identity` already binds `max_content_chars` through the
    division/tree digests, so a ratio-only change reuses a compatible
    completed record (D12/section 13).
    """
    merged = dict(synthesis)
    if "functions" in requested_field_paths:
        merged["functions"] = [dict(item) for item in ledger.functions]
    else:
        merged.pop("functions", None)
    if "classes" in requested_field_paths:
        merged["classes"] = [dict(item) for item in ledger.classes]
    else:
        merged.pop("classes", None)
    if "exports" in requested_field_paths:
        merged["exports"] = list(ledger.exports)
    else:
        merged.pop("exports", None)
    merged = clean_combined_report(merged, request.rel_path).value
    result = flat_combined_result(
        request.rel_path,
        request.language,
        list(request.imports),
        merged,
    )
    result.update(expected_analysis_identity(request.context.analysis_mode))
    bundle = request.context.resolved_shape_bundle
    if bundle.digest != NO_PROMPT_PROFILE_DIGEST:
        result["_prompt_profile_digest"] = bundle.digest
    result["_large_file_identity"] = large_file_identity
    return result


class Orchestrator:
    """
    Coordinates multi-agent processing for a single file.

    Parallel mode  (parallel=True, default):
        All 3 agents run concurrently → faster, recommended for API providers.

    Sequential mode (parallel=False):
        Agents run one after another → useful for local LLMs with limited VRAM.
        DocumentationAgent receives structure + dependency results as context.
    """

    def __init__(
        self,
        llm: LLMProvider,
        parallel: bool = True,
        max_content_chars: int = 12000,
        usage: UsageAccumulator | None = None,
        analysis_mode: str = "single",
        truncation_head_ratio: float = 0.70,
        response_correction_enabled: bool = False,
        correction_ledger: CorrectionLedger | None = None,
        call_tracker: CallManifestTracker | None = None,
    ) -> None:
        if analysis_mode not in VALID_ANALYSIS_MODES:
            raise ValueError(
                f"analysis_mode must be one of {VALID_ANALYSIS_MODES}; got "
                f"{analysis_mode!r}."
            )
        self.llm = llm
        self.parallel = parallel
        # Retained only to construct each agent's own defensive-truncation
        # ceiling below; per-call decisions for an orchestrated file come from
        # the immutable ``AgentCallContext`` passed into ``process()``.
        self.max_content_chars = max_content_chars
        self.analysis_mode = analysis_mode
        self.truncation_head_ratio = truncation_head_ratio
        self._call_tracker = call_tracker
        # The combined agent powers the default single-call path.
        self._file_agent = FileDocumentationAgent(
            llm, max_content_chars=max_content_chars, usage=usage,
            call_tracker=call_tracker,
        )
        self._synthesis_agent = FileSynthesisAgent(
            llm, max_content_chars=max_content_chars, usage=usage,
            call_tracker=call_tracker,
        )
        # The three legacy agents remain instantiated and are used by triple mode.
        self._structure_agent = StructureAgent(
            llm, max_content_chars=max_content_chars, usage=usage,
            call_tracker=call_tracker,
        )
        self._dependency_agent = DependencyAgent(
            llm, max_content_chars=max_content_chars, usage=usage,
            call_tracker=call_tracker,
        )
        self._doc_agent = DocumentationAgent(
            llm, max_content_chars=max_content_chars, usage=usage,
            call_tracker=call_tracker,
        )
        # Build exactly one shared correction component and inject it into all four
        # agents.  Only when a run-level ledger is supplied; a directly constructed
        # orchestrator (no ledger) leaves every agent's ``_correction`` as ``None``,
        # so an eligible response-contract failure is final with no correction call.
        if correction_ledger is not None:
            correction = ResponseCorrectionAgent(
                llm, usage, correction_ledger, response_correction_enabled,
                call_tracker=call_tracker,
            )
            for agent in (
                self._file_agent,
                self._structure_agent,
                self._dependency_agent,
                self._doc_agent,
                self._synthesis_agent,
            ):
                agent._correction = correction

    def bind_stop_event(self, stop_event: threading.Event | None) -> None:
        if self._call_tracker is not None:
            self._call_tracker.bind_stop_event(stop_event)

    def process_leaf_chunk(self, request: UnitChunkExecutionRequest) -> dict:
        """Document one immutable split leaf chunk under the fixed fragment
        contract.  Split is valid only in single mode (D2); there is no
        triple-mode leaf-chunk path."""
        call = PlannedCall(
            call_id=unit_documentation_call_id(
                request.rel_path,
                request.unit_id,
                request.chunk_id,
                request.unit_chunk_index,
                request.division_plan_digest,
                "single",
                _SPLIT_LEAF_AGENT,
                1,
            ),
            category="unit-documentation",
            owner=request.chunk_id,
            ordinal=1,
        )
        additional_attempt = self._additional_attempt(call, False)
        cleaned = self._file_agent.run_fragment(
            request, planned_call=call, additional_attempt=additional_attempt
        )
        result = dict(cleaned)
        result["chunk_id"] = request.chunk_id
        result["unit_id"] = request.unit_id
        return result

    def process_reduction_node(self, request: FileReductionExecutionRequest) -> dict:
        """Run one fixed internal unit-consolidation or general reduction call.

        Never receives raw source or authoritative coverage; the caller
        (`codedoc.core.execution`) attaches and verifies exact descendant
        coverage locally after this returns.
        """
        raw_narratives = tuple(
            capsule.get("narrative", capsule.get("description", ""))
            if isinstance(capsule, dict)
            else ""
            for capsule in request.child_capsules
        )
        narratives = refine_narrative_inputs(raw_narratives)
        call = PlannedCall(
            call_id=file_reduction_call_id(
                request.rel_path,
                request.node_id,
                request.reduction_tree_digest,
                request.reducer_revision,
                1,
            ),
            category="file-reduction",
            owner=request.node_id,
            ordinal=1,
        )
        additional_attempt = self._additional_attempt(call, False)
        return self._synthesis_agent.run_reduction(
            request, narratives, planned_call=call, additional_attempt=additional_attempt
        )

    def synthesize_divided_file(
        self,
        request: FileExecutionRequest,
        division_plan_digest: str,
        manifest_json: str,
    ) -> dict:
        """Run the single final synthesis call for a divided file."""
        context = request.context
        call = PlannedCall(
            call_id=file_synthesis_call_id(
                request.rel_path,
                division_plan_digest,
                SYNTHESIS_PROMPT_REVISION,
                1,
            ),
            category="file-synthesis",
            owner=request.rel_path,
            ordinal=1,
        )
        additional_attempt = self._additional_attempt(call, False)
        cleaned = self._synthesis_agent.run(
            request.rel_path,
            manifest_json,
            resolved_synthesis_shape(context.resolved_shape_bundle),
            call_context=context,
            planned_call=call,
            additional_attempt=additional_attempt,
        )
        return self._merge_single(
            request.rel_path, request.language, list(request.imports), cleaned
        )

    def process(self, request: FileExecutionRequest) -> dict:
        """
        Run all agents on one file and return the merged result.

        Args:
            request: the frozen execution request for this file — content,
                imports, and the immutable per-call context (mode, ceiling,
                ratio, and this file's resolved prompt-profile bundle) all come
                from *request*; nothing is re-read or recomputed here.

        Returns:
            Merged dict with structure and dependency analysis for this file.
        """
        file_path = request.rel_path
        language = request.language
        content = request.content
        imports = list(request.imports)
        context = request.context

        # Truncate once here so all three agents receive the exact same
        # string and the marker stays inside the configured ceiling.  One
        # WARNING per processed file — the agents' own fallback is DEBUG.
        original_chars = len(content)
        if original_chars > context.max_content_chars:
            content = truncate_for_llm(
                content,
                context.max_content_chars,
                head_fraction=context.truncation_head_ratio,
            )
            logger.warning(
                "Content truncated: %s (%d chars -> %d chars sent; configured "
                "ceiling max_content_chars=%d). Raise max_content_chars in "
                "config to include more content.",
                file_path,
                original_chars,
                len(content),
                context.max_content_chars,
            )

        logger.debug(
            "Running %s-mode analysis for %s with %s",
            context.analysis_mode,
            file_path,
            self.llm.provider_name,
        )

        # This file's already-resolved prompt-profile bundle (extension scope),
        # threaded through every agent prompt and the stamped record digest.
        bundle = context.resolved_shape_bundle
        additional_attempt = False

        if context.analysis_mode == "triple":
            result = self._process_triple(
                file_path, content, imports, language, bundle,
                context=context,
                additional_attempt=additional_attempt,
            )
        else:
            result = self._process_single(
                file_path, content, imports, language, bundle,
                context=context,
                additional_attempt=additional_attempt,
            )

        # Attach cache-identity keys to every successful flat result before it is
        # recorded.  Failed results (an error on structure/dependencies/
        # documentation) are never recorded, so they must not carry the identity.
        if result.get("state") == "checked" and not _result_has_agent_error(result):
            result.update(expected_analysis_identity(context.analysis_mode))
            # Stamp the truncation identity only for a file actually large
            # enough to be truncated (omit it otherwise so files that fit the
            # ceiling stay reusable across ceiling/ratio changes).  ``original_chars``
            # is the pre-truncation source length.
            mcr = expected_max_context_revision(
                original_chars,
                max_chars=context.max_content_chars,
                head_ratio=context.truncation_head_ratio,
            )
            if mcr is not None:
                result["_max_context_revision"] = mcr
            if bundle is not None and bundle.digest != NO_PROMPT_PROFILE_DIGEST:
                result["_prompt_profile_digest"] = bundle.digest
            # Ordinary-only path identity: refuses ordinary cross-path
            # identical-content reuse (a stored record must never document a
            # different path than the one it is reused into). Covers both a
            # truly ordinary file and an oversized truncate-path file alike --
            # never stamped by the split final-result assembly
            # (assemble_final_result), whose _large_file_identity already
            # binds the path.
            result["_ordinary_path_identity"] = expected_ordinary_path_identity(file_path)
        return result

    # ------------------------------------------------------------------
    # single mode (default)
    # ------------------------------------------------------------------

    def _process_single(
        self, file_path: str, content: str, imports: list[str], language: str,
        bundle=None,
        *, context=None, additional_attempt: bool = False,
    ) -> dict:
        """One combined provider call, merged into the flat record."""
        t_start = time.monotonic()
        shape = self._profile_block(bundle, "combined")
        cleaned = self._call_agent(
            self._file_agent, file_path, content, imports, language, shape,
            context=context,
            planned_call=self._planned_call(
                file_path, "single", "combined", 1,
            ),
            additional_attempt=additional_attempt,
        )
        elapsed = time.monotonic() - t_start

        if isinstance(cleaned, dict) and cleaned.get("error"):
            logger.warning(
                "[FILE] %s | combined fallback: %s",
                file_path,
                cleaned.get("error", "unknown"),
            )
            return self._merge_single_failure(file_path, language, imports, cleaned)

        logger.info("[FILE] %s | combined ok  %.1fs", file_path, elapsed)
        return self._merge_single(file_path, language, imports, cleaned)

    def _merge_single(
        self, file_path: str, language: str, imports: list[str], cleaned: dict
    ) -> dict:
        """Merge a cleaned combined response with deterministic identity fields."""
        return flat_combined_result(file_path, language, imports, cleaned)

    def _merge_single_failure(
        self, file_path: str, language: str, imports: list[str], failure: dict
    ) -> dict:
        """Flat record for a failed combined call: identity preserved, error on
        the ``documentation`` key so ``_agent_errors()`` detects one failure.

        The response-contract marker fields (``response_contract_final``,
        ``response_contract_diagnostic``, ``response_contract_correction_attempted``)
        are copied onto the same sub-result so the execution layer can classify the
        failure as a non-retryable response-contract failure.
        """
        documentation: dict = {
            "error": failure.get("error", "unknown"),
            "agent": failure.get("agent", "FileDocumentationAgent"),
        }
        for marker_key in (
            "response_contract_final",
            "response_contract_diagnostic",
            "response_contract_correction_attempted",
            "provider_failure",
        ):
            if marker_key in failure:
                documentation[marker_key] = failure[marker_key]
        return {
            "file_path": file_path,
            "language": language,
            "extension": extension_of(file_path),
            "imports": imports,
            "description": "",
            "role_in_system": "",
            "functions": [],
            "classes": [],
            "exports": [],
            "structure": {},
            "dependencies_analysis": {},
            "key_concepts": [],
            "usage_example": "",
            "documentation": documentation,
            "state": "checked",
        }

    # ------------------------------------------------------------------
    # triple mode (opt-in legacy path)
    # ------------------------------------------------------------------

    def _process_triple(
        self, file_path: str, content: str, imports: list[str], language: str,
        bundle=None,
        *, context=None, additional_attempt: bool = False,
    ) -> dict:
        t_start = time.monotonic()

        if self.parallel:
            structure, dependencies = self._run_parallel(
                file_path, content, imports, language, bundle, t_start,
                context=context,
                additional_attempt=additional_attempt,
            )
        else:
            structure, dependencies = self._run_sequential(
                file_path, content, imports, language, bundle, t_start,
                context=context,
                additional_attempt=additional_attempt,
            )

        # DocumentationAgent always gets the other agents' context
        t_doc_start = time.monotonic()
        doc_shape = self._profile_block(bundle, "documentation")
        doc_call = self._planned_call(
            file_path,
            "triple",
            "documentation",
            3,
        )
        doc_additional_attempt = self._additional_attempt(
            doc_call, additional_attempt
        )
        if doc_shape is None:
            documentation = self._doc_agent._safe_run_with_context(
                file_path, content, imports, language, structure, dependencies,
                call_context=context,
                planned_call=doc_call,
                additional_attempt=doc_additional_attempt,
            )
        else:
            documentation = self._doc_agent._safe_run_with_context(
                file_path, content, imports, language, structure, dependencies,
                doc_shape,
                call_context=context,
                planned_call=doc_call,
                additional_attempt=doc_additional_attempt,
            )
        elapsed_doc = time.monotonic() - t_doc_start
        if isinstance(documentation, dict) and documentation.get("error"):
            logger.warning(
                "[FILE] %s | documentation fallback: %s",
                file_path,
                documentation.get("error", "unknown"),
            )
        else:
            logger.info("[FILE] %s | documentation ok  %.1fs", file_path, elapsed_doc)

        return self._merge(file_path, language, imports, structure, dependencies, documentation)

    # ------------------------------------------------------------------
    # Internal runners (triple mode)
    # ------------------------------------------------------------------

    def _run_parallel(
        self, file_path: str, content: str, imports: list[str], language: str,
        bundle=None, t_start: float | None = None,
        *, context=None, additional_attempt: bool = False,
    ) -> tuple[dict, dict]:
        if t_start is None:
            t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_struct = pool.submit(
                self._call_agent, self._structure_agent,
                file_path, content, imports, language,
                self._profile_block(bundle, "structure"),
                context=context,
                planned_call=self._planned_call(
                    file_path, "triple", "structure", 1,
                ),
                additional_attempt=additional_attempt,
            )
            future_dep = pool.submit(
                self._call_agent, self._dependency_agent,
                file_path, content, imports, language,
                self._profile_block(bundle, "dependency"),
                context=context,
                planned_call=self._planned_call(
                    file_path, "triple", "dependency", 2,
                ),
                additional_attempt=additional_attempt,
            )
            concurrent.futures.wait((future_struct, future_dep))

        for future in (future_struct, future_dep):
            try:
                error = future.exception()
            except concurrent.futures.CancelledError:
                continue
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise error
        structure = future_struct.result()
        dependencies = future_dep.result()

        elapsed = time.monotonic() - t_start
        self._log_agent_result("structure", file_path, structure, elapsed)
        self._log_agent_result("dependencies", file_path, dependencies, elapsed)
        return structure, dependencies

    def _run_sequential(
        self, file_path: str, content: str, imports: list[str], language: str,
        bundle=None, t_start: float | None = None,
        *, context=None, additional_attempt: bool = False,
    ) -> tuple[dict, dict]:
        if t_start is None:
            t_start = time.monotonic()
        structure = self._call_agent(
            self._structure_agent, file_path, content, imports, language,
            self._profile_block(bundle, "structure"),
            context=context,
            planned_call=self._planned_call(
                file_path, "triple", "structure", 1,
            ),
            additional_attempt=additional_attempt,
        )
        elapsed_struct = time.monotonic() - t_start
        self._log_agent_result("structure", file_path, structure, elapsed_struct)

        t_dep = time.monotonic()
        dependencies = self._call_agent(
            self._dependency_agent, file_path, content, imports, language,
            self._profile_block(bundle, "dependency"),
            context=context,
            planned_call=self._planned_call(
                file_path, "triple", "dependency", 2,
            ),
            additional_attempt=additional_attempt,
        )
        elapsed_dep = time.monotonic() - t_dep
        self._log_agent_result("dependencies", file_path, dependencies, elapsed_dep)
        return structure, dependencies

    @staticmethod
    def _profile_block(bundle, agent: str):
        """Return the resolved shape block for *agent* from the file's bundle."""
        if bundle is None:
            return None
        return bundle.selections[agent].block

    def _call_agent(
        self, agent, file_path, content, imports, language, shape, *,
        context=None, planned_call=None, additional_attempt: bool = False,
    ):
        return agent._safe_run(
            file_path,
            content,
            imports,
            language,
            shape,
            call_context=context,
            planned_call=planned_call,
            additional_attempt=self._additional_attempt(
                planned_call, additional_attempt
            ),
        )

    def _additional_attempt(
        self,
        planned_call: PlannedCall | None,
        requested: bool,
    ) -> bool:
        if self._call_tracker is None or planned_call is None:
            return requested
        return self._call_tracker.call_was_attempted(planned_call.call_id)

    @staticmethod
    def _planned_call(
        file_path: str, mode: str, agent: str, ordinal: int,
    ) -> PlannedCall:
        return PlannedCall(
            call_id=documentation_call_id(
                file_path, mode, agent, ordinal
            ),
            category="file-documentation",
            owner=file_path,
            ordinal=ordinal,
        )

    def _log_agent_result(self, agent_label: str, file_path: str, result: dict, elapsed: float) -> None:
        if isinstance(result, dict) and result.get("error"):
            logger.warning(
                "[FILE] %s | %s fallback: %s",
                file_path,
                agent_label,
                result.get("error", "unknown"),
            )
        else:
            logger.info("[FILE] %s | %s ok  %.1fs", file_path, agent_label, elapsed)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(
        self,
        file_path: str,
        language: str,
        imports: list[str],
        structure: dict,
        dependencies: dict,
        documentation: dict,
    ) -> dict:
        """Combine all agent outputs into one flat result dict."""
        return {
            # Identity
            "file_path": file_path,
            "language": language,
            "extension": extension_of(file_path),

            # From parser (deterministic)
            "imports": imports,

            # From StructureAgent
            "description": (
                documentation.get("description")
                or structure.get("description", "")
            ),
            "role_in_system": (
                documentation.get("role_in_system")
                or structure.get("role_in_system", "")
            ),
            "functions": structure.get("functions", []),
            "classes": structure.get("classes", []),
            "exports": structure.get("exports", []),
            "structure": structure,

            # From DependencyAgent
            "dependencies_analysis": dependencies.get("dependencies_analysis", dependencies),

            # From DocumentationAgent
            "key_concepts": documentation.get("key_concepts", []),
            "usage_example": documentation.get("usage_example", ""),
            "documentation": documentation,

            # Status
            "state": "checked",
        }
