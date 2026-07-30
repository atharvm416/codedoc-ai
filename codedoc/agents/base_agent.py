"""
Abstract base agent.

Every agent (structure, dependency, documentation) inherits this class.
Provides shared prompt building, JSON extraction, and error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import CancelledError

from codedoc.agents.response_diagnostics import (
    extract_json_candidate,
    process_fixed_capsule_response,
    process_response,
)
from codedoc.core.execution_model import (
    AgentCallContext,
    CallManifestTracker,
    PlannedCall,
)
from codedoc.core.usage import UsageAccumulator
from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import AgentError, ResponseContractError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

TRUNCATION_MARKER = "\n... [truncated] ...\n"

# Head/tail split policy.  When a file exceeds the
# character ceiling we keep a larger leading segment (module docstring, imports,
# early definitions) and a bounded trailing segment (late definitions, entry
# points) with the marker between them, instead of a leading prefix only.  The
# fractions apply to the budget left after reserving room for the marker.  A
# ~70/30 head/tail split keeps import/context bias while making late class and
# function definitions visible.
TRUNCATION_HEAD_FRACTION = 0.70

# Strong exact-response contract clauses, interpolated once into every agent's
# ``_PROMPT_TEMPLATE`` so the four modules cannot drift apart.  Kept free of ``{``
# / ``}`` so it can be concatenated into a ``str.format`` template safely.  Each
# of the four assembled prompts must contain every clause verbatim.
EXACT_JSON_RESPONSE_RULES = (
    "- Return one JSON object only\n"
    "- No Markdown fences and no prose outside the JSON\n"
    "- Use exactly the requested top-level keys\n"
    "- Do not rename requested keys or invent aliases for them\n"
    "- Preserve the requested scalar/list/object type of each field\n"
    "- Provide every required field with a non-empty, valid value\n"
    "- Omit optional information only as the schema permits\n"
    "- Never invent facts to fill a field"
)


def truncate_for_llm(
    content: str,
    max_chars: int,
    head_fraction: float = TRUNCATION_HEAD_FRACTION,
) -> str:
    """Truncate *content* to a head-plus-tail slice within *max_chars*.

    Content above the ceiling keeps leading and trailing segments joined by
    :data:`TRUNCATION_MARKER`, preserving late definitions and entry points.
    Guarantees:

    - the result — both complete character slices plus the marker — never exceeds
      *max_chars* (the marker is counted inside the ceiling);
    - short content (``len <= max_chars``) is returned byte-for-byte unchanged;
    - slicing is by Unicode code point, so a character is never split;
    - the same bounded source is produced for every caller (orchestrator, each
      agent's defensive fallback, and dry-run estimation), so single and triple
      modes and the dry-run estimate all see exactly the same truncated text.

    The *head_fraction* parameter (default :data:`TRUNCATION_HEAD_FRACTION`
    = 0.70) is configurable via ``truncation_head_ratio`` in the config
    or ``--truncation-head-ratio`` on the CLI.  Callers that omit the argument
    continue to use the 0.70 default and produce byte-identical output.
    """
    if len(content) <= max_chars:
        return content
    marker = TRUNCATION_MARKER
    # Degenerate ceiling smaller than the marker itself: emit a marker prefix so
    # the result still signals truncation without exceeding the ceiling.
    if max_chars <= len(marker):
        return marker[:max_chars]
    budget = max_chars - len(marker)
    head_len = int(budget * head_fraction)
    tail_len = budget - head_len
    head = content[:head_len]
    tail = content[len(content) - tail_len:] if tail_len > 0 else ""
    return head + marker + tail


def _call_llm_counted(
    llm: LLMProvider,
    usage: UsageAccumulator | None,
    *,
    agent_name: str,
    prompt: str,
    system: str = "",
    call_tracker: CallManifestTracker | None = None,
    planned_call: PlannedCall | None = None,
    additional_attempt: bool = False,
) -> str:
    """The single provider-call accounting path for initial and correction calls.

    Records estimated input tokens before the call and a success (with estimated
    output tokens) or a failure after, isolating accounting errors from the call
    outcome, and wraps provider exceptions as
    ``AgentError(agent_name, "unknown", str(exc))``.  Both ``BaseAgent._call_llm``
    and ``ResponseCorrectionAgent`` delegate here so provider/usage accounting is
    never duplicated.
    """
    if call_tracker is not None:
        call_tracker.raise_if_cancelled()
    if call_tracker is not None and planned_call is None:
        raise RuntimeError(
            "call manifest tracking requires a planned call."
        )
    if call_tracker is not None and planned_call is not None:
        call_tracker.authorize(
            planned_call, additional_attempt=additional_attempt
        )
    if usage is not None:
        try:
            usage.record_input(system, prompt)
        except Exception:
            logger.debug("Usage accounting failed for input estimate", exc_info=True)
    if call_tracker is not None:
        call_tracker.raise_if_cancelled()
    try:
        raw = llm.complete_json(prompt, system=system)
    except CancelledError:
        raise
    except (KeyboardInterrupt, SystemExit):
        if call_tracker is not None:
            call_tracker.signal_stop()
        raise
    except Exception as exc:
        if usage is not None:
            try:
                usage.record_failure()
            except Exception:
                logger.debug("Usage accounting failed for failed call", exc_info=True)
        raise AgentError(agent_name, "unknown", str(exc)) from exc
    if usage is not None:
        try:
            usage.record_success(raw)
        except Exception:
            logger.debug("Usage accounting failed for successful call", exc_info=True)
    return raw


class BaseAgent(ABC):
    """
    Abstract agent.

    Subclasses implement `run()` which returns a dict of results.
    """

    #: Override in subclass — used in error messages and logs
    agent_name: str = "BaseAgent"

    def __init__(
        self,
        llm: LLMProvider,
        max_content_chars: int = 12000,
        usage: UsageAccumulator | None = None,
        call_tracker: CallManifestTracker | None = None,
    ) -> None:
        self.llm = llm
        self._max_content_chars = max_content_chars
        self._usage = usage
        self._call_tracker = call_tracker
        # Shared correction component, injected by the orchestrator after the four
        # agents are constructed.  ``None`` (the default for a directly constructed
        # agent, or an orchestrator built without a correction ledger) means an
        # eligible response-contract failure raises without any correction call.
        self._correction: "object | None" = None

    @abstractmethod
    def run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: "object | None" = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        """
        Analyse one file and return a result dict.

        Args:
            file_path: relative path string (for context in the prompt)
            content:   raw file content
            imports:   list of import strings extracted by the parser
            language:  detected language tag
            requested_shape: optional resolved prompt-profile block for this
                file/agent.  ``None`` means the developer-standard shape.

        Returns:
            dict of results — structure depends on the agent subclass
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _truncate(
        self,
        content: str,
        file_path: str = "",
        *,
        call_context: AgentCallContext | None = None,
    ) -> str:
        """Defensive truncation fallback for direct agent callers.

        Orchestrated runs truncate once in ``Orchestrator.process()`` before
        the content reaches any agent, so this is a no-op there.  The message
        is DEBUG so a normal orchestrated run never emits three warnings for
        one file.  The marker fits inside the ceiling.
        """
        max_chars = (
            call_context.max_content_chars
            if call_context is not None
            else self._max_content_chars
        )
        head_fraction = (
            call_context.truncation_head_ratio
            if call_context is not None
            else TRUNCATION_HEAD_FRACTION
        )
        if len(content) > max_chars:
            logger.debug(
                "Content truncated: %s (%d chars -> %d chars). "
                "Raise max_content_chars in config to include more content.",
                file_path or "file",
                len(content),
                max_chars,
            )
            return truncate_for_llm(
                content, max_chars, head_fraction=head_fraction
            )
        return content

    def _call_llm(
        self,
        prompt: str,
        system: str = "",
        *,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> str:
        """Call the LLM and return raw text. Wraps errors as AgentError.

        Delegates to the shared :func:`_call_llm_counted` accounting path so
        initial and correction calls are counted identically.
        """
        return _call_llm_counted(
            self.llm, self._usage,
            agent_name=self.agent_name, prompt=prompt, system=system,
            call_tracker=self._call_tracker,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )

    def _parse_json(self, raw: str, file_path: str) -> dict:
        """
        Extract and parse a JSON object from LLM output.
        Handles models that wrap JSON in markdown fences.

        Delegates candidate selection/parsing to the one shared extraction in
        :mod:`codedoc.agents.response_diagnostics`. Error messages contain only
        bounded structural metadata, never provider-response excerpts, so custom
        ``BaseAgent`` subclasses cannot leak echoed source or secrets through
        ``_safe_run`` warning logs.
        """
        outcome = extract_json_candidate(raw)
        if outcome.kind == "object":
            return outcome.value
        if outcome.kind == "json_parse_error":
            raise AgentError(
                self.agent_name, file_path,
                f"JSON parse error: {outcome.parse_error}"
            )
        # no_json_object or top_level_not_object: the legacy path never parsed a
        # brace-less candidate, so both surface the same historical message.
        raise AgentError(
            self.agent_name, file_path,
            "LLM did not return a JSON object."
        )

    def _finalize_response(
        self,
        raw: str,
        *,
        mode: str,
        agent: str,
        file_path: str,
        clean_reporter,
        resolved_shape,
        content: str,
        imports: list[str],
        language: str,
        shape_block: str,
        planned_call: PlannedCall | None = None,
    ) -> dict:
        """Validate *raw* through the canonical path, correcting once if eligible.

        Runs the shared :func:`~codedoc.agents.response_diagnostics.process_response`
        (extraction -> clean -> profile filter -> no-usable-fields ->
        registry-required-field validation).  On an eligible
        :class:`ResponseContractError`, when a shared correction component is
        injected (``self._correction is not None``) it calls ``repair`` exactly
        once with the bounded correction input and the identical validation path;
        otherwise the typed failure propagates without any correction call.
        """
        def _revalidate(text: str) -> dict:
            return process_response(
                text, mode=mode, agent=agent, file_path=file_path,
                clean_reporter=clean_reporter, resolved_shape=resolved_shape,
            )

        contract_error: ResponseContractError | None = None
        try:
            return _revalidate(raw)
        except ResponseContractError as exc:
            if self._correction is None:
                raise
            contract_error = exc

        # Invoke correction *outside* the ``except`` block so a provider fault
        # raised by the correction call is not implicitly chained (``__context__``)
        # to the initial ``ResponseContractError``.  Such a chain would make
        # ``_classify_failure`` return ``response_contract_final`` for a genuine
        # terminal-billing/global correction fault and defeat the run-level abort.
        correction_input = {
            "agent": agent,
            "mode": mode,
            "file_path": file_path,
            "language": language,
            "content": content,
            "imports": list(imports),
            "shape_block": shape_block,
            "original_response": raw,
        }
        return self._correction.repair(
            agent=agent,
            file_path=file_path,
            diagnostic=contract_error.diagnostic,
            correction_input=correction_input,
            revalidate=_revalidate,
            planned_call=planned_call,
        )

    def _finalize_fixed_response(
        self,
        raw: str,
        *,
        label: str,
        agent: str,
        file_path: str,
        clean_reporter,
        requested_paths: tuple[str, ...],
        required_paths: tuple[str, ...],
        content: str,
        shape_block: str,
        planned_call: PlannedCall | None = None,
    ) -> dict:
        """Validate *raw* against a fixed (non-profile) capsule contract.

        Split leaf and reduction capsules are developer-owned and never
        prompt-profile driven (D5/D6): this mirrors :meth:`_finalize_response`
        exactly, but runs :func:`~codedoc.agents.response_diagnostics.
        process_fixed_capsule_response` (no profile-filter stage) and passes
        empty ``imports`` to the shared correction prompt, since a fixed
        capsule correction never needs parser imports.
        """
        def _revalidate(text: str) -> dict:
            return process_fixed_capsule_response(
                text,
                label=label,
                agent=agent,
                file_path=file_path,
                clean_reporter=clean_reporter,
                requested_paths=requested_paths,
                required_paths=required_paths,
            )

        contract_error: ResponseContractError | None = None
        try:
            return _revalidate(raw)
        except ResponseContractError as exc:
            if self._correction is None:
                raise
            contract_error = exc

        correction_input = {
            "agent": agent,
            "mode": label,
            "file_path": file_path,
            "language": "",
            "content": content,
            "imports": [],
            "shape_block": shape_block,
            "original_response": raw,
        }
        return self._correction.repair(
            agent=agent,
            file_path=file_path,
            diagnostic=contract_error.diagnostic,
            correction_input=correction_input,
            revalidate=_revalidate,
            planned_call=planned_call,
        )

    def _agent_error_result(self, exc: Exception) -> dict:
        """Build the error-dict fallback for a failed agent run.

        For a :class:`ResponseContractError` the dict additionally carries the
        non-retryable marker ``response_contract_final``, a bounded json-safe
        diagnostic summary, and ``response_contract_correction_attempted`` so the
        execution layer can re-raise a typed, non-retryable failure and accounting
        can distinguish an opt-out rejection from an exhausted correction.  Every
        other exception returns the historical ``{"error", "agent"}`` dict.
        """
        result = {"error": str(exc), "agent": self.agent_name}
        if isinstance(exc, ResponseContractError):
            result["response_contract_final"] = True
            result["response_contract_correction_attempted"] = exc.correction_attempted
            if exc.diagnostic is not None:
                summary = getattr(exc.diagnostic, "as_summary", None)
                result["response_contract_diagnostic"] = (
                    summary() if callable(summary) else exc.diagnostic
                )
        return result

    def _safe_run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: "object | None" = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        """
        Wrapper that catches all errors and returns a fallback dict
        instead of crashing the pipeline.
        """
        try:
            if (
                requested_shape is None
                and call_context is None
                and planned_call is None
                and not additional_attempt
            ):
                return self.run(file_path, content, imports, language)
            return self.run(
                file_path,
                content,
                imports,
                language,
                requested_shape,
                call_context=call_context,
                planned_call=planned_call,
                additional_attempt=additional_attempt,
            )
        except AgentError as exc:
            logger.warning("%s failed on %s: %s", self.agent_name, file_path, exc)
            return self._agent_error_result(exc)
        except CancelledError:
            raise
        except Exception as exc:
            logger.warning("%s unexpected error on %s: %s", self.agent_name, file_path, exc)
            return self._agent_error_result(exc)
