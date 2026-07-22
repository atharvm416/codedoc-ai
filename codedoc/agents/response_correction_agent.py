"""The single shared targeted-response-correction component.

One :class:`ResponseCorrectionAgent` is constructed per run and injected into all
four agents.  It performs at most one targeted corrective provider call per
eligible response-contract failure, through the same usage-counted provider path
the agents use, and revalidates the corrected text through the identical
canonical :func:`~codedoc.agents.response_diagnostics.process_response` path.

It is intentionally not a :class:`~codedoc.agents.base_agent.BaseAgent` subclass:
it holds only the shared ``llm``, ``usage``, ``ledger`` and the ``enabled`` flag,
and never duplicates provider/usage accounting or the response-validation path.
Because ``repair`` is invoked at most once per agent per file, the one-call
guarantee is structural and needs no shared mutable per-file state, so triple-mode
parallel structure/dependency correction through the one shared instance is safe.
"""

from __future__ import annotations

from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, _call_llm_counted
from codedoc.agents.response_diagnostics import (
    MAX_CORRECTION_RESPONSE_CHARS,
    CorrectionLedger,
    ResponseDiagnostic,
)
from codedoc.core.error_classifier import _classify_failure
from codedoc.core.execution_model import CallManifestTracker, PlannedCall
from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import ResponseContractError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_CORRECTION_AGENT_NAME = "ResponseCorrectionAgent"

_SYSTEM = (
    "You are correcting a JSON response that failed a required schema contract. "
    "You respond ONLY with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """The previous {mode}/{agent} response for this {language} file did not \
satisfy the required JSON contract and must be corrected.

File: {file_path}
Rejection reason: {reason}
Fields removed or missing: {field_summary}

Imports found by static parser: {imports}

Code:
{content}

{shape_block}

Previous response (verbatim, may be malformed or incomplete):
{original_response}

Correction rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Repair the response to the exact JSON shape above
- Preserve every valid fact already present in the previous response
- Use the source and parser imports only to fill required missing information
- Output one complete replacement JSON response, not a patch or a diff
"""


class ResponseCorrectionAgent:
    """Run-scoped component that repairs one eligible response-contract failure."""

    def __init__(
        self,
        llm: LLMProvider,
        usage,
        ledger: CorrectionLedger,
        enabled: bool,
        call_tracker: CallManifestTracker | None = None,
    ) -> None:
        self._llm = llm
        self._usage = usage
        self._ledger = ledger
        self._enabled = bool(enabled)
        self._call_tracker = call_tracker

    def repair(
        self,
        *,
        agent: str,
        file_path: str,
        diagnostic: ResponseDiagnostic,
        correction_input: dict,
        revalidate,
        planned_call: PlannedCall | None = None,
    ) -> dict:
        """Attempt one targeted correction of a rejected response.

        Records the contract failure; if correction is disabled, raises with
        ``correction_attempted=False`` and makes zero provider calls.  Otherwise
        records the attempt, makes exactly one provider call through the shared
        usage-counted path, and revalidates the result through *revalidate* (the
        identical canonical ``process_response`` path).  A terminal-billing or
        global provider fault is re-raised unchanged (run-level abort; only the
        attempt is recorded); any other correction-call fault or an invalid
        corrected response fails the correction with ``correction_attempted=True``.
        """
        self._ledger.record_contract_failure()

        if not self._enabled:
            raise ResponseContractError(
                agent, file_path,
                "response correction is disabled; response-contract failure is final",
                diagnostic=diagnostic, correction_attempted=False,
            )

        self._ledger.record_attempt()
        system, prompt = self._build_prompt(correction_input, diagnostic)
        try:
            corrected_raw = _call_llm_counted(
                self._llm, self._usage,
                agent_name=_CORRECTION_AGENT_NAME, prompt=prompt, system=system,
                call_tracker=self._call_tracker,
                planned_call=planned_call,
                additional_attempt=True,
            )
        except Exception as exc:
            verdict = _classify_failure(exc, None)
            if verdict in ("terminal_billing", "global"):
                # Genuinely run-level: re-raise unchanged so the existing terminal
                # abort path stops the run with crash recovery left resumable.
                # Only the attempt is recorded (never a failure) so accounting is
                # honest about the aborted run.
                raise
            # Any other correction-call fault (rate-limit/timeout/connection/other
            # transport error) ends correction for this agent/file.
            self._ledger.record_failure()
            wrapped = ResponseContractError(
                agent, file_path,
                f"correction provider call failed ({verdict}); response-contract failure is final",
                diagnostic=diagnostic, correction_attempted=True,
            )
            wrapped.__cause__ = exc
            raise wrapped

        try:
            validated = revalidate(corrected_raw)
        except ResponseContractError as exc:
            self._ledger.record_failure()
            raise ResponseContractError(
                agent, file_path,
                "corrected response still failed the schema contract",
                diagnostic=exc.diagnostic, correction_attempted=True,
            ) from exc

        self._ledger.record_success()
        return validated

    # ------------------------------------------------------------------
    # Bounded correction prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _field_summary(diagnostic: ResponseDiagnostic) -> str:
        """Bounded, value-free field summary drawn from the diagnostic."""
        parts: list[str] = []
        for removed in diagnostic.removed:
            parts.append(f"{removed.field} ({removed.reason_code})")
        return "; ".join(parts) if parts else "(none)"

    def _build_prompt(
        self, correction_input: dict, diagnostic: ResponseDiagnostic
    ) -> tuple[str, str]:
        original = correction_input.get("original_response", "")
        prompt = _PROMPT_TEMPLATE.format(
            mode=correction_input.get("mode", ""),
            agent=correction_input.get("agent", ""),
            language=correction_input.get("language", ""),
            file_path=correction_input.get("file_path", ""),
            reason=diagnostic.reason_code,
            field_summary=self._field_summary(diagnostic),
            imports=correction_input.get("imports", []),
            content=correction_input.get("content", ""),
            shape_block=correction_input.get("shape_block", ""),
            original_response=original[:MAX_CORRECTION_RESPONSE_CHARS],
        )
        return _SYSTEM, prompt
