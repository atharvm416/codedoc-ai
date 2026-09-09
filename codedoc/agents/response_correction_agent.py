"""The single shared targeted-response-correction component.

One :class:`ResponseCorrectionAgent` is constructed per run and injected into all
four agents.  It performs at most one targeted corrective provider call per
eligible response-contract failure, through the same usage-counted provider path
the agents use, and revalidates the corrected text through the identical
canonical :func:`~codedoc.agents.response_diagnostics.process_response` path.

It is intentionally not a :class:`~codedoc.agents.base_agent.BaseAgent` subclass:
it holds only the shared ``llm``, ``usage``, ``ledger`` and the ``enabled`` flag,
and never duplicates provider/usage accounting or the response-validation path.
Because ``repair`` is invoked at most once per rejected response/agent invocation,
the one-call guarantee is structural and needs no shared mutable per-file state, so
triple-mode parallel structure/dependency correction through the one shared
instance is safe. A split file can therefore make more than one correction call
when more than one node (leaf/reducer/final) is rejected.
"""

from __future__ import annotations

import re
from concurrent.futures import CancelledError

import codedoc.core.file_division as _division
from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, _call_llm_counted
from codedoc.agents.narrative_terminology import NARRATIVE_TERMINOLOGY_RULES
from codedoc.agents.response_diagnostics import (
    MAX_CORRECTION_RESPONSE_CHARS,
    REASON_FIXED_CAP_EXCEEDED,
    REMOVAL_RESPONSE_CAP,
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
{terminology_rules}{cap_repair_rule}"""

# The narrative routes whose correction prompt also carries the conservative
# terminology rules. The dependency and internal-reduction routes are
# deliberately excluded because neither supplies terminology evidence to
# validate a narrative against, so the closed initialism rule is simply
# inapplicable on those routes -- not because the reduction route has no
# bounded narrative field. It has exactly one, ``narrative`` (bounded at
# ``MAX_REDUCTION_NARRATIVE_CHARS``); an over-cap ``narrative`` is handled by
# the shared fixed-capsule cap-repair rule below, not by terminology rules.
_TERMINOLOGY_CORRECTION_ROUTES = frozenset(
    {
        ("single", "combined"),      # single mode and the split final synthesis
        ("triple", "structure"),
        ("triple", "documentation"),
        ("split-leaf", "leaf"),
    }
)

# ---------------------------------------------------------------------------
# Fixed-capsule cap-repair instruction (F-1 / section 5.1)
# ---------------------------------------------------------------------------

_CAP_REPAIR_PROSE = "prose"
_CAP_REPAIR_IDENTIFIER = "identifier"

#: Closed fixed-capsule field-path -> (developer-owned bound-constant name,
#: contract category). The bound is a *name* resolved against
#: ``codedoc.core.file_division`` at render time, never a literal, so a bound
#: change there (or a test monkeypatch) moves the rendered target with it. This
#: table selects instruction *rendering* only -- which clauses appear and which
#: numeric target, if any. It is NOT the two-condition cap-repair gate, which
#: is evaluated first, on closed reason codes alone
#: (:func:`_cap_repair_applies`).
_FIELD_CAP_TABLE: dict[str, tuple[str, str]] = {
    "description": ("MAX_LEAF_DESCRIPTION_CHARS", _CAP_REPAIR_PROSE),
    "narrative": ("MAX_REDUCTION_NARRATIVE_CHARS", _CAP_REPAIR_PROSE),
    "functions[i].description": (
        "MAX_LEAF_SYMBOL_DESCRIPTION_CHARS", _CAP_REPAIR_PROSE,
    ),
    "classes[i].description": (
        "MAX_LEAF_SYMBOL_DESCRIPTION_CHARS", _CAP_REPAIR_PROSE,
    ),
    "functions[i].name": ("MAX_LEAF_SYMBOL_NAME_CHARS", _CAP_REPAIR_IDENTIFIER),
    "classes[i].name": ("MAX_LEAF_SYMBOL_NAME_CHARS", _CAP_REPAIR_IDENTIFIER),
    "functions[i].signature": (
        "MAX_LEAF_SYMBOL_SIGNATURE_CHARS", _CAP_REPAIR_IDENTIFIER,
    ),
    "classes[i].signature": (
        "MAX_LEAF_SYMBOL_SIGNATURE_CHARS", _CAP_REPAIR_IDENTIFIER,
    ),
    "exports[i]": ("MAX_LEAF_EXPORT_ITEM_CHARS", _CAP_REPAIR_IDENTIFIER),
}

_FIELD_INDEX_RE = re.compile(r"\[\d+\]")


def _normalized_field_path(field_path: str) -> str:
    """Collapse every ``[<int>]`` index segment to ``[i]`` so an indexed path
    the removal collector emits (``functions[3].description``) matches the
    closed table."""
    return _FIELD_INDEX_RE.sub("[i]", field_path)


def _prose_correction_target(hard_bound: int) -> int:
    """Aim-below target for a bounded prose field, derived from that field's own
    hard bound -- never a hardcoded literal.

    Reuses the reducer shape contract's existing headroom ratio
    (``MAX_REDUCTION_NARRATIVE_TARGET_CHARS`` / ``MAX_REDUCTION_NARRATIVE_CHARS``
    == 260/300); for a 300-bound field the target is 260, and it never exceeds
    the field's own hard bound.
    """
    target = (
        hard_bound
        * _division.MAX_REDUCTION_NARRATIVE_TARGET_CHARS
        // _division.MAX_REDUCTION_NARRATIVE_CHARS
    )
    return min(target, hard_bound)


def _resolve_field_cap(field_path: str) -> tuple[int, str] | None:
    """``(hard_bound, category)`` for a fixed-capsule field path, or ``None``
    when the path is outside the closed table.  Never parses ``RemovedField
    .detail``, shape-block prose, or any provider text."""
    entry = _FIELD_CAP_TABLE.get(_normalized_field_path(field_path))
    if entry is None:
        return None
    const_name, category = entry
    return getattr(_division, const_name), category


def _cap_repair_applies(diagnostic: ResponseDiagnostic) -> bool:
    """The exact fixed-capsule cap-repair gate. Both conditions are required:

    1. the top-level reason is ``fixed_cap_exceeded`` -- only
       :func:`process_fixed_capsule_response` raises it, so it discriminates the
       fixed leaf/reduction routes exactly; and
    2. the complete, untruncated observed removal-reason set contains a
       per-field ``response_cap`` overrun.

    Condition 1 alone also fires for an ``item_limit`` elevation (remedy: fewer
    items, not shorter prose); condition 2 alone also fires for a
    configurable-route global combined-cap eviction. The set is read from
    ``diagnostic.observed_removal_reasons`` -- never ``diagnostic.removed``,
    which is truncated at ``MAX_REMOVAL_ENTRIES`` and can drop the
    ``response_cap`` entry on a saturated capsule.
    """
    return (
        diagnostic.reason_code == REASON_FIXED_CAP_EXCEEDED
        and REMOVAL_RESPONSE_CAP in diagnostic.observed_removal_reasons
    )


def _cap_repair_rule(diagnostic: ResponseDiagnostic) -> str:
    """Render the shared cap-repair instruction once after the gate accepts.

    Precondition: :func:`_cap_repair_applies` returned True.

    Rewrite-shorter guidance (clauses 3 and 4) is emitted only for a *known*
    bounded prose field in the closed resolver table. A source-backed
    identifier field and a field path outside the closed table get the
    identifier-shaped conservative form: clauses 1, 2 and 5, then a deferral
    to the governing shape contract
    already supplied in the prompt, with no prose-shortening instruction and no
    numeric target. When the ``MAX_REMOVAL_ENTRIES``-bounded
    ``diagnostic.removed`` tuple has saturated and no ``response_cap`` field
    path survived at all, the field cannot even be classified -- it may be an
    over-cap prose field (the G-2 leaf ``description`` example) or an identifier
    -- so the fallback is category-neutral: it keeps clauses 1, 2 and 5, defers
    to the governing shape contract, and supplies no field-specific rewrite
    instruction and no numeric target, while neither commanding nor prohibiting
    prose shortening.
    """
    named: list[str] = []
    for removed in diagnostic.removed:
        if removed.reason_code == REMOVAL_RESPONSE_CAP and removed.field not in named:
            named.append(removed.field)

    prose_targets: list[str] = []
    identifier_fields: list[str] = []
    for field_path in named:
        resolved = _resolve_field_cap(field_path)
        if resolved is not None and resolved[1] == _CAP_REPAIR_PROSE:
            prose_targets.append(
                f"{field_path}: within {_prose_correction_target(resolved[0])} "
                "characters"
            )
        else:
            # An identifier field, or a path outside the closed table: the
            # conservative form -- clauses 1, 2, 5 and shape-contract deferral,
            # never an invented numeric target.
            identifier_fields.append(field_path)

    # Required precedence: clauses 1, 2, then 5, then the field-specific
    # remedy (prose target for clauses 3-4, or the shape-contract deferral for
    # an identifier / unknown / unavailable field). Clause 5 is its own
    # always-present line so it can never be dropped with the prose clauses.
    clause_1_2 = (
        "- Each field named below had its value rejected in full for exceeding "
        "that field's hard character cap, so that value is not a valid fact to "
        "keep and its previous wording must not be copied."
        if named
        else "- A field's value was rejected in full for exceeding its hard "
        "character cap, so that value is not a valid fact to keep and its "
        "previous wording must not be copied."
    )
    lines = [
        "",
        "Cap repair:",
        clause_1_2,  # clauses 1, 2
        '- This cap-repair rule overrides the general "Preserve every valid '
        'fact ..." instruction above for the affected field(s); every other '
        "valid field and fact from the previous response must still be "
        "preserved.",  # clause 5
    ]
    if prose_targets:
        lines.append(
            "- Rewrite each of these bounded prose fields as a shorter, "
            "meaning-preserving value that fits within its stated target, "
            "comfortably below the hard cap rather than at or just under it -- "
            + "; ".join(prose_targets)
            + "."  # clauses 3, 4
        )
    if identifier_fields:
        lines.append(
            "- For each of these fields, produce a valid replacement by "
            "following the governing shape contract for that field already "
            "stated above (the fixed shape block and the signature and export "
            "contracts it carries); this rule states no length target of its "
            "own, and you must not invent a shorter identifier, a shortened "
            "name, or a placeholder value -- " + "; ".join(identifier_fields) + "."
        )
    if not named:
        # The over-cap ``response_cap`` field path did not survive the bounded
        # ``removed`` tuple, so its category is unknown: it may be an over-cap
        # prose field (leaf ``description`` -- the G-2 example) or a
        # source-backed identifier. The fallback is therefore category-neutral
        # -- it neither commands prose shortening nor prohibits it, states no
        # target, and invents nothing -- and defers to the shape contract.
        lines.append(
            "- The over-cap field could not be identified from this "
            "diagnostic. Follow the governing shape contract supplied above "
            "for the affected field. This fallback supplies no field-specific "
            "rewrite instruction and no numeric target, and must not be used "
            "to invent an identifier, a shortened name, a placeholder value, a "
            "field identity, or an unstated limit."
        )
    return "\n".join(lines)


#: Section 5.3: an explicit, deterministic acceptance ceiling for the rendered
#: field summary. The summary is already structurally bounded well below this
#: (measured worst case ~9,982 chars), so this is an assertable invariant, not a
#: truncation point -- silently dropping an entry could drop the very field the
#: cap-repair rule names.
_FIELD_SUMMARY_CEILING_CHARS = 10_000


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
        if self._call_tracker is not None:
            self._call_tracker.raise_if_cancelled()

        if not self._enabled:
            raise ResponseContractError(
                agent, file_path,
                "response correction is disabled; response-contract failure is "
                f"final ({diagnostic.reason_code})",
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
        except CancelledError:
            raise
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
                f"correction provider call failed ({verdict}); response-contract "
                f"failure is final ({diagnostic.reason_code})",
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
                "corrected response still failed the schema contract "
                f"({exc.diagnostic.reason_code})",
                diagnostic=exc.diagnostic, correction_attempted=True,
            ) from exc

        self._ledger.record_success()
        return validated

    # ------------------------------------------------------------------
    # Bounded correction prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _field_summary(diagnostic: ResponseDiagnostic) -> str:
        """Bounded, value-free field summary drawn from the diagnostic.

        Field path and closed reason code only -- never a field value and never
        ``RemovedField.detail``. The result is structurally bounded at
        ``MAX_REMOVAL_ENTRIES`` x (``MAX_PATH_CHARS`` + reason code +
        separators); section 5.3's ``_FIELD_SUMMARY_CEILING_CHARS`` is asserted
        as an invariant here rather than enforced by truncation.
        """
        parts: list[str] = []
        for removed in diagnostic.removed:
            parts.append(f"{removed.field} ({removed.reason_code})")
        rendered = "; ".join(parts) if parts else "(none)"
        assert len(rendered) <= _FIELD_SUMMARY_CEILING_CHARS, (
            "rendered correction field summary exceeded its 10,000-char ceiling "
            "(section 12 stop condition)"
        )
        return rendered

    def _build_prompt(
        self, correction_input: dict, diagnostic: ResponseDiagnostic
    ) -> tuple[str, str]:
        original = correction_input.get("original_response", "")
        mode = correction_input.get("mode", "")
        agent = correction_input.get("agent", "")
        terminology_rules = (
            NARRATIVE_TERMINOLOGY_RULES + "\n"
            if (mode, agent) in _TERMINOLOGY_CORRECTION_ROUTES
            else ""
        )
        # The cap-repair rule appears only when the exact fixed-capsule gate
        # holds. It can therefore never reach a configurable single/triple
        # correction (whose ``process_response`` path never raises
        # ``fixed_cap_exceeded``), so those prompts stay byte-identical.
        cap_repair_rule = (
            _cap_repair_rule(diagnostic) if _cap_repair_applies(diagnostic) else ""
        )
        prompt = _PROMPT_TEMPLATE.format(
            mode=mode,
            agent=agent,
            language=correction_input.get("language", ""),
            file_path=correction_input.get("file_path", ""),
            reason=diagnostic.reason_code,
            field_summary=self._field_summary(diagnostic),
            imports=correction_input.get("imports", []),
            content=correction_input.get("content", ""),
            shape_block=correction_input.get("shape_block", ""),
            original_response=original[:MAX_CORRECTION_RESPONSE_CHARS],
            terminology_rules=terminology_rules,
            cap_repair_rule=cap_repair_rule,
        )
        return _SYSTEM, prompt
