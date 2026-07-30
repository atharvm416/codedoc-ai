"""Fixed intermediate reduction and final synthesis agent for deterministically
divided files.

Both prompt variants live in this one module (no second synthesis-agent
module):

- the fixed internal reduction (narrative-only) prompt, used for both
  unit-consolidation and general `file-reduction` nodes; and
- the final synthesis prompt, which alone uses the resolved (possibly
  user-customized) final file-level shape.
"""

from __future__ import annotations

from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, BaseAgent
from codedoc.agents.response_cleaning import clean_combined_report, clean_reduction_capsule_report
from codedoc.core.execution_model import (
    AgentCallContext,
    FileReductionExecutionRequest,
    PlannedCall,
)
from codedoc.core.file_division import (
    FINAL_SYNTHESIS_REVISION,
    REDUCTION_ENVELOPE_OVERHEAD_CHARS,
    DivisionInternalDefect,
    render_reduction_child_manifest,
)
from codedoc.core.prompt_profiles import ResolvedShapeBlock, default_shape_block

# Compatibility alias: retained under its historical name because
# ``codedoc.agents.orchestrator`` and call-ID construction import it from here.
SYNTHESIS_PROMPT_REVISION = FINAL_SYNTHESIS_REVISION

REDUCTION_CAPSULE_REQUESTED_PATHS: tuple[str, ...] = ("narrative",)
REDUCTION_CAPSULE_REQUIRED_PATHS: tuple[str, ...] = ("narrative",)

# ---------------------------------------------------------------------------
# Fixed internal reduction (narrative-only) prompt — D6/section 9
# ---------------------------------------------------------------------------

_REDUCTION_SYSTEM = (
    "You are a senior technical writer refining already-reviewed narrative "
    "summaries from sibling fragments of the same source file into one "
    "combined narrative. You respond ONLY with valid JSON — no markdown, no "
    "explanation."
)

_REDUCTION_SHAPE_BLOCK = (
    "Return exactly this fixed internal JSON shape:\n"
    '{\n  "narrative": "<required, non-empty refined narrative>"\n}'
)

_REDUCTION_PROMPT_TEMPLATE = """Refine one combined narrative from {child_count} already-reviewed \
child narrative(s) belonging to the same source file. This is an internal \
reduction step, not the final file-level documentation.

File: {file_path}
Reduction phase: {phase}
Tree level: {level}

Child narratives, in source order (already de-duplicated):
{child_narratives}

{shape_block}

Rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Combine and refine only the supplied child narratives
- Do not invent facts, symbols, files, or relationships not present above
- Do not invent source ranges, unit/chunk identifiers, or coverage claims
- Do not list functions, classes, or exports — those are tracked separately \
and are never part of this response
- Prefer a concise combined narrative over restating every child verbatim
- Omit a key instead of returning an empty list, empty object, or null
- Do not include duplicate fields
"""


def build_reduction_prompt(
    request: FileReductionExecutionRequest, child_narratives: tuple[str, ...]
) -> tuple[str, str]:
    """Return ``(system, prompt)`` for one fixed internal reduction request.

    *child_narratives* must already be the deterministically de-duplicated,
    source-ordered narrative text for `request.child_ids` (see
    :func:`codedoc.core.file_division.refine_narrative_inputs`).
    """
    rendered = (
        render_reduction_child_manifest(child_narratives)
        or "(no narrative text)"
    )
    prompt = _REDUCTION_PROMPT_TEMPLATE.format(
        child_count=len(request.child_ids),
        file_path=request.rel_path,
        phase=request.phase,
        level=request.level,
        child_narratives=rendered,
        shape_block=_REDUCTION_SHAPE_BLOCK,
    )
    return _REDUCTION_SYSTEM, prompt


# ---------------------------------------------------------------------------
# Final synthesis prompt — D10
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a senior software engineer and technical writer synthesizing "
    "the final file-level documentation for one deterministically divided "
    "source file, grounded in already-reviewed narrative and structural "
    "facts. Respond only with valid JSON."
)

_PROMPT_TEMPLATE = """Synthesize one final file-level documentation JSON object for a large file \
that was analyzed in bounded fragments.

File: {file_path}
Deterministic grounding manifest (path, language, imports, refined root \
narrative(s), exact coverage count, and a bounded structural-fact-ledger \
synopsis):
{manifest}

{shape_block}

Rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Use only the supplied root narrative(s), imports, and fact-ledger synopsis
- Do not invent source ranges, internal unit/chunk identifiers, or \
relationships not present above
- functions, classes, and exports you return must come only from the \
supplied fact-ledger synopsis
- Do not include duplicate fields
"""


def build_prompt(
    file_path: str,
    manifest: str,
    requested_shape: ResolvedShapeBlock | None = None,
) -> tuple[str, str]:
    shape_block = (
        requested_shape.text
        if requested_shape is not None
        else default_shape_block("single", "combined")
    )
    return _SYSTEM, _PROMPT_TEMPLATE.format(
        file_path=file_path,
        manifest=manifest,
        shape_block=shape_block,
    )


class FileSynthesisAgent(BaseAgent):
    """Fixed internal reduction calls plus the final resolved-shape synthesis
    call for one divided file."""

    agent_name = "FileSynthesisAgent"

    def run_reduction(
        self,
        request: FileReductionExecutionRequest,
        child_narratives: tuple[str, ...],
        *,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        """Run one fixed internal unit-consolidation or general reduction call.

        Never receives raw source, a user-customized shape, or authoritative
        coverage IDs; the caller attaches and verifies exact descendant
        coverage locally (D6/section 9).
        """
        rendered_children = render_reduction_child_manifest(child_narratives)
        if (
            len(rendered_children) + REDUCTION_ENVELOPE_OVERHEAD_CHARS
            > request.context.max_content_chars
        ):
            raise DivisionInternalDefect(
                "planned reduction child manifest exceeds "
                "max_content_chars."
            )
        system, prompt = build_reduction_prompt(request, child_narratives)
        raw = self._call_llm(
            prompt,
            system=system,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )
        return self._finalize_fixed_response(
            raw,
            label="split-reduction",
            agent="reduction",
            file_path=request.rel_path,
            clean_reporter=clean_reduction_capsule_report,
            requested_paths=REDUCTION_CAPSULE_REQUESTED_PATHS,
            required_paths=REDUCTION_CAPSULE_REQUIRED_PATHS,
            content=rendered_children,
            shape_block=_REDUCTION_SHAPE_BLOCK,
            planned_call=planned_call,
        )

    def run(
        self,
        file_path: str,
        manifest: str,
        requested_shape: ResolvedShapeBlock | None = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        if call_context is not None and len(manifest) > call_context.max_content_chars:
            raise ValueError("synthesis manifest exceeds max_content_chars.")
        if requested_shape is None:
            raise ValueError("split-file synthesis requires a resolved shape.")
        system, prompt = build_prompt(file_path, manifest, requested_shape)
        raw = self._call_llm(
            prompt,
            system=system,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )
        return self._finalize_response(
            raw,
            mode="single",
            agent="combined",
            file_path=file_path,
            clean_reporter=clean_combined_report,
            resolved_shape=requested_shape,
            content=manifest,
            imports=[],
            language="",
            shape_block=requested_shape.text if requested_shape else default_shape_block("single", "combined"),
            planned_call=planned_call,
        )
