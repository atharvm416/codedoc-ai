"""
Structure Agent.

Analyses a file's internal structure:
  - Key functions and their purpose
  - Classes and their responsibility
  - Exported symbols
  - The file's role in the overall system
"""

from __future__ import annotations

from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, BaseAgent
from codedoc.core.execution_model import AgentCallContext, PlannedCall
from codedoc.agents.response_cleaning import clean_structure_report
from codedoc.core.prompt_profiles import (
    ResolvedShapeBlock,
    default_shape_block,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a senior software engineer analysing source code. "
    "You respond ONLY with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """Analyse this {language} file and respond with a JSON object.

File: {file_path}
Imports: {imports}

Code:
{content}

{shape_block}

Rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Use only information from the provided code
- functions and classes must be ones DEFINED IN this file — never list an imported
  or re-exported name as a local function or class unless it is also defined here
- exports are names this module deliberately exposes (including intentional
  re-exports in a package initializer); an ordinary imported helper is not an export
- For a package initializer (e.g. __init__.py), describe imported names as
  re-exports, not as locally implemented classes or functions
- Keep descriptions concise (1-2 sentences)
- If the Code above ends with a truncation marker ("... [truncated] ..."), part
  of the middle of the file is omitted; report only what is visible in the
  supplied head and tail slices — never infer the omitted middle
- If functions, classes, or exports are not present, omit that key instead of returning an empty list
- Do not include empty arrays, empty objects, null values, or duplicate fields
"""


def build_prompt(
    file_path: str,
    content: str,
    imports: list[str],
    language: str,
    requested_shape: ResolvedShapeBlock | None = None,
) -> tuple[str, str]:
    """Return ``(system, prompt)`` exactly as sent to the provider.

    *content* must already be truncated by the caller.  Used by ``run()`` and
    by dry-run usage estimation so estimates match real prompts.

    *requested_shape* supplies the requested-shape block; ``None`` reproduces the
    developer-standard block byte for byte.
    """
    shape_block = (
        requested_shape.text
        if requested_shape is not None
        else default_shape_block("triple", "structure")
    )
    prompt = _PROMPT_TEMPLATE.format(
        language=language,
        file_path=file_path,
        imports=imports,
        content=content,
        shape_block=shape_block,
    )
    return _SYSTEM, prompt


class StructureAgent(BaseAgent):
    agent_name = "StructureAgent"

    def run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: ResolvedShapeBlock | None = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        if call_context is not None:
            requested_shape = call_context.resolved_shape_bundle.selections[
                "structure"
            ].block
        truncated = self._truncate(content, file_path, call_context=call_context)
        system, prompt = build_prompt(
            file_path, truncated, imports, language, requested_shape,
        )
        shape_block = (
            requested_shape.text
            if requested_shape is not None
            else default_shape_block("triple", "structure")
        )
        raw = self._call_llm(
            prompt,
            system=system,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )
        cleaned = self._finalize_response(
            raw,
            mode="triple",
            agent="structure",
            file_path=file_path,
            clean_reporter=clean_structure_report,
            resolved_shape=requested_shape,
            content=truncated,
            imports=imports,
            language=language,
            shape_block=shape_block,
            planned_call=planned_call,
        )

        logger.debug(
            "StructureAgent: %s → %d functions, %d classes",
            file_path,
            len(cleaned.get("functions", [])),
            len(cleaned.get("classes", [])),
        )
        return cleaned
