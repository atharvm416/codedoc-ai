"""
Documentation Agent.

Takes the outputs from StructureAgent and DependencyAgent and
generates the final polished documentation for a file:
  - Enriched description
  - Clear role in system
  - Human-readable summary suitable for the .md output
"""

from __future__ import annotations

from concurrent.futures import CancelledError

from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, BaseAgent
from codedoc.core.execution_model import AgentCallContext, PlannedCall
from codedoc.agents.response_cleaning import clean_documentation_report
from codedoc.core.prompt_profiles import (
    ResolvedShapeBlock,
    default_shape_block,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a technical writer creating clear, accurate documentation for a software codebase. "
    "You respond ONLY with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """Generate documentation for this {language} file.

File: {file_path}

Structure analysis:
{structure}

Dependency analysis:
{dependencies}

Code:
{content}

{shape_block}

Rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Base the documentation only on the provided code and analyses
- Write for a developer who is new to this codebase
- Be specific — avoid generic phrases like 'this file contains utility functions'
- If there are no noteworthy key_concepts, omit key_concepts instead of returning an empty list
- Include usage_example only when it is directly supported by this file's real
  public API and project/package path; omit it when caller context is absent or
  uncertain rather than inventing one
- Never invent placeholder paths or packages such as path/to/file, example.py,
  my_module, your_project, your_package, your_app, or your_package_name
- If the Code above ends with a truncation marker ("... [truncated] ..."), part
  of the middle of the file is omitted; report only what is visible in the
  supplied head and tail slices and the analyses — never infer the omitted middle
- Do not include empty arrays, empty objects, null values, or duplicate fields
"""


def build_prompt(
    file_path: str,
    content: str,
    language: str,
    structure: dict,
    dependencies: dict,
    requested_shape: ResolvedShapeBlock | None = None,
) -> tuple[str, str]:
    """Return ``(system, prompt)`` exactly as sent to the provider.

    *content* must already be truncated by the caller.  Dry-run estimation
    calls this with empty *structure* / *dependencies* objects (those agent
    responses do not exist yet), which makes the estimate a lower bound.
    """
    import json as _json

    shape_block = (
        requested_shape.text
        if requested_shape is not None
        else default_shape_block("triple", "documentation")
    )
    prompt = _PROMPT_TEMPLATE.format(
        language=language,
        file_path=file_path,
        structure=_json.dumps(structure, indent=2) if structure else "{}",
        dependencies=_json.dumps(dependencies, indent=2) if dependencies else "{}",
        content=content,
        shape_block=shape_block,
    )
    return _SYSTEM, prompt


class DocumentationAgent(BaseAgent):
    agent_name = "DocumentationAgent"

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
        # This agent also receives pre-computed structure/dependency results
        # passed via content slot as structured context — see orchestrator
        return self._run_with_context(
            file_path, content, imports, language, {}, {}, requested_shape,
            call_context=call_context,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )

    def run_with_context(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        structure: dict,
        dependencies: dict,
        requested_shape: ResolvedShapeBlock | None = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        return self._run_with_context(
            file_path, content, imports, language, structure, dependencies,
            requested_shape,
            call_context=call_context,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )

    def _safe_run_with_context(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        structure: dict,
        dependencies: dict,
        requested_shape: ResolvedShapeBlock | None = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        """Run with upstream agent context and return the standard fallback."""
        try:
            return self.run_with_context(
                file_path,
                content,
                imports,
                language,
                structure,
                dependencies,
                requested_shape,
                call_context=call_context,
                planned_call=planned_call,
                additional_attempt=additional_attempt,
            )
        except CancelledError:
            raise
        except Exception as exc:
            logger.warning("DocumentationAgent failed on %s: %s", file_path, exc)
            return self._agent_error_result(exc)

    def _run_with_context(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        structure: dict,
        dependencies: dict,
        requested_shape: ResolvedShapeBlock | None = None,
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        if call_context is not None:
            requested_shape = call_context.resolved_shape_bundle.selections[
                "documentation"
            ].block
        truncated = self._truncate(content, file_path, call_context=call_context)
        system, prompt = build_prompt(
            file_path,
            truncated,
            language,
            structure,
            dependencies,
            requested_shape,
        )
        shape_block = (
            requested_shape.text
            if requested_shape is not None
            else default_shape_block("triple", "documentation")
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
            agent="documentation",
            file_path=file_path,
            clean_reporter=clean_documentation_report,
            resolved_shape=requested_shape,
            content=truncated,
            imports=imports,
            language=language,
            shape_block=shape_block,
            planned_call=planned_call,
        )

        logger.debug("DocumentationAgent: completed %s", file_path)
        return cleaned
