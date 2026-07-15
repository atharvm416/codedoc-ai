"""
Dependency Agent.

Analyses import relationships:
  - What each import is likely used for
  - Whether imports are internal (project files) or external (packages)
  - Dependency risk signals (circular, unused, missing)
"""

from __future__ import annotations

from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, BaseAgent
from codedoc.agents.response_cleaning import clean_dependency_report
from codedoc.core.prompt_profiles import (
    ResolvedShapeBlock,
    default_shape_block,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a software architect specialising in dependency analysis. "
    "You respond ONLY with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """Analyse the imports of this {language} file and respond with a JSON object.

File: {file_path}
Imports found by static parser: {imports}

Code:
{content}

{shape_block}

Rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Internal imports start with ./ or ../ (JS/TS) or are relative paths (Python/Dart/Java)
- External imports are package names (e.g. react, lodash, django, flutter/material.dart)
- dependency_refs should list every imported dependency using stable normalized names
- catalog_updates should contain reusable dependency knowledge worth storing once for the whole project
- Add a catalog_updates item when the dependency has a meaningful library/framework/project role, or when this file reveals a better purpose than a generic one
- Do not add catalog_updates for obvious language utilities unless there is a project-specific reason
- usage_notes should describe file-specific or non-obvious usage that should not become global catalog text
- Omit generic repeated usage_notes for common imports such as typing, datetime, pydantic, flutter/material.dart, or provider unless this file uses them in a special way
- Omit warnings if no concerns are found
- Omit dependencies_analysis entirely when there are no internal dependencies, external dependencies, dependency_refs, catalog_updates, usage_notes, or warnings
- Do not include empty arrays, empty objects, null values, or duplicate fields
- Do not invent imports not present in the provided list
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
    """
    shape_block = (
        requested_shape.text
        if requested_shape is not None
        else default_shape_block("triple", "dependency")
    )
    prompt = _PROMPT_TEMPLATE.format(
        language=language,
        file_path=file_path,
        imports=imports,
        content=content,
        shape_block=shape_block,
    )
    return _SYSTEM, prompt


class DependencyAgent(BaseAgent):
    agent_name = "DependencyAgent"

    def run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: ResolvedShapeBlock | None = None,
    ) -> dict:
        truncated = self._truncate(content, file_path)
        system, prompt = build_prompt(
            file_path, truncated, imports, language, requested_shape,
        )
        shape_block = (
            requested_shape.text
            if requested_shape is not None
            else default_shape_block("triple", "dependency")
        )
        raw = self._call_llm(prompt, system=system)
        cleaned = self._finalize_response(
            raw,
            mode="triple",
            agent="dependency",
            file_path=file_path,
            clean_reporter=clean_dependency_report,
            resolved_shape=requested_shape,
            content=truncated,
            imports=imports,
            language=language,
            shape_block=shape_block,
        )

        dep_analysis = cleaned.get("dependencies_analysis", {})
        logger.debug(
            "DependencyAgent: %s → %d internal, %d external, %d warnings",
            file_path,
            len(dep_analysis.get("internal", [])),
            len(dep_analysis.get("external", [])),
            len(dep_analysis.get("warnings", [])),
        )
        return cleaned
