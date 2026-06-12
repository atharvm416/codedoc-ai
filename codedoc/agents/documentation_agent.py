"""
Documentation Agent.

Takes the outputs from StructureAgent and DependencyAgent and
generates the final polished documentation for a file:
  - Enriched description
  - Clear role in system
  - Human-readable summary suitable for the .md output
"""

from __future__ import annotations

from codedoc.agents.base_agent import BaseAgent
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

Return EXACTLY this JSON shape:
{{
  "description": "<clear, detailed paragraph describing what this file does and why it exists>",
  "role_in_system": "<how this file connects to and supports the rest of the codebase>",
  "key_concepts": ["<important concept or pattern used in this file>"],
  "usage_example": "<one-line example of how another file would import or use this file, or empty string>"
}}

Rules:
- Base the documentation only on the provided code and analyses
- Write for a developer who is new to this codebase
- Be specific — avoid generic phrases like 'this file contains utility functions'
- If there are no noteworthy key_concepts, omit key_concepts instead of returning an empty list
- If there is no real usage example from the codebase context, omit usage_example instead of inventing placeholder package names
- Never use placeholder package names such as your_project, your_package, your_app, or your_package_name
- Do not include empty arrays, empty objects, null values, or duplicate fields
"""


def build_prompt(
    file_path: str,
    content: str,
    language: str,
    structure: dict,
    dependencies: dict,
) -> tuple[str, str]:
    """Return ``(system, prompt)`` exactly as sent to the provider.

    *content* must already be truncated by the caller.  Dry-run estimation
    calls this with empty *structure* / *dependencies* objects (those agent
    responses do not exist yet), which makes the estimate a lower bound.
    """
    import json as _json

    prompt = _PROMPT_TEMPLATE.format(
        language=language,
        file_path=file_path,
        structure=_json.dumps(structure, indent=2) if structure else "{}",
        dependencies=_json.dumps(dependencies, indent=2) if dependencies else "{}",
        content=content,
    )
    return _SYSTEM, prompt


class DocumentationAgent(BaseAgent):
    agent_name = "DocumentationAgent"

    def run(self, file_path: str, content: str, imports: list[str], language: str) -> dict:
        # This agent also receives pre-computed structure/dependency results
        # passed via content slot as structured context — see orchestrator
        return self._run_with_context(file_path, content, imports, language, {}, {})

    def run_with_context(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        structure: dict,
        dependencies: dict,
    ) -> dict:
        return self._run_with_context(
            file_path, content, imports, language, structure, dependencies
        )

    def _run_with_context(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        structure: dict,
        dependencies: dict,
    ) -> dict:
        system, prompt = build_prompt(
            file_path,
            self._truncate(content, file_path),
            language,
            structure,
            dependencies,
        )
        raw = self._call_llm(prompt, system=system)
        result = self._parse_json(raw, file_path)

        logger.debug("DocumentationAgent: completed %s", file_path)
        return result
