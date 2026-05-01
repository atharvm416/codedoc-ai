"""
Dependency Agent.

Analyses import relationships:
  - What each import is likely used for
  - Whether imports are internal (project files) or external (packages)
  - Dependency risk signals (circular, unused, missing)
"""

from __future__ import annotations

from codedoc.agents.base_agent import BaseAgent
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

Return EXACTLY this JSON shape:
{{
  "dependencies_analysis": {{
    "internal": ["<relative import paths that are project files>"],
    "external": ["<package names from node_modules / pip / pub / maven>"],
    "usage_notes": [
      {{"import": "<import string>", "used_for": "<brief note on why it is imported>"}}
    ],
    "warnings": ["<any dependency concern, e.g. unused import, potential cycle>"]
  }}
}}

Rules:
- Internal imports start with ./ or ../ (JS/TS) or are relative paths (Python/Dart/Java)
- External imports are package names (e.g. react, lodash, django, flutter/material.dart)
- warnings list may be empty if no concerns are found
- Do not invent imports not present in the provided list
"""


class DependencyAgent(BaseAgent):
    agent_name = "DependencyAgent"

    def run(self, file_path: str, content: str, imports: list[str], language: str) -> dict:
        prompt = _PROMPT_TEMPLATE.format(
            language=language,
            file_path=file_path,
            imports=imports,
            content=self._truncate(content),
        )
        raw = self._call_llm(prompt, system=_SYSTEM)
        result = self._parse_json(raw, file_path)

        dep_analysis = result.get("dependencies_analysis", {})
        logger.debug(
            "DependencyAgent: %s → %d internal, %d external, %d warnings",
            file_path,
            len(dep_analysis.get("internal", [])),
            len(dep_analysis.get("external", [])),
            len(dep_analysis.get("warnings", [])),
        )
        return result