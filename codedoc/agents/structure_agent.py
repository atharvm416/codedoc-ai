"""
Structure Agent.

Analyses a file's internal structure:
  - Key functions and their purpose
  - Classes and their responsibility
  - Exported symbols
  - The file's role in the overall system
"""

from __future__ import annotations

from codedoc.agents.base_agent import BaseAgent
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

Return EXACTLY this JSON shape:
{{
  "description": "<one paragraph describing what this file does>",
  "role_in_system": "<how this file fits into the broader system>",
  "functions": [
    {{"name": "<fn name>", "description": "<what it does>"}}
  ],
  "classes": [
    {{"name": "<class name>", "description": "<what it does>"}}
  ],
  "exports": ["<exported symbol>"]
}}

Rules:
- Use only information from the provided code
- Do not hallucinate functions or classes that are not in the code
- Keep descriptions concise (1-2 sentences)
- If functions, classes, or exports are not present, omit that key instead of returning an empty list
- Do not include empty arrays, empty objects, null values, or duplicate fields
"""


class StructureAgent(BaseAgent):
    agent_name = "StructureAgent"

    def run(self, file_path: str, content: str, imports: list[str], language: str) -> dict:
        prompt = _PROMPT_TEMPLATE.format(
            language=language,
            file_path=file_path,
            imports=imports,
            content=self._truncate(content, file_path),
        )
        raw = self._call_llm(prompt, system=_SYSTEM)
        result = self._parse_json(raw, file_path)

        logger.debug(
            "StructureAgent: %s → %d functions, %d classes",
            file_path,
            len(result.get("functions", [])),
            len(result.get("classes", [])),
        )
        return result
