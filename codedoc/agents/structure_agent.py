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
from codedoc.agents.response_cleaning import clean_structure_response
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
    {{"name": "<function or method defined IN this file>", "description": "<what it does>"}}
  ],
  "classes": [
    {{"name": "<class defined IN this file>", "description": "<what it does>"}}
  ],
  "exports": ["<symbol this module deliberately exposes, including intentional re-exports>"]
}}

Rules:
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
    file_path: str, content: str, imports: list[str], language: str
) -> tuple[str, str]:
    """Return ``(system, prompt)`` exactly as sent to the provider.

    *content* must already be truncated by the caller.  Used by ``run()`` and
    by dry-run usage estimation so estimates match real prompts.
    """
    prompt = _PROMPT_TEMPLATE.format(
        language=language,
        file_path=file_path,
        imports=imports,
        content=content,
    )
    return _SYSTEM, prompt


class StructureAgent(BaseAgent):
    agent_name = "StructureAgent"

    def run(self, file_path: str, content: str, imports: list[str], language: str) -> dict:
        system, prompt = build_prompt(
            file_path, self._truncate(content, file_path), imports, language
        )
        raw = self._call_llm(prompt, system=system)
        result = self._parse_json(raw, file_path)
        cleaned = clean_structure_response(result, file_path)

        logger.debug(
            "StructureAgent: %s → %d functions, %d classes",
            file_path,
            len(cleaned.get("functions", [])),
            len(cleaned.get("classes", [])),
        )
        return cleaned
