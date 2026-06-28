"""Combined per-file documentation agent (0.10.0 — default ``single`` mode).

This is the default one-call analysis path.  A single provider call produces the
same flat record the legacy three-agent path produced, by asking the model for
one combined JSON object covering structure, dependency analysis, and
documentation.  The deterministic identity/import fields (file path, language,
extension, parser imports) are merged by the :class:`~codedoc.agents.orchestrator.Orchestrator`
and can never be replaced by model output.

The opt-in ``triple`` mode still runs ``StructureAgent`` / ``DependencyAgent`` /
``DocumentationAgent``; 0.10.1 makes them share this module's strict cleaners via
:mod:`codedoc.agents.response_cleaning`.

Factuality boundary: the current parser contract supplies deterministic imports
but no deterministic function/class inventory, so ``functions``, ``classes``,
``exports``, descriptions, concepts, and usage examples produced here are
bounded *model enrichment*, not verified AST facts.  They are cleaned and
capped, never presented as parser-verified.

0.10.1: the response-cleaning primitives, bounds, and ``clean_combined_response``
moved to :mod:`codedoc.agents.response_cleaning` so single and triple modes share
one strict contract.  They are re-exported here for backward compatibility.
"""

from __future__ import annotations

from codedoc.agents.base_agent import BaseAgent
from codedoc.core.prompt_profiles import (
    ResolvedShapeBlock,
    default_shape_block,
    filter_cleaned_response_for_profile,
)

# Re-export the shared bounds + cleaners so existing imports of
# ``file_documentation_agent.MAX_*`` / ``clean_combined_response`` keep working.
from codedoc.agents.response_cleaning import (  # noqa: F401  (re-export)
    MAX_CATALOG_UPDATE_ITEMS,
    MAX_COMBINED_RESPONSE_CHARS,
    MAX_DEPENDENCY_ITEMS_PER_LIST,
    MAX_DEPENDENCY_NAME_CHARS,
    MAX_DEPENDENCY_PURPOSE_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_EXPORT_ITEMS,
    MAX_KEY_CONCEPT_CHARS,
    MAX_KEY_CONCEPT_ITEMS,
    MAX_LIST_ITEM_CHARS,
    MAX_ROLE_CHARS,
    MAX_SYMBOL_DESCRIPTION_CHARS,
    MAX_SYMBOL_ITEMS_PER_KIND,
    MAX_SYMBOL_NAME_CHARS,
    MAX_USAGE_EXAMPLE_CHARS,
    MAX_USAGE_NOTE_ITEMS,
    _clean_dependencies,
    _clean_object_list,
    _clean_scalar,
    _clean_str_list,
    _clean_symbols,
    _enforce_global_cap,
    clean_combined_response,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a senior software engineer and technical writer analysing source "
    "code. You respond ONLY with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """Analyse this {language} file and respond with a single JSON object.

File: {file_path}
Imports found by static parser: {imports}

Code:
{content}

{shape_block}

Rules:
- Use only information from the provided code and parser imports
- functions and classes must be ones DEFINED IN this file — never list an imported
  or re-exported name as a local function or class unless it is also defined here
- exports are names this module deliberately exposes (including intentional
  re-exports in a package initializer); an ordinary imported helper is not an export
- For a package initializer (e.g. __init__.py), describe imported names as
  re-exports, not as locally implemented classes or functions
- Do not hallucinate functions, classes, exports, imports, or relationships not in the code
- Internal imports are relative project paths; external imports are package names
- dependency_refs should list every imported dependency using stable normalized names
- catalog_updates should hold reusable project-level dependency knowledge; usage_notes hold file-specific notes
- Do not invent imports not present in the provided list
- Be specific; avoid generic phrases like 'this file contains utility functions'
- Include usage_example only when it is directly supported by this file's real
  public API and project/package path; omit it (use an empty string) when caller
  context is absent or uncertain
- Never invent placeholder paths or packages such as path/to/file, example.py,
  my_module, your_project, your_package, or your_app
- If the Code above ends with a truncation marker ("... [truncated] ..."), part
  of the middle of the file is omitted; report only what is visible in the
  supplied head and tail slices and the parser imports — never infer the omitted middle
- Omit a key instead of returning an empty list, empty object, or null
- Do not include duplicate fields
"""


def build_prompt(
    file_path: str,
    content: str,
    imports: list[str],
    language: str,
    requested_shape: ResolvedShapeBlock | None = None,
) -> tuple[str, str]:
    """Return ``(system, prompt)`` exactly as sent to the provider.

    *content* must already be truncated by the caller.  Used by ``run()`` and by
    dry-run usage estimation so estimates match real prompts.  In ``single`` mode
    the dry-run estimate is exact because the prompt embeds only known inputs.

    *requested_shape* supplies the requested-shape block.  When ``None`` the
    developer-standard block is used, reproducing the 0.10.3 prompt byte for byte.
    """
    shape_block = (
        requested_shape.text
        if requested_shape is not None
        else default_shape_block("single", "combined")
    )
    prompt = _PROMPT_TEMPLATE.format(
        language=language,
        file_path=file_path,
        imports=imports,
        content=content,
        shape_block=shape_block,
    )
    return _SYSTEM, prompt


class FileDocumentationAgent(BaseAgent):
    """Default ``single``-mode agent: one combined call per file."""

    agent_name = "FileDocumentationAgent"

    def run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: ResolvedShapeBlock | None = None,
    ) -> dict:
        system, prompt = build_prompt(
            file_path, self._truncate(content, file_path), imports, language,
            requested_shape,
        )
        raw = self._call_llm(prompt, system=system)
        result = self._parse_json(raw, file_path)
        cleaned = clean_combined_response(result, file_path)
        cleaned = filter_cleaned_response_for_profile(
            cleaned, requested_shape, mode="single", agent="combined"
        )
        logger.debug(
            "FileDocumentationAgent: %s → %d functions, %d classes",
            file_path,
            len(cleaned.get("functions", [])),
            len(cleaned.get("classes", [])),
        )
        return cleaned
