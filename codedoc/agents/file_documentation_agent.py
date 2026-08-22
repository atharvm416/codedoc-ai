"""Combined per-file documentation agent (default ``single`` mode).

This is the default one-call analysis path.  A single provider call produces the
same flat record the legacy three-agent path produced, by asking the model for
one combined JSON object covering structure, dependency analysis, and
documentation.  The deterministic identity/import fields (file path, language,
extension, parser imports) are merged by the :class:`~codedoc.agents.orchestrator.Orchestrator`
and can never be replaced by model output.

The opt-in ``triple`` mode still runs ``StructureAgent`` / ``DependencyAgent`` /
``DocumentationAgent``; both share this module's strict cleaners via
:mod:`codedoc.agents.response_cleaning`.

Factuality boundary: the current parser contract supplies deterministic imports
but no deterministic function/class inventory, so ``functions``, ``classes``,
``exports``, descriptions, concepts, and usage examples produced here are
bounded *model enrichment*, not verified AST facts.  They are cleaned and
capped, never presented as parser-verified.

The response-cleaning primitives, bounds, and ``clean_combined_response``
moved to :mod:`codedoc.agents.response_cleaning` so single and triple modes share
one strict contract.  They are re-exported here for backward compatibility.
"""

from __future__ import annotations

from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES, BaseAgent
from codedoc.core.execution_model import AgentCallContext, PlannedCall, UnitChunkExecutionRequest
from codedoc.core.file_division import (
    MAX_LEAF_DESCRIPTION_CHARS,
    MAX_LEAF_EXPORT_ITEM_CHARS,
    MAX_LEAF_EXPORT_ITEMS,
    MAX_LEAF_SYMBOL_DESCRIPTION_CHARS,
    MAX_LEAF_SYMBOL_ITEMS_PER_KIND,
    MAX_LEAF_SYMBOL_NAME_CHARS,
    MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
    MAX_LEAF_PROMPT_METADATA_CHARS,
    render_leaf_prompt_metadata,
)
from codedoc.core.prompt_profiles import (
    ResolvedShapeBlock,
    default_shape_block,
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
    MAX_EXPORT_ITEM_CHARS,
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
    clean_combined_report,
    clean_combined_response,
    clean_leaf_capsule_report,
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
""" + EXACT_JSON_RESPONSE_RULES + """
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
    developer-standard block is used, reproducing the frozen prompt byte for byte.
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


# ---------------------------------------------------------------------------
# Fixed internal fragment (split leaf chunk) prompt — D5/section 7
# ---------------------------------------------------------------------------
# Completely independent of prompt-profile customization: chunk prompt bytes
# are fixed and versioned by CodeDoc (`LEAF_CAPSULE_SCHEMA_REVISION`), never
# derived from `resolved_synthesis_shape(...)` or any per-file bundle.

LEAF_CAPSULE_REQUESTED_PATHS: tuple[str, ...] = ("description", "functions", "classes", "exports")
LEAF_CAPSULE_REQUIRED_PATHS: tuple[str, ...] = ("description",)

_FRAGMENT_SYSTEM = (
    "You are a senior software engineer analysing one bounded fragment of a "
    "larger source file. You respond ONLY with valid JSON — no markdown, no "
    "explanation."
)

_FRAGMENT_SHAPE_BLOCK = (
    "Return exactly this fixed internal JSON shape (never the final "
    'file-level shape):\n{\n  "description": "<required, non-empty; a '
    'truthful minimal description such as \'Comment-only fragment; no '
    "executable symbols' is valid for a fragment with no executable "
    'code>",\n  "functions": [{"name": "...", "signature": "...", '
    '"description": "..."}, ...], '
    '(optional)\n  "classes": [{"name": "...", "signature": "...", '
    '"description": "..."}, ...], '
    '(optional)\n  "exports": ["...", ...] (optional)\n}\n'
    "Hard bounds: description <= "
    f"{MAX_LEAF_DESCRIPTION_CHARS} characters; functions <= "
    f"{MAX_LEAF_SYMBOL_ITEMS_PER_KIND} items; classes <= "
    f"{MAX_LEAF_SYMBOL_ITEMS_PER_KIND} items; each symbol name <= "
    f"{MAX_LEAF_SYMBOL_NAME_CHARS} characters; each symbol signature <= "
    f"{MAX_LEAF_SYMBOL_SIGNATURE_CHARS} characters; each symbol description <= "
    f"{MAX_LEAF_SYMBOL_DESCRIPTION_CHARS} characters; exports <= "
    f"{MAX_LEAF_EXPORT_ITEMS} items; each export <= "
    f"{MAX_LEAF_EXPORT_ITEM_CHARS} characters."
)

_FRAGMENT_PROMPT_TEMPLATE = """This is one bounded fragment of a larger {language} file. It is NOT the \
whole file.

File: {file_path}
Fragment position: fragment {fragment_index} of {fragment_count} within this \
leaf call group; overall \
fragment {global_index} of {global_count} in this file.
Fragment metadata (ordered JSON; semantic_unit_indexes are 1-based): \
{fragment_metadata}
Continues an earlier fragment of the same unit: {continuation_before}
Continues into a later fragment of the same unit: {continuation_after}
Known symbol name(s) anchored in this fragment: {known_symbols}

Visible fragment source:
{payload}

{shape_block}

Rules:
""" + EXACT_JSON_RESPONSE_RULES + """
- Analyze only the visible fragment source above
- Do not infer this file's overall role, its whole-file dependencies, its \
key concepts, or a usage example — those belong only to final file-level \
synthesis, never to a fragment
- Do not assume, invent, or describe content from an earlier or later \
fragment you cannot see, even when this fragment continues a larger unit
- Do not invent source ranges, IDs, or coverage claims; report only what is \
directly visible
- functions and classes must be ones actually visible and defined in this \
fragment
- include "signature" for every function or method whose language expresses \
one, copied from the visible declaration (parameter types/arity), so that \
overloads sharing a name stay distinguishable; omit it only when the \
language has no signature or the declaration is not visible here
- If this fragment continues into a later fragment, describe only the \
portion visible here
- If the fragment contains only comments, whitespace, or other \
non-executable text, still provide a truthful minimal description and omit \
functions/classes/exports
- Omit a key instead of returning an empty list, empty object, or null
- Do not include duplicate fields
"""


def build_fragment_prompt(request: UnitChunkExecutionRequest) -> tuple[str, str]:
    """Return ``(system, prompt)`` for one fixed internal fragment request.

    *request* already carries every deterministic metadata field the prompt
    needs; this never reopens source, recomputes division/tree topology, or
    accepts a prompt-profile shape.
    """
    known_symbols = ", ".join(request.known_symbols) if request.known_symbols else "(none known)"
    fragment_metadata = render_leaf_prompt_metadata(
        group_unit_id=request.unit_id,
        semantic_units=request.semantic_units,
        unit_indexes=request.unit_indexes,
        unit_count=request.unit_count,
        owning_ranges=request.owning_ranges,
    )
    if len(fragment_metadata) > MAX_LEAF_PROMPT_METADATA_CHARS:
        raise ValueError("fragment prompt metadata exceeds its fixed bound.")
    prompt = _FRAGMENT_PROMPT_TEMPLATE.format(
        language=request.language,
        file_path=request.rel_path,
        fragment_index=request.unit_chunk_index + 1,
        fragment_count=request.unit_chunk_count,
        global_index=request.global_index + 1,
        global_count=request.global_count,
        fragment_metadata=fragment_metadata,
        continuation_before=request.continuation_before,
        continuation_after=request.continuation_after,
        known_symbols=known_symbols,
        payload=request.payload,
        shape_block=_FRAGMENT_SHAPE_BLOCK,
    )
    return _FRAGMENT_SYSTEM, prompt


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
        *,
        call_context: AgentCallContext | None = None,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        if call_context is not None:
            requested_shape = call_context.resolved_shape_bundle.selections[
                "combined"
            ].block
        truncated = self._truncate(content, file_path, call_context=call_context)
        system, prompt = build_prompt(
            file_path, truncated, imports, language, requested_shape,
        )
        shape_block = (
            requested_shape.text
            if requested_shape is not None
            else default_shape_block("single", "combined")
        )
        raw = self._call_llm(
            prompt,
            system=system,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )
        cleaned = self._finalize_response(
            raw,
            mode="single",
            agent="combined",
            file_path=file_path,
            clean_reporter=clean_combined_report,
            resolved_shape=requested_shape,
            content=truncated,
            imports=imports,
            language=language,
            shape_block=shape_block,
            planned_call=planned_call,
        )
        logger.debug(
            "FileDocumentationAgent: %s → %d functions, %d classes",
            file_path,
            len(cleaned.get("functions", [])),
            len(cleaned.get("classes", [])),
        )
        return cleaned

    def run_fragment(
        self,
        request: UnitChunkExecutionRequest,
        *,
        planned_call: PlannedCall | None = None,
        additional_attempt: bool = False,
    ) -> dict:
        """Document one immutable split leaf chunk under the fixed internal
        fragment contract (D5/section 7).

        The fixed fragment prompt is completely independent of any
        prompt-profile customization — a customized final shape never reaches
        this call — and the returned capsule is cleaned under the fixed
        internal schema (`description` required; `functions`/`classes`/
        `exports` optional), never the final synthesis shape.
        """
        system, prompt = build_fragment_prompt(request)
        raw = self._call_llm(
            prompt,
            system=system,
            planned_call=planned_call,
            additional_attempt=additional_attempt,
        )
        cleaned = self._finalize_fixed_response(
            raw,
            label="split-leaf",
            agent="leaf",
            file_path=request.rel_path,
            clean_reporter=clean_leaf_capsule_report,
            requested_paths=LEAF_CAPSULE_REQUESTED_PATHS,
            required_paths=LEAF_CAPSULE_REQUIRED_PATHS,
            content=request.payload,
            shape_block=_FRAGMENT_SHAPE_BLOCK,
            planned_call=planned_call,
        )
        logger.debug(
            "FileDocumentationAgent: %s fragment %d/%d -> %d functions, %d classes",
            request.rel_path,
            request.global_index + 1,
            request.global_count,
            len(cleaned.get("functions", [])),
            len(cleaned.get("classes", [])),
        )
        return cleaned
