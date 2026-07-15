"""
Agent Orchestrator.

Coordinates per-file analysis in one of two selectable modes:

- ``single`` (default): one combined :class:`FileDocumentationAgent` call per
  file (one provider call), then merges the cleaned response with the
  deterministic identity/import fields.
- ``triple``: the legacy path running StructureAgent, DependencyAgent, and
  DocumentationAgent (three provider calls), merging their outputs.

Both modes return the identical flat record consumed by SafeWriter and
``project_view.py``.
"""

from __future__ import annotations

import concurrent.futures
import time

from codedoc.agents.base_agent import truncate_for_llm
from codedoc.agents.file_documentation_agent import FileDocumentationAgent
from codedoc.agents.structure_agent import StructureAgent
from codedoc.agents.dependency_agent import DependencyAgent
from codedoc.agents.documentation_agent import DocumentationAgent
from codedoc.agents.response_correction_agent import ResponseCorrectionAgent
from codedoc.agents.response_diagnostics import CorrectionLedger
from codedoc.core.record_meta import (
    expected_analysis_identity,
    expected_max_context_revision,
)
from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST, ResolvedProfile
from codedoc.core.usage import UsageAccumulator
from codedoc.llm.base import LLMProvider
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

VALID_ANALYSIS_MODES = ("single", "triple")


def initial_calls_per_file(analysis_mode: str) -> int:
    """Return the number of provider calls one file makes on its initial attempt.

    Canonical helper used by both mode statistics and the planning multiplier so
    the per-file call count is defined in exactly one place.
    """
    return 3 if analysis_mode == "triple" else 1


def _result_has_agent_error(result: dict) -> bool:
    """Whether a merged result carries an agent error on any sub-result.

    Mirrors the execution-layer ``_agent_errors`` check without importing it
    (execution imports this module, so the dependency must not be reversed).
    """
    for key in ("structure", "dependencies_analysis", "documentation"):
        value = result.get(key)
        if isinstance(value, dict) and value.get("error"):
            return True
    return False


class Orchestrator:
    """
    Coordinates multi-agent processing for a single file.

    Parallel mode  (parallel=True, default):
        All 3 agents run concurrently → faster, recommended for API providers.

    Sequential mode (parallel=False):
        Agents run one after another → useful for local LLMs with limited VRAM.
        DocumentationAgent receives structure + dependency results as context.
    """

    def __init__(
        self,
        llm: LLMProvider,
        parallel: bool = True,
        max_content_chars: int = 12000,
        usage: UsageAccumulator | None = None,
        analysis_mode: str = "single",
        truncation_head_ratio: float = 0.70,
        resolved_profile: ResolvedProfile | None = None,
        response_correction_enabled: bool = False,
        correction_ledger: CorrectionLedger | None = None,
    ) -> None:
        if analysis_mode not in VALID_ANALYSIS_MODES:
            raise ValueError(
                f"analysis_mode must be one of {VALID_ANALYSIS_MODES}; got "
                f"{analysis_mode!r}."
            )
        self.llm = llm
        self.parallel = parallel
        self.max_content_chars = max_content_chars
        self.analysis_mode = analysis_mode
        self.truncation_head_ratio = truncation_head_ratio
        self.resolved_profile = resolved_profile
        # The combined agent powers the default single-call path.
        self._file_agent = FileDocumentationAgent(
            llm, max_content_chars=max_content_chars, usage=usage
        )
        # The three legacy agents remain instantiated and are used by triple mode.
        self._structure_agent = StructureAgent(llm, max_content_chars=max_content_chars, usage=usage)
        self._dependency_agent = DependencyAgent(llm, max_content_chars=max_content_chars, usage=usage)
        self._doc_agent = DocumentationAgent(llm, max_content_chars=max_content_chars, usage=usage)
        # Build exactly one shared correction component and inject it into all four
        # agents.  Only when a run-level ledger is supplied; a directly constructed
        # orchestrator (no ledger) leaves every agent's ``_correction`` as ``None``,
        # so an eligible response-contract failure is final with no correction call.
        if correction_ledger is not None:
            correction = ResponseCorrectionAgent(
                llm, usage, correction_ledger, response_correction_enabled
            )
            for agent in (
                self._file_agent,
                self._structure_agent,
                self._dependency_agent,
                self._doc_agent,
            ):
                agent._correction = correction

    def process(
        self,
        descriptor: dict,
        content: str,
        imports: list[str],
    ) -> dict:
        """
        Run all agents on one file and return the merged result.

        Args:
            descriptor: file descriptor from scanner
            content:    raw file content
            imports:    import strings from parser

        Returns:
            Merged dict with structure and dependency analysis for this file.
        """
        file_path = descriptor["rel_path"]
        language = descriptor.get("language", "generic")

        # Truncate once here so all three agents receive the exact same
        # string and the marker stays inside the configured ceiling.  One
        # WARNING per processed file — the agents' own fallback is DEBUG.
        original_chars = len(content)
        if original_chars > self.max_content_chars:
            content = truncate_for_llm(
                content,
                self.max_content_chars,
                head_fraction=self.truncation_head_ratio,
            )
            logger.warning(
                "Content truncated: %s (%d chars -> %d chars sent; configured "
                "ceiling max_content_chars=%d). Raise max_content_chars in "
                "config to include more content.",
                file_path,
                original_chars,
                len(content),
                self.max_content_chars,
            )

        logger.debug(
            "Running %s-mode analysis for %s with %s",
            self.analysis_mode,
            file_path,
            self.llm.provider_name,
        )

        # Resolve one bundle per file (extension scope) and thread it through every
        # agent prompt and the stamped record digest.  ``None`` when no profile is
        # active, exactly reproducing the developer-standard prompts.
        bundle = (
            self.resolved_profile.resolve_bundle(
                self.resolved_profile.scope_for(descriptor)
            )
            if self.resolved_profile is not None
            else None
        )

        if self.analysis_mode == "triple":
            result = self._process_triple(descriptor, content, imports, language, bundle)
        else:
            result = self._process_single(descriptor, content, imports, language, bundle)

        # Attach cache-identity keys to every successful flat result before it is
        # recorded.  Failed results (an error on structure/dependencies/
        # documentation) are never recorded, so they must not carry the identity.
        if result.get("state") == "checked" and not _result_has_agent_error(result):
            result.update(expected_analysis_identity(self.analysis_mode))
            # Stamp the truncation identity only for a file actually large
            # enough to be truncated (omit it otherwise so files that fit the
            # ceiling stay reusable across ceiling/ratio changes).  ``original_chars``
            # is the pre-truncation source length.
            mcr = expected_max_context_revision(
                original_chars,
                max_chars=self.max_content_chars,
                head_ratio=self.truncation_head_ratio,
            )
            if mcr is not None:
                result["_max_context_revision"] = mcr
            if bundle is not None and bundle.digest != NO_PROMPT_PROFILE_DIGEST:
                result["_prompt_profile_digest"] = bundle.digest
        return result

    # ------------------------------------------------------------------
    # single mode (default)
    # ------------------------------------------------------------------

    def _process_single(
        self, descriptor: dict, content: str, imports: list[str], language: str,
        bundle=None,
    ) -> dict:
        """One combined provider call, merged into the flat record."""
        file_path = descriptor["rel_path"]
        t_start = time.monotonic()
        shape = self._profile_block(bundle, "combined")
        cleaned = self._call_agent(
            self._file_agent, file_path, content, imports, language, shape
        )
        elapsed = time.monotonic() - t_start

        if isinstance(cleaned, dict) and cleaned.get("error"):
            logger.warning(
                "[FILE] %s | combined fallback: %s",
                file_path,
                cleaned.get("error", "unknown"),
            )
            return self._merge_single_failure(descriptor, imports, cleaned)

        logger.info("[FILE] %s | combined ok  %.1fs", file_path, elapsed)
        return self._merge_single(descriptor, imports, cleaned)

    def _merge_single(self, descriptor: dict, imports: list[str], cleaned: dict) -> dict:
        """Merge a cleaned combined response with deterministic identity fields."""
        description = cleaned.get("description", "")
        role = cleaned.get("role_in_system", "")
        functions = cleaned.get("functions", [])
        classes = cleaned.get("classes", [])
        exports = cleaned.get("exports", [])
        dependencies_analysis = cleaned.get("dependencies_analysis", {})
        key_concepts = cleaned.get("key_concepts", [])
        usage_example = cleaned.get("usage_example", "")

        structure_view = {
            "description": description,
            "role_in_system": role,
            "functions": functions,
            "classes": classes,
            "exports": exports,
        }
        documentation_view = {
            "description": description,
            "role_in_system": role,
            "key_concepts": key_concepts,
            "usage_example": usage_example,
        }
        return {
            # Identity / parser (deterministic — never from model output)
            "file_path": descriptor["rel_path"],
            "language": descriptor.get("language", ""),
            "extension": descriptor.get("extension", ""),
            "imports": imports,
            # Combined model enrichment
            "description": description,
            "role_in_system": role,
            "functions": functions,
            "classes": classes,
            "exports": exports,
            "structure": structure_view,
            "dependencies_analysis": dependencies_analysis,
            "key_concepts": key_concepts,
            "usage_example": usage_example,
            "documentation": documentation_view,
            "state": "checked",
        }

    def _merge_single_failure(
        self, descriptor: dict, imports: list[str], failure: dict
    ) -> dict:
        """Flat record for a failed combined call: identity preserved, error on
        the ``documentation`` key so ``_agent_errors()`` detects one failure.

        The response-contract marker fields (``response_contract_final``,
        ``response_contract_diagnostic``, ``response_contract_correction_attempted``)
        are copied onto the same sub-result so the execution layer can classify the
        failure as a non-retryable response-contract failure.
        """
        documentation: dict = {
            "error": failure.get("error", "unknown"),
            "agent": failure.get("agent", "FileDocumentationAgent"),
        }
        for marker_key in (
            "response_contract_final",
            "response_contract_diagnostic",
            "response_contract_correction_attempted",
        ):
            if marker_key in failure:
                documentation[marker_key] = failure[marker_key]
        return {
            "file_path": descriptor["rel_path"],
            "language": descriptor.get("language", ""),
            "extension": descriptor.get("extension", ""),
            "imports": imports,
            "description": "",
            "role_in_system": "",
            "functions": [],
            "classes": [],
            "exports": [],
            "structure": {},
            "dependencies_analysis": {},
            "key_concepts": [],
            "usage_example": "",
            "documentation": documentation,
            "state": "checked",
        }

    # ------------------------------------------------------------------
    # triple mode (opt-in legacy path)
    # ------------------------------------------------------------------

    def _process_triple(
        self, descriptor: dict, content: str, imports: list[str], language: str,
        bundle=None,
    ) -> dict:
        file_path = descriptor["rel_path"]
        t_start = time.monotonic()

        if self.parallel:
            structure, dependencies = self._run_parallel(
                file_path, content, imports, language, bundle, t_start
            )
        else:
            structure, dependencies = self._run_sequential(
                file_path, content, imports, language, bundle, t_start
            )

        # DocumentationAgent always gets the other agents' context
        t_doc_start = time.monotonic()
        doc_shape = self._profile_block(bundle, "documentation")
        if doc_shape is None:
            documentation = self._doc_agent._safe_run_with_context(
                file_path, content, imports, language, structure, dependencies
            )
        else:
            documentation = self._doc_agent._safe_run_with_context(
                file_path, content, imports, language, structure, dependencies,
                doc_shape,
            )
        elapsed_doc = time.monotonic() - t_doc_start
        if isinstance(documentation, dict) and documentation.get("error"):
            logger.warning(
                "[FILE] %s | documentation fallback: %s",
                file_path,
                documentation.get("error", "unknown"),
            )
        else:
            logger.info("[FILE] %s | documentation ok  %.1fs", file_path, elapsed_doc)

        return self._merge(descriptor, imports, structure, dependencies, documentation)

    # ------------------------------------------------------------------
    # Internal runners (triple mode)
    # ------------------------------------------------------------------

    def _run_parallel(
        self, file_path: str, content: str, imports: list[str], language: str,
        bundle=None, t_start: float | None = None,
    ) -> tuple[dict, dict]:
        if t_start is None:
            t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_struct = pool.submit(
                self._call_agent, self._structure_agent,
                file_path, content, imports, language,
                self._profile_block(bundle, "structure"),
            )
            future_dep = pool.submit(
                self._call_agent, self._dependency_agent,
                file_path, content, imports, language,
                self._profile_block(bundle, "dependency"),
            )
            structure = future_struct.result()
            dependencies = future_dep.result()

        elapsed = time.monotonic() - t_start
        self._log_agent_result("structure", file_path, structure, elapsed)
        self._log_agent_result("dependencies", file_path, dependencies, elapsed)
        return structure, dependencies

    def _run_sequential(
        self, file_path: str, content: str, imports: list[str], language: str,
        bundle=None, t_start: float | None = None,
    ) -> tuple[dict, dict]:
        if t_start is None:
            t_start = time.monotonic()
        structure = self._call_agent(
            self._structure_agent, file_path, content, imports, language,
            self._profile_block(bundle, "structure"),
        )
        elapsed_struct = time.monotonic() - t_start
        self._log_agent_result("structure", file_path, structure, elapsed_struct)

        t_dep = time.monotonic()
        dependencies = self._call_agent(
            self._dependency_agent, file_path, content, imports, language,
            self._profile_block(bundle, "dependency"),
        )
        elapsed_dep = time.monotonic() - t_dep
        self._log_agent_result("dependencies", file_path, dependencies, elapsed_dep)
        return structure, dependencies

    @staticmethod
    def _profile_block(bundle, agent: str):
        """Return the resolved shape block for *agent* from the file's bundle."""
        if bundle is None:
            return None
        return bundle.selections[agent].block

    @staticmethod
    def _call_agent(agent, file_path, content, imports, language, shape):
        if shape is None:
            return agent._safe_run(file_path, content, imports, language)
        return agent._safe_run(file_path, content, imports, language, shape)

    def _log_agent_result(self, agent_label: str, file_path: str, result: dict, elapsed: float) -> None:
        if isinstance(result, dict) and result.get("error"):
            logger.warning(
                "[FILE] %s | %s fallback: %s",
                file_path,
                agent_label,
                result.get("error", "unknown"),
            )
        else:
            logger.info("[FILE] %s | %s ok  %.1fs", file_path, agent_label, elapsed)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(
        self,
        descriptor: dict,
        imports: list[str],
        structure: dict,
        dependencies: dict,
        documentation: dict,
    ) -> dict:
        """Combine all agent outputs into one flat result dict."""
        return {
            # Identity
            "file_path": descriptor["rel_path"],
            "language": descriptor.get("language", ""),
            "extension": descriptor.get("extension", ""),

            # From parser (deterministic)
            "imports": imports,

            # From StructureAgent
            "description": (
                documentation.get("description")
                or structure.get("description", "")
            ),
            "role_in_system": (
                documentation.get("role_in_system")
                or structure.get("role_in_system", "")
            ),
            "functions": structure.get("functions", []),
            "classes": structure.get("classes", []),
            "exports": structure.get("exports", []),
            "structure": structure,

            # From DependencyAgent
            "dependencies_analysis": dependencies.get("dependencies_analysis", dependencies),

            # From DocumentationAgent
            "key_concepts": documentation.get("key_concepts", []),
            "usage_example": documentation.get("usage_example", ""),
            "documentation": documentation,

            # Status
            "state": "checked",
        }
