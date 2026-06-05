"""
Agent Orchestrator.

Runs StructureAgent, DependencyAgent, and DocumentationAgent
in parallel using threads (safe for I/O-bound LLM calls).
Merges their outputs into a single result dict.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

from codedoc.agents.structure_agent import StructureAgent
from codedoc.agents.dependency_agent import DependencyAgent
from codedoc.agents.documentation_agent import DocumentationAgent
from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import AgentError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """
    Coordinates multi-agent processing for a single file.

    Parallel mode  (parallel=True, default):
        All 3 agents run concurrently → faster, recommended for API providers.

    Sequential mode (parallel=False):
        Agents run one after another → useful for local LLMs with limited VRAM.
        DocumentationAgent receives structure + dependency results as context.
    """

    def __init__(self, llm: LLMProvider, parallel: bool = True, max_content_chars: int = 12000) -> None:
        self.llm = llm
        self.parallel = parallel
        self._structure_agent = StructureAgent(llm, max_content_chars=max_content_chars)
        self._dependency_agent = DependencyAgent(llm, max_content_chars=max_content_chars)
        self._doc_agent = DocumentationAgent(llm, max_content_chars=max_content_chars)

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

        logger.debug("Running agents for %s with %s", file_path, self.llm.provider_name)

        t_start = time.monotonic()

        if self.parallel:
            structure, dependencies = self._run_parallel(
                file_path, content, imports, language, t_start
            )
        else:
            structure, dependencies = self._run_sequential(
                file_path, content, imports, language, t_start
            )

        # DocumentationAgent always gets the other agents' context
        t_doc_start = time.monotonic()
        documentation = self._doc_agent._safe_run_with_context(
            file_path, content, imports, language, structure, dependencies
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
    # Internal runners
    # ------------------------------------------------------------------

    def _run_parallel(
        self, file_path: str, content: str, imports: list[str], language: str,
        t_start: float | None = None,
    ) -> tuple[dict, dict]:
        if t_start is None:
            t_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_struct = pool.submit(
                self._structure_agent._safe_run, file_path, content, imports, language
            )
            future_dep = pool.submit(
                self._dependency_agent._safe_run, file_path, content, imports, language
            )
            structure = future_struct.result()
            dependencies = future_dep.result()

        elapsed = time.monotonic() - t_start
        self._log_agent_result("structure", file_path, structure, elapsed)
        self._log_agent_result("dependencies", file_path, dependencies, elapsed)
        return structure, dependencies

    def _run_sequential(
        self, file_path: str, content: str, imports: list[str], language: str,
        t_start: float | None = None,
    ) -> tuple[dict, dict]:
        if t_start is None:
            t_start = time.monotonic()
        structure = self._structure_agent._safe_run(file_path, content, imports, language)
        elapsed_struct = time.monotonic() - t_start
        self._log_agent_result("structure", file_path, structure, elapsed_struct)

        t_dep = time.monotonic()
        dependencies = self._dependency_agent._safe_run(file_path, content, imports, language)
        elapsed_dep = time.monotonic() - t_dep
        self._log_agent_result("dependencies", file_path, dependencies, elapsed_dep)
        return structure, dependencies

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


# Patch BaseAgent to support _safe_run_with_context used above
def _safe_run_with_context(
    self,
    file_path: str,
    content: str,
    imports: list[str],
    language: str,
    structure: dict,
    dependencies: dict,
) -> dict:
    try:
        return self.run_with_context(file_path, content, imports, language, structure, dependencies)
    except Exception as exc:
        logger.warning("DocumentationAgent failed on %s: %s", file_path, exc)
        return {"error": str(exc), "agent": "DocumentationAgent"}


DocumentationAgent._safe_run_with_context = _safe_run_with_context
