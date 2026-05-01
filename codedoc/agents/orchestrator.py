"""
Agent Orchestrator.

Runs StructureAgent, DependencyAgent, and DocumentationAgent
in parallel using threads (safe for I/O-bound LLM calls).
Merges their outputs into a single result dict.
"""

from __future__ import annotations

import concurrent.futures
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

    def __init__(self, llm: LLMProvider, parallel: bool = True) -> None:
        self.llm = llm
        self.parallel = parallel
        self._structure_agent = StructureAgent(llm)
        self._dependency_agent = DependencyAgent(llm)
        self._doc_agent = DocumentationAgent(llm)

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
            Merged dict ready for output.write_outputs()
        """
        file_path = descriptor["rel_path"]
        language = descriptor.get("language", "generic")

        logger.info("Processing: %s", file_path)

        if self.parallel:
            structure, dependencies = self._run_parallel(
                file_path, content, imports, language
            )
        else:
            structure, dependencies = self._run_sequential(
                file_path, content, imports, language
            )

        # DocumentationAgent always gets the other agents' context
        documentation = self._doc_agent._safe_run_with_context(
            file_path, content, imports, language, structure, dependencies
        )

        return self._merge(descriptor, imports, structure, dependencies, documentation)

    # ------------------------------------------------------------------
    # Internal runners
    # ------------------------------------------------------------------

    def _run_parallel(
        self, file_path: str, content: str, imports: list[str], language: str
    ) -> tuple[dict, dict]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_struct = pool.submit(
                self._structure_agent._safe_run, file_path, content, imports, language
            )
            future_dep = pool.submit(
                self._dependency_agent._safe_run, file_path, content, imports, language
            )
            structure = future_struct.result()
            dependencies = future_dep.result()
        return structure, dependencies

    def _run_sequential(
        self, file_path: str, content: str, imports: list[str], language: str
    ) -> tuple[dict, dict]:
        structure = self._structure_agent._safe_run(file_path, content, imports, language)
        dependencies = self._dependency_agent._safe_run(file_path, content, imports, language)
        return structure, dependencies

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
            "dependencies_analysis": dependencies.get("dependencies_analysis", {}),

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