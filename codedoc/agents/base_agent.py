"""
Abstract base agent.

Every agent (structure, dependency, documentation) inherits this class.
Provides shared prompt building, JSON extraction, and error handling.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from codedoc.core.usage import UsageAccumulator
from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import AgentError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

TRUNCATION_MARKER = "\n... [truncated]"


def truncate_for_llm(content: str, max_chars: int) -> str:
    """Truncate *content* so the result — marker included — never exceeds
    *max_chars*.

    This is the single truncation helper shared by the orchestrator, the
    agents' defensive fallback, and dry-run prompt estimation, so estimated
    source size always matches what real execution sends.
    """
    if len(content) <= max_chars:
        return content
    keep = max(0, max_chars - len(TRUNCATION_MARKER))
    return content[:keep] + TRUNCATION_MARKER[: max_chars - keep]


class BaseAgent(ABC):
    """
    Abstract agent.

    Subclasses implement `run()` which returns a dict of results.
    """

    #: Override in subclass — used in error messages and logs
    agent_name: str = "BaseAgent"

    def __init__(
        self,
        llm: LLMProvider,
        max_content_chars: int = 12000,
        usage: UsageAccumulator | None = None,
    ) -> None:
        self.llm = llm
        self._max_content_chars = max_content_chars
        self._usage = usage

    @abstractmethod
    def run(self, file_path: str, content: str, imports: list[str], language: str) -> dict:
        """
        Analyse one file and return a result dict.

        Args:
            file_path: relative path string (for context in the prompt)
            content:   raw file content
            imports:   list of import strings extracted by the parser
            language:  detected language tag

        Returns:
            dict of results — structure depends on the agent subclass
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _truncate(self, content: str, file_path: str = "") -> str:
        """Defensive truncation fallback for direct agent callers.

        Orchestrated runs truncate once in ``Orchestrator.process()`` before
        the content reaches any agent, so this is a no-op there.  The message
        is DEBUG so a normal orchestrated run never emits three warnings for
        one file.  The marker fits inside the ceiling.
        """
        if len(content) > self._max_content_chars:
            logger.debug(
                "Content truncated: %s (%d chars -> %d chars). "
                "Raise max_content_chars in config to include more content.",
                file_path or "file",
                len(content),
                self._max_content_chars,
            )
            return truncate_for_llm(content, self._max_content_chars)
        return content

    def _call_llm(self, prompt: str, system: str = "") -> str:
        """Call the LLM and return raw text. Wraps errors as AgentError.

        When a :class:`UsageAccumulator` is attached, every provider attempt
        records estimated input tokens immediately before the call, then a
        success (with estimated output tokens) or a failure.  Accounting
        problems never change the outcome of the provider call itself.
        """
        usage = self._usage
        if usage is not None:
            try:
                usage.record_input(system, prompt)
            except Exception:
                logger.debug("Usage accounting failed for input estimate", exc_info=True)
        try:
            raw = self.llm.complete_json(prompt, system=system)
        except Exception as exc:
            if usage is not None:
                try:
                    usage.record_failure()
                except Exception:
                    logger.debug("Usage accounting failed for failed call", exc_info=True)
            raise AgentError(self.agent_name, "unknown", str(exc)) from exc
        if usage is not None:
            try:
                usage.record_success(raw)
            except Exception:
                logger.debug("Usage accounting failed for successful call", exc_info=True)
        return raw

    def _parse_json(self, raw: str, file_path: str) -> dict:
        """
        Extract and parse a JSON object from LLM output.
        Handles models that wrap JSON in markdown fences.
        """
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        # Find the outermost { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise AgentError(
                self.agent_name, file_path,
                f"LLM did not return a JSON object. Raw output: {raw[:200]}"
            )
        json_str = cleaned[start:end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise AgentError(
                self.agent_name, file_path,
                f"JSON parse error: {exc}. Raw snippet: {json_str[:200]}"
            ) from exc

    def _safe_run(self, file_path: str, content: str, imports: list[str], language: str) -> dict:
        """
        Wrapper that catches all errors and returns a fallback dict
        instead of crashing the pipeline.
        """
        try:
            return self.run(file_path, content, imports, language)
        except AgentError as exc:
            logger.warning("%s failed on %s: %s", self.agent_name, file_path, exc)
            return {"error": str(exc), "agent": self.agent_name}
        except Exception as exc:
            logger.warning("%s unexpected error on %s: %s", self.agent_name, file_path, exc)
            return {"error": str(exc), "agent": self.agent_name}
