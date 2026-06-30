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

TRUNCATION_MARKER = "\n... [truncated] ...\n"

# Head/tail split policy.  When a file exceeds the
# character ceiling we keep a larger leading segment (module docstring, imports,
# early definitions) and a bounded trailing segment (late definitions, entry
# points) with the marker between them, instead of a leading prefix only.  The
# fractions apply to the budget left after reserving room for the marker.  A
# ~70/30 head/tail split keeps import/context bias while making late class and
# function definitions visible.
TRUNCATION_HEAD_FRACTION = 0.70


def truncate_for_llm(
    content: str,
    max_chars: int,
    head_fraction: float = TRUNCATION_HEAD_FRACTION,
) -> str:
    """Truncate *content* to a head-plus-tail slice within *max_chars*.

    Content above the ceiling keeps leading and trailing segments joined by
    :data:`TRUNCATION_MARKER`, preserving late definitions and entry points.
    Guarantees:

    - the result — both complete character slices plus the marker — never exceeds
      *max_chars* (the marker is counted inside the ceiling);
    - short content (``len <= max_chars``) is returned byte-for-byte unchanged;
    - slicing is by Unicode code point, so a character is never split;
    - the same bounded source is produced for every caller (orchestrator, each
      agent's defensive fallback, and dry-run estimation), so single and triple
      modes and the dry-run estimate all see exactly the same truncated text.

    The *head_fraction* parameter (default :data:`TRUNCATION_HEAD_FRACTION`
    = 0.70) is configurable via ``truncation_head_ratio`` in the config
    or ``--truncation-head-ratio`` on the CLI.  Callers that omit the argument
    continue to use the 0.70 default and produce byte-identical output.
    """
    if len(content) <= max_chars:
        return content
    marker = TRUNCATION_MARKER
    # Degenerate ceiling smaller than the marker itself: emit a marker prefix so
    # the result still signals truncation without exceeding the ceiling.
    if max_chars <= len(marker):
        return marker[:max_chars]
    budget = max_chars - len(marker)
    head_len = int(budget * head_fraction)
    tail_len = budget - head_len
    head = content[:head_len]
    tail = content[len(content) - tail_len:] if tail_len > 0 else ""
    return head + marker + tail


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
    def run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: "object | None" = None,
    ) -> dict:
        """
        Analyse one file and return a result dict.

        Args:
            file_path: relative path string (for context in the prompt)
            content:   raw file content
            imports:   list of import strings extracted by the parser
            language:  detected language tag
            requested_shape: optional resolved prompt-profile block for this
                file/agent.  ``None`` means the developer-standard shape.

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

    def _safe_run(
        self,
        file_path: str,
        content: str,
        imports: list[str],
        language: str,
        requested_shape: "object | None" = None,
    ) -> dict:
        """
        Wrapper that catches all errors and returns a fallback dict
        instead of crashing the pipeline.
        """
        try:
            if requested_shape is None:
                return self.run(file_path, content, imports, language)
            return self.run(file_path, content, imports, language, requested_shape)
        except AgentError as exc:
            logger.warning("%s failed on %s: %s", self.agent_name, file_path, exc)
            return {"error": str(exc), "agent": self.agent_name}
        except Exception as exc:
            logger.warning("%s unexpected error on %s: %s", self.agent_name, file_path, exc)
            return {"error": str(exc), "agent": self.agent_name}
