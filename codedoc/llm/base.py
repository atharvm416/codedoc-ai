"""
Abstract base class for all LLM providers.

Every provider (OpenAI, Anthropic, Ollama, LM Studio, etc.) must
implement this interface. The rest of the system only talks to this
contract — swapping providers requires zero changes outside llm/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract LLM provider.

    Subclasses implement `complete()` and optionally `is_available()`.
    """

    @abstractmethod
    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """
        Send a prompt and return the text response.

        Args:
            prompt:      The user message / task description.
            system:      Optional system-level instruction.
            temperature: Sampling temperature (default low for determinism).

        Returns:
            Raw string response from the model.

        Raises:
            LLMError: on any provider-side failure.
        """

    def complete_json(self, prompt: str, system: str = "") -> str:
        """
        Convenience wrapper: asks the model to respond with valid JSON only.
        Subclasses may override for native JSON mode support.
        """
        json_system = (
            (system + "\n\n" if system else "")
            + "Respond ONLY with valid JSON. No markdown fences, no explanation, no preamble."
        )
        return self.complete(prompt, system=json_system, temperature=0.0)

    def is_available(self) -> bool:
        """
        Optional liveness check.
        Returns True if the provider can currently accept requests.
        Default implementation always returns True.
        """
        return True

    @property
    def provider_name(self) -> str:
        """Human-readable name for logging."""
        return self.__class__.__name__