"""
API-based LLM providers.

Supports:
  - OpenAI  (and any OpenAI-compatible endpoint: Together, Groq, etc.)
  - Anthropic Claude
  - Google Gemini
"""

from __future__ import annotations

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import LLMError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI chat completions provider.
    Also works with any OpenAI-compatible API (Together, Groq, LiteLLM proxy).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("OpenAI", "openai package not installed. Run: pip install openai") from exc

        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("OpenAI provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError("OpenAI", str(exc)) from exc

    @property
    def provider_name(self) -> str:
        return f"OpenAI({self._model})"


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.
    """

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise LLMError("Anthropic", "anthropic package not installed. Run: pip install anthropic") from exc

        self._model = model
        self._api_key = api_key
        self._anthropic = _anthropic
        logger.info("Anthropic provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        try:
            client = self._anthropic.Anthropic(api_key=self._api_key)
            kwargs: dict = dict(
                model=self._model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            response = client.messages.create(**kwargs)
            return response.content[0].text or ""
        except Exception as exc:
            raise LLMError("Anthropic", str(exc)) from exc

    @property
    def provider_name(self) -> str:
        return f"Anthropic({self._model})"


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider using the official google-genai SDK.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "Gemini",
                "google-genai package not installed. Run: pip install google-genai",
            ) from exc

        self._model = model
        self._client = genai.Client(api_key=api_key)
        self._types = types
        logger.info("Gemini provider ready - model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        contents = prompt if not system else f"{system}\n\n{prompt}"
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._types.GenerateContentConfig(temperature=temperature),
            )
            return response.text or ""
        except Exception as exc:
            raise LLMError("Gemini", str(exc)) from exc

    def complete_json(self, prompt: str, system: str = "") -> str:
        json_system = (
            (system + "\n\n" if system else "")
            + "Respond ONLY with valid JSON. No markdown fences, no explanation, no preamble."
        )
        contents = f"{json_system}\n\n{prompt}"
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""
        except Exception as exc:
            raise LLMError("Gemini", str(exc)) from exc

    @property
    def provider_name(self) -> str:
        return f"Gemini({self._model})"
