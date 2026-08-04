"""
API-based LLM providers.

Supports:
  - OpenAI  (and any OpenAI-compatible endpoint: LiteLLM, Together, Groq, etc.)
  - Anthropic Claude
  - Google Gemini

All three providers follow the same contract:
  - Client is instantiated once in __init__ and reused across all calls.
  - complete()      → free-text response at a given temperature.
  - complete_json() → JSON-only response using each provider's native JSON mode
                      where available, with a text-instruction fallback.
  - provider_name   → human-readable string used in logs.
"""

from __future__ import annotations

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import LLMError, bounded_exception_summary
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Instruction appended to every JSON request (used by all providers).
_JSON_INSTRUCTION = (
    "Respond ONLY with valid JSON. No markdown fences, no explanation, no preamble."
)


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """
    OpenAI chat completions provider.

    Also works with any OpenAI-compatible API endpoint — LiteLLM proxy,
    Together.ai, Groq, a custom gateway, etc. — by passing ``base_url``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "OpenAI", "openai package not installed. Run: pip install openai"
            ) from exc

        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("OpenAI provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        messages: list[dict] = []
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
            raise LLMError("OpenAI", bounded_exception_summary(exc)) from exc

    def complete_json(self, prompt: str, system: str = "") -> str:
        """
        Uses OpenAI's native ``response_format={"type": "json_object"}`` mode
        for guaranteed JSON output rather than relying on text instructions alone.
        """
        json_system = (
            (system + "\n\n" if system else "") + _JSON_INSTRUCTION
        )
        messages: list[dict] = [
            {"role": "system", "content": json_system},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError("OpenAI", bounded_exception_summary(exc)) from exc

    @property
    def provider_name(self) -> str:
        return f"OpenAI({self._model})"


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider.

    Uses the ``anthropic`` SDK's Messages API.  The system prompt is passed
    via the top-level ``system`` parameter (not inside the messages list),
    which is the idiomatic Anthropic pattern.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise LLMError(
                "Anthropic",
                "anthropic package not installed. Run: pip install anthropic",
            ) from exc

        self._model = model
        self._client = _anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        try:
            kwargs: dict = dict(
                model=self._model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            return response.content[0].text or ""
        except Exception as exc:
            raise LLMError("Anthropic", bounded_exception_summary(exc)) from exc

    # Anthropic does not expose a dedicated JSON-mode parameter in the
    # standard Messages API, so we rely on the base class text-instruction
    # approach (complete_json from LLMProvider) combined with the JSON
    # extraction logic in BaseAgent._parse_json.

    @property
    def provider_name(self) -> str:
        return f"Anthropic({self._model})"


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    Google Gemini provider using the official ``google-genai`` SDK.

    System instructions are passed via ``GenerateContentConfig.system_instruction``
    (the idiomatic Gemini pattern) rather than being prepended to the user prompt.
    JSON mode uses ``response_mime_type="application/json"`` for native enforcement.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ) -> None:
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
        logger.info("Gemini provider ready — model: %s", model)

    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        try:
            config = self._types.GenerateContentConfig(
                temperature=temperature,
                **({"system_instruction": system} if system else {}),
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            raise LLMError("Gemini", bounded_exception_summary(exc)) from exc

    def complete_json(self, prompt: str, system: str = "") -> str:
        """
        Uses Gemini's native ``response_mime_type="application/json"`` for
        guaranteed JSON output, with the JSON instruction passed as the
        system_instruction rather than concatenated into the user prompt.
        """
        json_system = (
            (system + "\n\n" if system else "") + _JSON_INSTRUCTION
        )
        try:
            config = self._types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=json_system,
                response_mime_type="application/json",
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            raise LLMError("Gemini", bounded_exception_summary(exc)) from exc

    @property
    def provider_name(self) -> str:
        return f"Gemini({self._model})"
