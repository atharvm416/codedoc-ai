"""
Local LLM provider — Ollama and LM Studio.

Both expose an OpenAI-compatible /v1/chat/completions endpoint,
so we use the openai SDK pointed at localhost.
"""

from __future__ import annotations

import requests

from codedoc.llm.base import LLMProvider
from codedoc.utils.errors import LLMError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"


class LocalProvider(LLMProvider):
    """
    Connects to a locally running LLM via OpenAI-compatible API.
    Works with Ollama, LM Studio, llama.cpp server, etc.
    """

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = DEFAULT_OLLAMA_URL,
        api_key: str = "ollama",          # Ollama ignores the key; LM Studio needs any non-empty string
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("Local", "openai package not installed. Run: pip install openai") from exc

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = OpenAI(api_key=api_key, base_url=self._base_url)
        logger.info("Local provider ready — model: %s @ %s", model, base_url)

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
            raise LLMError("Local", f"Local LLM call failed: {exc}") from exc

    def is_available(self) -> bool:
        """Ping the local server to check it's running."""
        try:
            resp = requests.get(f"{self._base_url}/models", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    @property
    def provider_name(self) -> str:
        return f"Local({self._model}@{self._base_url})"