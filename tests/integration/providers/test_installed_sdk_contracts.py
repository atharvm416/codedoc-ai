"""Offline contracts against the actual installed provider SDK interfaces."""

from __future__ import annotations

import pytest

from codedoc.llm.api_provider import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
)


@pytest.mark.parametrize(
    ("factory", "client_surface"),
    (
        (
            lambda: OpenAIProvider("test-key", model="gpt-test"),
            lambda provider: provider._client.chat.completions.create,
        ),
        (
            lambda: AnthropicProvider("test-key", model="claude-test"),
            lambda provider: provider._client.messages.create,
        ),
        (
            lambda: GeminiProvider("test-key", model="gemini-test"),
            lambda provider: provider._client.models.generate_content,
        ),
    ),
)
def test_installed_sdk_constructor_and_request_surface_are_compatible(
    factory, client_surface
) -> None:
    """SDK imports/construction must stay compatible without network calls."""

    provider = factory()
    try:
        assert callable(client_surface(provider))
    finally:
        close = getattr(provider._client, "close", None)
        if callable(close):
            close()
