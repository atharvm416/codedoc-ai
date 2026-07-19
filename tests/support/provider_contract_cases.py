"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

_JSON_HINT = "Respond ONLY with valid JSON"

def _make(provider_cls_name):
    from codedoc.llm import api_provider

    return getattr(api_provider, provider_cls_name)
