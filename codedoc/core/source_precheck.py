"""Deterministic source-sufficiency checks."""

from __future__ import annotations


def insufficient_source(content: str) -> tuple[bool, str]:
    """Return whether *content* is empty or whitespace-only."""
    candidate = content[1:] if content.startswith("\ufeff") else content
    if candidate.strip() == "":
        return True, "empty_or_whitespace_only"
    return False, ""
