"""Tests organized by feature ownership."""

from __future__ import annotations

import pytest
from codedoc.core.source_precheck import insufficient_source

@pytest.mark.parametrize(
    "content",
    ["", "   ", "\n\t\r\n", "\ufeff"],
)
def test_empty_or_whitespace_source_is_insufficient(content):
    assert insufficient_source(content) == (True, "empty_or_whitespace_only")

@pytest.mark.parametrize("content", ["x", "\ufffd"])
def test_non_whitespace_source_is_not_insufficient(content):
    assert insufficient_source(content) == (False, "")
