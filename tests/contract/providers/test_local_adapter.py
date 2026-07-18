"""Contract for the reserved LocalProvider implementation, which is intentionally excluded from the public factory and CLI."""

from __future__ import annotations

import urllib.error

def test_dormant_local_provider_liveness_uses_standard_library(monkeypatch):
    from codedoc.llm.local_provider import LocalProvider

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    provider = object.__new__(LocalProvider)
    provider._base_url = "http://localhost:11434/v1"
    monkeypatch.setattr(
        "codedoc.llm.local_provider.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(),
    )
    assert provider.is_available() is True

    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(
        "codedoc.llm.local_provider.urllib.request.urlopen",
        unavailable,
    )
    assert provider.is_available() is False
