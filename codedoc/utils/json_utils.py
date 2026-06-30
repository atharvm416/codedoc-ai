"""Strict JSON text loading with nested duplicate-key rejection.

This is a deliberately dependency-light module: it imports **only** the standard
library so it can be shared by both file parse sites (``loader.load_config`` for
``codedoc.config.json`` and ``prompt_profiles._read_profile_file`` for external
profile files) and by the AI control-response parsers (the customization reviewer
and the routing agent) without creating an import cycle between ``loader.py`` and
``prompt_profiles.py``.

Standard ``json.loads`` keeps the *last* value when an object contains the same
key twice.  For configuration and for security/routing control responses that is
unsafe: a duplicate ``"verdict"``, binding field, or generated ``triple`` member
could silently override an earlier one.  :func:`loads_no_duplicate_keys` installs
an ``object_pairs_hook`` that fails closed on a repeated key at **every** nesting
depth instead.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["DuplicateJSONKeyError", "loads_no_duplicate_keys"]


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object contains the same key more than once.

    Subclasses :class:`ValueError` (which :class:`json.JSONDecodeError` also
    subclasses) so existing ``except (ValueError, json.JSONDecodeError)`` callers
    keep working, while callers that want to preserve a path-specific message can
    catch this type explicitly.  ``key`` is the offending object key.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON key {key!r}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` that rejects any duplicate key in one object.

    Runs for every JSON object at every nesting depth, so a repeated key nested
    arbitrarily deep is still rejected.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(key)
        result[key] = value
    return result


def loads_no_duplicate_keys(text: str) -> Any:
    """Parse *text* as JSON, rejecting duplicate object keys at any depth.

    Behaves exactly like :func:`json.loads` for valid, duplicate-free input.
    Raises :class:`DuplicateJSONKeyError` (a :class:`ValueError`) on a repeated
    key and :class:`json.JSONDecodeError` on malformed JSON.
    """
    return json.loads(text, object_pairs_hook=_reject_duplicate_pairs)
