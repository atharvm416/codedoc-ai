"""Shared strict response cleaning for both analysis modes (0.10.1, Workstream G4).

Single mode (``FileDocumentationAgent``) historically owned the only schema
cleaning; triple mode passed the legacy agents' raw parsed dicts straight
through.  This module promotes those field-level cleaners to one shared place so
both modes enforce identical bounds and factuality shape:

- documented keys and types only; unknown keys dropped;
- trimmed, non-empty strings; booleans rejected as scalar/list/object values;
- malformed symbol / dependency objects removed;
- order-preserving de-duplication;
- the same named count and text bounds;
- no model replacement of deterministic identity fields.

``clean_combined_response`` cleans the single-mode one-call object (with the
global response cap).  ``clean_structure_response`` / ``clean_dependency_response``
/ ``clean_documentation_response`` clean the equivalent triple-mode subresponses
with the same per-field limits.  A malformed subresponse continues through the
established agent-error / retry / usage / recovery paths because the caller wraps
``run`` in ``_safe_run``.
"""

from __future__ import annotations

import json

from codedoc.agents.base_agent import TRUNCATION_MARKER
from codedoc.utils.errors import AgentError

# ---------------------------------------------------------------------------
# Centralized response bounds (single source of truth for prompt + cleaning)
# ---------------------------------------------------------------------------

MAX_COMBINED_RESPONSE_CHARS = 12000
MAX_DESCRIPTION_CHARS = 1200
MAX_ROLE_CHARS = 800
MAX_USAGE_EXAMPLE_CHARS = 2000
MAX_SYMBOL_ITEMS_PER_KIND = 12
MAX_SYMBOL_NAME_CHARS = 128
MAX_SYMBOL_DESCRIPTION_CHARS = 300
MAX_EXPORT_ITEMS = 32
MAX_DEPENDENCY_ITEMS_PER_LIST = 32
MAX_LIST_ITEM_CHARS = 256
MAX_CATALOG_UPDATE_ITEMS = 16
MAX_USAGE_NOTE_ITEMS = 16
MAX_DEPENDENCY_NAME_CHARS = 128
MAX_DEPENDENCY_PURPOSE_CHARS = 400
MAX_KEY_CONCEPT_ITEMS = 16
MAX_KEY_CONCEPT_CHARS = 300


# ---------------------------------------------------------------------------
# Primitive cleaners
# ---------------------------------------------------------------------------

def _canonical(obj: dict) -> str:
    """Canonical compact JSON (sorted keys) used for object de-duplication."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _serialized_len(obj: dict) -> int:
    """Length of the canonical compact-JSON serialization, for cap enforcement."""
    return len(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _clean_scalar(value: object, max_chars: int) -> str | None:
    """Return a trimmed, capped string, or ``None`` if not a usable string.

    Booleans are never accepted as strings; non-strings, empty strings, and
    whitespace-only strings are rejected.  Trimming happens before the length
    cap so the cap measures real content.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:max_chars]


def _clean_str_list(value: object, max_items: int, max_item_chars: int) -> list[str]:
    """Clean a list of strings: trim, drop empties/non-strings, de-dup, cap."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        cleaned = _clean_scalar(item, max_item_chars)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _clean_symbols(value: object) -> list[dict]:
    """Clean a function/class symbol list. ``name`` is required and non-empty;
    ``description`` is optional.  Unknown keys are dropped; objects are de-duped
    by canonical compact JSON; counts are capped."""
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _clean_scalar(item.get("name"), MAX_SYMBOL_NAME_CHARS)
        if name is None:
            continue
        cleaned: dict = {"name": name}
        desc = _clean_scalar(item.get("description"), MAX_SYMBOL_DESCRIPTION_CHARS)
        if desc is not None:
            cleaned["description"] = desc
        key = _canonical(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= MAX_SYMBOL_ITEMS_PER_KIND:
            break
    return out


def _clean_object_list(
    value: object,
    required_fields: tuple[str, ...],
    field_caps: dict[str, int],
    max_items: int,
    type_field: str | None = None,
) -> list[dict]:
    """Clean a list of fixed-schema objects.

    Each object must contain exactly *required_fields* (after trimming, all
    non-empty).  Unknown keys are removed.  When *type_field* is given it must be
    ``"internal"`` or ``"external"``.  Objects are de-duped by canonical compact
    JSON, preserving order, and the count is capped.
    """
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        cleaned: dict = {}
        ok = True
        for field in required_fields:
            val = _clean_scalar(item.get(field), field_caps[field])
            if val is None:
                ok = False
                break
            cleaned[field] = val
        if not ok:
            continue
        if type_field is not None and cleaned.get(type_field) not in ("internal", "external"):
            continue
        key = _canonical(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _clean_dependencies(value: object) -> dict:
    """Clean the ``dependencies_analysis`` sub-object.  Empty sub-lists are
    dropped; an all-empty result returns ``{}`` (removed by the caller)."""
    if not isinstance(value, dict):
        return {}
    deps: dict = {}
    for field in ("internal", "external", "dependency_refs", "warnings"):
        items = _clean_str_list(
            value.get(field), MAX_DEPENDENCY_ITEMS_PER_LIST, MAX_LIST_ITEM_CHARS
        )
        if items:
            deps[field] = items
    catalog = _clean_object_list(
        value.get("catalog_updates"),
        required_fields=("name", "type", "used_for"),
        field_caps={
            "name": MAX_DEPENDENCY_NAME_CHARS,
            "type": MAX_DEPENDENCY_NAME_CHARS,
            "used_for": MAX_DEPENDENCY_PURPOSE_CHARS,
        },
        max_items=MAX_CATALOG_UPDATE_ITEMS,
        type_field="type",
    )
    if catalog:
        deps["catalog_updates"] = catalog
    usage_notes = _clean_object_list(
        value.get("usage_notes"),
        required_fields=("import", "used_for"),
        field_caps={
            "import": MAX_DEPENDENCY_NAME_CHARS,
            "used_for": MAX_DEPENDENCY_PURPOSE_CHARS,
        },
        max_items=MAX_USAGE_NOTE_ITEMS,
    )
    if usage_notes:
        deps["usage_notes"] = usage_notes
    return deps


# Lower-priority-first trim order for the global cap: (parent, field).  A parent
# of ``None`` means a top-level field; otherwise the field lives inside that
# parent sub-dict (``dependencies_analysis``).
_LIST_TRIM_ORDER: tuple[tuple[str | None, str], ...] = (
    ("dependencies_analysis", "warnings"),
    ("dependencies_analysis", "usage_notes"),
    (None, "key_concepts"),
    ("dependencies_analysis", "catalog_updates"),
    ("dependencies_analysis", "dependency_refs"),
    ("dependencies_analysis", "external"),
    ("dependencies_analysis", "internal"),
    (None, "exports"),
    (None, "classes"),
    (None, "functions"),
)

_SCALAR_TRIM_ORDER: tuple[tuple[str, int], ...] = (
    ("usage_example", MAX_USAGE_EXAMPLE_CHARS),
    ("role_in_system", MAX_ROLE_CHARS),
    ("description", MAX_DESCRIPTION_CHARS),
)


def _scalar_candidate(value: str, keep: int, cap: int) -> str:
    """Return *value* truncated to *keep* characters, appending the truncation
    marker only when it fits inside the field *cap*.  Never cuts mid-marker."""
    keep = max(0, keep)
    if keep >= len(value):
        return value
    prefix = value[:keep]
    if keep + len(TRUNCATION_MARKER) <= cap:
        return prefix + TRUNCATION_MARKER
    return prefix


def _enforce_global_cap(cleaned: dict) -> dict:
    """Enforce ``MAX_COMBINED_RESPONSE_CHARS`` against the canonical compact-JSON
    serialization.  Retains complete values only, trimming list items from the
    end in lower-priority-first order, then truncating scalar prose.  Mutates and
    returns *cleaned*."""
    if _serialized_len(cleaned) <= MAX_COMBINED_RESPONSE_CHARS:
        return cleaned

    for parent, field in _LIST_TRIM_ORDER:
        container = cleaned if parent is None else cleaned.get(parent)
        if not isinstance(container, dict):
            continue
        lst = container.get(field)
        if not isinstance(lst, list):
            continue
        while lst and _serialized_len(cleaned) > MAX_COMBINED_RESPONSE_CHARS:
            lst.pop()
        if not lst:
            container.pop(field, None)
            if parent is not None and not container:
                cleaned.pop(parent, None)
        if _serialized_len(cleaned) <= MAX_COMBINED_RESPONSE_CHARS:
            return cleaned

    for field, cap in _SCALAR_TRIM_ORDER:
        if _serialized_len(cleaned) <= MAX_COMBINED_RESPONSE_CHARS:
            return cleaned
        value = cleaned.get(field)
        if not isinstance(value, str):
            continue
        # Largest prefix length that keeps the whole response within the cap.
        lo, hi, best = 0, len(value), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = dict(cleaned)
            trial[field] = _scalar_candidate(value, mid, cap)
            if _serialized_len(trial) <= MAX_COMBINED_RESPONSE_CHARS:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        candidate = _scalar_candidate(value, best, cap)
        if candidate:
            cleaned[field] = candidate
        else:
            cleaned.pop(field, None)

    return cleaned


# ---------------------------------------------------------------------------
# Mode-level cleaners
# ---------------------------------------------------------------------------

def clean_combined_response(raw_obj: object, file_path: str) -> dict:
    """Strictly clean a combined-agent (single-mode) response.

    Raises :class:`AgentError` when *raw_obj* is not an object or retains no
    usable documentation field after cleaning and cap enforcement.
    """
    if not isinstance(raw_obj, dict):
        raise AgentError(
            "FileDocumentationAgent",
            file_path,
            "combined response was not a JSON object",
        )

    cleaned: dict = {}
    description = _clean_scalar(raw_obj.get("description"), MAX_DESCRIPTION_CHARS)
    if description is not None:
        cleaned["description"] = description
    role = _clean_scalar(raw_obj.get("role_in_system"), MAX_ROLE_CHARS)
    if role is not None:
        cleaned["role_in_system"] = role
    functions = _clean_symbols(raw_obj.get("functions"))
    if functions:
        cleaned["functions"] = functions
    classes = _clean_symbols(raw_obj.get("classes"))
    if classes:
        cleaned["classes"] = classes
    exports = _clean_str_list(raw_obj.get("exports"), MAX_EXPORT_ITEMS, MAX_SYMBOL_NAME_CHARS)
    if exports:
        cleaned["exports"] = exports
    dependencies = _clean_dependencies(raw_obj.get("dependencies_analysis"))
    if dependencies:
        cleaned["dependencies_analysis"] = dependencies
    key_concepts = _clean_str_list(
        raw_obj.get("key_concepts"), MAX_KEY_CONCEPT_ITEMS, MAX_KEY_CONCEPT_CHARS
    )
    if key_concepts:
        cleaned["key_concepts"] = key_concepts
    usage_example = _clean_scalar(raw_obj.get("usage_example"), MAX_USAGE_EXAMPLE_CHARS)
    if usage_example is not None:
        cleaned["usage_example"] = usage_example

    cleaned = _enforce_global_cap(cleaned)

    if not cleaned:
        raise AgentError(
            "FileDocumentationAgent",
            file_path,
            "combined response had no usable documentation fields after cleaning",
        )
    return cleaned


def clean_structure_response(raw_obj: object, file_path: str) -> dict:
    """Clean a triple-mode ``StructureAgent`` response to the bounded schema.

    Returns a (possibly partial) dict with only documented keys; the triple-mode
    merge tolerates missing keys.  Raises :class:`AgentError` only when *raw_obj*
    is not an object so a malformed subresponse follows the agent-error path.
    """
    if not isinstance(raw_obj, dict):
        raise AgentError("StructureAgent", file_path, "structure response was not a JSON object")
    cleaned: dict = {}
    description = _clean_scalar(raw_obj.get("description"), MAX_DESCRIPTION_CHARS)
    if description is not None:
        cleaned["description"] = description
    role = _clean_scalar(raw_obj.get("role_in_system"), MAX_ROLE_CHARS)
    if role is not None:
        cleaned["role_in_system"] = role
    functions = _clean_symbols(raw_obj.get("functions"))
    if functions:
        cleaned["functions"] = functions
    classes = _clean_symbols(raw_obj.get("classes"))
    if classes:
        cleaned["classes"] = classes
    exports = _clean_str_list(raw_obj.get("exports"), MAX_EXPORT_ITEMS, MAX_SYMBOL_NAME_CHARS)
    if exports:
        cleaned["exports"] = exports
    return cleaned


def clean_dependency_response(raw_obj: object, file_path: str) -> dict:
    """Clean a triple-mode ``DependencyAgent`` response.

    Preserves the ``{"dependencies_analysis": {...}}`` shape the orchestrator
    merge expects.  Raises :class:`AgentError` only when *raw_obj* is not an
    object.  An empty cleaned analysis yields ``{}`` (no dependency fields), which
    the merge tolerates.
    """
    if not isinstance(raw_obj, dict):
        raise AgentError("DependencyAgent", file_path, "dependency response was not a JSON object")
    analysis = raw_obj.get("dependencies_analysis", raw_obj)
    cleaned_analysis = _clean_dependencies(analysis)
    if cleaned_analysis:
        return {"dependencies_analysis": cleaned_analysis}
    return {}


def clean_documentation_response(raw_obj: object, file_path: str) -> dict:
    """Clean a triple-mode ``DocumentationAgent`` response to the bounded schema.

    Returns a (possibly partial) dict with only documented keys.  Raises
    :class:`AgentError` only when *raw_obj* is not an object.
    """
    if not isinstance(raw_obj, dict):
        raise AgentError(
            "DocumentationAgent", file_path, "documentation response was not a JSON object"
        )
    cleaned: dict = {}
    description = _clean_scalar(raw_obj.get("description"), MAX_DESCRIPTION_CHARS)
    if description is not None:
        cleaned["description"] = description
    role = _clean_scalar(raw_obj.get("role_in_system"), MAX_ROLE_CHARS)
    if role is not None:
        cleaned["role_in_system"] = role
    key_concepts = _clean_str_list(
        raw_obj.get("key_concepts"), MAX_KEY_CONCEPT_ITEMS, MAX_KEY_CONCEPT_CHARS
    )
    if key_concepts:
        cleaned["key_concepts"] = key_concepts
    usage_example = _clean_scalar(raw_obj.get("usage_example"), MAX_USAGE_EXAMPLE_CHARS)
    if usage_example is not None:
        cleaned["usage_example"] = usage_example
    return cleaned
