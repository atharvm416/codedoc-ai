"""Shared strict response cleaning for both analysis modes.

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

Each mode-level cleaner has a paired ``*_report`` reporter that runs the exact
same code paths with a bounded removal collector and returns a
:class:`~codedoc.agents.response_diagnostics.CleanResult`; the collector never
changes whether a value is accepted.  The primitive cleaners accept an optional
``collector`` so the reporters reuse them without a second acceptance path.
"""

from __future__ import annotations

import json

from codedoc.agents.base_agent import TRUNCATION_MARKER
from codedoc.agents.response_diagnostics import (
    REMOVAL_DUPLICATE,
    REMOVAL_INVALID_VALUE,
    REMOVAL_ITEM_LIMIT,
    REMOVAL_RESPONSE_CAP,
    REMOVAL_UNKNOWN_FIELD,
    REMOVAL_WRONG_TYPE,
    CleanResult,
    RemovalCollector,
    scalar_removal_reason,
    type_detail,
)
from codedoc.core.file_division import (
    MAX_LEAF_DESCRIPTION_CHARS,
    MAX_LEAF_EXPORT_ITEM_CHARS,
    MAX_LEAF_EXPORT_ITEMS,
    MAX_LEAF_SYMBOL_DESCRIPTION_CHARS,
    MAX_LEAF_SYMBOL_ITEMS_PER_KIND,
    MAX_LEAF_SYMBOL_NAME_CHARS,
    MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
    MAX_REDUCTION_NARRATIVE_CHARS,
)
from codedoc.core.result_assembly import (
    MAX_PUBLIC_EXPORT_ITEM_CHARS,
    MAX_PUBLIC_EXPORT_ITEMS,
    MAX_PUBLIC_SYMBOL_DESCRIPTION_CHARS,
    MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND,
    MAX_PUBLIC_SYMBOL_NAME_CHARS,
)
from codedoc.utils.errors import AgentError

# ---------------------------------------------------------------------------
# Centralized response bounds (single source of truth for prompt + cleaning)
# ---------------------------------------------------------------------------

MAX_COMBINED_RESPONSE_CHARS = 12000
MAX_DESCRIPTION_CHARS = 1200
MAX_ROLE_CHARS = 800
MAX_USAGE_EXAMPLE_CHARS = 2000
MAX_SYMBOL_ITEMS_PER_KIND = MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND
MAX_SYMBOL_NAME_CHARS = MAX_PUBLIC_SYMBOL_NAME_CHARS
MAX_SYMBOL_DESCRIPTION_CHARS = MAX_PUBLIC_SYMBOL_DESCRIPTION_CHARS
MAX_EXPORT_ITEMS = MAX_PUBLIC_EXPORT_ITEMS
MAX_EXPORT_ITEM_CHARS = MAX_PUBLIC_EXPORT_ITEM_CHARS
MAX_DEPENDENCY_ITEMS_PER_LIST = 32
MAX_LIST_ITEM_CHARS = 256
MAX_CATALOG_UPDATE_ITEMS = 16
MAX_USAGE_NOTE_ITEMS = 16
MAX_DEPENDENCY_NAME_CHARS = 128
MAX_DEPENDENCY_PURPOSE_CHARS = 400
MAX_KEY_CONCEPT_ITEMS = 16
MAX_KEY_CONCEPT_CHARS = 300

# Documented top-level keys per mode (used only for ``unknown_field`` reporting;
# acceptance still comes from the field cleaners below).
_COMBINED_KEYS = (
    "description", "role_in_system", "functions", "classes", "exports",
    "dependencies_analysis", "key_concepts", "usage_example",
)
_STRUCTURE_KEYS = ("description", "role_in_system", "functions", "classes", "exports")
_DOCUMENTATION_KEYS = ("description", "role_in_system", "key_concepts", "usage_example")
_DEPENDENCY_MEMBER_KEYS = (
    "internal", "external", "dependency_refs", "warnings",
    "catalog_updates", "usage_notes",
)


# ---------------------------------------------------------------------------
# Removal reporting helper
# ---------------------------------------------------------------------------

def _report(collector, field: str, reason_code: str, detail: str) -> None:
    if collector is not None:
        collector.add(field, reason_code, detail)


def _report_unknown_keys(collector, raw_obj: dict, known: tuple[str, ...]) -> None:
    if collector is None:
        return
    known_set = set(known)
    for key in raw_obj:
        if key not in known_set:
            _report(collector, str(key), REMOVAL_UNKNOWN_FIELD, type_detail(raw_obj[key]))


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


def _clean_str_list(
    value: object,
    max_items: int,
    max_item_chars: int,
    collector=None,
    path: str = "",
) -> list[str]:
    """Clean a list of strings: trim, drop empties/non-strings, de-dup, cap.

    Acceptance is unchanged (first *max_items* trimmed, unique, non-empty
    strings).  When a *collector* is given, every dropped item is reported with
    its stable ``path[index]``.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        cleaned = _clean_scalar(item, max_item_chars)
        if cleaned is None:
            _report(collector, f"{path}[{index}]", scalar_removal_reason(item), type_detail(item))
            continue
        if cleaned in seen:
            _report(collector, f"{path}[{index}]", REMOVAL_DUPLICATE, type_detail(item))
            continue
        if len(out) >= max_items:
            _report(collector, f"{path}[{index}]", REMOVAL_ITEM_LIMIT, f"index {index} over cap {max_items}")
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _clean_symbols(value: object, collector=None, path: str = "") -> list[dict]:
    """Clean a function/class symbol list. ``name`` is required and non-empty;
    ``description`` is optional.  Unknown keys are dropped; objects are de-duped
    by canonical compact JSON; counts are capped."""
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            _report(collector, item_path, REMOVAL_WRONG_TYPE, type_detail(item))
            continue
        for field, field_value in item.items():
            if field not in ("name", "description"):
                _report(
                    collector,
                    f"{item_path}.{field}",
                    REMOVAL_UNKNOWN_FIELD,
                    type_detail(field_value),
                )
        name = _clean_scalar(item.get("name"), MAX_SYMBOL_NAME_CHARS)
        if name is None:
            _report(collector, f"{item_path}.name", scalar_removal_reason(item.get("name")),
                    type_detail(item.get("name")))
            continue
        cleaned: dict = {"name": name}
        desc = _clean_scalar(item.get("description"), MAX_SYMBOL_DESCRIPTION_CHARS)
        if desc is not None:
            cleaned["description"] = desc
        elif "description" in item:
            _report(
                collector,
                f"{item_path}.description",
                scalar_removal_reason(item["description"]),
                type_detail(item["description"]),
            )
        key = _canonical(cleaned)
        if key in seen:
            _report(collector, item_path, REMOVAL_DUPLICATE, type_detail(item))
            continue
        if len(out) >= MAX_SYMBOL_ITEMS_PER_KIND:
            _report(collector, item_path, REMOVAL_ITEM_LIMIT,
                    f"index {index} over cap {MAX_SYMBOL_ITEMS_PER_KIND}")
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _clean_object_list(
    value: object,
    required_fields: tuple[str, ...],
    field_caps: dict[str, int],
    max_items: int,
    type_field: str | None = None,
    collector=None,
    path: str = "",
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
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            _report(collector, item_path, REMOVAL_WRONG_TYPE, type_detail(item))
            continue
        for field, field_value in item.items():
            if field not in required_fields:
                _report(
                    collector,
                    f"{item_path}.{field}",
                    REMOVAL_UNKNOWN_FIELD,
                    type_detail(field_value),
                )
        cleaned: dict = {}
        ok = True
        for field in required_fields:
            val = _clean_scalar(item.get(field), field_caps[field])
            if val is None:
                _report(collector, f"{item_path}.{field}", scalar_removal_reason(item.get(field)),
                        type_detail(item.get(field)))
                ok = False
                continue
            cleaned[field] = val
        if not ok:
            continue
        if type_field is not None and cleaned.get(type_field) not in ("internal", "external"):
            _report(collector, f"{item_path}.{type_field}", REMOVAL_INVALID_VALUE,
                    "not in {internal, external}")
            continue
        key = _canonical(cleaned)
        if key in seen:
            _report(collector, item_path, REMOVAL_DUPLICATE, type_detail(item))
            continue
        if len(out) >= max_items:
            _report(collector, item_path, REMOVAL_ITEM_LIMIT, f"index {index} over cap {max_items}")
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _clean_dependencies(value: object, collector=None) -> dict:
    """Clean the ``dependencies_analysis`` sub-object.  Empty sub-lists are
    dropped; an all-empty result returns ``{}`` (removed by the caller)."""
    if not isinstance(value, dict):
        return {}
    base = "dependencies_analysis"
    if collector is not None:
        for member in value:
            if member not in _DEPENDENCY_MEMBER_KEYS:
                _report(collector, f"{base}.{member}", REMOVAL_UNKNOWN_FIELD,
                        type_detail(value[member]))
    deps: dict = {}
    for field in ("internal", "external", "dependency_refs", "warnings"):
        raw_member = value.get(field)
        if collector is not None and field in value and not isinstance(raw_member, list):
            _report(collector, f"{base}.{field}", REMOVAL_WRONG_TYPE, type_detail(raw_member))
        items = _clean_str_list(
            raw_member, MAX_DEPENDENCY_ITEMS_PER_LIST, MAX_LIST_ITEM_CHARS,
            collector=collector, path=f"{base}.{field}",
        )
        if items:
            deps[field] = items
    raw_catalog = value.get("catalog_updates")
    if collector is not None and "catalog_updates" in value and not isinstance(raw_catalog, list):
        _report(collector, f"{base}.catalog_updates", REMOVAL_WRONG_TYPE, type_detail(raw_catalog))
    catalog = _clean_object_list(
        raw_catalog,
        required_fields=("name", "type", "used_for"),
        field_caps={
            "name": MAX_DEPENDENCY_NAME_CHARS,
            "type": MAX_DEPENDENCY_NAME_CHARS,
            "used_for": MAX_DEPENDENCY_PURPOSE_CHARS,
        },
        max_items=MAX_CATALOG_UPDATE_ITEMS,
        type_field="type",
        collector=collector,
        path=f"{base}.catalog_updates",
    )
    if catalog:
        deps["catalog_updates"] = catalog
    raw_notes = value.get("usage_notes")
    if collector is not None and "usage_notes" in value and not isinstance(raw_notes, list):
        _report(collector, f"{base}.usage_notes", REMOVAL_WRONG_TYPE, type_detail(raw_notes))
    usage_notes = _clean_object_list(
        raw_notes,
        required_fields=("import", "used_for"),
        field_caps={
            "import": MAX_DEPENDENCY_NAME_CHARS,
            "used_for": MAX_DEPENDENCY_PURPOSE_CHARS,
        },
        max_items=MAX_USAGE_NOTE_ITEMS,
        collector=collector,
        path=f"{base}.usage_notes",
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


def _enforce_global_cap(cleaned: dict, collector=None) -> dict:
    """Enforce ``MAX_COMBINED_RESPONSE_CHARS`` against the canonical compact-JSON
    serialization.  Retains complete values only, trimming list items from the
    end in lower-priority-first order, then truncating scalar prose.  Mutates and
    returns *cleaned*.

    When a *collector* is given, complete values removed by the cap (dropped list
    items and fields removed entirely) are reported as ``response_cap``; a
    truncated scalar retains a value and is not reported as a removal.
    """
    if _serialized_len(cleaned) <= MAX_COMBINED_RESPONSE_CHARS:
        return cleaned

    for parent, field in _LIST_TRIM_ORDER:
        container = cleaned if parent is None else cleaned.get(parent)
        if not isinstance(container, dict):
            continue
        lst = container.get(field)
        if not isinstance(lst, list):
            continue
        popped = 0
        while lst and _serialized_len(cleaned) > MAX_COMBINED_RESPONSE_CHARS:
            lst.pop()
            popped += 1
        if popped:
            field_path = field if parent is None else f"{parent}.{field}"
            _report(collector, field_path, REMOVAL_RESPONSE_CAP,
                    f"{popped} item(s) dropped by response cap")
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
            _report(collector, field, REMOVAL_RESPONSE_CAP, "field removed by response cap")

    return cleaned


# ---------------------------------------------------------------------------
# Mode-level field cleaning (shared body for cleaner + reporter)
# ---------------------------------------------------------------------------

def _put_scalar(cleaned: dict, key: str, raw_obj: dict, max_chars: int, collector) -> None:
    value = _clean_scalar(raw_obj.get(key), max_chars)
    if value is not None:
        cleaned[key] = value
    elif collector is not None and key in raw_obj:
        _report(collector, key, scalar_removal_reason(raw_obj[key]), type_detail(raw_obj[key]))


def _put_str_list(
    cleaned: dict, key: str, raw_obj: dict, max_items: int, max_item_chars: int, collector
) -> None:
    raw_value = raw_obj.get(key)
    if collector is not None and key in raw_obj and not isinstance(raw_value, list):
        _report(collector, key, REMOVAL_WRONG_TYPE, type_detail(raw_value))
    items = _clean_str_list(raw_value, max_items, max_item_chars, collector=collector, path=key)
    if items:
        cleaned[key] = items


def _put_symbols(cleaned: dict, key: str, raw_obj: dict, collector) -> None:
    raw_value = raw_obj.get(key)
    if collector is not None and key in raw_obj and not isinstance(raw_value, list):
        _report(collector, key, REMOVAL_WRONG_TYPE, type_detail(raw_value))
    symbols = _clean_symbols(raw_value, collector=collector, path=key)
    if symbols:
        cleaned[key] = symbols


def _put_dependencies(cleaned: dict, raw_obj: dict, collector) -> None:
    raw_value = raw_obj.get("dependencies_analysis")
    if collector is not None and "dependencies_analysis" in raw_obj and not isinstance(raw_value, dict):
        _report(collector, "dependencies_analysis", REMOVAL_WRONG_TYPE, type_detail(raw_value))
    deps = _clean_dependencies(raw_value, collector=collector)
    if deps:
        cleaned["dependencies_analysis"] = deps


def _explicit_empty_list_paths(
    raw_obj: dict,
    fields: tuple[str, ...],
    *,
    parent: str | None = None,
) -> tuple[str, ...]:
    """Return paths explicitly supplied as correctly typed empty lists.

    The historical persisted shape omits empty optional collections.  Reporters
    carry these paths separately so response validation can distinguish a valid
    "there are none" answer from a missing, wrong-typed, or unusable field.
    """
    return tuple(
        f"{parent}.{field}" if parent else field
        for field in fields
        if isinstance(raw_obj.get(field), list) and not raw_obj[field]
    )


def _dependency_empty_paths(raw_obj: dict) -> tuple[str, ...]:
    analysis = raw_obj.get("dependencies_analysis", raw_obj)
    if not isinstance(analysis, dict):
        return ()
    return _explicit_empty_list_paths(
        analysis,
        _DEPENDENCY_MEMBER_KEYS,
        parent="dependencies_analysis",
    )


def _clean_combined_fields(raw_obj: dict, collector=None) -> dict:
    """Shared combined-agent field cleaning used by cleaner and reporter."""
    _report_unknown_keys(collector, raw_obj, _COMBINED_KEYS)
    cleaned: dict = {}
    _put_scalar(cleaned, "description", raw_obj, MAX_DESCRIPTION_CHARS, collector)
    _put_scalar(cleaned, "role_in_system", raw_obj, MAX_ROLE_CHARS, collector)
    _put_symbols(cleaned, "functions", raw_obj, collector)
    _put_symbols(cleaned, "classes", raw_obj, collector)
    _put_str_list(cleaned, "exports", raw_obj, MAX_EXPORT_ITEMS, MAX_EXPORT_ITEM_CHARS, collector)
    _put_dependencies(cleaned, raw_obj, collector)
    _put_str_list(cleaned, "key_concepts", raw_obj, MAX_KEY_CONCEPT_ITEMS, MAX_KEY_CONCEPT_CHARS, collector)
    _put_scalar(cleaned, "usage_example", raw_obj, MAX_USAGE_EXAMPLE_CHARS, collector)
    cleaned = _enforce_global_cap(cleaned, collector)
    return cleaned


def _clean_structure_fields(raw_obj: dict, collector=None) -> dict:
    _report_unknown_keys(collector, raw_obj, _STRUCTURE_KEYS)
    cleaned: dict = {}
    _put_scalar(cleaned, "description", raw_obj, MAX_DESCRIPTION_CHARS, collector)
    _put_scalar(cleaned, "role_in_system", raw_obj, MAX_ROLE_CHARS, collector)
    _put_symbols(cleaned, "functions", raw_obj, collector)
    _put_symbols(cleaned, "classes", raw_obj, collector)
    _put_str_list(cleaned, "exports", raw_obj, MAX_EXPORT_ITEMS, MAX_EXPORT_ITEM_CHARS, collector)
    return cleaned


def _clean_documentation_fields(raw_obj: dict, collector=None) -> dict:
    _report_unknown_keys(collector, raw_obj, _DOCUMENTATION_KEYS)
    cleaned: dict = {}
    _put_scalar(cleaned, "description", raw_obj, MAX_DESCRIPTION_CHARS, collector)
    _put_scalar(cleaned, "role_in_system", raw_obj, MAX_ROLE_CHARS, collector)
    _put_str_list(cleaned, "key_concepts", raw_obj, MAX_KEY_CONCEPT_ITEMS, MAX_KEY_CONCEPT_CHARS, collector)
    _put_scalar(cleaned, "usage_example", raw_obj, MAX_USAGE_EXAMPLE_CHARS, collector)
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
    cleaned = _clean_combined_fields(raw_obj)
    if not cleaned:
        raise AgentError(
            "FileDocumentationAgent",
            file_path,
            "combined response had no usable documentation fields after cleaning",
        )
    return cleaned


def clean_combined_report(raw_obj: object, file_path: str) -> CleanResult:
    """Clean a combined response, returning the value and bounded removals.

    Never raises for an empty or non-object input: :func:`process_response` owns
    the top-level object check and the ``no_usable_fields`` decision.
    """
    collector = RemovalCollector()
    source = raw_obj if isinstance(raw_obj, dict) else {}
    cleaned = _clean_combined_fields(source, collector)
    valid_empty_paths = list(
        _explicit_empty_list_paths(
            source, ("functions", "classes", "exports", "key_concepts")
        )
    )
    valid_empty_paths.extend(_dependency_empty_paths(source))
    return CleanResult(
        value=cleaned,
        removed=collector.finalize(),
        valid_empty_paths=tuple(valid_empty_paths),
    )


def clean_structure_response(raw_obj: object, file_path: str) -> dict:
    """Clean a triple-mode ``StructureAgent`` response to the bounded schema.

    Returns a (possibly partial) dict with only documented keys; the triple-mode
    merge tolerates missing keys.  Raises :class:`AgentError` only when *raw_obj*
    is not an object so a malformed subresponse follows the agent-error path.
    """
    if not isinstance(raw_obj, dict):
        raise AgentError("StructureAgent", file_path, "structure response was not a JSON object")
    return _clean_structure_fields(raw_obj)


def clean_structure_report(raw_obj: object, file_path: str) -> CleanResult:
    collector = RemovalCollector()
    source = raw_obj if isinstance(raw_obj, dict) else {}
    cleaned = _clean_structure_fields(source, collector)
    return CleanResult(
        value=cleaned,
        removed=collector.finalize(),
        valid_empty_paths=_explicit_empty_list_paths(
            source, ("functions", "classes", "exports")
        ),
    )


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


def clean_dependency_report(raw_obj: object, file_path: str) -> CleanResult:
    collector = RemovalCollector()
    source = raw_obj if isinstance(raw_obj, dict) else {}
    analysis = source.get("dependencies_analysis", source)
    cleaned_analysis = _clean_dependencies(analysis, collector=collector)
    value = {"dependencies_analysis": cleaned_analysis} if cleaned_analysis else {}
    return CleanResult(
        value=value,
        removed=collector.finalize(),
        valid_empty_paths=_dependency_empty_paths(source),
    )


def clean_documentation_response(raw_obj: object, file_path: str) -> dict:
    """Clean a triple-mode ``DocumentationAgent`` response to the bounded schema.

    Returns a (possibly partial) dict with only documented keys.  Raises
    :class:`AgentError` only when *raw_obj* is not an object.
    """
    if not isinstance(raw_obj, dict):
        raise AgentError(
            "DocumentationAgent", file_path, "documentation response was not a JSON object"
        )
    return _clean_documentation_fields(raw_obj)


def clean_documentation_report(raw_obj: object, file_path: str) -> CleanResult:
    collector = RemovalCollector()
    source = raw_obj if isinstance(raw_obj, dict) else {}
    cleaned = _clean_documentation_fields(source, collector)
    return CleanResult(
        value=cleaned,
        removed=collector.finalize(),
        valid_empty_paths=_explicit_empty_list_paths(source, ("key_concepts",)),
    )


# ---------------------------------------------------------------------------
# Fixed internal split leaf-capsule / reduction-capsule cleaning (D5/D6/sections 7 and 9)
# ---------------------------------------------------------------------------
#
# These schemas are fixed, developer-owned, and never derived from a
# prompt-profile registry: split internal work is not user-configurable.  A
# leaf capsule's only required field is ``description``; a reduction capsule
# carries a single refined ``narrative`` field. Symbol/export leaf bounds align
# with the corresponding whole-file fields; the leaf description and reduction
# narrative share the exact child-narrative ceiling. Reducer fan-in is sized
# from the rendered narratives, independently of leaf capsule JSON size (see
# ``codedoc.core.file_division``).

_LEAF_CAPSULE_KEYS = ("description", "functions", "classes", "exports")
_REDUCTION_CAPSULE_KEYS = ("narrative",)


def _clean_fixed_scalar(
    value: object,
    max_chars: int,
    *,
    collector=None,
    path: str,
) -> str | None:
    """Clean a fixed-capsule scalar without silently truncating facts."""
    if isinstance(value, bool) or not isinstance(value, str):
        _report(
            collector,
            path,
            scalar_removal_reason(value),
            type_detail(value),
        )
        return None
    trimmed = value.strip()
    if not trimmed:
        _report(
            collector,
            path,
            scalar_removal_reason(value),
            type_detail(value),
        )
        return None
    if len(trimmed) > max_chars:
        _report(
            collector,
            path,
            REMOVAL_RESPONSE_CAP,
            f"{len(trimmed)} chars over cap {max_chars}",
        )
        return None
    return trimmed


def _clean_fixed_str_list(
    value: object,
    max_items: int,
    max_item_chars: int,
    *,
    collector=None,
    path: str,
) -> list[str]:
    """Clean a fixed-capsule string list without truncating any item."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        cleaned = _clean_fixed_scalar(
            item,
            max_item_chars,
            collector=collector,
            path=item_path,
        )
        if cleaned is None:
            continue
        if cleaned in seen:
            _report(
                collector,
                item_path,
                REMOVAL_DUPLICATE,
                type_detail(item),
            )
            continue
        if len(out) >= max_items:
            _report(
                collector,
                item_path,
                REMOVAL_ITEM_LIMIT,
                f"index {index} over cap {max_items}",
            )
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _clean_leaf_symbols(value: object, collector=None, path: str = "") -> list[dict]:
    """Clean a leaf-capsule function/class list under its fixed bounds."""
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            _report(collector, item_path, REMOVAL_WRONG_TYPE, type_detail(item))
            continue
        for field, field_value in item.items():
            if field not in ("name", "description", "signature"):
                _report(collector, f"{item_path}.{field}", REMOVAL_UNKNOWN_FIELD, type_detail(field_value))
        name = _clean_fixed_scalar(
            item.get("name"),
            MAX_LEAF_SYMBOL_NAME_CHARS,
            collector=collector,
            path=f"{item_path}.name",
        )
        if name is None:
            continue
        cleaned: dict = {"name": name}
        if "description" in item:
            desc = _clean_fixed_scalar(
                item["description"],
                MAX_LEAF_SYMBOL_DESCRIPTION_CHARS,
                collector=collector,
                path=f"{item_path}.description",
            )
            if desc is not None:
                cleaned["description"] = desc
        # Optional, but load-bearing: the ledger's semantic key uses the
        # signature to keep same-named overloads distinct (D7/section 8). Dropping it
        # here would silently collapse `run(int)` and `run(str)` into one fact.
        if "signature" in item:
            signature = _clean_fixed_scalar(
                item["signature"],
                MAX_LEAF_SYMBOL_SIGNATURE_CHARS,
                collector=collector,
                path=f"{item_path}.signature",
            )
            if signature is not None:
                cleaned["signature"] = signature
        if len(out) >= MAX_LEAF_SYMBOL_ITEMS_PER_KIND:
            _report(collector, item_path, REMOVAL_ITEM_LIMIT,
                    f"index {index} over cap {MAX_LEAF_SYMBOL_ITEMS_PER_KIND}")
            continue
        out.append(cleaned)
    return out


def _clean_leaf_capsule_fields(raw_obj: dict, collector=None) -> dict:
    _report_unknown_keys(collector, raw_obj, _LEAF_CAPSULE_KEYS)
    cleaned: dict = {}
    if "description" in raw_obj:
        description = _clean_fixed_scalar(
            raw_obj["description"],
            MAX_LEAF_DESCRIPTION_CHARS,
            collector=collector,
            path="description",
        )
        if description is not None:
            cleaned["description"] = description
    for key in ("functions", "classes"):
        raw_value = raw_obj.get(key)
        if collector is not None and key in raw_obj and not isinstance(raw_value, list):
            _report(collector, key, REMOVAL_WRONG_TYPE, type_detail(raw_value))
        symbols = _clean_leaf_symbols(raw_value, collector=collector, path=key)
        if symbols:
            cleaned[key] = symbols
    raw_exports = raw_obj.get("exports")
    if (
        collector is not None
        and "exports" in raw_obj
        and not isinstance(raw_exports, list)
    ):
        _report(
            collector,
            "exports",
            REMOVAL_WRONG_TYPE,
            type_detail(raw_exports),
        )
    exports = _clean_fixed_str_list(
        raw_exports,
        MAX_LEAF_EXPORT_ITEMS,
        MAX_LEAF_EXPORT_ITEM_CHARS,
        collector=collector,
        path="exports",
    )
    if exports:
        cleaned["exports"] = exports
    return cleaned


def clean_leaf_capsule_report(raw_obj: object, file_path: str) -> CleanResult:
    """Clean one fixed internal leaf-fragment capsule.

    Never raises for an empty or non-object input: the shared response path
    owns the top-level-object and required-field decisions.  Unlike the
    whole-file cleaners, this never includes ``role_in_system``,
    ``dependencies_analysis``, ``key_concepts``, or ``usage_example`` — those
    are whole-file inferences a fragment must never make (D5).
    """
    collector = RemovalCollector()
    source = raw_obj if isinstance(raw_obj, dict) else {}
    cleaned = _clean_leaf_capsule_fields(source, collector)
    valid_empty_paths = _explicit_empty_list_paths(source, ("functions", "classes", "exports"))
    return CleanResult(
        value=cleaned,
        removed=collector.finalize(),
        valid_empty_paths=valid_empty_paths,
        removal_reason_codes=collector.reason_codes(),
    )


def _clean_reduction_capsule_fields(raw_obj: dict, collector=None) -> dict:
    _report_unknown_keys(collector, raw_obj, _REDUCTION_CAPSULE_KEYS)
    cleaned: dict = {}
    if "narrative" in raw_obj:
        narrative = _clean_fixed_scalar(
            raw_obj["narrative"],
            MAX_REDUCTION_NARRATIVE_CHARS,
            collector=collector,
            path="narrative",
        )
        if narrative is not None:
            cleaned["narrative"] = narrative
    return cleaned


def clean_reduction_capsule_report(raw_obj: object, file_path: str) -> CleanResult:
    """Clean one fixed internal reduction (narrative-only) capsule.

    A reduction call refines prose only (section 9): it never carries — and can
    never reintroduce — functions/classes/exports; those live solely in the
    locally maintained fact ledger.
    """
    collector = RemovalCollector()
    source = raw_obj if isinstance(raw_obj, dict) else {}
    cleaned = _clean_reduction_capsule_fields(source, collector)
    return CleanResult(
        value=cleaned,
        removed=collector.finalize(),
        valid_empty_paths=(),
        removal_reason_codes=collector.reason_codes(),
    )
