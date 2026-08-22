"""Structured, bounded response diagnostics and the canonical response path.

This module is the single place that turns a raw provider response into either a
cleaned, profile-filtered, required-field-validated dict or a typed
:class:`~codedoc.utils.errors.ResponseContractError` carrying a bounded
:class:`ResponseDiagnostic`.  It owns:

- the immutable diagnostic models (:class:`RemovedField`, :class:`ResponseDiagnostic`)
  and the two closed reason-code sets;
- the diagnostic bound constants and the shared ``MAX_CORRECTION_RESPONSE_CHARS``;
- the one Markdown-fence / brace JSON-candidate extraction used by both
  :func:`process_response` and ``BaseAgent._parse_json``;
- :func:`process_response`, the canonical initial-and-correction response path;
- the canonical registry-required-field validation; and
- the thread-safe :class:`CorrectionLedger`.

Every diagnostic collection is bounded before it is attached to an exception or
logged.  No raw provider response, source, prompt, or credential text is ever
placed in a diagnostic, a log line, or the ledger.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

from codedoc.core.prompt_profiles import (
    ResolvedShapeBlock,
    default_field_paths,
    filter_cleaned_response_for_profile,
    iter_fields,
)
from codedoc.utils.errors import ResponseContractError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Bounds (single source of truth for every diagnostic collection)
# ---------------------------------------------------------------------------

#: Maximum expected / returned / retained key paths recorded per diagnostic.
MAX_DIAGNOSTIC_KEYS = 32
#: Maximum removal entries recorded per diagnostic.
MAX_REMOVAL_ENTRIES = 64
#: Maximum characters retained for one recorded field path.
MAX_PATH_CHARS = 128
#: Maximum characters retained for one bounded type/size detail summary.
MAX_DETAIL_CHARS = 80
#: Maximum characters retained for a normalized ``json.JSONDecodeError`` message.
MAX_PARSE_ERROR_CHARS = 160

#: Fixed marker appended to a truncated key collection.
OMITTED_KEYS_MARKER = "...(truncated)"
#: Fixed field label used for the appended removal-list omitted-count marker.
OMITTED_REMOVALS_FIELD = "..."

#: Shared cap on the original response text passed into a correction prompt,
#: applied identically to every agent.
MAX_CORRECTION_RESPONSE_CHARS = 8000

# ---------------------------------------------------------------------------
# Closed reason-code sets
# ---------------------------------------------------------------------------

# Top-level rejection reasons — exactly one per rejected response, first match
# wins in this fixed precedence order.
REASON_NO_JSON_OBJECT = "no_json_object"
REASON_JSON_PARSE_ERROR = "json_parse_error"
REASON_TOP_LEVEL_NOT_OBJECT = "top_level_not_object"
REASON_FIXED_CAP_EXCEEDED = "fixed_cap_exceeded"
REASON_NO_USABLE_FIELDS = "no_usable_fields"
REASON_MISSING_REQUIRED = "missing_required"

TOP_LEVEL_REASONS = (
    REASON_NO_JSON_OBJECT,
    REASON_JSON_PARSE_ERROR,
    REASON_TOP_LEVEL_NOT_OBJECT,
    REASON_FIXED_CAP_EXCEEDED,
    REASON_NO_USABLE_FIELDS,
    REASON_MISSING_REQUIRED,
)

# Per-field removal reasons are recorded in ``removed``. Ordinary configurable
# response cleaning may retain other usable fields after any of these removals.
# Fixed split capsules are stricter: item/response-cap removals are elevated to
# ``fixed_cap_exceeded`` so bounded facts are never silently discarded.
REMOVAL_UNKNOWN_FIELD = "unknown_field"
REMOVAL_WRONG_TYPE = "wrong_type"
REMOVAL_EMPTY_VALUE = "empty_value"
REMOVAL_INVALID_VALUE = "invalid_value"
REMOVAL_DUPLICATE = "duplicate"
REMOVAL_ITEM_LIMIT = "item_limit"
REMOVAL_RESPONSE_CAP = "response_cap"
REMOVAL_NOT_REQUESTED = "not_requested"

PER_FIELD_REASONS = (
    REMOVAL_UNKNOWN_FIELD,
    REMOVAL_WRONG_TYPE,
    REMOVAL_EMPTY_VALUE,
    REMOVAL_INVALID_VALUE,
    REMOVAL_DUPLICATE,
    REMOVAL_ITEM_LIMIT,
    REMOVAL_RESPONSE_CAP,
    REMOVAL_NOT_REQUESTED,
)

# Extraction stages that begin the fixed diagnostic stage vocabulary.
STAGE_JSON_EXTRACT = "json_extract"
STAGE_JSON_PARSE = "json_parse"
STAGE_TOP_LEVEL = "top_level"
STAGE_CLEAN = "clean"
STAGE_PROFILE_FILTER = "profile_filter"
STAGE_REQUIRED_FIELDS = "required_fields"

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ---------------------------------------------------------------------------
# Immutable diagnostic models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RemovedField:
    """One removed field.  ``detail`` is a bounded type/size summary, never the
    field value."""

    field: str
    reason_code: str
    detail: str


@dataclass(frozen=True)
class ResponseDiagnostic:
    """Bounded, immutable summary of one rejected response.

    Carries only stage, reason code, bounded key names, bounded removal
    summaries, and a bounded parse message/position — never raw response,
    source, prompt, or instruction text.
    """

    stage: str
    reason_code: str
    agent: str
    file_path: str
    expected_keys: tuple[str, ...] = ()
    returned_keys: tuple[str, ...] = ()
    returned_types: tuple[str, ...] = ()
    retained_keys: tuple[str, ...] = ()
    removed: tuple[RemovedField, ...] = ()
    parse_error: str | None = None
    parse_position: int | None = None
    response_chars: int = 0

    def as_summary(self) -> dict:
        """Return a bounded json-safe dict for cross-boundary marker transport."""
        return {
            "stage": self.stage,
            "reason_code": self.reason_code,
            "agent": self.agent,
            "expected_keys": list(self.expected_keys),
            "returned_keys": list(self.returned_keys),
            "returned_types": list(self.returned_types),
            "retained_keys": list(self.retained_keys),
            "removed": [
                {"field": r.field, "reason_code": r.reason_code, "detail": r.detail}
                for r in self.removed
            ],
            "parse_error": self.parse_error,
            "parse_position": self.parse_position,
            "response_chars": self.response_chars,
        }


@dataclass(frozen=True)
class CleanResult:
    """The cleaned value paired with the bounded removals a reporter observed."""

    value: dict
    removed: tuple[RemovedField, ...] = ()
    # Explicit, correctly typed empty optional collections are valid responses
    # even though the historical cleaners omit them from the persisted value.
    valid_empty_paths: tuple[str, ...] = ()
    # Complete reason-code set observed before the bounded diagnostic list was
    # truncated. Fixed capsules use this to enforce losslessness even when
    # many earlier removals exhaust the user-facing diagnostic detail budget.
    removal_reason_codes: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Bounded detail helpers (shared by the response-cleaning reporters)
# ---------------------------------------------------------------------------

def type_detail(value: object) -> str:
    """Return a bounded type/size summary of *value*, never the value itself."""
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "null"
    if isinstance(value, str):
        return f"str[{len(value)}]"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)} keys]"
    return type(value).__name__


def scalar_removal_reason(value: object) -> str:
    """Classify why a scalar was dropped: wrong type, or an empty/blank string."""
    if isinstance(value, bool) or not isinstance(value, str):
        return REMOVAL_WRONG_TYPE
    return REMOVAL_EMPTY_VALUE


def _sanitize(text: str, cap: int) -> str:
    """Replace control characters and cap length."""
    cleaned = _CONTROL_CHARS_RE.sub(" ", text)
    return cleaned[:cap]


# ---------------------------------------------------------------------------
# Bounded removal collector (threaded through the cleaner reporters)
# ---------------------------------------------------------------------------

class RemovalCollector:
    """Collects bounded :class:`RemovedField` entries in first-seen order."""

    __slots__ = ("_items", "_omitted", "_reason_codes")

    def __init__(self) -> None:
        self._items: list[RemovedField] = []
        self._omitted = 0
        self._reason_codes: set[str] = set()

    def add(self, field: str, reason_code: str, detail: str) -> None:
        self._reason_codes.add(reason_code)
        if self._omitted:
            self._omitted += 1
            return
        if len(self._items) >= MAX_REMOVAL_ENTRIES:
            # Reserve one of the declared MAX_REMOVAL_ENTRIES slots for the
            # omitted-count marker once truncation actually occurs.
            if self._omitted == 0:
                self._items.pop()
                self._omitted = 2  # the displaced item plus this item
            else:
                self._omitted += 1
            return
        self._items.append(
            RemovedField(
                field=_sanitize(field, MAX_PATH_CHARS),
                reason_code=reason_code,
                detail=_sanitize(detail, MAX_DETAIL_CHARS),
            )
        )

    def finalize(self) -> tuple[RemovedField, ...]:
        items = list(self._items)
        if self._omitted:
            items.append(
                RemovedField(
                    field=OMITTED_REMOVALS_FIELD,
                    reason_code=REMOVAL_ITEM_LIMIT,
                    detail=f"+{self._omitted} more removals",
                )
            )
        return tuple(items)

    def reason_codes(self) -> frozenset[str]:
        """Return every observed reason, including omitted diagnostic entries."""
        return frozenset(self._reason_codes)


def _cap_keys(keys) -> tuple[str, ...]:
    """Bound a key collection: cap count and per-key length, preserve order."""
    out: list[str] = []
    extra = 0
    for key in keys:
        if len(out) >= MAX_DIAGNOSTIC_KEYS:
            extra += 1
            continue
        out.append(_sanitize(str(key), MAX_PATH_CHARS))
    if extra:
        # The marker is part of the declared collection maximum, not an extra
        # 33rd entry.
        out[-1] = OMITTED_KEYS_MARKER
    return tuple(out)


def _cap_removed(removed) -> tuple[RemovedField, ...]:
    """Bound an already-built removal tuple to the hard entry cap."""
    removed = tuple(removed)
    if len(removed) <= MAX_REMOVAL_ENTRIES:
        return removed
    kept = list(removed[:MAX_REMOVAL_ENTRIES - 1])
    kept.append(
        RemovedField(
            field=OMITTED_REMOVALS_FIELD,
            reason_code=REMOVAL_ITEM_LIMIT,
            detail=f"+{len(removed) - (MAX_REMOVAL_ENTRIES - 1)} more removals",
        )
    )
    return tuple(kept)


# ---------------------------------------------------------------------------
# Shared JSON-candidate extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?")


@dataclass(frozen=True)
class ExtractOutcome:
    """Outcome of JSON-candidate extraction from raw provider text.

    ``kind`` is one of ``"object"`` (``value`` is the parsed dict),
    ``"top_level_not_object"`` (``value`` is a parsed non-object JSON value),
    ``"json_parse_error"`` (``parse_error`` / ``parse_position`` populated), or
    ``"no_json_object"``.  ``json_str`` is the brace-delimited candidate string
    when one was selected (used only by the legacy ``_parse_json`` snippet path).
    """

    kind: str
    value: object = None
    json_str: str | None = None
    parse_error: str | None = None
    parse_position: int | None = None


def extract_json_candidate(raw: str) -> ExtractOutcome:
    """Extract and parse a JSON candidate from raw provider text.

    Compatibility algorithm (identical acceptance to the legacy ``_parse_json``):

    1. remove the currently supported Markdown-fence tokens and surrounding
       whitespace;
    2. when both an opening and closing brace exist, select the text from the
       first ``{`` through the last ``}`` and parse it;
    3. when no brace-delimited candidate exists, parse the whole cleaned text
       once — a parsed array/scalar/``null`` yields ``top_level_not_object`` and
       text that is not any JSON value yields ``no_json_object``;
    4. a selected brace candidate that fails ``json.loads`` yields
       ``json_parse_error`` with the bounded normalized message and position.
    """
    cleaned = _FENCE_RE.sub("", raw).strip().rstrip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        # No brace-delimited candidate: try to parse the whole cleaned text once.
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            return ExtractOutcome(kind=REASON_NO_JSON_OBJECT)
        if isinstance(value, dict):
            # Reachable only for a degenerate ``{`` / ``}`` asymmetry; keep it
            # consistent with the object path.
            return ExtractOutcome(kind="object", value=value)
        return ExtractOutcome(kind=REASON_TOP_LEVEL_NOT_OBJECT, value=value)

    json_str = cleaned[start:end + 1]
    try:
        value = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return ExtractOutcome(
            kind=REASON_JSON_PARSE_ERROR,
            json_str=json_str,
            parse_error=_sanitize(str(exc), MAX_PARSE_ERROR_CHARS),
            parse_position=getattr(exc, "pos", None),
        )
    if isinstance(value, dict):
        return ExtractOutcome(kind="object", value=value, json_str=json_str)
    return ExtractOutcome(kind=REASON_TOP_LEVEL_NOT_OBJECT, value=value, json_str=json_str)


# ---------------------------------------------------------------------------
# Requested-path / required-field helpers
# ---------------------------------------------------------------------------

def _requested_paths(
    mode: str, agent: str, resolved_shape: ResolvedShapeBlock | None
) -> tuple[str, ...]:
    """Effective requested field paths for ``(mode, agent)``."""
    if resolved_shape is not None and resolved_shape.active:
        return resolved_shape.requested_field_paths
    return default_field_paths(mode, agent)


def _leaf_paths(value: dict) -> list[str]:
    """Flatten a cleaned/filtered dict into registered leaf paths.

    ``dependencies_analysis`` is expanded into ``dependencies_analysis.<member>``
    paths so a surviving nested member counts as a retained requested field.
    """
    paths: list[str] = []
    for key, member in value.items():
        if key == "dependencies_analysis" and isinstance(member, dict):
            for sub in member:
                paths.append(f"dependencies_analysis.{sub}")
        else:
            paths.append(key)
    return paths


def required_field_paths(mode: str, agent: str) -> tuple[str, ...]:
    """Registry-required leaf paths for ``(mode, agent)``.

    Iterates the flattened registry leaves (every ``ShapeField``, including
    ``dependencies_analysis`` members) and keeps each ``fld.path`` where
    ``fld.required is True``.  Never iterates ``PROMPT_SHAPE_REGISTRY`` directly,
    whose entries include ``ContainerField`` (no ``required`` attribute).
    """
    return tuple(fld.path for fld in iter_fields(mode, agent) if fld.required)


def _filter_with_report(
    cleaned: dict,
    resolved_shape: ResolvedShapeBlock | None,
    mode: str,
    agent: str,
) -> tuple[dict, list[RemovedField]]:
    """Profile-filter *cleaned* and report each dropped registered field.

    Uses the canonical :func:`filter_cleaned_response_for_profile` for acceptance
    (so filtering behaviour is unchanged) and diffs its input against its output
    to report each dropped registered field/member as ``not_requested``.
    """
    filtered = filter_cleaned_response_for_profile(
        cleaned, resolved_shape, mode=mode, agent=agent
    )
    if filtered is cleaned or filtered == cleaned:
        return filtered, []
    removed: list[RemovedField] = []
    for key, member in cleaned.items():
        if key == "dependencies_analysis" and isinstance(member, dict):
            kept = filtered.get("dependencies_analysis")
            kept_members = kept if isinstance(kept, dict) else {}
            for sub, sub_value in member.items():
                if sub not in kept_members:
                    removed.append(
                        RemovedField(
                            field=f"dependencies_analysis.{sub}",
                            reason_code=REMOVAL_NOT_REQUESTED,
                            detail=type_detail(sub_value),
                        )
                    )
        elif key not in filtered:
            removed.append(
                RemovedField(
                    field=key,
                    reason_code=REMOVAL_NOT_REQUESTED,
                    detail=type_detail(member),
                )
            )
    return filtered, removed


# ---------------------------------------------------------------------------
# Canonical response path
# ---------------------------------------------------------------------------

def process_response(
    raw: str,
    *,
    mode: str,
    agent: str,
    file_path: str,
    clean_reporter,
    resolved_shape: ResolvedShapeBlock | None,
) -> dict:
    """Turn a raw provider response into a validated dict, or raise.

    Runs, in order: JSON candidate selection -> JSON parse -> top-level object
    check -> ``clean_reporter`` -> profile filtering -> ``no_usable_fields`` ->
    registry-required-field validation.  Returns the cleaned, profile-filtered
    dict, or raises :class:`ResponseContractError` with a fully populated,
    bounded :class:`ResponseDiagnostic` and ``correction_attempted=False``.

    The same function is used for both initial and corrected responses.
    """
    requested = _requested_paths(mode, agent, resolved_shape)
    requested_set = set(requested)
    response_chars = len(raw)

    outcome = extract_json_candidate(raw)
    if outcome.kind == REASON_NO_JSON_OBJECT:
        _raise(mode, agent, file_path, STAGE_JSON_EXTRACT, REASON_NO_JSON_OBJECT,
               expected=requested, response_chars=response_chars)
    if outcome.kind == REASON_JSON_PARSE_ERROR:
        _raise(mode, agent, file_path, STAGE_JSON_PARSE, REASON_JSON_PARSE_ERROR,
               expected=requested, response_chars=response_chars,
               parse_error=outcome.parse_error, parse_position=outcome.parse_position)
    if outcome.kind == REASON_TOP_LEVEL_NOT_OBJECT:
        _raise(mode, agent, file_path, STAGE_TOP_LEVEL, REASON_TOP_LEVEL_NOT_OBJECT,
               expected=requested, response_chars=response_chars,
               removed=(RemovedField("<top-level>", REMOVAL_WRONG_TYPE,
                                     type_detail(outcome.value)),))

    parsed: dict = outcome.value  # a dict at this point
    returned_keys = tuple(parsed.keys())

    clean_result: CleanResult = clean_reporter(parsed, file_path)
    cleaned = clean_result.value
    removed = list(clean_result.removed)

    filtered, not_requested = _filter_with_report(cleaned, resolved_shape, mode, agent)
    removed.extend(not_requested)

    retained_set = {p for p in _leaf_paths(filtered) if p in requested_set}
    retained_set.update(
        p for p in clean_result.valid_empty_paths if p in requested_set
    )
    # Registry/request order is stable and avoids duplicates when an explicitly
    # empty path is also materialized by a future cleaner.
    retained = tuple(p for p in requested if p in retained_set)

    returned_types = tuple(
        f"{key}:{type_detail(value)}" for key, value in parsed.items()
    )

    if requested_set and not retained:
        _raise(mode, agent, file_path, STAGE_PROFILE_FILTER, REASON_NO_USABLE_FIELDS,
               expected=requested, returned=returned_keys, retained=retained,
               returned_types=returned_types, removed=removed,
               response_chars=response_chars)

    retained_leaves = set(_leaf_paths(filtered))
    missing = [
        path
        for path in required_field_paths(mode, agent)
        if path in requested_set and path not in retained_leaves
    ]
    if missing:
        _raise(mode, agent, file_path, STAGE_REQUIRED_FIELDS, REASON_MISSING_REQUIRED,
               expected=requested, returned=returned_keys, retained=retained,
               returned_types=returned_types, removed=removed,
               response_chars=response_chars,
               reason_detail="; ".join(missing))

    return filtered


def process_fixed_capsule_response(
    raw: str,
    *,
    label: str,
    agent: str,
    file_path: str,
    clean_reporter,
    requested_paths: tuple[str, ...],
    required_paths: tuple[str, ...],
) -> dict:
    """Turn a raw provider response into a validated fixed-schema capsule.

    The split leaf/reduction internal contract is never prompt-profile
    driven (D5/D6): there is no filtering stage, and the requested/required
    paths are the fixed capsule schema itself, not a registry lookup.
    Otherwise follows :func:`process_response` through JSON candidate
    selection -> JSON parse -> top-level object check -> ``clean_reporter``.
    It then rejects lossy item/response-cap removals before
    ``no_usable_fields`` and required-field validation. It uses the same bounded
    :class:`ResponseDiagnostic` / :class:`~codedoc.utils.errors.
    ResponseContractError` shape and the same correction path.

    *label* is a bounded descriptive stage tag (e.g. ``"split-leaf"`` or
    ``"split-reduction"``) carried only for diagnostics; it never selects a
    registry.
    """
    requested_set = set(requested_paths)
    response_chars = len(raw)

    outcome = extract_json_candidate(raw)
    if outcome.kind == REASON_NO_JSON_OBJECT:
        _raise(label, agent, file_path, STAGE_JSON_EXTRACT, REASON_NO_JSON_OBJECT,
               expected=requested_paths, response_chars=response_chars)
    if outcome.kind == REASON_JSON_PARSE_ERROR:
        _raise(label, agent, file_path, STAGE_JSON_PARSE, REASON_JSON_PARSE_ERROR,
               expected=requested_paths, response_chars=response_chars,
               parse_error=outcome.parse_error, parse_position=outcome.parse_position)
    if outcome.kind == REASON_TOP_LEVEL_NOT_OBJECT:
        _raise(label, agent, file_path, STAGE_TOP_LEVEL, REASON_TOP_LEVEL_NOT_OBJECT,
               expected=requested_paths, response_chars=response_chars,
               removed=(RemovedField("<top-level>", REMOVAL_WRONG_TYPE,
                                     type_detail(outcome.value)),))

    parsed: dict = outcome.value
    returned_keys = tuple(parsed.keys())

    clean_result: CleanResult = clean_reporter(parsed, file_path)
    cleaned = clean_result.value
    removed = list(clean_result.removed)

    retained_set = {p for p in _leaf_paths(cleaned) if p in requested_set}
    retained_set.update(p for p in clean_result.valid_empty_paths if p in requested_set)
    retained = tuple(p for p in requested_paths if p in retained_set)

    returned_types = tuple(f"{key}:{type_detail(value)}" for key, value in parsed.items())

    if clean_result.removal_reason_codes.intersection(
        (REMOVAL_ITEM_LIMIT, REMOVAL_RESPONSE_CAP)
    ):
        _raise(
            label,
            agent,
            file_path,
            STAGE_CLEAN,
            REASON_FIXED_CAP_EXCEEDED,
            expected=requested_paths,
            returned=returned_keys,
            retained=retained,
            returned_types=returned_types,
            removed=removed,
            response_chars=response_chars,
            reason_detail=(
                "fixed split capsules must be corrected rather than "
                "published after bounded facts were removed"
            ),
        )

    if requested_set and not retained:
        _raise(label, agent, file_path, STAGE_CLEAN, REASON_NO_USABLE_FIELDS,
               expected=requested_paths, returned=returned_keys, retained=retained,
               returned_types=returned_types, removed=removed, response_chars=response_chars)

    retained_leaves = set(_leaf_paths(cleaned))
    missing = [path for path in required_paths if path not in retained_leaves]
    if missing:
        _raise(label, agent, file_path, STAGE_REQUIRED_FIELDS, REASON_MISSING_REQUIRED,
               expected=requested_paths, returned=returned_keys, retained=retained,
               returned_types=returned_types, removed=removed, response_chars=response_chars,
               reason_detail="; ".join(missing))

    return cleaned


def _raise(
    mode: str,
    agent: str,
    file_path: str,
    stage: str,
    reason: str,
    *,
    expected=(),
    returned=(),
    returned_types=(),
    retained=(),
    removed=(),
    parse_error: str | None = None,
    parse_position: int | None = None,
    response_chars: int = 0,
    reason_detail: str | None = None,
) -> "None":
    """Build a bounded diagnostic, log it, and raise ``ResponseContractError``."""
    diagnostic = ResponseDiagnostic(
        stage=stage,
        reason_code=reason,
        agent=agent,
        file_path=file_path,
        expected_keys=_cap_keys(expected),
        returned_keys=_cap_keys(returned),
        returned_types=_cap_keys(returned_types),
        retained_keys=_cap_keys(retained),
        removed=_cap_removed(removed),
        parse_error=parse_error,
        parse_position=parse_position,
        response_chars=response_chars,
    )
    _log_rejection(diagnostic)
    message = f"response contract failed ({reason})"
    if reason_detail:
        message += f": {reason_detail}"
    raise ResponseContractError(
        agent, file_path, message,
        diagnostic=diagnostic, correction_attempted=False,
    )


def _log_rejection(diagnostic: ResponseDiagnostic) -> None:
    """Emit one concise WARNING line, plus bounded structural DEBUG metadata.

    Never logs source content, prompts, API keys, full raw responses, or any
    provider request/response body.
    """
    logger.warning(
        "Response rejected: file=%s agent=%s stage=%s reason=%s "
        "expected=[%s] returned=[%s] retained=[%s]",
        diagnostic.file_path,
        diagnostic.agent,
        diagnostic.stage,
        diagnostic.reason_code,
        ",".join(diagnostic.expected_keys),
        ",".join(diagnostic.returned_keys),
        ",".join(diagnostic.retained_keys),
    )
    if logger.isEnabledFor(10):  # logging.DEBUG
        logger.debug(
            "Response rejected (detail): file=%s agent=%s returned_types=%s removed=%s "
            "parse_position=%s response_chars=%d",
            diagnostic.file_path,
            diagnostic.agent,
            diagnostic.returned_types,
            [(r.field, r.reason_code, r.detail) for r in diagnostic.removed],
            diagnostic.parse_position,
            diagnostic.response_chars,
        )


# ---------------------------------------------------------------------------
# Run-level correction statistics
# ---------------------------------------------------------------------------

class CorrectionLedger:
    """Thread-safe run-level correction statistics.

    Accumulates counts only — never any response, profile, or source text.  One
    ledger is built per run and shared by the single ``ResponseCorrectionAgent``.
    """

    __slots__ = ("_lock", "_enabled", "_contract_failures", "_attempts",
                 "_succeeded", "_failed")

    def __init__(self, enabled: bool) -> None:
        self._lock = threading.Lock()
        self._enabled = bool(enabled)
        self._contract_failures = 0
        self._attempts = 0
        self._succeeded = 0
        self._failed = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record_contract_failure(self) -> None:
        with self._lock:
            self._contract_failures += 1

    def record_attempt(self) -> None:
        with self._lock:
            self._attempts += 1

    def record_success(self) -> None:
        with self._lock:
            self._succeeded += 1

    def record_failure(self) -> None:
        with self._lock:
            self._failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "response_correction_enabled": self._enabled,
                "response_contract_failures": self._contract_failures,
                "response_correction_calls_attempted": self._attempts,
                "response_correction_calls_succeeded": self._succeeded,
                "response_correction_calls_failed": self._failed,
            }
