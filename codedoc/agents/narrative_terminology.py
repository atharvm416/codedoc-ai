"""One conservative narrative-terminology authority.

Two things live here and nowhere else:

* :data:`NARRATIVE_TERMINOLOGY_RULES` -- the verbatim prompt clauses that every
  applicable initial and correction prompt renders unchanged, so the rules
  cannot drift between routes or be removed by a custom prompt profile.
* :func:`validate_narrative_terminology` -- a closed, deterministic check that
  runs inside the canonical response-contract path on already-cleaned bounded
  narrative values only. It removes a narrative value that carries an acronym
  expansion proven unsupported by the four closed conditions of plan section
  5.6; the surrounding response-contract path then corrects or rejects the
  response exactly as it does for any other removal. It never rewrites prose,
  never guesses, and never treats model output as evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from codedoc.agents.response_diagnostics import (
    MAX_DETAIL_CHARS,
    REMOVAL_UNSUPPORTED_TERMINOLOGY,
    RemovedField,
)

# ---------------------------------------------------------------------------
# Prompt rules (rendered verbatim into every applicable prompt)
# ---------------------------------------------------------------------------

# Kept free of ``{`` / ``}`` so it concatenates safely into a ``str.format``
# template, mirroring ``EXACT_JSON_RESPONSE_RULES``.
NARRATIVE_TERMINOLOGY_RULES = (
    "- Do not expand an acronym unless its expansion appears verbatim in the "
    "supplied source, parser metadata, or trusted project metadata\n"
    "- If an acronym is not defined in that evidence, keep the acronym as "
    "written and do not guess what it stands for\n"
    "- Do not call a file \"the entry point\" only because it defines or exports "
    "a root component; reserve \"entry point\" for visible startup or bootstrap "
    "behavior, or an explicit trusted entry declaration\n"
    "- Do not turn a filename into a function, class, component, or other "
    "declaration name"
)

# Value-safe fixed detail string for every terminology removal. It names the
# rule, never the offending phrase, acronym, field value, or any source text.
_REMOVAL_DETAIL = "closed initialism rule: unsupported acronym expansion"[
    :MAX_DETAIL_CHARS
]


# ---------------------------------------------------------------------------
# Immutable per-call evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminologyEvidence:
    """Immutable trusted evidence for one analysis call.

    ``source_text`` is the exact planned/truncated source string the call was
    given (never a fresh filesystem read). ``metadata_text`` holds trusted
    deterministic project/parser metadata (symbol names and signatures)
    concatenated into one searchable blob. The initial response and its one
    correction are validated against the same instance.
    """

    source_text: str = ""
    metadata_text: str = ""

    def has_source(self) -> bool:
        return bool(self.source_text and self.source_text.strip())


def terminology_metadata_text(*symbol_sources: object) -> str:
    """Join parser-owned symbol names / signatures into one searchable blob.

    Accepts any mix of ``None``, a list of ``{name, signature?}`` mappings, or a
    list of plain-string names (the split-leaf ``known_symbols`` shape). Only
    parser-owned identifiers reach here -- never model narrative.
    """
    parts: list[str] = []
    for source in symbol_sources:
        if not isinstance(source, (list, tuple)):
            continue
        for item in source:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                for key in ("name", "qualified_name", "signature"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Closed initialism grammar (deliberately narrow -- not a language parser)
# ---------------------------------------------------------------------------

# An acronym token: 2-6 uppercase letters/digits, letter first, standalone
# (not part of a longer identifier such as ``DPRTab``). There is no
# knowledge-based allowlist or denylist: plan section 5.6 permits an expansion
# only when it appears verbatim in trusted evidence, so every acronym token in
# the trusted source is checked the same way.
_ACRONYM_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9]{1,5})(?![A-Za-z0-9_])")

# A significant Title-Case word contributes one initial: an initial capital
# then one-or-more lowercase letters.
_SIGNIFICANT_WORD = re.compile(r"[A-Z][a-z]+")
# Lowercase connectors may sit *between* significant words without contributing
# an initial. They are matched lowercase only.
_CONNECTOR_WORDS = frozenset(
    {"of", "the", "and", "for", "to", "in", "on", "a", "an", "with", "by"}
)
_WORD_TOKEN = re.compile(r"[A-Za-z]+")

_MIN_PHRASE_WORDS = 2
_MAX_PHRASE_WORDS = 6


def _source_acronyms(evidence: TerminologyEvidence) -> set[str]:
    """Acronym tokens that appear literally in the trusted source (condition 1).

    Metadata identifiers are trusted for *supporting* an expansion (condition 3)
    but not for *triggering* the rule, so only ``source_text`` is scanned here.
    """
    return set(_ACRONYM_TOKEN.findall(evidence.source_text))


def _phrase_supported(phrase: str, evidence: TerminologyEvidence) -> bool:
    """Condition 3: the complete candidate phrase occurs *exactly and verbatim*
    in the trusted evidence. No case-folding, whitespace collapsing, or
    punctuation normalization -- a near match is not evidence. The exact text
    must also be bounded by non-identifier characters so a prefix inside a
    longer word (for example, ``Report`` inside ``Reporter``) is not mistaken
    for the complete phrase.
    """

    def _is_identifier_char(char: str) -> bool:
        return char.isascii() and (char.isalnum() or char == "_")

    def _contains_complete_phrase(haystack: str) -> bool:
        start = 0
        while True:
            position = haystack.find(phrase, start)
            if position < 0:
                return False
            end = position + len(phrase)
            left_is_clear = position == 0 or not _is_identifier_char(
                haystack[position - 1]
            )
            right_is_clear = end == len(haystack) or not _is_identifier_char(
                haystack[end]
            )
            if left_is_clear and right_is_clear:
                return True
            start = position + 1

    return _contains_complete_phrase(
        evidence.source_text
    ) or _contains_complete_phrase(evidence.metadata_text)


def _candidate_phrases(value: str) -> Iterable[tuple[str, str]]:
    """Yield ``(phrase_text, initials)`` for every contiguous run of 2..6
    significant Title-Case words (allowed lowercase connectors permitted
    between them), including qualifying subspans inside a longer run.

    Deterministic left-to-right, bounded: each run of *N* significant words
    yields at most ``N * (_MAX_PHRASE_WORDS - 1)`` subspans, each of at most
    ``_MAX_PHRASE_WORDS`` initials. ``phrase_text`` is the exact slice of
    *value* from the first significant word to the last (connectors and their
    surrounding spacing included verbatim); leading/trailing connectors are
    excluded.
    """
    # Classify each word token; a run is a maximal sequence of significant /
    # connector tokens separated only by whitespace.
    run: list[tuple[int, int, str, bool]] = []  # (start, end, text, is_significant)
    prev_end = -1

    def _emit(current: list[tuple[int, int, str, bool]]):
        sig = [i for i, tok in enumerate(current) if tok[3]]
        for a in range(len(sig)):
            for b in range(a + _MIN_PHRASE_WORDS - 1,
                           min(a + _MAX_PHRASE_WORDS, len(sig))):
                start = current[sig[a]][0]
                end = current[sig[b]][1]
                initials = "".join(current[sig[k]][2][0] for k in range(a, b + 1))
                yield value[start:end], initials

    for match in _WORD_TOKEN.finditer(value):
        text = match.group(0)
        is_sig = bool(_SIGNIFICANT_WORD.fullmatch(text))
        is_conn = (not is_sig) and text in _CONNECTOR_WORDS
        gap_is_ws_only = (
            prev_end == -1 or not value[prev_end:match.start()].strip()
        )
        if (is_sig or is_conn) and (not run or gap_is_ws_only):
            run.append((match.start(), match.end(), text, is_sig))
        else:
            if run:
                yield from _emit(run)
            run = [(match.start(), match.end(), text, is_sig)] if (is_sig or is_conn) else []
        prev_end = match.end()
    if run:
        yield from _emit(run)


def _has_unsupported_expansion(
    value: str, acronyms: set[str], evidence: TerminologyEvidence
) -> bool:
    """True when *value* contains a candidate Title-Case phrase whose initials
    equal a source acronym and whose exact text is absent verbatim from all
    trusted evidence. Every branch that cannot decide safely returns ``False``.
    """
    if not acronyms or not isinstance(value, str):
        return False
    for phrase, initials in _candidate_phrases(value):
        if initials not in acronyms:
            continue
        if _phrase_supported(phrase, evidence):
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# The closed check, run inside the canonical response-contract path
# ---------------------------------------------------------------------------

# The only fields whose whole value is bounded model narrative.
_SCALAR_NARRATIVE_FIELDS = ("description", "role_in_system", "usage_example")
_LIST_NARRATIVE_FIELDS = ("key_concepts",)
_SYMBOL_LIST_FIELDS = ("functions", "classes")


def validate_narrative_terminology(
    cleaned: object,
    evidence: TerminologyEvidence | None,
    *,
    mode: str = "",
    agent: str = "",
) -> tuple[object, list[RemovedField]]:
    """Return ``(cleaned_or_copy, removed)``.

    Removes any bounded narrative value that carries a closed-rule-provable
    unsupported acronym expansion. A no-op -- returns the input unchanged and an
    empty list -- whenever there is no trusted source, no source acronym, or the
    prose cannot be classified. Never rewrites a value.
    """
    if (
        not isinstance(cleaned, dict)
        or evidence is None
        or not evidence.has_source()
    ):
        return cleaned, []
    acronyms = _source_acronyms(evidence)
    if not acronyms:
        return cleaned, []

    removed: list[RemovedField] = []
    out = dict(cleaned)

    def _flag(field: str) -> None:
        removed.append(
            RemovedField(
                field=field,
                reason_code=REMOVAL_UNSUPPORTED_TERMINOLOGY,
                detail=_REMOVAL_DETAIL,
            )
        )

    for field in _SCALAR_NARRATIVE_FIELDS:
        value = out.get(field)
        if isinstance(value, str) and _has_unsupported_expansion(
            value, acronyms, evidence
        ):
            del out[field]
            _flag(field)

    for field in _LIST_NARRATIVE_FIELDS:
        items = out.get(field)
        if not isinstance(items, list):
            continue
        kept: list = []
        for index, item in enumerate(items):
            if isinstance(item, str) and _has_unsupported_expansion(
                item, acronyms, evidence
            ):
                _flag(f"{field}[{index}]")
            else:
                kept.append(item)
        if len(kept) != len(items):
            out[field] = kept

    for field in _SYMBOL_LIST_FIELDS:
        items = out.get(field)
        if not isinstance(items, list):
            continue
        changed = False
        rebuilt: list = []
        for index, item in enumerate(items):
            if (
                isinstance(item, Mapping)
                and isinstance(item.get("description"), str)
                and _has_unsupported_expansion(
                    item["description"], acronyms, evidence
                )
            ):
                rebuilt.append(
                    {k: v for k, v in item.items() if k != "description"}
                )
                _flag(f"{field}[{index}].description")
                changed = True
            else:
                rebuilt.append(item)
        if changed:
            out[field] = rebuilt

    if not removed:
        return cleaned, []
    return out, removed
