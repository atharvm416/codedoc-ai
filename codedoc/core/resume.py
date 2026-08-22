"""Resume and recovery helpers for the documentation pipeline.

This module owns:

- loading stable per-file documentation from the exact selected prior JSON /
  Markdown output, with one strict opposite-format sibling fallback only when
  the selected target is absent (no directory or legacy-candidate probing);
- the both-mode cross-document identity consistency check;
- the exact single-recovery-file reader and its versioned run identity;
- reconstructing the internal documentation shape from a public JSON record;
- building the final documentation records handed to ``write_project_outputs``.

It depends on the read-only document reader, output/project-view helpers,
hashing, and paths.  It must not scan source files, create providers, or
schedule agent work.

Recovery is exactly one fixed file, ``<output_dir>/crash_recovery.json`` (see
:data:`RECOVERY_FILENAME`).  There is no candidate walk, numbered suffix, legacy
overlay, or sibling lookup: a missing file means a fresh recovery state, and an
owned in-progress file is reused only when its versioned identity matches the
current run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.file_division import SplitTreeState, canonical_json
from codedoc.core.markdown_view import markdown_to_view
from codedoc.core.project_view import read_codedoc_meta
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    carry_private_keys,
    normalized_identity_value,
)
from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# The single, fixed crash-recovery filename.  Crash-recovery data lives in its
# own dedicated file in the output directory so a run never overwrites the user's
# last stable completed output while it is still in progress.  This is a fixed
# internal constant — there is no configuration key, environment variable, or CLI
# flag for it — and the same constant guards ``--output`` (see
# ``loader._resolve_output_spec``) and path-distinctness validation.
RECOVERY_FILENAME = "crash_recovery.json"

# Supported versioned-recovery-identity schema version.
_RECOVERY_IDENTITY_VERSION = 1


@dataclass(frozen=True, slots=True)
class RecoveryState:
    """Deeply immutable contents of one compatible recovery file."""

    records: tuple[tuple[str, str], ...] = ()
    partial_files: tuple[SplitTreeState, ...] = ()

    def records_by_path(self) -> dict[str, dict]:
        """Materialize defensive record dictionaries for routing."""
        import json

        return {rel_path: json.loads(payload) for rel_path, payload in self.records}


# ---------------------------------------------------------------------------
# Existing-docs loader (selected target, then exact format sibling)
# ---------------------------------------------------------------------------

def _load_existing_file_docs(
    json_target: Path | None,
    md_target: Path | None,
    output_format: str,
) -> dict[str, dict]:
    """Load stable records from the selected target or its exact format sibling.

    - ``json``: load JSON when it exists; otherwise strictly validate and load
      the exact Markdown sibling.
    - ``md``: load Markdown when it exists; otherwise strictly validate and load
      the exact JSON sibling.
    - ``both``: load both when present; run the canonical cross-document identity
      comparison and raise :class:`ConfigError` on any mismatch before returning.
      JSON is authoritative when both are valid; when only one exists it is used.

    An existing selected target is always authoritative in single-format mode,
    so its sibling is not inspected.  Fallback probes no directory, recovery
    file, or alternate name.  Unlike optional exact-target reuse, a present
    fallback sibling is strict: foreign or malformed data raises before paid
    work. Recovery reuse is handled separately by
    :func:`load_recovery_records_if_compatible`.
    """
    if output_format == "json":
        if json_target is not None and json_target.exists():
            return _load_json_records(json_target)
        if md_target is not None and md_target.exists():
            return _load_strict_fallback_records(md_target, "Markdown")
        return {}
    if output_format == "md":
        if md_target is not None and md_target.exists():
            return _load_md_records(md_target)
        if json_target is not None and json_target.exists():
            return _load_strict_fallback_records(json_target, "JSON")
        return {}

    # both — JSON authoritative; cross-check consistency when both exist.
    json_records = _load_json_records(json_target)
    md_records = _load_md_records(md_target)
    if (
        json_target is not None
        and md_target is not None
        and json_target.exists()
        and md_target.exists()
    ):
        _assert_cross_document_consistency(json_target, md_target)
        return json_records
    return json_records or md_records


def _load_strict_fallback_records(path: Path, label: str) -> dict[str, dict]:
    """Return records from an owned opposite-format fallback or fail closed."""
    try:
        return records_by_path(read_codedoc_document(path))
    except (ConfigError, FileNotFoundError) as exc:
        raise ConfigError(
            f"The requested output target does not exist, but its {label} "
            f"conversion sibling '{path.name}' is not a readable CodeDoc "
            f"document: {exc}. Rename or remove that sibling, or choose a "
            "different output path, before rerunning. No provider was contacted."
        ) from exc


def _load_json_records(json_target: Path | None) -> dict[str, dict]:
    """Load per-file records from the exact JSON target, or ``{}``."""
    if json_target is None or not json_target.exists():
        return {}
    try:
        return records_by_path(read_codedoc_document(json_target))
    except (ConfigError, FileNotFoundError) as exc:
        # A foreign/malformed exact target is blocked by the ownership preflight
        # before provider contact; here (planning) treat it as no reuse.
        logger.debug("Exact JSON target '%s' not reusable: %s", json_target.name, exc)
        return {}


def _load_md_records(md_target: Path | None) -> dict[str, dict]:
    """Load per-file records from the exact Markdown target, or ``{}``."""
    if md_target is None or not md_target.exists():
        return {}
    try:
        return _load_existing_file_docs_from_md(md_target)
    except Exception as exc:  # noqa: BLE001 — optional reuse source
        logger.debug("Exact Markdown target '%s' not reusable: %s", md_target.name, exc)
        return {}


def _assert_cross_document_consistency(json_target: Path, md_target: Path) -> None:
    """Raise :class:`ConfigError` if the two both-mode targets disagree.

    Compares one canonical identity: the entry file,
    the exact sorted set of documented file paths, and — for every path — its
    content ``hash`` plus every normalized :data:`CACHE_IDENTITY_KEYS` value (all
    of which round-trip losslessly through the Markdown embedded view).  Names the
    first deterministic mismatch and both values.  This is an unconditional
    read-only preflight check that runs before provider construction and any
    mutation.
    """
    jdoc = read_codedoc_document(json_target)
    jrecords = records_by_path(jdoc)
    mrecords = _load_existing_file_docs_from_md(md_target)
    try:
        mmeta = read_codedoc_meta(md_target) or {}
    except ConfigError:
        mmeta = {}

    def _conflict(field: str, jval: object, mval: object) -> None:
        raise ConfigError(
            f"The selected JSON output '{json_target.name}' and Markdown output "
            f"'{md_target.name}' disagree on {field}: JSON has {jval!r}, Markdown "
            f"has {mval!r}. Regenerate one target, or select a single format, so "
            "the incremental state is unambiguous."
        )

    if jdoc.entry_file != mmeta.get("entry_file"):
        _conflict("the entry file", jdoc.entry_file, mmeta.get("entry_file"))
    jpaths, mpaths = sorted(jrecords), sorted(mrecords)
    if jpaths != mpaths:
        _conflict("the set of documented file paths", jpaths, mpaths)
    for path in jpaths:
        jr, mr = jrecords[path], mrecords[path]
        if jr.get("hash", "") != mr.get("hash", ""):
            _conflict(f"the content hash of '{path}'", jr.get("hash", ""), mr.get("hash", ""))
        for key in sorted(CACHE_IDENTITY_KEYS):
            jval = normalized_identity_value(key, jr)
            mval = normalized_identity_value(key, mr)
            if jval != mval:
                _conflict(f"the {key} of '{path}'", jval, mval)


def _load_existing_file_docs_from_md(md_path: Path) -> dict[str, dict]:
    """Parse an existing MD output file into a file-record dict."""
    content = md_path.read_text(encoding="utf-8-sig", errors="replace")

    file_hashes: dict[str, str] = {}
    try:
        meta = read_codedoc_meta(md_path)
        file_hashes = meta.get("file_hashes") or {}
    except ConfigError:
        pass

    view = markdown_to_view(content)
    result: dict[str, dict] = {}
    for file_record in view.get("files", []):
        path = file_record.get("path")
        if path:
            # Prefer the hash from the lightweight metadata comment (authoritative
            # for incremental reuse).  When the embedded view already contains a
            # hash (embedded-view Markdown), use that as the fallback so we never
            # overwrite a good hash with an empty string.
            file_hash = file_hashes.get(path) or file_record.get("hash", "")
            result[path] = {**file_record, "hash": file_hash}
    return result


# ---------------------------------------------------------------------------
# Exact single-recovery-file identity and reader
# ---------------------------------------------------------------------------

def _normalized_path(path: Path | str | None) -> str | None:
    """Normalize a path to an absolute, platform-case string without following
    symlinks (so an out-of-root alias is not chased into a different project)."""
    if path is None:
        return None
    return os.path.normcase(os.path.abspath(str(path)))


def build_recovery_identity(
    *,
    project_root: Path,
    json_target: Path | None,
    md_target: Path | None,
    entry_file: str | None,
    documentation_scope: str,
    analysis_mode: str,
    analysis_revision: str,
    large_file_strategy: str = "truncate",
) -> dict:
    """Build the versioned ``_codedoc.recovery_identity`` object for this run.

    Bounded and free of source/credential data.  The identity binds run-wide
    routing and output choices only; it does **not** bind a profile-wide digest.
    Per-file ``_prompt_profile_digest`` values in ``CACHE_IDENTITY_KEYS`` already
    re-validate every recovered record individually, so a language-keyed or
    path-keyed run-level map would only reject otherwise-resumable recovery for
    unrelated edits.

    ``large_file_strategy`` is emitted **only** for a ``split`` run.  A
    ``truncate`` run omits it, so default recovery-identity bytes are unchanged,
    and an absent stored value is normalized back to ``"truncate"`` when
    comparing (see :data:`_LEGACY_IDENTITY_DEFAULTS`).  Because absence has an
    explicit backward-compatible meaning, a legacy recovery file stays readable
    and ``_RECOVERY_IDENTITY_VERSION`` remains 1 (only widening the compared set
    without such a default, or changing a retained field's meaning, would require
    a version bump).
    """
    identity = {
        "version": _RECOVERY_IDENTITY_VERSION,
        "project_root": _normalized_path(project_root),
        "json_target": _normalized_path(json_target),
        "md_target": _normalized_path(md_target),
        "entry_file": entry_file,
        "documentation_scope": documentation_scope,
        "analysis_mode": analysis_mode,
        "analysis_revision": analysis_revision,
    }
    if large_file_strategy == "split":
        identity["large_file_strategy"] = "split"
    return identity


_RECOVERY_IDENTITY_FIELDS = (
    "project_root",
    "json_target",
    "md_target",
    "entry_file",
    "documentation_scope",
    "analysis_mode",
    "analysis_revision",
    "large_file_strategy",
)

# Identity fields whose absence carries an explicit backward-compatible value.
# A legacy or default recovery file has no ``large_file_strategy`` and must stay
# compatible with a ``truncate`` run; only a ``split`` run stores the field, so a
# strategy change is a genuine mismatch instead of a silent reinterpretation of
# incompatible split checkpoints.
_LEGACY_IDENTITY_DEFAULTS = {"large_file_strategy": "truncate"}


def _identity_field(identity: dict, field: str) -> object:
    """Return one identity field, normalizing a legacy absent value."""
    value = identity.get(field)
    if value is None and field in _LEGACY_IDENTITY_DEFAULTS:
        return _LEGACY_IDENTITY_DEFAULTS[field]
    return value


def _first_identity_mismatch(
    stored: dict, expected: dict
) -> tuple[str, object, object] | None:
    """Return ``(field, expected, found)`` for the first mismatching identity
    field, or ``None`` when every field matches."""
    if stored.get("version") != expected["version"]:
        return ("version", expected["version"], stored.get("version"))
    for field in _RECOVERY_IDENTITY_FIELDS:
        found = _identity_field(stored, field)
        wanted = _identity_field(expected, field)
        if found != wanted:
            return (field, wanted, found)
    return None


def _recovery_remedy(recovery_path: Path) -> str:
    """Preserve-first remedy for a generic recovery conflict.

    Ordering matters: this file can hold already-paid checkpoints, so the
    non-destructive options come first and deletion is described only as a
    deliberate discard — matching the D11 predecessor remedy below and the
    published recovery contract. Your stable completed output is never
    touched by any of these choices.
    """
    return (
        f"To return to that work, restore the prior run's configuration. "
        "Compatible completed ordinary and split records may then be reused, "
        "and compatible current schema-4 split node checkpoints may resume; "
        "forced, stale, identity-mismatched, legacy, foreign, or unsupported "
        "state is rerun or preserved and blocked according to the documented "
        "remedy. To continue with the current configuration instead, move "
        f"'{recovery_path.name}' aside (for example, rename it) so a fresh "
        f"recovery file is created; you may instead delete it, but that "
        f"explicitly discards any completed work it still holds. Your prior "
        f"stable completed output is untouched either way."
    )


def _legacy_split_partial_remedy(recovery_path: Path, rel_paths: tuple[str, ...]) -> str:
    """The D11 preserve-or-move-aside remedy for a version-1 (predecessor
    ordered-prefix) split partial.

    This build never resumes or executes a version-1 split partial: it is
    migration-readable only.  The message never suggests silently discarding
    paid work; deletion is offered only as an explicit, named choice.
    """
    named = ", ".join(f"'{rel_path}'" for rel_path in rel_paths)
    return (
        f"'{recovery_path.name}' in the output directory contains predecessor "
        f"(schema version 1) split checkpoint data for {named} that this build "
        "does not resume or execute. To finish that in-progress split work, "
        "re-run the exact predecessor CodeDoc build that created it. To "
        f"continue with this build, move '{recovery_path.name}' aside (for "
        "example, rename it) before re-running so a fresh recovery file is "
        "created; you may instead delete it, but that explicitly discards "
        "those predecessor split checkpoints. Your prior stable completed "
        "output is untouched either way."
    )


def load_recovery_records_if_compatible(
    recovery_path: Path,
    identity: dict,
    *,
    include_partial_files: bool = False,
) -> RecoveryState:
    """Return one frozen compatible recovery state, else an empty state.

    Performs only the exact-path ownership/status/identity validation:

    - a missing file yields ``{}`` (a fresh recovery state);
    - a foreign/malformed file, a completed (non-``in_progress``) document, an
      unknown/unsupported identity ``version``, or an owned in-progress file whose
      identity does not match the current run **blocks** with an actionable
      :class:`ConfigError` — it is never renamed, relocated, or silently deleted.

    It does not select alternate names, read stable outputs, delete files, or
    mutate ``SafeWriter``.  Partial loading is disabled by default; only an
    explicit split-strategy caller enables it, so split-only state is neither
    normalized nor returned across the default/truncate strategy boundary.
    """
    if not recovery_path.exists():
        return RecoveryState()
    try:
        document = read_codedoc_document(
            recovery_path,
            include_partial_files=include_partial_files,
        )
    except (ConfigError, FileNotFoundError) as exc:
        raise ConfigError(
            f"'{recovery_path.name}' in the output directory is not a readable "
            f"CodeDoc recovery file ({exc}). {_recovery_remedy(recovery_path)}"
        ) from exc
    if not document.in_progress:
        raise ConfigError(
            f"'{recovery_path.name}' in the output directory is a completed "
            "document, not an in-progress recovery file. "
            f"{_recovery_remedy(recovery_path)}"
        )
    stored = document.metadata.get("recovery_identity")
    if not isinstance(stored, dict):
        raise ConfigError(
            f"'{recovery_path.name}' carries no recovery identity and cannot be "
            f"safely resumed. {_recovery_remedy(recovery_path)}"
        )
    if stored.get("version") != _RECOVERY_IDENTITY_VERSION:
        raise ConfigError(
            f"'{recovery_path.name}' has an unsupported recovery identity version "
            f"{stored.get('version')!r} (this build supports "
            f"{_RECOVERY_IDENTITY_VERSION}). {_recovery_remedy(recovery_path)}"
        )
    mismatch = _first_identity_mismatch(stored, identity)
    if mismatch is not None:
        field, expected, found = mismatch
        raise ConfigError(
            f"'{recovery_path.name}' in the output directory was written for a "
            f"different run: {field} expected {expected!r} but the recovery file "
            f"has {found!r}. {_recovery_remedy(recovery_path)}"
        )
    if document.legacy_split_partial_rels:
        raise ConfigError(_legacy_split_partial_remedy(recovery_path, document.legacy_split_partial_rels))
    records = records_by_path(document)
    return RecoveryState(
        records=tuple(
            (rel_path, canonical_json(records[rel_path]))
            for rel_path in sorted(records)
        ),
        partial_files=tuple(document.partial_files),
    )


# ---------------------------------------------------------------------------
# Record reconstruction
# ---------------------------------------------------------------------------

def _public_record_to_doc(file_record: dict) -> dict:
    """Convert a public JSON file record back to a documentation dict."""
    links = file_record.get("links", {})
    deps = file_record.get("_deps") or {
        "external": links.get("external_dependencies", []),
    }
    doc = {
        "file_path": file_record.get("path", ""),
        "language": file_record.get("language", ""),
        "imports": file_record.get("imports", []),
        "description": file_record.get("description", ""),
        "role_in_system": file_record.get("role_in_system", ""),
        "key_concepts": file_record.get("key_concepts", []),
        "functions": file_record.get("functions", []),
        "classes": file_record.get("classes", []),
        "exports": file_record.get("exports", []),
        "usage_example": file_record.get("usage_example", ""),
        "dependencies_analysis": deps,
    }
    # A predecessor record's `division`/`documentation_units` fields (D11) are
    # never reconstructed here: they are discarded on read and never carried
    # into new JSON, Markdown, embedded views, or reverse conversion.
    cleaned = {k: v for k, v in doc.items() if v not in (None, "", [], {}, {"external": [], "internal": []})}
    # Carry registered private keys from the public record into the
    # reconstructed flat documentation result so they survive resume/reuse.
    carry_private_keys(file_record, cleaned)
    return cleaned


def _build_documentation_records(
    rel_paths: set,
    content_hashes: dict,
    ordered_paths: list,
    existing_docs: dict,
    new_results: dict,
) -> list[dict]:
    """Build documentation records for write_project_outputs."""
    records = []
    for rel_path in ordered_paths:
        if rel_path not in rel_paths:
            continue

        if rel_path in new_results:
            result = new_results[rel_path]
            if isinstance(result, dict) and result.get("path") and not result.get("file_path"):
                doc = _public_record_to_doc(result)
            else:
                doc = dict(result)
        elif rel_path in existing_docs:
            doc = _public_record_to_doc(existing_docs[rel_path])
        else:
            continue

        if rel_path in new_results:
            file_hash = content_hashes.get(rel_path)
            if not file_hash:
                raise ValueError(
                    f"new documentation result for {rel_path!r} has no frozen "
                    "planning-snapshot hash."
                )
        else:
            file_hash = existing_docs.get(rel_path, {}).get("hash", "")

        records.append({
            "hash": file_hash,
            "file_path": rel_path,
            "language": doc.get("language", ""),
            "documentation": doc,
        })
    return records
