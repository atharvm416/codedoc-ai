"""Centralized read-only parser for CodeDoc JSON / Markdown documents.

This module owns CodeDoc document parsing and structural ownership validation.
It is strictly read-only: it never writes, migrates, renames, or deletes.
Source priority and missing/malformed *policy* remain with callers — this
module only decides whether a given file is a structurally valid CodeDoc
document and, if so, returns its normalized contents.

Acceptance (encoded explicitly):

- current completed JSON with strict ``last_run`` / ``files`` shape;
- legacy completed JSON with valid ``_codedoc`` metadata and schema ``1.3`` or
  ``1.4``;
- in-progress crash-recovery JSON (crash banner, ``status=in_progress``, or
  ``live_backup``) with schema ``1.4`` or a missing schema;
- stale-build migration JSON with valid ``_codedoc`` ownership and schema
  ``1.3`` / ``1.4`` / missing — only when ``legacy_role="stale_build"``;
- Markdown metadata schema ``1.3`` or ``1.4``;
- current Markdown with a valid embedded lossless view;
- legacy Markdown without an embedded view (visible parser).

Versionless completed documents must satisfy stricter current or legacy
ownership shape checks. Unknown newer schemas, malformed versions, and
unsupported extensions fail
closed (``ConfigError``).  Missing files raise ``FileNotFoundError`` — callers
decide whether "missing" is acceptable.

Document envelope contract:

- current completed JSON is recognized by its strict versionless public shape:
  ``last_run`` plus ``files`` with an ``entry_file`` in ``last_run``;
- legacy completed JSON with top-level ``_codedoc`` / ``project`` / ``run`` is
  still accepted for compatibility;
- in-progress crash-recovery JSON still carries ``_codedoc`` because recovery
  identity is internal state, not the completed public contract;
- wrappers such as ``_codedoc`` and ``_crash_safety`` are removed from the public
  view returned by this reader.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from codedoc.core.markdown_view import (
    _CODEDOC_META_COMMENT_RE,
    markdown_to_view,
    read_embedded_view_result,
)
from codedoc.core.project_view import SCHEMA_VERSION
from codedoc.utils.errors import ConfigError

# Accepted schema versions, parsed as integer components rather than floats.
# Importing SCHEMA_VERSION keeps this synchronized with its production owner.
_CURRENT_SCHEMA_COMPONENTS = tuple(int(part) for part in SCHEMA_VERSION.split("."))
_ACCEPTED_SCHEMAS: frozenset[tuple[int, ...]] = frozenset({
    (1, 3),
    _CURRENT_SCHEMA_COMPONENTS,
})
_VALID_LEGACY_ROLES = (None, "stale_build")
_LAST_RUN_ENTRY_SOURCES = {"explicit", "recovered", "auto-detected", "none"}
_LAST_RUN_DOCUMENTATION_SCOPES = {"entry", "all"}
_LAST_RUN_ANALYSIS_MODES = {"single", "triple"}
_LAST_RUN_INTEGER_FIELDS = {
    "files_scanned",
    "files_selected",
    "files_documented_by_llm",
    "files_failed",
    "files_unattempted",
    "files_reused_unchanged",
    "files_reused_identical_content",
    "files_resumed_from_recovery",
}


@dataclass(frozen=True)
class CodedocDocument:
    """Normalized, defensively-copied contents of a CodeDoc document."""

    path: Path
    format: str  # "json" or "md"
    metadata: dict
    schema_version: str | None
    entry_file: str | None
    file_hashes: dict[str, str]
    files: tuple[dict, ...]
    in_progress: bool
    view: dict


def read_codedoc_document(
    path: Path,
    *,
    legacy_role: str | None = None,
) -> CodedocDocument:
    """Parse and validate a CodeDoc document at *path* (read-only).

    ``legacy_role`` must be ``None`` or ``"stale_build"``.  Only the stale-build
    migration call site passes ``"stale_build"``; any other value is rejected.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ConfigError
        When the file is foreign, malformed, has an unsupported extension, or
        carries an unknown / invalid schema version.
    """
    if legacy_role not in _VALID_LEGACY_ROLES:
        raise ValueError(f"Unknown legacy_role: {legacy_role!r}")

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Cannot read '{path}': {exc}") from exc

    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"'{path}' is not valid UTF-8 and cannot be read as a CodeDoc file."
        ) from exc

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json(path, content, legacy_role)
    if suffix == ".md":
        return _read_markdown(path, content)

    raise ConfigError(
        f"'{path}' has unsupported extension '{suffix}'. "
        "CodeDoc output files must end in .json or .md."
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _read_json(path: Path, content: str, legacy_role: str | None) -> CodedocDocument:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"'{path}' is not a valid CodeDoc file: JSON parse error."
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"'{path}' does not appear to be a valid CodeDoc file (not a JSON object)."
        )

    meta_obj = data.get("_codedoc")
    if meta_obj is not None and not isinstance(meta_obj, dict):
        raise ConfigError(
            f"'{path}' does not appear to be a valid CodeDoc file. "
            "The '_codedoc' metadata block is malformed."
        )
    meta = meta_obj if isinstance(meta_obj, dict) else {}
    if meta_obj is not None:
        _validate_metadata(path, meta)

    in_progress = (
        "_crash_safety" in data
        or meta.get("status") == "in_progress"
        or meta.get("live_backup") is True
    )
    if in_progress and meta_obj is None:
        raise ConfigError(
            f"'{path}' is an incomplete CodeDoc recovery file but is missing "
            "the internal '_codedoc' recovery metadata."
        )

    meta_schema = meta.get("schema_version")
    view_schema = data.get("schema_version")
    # When both the metadata block and the public view declare a schema they
    # must agree.
    if (
        meta_schema is not None
        and view_schema is not None
        and str(meta_schema) != str(view_schema)
    ):
        raise ConfigError(
            f"'{path}' has conflicting schema versions "
            f"('{meta_schema}' in _codedoc vs '{view_schema}' in the view)."
        )
    schema = meta_schema if meta_schema is not None else view_schema

    _validate_json_schema(path, schema, in_progress=in_progress, legacy_role=legacy_role)

    entry_file = _entry_file_from_view(data, meta)
    if meta_obj is None:
        meta = {"entry_file": entry_file}
    files = _extract_files(path, data.get("files"))
    if not in_progress and legacy_role != "stale_build":
        if meta_obj is None:
            _validate_versionless_completed_view(
                path, data, meta, files, allow_legacy=False
            )
        elif schema is None:
            _validate_versionless_completed_view(path, data, meta, files)

    file_hashes = {
        f["path"]: f.get("hash", "")
        for f in files
        if isinstance(f, dict) and f.get("path")
    }

    # The public view is the payload without the ownership / crash wrappers.
    view = {
        k: copy.deepcopy(v)
        for k, v in data.items()
        if k not in ("_codedoc", "_crash_safety")
    }

    return CodedocDocument(
        path=path,
        format="json",
        metadata=copy.deepcopy(meta),
        schema_version=str(schema) if schema is not None else None,
        entry_file=entry_file,
        file_hashes=file_hashes,
        files=tuple(files),
        in_progress=in_progress,
        view=view,
    )


def _validate_json_schema(
    path: Path,
    schema,
    *,
    in_progress: bool,
    legacy_role: str | None,
) -> None:
    if schema is None:
        # Recovery/stale-build envelopes are accepted directly. Completed
        # current documents receive strict structural validation in the caller.
        return

    components = _parse_schema_components(schema)
    if components is None:
        raise ConfigError(f"'{path}' has a malformed schema version '{schema}'.")

    if components in _ACCEPTED_SCHEMAS:
        return

    raise ConfigError(
        f"'{path}' has an unsupported CodeDoc schema version '{schema}'. "
        "This build accepts CodeDoc schema 1.3 and 1.4."
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _read_markdown(path: Path, content: str) -> CodedocDocument:
    # Lightweight metadata comment (authoritative for incremental hashes).
    lightweight_meta: dict | None = None
    meta_match = _CODEDOC_META_COMMENT_RE.search(content)
    if meta_match:
        try:
            parsed = json.loads(meta_match.group(1))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"'{path}' has a malformed CodeDoc metadata comment."
            ) from exc
        if not isinstance(parsed, dict):
            raise ConfigError(
                f"'{path}' has a malformed CodeDoc metadata comment "
                "(expected a JSON object)."
            )
        lightweight_meta = parsed
        _validate_metadata(path, lightweight_meta, markdown=True)
    elif "<!-- codedoc-ai:" in content:
        raise ConfigError(
            f"'{path}' has a malformed CodeDoc metadata comment."
        )

    # Embedded lossless view — tri-state.  A present-but-corrupt block means the
    # document is malformed; we must not silently trust visible prose.
    embedded = read_embedded_view_result(content)
    if embedded.state == "invalid":
        raise ConfigError(
            f"'{path}' contains a corrupt embedded CodeDoc view and cannot be "
            "trusted for ownership or resume."
        )

    if lightweight_meta is None and embedded.state != "valid":
        # No valid CodeDoc marker of any kind — foreign (this intentionally
        # tightens the old prefix-only match: malformed marker-only Markdown is
        # foreign and must not be overwritten).
        raise ConfigError(
            f"'{path}' does not appear to be a valid CodeDoc file. "
            "The metadata comment (<!-- codedoc-ai: ... -->) is missing or malformed."
        )

    emb_view = embedded.view if embedded.state == "valid" else None
    emb_project = emb_view.get("project") if isinstance(emb_view, dict) else None
    if emb_project is not None and not isinstance(emb_project, dict):
        raise ConfigError(
            f"'{path}' contains a malformed embedded CodeDoc view: "
            "'project' must be an object."
        )

    lw_schema = lightweight_meta.get("schema_version") if lightweight_meta else None
    emb_schema = emb_view.get("schema_version") if emb_view else None
    lw_entry = lightweight_meta.get("entry_file") if lightweight_meta else None
    emb_last_run = emb_view.get("last_run") if isinstance(emb_view, dict) else None
    emb_entry = None
    if isinstance(emb_last_run, dict):
        emb_entry = emb_last_run.get("entry_file")
    if emb_entry is None and emb_project is not None:
        emb_entry = emb_project.get("entry_file")

    if lightweight_meta is not None and emb_view is not None:
        if (
            lw_schema is not None
            and emb_schema is not None
            and str(lw_schema) != str(emb_schema)
        ):
            raise ConfigError(
                f"'{path}' has conflicting schema versions between its metadata "
                f"comment ('{lw_schema}') and embedded view ('{emb_schema}')."
            )
        if lw_entry is not None and emb_entry is not None and lw_entry != emb_entry:
            raise ConfigError(
                f"'{path}' has conflicting entry files between its metadata "
                f"comment ('{lw_entry}') and embedded view ('{emb_entry}')."
            )

    schema = lw_schema if lw_schema is not None else emb_schema
    _validate_markdown_schema(path, schema)
    if schema is None:
        if lightweight_meta is None or emb_view is None:
            raise ConfigError(
                f"'{path}' is missing the complete versionless CodeDoc ownership "
                "envelope (metadata comment plus embedded project view)."
            )
        embedded_files = _extract_files(path, emb_view.get("files"))
        _validate_versionless_completed_view(
            path, emb_view, lightweight_meta, embedded_files
        )

    # Prefer the valid embedded view; fall back to the visible parser only when
    # the embedded block is absent.
    if emb_view is not None:
        view = copy.deepcopy(emb_view)
    else:
        view = markdown_to_view(content)

    files = _extract_files(path, view.get("files"))

    # Lightweight metadata file_hashes are authoritative; embedded / per-file
    # hashes are only a fallback.
    lw_hashes = lightweight_meta.get("file_hashes") if lightweight_meta else None
    file_hashes: dict[str, str] = dict(lw_hashes) if isinstance(lw_hashes, dict) else {}
    for f in files:
        p = f.get("path")
        if p and p not in file_hashes:
            h = f.get("hash", "")
            if h:
                file_hashes[p] = h

    entry_file = lw_entry if lightweight_meta is not None else emb_entry
    metadata = (
        copy.deepcopy(lightweight_meta)
        if lightweight_meta is not None
        else {
            "entry_file": emb_entry,
            "file_hashes": dict(file_hashes),
        }
    )

    return CodedocDocument(
        path=path,
        format="md",
        metadata=metadata,
        schema_version=str(schema) if schema is not None else None,
        entry_file=entry_file,
        file_hashes=file_hashes,
        files=tuple(files),
        in_progress=False,
        view=view,
    )


def _validate_markdown_schema(path: Path, schema) -> None:
    if schema is None:
        return
    components = _parse_schema_components(schema)
    if components is None:
        raise ConfigError(f"'{path}' has a malformed schema version '{schema}'.")
    if components in _ACCEPTED_SCHEMAS:
        return
    raise ConfigError(
        f"'{path}' has an unsupported CodeDoc schema version '{schema}'. "
        "This build accepts CodeDoc schema 1.3 and 1.4."
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_schema_components(schema) -> tuple[int, ...] | None:
    """Parse a schema string into integer components, or None if malformed."""
    try:
        parts = str(schema).split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return None


def _validate_metadata(
    path: Path,
    metadata: dict,
    *,
    markdown: bool = False,
) -> None:
    entry_file = metadata.get("entry_file")
    if entry_file is not None and not isinstance(entry_file, str):
        raise ConfigError(
            f"'{path}' is malformed: '_codedoc.entry_file' must be a string or null."
        )
    if markdown and "file_hashes" in metadata:
        hashes = metadata["file_hashes"]
        if not isinstance(hashes, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in hashes.items()
        ):
            raise ConfigError(
                f"'{path}' is malformed: Markdown 'file_hashes' must map "
                "string paths to string hashes."
            )


def _extract_files(path: Path, files_obj) -> list[dict]:
    """Validate and defensively copy the per-file collection.

    Rejects a non-list ``files`` value, non-dict records, and duplicate
    non-empty file paths.
    """
    if files_obj is None:
        return []
    if not isinstance(files_obj, list):
        raise ConfigError(
            f"'{path}' is malformed: 'files' must be a list, got "
            f"{type(files_obj).__name__}."
        )
    seen: set[str] = set()
    out: list[dict] = []
    for record in files_obj:
        if not isinstance(record, dict):
            raise ConfigError(
                f"'{path}' is malformed: each file record must be an object."
            )
        rec_path = record.get("path")
        if rec_path:
            if rec_path in seen:
                raise ConfigError(
                    f"'{path}' is malformed: duplicate file path '{rec_path}'."
                )
            seen.add(rec_path)
        out.append(copy.deepcopy(record))
    return out


def _entry_file_from_view(data: dict, metadata: dict) -> str | None:
    """Return entry identity from new, legacy, or metadata-backed documents."""
    if isinstance(data.get("last_run"), dict):
        entry = data["last_run"].get("entry_file")
        if entry is not None:
            return entry
    if isinstance(data.get("project"), dict):
        entry = data["project"].get("entry_file")
        if entry is not None:
            return entry
    return metadata.get("entry_file")


def _validate_versionless_completed_view(
    path: Path,
    data: dict,
    metadata: dict,
    files: list[dict],
    *,
    allow_legacy: bool = True,
) -> None:
    """Require a recognized versionless completed CodeDoc output shape."""
    project = data.get("project")
    run = data.get("run")
    last_run = data.get("last_run")
    if "last_run" in data:
        _validate_last_run(path, last_run)

    if isinstance(project, dict) or isinstance(run, dict):
        if not allow_legacy:
            raise ConfigError(
                f"'{path}' is not a valid current CodeDoc output: legacy "
                "'project'/'run' blocks require legacy ownership metadata."
            )
        _validate_legacy_project_run(path, project, run, metadata, files)
        return

    if not isinstance(last_run, dict):
        raise ConfigError(
            f"'{path}' is not a valid versionless CodeDoc output: canonical "
            "'last_run' object is required."
        )

    if "entry_file" not in last_run:
        raise ConfigError(
            f"'{path}' is not a valid versionless CodeDoc output: "
            "'last_run.entry_file' is required."
        )
    entry = last_run.get("entry_file")
    if entry is not None and not isinstance(entry, str):
        raise ConfigError(f"'{path}' has a malformed last_run entry_file.")
    if metadata.get("entry_file") != entry:
        raise ConfigError(
            f"'{path}' has contradictory versionless ownership metadata and "
            "last_run entry file."
        )
    if len(files) > last_run["files_selected"]:
        raise ConfigError(f"'{path}' has contradictory versionless file counts.")


def _validate_legacy_project_run(
    path: Path,
    project: object,
    run: object,
    metadata: dict,
    files: list[dict],
) -> None:
    """Validate the legacy project/run envelope."""
    if not isinstance(project, dict) or not isinstance(run, dict):
        raise ConfigError(
            f"'{path}' is not a valid versionless CodeDoc output: legacy "
            "'project' and 'run' objects must either both be present or both "
            "be absent."
        )
    required_project = {"entry_file", "file_count", "languages", "folders"}
    required_run = {
        "files_checked",
        "files_failed",
        "files_skipped",
        "files_reused",
        "files_documented",
    }
    if not required_project.issubset(project) or not required_run.issubset(run):
        raise ConfigError(
            f"'{path}' is not a valid versionless CodeDoc output: the canonical "
            "project/run fields are incomplete."
        )

    entry = project.get("entry_file")
    file_count = project.get("file_count")
    if entry is not None and not isinstance(entry, str):
        raise ConfigError(f"'{path}' has a malformed project entry file.")
    if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count < 0:
        raise ConfigError(f"'{path}' has a malformed project file count.")
    for field in ("languages", "folders"):
        value = project.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ConfigError(f"'{path}' has a malformed project {field} list.")
    for field in required_run:
        value = run.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"'{path}' has a malformed run field '{field}'.")

    if "entry_file" not in metadata or metadata.get("entry_file") != entry:
        raise ConfigError(
            f"'{path}' has contradictory versionless ownership metadata and "
            "project entry file."
        )
    if file_count != len(files) or run.get("files_documented") != len(files):
        raise ConfigError(f"'{path}' has contradictory versionless file counts.")


def _validate_last_run(path: Path, last_run: object) -> None:
    if not isinstance(last_run, dict):
        raise ConfigError(f"'{path}' has a malformed last_run block.")

    required = {
        "entry_source",
        "documentation_scope",
        "analysis_mode",
        *_LAST_RUN_INTEGER_FIELDS,
    }
    missing = required - set(last_run)
    if missing:
        raise ConfigError(
            f"'{path}' has an incomplete last_run block: missing {sorted(missing)}."
        )

    if last_run.get("entry_source") not in _LAST_RUN_ENTRY_SOURCES:
        raise ConfigError(f"'{path}' has a malformed last_run entry_source.")
    if last_run.get("documentation_scope") not in _LAST_RUN_DOCUMENTATION_SCOPES:
        raise ConfigError(f"'{path}' has a malformed last_run documentation_scope.")
    if last_run.get("analysis_mode") not in _LAST_RUN_ANALYSIS_MODES:
        raise ConfigError(f"'{path}' has a malformed last_run analysis_mode.")
    if "entry_file" in last_run and last_run.get("entry_file") is not None and not isinstance(
        last_run.get("entry_file"), str
    ):
        raise ConfigError(f"'{path}' has a malformed last_run entry_file.")

    for field in _LAST_RUN_INTEGER_FIELDS:
        value = last_run.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"'{path}' has a malformed last_run field '{field}'.")

    partition_total = (
        last_run["files_reused_unchanged"]
        + last_run["files_reused_identical_content"]
        + last_run["files_documented_by_llm"]
        + last_run["files_failed"]
        + last_run["files_unattempted"]
    )
    if last_run["files_selected"] != partition_total:
        raise ConfigError(f"'{path}' has contradictory last_run file counts.")


def records_by_path(document: CodedocDocument) -> dict[str, dict]:
    """Return a fresh ``{path: record}`` map from a parsed document.

    Each call returns independent copies — callers must not share one cached
    parsed object across mutation paths.
    """
    return {
        copy.deepcopy(f["path"]): copy.deepcopy(f)
        for f in document.files
        if isinstance(f, dict) and f.get("path")
    }
