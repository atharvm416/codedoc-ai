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

from codedoc.core.file_division import (
    SPLIT_PARTIAL_SCHEMA_VERSION,
    DuplicateCanonicalKeyError,
    QuarantineEntry,
    ReductionNodeState,
    SplitTreeState,
    canonical_json,
    is_legacy_split_partial,
    load_canonical_json_object,
    node_state_public_json,
)
from codedoc.core.markdown_view import (
    _CODEDOC_META_COMMENT_RE,
    markdown_to_view,
    read_embedded_view_result,
)
from codedoc.core.project_view import SCHEMA_VERSION, sanitize_public_view
from codedoc.parser.source_structure import normalize_rel_path
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
_LAST_RUN_OPTIONAL_INTEGER_FIELDS = {
    "files_skipped_insufficient_source",
}
_LAST_RUN_SPLIT_INTEGER_FIELDS = {
    "split_ordinary_files",
    "split_syntax_files",
    "split_lexical_files",
    "split_blocked_files",
    "split_divided_files",
    "split_units",
    "split_chunks",
    "split_continuation_groups",
    "split_unit_consolidation_levels",
    "split_unit_consolidation_calls_planned",
    "split_general_reduction_levels",
    "split_general_reduction_calls_planned",
    "split_final_synthesis_calls_planned",
    "split_restored_complete_chunks",
    "split_restored_unit_consolidation_calls",
    "split_restored_general_reduction_calls",
    "split_restored_final_synthesis_calls",
    "file_documentation_calls_planned",
    "unit_documentation_calls_planned",
    "file_reduction_calls_planned",
    "synthesis_calls_planned",
}
_LAST_RUN_SPLIT_OPTIONAL_INTEGER_FIELDS = {
    "split_completed_files_reused",
    "split_partial_files_resumed",
    "split_unpaid_nodes",
    "split_reexecuted_nodes",
    "split_quarantined_nodes",
    "split_recovery_conflict_files",
}
# The frozen capacity-block reasons (design decision D6). There is no
# "capacity fallback": a blocked file has no provider action at all (D8).
_BLOCKED_REASONS = {
    "atom-cap",
    "symbol-cap",
    "unit-cap",
    "chunk-cap",
    "reduction-envelope-cap",
    "reduction-fan-in-cap",
    "reduction-depth-cap",
    "final-synthesis-envelope-cap",
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
    partial_files: tuple[SplitTreeState, ...] = ()
    # rel_paths whose stored split partial structurally matches the
    # predecessor (schema version 1) ordered-prefix payload.  Migration-
    # readable only: D11 requires callers to fail closed with the
    # preserve-or-move-aside remedy rather than resume or execute these.
    legacy_split_partial_rels: tuple[str, ...] = ()


def read_codedoc_document(
    path: Path,
    *,
    legacy_role: str | None = None,
    include_partial_files: bool = True,
) -> CodedocDocument:
    """Parse and validate a CodeDoc document at *path* (read-only).

    ``legacy_role`` must be ``None`` or ``"stale_build"``.  Only the stale-build
    migration call site passes ``"stale_build"``; any other value is rejected.
    ``include_partial_files=False`` leaves split-only partial recovery metadata
    out of the normalized document while retaining the ordinary recovery
    ownership and completed-record envelope.

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
        return _read_json(
            path,
            content,
            legacy_role,
            include_partial_files=include_partial_files,
        )
    if suffix == ".md":
        return _read_markdown(path, content)

    raise ConfigError(
        f"'{path}' has unsupported extension '{suffix}'. "
        "CodeDoc output files must end in .json or .md."
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _read_json(
    path: Path,
    content: str,
    legacy_role: str | None,
    *,
    include_partial_files: bool,
) -> CodedocDocument:
    try:
        # Duplicate-aware: a duplicate key anywhere in the document (recovery
        # or completed) fails closed rather than silently keeping the last
        # value, matching the split-partial container's own strictness
        # (section 9/14). This does not require canonical (sorted/compact)
        # form -- the recovery file is written pretty-printed -- so the
        # underlying pairs hook is reused directly rather than the stricter
        # `load_canonical_json_object`.
        data = load_canonical_json_object(content, require_canonical=False)
    except DuplicateCanonicalKeyError as exc:
        raise ConfigError(
            f"'{path}' is not a valid CodeDoc file: duplicate JSON key {exc}."
        ) from exc
    except ValueError as exc:
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
    if not in_progress:
        view = sanitize_public_view(view)
        files = _extract_files(path, view.get("files"))

    metadata = copy.deepcopy(meta)
    if not include_partial_files:
        metadata.pop("partial_files", None)

    partial_files: tuple[SplitTreeState, ...] = ()
    legacy_split_partial_rels: tuple[str, ...] = ()
    if include_partial_files:
        partial_files, legacy_split_partial_rels = _partial_files_from_meta(meta)

    return CodedocDocument(
        path=path,
        format="json",
        metadata=metadata,
        schema_version=str(schema) if schema is not None else None,
        entry_file=entry_file,
        file_hashes=file_hashes,
        files=tuple(files),
        in_progress=in_progress,
        view=view,
        partial_files=partial_files,
        legacy_split_partial_rels=legacy_split_partial_rels,
    )


def _partial_files_from_meta(
    meta: dict,
) -> tuple[tuple[SplitTreeState, ...], tuple[str, ...]]:
    """Parse `partial_files` into `(schema-4 tree states, legacy v1 rel_paths)`.

    A predecessor (schema version 1) ordered-prefix payload is
    migration-readable only (D11): it is recognized structurally so the
    caller can fail closed with the preserve-or-move-aside remedy, but it is
    never parsed into a usable `SplitTreeState` and never silently dropped.
    A dormant schema-2 (or any other unsupported) container fails the exact
    ``schema_version`` check below and is likewise never migrated or
    executed (section 16) -- the caller's ConfigError is the preserve-first
    block.
    """
    if "partial_files" not in meta:
        return (), ()
    raw = meta["partial_files"]
    if not isinstance(raw, dict):
        raise ConfigError("CodeDoc recovery partial_files must be an object.")
    partials: list[SplitTreeState] = []
    legacy_rel_paths: list[str] = []
    normalized_rel_paths: set[str] = set()
    container_required = {
        "schema_version",
        "owner",
        "rel_path",
        "content_hash",
        "division_plan_digest",
        "reduction_tree_digest",
        "nodes",
    }
    container_allowed = container_required | {"quarantine"}
    node_required = {
        "node_id",
        "node_type",
        "rel_path",
        "content_hash",
        "division_plan_digest",
        "execution_identity_digest",
        "input_digest",
        "unit_id",
        "child_ids",
        "coverage_leaf_ids",
        "result_json",
    }
    quarantine_required = {"node_id", "reason", "raw_json"}

    for rel_path in sorted(raw):
        value = raw[rel_path]
        if not isinstance(value, dict):
            raise ConfigError("CodeDoc recovery contains a malformed split-partial container.")
        try:
            normalized_rel_path = normalize_rel_path(rel_path)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "CodeDoc recovery contains a split-partial container with an invalid path."
            ) from exc
        if normalized_rel_path in normalized_rel_paths:
            raise ConfigError(
                "CodeDoc recovery contains split-partial containers with aliased paths."
            )
        normalized_rel_paths.add(normalized_rel_path)
        if is_legacy_split_partial(value):
            legacy_rel_paths.append(normalized_rel_path)
            continue
        if value.get("schema_version") != SPLIT_PARTIAL_SCHEMA_VERSION:
            raise ConfigError(
                "CodeDoc recovery contains an unsupported split-partial schema version."
            )
        if set(value) - container_allowed or container_required - set(value):
            raise ConfigError(
                "CodeDoc recovery contains a split-partial container with an unknown "
                "or missing field."
            )
        raw_nodes = value.get("nodes")
        if not isinstance(raw_nodes, list):
            raise ConfigError("CodeDoc recovery contains a malformed split-partial container.")
        nodes: list[ReductionNodeState] = []
        nodes_quarantine: list[QuarantineEntry] = []
        seen_node_ids: set[str] = set()
        for node_value in raw_nodes:
            if not isinstance(node_value, dict):
                raise ConfigError("CodeDoc recovery contains a malformed split-partial node.")
            node_id = node_value.get("node_id")
            if not isinstance(node_id, str) or not node_id:
                raise ConfigError(
                    "CodeDoc recovery contains a split-partial node without an ID."
                )
            if node_id in seen_node_ids:
                raise ConfigError(
                    "CodeDoc recovery contains a duplicate split-partial node ID."
                )
            seen_node_ids.add(node_id)
            raw_node_json = canonical_json(node_value)
            if set(node_value) - node_required or node_required - set(node_value):
                # Strictly reject this object as executable while preserving a
                # bounded copy beside valid siblings (section 9/14). A
                # per-node reduction_tree_digest follows this exact path.
                nodes_quarantine.append(
                    QuarantineEntry(
                        node_id=node_id,
                        reason="live-schema-mismatch",
                        raw_json=raw_node_json,
                    )
                )
                continue
            try:
                node = ReductionNodeState(
                    node_id=node_id,
                    node_type=node_value.get("node_type"),
                    rel_path=node_value.get("rel_path"),
                    content_hash=node_value.get("content_hash", ""),
                    division_plan_digest=node_value.get("division_plan_digest", ""),
                    execution_identity_digest=node_value.get("execution_identity_digest", ""),
                    input_digest=node_value.get("input_digest", ""),
                    unit_id=node_value.get("unit_id"),
                    child_ids=tuple(node_value.get("child_ids", ())),
                    coverage_leaf_ids=tuple(node_value.get("coverage_leaf_ids", ())),
                    result_json=node_value.get("result_json", ""),
                )
            except (TypeError, ValueError):
                nodes_quarantine.append(
                    QuarantineEntry(
                        node_id=node_id,
                        reason="live-schema-mismatch",
                        raw_json=raw_node_json,
                    )
                )
                continue
            nodes.append(node)

        raw_quarantine = value.get("quarantine", [])
        if not isinstance(raw_quarantine, list):
            raise ConfigError(
                "CodeDoc recovery contains a malformed split-partial quarantine map."
            )
        quarantine: list[QuarantineEntry] = []
        for entry_value in raw_quarantine:
            if not isinstance(entry_value, dict):
                raise ConfigError("CodeDoc recovery contains a malformed quarantine entry.")
            if set(entry_value) != quarantine_required:
                raise ConfigError(
                    "CodeDoc recovery contains a quarantine entry with an unknown "
                    "or missing field."
                )
            try:
                quarantine.append(
                    QuarantineEntry(
                        node_id=entry_value.get("node_id"),
                        reason=entry_value.get("reason"),
                        raw_json=entry_value.get("raw_json", ""),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "CodeDoc recovery contains a malformed quarantine entry."
                ) from exc

        quarantine.extend(nodes_quarantine)

        try:
            partial_header = SplitTreeState(
                schema_version=value.get("schema_version"),
                owner=value.get("owner"),
                rel_path=value.get("rel_path"),
                content_hash=value.get("content_hash", ""),
                division_plan_digest=value.get("division_plan_digest", ""),
                reduction_tree_digest=value.get("reduction_tree_digest", ""),
                nodes=(),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "CodeDoc recovery contains a malformed split-partial container."
            ) from exc
        if partial_header.rel_path != normalized_rel_path:
            raise ConfigError(
                "CodeDoc recovery contains a split-partial container with a mismatched path."
            )
        matching_nodes_list: list[ReductionNodeState] = []
        for node in nodes:
            if (
                node.rel_path == partial_header.rel_path
                and node.content_hash == partial_header.content_hash
                and node.division_plan_digest == partial_header.division_plan_digest
            ):
                matching_nodes_list.append(node)
            else:
                quarantine.append(
                    QuarantineEntry(
                        node_id=node.node_id,
                        reason="stale-identity",
                        raw_json=canonical_json(node_state_public_json(node)),
                    )
                )
        matching_nodes = tuple(matching_nodes_list)
        try:
            partial = SplitTreeState(
                schema_version=partial_header.schema_version,
                owner=partial_header.owner,
                rel_path=partial_header.rel_path,
                content_hash=partial_header.content_hash,
                division_plan_digest=partial_header.division_plan_digest,
                reduction_tree_digest=partial_header.reduction_tree_digest,
                nodes=matching_nodes,
                quarantine=tuple(quarantine),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "CodeDoc recovery contains a malformed split-partial container."
            ) from exc
        partials.append(partial)
    return tuple(partials), tuple(legacy_rel_paths)


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
    emb_schema = embedded.source_schema_version
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
    view = sanitize_public_view(view)

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
    for field in _LAST_RUN_OPTIONAL_INTEGER_FIELDS:
        if field not in last_run:
            continue
        value = last_run[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"'{path}' has a malformed last_run field '{field}'.")

    split_strategy = last_run.get("large_file_strategy")
    present_split_fields = (
        _LAST_RUN_SPLIT_INTEGER_FIELDS
        | _LAST_RUN_SPLIT_OPTIONAL_INTEGER_FIELDS
    ) & set(last_run)
    if split_strategy is None and present_split_fields:
        raise ConfigError(f"'{path}' has split statistics without split strategy.")
    if split_strategy is not None:
        if split_strategy != "split":
            raise ConfigError(f"'{path}' has a malformed large_file_strategy.")
        # Split is valid only in single mode (D2); a document claiming split
        # statistics under triple mode is internally contradictory.
        if last_run.get("analysis_mode") != "single":
            raise ConfigError(
                f"'{path}' has split statistics with a non-single analysis_mode."
            )
        missing_split_fields = _LAST_RUN_SPLIT_INTEGER_FIELDS - set(last_run)
        if missing_split_fields:
            raise ConfigError(
                f"'{path}' has incomplete split statistics: "
                f"missing {sorted(missing_split_fields)}."
            )
        for field in _LAST_RUN_SPLIT_INTEGER_FIELDS:
            value = last_run[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"'{path}' has a malformed last_run field '{field}'."
                )
        for field in _LAST_RUN_SPLIT_OPTIONAL_INTEGER_FIELDS:
            if field not in last_run:
                continue
            value = last_run[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"'{path}' has a malformed last_run field '{field}'."
                )
        reasons = last_run.get("split_blocked_by_reason")
        if not isinstance(reasons, dict) or any(
            reason not in _BLOCKED_REASONS
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for reason, count in reasons.items()
        ):
            raise ConfigError(f"'{path}' has malformed split-blocked statistics.")
        if sum(reasons.values()) != last_run["split_blocked_files"]:
            raise ConfigError(f"'{path}' has contradictory split-blocked statistics.")
        if (
            last_run["split_syntax_files"] + last_run["split_lexical_files"]
            != last_run["split_divided_files"]
        ):
            raise ConfigError(f"'{path}' has contradictory split file-count statistics.")
        # Every restored count is <= its total, and each "*_calls_planned"
        # remainder plus its restored count equals its total (the frozen
        # run-statistics definitions).
        restored_pairs = (
            ("split_restored_complete_chunks", "split_chunks"),
            ("split_restored_unit_consolidation_calls", "split_unit_consolidation_calls_planned"),
            ("split_restored_general_reduction_calls", "split_general_reduction_calls_planned"),
            ("split_restored_final_synthesis_calls", "split_final_synthesis_calls_planned"),
        )
        for restored_field, total_field in restored_pairs:
            if last_run[restored_field] > last_run[total_field]:
                raise ConfigError(
                    f"'{path}' has a restored split count exceeding its total "
                    f"('{restored_field}' > '{total_field}')."
                )
        if (
            last_run["unit_documentation_calls_planned"]
            != last_run["split_chunks"] - last_run["split_restored_complete_chunks"]
        ):
            raise ConfigError(f"'{path}' has contradictory leaf call-count statistics.")
        expected_reduction_remaining = (
            last_run["split_unit_consolidation_calls_planned"]
            - last_run["split_restored_unit_consolidation_calls"]
        ) + (
            last_run["split_general_reduction_calls_planned"]
            - last_run["split_restored_general_reduction_calls"]
        )
        if last_run["file_reduction_calls_planned"] != expected_reduction_remaining:
            raise ConfigError(f"'{path}' has contradictory reduction call-count statistics.")
        if (
            last_run["synthesis_calls_planned"]
            != last_run["split_final_synthesis_calls_planned"]
            - last_run["split_restored_final_synthesis_calls"]
        ):
            raise ConfigError(f"'{path}' has contradictory synthesis call-count statistics.")
        if "split_unpaid_nodes" in last_run and last_run["split_unpaid_nodes"] != (
            last_run["unit_documentation_calls_planned"]
            + last_run["file_reduction_calls_planned"]
            + last_run["synthesis_calls_planned"]
        ):
            raise ConfigError(f"'{path}' has contradictory split unpaid-node statistics.")
        if (
            "split_reexecuted_nodes" in last_run
            and "split_unpaid_nodes" in last_run
            and last_run["split_reexecuted_nodes"] > last_run["split_unpaid_nodes"]
        ):
            raise ConfigError(f"'{path}' has contradictory split reexecution statistics.")
        if (
            "split_completed_files_reused" in last_run
            and last_run["split_completed_files_reused"]
            > last_run["files_reused_unchanged"]
        ):
            raise ConfigError(f"'{path}' has contradictory completed-split reuse statistics.")
        if (
            "split_partial_files_resumed" in last_run
            and last_run["split_partial_files_resumed"]
            > last_run["files_resumed_from_recovery"]
        ):
            raise ConfigError(f"'{path}' has contradictory split recovery statistics.")

    partition_total = (
        last_run["files_reused_unchanged"]
        + last_run["files_reused_identical_content"]
        + last_run["files_documented_by_llm"]
        + last_run["files_failed"]
        + last_run["files_unattempted"]
        + last_run.get("files_skipped_insufficient_source", 0)
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
