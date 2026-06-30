"""Build clean public project documentation views from cached records.

Markdown serialization and parsing live in :mod:`codedoc.core.markdown_view`.
This module retains project-view
*assembly*: ``build_project_view``, ``json_from_view``, ``clean_file_record`` /
``_clean_file``, folder/tree/graph-index construction, dependency-catalog
assembly, empty-value pruning, usage-example sanitization, and
``read_codedoc_meta`` (which delegates to the centralized document reader).

The serializer helpers that moved are still importable from this module as a
deprecated one-release compatibility shim (see ``__getattr__`` at the bottom),
which forwards lazily to :mod:`codedoc.core.markdown_view` to avoid an import
cycle.  Import them from ``codedoc.core.markdown_view`` directly in new code.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from codedoc.core.dependency_kind import (
    KIND_EXTERNAL,
    KIND_SDK,
    _NODE_LANGUAGES,
    classify_non_project_dependency,
)
from codedoc.core.record_meta import carry_private_keys
from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.4"

# Placeholder import/package names that LLMs emit in usage examples.
# These are never meaningful to end-users and must be stripped before output.
# Uses \b word boundaries so 'example_package' does NOT match inside a longer
# real path like 'my_real_example_package_utils' (since '_' is a \w character,
# there is no word boundary between the prefix and the suffix).
_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:your_package_name|your_package|your_project|your_app"
    r"|example_package|my_package)\b"
    r"|package:example/",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public view construction
# ---------------------------------------------------------------------------

def build_project_view(
    records: list[dict],
    stats: dict,
    entry_file: str | None = None,
    graph_edges: list[dict] | None = None,
    reachable_rels: set[str] | frozenset[str] | None = None,
    unresolved_imports_by_path: dict[str, list[str]] | None = None,
) -> dict:
    """Return a compact language-neutral view for public JSON/Markdown output."""
    graph_edges = graph_edges or []
    files = [_clean_file(record) for record in records]
    paths = [file["path"] for file in files]
    internal_by_from, imported_by = _edge_indexes(graph_edges)
    project_import_roots = _project_import_roots(files)

    for file in files:
        path = file["path"]
        file["reachable_from_entry"] = (
            True if reachable_rels is None else path in reachable_rels
        )
        internal_paths = internal_by_from.get(path, [])
        # External/sdk links are projected deterministically
        # from this file's parser imports + finalized graph edges, never from
        # model output, so single and triple modes produce identical links.
        # For Python and generic-parser languages, use the
        # per-file unresolved imports (graph-filtered) as the authoritative source.
        unresolved = (
            unresolved_imports_by_path.get(path)
            if unresolved_imports_by_path is not None
            else None
        )
        external, sdk = _project_dependency_links(
            file, internal_paths, project_import_roots, unresolved_imports=unresolved
        )
        links = {
            "internal_dependencies": internal_paths,
            "imported_by": imported_by.get(path, []),
            "external_dependencies": external,
            "sdk_dependencies": sdk,
        }
        links = {key: value for key, value in links.items() if value}
        if links:
            file["links"] = links

    folders = _folder_view(files)
    languages = sorted({file["language"] for file in files if file.get("language")})
    dependency_catalog = _dependency_catalog(files)

    view = {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "entry_file": entry_file,
            "file_count": len(files),
            "languages": languages,
            "folders": [folder["path"] for folder in folders],
        },
        "run": {
            "files_checked": stats.get("checked", 0),
            "files_failed": stats.get("failed", 0),
            "files_skipped": stats.get("skipped", 0),
            "files_reused": stats.get("reused", 0),
            "files_documented": len(files),
        },
        "tree": _tree(paths),
        "folders": folders,
        "dependency_catalog": dependency_catalog,
        "dependency_graph": graph_edges,
        "files": files,
    }
    return {key: value for key, value in view.items() if value not in (None, "", [], {})}


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def json_from_view(view: dict, error_summary: str = "") -> str:
    """Render a public project view as formatted JSON."""
    payload = dict(view)
    if error_summary and error_summary != "No errors.":
        payload["errors"] = error_summary
    # Embed _codedoc metadata block first so it's easy to find
    project = view.get("project", {})
    meta_block = {
        "_codedoc": {
            "entry_file": project.get("entry_file"),
            "schema_version": view.get("schema_version", SCHEMA_VERSION),
        }
    }
    # Determinism: completed output carries no run-varying timestamp.
    # A caller-provided legacy view may still contain ``generated_at`` — never
    # propagate it into the completed payload.
    payload.pop("generated_at", None)
    ordered = {**meta_block, **payload}
    return json.dumps(ordered, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Metadata comment reader
# ---------------------------------------------------------------------------

def read_codedoc_meta(file_path: Path) -> dict:
    """
    Read CodeDoc metadata from a previously generated .json or .md output file.

    Returns a dict containing at least ``schema_version``.  ``entry_file`` may
    be ``None`` — callers must handle that case (a valid CodeDoc file can have
    a null entry when auto-detection was used or when no entry was specified on
    the first run).

    Raises :class:`~codedoc.utils.errors.ConfigError` when the file cannot be
    read, is not a recognised CodeDoc output, or is missing structural metadata
    (e.g. the ``_codedoc`` block for JSON, or the ``<!-- codedoc-ai: ... -->``
    comment for Markdown).

    Parsing and structural ownership are delegated to the centralized
    read-only document reader.  A function-local import keeps the dependency
    acyclic (``document`` imports low-level helpers from this module at module
    load time).
    """
    from codedoc.core.document import read_codedoc_document

    try:
        document = read_codedoc_document(file_path)
    except FileNotFoundError as exc:
        raise ConfigError(f"Cannot read '{file_path}': file does not exist.") from exc

    return dict(document.metadata)


# ---------------------------------------------------------------------------
# Public wrapper for pipeline record conversion
# ---------------------------------------------------------------------------

def clean_file_record(record: dict) -> dict:
    """Public wrapper around the internal ``_clean_file`` helper.

    Converts a raw pipeline record into the clean public file entry stored
    in the JSON / Markdown output.  Used by :class:`~codedoc.core.safe_writer.SafeWriter`
    to produce partial file entries that are structurally identical to what
    the final ``write_project_outputs`` call would produce.

    Parameters
    ----------
    record:
        Dict with keys: ``hash``, ``file_path``, ``language``, and
        ``documentation`` (the full orchestrator result dict).

    Returns
    -------
    dict
        Clean public file entry with ``path``, ``language``, ``description``,
        ``hash``, etc.
    """
    return _clean_file(record)


# ---------------------------------------------------------------------------
# Internal file record cleaner
# ---------------------------------------------------------------------------

def _clean_file(record: dict) -> dict:
    result = record.get("documentation", {}) or {}
    language = result.get("language") or record.get("language", "")
    dependencies = result.get("dependencies_analysis", {})

    # Public external/sdk dependency links are no longer
    # derived from the model's ``dependencies_analysis.external``.  They are
    # projected deterministically from the parser ``imports`` and finalized graph
    # edges in :func:`build_project_view`, so the same source produces identical
    # links in single and triple modes.  The model dependency fields below remain
    # *bounded enrichment*: they may only supply ``used_for`` text for a
    # dependency that the deterministic projection already admits.
    usage_notes = dependencies.get("usage_notes", []) if isinstance(dependencies, dict) else []
    dependency_refs = (
        dependencies.get("dependency_refs", []) if isinstance(dependencies, dict) else []
    )
    catalog_updates = (
        dependencies.get("catalog_updates", []) if isinstance(dependencies, dict) else []
    )

    file = {
        "hash": record.get("hash", ""),
        "path": record.get("file_path") or result.get("file_path", ""),
        "language": language,
        "description": result.get("description", ""),
        "role_in_system": result.get("role_in_system", ""),
        "imports": result.get("imports", []),
        "functions": result.get("functions", []),
        "classes": result.get("classes", []),
        "exports": result.get("exports", []),
        "key_concepts": result.get("key_concepts", []),
        "usage_example": _sanitize_usage_example(result.get("usage_example", "")),
        "_deps": {k: v for k, v in dependencies.items() if v not in (None, "", [], {})} if isinstance(dependencies, dict) else {},
        "dependency_refs": dependency_refs,
        "dependency_usage": _dependency_usage_map(usage_notes),
        "dependency_catalog_updates": _clean_catalog_updates(catalog_updates),
    }
    cleaned = {key: value for key, value in file.items() if value not in (None, "", [], {})}

    # Registered private keys survive empty-value pruning.  Carry from the
    # nested orchestrator result first, then the top-level record so a persisted
    # top-level value wins when both layers contain the same key.
    carry_private_keys(result, cleaned)
    carry_private_keys(record, cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Usage-example sanitizer
# ---------------------------------------------------------------------------

def _sanitize_usage_example(usage_example: str) -> str:
    """Remove usage examples that contain LLM placeholder package/import names.

    LLMs sometimes emit template strings like::

        import 'package:your_package/core/theme/app_theme.dart';

    or::

        from your_project import MyClass

    These are never meaningful to users.  When a placeholder is detected the
    whole ``usage_example`` is discarded (returned as an empty string) so it
    does not appear in JSON or Markdown output.

    The check is case-insensitive and uses word boundaries (``\\b``) so that
    ``example_package`` only matches as a standalone word and does NOT trigger
    on real paths like ``my_real_example_package_helper`` where ``_`` acts as a
    word character and prevents the boundary from forming.

    No LLM call is made.  If the example cannot be deterministically fixed, it
    is silently removed rather than kept incorrect.
    """
    if not usage_example:
        return usage_example
    if _PLACEHOLDER_PATTERN.search(usage_example):
        return ""
    return usage_example


# ---------------------------------------------------------------------------
# Pruning utility
# ---------------------------------------------------------------------------

def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        pruned = {
            key: _prune_empty(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return {
            key: item
            for key, item in pruned.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            item
            for item in (_prune_empty(item) for item in value)
            if item not in (None, "", [], {})
        ]
    return value


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _dependency_usage_map(usage_notes: list) -> dict[str, str]:
    """Map each raw usage-note import to its ``used_for`` text.

    Names are kept *raw* here; classification (external vs sdk) and
    canonicalization are deferred to :func:`_dependency_catalog`, which knows
    each file's language.
    """
    usage: dict[str, str] = {}
    for note in usage_notes:
        if not isinstance(note, dict):
            continue
        name = str(note.get("import", "") or "").strip()
        used_for = str(note.get("used_for", "")).strip()
        if name and used_for and name not in usage:
            usage[name] = used_for
    return usage


_RECOGNIZED_TYPE_HINTS = ("internal", "external", "sdk")


def _clean_catalog_updates(catalog_updates: list) -> list[dict]:
    """Shape-clean agent catalog updates only.

    Runs *before* graph links are attached, so it cannot validate
    internal hints or classify names.  It retains the trimmed raw name, a
    recognized raw type hint, and the trimmed ``used_for``.  All
    classification and internal-hint validation happen later in
    :func:`_dependency_catalog`, after links exist.
    """
    updates = []
    for item in catalog_updates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        type_hint = item.get("type") if item.get("type") in _RECOGNIZED_TYPE_HINTS else ""
        update = {
            "name": name,
            "type": type_hint,
            "used_for": str(item.get("used_for", "")).strip(),
        }
        updates.append({key: value for key, value in update.items() if value not in ("", None)})
    return updates


def _normalize_internal_candidate(name: Any) -> str:
    """Normalize an agent-provided internal hint for exact-path comparison."""
    value = str(name or "").strip()
    if value.startswith("package:"):
        value = value[len("package:"):]
    return value.replace("\\", "/")


# Catalog type tag for graph-resolved internal dependencies.  External / SDK
# tags come from the deterministic classifier (KIND_EXTERNAL / KIND_SDK).
KIND_INTERNAL = "internal"


def _project_dependency_links(
    file: dict,
    internal_paths: list[str],
    project_import_roots: set[str],
    unresolved_imports: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Project deterministic ``(external, sdk)`` links for one file.

    Dependency identity is no longer taken from model
    type labels; it is projected through :func:`classify_non_project_dependency`
    from an authoritative name source, then de-duplicated and sorted so both
    analysis modes emit identical links for identical source.

    The authoritative name source depends on the language:

    - **Python** — the unresolved parser imports (graph-filtered): imports that
      did not resolve to an internal project file.  A relative import yields an
      empty canonical and is skipped.  A Python import whose canonical root names
      a project package *and* resolves to one of this file's finalized internal
      links (e.g. ``codedoc.*``) is dropped as a false external.  Model output
      can never add, remove, or reclassify a link.  Falls back to all parser
      ``imports`` when ``unresolved_imports`` is ``None``.
    - **Generic-parser languages** (Dart, Java, Kotlin, C#, Swift, Go, Ruby,
      Rust, C/C++, HTML) — the unresolved parser imports (graph-filtered).
      For Dart, only imports starting with ``dart:`` or ``package:`` are
      classified; bare filenames and relative paths that the graph did not
      resolve are skipped to prevent bogus external entries.  Falls back to
      model ``_deps.external`` when ``unresolved_imports`` is ``None``.
    - **React/Node family** (JS, TS, JSX, TSX) — always uses model
      ``_deps.external``, because ``react_parser`` deliberately omits bare npm
      packages and the parser cannot enumerate them.

    Either way, model ``catalog_updates`` / ``dependency_refs`` / ``usage_notes``
    can only enrich an admitted dependency with ``used_for`` text in
    :func:`_dependency_catalog`; they never create a public link.
    """
    language = str(file.get("language", "") or "").lower()

    if language in _NODE_LANGUAGES:
        # React/Node: model _deps.external is the only source for npm packages.
        deps = file.get("_deps", {})
        names = deps.get("external", []) if isinstance(deps, dict) else []
        suppress_project_roots = False
    elif language == "python":
        if unresolved_imports is not None:
            names = unresolved_imports
        else:
            names = file.get("imports", [])
        suppress_project_roots = True
    else:
        # Generic-parser languages: use graph-filtered unresolved imports.
        if unresolved_imports is not None:
            if language == "dart":
                # Dart: skip bare filenames and relative imports; only dart:*
                # and package:* are unambiguously external / SDK.
                names = [
                    imp for imp in unresolved_imports
                    if imp.startswith(("dart:", "package:"))
                ]
            else:
                names = unresolved_imports
        else:
            # Fallback when caller does not supply graph-filtered imports.
            deps = file.get("_deps", {})
            names = deps.get("external", []) if isinstance(deps, dict) else []
        suppress_project_roots = False

    if not isinstance(names, list):
        return [], []

    internal_roots: set[str] = set()
    for internal_path in internal_paths:
        internal_roots.update(_python_import_roots_for_path(internal_path))

    external_set: set[str] = set()
    sdk_set: set[str] = set()
    for name in names:
        dep = classify_non_project_dependency(name, language)
        if not dep.canonical:
            continue
        is_resolved_python_project_root = (
            suppress_project_roots
            and dep.canonical in project_import_roots
            and dep.canonical in internal_roots
        )
        if is_resolved_python_project_root:
            continue
        if dep.kind == KIND_SDK:
            sdk_set.add(dep.canonical)
        else:
            external_set.add(dep.canonical)
    return sorted(external_set), sorted(sdk_set)


def _project_import_roots(files: list[dict]) -> set[str]:
    """Return exact Python import roots represented by project file paths.

    Python modules contribute their filename stem (except ``__init__``), and
    package paths contribute each directory segment. This represents root-level
    modules, conventional packages, and ``src`` layouts. Admission still
    requires matching finalized internal-link evidence for the same importing
    file.
    """
    roots: set[str] = set()
    for file in files:
        if str(file.get("language", "") or "").lower() != "python":
            continue
        roots.update(_python_import_roots_for_path(file.get("path", "")))
    return roots


def _python_import_roots_for_path(path: Any) -> set[str]:
    """Return exact import-root candidates represented by one Python path."""
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        return set()
    parts = PurePosixPath(normalized).parts
    if not parts:
        return set()
    roots = {part for part in parts[:-1] if part not in ("", ".", "..")}
    stem = PurePosixPath(parts[-1]).stem
    if stem and stem != "__init__":
        roots.add(stem)
    return roots


def _eligible_catalog_keys(
    file: dict,
    project_import_roots: set[str],
) -> set[tuple[str, str]]:
    """Return the ``(type, canonical_name)`` keys this file's finalized links
    authorize for the dependency catalog.

    Only graph-resolved internal links and deterministically classified
    external / SDK links produce eligible keys.  A Python external candidate
    whose canonical root is a project package root is dropped as a false
    external — project code resolved internally via the graph, not a third-party
    package.  Model ``catalog_updates``, ``dependency_refs``, and ``usage_notes``
    can never widen this set; they may only supply ``used_for`` text for a key
    that is already eligible.
    """
    links = file.get("links", {})
    if not isinstance(links, dict):
        return set()
    keys: set[tuple[str, str]] = set()
    for internal_path in links.get("internal_dependencies", []):
        if internal_path:
            keys.add((KIND_INTERNAL, internal_path))
    internal_roots: set[str] = set()
    for path in links.get("internal_dependencies", []):
        internal_roots.update(_python_import_roots_for_path(path))
    language = str(file.get("language", "") or "").lower()
    for name in links.get("external_dependencies", []):
        if not name:
            continue
        is_resolved_python_project_root = (
            language == "python"
            and name in project_import_roots
            and name in internal_roots
        )
        if not is_resolved_python_project_root:
            keys.add((KIND_EXTERNAL, name))
    for name in links.get("sdk_dependencies", []):
        if name:
            keys.add((KIND_SDK, name))
    return keys


def _dependency_catalog(files: list[dict]) -> list[dict]:
    """Build the deduplicated dependency catalog, grouped by (type, name).

    Admission is evidence-based: an entry's ``(type, canonical_name)`` key must
    be authorized by at least one file's finalized links
    (:func:`_eligible_catalog_keys`).  Graph resolution and deterministic
    classification are authoritative; model ``catalog_updates``,
    ``dependency_refs``, and ``usage_notes`` may enrich an eligible dependency
    with ``used_for`` text but can never create or retype one.  Every emitted
    entry carries non-empty ``used_for`` text and at least one backing file.
    """
    # Keyed by (type, canonical_name) so external/sdk/internal never collide.
    catalog: dict[tuple[str, str], dict] = {}
    project_import_roots = _project_import_roots(files)

    def _entry(type_: str, canonical: str) -> dict:
        key = (type_, canonical)
        item = catalog.get(key)
        if item is None:
            item = {"name": canonical, "type": type_, "used_for": "", "files": set()}
            catalog[key] = item
        return item

    for file in files:
        path = file.get("path", "")
        language = file.get("language", "")
        usage_raw = file.pop("dependency_usage", {})
        catalog_updates = file.pop("dependency_catalog_updates", [])
        dependency_refs = file.pop("dependency_refs", [])

        eligible = _eligible_catalog_keys(file, project_import_roots)
        if not eligible:
            continue

        # Classify this file's usage notes into used_for text, restricted to
        # eligible external/SDK keys.  First note wins per key within a file.
        usage: dict[tuple[str, str], str] = {}
        for raw_name, used_for in usage_raw.items():
            if not used_for:
                continue
            dep = classify_non_project_dependency(raw_name, language)
            key = (dep.kind, dep.canonical)
            if dep.canonical and key in eligible and key not in usage:
                usage[key] = used_for

        # 1. Agent catalog updates may carry used_for text + a type hint, but
        #    only for an eligible key.  An unresolved internal hint, or a name
        #    with no matching finalized link, is discarded — never reclassified
        #    into a fabricated entry.
        for update in catalog_updates:
            raw_name = update.get("name", "")
            type_hint = update.get("type", "")
            used_for = update.get("used_for", "")
            if not used_for:
                continue
            matched: tuple[str, str] | None = None
            norm = _normalize_internal_candidate(raw_name)
            if norm and (KIND_INTERNAL, norm) in eligible:
                matched = (KIND_INTERNAL, norm)
            if matched is None:
                # An unresolved internal hint is discarded rather than
                # reclassified. For other hints, deterministic external/SDK
                # classification wins over the model-provided type.
                if type_hint == "internal":
                    continue
                dep = classify_non_project_dependency(raw_name, language)
                if dep.canonical and (dep.kind, dep.canonical) in eligible:
                    matched = (dep.kind, dep.canonical)
            if matched is None:
                continue
            item = _entry(*matched)
            item["files"].add(path)
            if _should_replace_catalog_text(item["used_for"], used_for):
                item["used_for"] = used_for

        # 2. Resolved external / SDK links from this file's classification are
        #    the authority for membership.  Attach the file and any usage text,
        #    but never create a text-less entry (A4 drops those below anyway).
        for key in eligible:
            if key[0] == KIND_INTERNAL:
                continue
            item = catalog.get(key)
            if item is None:
                if key not in usage:
                    continue
                item = _entry(*key)
            item["files"].add(path)
            if not item["used_for"] and usage.get(key):
                item["used_for"] = usage[key]

        # 3. Dependency references — attach only to an eligible key, and only to
        #    an existing entry or one the agent gave usage text for.
        for raw_ref in dependency_refs:
            dep = classify_non_project_dependency(raw_ref, language)
            key = (dep.kind, dep.canonical)
            if not dep.canonical or key not in eligible:
                continue
            item = catalog.get(key)
            if item is None:
                if key not in usage:
                    continue
                item = _entry(*key)
            item["files"].add(path)
            if not item["used_for"] and usage.get(key):
                item["used_for"] = usage[key]

        # 4. Remaining usage notes that match an existing eligible entry.
        for key, used_for in usage.items():
            item = catalog.get(key)
            if item is None:
                continue
            item["files"].add(path)
            if not item["used_for"]:
                item["used_for"] = used_for

    result = []
    for item in catalog.values():
        files_used = sorted(p for p in item.pop("files") if p)
        # A4: every emitted entry needs non-empty used_for text and at least one
        # backing file whose finalized links contain the key.
        if not item["used_for"] or not files_used:
            continue
        item["files"] = files_used
        item["file_count"] = len(files_used)
        result.append(item)

    return sorted(result, key=lambda item: (-item["file_count"], item["name"], item["type"]))


def _should_replace_catalog_text(existing: str, candidate: str) -> bool:
    if not candidate:
        return False
    if not existing:
        return True
    return _specificity_score(candidate) > _specificity_score(existing)


def _specificity_score(text: str) -> tuple[int, int]:
    project_words = (
        "project",
        "application",
        "app",
        "api",
        "database",
        "route",
        "screen",
        "service",
        "model",
        "schema",
        "state",
    )
    lowered = text.lower()
    return (sum(1 for word in project_words if word in lowered), len(text))


def _edge_indexes(graph_edges: list[dict]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    internal_by_from: dict[str, list[str]] = defaultdict(list)
    imported_by: dict[str, list[str]] = defaultdict(list)
    for edge in graph_edges:
        source = edge.get("from", "")
        target = edge.get("to", "")
        if not source or not target:
            continue
        internal_by_from[source].append(target)
        imported_by[target].append(source)
    return (
        {key: sorted(set(value)) for key, value in internal_by_from.items()},
        {key: sorted(set(value)) for key, value in imported_by.items()},
    )


def _folder_view(files: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for file in files:
        folder = _top_folder(file["path"])
        grouped[folder].append(file)

    folders = []
    for folder, folder_files in sorted(grouped.items()):
        languages = sorted({f.get("language", "") for f in folder_files if f.get("language")})
        concepts = Counter(
            concept
            for file in folder_files
            for concept in file.get("key_concepts", [])
            if isinstance(concept, str)
        )
        common_concepts = [name for name, _ in concepts.most_common(5)]
        folders.append(
            {
                "path": folder,
                "summary": _folder_summary(folder, len(folder_files), languages, common_concepts),
                "file_count": len(folder_files),
                "languages": languages,
                "files": [file["path"] for file in sorted(folder_files, key=lambda f: f["path"])],
                "key_concepts": common_concepts,
            }
        )
    return folders


def _folder_summary(
    folder: str,
    file_count: int,
    languages: list[str],
    concepts: list[str],
) -> str:
    language_text = ", ".join(languages) if languages else "source"
    concept_text = f" Common concepts: {', '.join(concepts)}." if concepts else ""
    if folder == ".":
        return f"Root-level {language_text} files ({file_count} file(s)).{concept_text}"
    return f"Files under `{folder}` ({file_count} {language_text} file(s)).{concept_text}"


def _tree(paths: list[str]) -> dict:
    root: dict[str, Any] = {}
    for path in sorted(paths):
        parts = PurePosixPath(path).parts
        current = root
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = {"type": "file", "path": path}
    return root


def _top_folder(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[0] if len(parts) > 1 else "."


# ---------------------------------------------------------------------------
# Compatibility shim (deprecated; import from codedoc.core.markdown_view)
# ---------------------------------------------------------------------------

_MARKDOWN_VIEW_COMPAT_NAMES = frozenset(
    {
        "EmbeddedViewResult",
        "json_from_markdown",
        "markdown_from_json",
        "markdown_from_view",
        "markdown_to_view",
        "read_embedded_view",
        "read_embedded_view_result",
        "_build_full_view_comment",
        "_build_meta_comment",
        "_public_view_for_embedding",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily forward moved serializer/parser helpers to ``markdown_view``.

    Markdown serialization/parsing moved to :mod:`codedoc.core.markdown_view`
    during internal decomposition.  Repository tests and documented integrations still import those
    names from this module, so they are forwarded here for one release.  The
    import is function-local so ``markdown_view`` (which imports a few pure
    assembly helpers from this module at load time) does not create a cycle.
    No runtime warning is emitted.
    """
    if name not in _MARKDOWN_VIEW_COMPAT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from codedoc.core import markdown_view as _mv

    return getattr(_mv, name)
