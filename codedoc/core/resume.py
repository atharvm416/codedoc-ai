"""Resume and recovery helpers for the documentation pipeline.

0.9.4 — extracted verbatim from ``codedoc.pipeline`` as part of the internal
decomposition.  This module owns:

- live-backup path resolution;
- loading existing per-file documentation from prior JSON / Markdown output;
- reconstructing the internal documentation shape from a public JSON record;
- building the final documentation records handed to ``write_project_outputs``;
- cleaning up stale 0.7.x build files and the legacy ``codedoc_db.json``.

It depends on the read-only document reader, output/project-view helpers,
hashing, and paths.  It must not scan source files, create providers, or
schedule agent work.
"""

from __future__ import annotations

from pathlib import Path

from codedoc.core.db import compute_file_hash
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.markdown_view import markdown_to_view
from codedoc.core.output import BUILD_FILENAME
from codedoc.core.project_view import read_codedoc_meta
from codedoc.core.record_meta import carry_private_keys
from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Live backup path helper (Work Item 1)
# ---------------------------------------------------------------------------

def _resolve_live_backup_path(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
) -> Path:
    """Return the absolute live JSON backup path for the given output scenario.

    | output_format | md_filename    | json_filename | Live backup path         |
    |---------------|----------------|---------------|--------------------------|
    | json / both   | (any)          | codedoc.json  | output_dir/codedoc.json  |
    | md            | codedoc.md     | (ignored)     | output_dir/codedoc.json  |
    | md            | report.md      | (ignored)     | output_dir/report.json   |

    For MD-only runs the backup is a JSON sibling of the Markdown file,
    derived from ``md_filename`` stem (not from ``json_filename``, which
    ``_resolve_output_spec`` leaves as the default ``codedoc.json`` even when
    the user requested ``--output docs/report.md``).
    """
    if output_format == "md":
        md_stem = Path(md_filename).stem
        return output_dir / f"{md_stem}.json"
    return output_dir / json_filename


# ---------------------------------------------------------------------------
# Existing-docs loader
# ---------------------------------------------------------------------------

def _load_existing_file_docs(
    output_dir: Path,
    json_filename: str,
    md_filename: str = "codedoc.md",
    live_backup_path: Path | None = None,
    read_only: bool = False,
) -> dict[str, dict]:
    """Load per-file documentation from existing output files.

    With ``read_only=True`` (planning / dry-run) a stale 0.7.x build file is
    skipped without being unlinked; routing results are identical.  Real runs
    delete the stale file later, behind the mutation boundary, via
    :func:`_cleanup_stale_build_file`.

    Priority order
    --------------
    1. **Live backup path** (e.g. ``report.json`` for ``--output report.md``)
       when it differs from the default JSON path.  This handles the named-MD
       case where ``_resolve_output_spec`` leaves ``output_json_filename`` as
       ``codedoc.json`` even though the live backup is ``report.json``.
    2. **Final JSON output** (``json_filename``).  Covers the JSON/both case
       and the default-MD case where the live backup IS ``codedoc.json``.
    3. **Stale build file** (``.codedoc_build.json``).  Migration fallback for
       0.7.x interrupted MD runs.  Overlaid only when newer than the JSON.
    4. **Markdown fallback** — same-stem sibling or configured ``md_filename``.

    Returns a dict mapping rel_path → file record dict.
    """
    existing: dict[str, dict] = {}

    # 1. Live backup when it differs from the standard JSON path (named-MD case).
    json_path = output_dir / json_filename
    if live_backup_path and live_backup_path.resolve() != json_path.resolve():
        if live_backup_path.exists():
            try:
                existing = records_by_path(read_codedoc_document(live_backup_path))
                if existing:
                    logger.info(
                        "Loaded %d existing record(s) from live backup '%s'.",
                        len(existing),
                        live_backup_path.name,
                    )
                    return existing
            except (ConfigError, FileNotFoundError) as exc:
                logger.debug(
                    "Optional resume candidate '%s' rejected: %s",
                    live_backup_path.name,
                    exc,
                )

    # 2. Final JSON output (also the live backup for JSON/both and default-MD runs).
    if json_path.exists():
        try:
            existing = records_by_path(read_codedoc_document(json_path))
        except (ConfigError, FileNotFoundError) as exc:
            logger.debug(
                "Optional resume candidate '%s' rejected: %s",
                json_path.name,
                exc,
            )

    # 3. Stale 0.7.x build file migration fallback.
    build_path = output_dir / BUILD_FILENAME
    if build_path.exists():
        try:
            build_is_stale = (
                json_path.exists()
                and build_path.stat().st_mtime < json_path.stat().st_mtime
            )
            if build_is_stale:
                if read_only:
                    logger.info(
                        "Build file '%s' is older than '%s' — treating as stale "
                        "(read-only mode: not removed).",
                        BUILD_FILENAME,
                        json_filename,
                    )
                else:
                    logger.info(
                        "Build file '%s' is older than '%s' — treating as stale and removing it.",
                        BUILD_FILENAME,
                        json_filename,
                    )
                    try:
                        build_path.unlink()
                    except Exception:
                        pass
            else:
                # The reader parses only; age comparison, merge policy, and
                # deletion remain here in the pipeline (legacy stale-build role).
                build_files = records_by_path(
                    read_codedoc_document(build_path, legacy_role="stale_build")
                )
                if build_files:
                    logger.info(
                        "Build file '%s' found (%d record(s)) — merging (migration from 0.7.x).",
                        BUILD_FILENAME,
                        len(build_files),
                    )
                    existing.update(build_files)
        except Exception:
            pass

    if existing:
        return existing

    # 4. Markdown fallback.
    stem_sibling = output_dir / (Path(json_filename).stem + ".md")
    configured_md = output_dir / md_filename
    md_candidates: list[Path] = [stem_sibling]
    if configured_md != stem_sibling:
        md_candidates.append(configured_md)
    # Also check the live backup sibling when named-MD
    if live_backup_path:
        md_sibling = live_backup_path.with_suffix(".md")
        if md_sibling not in md_candidates:
            md_candidates.insert(0, md_sibling)

    for md_path in md_candidates:
        if md_path.exists():
            try:
                return _load_existing_file_docs_from_md(md_path)
            except Exception:
                pass

    return {}


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
            # hash (0.8.1+ Markdown), use that as the fallback so we never
            # overwrite a good hash with an empty string.
            file_hash = file_hashes.get(path) or file_record.get("hash", "")
            result[path] = {**file_record, "hash": file_hash}
    return result


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
    cleaned = {k: v for k, v in doc.items() if v not in (None, "", [], {}, {"external": [], "internal": []})}
    # 0.9.3: carry registered private keys from the public record into the
    # reconstructed flat documentation result so they survive resume/reuse.
    carry_private_keys(file_record, cleaned)
    return cleaned


def _build_documentation_records(
    rel_paths: set,
    file_map: dict,
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
            descriptor = file_map.get(rel_path, {})
            try:
                file_hash = compute_file_hash(descriptor["path"]) if descriptor.get("path") else ""
            except Exception:
                file_hash = existing_docs.get(rel_path, {}).get("hash", "")
        else:
            file_hash = existing_docs.get(rel_path, {}).get("hash", "")

        records.append({
            "hash": file_hash,
            "file_path": rel_path,
            "language": doc.get("language", ""),
            "documentation": doc,
        })
    return records


def _cleanup_stale_build_file(output_dir: Path, json_filename: str) -> None:
    """Remove a stale 0.7.x build file — mutation-phase counterpart of the
    skip in ``_load_existing_file_docs(read_only=True)``."""
    build_path = output_dir / BUILD_FILENAME
    json_path = output_dir / json_filename
    try:
        if (
            build_path.exists()
            and json_path.exists()
            and build_path.stat().st_mtime < json_path.stat().st_mtime
        ):
            build_path.unlink()
            logger.info(
                "Removed stale build file '%s' (older than '%s').",
                BUILD_FILENAME,
                json_filename,
            )
    except Exception:
        pass


def _remove_legacy_db(output_dir: Path) -> None:
    """Remove codedoc_db.json left over from earlier versions."""
    legacy = output_dir / "codedoc_db.json"
    if legacy.exists():
        try:
            legacy.unlink()
            logger.info("Removed legacy codedoc_db.json (no longer used since 0.6.4)")
        except Exception:
            pass
