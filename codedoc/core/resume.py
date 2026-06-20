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
from codedoc.utils.errors import ConfigError, OutputError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# 0.9.8: crash-recovery (live-backup) data lives in its own dedicated file so a
# run never overwrites the user's last stable completed output while it is still
# in progress.  The recovery file is named ``crash_recovery_<stem>.json`` where
# ``<stem>`` is the stem of the final output file.  This is a fixed internal
# constant — there is no configuration key, environment variable, or CLI flag
# for it, and the same constant guards ``--output`` (see ``loader._resolve_output_spec``).
CRASH_RECOVERY_PREFIX = "crash_recovery_"

# Bound on the suffix walk for an occupied/invalid recovery name (Workstream B).
_MAX_RECOVERY_CANDIDATES = 1000


# ---------------------------------------------------------------------------
# Recovery path helpers (Work Item 1 / 0.9.8)
# ---------------------------------------------------------------------------

def _resolve_live_backup_path(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
) -> Path:
    """Return the *base* crash-recovery file path for the given output scenario.

    The recovery file is always a distinct file derived from the final output
    stem, in the output directory — it is **never** equal to the stable JSON
    output, the Markdown output, or the diagnostic log:

    | output_format | stem source   | base recovery path                    |
    |---------------|---------------|---------------------------------------|
    | json / both   | json_filename | output_dir/crash_recovery_<stem>.json |
    | md            | md_filename   | output_dir/crash_recovery_<stem>.json |

    For MD-only runs the stem is taken from ``md_filename`` (not from
    ``json_filename``, which ``_resolve_output_spec`` leaves as the default
    ``codedoc.json`` even when the user requested ``--output docs/report.md``).

    Resolution is pure and does not touch the disk.  Suffix selection for an
    occupied/invalid name happens in :func:`select_active_recovery_path`.
    """
    if output_format == "md":
        stem = Path(md_filename).stem
    else:
        stem = Path(json_filename).stem
    return output_dir / f"{CRASH_RECOVERY_PREFIX}{stem}.json"


def _resolve_legacy_backup_path(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
) -> Path:
    """Return the *legacy* (pre-0.9.8) live-backup location for this scenario.

    Earlier versions aliased the live backup onto the final JSON (for
    ``json``/``both``) or onto a JSON sibling of the Markdown file (for ``md``).
    An interrupted pre-0.9.8 run may therefore have left in-progress
    ``_crash_safety`` records at that location.  This release reads that file as
    a *legacy in-progress overlay* during resume; the value is otherwise pure.
    """
    if output_format == "md":
        return output_dir / f"{Path(md_filename).stem}.json"
    return output_dir / json_filename


def _recovery_candidate(base_recovery_path: Path, n: int) -> Path:
    """Return the n-th recovery candidate.

    ``n == 1`` is the base name itself; ``n >= 2`` appends the literal ``(n)``
    suffix before ``.json`` (``crash_recovery_<stem>(2).json``, then ``(3)`` …).
    """
    if n == 1:
        return base_recovery_path
    return base_recovery_path.with_name(
        f"{base_recovery_path.stem}({n}){base_recovery_path.suffix}"
    )


def select_active_recovery_path(base_recovery_path: Path) -> tuple[Path, bool]:
    """Deterministically choose the active recovery file by walking candidates.

    Walks ``crash_recovery_<stem>.json``, ``crash_recovery_<stem>(2).json``,
    ``…(3).json`` … and applies, for each candidate, the first matching rule:

    1. **Absent** → use it for a fresh run (no resume); returns ``(path, False)``.
    2. **Present and a valid, codedoc-owned in-progress recovery document** →
       use it and resume from its records; returns ``(path, True)``.
    3. **Present but completed, unreadable, malformed, or foreign-owned** →
       leave it untouched and advance to the next suffix.

    The walk stops at the first name resolved by rule 1 or 2.  It is capped at
    exactly :data:`_MAX_RECOVERY_CANDIDATES` candidates; if every candidate is
    occupied by a non-resumable, non-absent file an :class:`OutputError` naming
    the output directory and the exhausted range is raised rather than looping
    unbounded.

    The walk is read-only — it never creates, mutates, or deletes a candidate.
    """
    for n in range(1, _MAX_RECOVERY_CANDIDATES + 1):
        candidate = _recovery_candidate(base_recovery_path, n)
        if not candidate.exists():
            return candidate, False
        try:
            document = read_codedoc_document(candidate)
        except (ConfigError, FileNotFoundError):
            # Unreadable / malformed / foreign — preserve and advance.
            continue
        if document.in_progress:
            return candidate, True
        # A clean completed CodeDoc JSON at a reserved recovery name is not a
        # live recovery source; preserve it rather than overwrite, and advance.

    last = _recovery_candidate(base_recovery_path, _MAX_RECOVERY_CANDIDATES)
    raise OutputError(
        str(base_recovery_path.parent),
        f"all {_MAX_RECOVERY_CANDIDATES} crash-recovery candidates from "
        f"'{base_recovery_path.name}' to '{last.name}' in '{base_recovery_path.parent}' "
        "are occupied by non-resumable files; remove or relocate them to continue.",
    )


# ---------------------------------------------------------------------------
# Existing-docs loader
# ---------------------------------------------------------------------------

def _load_existing_file_docs(
    output_dir: Path,
    json_filename: str,
    md_filename: str = "codedoc.md",
    live_backup_path: Path | None = None,
    read_only: bool = False,
    *,
    output_format: str | None = None,
    legacy_recovery_path: Path | None = None,
) -> dict[str, dict]:
    """Load per-file documentation from existing output files.

    With ``read_only=True`` (planning / dry-run) a stale 0.7.x build file is
    skipped without being unlinked; routing results are identical.  Real runs
    delete the stale file later, behind the mutation boundary, via
    :func:`_cleanup_stale_build_file`.

    0.9.8 — the dedicated recovery file (``live_backup_path``, now always a
    distinct ``crash_recovery_<stem>.json``) is no longer a short-circuit source.
    The stable completed output is read first as the reuse baseline, and the
    active in-progress recovery records overlay it.  Records are merged by
    project-relative path in a fixed oldest-to-newest precedence; a path present
    in a later source replaces the whole earlier record (fields are never merged
    across records):

    1. **Clean stable completed output** as the baseline:
       - ``json`` / ``both``: ``json_filename`` (clean *or* a legacy in-progress
         ``_crash_safety`` document — either way the reuse base for this path);
       - ``md``: the Markdown output (same-stem sibling or configured
         ``md_filename``).
       The stale 0.7.x build file (``.codedoc_build.json``) is migrated into this
       baseline when newer than the JSON.
    2. **Legacy in-progress stable-path records** overlay the baseline — for
       ``md`` runs this is the pre-0.9.8 JSON sibling (``legacy_recovery_path``)
       when it is an in-progress document.  (For ``json``/``both`` the legacy
       location *is* the baseline path, so it is already covered by step 1.)
    3. **Active dedicated recovery records** (``live_backup_path``) overlay both
       — newest work from an interrupted run.

    Returns a dict mapping rel_path → file record dict.
    """
    existing: dict[str, dict] = {}
    json_path = output_dir / json_filename
    # ``output_format`` is accepted for call-site clarity and forward use; the
    # overlay logic below is format-agnostic (the stable JSON, the Markdown
    # fallback, the legacy sibling, and the active recovery file are each read
    # when present), so no explicit branching on it is required here.
    _ = output_format

    # --- 1a. Stable JSON baseline (json/both; also cross-format reuse for md).
    if json_path.exists():
        try:
            existing.update(records_by_path(read_codedoc_document(json_path)))
        except (ConfigError, FileNotFoundError) as exc:
            logger.debug(
                "Optional resume candidate '%s' rejected: %s", json_path.name, exc
            )

    # --- 1b. Stale 0.7.x build file migration overlay.
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

    # --- 1c. Markdown baseline when no JSON/build baseline exists yet.
    if not existing:
        stem_sibling = output_dir / (Path(json_filename).stem + ".md")
        configured_md = output_dir / md_filename
        md_candidates: list[Path] = [stem_sibling]
        if configured_md != stem_sibling:
            md_candidates.append(configured_md)
        for md_path in md_candidates:
            if md_path.exists():
                try:
                    existing = _load_existing_file_docs_from_md(md_path)
                    break
                except Exception:
                    pass

    # --- 2. Legacy in-progress stable-path overlay (md old JSON sibling).
    if (
        legacy_recovery_path is not None
        and legacy_recovery_path.resolve() != json_path.resolve()
        and legacy_recovery_path.exists()
    ):
        try:
            legacy_doc = read_codedoc_document(legacy_recovery_path)
            if legacy_doc.in_progress:
                legacy_records = records_by_path(legacy_doc)
                if legacy_records:
                    logger.info(
                        "Overlaying %d legacy in-progress record(s) from '%s'.",
                        len(legacy_records),
                        legacy_recovery_path.name,
                    )
                    existing.update(legacy_records)
        except (ConfigError, FileNotFoundError) as exc:
            logger.debug(
                "Optional legacy recovery '%s' rejected: %s",
                legacy_recovery_path.name,
                exc,
            )

    # --- 3. Active dedicated recovery overlay (newest), all formats.
    if (
        live_backup_path is not None
        and live_backup_path.resolve() != json_path.resolve()
        and live_backup_path.exists()
    ):
        try:
            recovery_doc = read_codedoc_document(live_backup_path)
            if recovery_doc.in_progress:
                recovery_records = records_by_path(recovery_doc)
                if recovery_records:
                    logger.info(
                        "Overlaying %d in-progress record(s) from recovery file '%s'.",
                        len(recovery_records),
                        live_backup_path.name,
                    )
                    existing.update(recovery_records)
        except (ConfigError, FileNotFoundError) as exc:
            logger.debug(
                "Optional recovery candidate '%s' rejected: %s",
                live_backup_path.name,
                exc,
            )

    return existing


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


def _cleanup_legacy_recovery(
    legacy_recovery_path: Path | None,
    do_not_delete: set[Path],
) -> None:
    """Remove a migrated legacy in-progress recovery sibling after clean success.

    0.9.8 — for ``md`` runs the pre-0.9.8 live backup was a JSON sibling of the
    Markdown file (e.g. ``codedoc.json`` next to ``codedoc.md``).  After a clean
    completion the new layout writes only the Markdown stable output and the
    dedicated recovery file is removed by :meth:`SafeWriter.delete`; this removes
    the now-migrated legacy sibling too, so the old "only the Markdown remains"
    guarantee holds and the user is never required to delete anything by hand.

    Acts only on a distinct, codedoc-owned, **in-progress** legacy file — never a
    stable output target (``do_not_delete``) and never a clean completed
    document.  For ``json``/``both`` the legacy location *is* the stable JSON
    output (already overwritten cleanly by ``write_project_outputs``), so it is in
    ``do_not_delete`` and is never touched here.  A deletion failure is
    non-fatal: the data is already in the stable output and a leftover is re-read
    harmlessly on the next run.
    """
    if legacy_recovery_path is None or not legacy_recovery_path.exists():
        return
    resolved = legacy_recovery_path.resolve()
    if any(resolved == p.resolve() for p in do_not_delete):
        return
    try:
        document = read_codedoc_document(legacy_recovery_path)
    except (ConfigError, FileNotFoundError):
        return
    if not document.in_progress:
        return
    try:
        legacy_recovery_path.unlink()
        logger.info(
            "Removed migrated legacy in-progress recovery file '%s'.",
            legacy_recovery_path.name,
        )
    except OSError as exc:
        logger.debug(
            "Could not remove migrated legacy recovery file '%s' (non-fatal): %s",
            legacy_recovery_path.name,
            exc,
        )


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
