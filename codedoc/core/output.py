"""Write final artifacts with ownership checks and atomic replacement.

Every output format writes only the exact selected final target(s) and relies on
``SafeWriter`` recovery for in-progress crash-safety — there is no intermediate
build file.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from codedoc.core.block_manager import atomic_write_text
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.io_diagnostics import category_reason, classify_os_error, describe_cause
from codedoc.core.markdown_view import markdown_from_view
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.utils.errors import ConfigError, OutputError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_JSON = "codedoc.json"
PROJECT_MARKDOWN = "codedoc.md"


def inspect_output_ownership(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
    live_backup_path: Path | None = None,
) -> list[dict]:
    """Read-only ownership inspection of every final output target.

    Returns a list of conflict dicts — ``{"path": str, "message": str}`` — one
    per foreign-owned target.  Never raises and never touches the filesystem
    beyond reading, so dry-run can report conflicts as warnings while a real
    run converts the first conflict into a :class:`ConfigError` via
    :func:`preflight_output_targets`.

    When the output directory does not yet exist all targets are new and the
    list is empty.
    """
    if not output_dir.exists():
        return []

    conflicts: list[dict] = []

    def _add_conflict(path: Path) -> None:
        conflicts.append({"path": str(path), "message": _foreign_file_message(path)})

    if output_format in ("json", "both"):
        target = output_dir / json_filename
        if not _is_codedoc_owned(target):
            _add_conflict(target)
    if output_format in ("md", "both"):
        target = output_dir / md_filename
        if not _is_codedoc_owned(target):
            _add_conflict(target)

    # Compatibility for callers using the legacy API. The current pipeline does
    # not pass the dedicated recovery path here because the fixed recovery file
    # is validated through the exact recovery-reader path, not via output-target
    # ownership inspection.
    if output_format == "md" and live_backup_path is not None:
        if not _is_codedoc_owned(live_backup_path):
            _add_conflict(live_backup_path)

    return conflicts


def preflight_output_targets(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
    live_backup_path: Path | None = None,
) -> None:
    """Raise ConfigError if any final output target is foreign-owned.

    Call this before scan_files() and create_provider() so a foreign file
    that would block the final write fails immediately without wasting tokens.
    When the output directory does not yet exist all targets are new — returns
    immediately without raising.
    """
    conflicts = inspect_output_ownership(
        output_dir, output_format, json_filename, md_filename, live_backup_path
    )
    if conflicts:
        raise ConfigError(conflicts[0]["message"])


def preflight_output_accessibility(output_dir: Path) -> None:
    """Verify *output_dir* can actually be written before any provider is created.

    This is a non-destructive, provider-free probe run
    only for a real run that may write output (never for ``--dry-run``).  It:

    - creates *output_dir* when absent (a real run will need it anyway);
    - exercises create → UTF-8 write → flush → fsync → atomic rename → cleanup
      using uniquely named CodeDoc probe files inside the directory;
    - never overwrites or deletes a user file (probe names are random and unique);
    - cleans both probe paths on success and best-effort on failure;
    - raises a classified :class:`OutputError` before provider creation on the
      first failure.

    It is an early diagnostic, not a guarantee — a file can still become locked
    afterwards, which is why the bounded replacement retry and fatal recovery
    semantics remain.  The authoritative recovery-target check stays with
    ``SafeWriter.initialize_empty()``; this probe never touches the recovery file.
    """
    token = uuid.uuid4().hex
    src = output_dir / f".codedoc_preflight_{token}.src.tmp"
    dst = output_dir / f".codedoc_preflight_{token}.dst.tmp"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # create + UTF-8 write + flush + fsync on the unique source probe.
        fd = os.open(str(src), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("codedoc preflight\n")
            handle.flush()
            os.fsync(handle.fileno())
        # atomic rename to the unique destination probe, then remove it.
        os.replace(str(src), str(dst))
    except OSError as exc:
        _silent_remove(src)
        _silent_remove(dst)
        category = classify_os_error(exc)
        cause = describe_cause(exc)
        detail = f"{category_reason(category)} ({cause})" if cause else category_reason(category)
        raise OutputError(
            str(output_dir),
            f"output directory '{output_dir}' is not writable: {detail}. "
            "No provider was contacted — choose a writable output directory or "
            "correct local permissions, then rerun.",
        ) from exc
    finally:
        _silent_remove(src)
        _silent_remove(dst)


def _silent_remove(path: Path) -> None:
    """Remove *path* if present, swallowing cleanup-time OSErrors."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def read_existing_records(path: Path) -> dict[str, dict] | None:
    """Read per-file records from a codedoc JSON output file, read-only.

    Returns ``{rel_path: record}`` for a codedoc-owned JSON file, an empty
    dict for an owned file with no records, and ``None`` when the file is
    missing, unreadable, or not codedoc-owned.  Never writes or deletes
    anything — safe for planning and dry-run.

    Parsing/validation is delegated to the centralized document reader.
    Malformed or foreign files return ``None`` for this optional discovery use;
    ownership is enforced separately by :func:`inspect_output_ownership`.
    """
    if not path.exists():
        return None
    try:
        document = read_codedoc_document(path)
    except (ConfigError, FileNotFoundError):
        return None
    return records_by_path(document)


def write_project_outputs(
    records: list[dict],
    stats: dict,
    output_dir: Path,
    error_summary: str = "",
    output_format: str = "json",
    entry_file: str | None = None,
    graph_edges: list[dict] | None = None,
    json_filename: str = PROJECT_JSON,
    md_filename: str = PROJECT_MARKDOWN,
    reachable_rels: set[str] | frozenset[str] | None = None,
    unresolved_imports_by_path: dict[str, list[str]] | None = None,
) -> tuple[Path | None, Path | None]:
    """Write the final combined output file(s).

    For ``format="json"`` and ``format="both"`` this writes the final clean JSON
    payload (no ``_crash_safety`` banner and no ``status = "in_progress"``).
    In-progress work is staged separately by ``SafeWriter``.

    For ``format="md"`` the Markdown file is written directly without an
    intermediate build file.  If the Markdown conversion fails the live JSON
    backup written by ``SafeWriter`` throughout the run is preserved, so the
    user retains their processed results and the next run can resume.

    Before overwriting any output file the function checks that the existing
    file (if any) was produced by codedoc.  A non-codedoc file at the target
    path stops the run with a :class:`ConfigError`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format not in ("json", "md", "both"):
        raise OutputError(str(output_dir), f"Unsupported output format: {output_format}")

    json_path = output_dir / json_filename if output_format in ("json", "both") else None
    md_path = output_dir / md_filename if output_format in ("md", "both") else None

    for path in (json_path, md_path):
        if path:
            _check_file_ownership(path)

    try:
        view = build_project_view(
            records,
            stats,
            entry_file,
            graph_edges,
            reachable_rels=reachable_rels,
            unresolved_imports_by_path=unresolved_imports_by_path,
        )

        # Render both complete payloads before mutating any final target, so a
        # render failure can never leave one artifact rewritten and the other
        # stale.  ``both`` mode guarantees per-artifact atomicity, not a
        # cross-file transaction.
        json_text = json_from_view(view, error_summary) if json_path else None
        md_text = markdown_from_view(view, error_summary) if md_path else None

        # Markdown first, JSON last. If either stable write fails, the dedicated
        # recovery file still exists because the caller deletes it only after
        # this function returns successfully.
        if md_path is not None and md_text is not None:
            atomic_write_text(md_path, md_text)
            logger.info("Markdown output: %s", md_path)

        if json_path is not None and json_text is not None:
            atomic_write_text(json_path, json_text)
            logger.info("JSON output: %s", json_path)

    except Exception as exc:
        raise OutputError(str(output_dir), str(exc)) from exc

    written = [p.name for p in (json_path, md_path) if p]
    logger.debug("Output written: %s", ", ".join(written))
    return json_path, md_path


# ---------------------------------------------------------------------------
# Ownership guard
# ---------------------------------------------------------------------------

def _is_codedoc_owned(path: Path) -> bool:
    """Read-only check that *path* may be overwritten by codedoc.

    Files that do not yet exist, or whose extension is not ``.json`` / ``.md``,
    are always allowed through.  Malformed or foreign files return ``False``.

    Ownership is decided by the centralized document reader.  This
    intentionally tightens the old behaviour — Markdown that merely *contains*
    a ``<!-- codedoc-ai:`` marker but whose metadata is malformed is now treated
    as foreign and will not be overwritten.
    """
    if not path.exists():
        return True

    suffix = path.suffix.lower()
    if suffix not in (".json", ".md"):
        # Non-CodeDoc target extensions are not ours to guard.
        return True
    try:
        read_codedoc_document(path)
        return True
    except (ConfigError, FileNotFoundError):
        return False


def _foreign_file_message(path: Path) -> str:
    return (
        f"'{path.name}' already exists but does not appear to be a codedoc "
        "output file.\n"
        "codedoc will not overwrite it to protect your data.\n\n"
        "To resolve this, choose one of:\n"
        f"  • Use a different output directory:   codedoc --output my_docs/\n"
        f"  • Delete or rename the conflicting file:  {path}"
    )


def _check_file_ownership(path: Path) -> None:
    """Verify that *path* was written by codedoc before allowing an overwrite.

    Malformed or foreign files raise :class:`ConfigError`.  This is the
    mutation-time guard used by :func:`write_project_outputs`; the read-only
    variant for planning/dry-run is :func:`inspect_output_ownership`.
    """
    if not _is_codedoc_owned(path):
        raise ConfigError(_foreign_file_message(path))


# ---------------------------------------------------------------------------
# Artifact-path collision validation
# ---------------------------------------------------------------------------

def validate_distinct_artifact_paths(paths: dict[str, Path | None]) -> None:
    """Reject two distinct generated artifacts targeting the same path.

    *paths* maps a logical artifact name (e.g. ``"json"``, ``"markdown"``,
    ``"live_backup"``) to its target ``Path``.  ``None`` values are ignored.  The
    check is read-only: targets are normalized to absolute paths without being
    created, existing aliases are resolved where possible, and case behavior is
    detected from the target filesystem without creating probe files. Raises
    :class:`ConfigError` naming both logical artifacts when two of them resolve to
    the same path.

    The dedicated crash-recovery file (``crash_recovery.json``) is its own
    ``"live_backup"`` artifact, always distinct from the final JSON (``"json"``)
    and Markdown (``"markdown"``) targets, so any overlap between them is a real
    collision.  The helper is generic: any future generated artifact can join the
    same call.
    """
    seen: dict[str, tuple[str, Path]] = {}
    for name, path in paths.items():
        if path is None:
            continue
        resolved = Path(path).resolve()
        case_insensitive = _filesystem_is_case_insensitive(resolved.parent)
        key = str(resolved).casefold() if case_insensitive else str(resolved)
        if key in seen:
            other, _other_path = seen[key]
            raise ConfigError(
                f"Output artifacts '{other}' and '{name}' would be written to "
                f"the same path:\n  {Path(path).resolve()}\n"
                "Choose distinct output targets so one cannot overwrite the other."
            )
        seen[key] = (name, resolved)


def _filesystem_is_case_insensitive(path: Path) -> bool:
    """Detect case behavior from an existing ancestor without writing probes."""
    candidate = Path(path).resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        swapped = _swap_case_letter(current.name)
        if not swapped:
            continue
        alternate = current.with_name(swapped)
        try:
            return alternate.exists() and alternate.samefile(current)
        except OSError:
            continue
    return False


def _swap_case_letter(value: str) -> str:
    for index, char in enumerate(value):
        if "a" <= char <= "z":
            return value[:index] + char.upper() + value[index + 1:]
        if "A" <= char <= "Z":
            return value[:index] + char.lower() + value[index + 1:]
    return ""


# ---------------------------------------------------------------------------
# Backward-compatible summary writer
# ---------------------------------------------------------------------------

def write_summary(stats: dict, output_dir: Path, error_summary: str = "") -> Path:
    """Backward compatible summary writer for older callers.

    The helper still writes only aggregate counters, but it now uses the normal
    CodeDoc Markdown envelope so the resulting file is recognized as owned and
    can be safely overwritten by a later pipeline run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / PROJECT_MARKDOWN
    view = build_project_view([], stats, entry_file=stats.get("entry_file"))
    # Route through the canonical atomic writer so a failed write leaves
    # the prior file intact instead of truncating it in place.
    atomic_write_text(summary_path, markdown_from_view(view, error_summary))
    return summary_path
