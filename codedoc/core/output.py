"""Output writer for codedoc.

0.8.0 changes
-------------
- The intermediate ``.codedoc_build.json`` write for ``format="md"`` has been
  removed.  Crash safety for MD-only runs is now provided by the always-on
  live JSON backup written by ``SafeWriter`` throughout the pipeline run.
  ``write_project_outputs`` writes the final Markdown directly without an
  intermediate build file.
- ``BUILD_FILENAME`` is kept as a constant so ``_load_existing_file_docs``
  can still read and migrate stale ``.codedoc_build.json`` files left by
  0.7.x runs.
- ``_check_file_ownership`` is unchanged — it still validates both ``.json``
  and ``.md`` files before any overwrite.
"""

from __future__ import annotations

from pathlib import Path

from codedoc.core.block_manager import atomic_write_text
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.markdown_view import markdown_from_view
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.utils.errors import ConfigError, OutputError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_JSON = "codedoc.json"
PROJECT_MARKDOWN = "codedoc.md"

# Kept for reading/migrating stale 0.7.x build files.  No longer written by
# write_project_outputs in 0.8.0.
BUILD_FILENAME = ".codedoc_build.json"


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

    # For md-only runs the live backup is a JSON sibling of the MD file.
    # A foreign sibling would block SafeWriter.initialize_empty() after
    # scanning — same acceptance rules as SafeWriter.load(): the file is
    # accepted if it does not exist or contains a _codedoc key.
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


def read_existing_records(path: Path) -> dict[str, dict] | None:
    """Read per-file records from a codedoc JSON output file, read-only.

    Returns ``{rel_path: record}`` for a codedoc-owned JSON file, an empty
    dict for an owned file with no records, and ``None`` when the file is
    missing, unreadable, or not codedoc-owned.  Never writes or deletes
    anything — safe for planning and dry-run.

    0.9.3: parsing/validation is delegated to the centralized document reader.
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
) -> tuple[Path | None, Path | None]:
    """Write the final combined output file(s).

    For ``format="json"`` and ``format="both"`` the JSON output also acts as
    the live backup written throughout the run, so ``write_project_outputs``
    simply overwrites it with the final clean payload (no ``_crash_safety``
    banner, no ``status = "in_progress"``).

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
        view = build_project_view(records, stats, entry_file, graph_edges)

        # Render both complete payloads before mutating any final target, so a
        # render failure can never leave one artifact rewritten and the other
        # stale.  ``both`` mode guarantees per-artifact atomicity, not a
        # cross-file transaction.
        json_text = json_from_view(view, error_summary) if json_path else None
        md_text = markdown_from_view(view, error_summary) if md_path else None

        # Markdown first, JSON last: the JSON path is also the live backup, so
        # it must remain the prior valid backup until Markdown has succeeded.
        # If Markdown fails, the previous JSON live backup is untouched.  If the
        # final JSON replacement fails after Markdown, no target is partially
        # truncated — Markdown holds the new document and JSON holds the prior
        # complete live backup.
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

    0.9.3: ownership is decided by the centralized document reader.  This
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
        f"  • Use a different output directory:   codedoc run --output my_docs/\n"
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

    *paths* maps a logical artifact name (e.g. ``"json_live_backup"``,
    ``"markdown"``, ``"live_backup"``, ``"error_log"``) to its target ``Path``.
    ``None`` values are ignored.  The check is read-only: targets are normalized
    to absolute paths without being created, existing aliases are resolved where
    possible, and case behavior is detected from the target filesystem without
    creating probe files. Raises :class:`ConfigError` naming both
    logical artifacts when two of them resolve to the same path.

    The intentional final-JSON / live-backup phase alias is expressed by passing
    that single path once under one logical name (``"json_live_backup"``), so it
    is never mistaken for a collision.  The helper is generic: any future
    generated artifact can join the same validation call.
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
    """Backward compatible summary writer for older callers."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / PROJECT_MARKDOWN
    lines = [
        "# codedoc run summary\n\n",
        f"- Files checked: {stats.get('checked', 0)}\n",
        f"- Files failed: {stats.get('failed', 0)}\n",
        f"- Files skipped: {stats.get('skipped', 0)}\n",
        f"- Files reused from cache: {stats.get('reused', 0)}\n",
    ]
    if error_summary:
        lines += ["\n## Errors\n\n```\n", error_summary, "\n```\n"]
    # 0.9.6: route through the canonical atomic writer so a failed write leaves
    # the prior file intact instead of truncating it in place.
    atomic_write_text(summary_path, "".join(lines))
    return summary_path
