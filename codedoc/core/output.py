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

import json
import re
from pathlib import Path

from codedoc.core.project_view import build_project_view, json_from_view, markdown_from_view
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
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("_codedoc"), dict):
        return None
    return {
        f["path"]: f
        for f in data.get("files", [])
        if isinstance(f, dict) and f.get("path")
    }


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

        if json_path:
            _write_project_json(view, error_summary, json_path)
            logger.info("JSON output: %s", json_path)

        if md_path:
            _write_project_markdown(view, error_summary, md_path)
            logger.info("Markdown output: %s", md_path)

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
    """
    if not path.exists():
        return True

    suffix = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        if suffix == ".json":
            data = json.loads(content)
            return isinstance(data, dict) and isinstance(data.get("_codedoc"), dict)
        if suffix == ".md":
            return bool(re.search(r"<!--\s*codedoc-ai:", content))
        return True
    except Exception:
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
# Internal helpers
# ---------------------------------------------------------------------------

def _write_project_json(view: dict, error_summary: str, path: Path) -> None:
    path.write_text(json_from_view(view, error_summary), encoding="utf-8")


def _write_project_markdown(view: dict, error_summary: str, path: Path) -> None:
    path.write_text(markdown_from_view(view, error_summary), encoding="utf-8")


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
    summary_path.write_text("".join(lines), encoding="utf-8")
    return summary_path
