"""Output writer for codedoc."""

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

# Internal build file used as a JSON intermediate during MD-only runs.
# Dot-prefix marks it as a system file; it is deleted after a successful
# MD write and preserved (as a data backup) on failure.
BUILD_FILENAME = ".codedoc_build.json"


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
    """
    Write selected combined output file(s).

    For ``format="md"`` an internal JSON build file (``.codedoc_build.json``)
    is written first.  If the Markdown conversion succeeds it is deleted; if it
    fails it is preserved so the user retains their processed results and the
    next run can convert without re-calling the LLM.

    For ``format="both"`` the JSON output itself acts as the durable
    intermediate — if the MD write fails the JSON is already on disk.

    Before overwriting any final output file the function checks that the
    existing file (if any) was produced by codedoc.  If a non-codedoc file is
    found at the target path the run stops with a :class:`ConfigError` so user
    data is never silently overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format not in ("json", "md", "both"):
        raise OutputError(str(output_dir), f"Unsupported output format: {output_format}")

    json_path = output_dir / json_filename if output_format in ("json", "both") else None
    md_path = output_dir / md_filename if output_format in ("md", "both") else None
    # Internal intermediate for MD-only runs.  Not used when JSON is also
    # being written (format="both") because json_path already fills that role.
    build_path = output_dir / BUILD_FILENAME if output_format == "md" else None

    # Ownership check for all output paths — final user-facing files and the
    # internal build file.  Performed BEFORE the try/except so ConfigError
    # propagates directly (not wrapped in OutputError).
    for path in (json_path, md_path, build_path):
        if path:
            _check_file_ownership(path)

    try:
        view = build_project_view(records, stats, entry_file, graph_edges)

        if json_path:
            _write_project_json(view, error_summary, json_path)
            logger.info("JSON output: %s", json_path)

        if md_path:
            if build_path:
                # MD-only: write a complete JSON to the internal build file
                # before attempting the MD conversion.  This guarantees the
                # user's processed results survive a crash or conversion error.
                _write_project_json(view, error_summary, build_path)
                logger.debug("Build file written: %s", build_path.name)

            _write_project_markdown(view, error_summary, md_path)
            logger.info("Markdown output: %s", md_path)

            if build_path:
                _delete_build_file(build_path)

    except Exception as exc:
        if build_path and build_path.exists():
            logger.error(
                "Output write failed — processed results are preserved at '%s'. "
                "Re-run the same command to convert them to Markdown.",
                build_path.name,
            )
        raise OutputError(str(output_dir), str(exc)) from exc

    written = [p.name for p in (json_path, md_path) if p]
    logger.debug("Output written: %s", ", ".join(written))
    return json_path, md_path


# ---------------------------------------------------------------------------
# Ownership guard
# ---------------------------------------------------------------------------

def _check_file_ownership(path: Path) -> None:
    """
    Verify that *path* was written by codedoc before allowing an overwrite.

    If the file exists and does **not** carry valid codedoc metadata
    (``_codedoc`` JSON block for ``.json`` files; ``<!-- codedoc-ai: ... -->``
    comment for ``.md`` files) a :class:`~codedoc.utils.errors.ConfigError`
    is raised so the run stops before any data is lost.

    Files that do not yet exist, or whose extension is not ``.json`` /
    ``.md``, are always allowed through.
    """
    if not path.exists():
        return  # New file — always safe to create.

    suffix = path.suffix.lower()
    try:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        if suffix == ".json":
            data = json.loads(content)
            if isinstance(data, dict) and isinstance(data.get("_codedoc"), dict):
                return  # Valid codedoc JSON — safe to overwrite.
        elif suffix == ".md":
            if re.search(r"<!--\s*codedoc-ai:", content):
                return  # Valid codedoc Markdown — safe to overwrite.
        else:
            return  # Unknown extension — don't block.
    except Exception:
        pass  # Malformed / unreadable file — treat as non-codedoc.

    raise ConfigError(
        f"'{path.name}' already exists but does not appear to be a codedoc "
        "output file.\n"
        "codedoc will not overwrite it to protect your data.\n\n"
        "To resolve this, choose one of:\n"
        f"  • Use a different output directory:   codedoc run --output my_docs/\n"
        f"  • Delete or rename the conflicting file:  {path}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_project_json(view: dict, error_summary: str, path: Path) -> None:
    path.write_text(json_from_view(view, error_summary), encoding="utf-8")


def _write_project_markdown(view: dict, error_summary: str, path: Path) -> None:
    path.write_text(markdown_from_view(view, error_summary), encoding="utf-8")


def _delete_build_file(path: Path) -> None:
    """Remove the internal build file after a successful MD write."""
    try:
        if path.exists():
            path.unlink()
            logger.debug("Build file removed after successful MD write: %s", path.name)
    except Exception as exc:
        logger.warning(
            "Could not remove build file '%s' — you can delete it manually: %s",
            path.name,
            exc,
        )


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
    if error_summary and error_summary != "No errors.":
        lines += ["\n## Errors\n\n```\n", error_summary, "\n```\n"]
    summary_path.write_text("".join(lines), encoding="utf-8")
    return summary_path
