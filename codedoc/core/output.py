"""Output writer for codedoc."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from codedoc.core.project_view import build_project_view, markdown_from_view
from codedoc.utils.errors import OutputError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_JSON = "codedoc.json"
PROJECT_MARKDOWN = "codedoc.md"


def write_project_outputs(
    records: list[dict],
    stats: dict,
    output_dir: Path,
    error_summary: str = "",
    output_format: str = "json",
    entry_file: str | None = None,
    graph_edges: list[dict] | None = None,
) -> tuple[Path | None, Path | None]:
    """Write selected combined output file(s)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format not in ("json", "md", "both"):
        raise OutputError(str(output_dir), f"Unsupported output format: {output_format}")

    selected = _selected_output_names(output_format)
    _remove_unselected_outputs(output_dir, selected)
    json_path = output_dir / PROJECT_JSON if PROJECT_JSON in selected else None
    md_path = output_dir / PROJECT_MARKDOWN if PROJECT_MARKDOWN in selected else None

    try:
        view = build_project_view(records, stats, entry_file, graph_edges)
        if json_path:
            _write_project_json(view, error_summary, json_path)
        if md_path:
            _write_project_markdown(view, error_summary, md_path)
    except Exception as exc:
        raise OutputError(str(output_dir), str(exc)) from exc

    written = [path.name for path in (json_path, md_path) if path]
    logger.debug("Output written: %s", ", ".join(written))
    return json_path, md_path


def write_outputs(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Backward compatible wrapper that writes a one-file project document."""
    record = {
        "id": "",
        "hash": "",
        "file_path": result.get("file_path", ""),
        "format": result.get("extension", "").lstrip("."),
        "language": result.get("language", ""),
        "last_processed": datetime.now(timezone.utc).isoformat(),
        "git_commit": None,
        "author": None,
        "documentation": result,
    }
    return write_project_outputs([record], {"checked": 1, "failed": 0, "skipped": 0}, output_dir)


def _write_project_json(
    view: dict,
    error_summary: str,
    path: Path,
) -> None:
    payload = dict(view)
    if error_summary and error_summary != "No errors.":
        payload["errors"] = error_summary

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_project_markdown(
    view: dict,
    error_summary: str,
    path: Path,
) -> None:
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
    if error_summary and error_summary != "No errors.":
        lines += ["\n## Errors\n\n```\n", error_summary, "\n```\n"]
    summary_path.write_text("".join(lines), encoding="utf-8")
    return summary_path


def _selected_output_names(output_format: str) -> set[str]:
    if output_format == "json":
        return {PROJECT_JSON}
    if output_format == "md":
        return {PROJECT_MARKDOWN}
    return {PROJECT_JSON, PROJECT_MARKDOWN}


def _remove_unselected_outputs(output_dir: Path, keep: set[str]) -> None:
    """Remove old docs so the output directory keeps only selected docs files."""
    for pattern in ("*.json", "*.md"):
        for path in output_dir.glob(pattern):
            if path.name in keep:
                continue
            try:
                path.unlink()
            except OSError as exc:
                logger.debug("Could not remove legacy output %s: %s", path, exc)

