"""Output writer for codedoc."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from codedoc.utils.errors import OutputError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.1"


def write_outputs(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown documentation for one file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_stem(result.get("file_path", "unknown"))

    json_path = output_dir / f"{safe}.json"
    md_path = output_dir / f"{safe}.md"

    try:
        _write_json(result, json_path)
        _write_markdown(result, md_path)
    except Exception as exc:
        raise OutputError(str(output_dir), str(exc)) from exc

    logger.debug("Output written: %s + %s", json_path.name, md_path.name)
    return json_path, md_path


def _write_json(result: dict, path: Path) -> None:
    payload = {
        "_schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_path": result.get("file_path", ""),
        "language": result.get("language", ""),
        "extension": result.get("extension", ""),
        "imports": result.get("imports", []),
        "description": result.get("description", ""),
        "role_in_system": result.get("role_in_system", ""),
        "functions": result.get("functions", []),
        "classes": result.get("classes", []),
        "exports": result.get("exports", []),
        "dependencies_analysis": result.get("dependencies_analysis", {}),
        "key_concepts": result.get("key_concepts", []),
        "usage_example": result.get("usage_example", ""),
        "structure": result.get("structure", {}),
        "documentation": result.get("documentation", {}),
        "state": result.get("state", "checked"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(result: dict, path: Path) -> None:
    file_path = result.get("file_path", "unknown")
    description = result.get("description") or "_No description generated._"
    imports = result.get("imports", [])
    functions = result.get("functions", [])
    classes = result.get("classes", [])
    exports = result.get("exports", [])
    role = result.get("role_in_system", "")
    language = result.get("language", "")
    key_concepts = result.get("key_concepts", [])
    usage_example = result.get("usage_example", "")
    dependencies = result.get("dependencies_analysis", {})

    lines: list[str] = [
        f"# {file_path}\n\n",
        f"**Language:** {language}  \n",
        "**State:** checked  \n\n",
        "## Description\n\n",
        f"{description}\n\n",
    ]

    if role:
        lines += ["## Role in system\n\n", f"{role}\n\n"]

    if key_concepts:
        lines.append("## Key concepts\n\n")
        for concept in key_concepts:
            lines.append(f"- {concept}\n")
        lines.append("\n")

    if imports:
        lines.append("## Imports\n\n")
        for imp in imports:
            lines.append(f"- `{imp}`\n")
        lines.append("\n")

    _append_named_items(lines, "Functions", functions)
    _append_named_items(lines, "Classes", classes)

    if exports:
        lines.append("## Exports\n\n")
        for item in exports:
            lines.append(f"- `{item}`\n")
        lines.append("\n")

    warnings = dependencies.get("warnings", []) if isinstance(dependencies, dict) else []
    usage_notes = dependencies.get("usage_notes", []) if isinstance(dependencies, dict) else []
    if usage_notes or warnings:
        lines.append("## Dependency notes\n\n")
        for note in usage_notes:
            if isinstance(note, dict):
                lines.append(f"- `{note.get('import', '')}`: {note.get('used_for', '')}\n")
        for warning in warnings:
            lines.append(f"- Warning: {warning}\n")
        lines.append("\n")

    if usage_example:
        lines += ["## Usage example\n\n", "```text\n", usage_example, "\n```\n"]

    path.write_text("".join(lines), encoding="utf-8")


def write_summary(stats: dict, output_dir: Path, error_summary: str = "") -> Path:
    """Write a run summary and refresh the machine-readable docs index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_index(output_dir)

    summary_path = output_dir / "_summary.md"
    lines = [
        "# codedoc run summary\n\n",
        f"- Files checked: {stats.get('checked', 0)}\n",
        f"- Files failed: {stats.get('failed', 0)}\n",
        f"- Files skipped: {stats.get('skipped', 0)}\n",
    ]
    if error_summary and error_summary != "No errors.":
        lines += ["\n## Errors\n\n```\n", error_summary, "\n```\n"]
    summary_path.write_text("".join(lines), encoding="utf-8")
    return summary_path


def _write_index(output_dir: Path) -> None:
    entries = []
    for json_file in sorted(output_dir.glob("*.json")):
        if json_file.name == "_index.json":
            continue
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        safe = _safe_stem(payload.get("file_path", json_file.stem))
        entries.append(
            {
                "file_path": payload.get("file_path", ""),
                "language": payload.get("language", ""),
                "description": payload.get("description", ""),
                "json": json_file.name,
                "markdown": f"{safe}.md",
                "imports": payload.get("imports", []),
            }
        )

    index = {
        "_schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    (output_dir / "_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_named_items(lines: list[str], title: str, items: list) -> None:
    if not items:
        return
    lines.append(f"## {title}\n\n")
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- **`{item.get('name', '')}`**: {item.get('description', '')}\n")
        else:
            lines.append(f"- `{item}`\n")
    lines.append("\n")


def _safe_stem(file_path: str) -> str:
    """Convert a relative path like src/App.tsx to a safe file stem."""
    stem = re.sub(r"[/\\]", "__", file_path)
    stem = re.sub(r"[^\w\-.]", "_", stem)
    return stem
