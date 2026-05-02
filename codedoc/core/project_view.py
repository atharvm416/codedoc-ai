"""Build clean public project documentation views from cached records."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "1.3"


def build_project_view(
    records: list[dict],
    stats: dict,
    entry_file: str | None = None,
    graph_edges: list[dict] | None = None,
) -> dict:
    """Return a compact language-neutral view for public JSON/Markdown output."""
    graph_edges = graph_edges or []
    files = [_clean_file(record) for record in records]
    paths = [file["path"] for file in files]
    internal_by_from, imported_by = _edge_indexes(graph_edges)

    for file in files:
        path = file["path"]
        file["links"] = {
            "internal_dependencies": internal_by_from.get(path, []),
            "imported_by": imported_by.get(path, []),
            "external_dependencies": file.pop("external_dependencies", []),
        }

    folders = _folder_view(files)
    languages = sorted({file["language"] for file in files if file.get("language")})

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
        "dependency_graph": graph_edges,
        "files": files,
    }


def markdown_from_view(view: dict, error_summary: str = "") -> str:
    """Render a public project view as Markdown."""
    lines: list[str] = ["# codedoc project documentation\n\n"]
    project = view.get("project", {})
    run = view.get("run", {})

    lines += [
        "## Project Overview\n\n",
        f"- Entry file: `{project.get('entry_file') or 'not specified'}`\n",
        f"- Files documented: {project.get('file_count', 0)}\n",
        f"- Languages: {', '.join(project.get('languages', [])) or 'unknown'}\n",
        f"- Folders: {', '.join(f'`{f}`' for f in project.get('folders', [])) or 'none'}\n\n",
        "## Run Summary\n\n",
        f"- Files checked: {run.get('files_checked', 0)}\n",
        f"- Files failed: {run.get('files_failed', 0)}\n",
        f"- Files skipped: {run.get('files_skipped', 0)}\n",
        f"- Files reused from cache: {run.get('files_reused', 0)}\n\n",
    ]

    lines += ["## Project Tree\n\n", "```text\n"]
    lines.extend(_render_tree_lines(view.get("tree", {})))
    lines += ["```\n\n"]

    folders = view.get("folders", [])
    if folders:
        lines.append("## Folder Map\n\n")
        for folder in folders:
            lines += [
                f"### {folder['path']}\n\n",
                f"{folder['summary']}\n\n",
                f"- Files: {folder['file_count']}\n",
            ]
            if folder.get("languages"):
                lines.append(f"- Languages: {', '.join(folder['languages'])}\n")
            lines.append("\n")

    edges = view.get("dependency_graph", [])
    if edges:
        lines.append("## Dependency Map\n\n")
        for edge in edges:
            lines.append(f"- `{edge['from']}` -> `{edge['to']}`\n")
        lines.append("\n")

    files = view.get("files", [])
    if files:
        lines.append("## Files\n\n")
        for file in files:
            _append_file_markdown(lines, file)

    if error_summary and error_summary != "No errors.":
        lines += ["## Errors\n\n```text\n", error_summary, "\n```\n"]

    return "".join(lines)


def _clean_file(record: dict) -> dict:
    result = record.get("documentation", {}) or {}
    dependencies = result.get("dependencies_analysis", {})
    external = dependencies.get("external", []) if isinstance(dependencies, dict) else []

    file = {
        "id": record.get("id", ""),
        "hash": record.get("hash", ""),
        "path": record.get("file_path") or result.get("file_path", ""),
        "format": record.get("format", ""),
        "language": result.get("language") or record.get("language", ""),
        "last_processed": record.get("last_processed", ""),
        "description": result.get("description", ""),
        "role_in_system": result.get("role_in_system", ""),
        "imports": result.get("imports", []),
        "functions": result.get("functions", []),
        "classes": result.get("classes", []),
        "exports": result.get("exports", []),
        "key_concepts": result.get("key_concepts", []),
        "usage_example": result.get("usage_example", ""),
        "state": result.get("state", "checked"),
        "external_dependencies": external,
    }
    return {key: value for key, value in file.items() if value not in (None, "", [], {})}


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


def _render_tree_lines(tree: dict, indent: int = 0) -> list[str]:
    lines: list[str] = []
    for name, value in sorted(tree.items()):
        prefix = "  " * indent
        if isinstance(value, dict) and value.get("type") == "file":
            lines.append(f"{prefix}{name}\n")
        else:
            lines.append(f"{prefix}{name}/\n")
            lines.extend(_render_tree_lines(value, indent + 1))
    return lines


def _append_file_markdown(lines: list[str], file: dict) -> None:
    lines += [
        f"### {file.get('path', 'unknown')}\n\n",
        f"**ID:** `{file.get('id', '')}`  \n",
        f"**Format:** {file.get('format', '')}  \n",
        f"**Language:** {file.get('language', '')}  \n\n",
    ]
    if file.get("description"):
        lines += ["**Description:** ", file["description"], "\n\n"]
    if file.get("role_in_system"):
        lines += ["**Role:** ", file["role_in_system"], "\n\n"]
    _append_list(lines, "Imports", file.get("imports", []), code=True)
    links = file.get("links", {})
    _append_list(lines, "Internal Dependencies", links.get("internal_dependencies", []), code=True)
    _append_list(lines, "Imported By", links.get("imported_by", []), code=True)
    _append_list(lines, "External Dependencies", links.get("external_dependencies", []), code=True)
    _append_named_items(lines, "Functions", file.get("functions", []))
    _append_named_items(lines, "Classes", file.get("classes", []))
    _append_list(lines, "Exports", file.get("exports", []), code=True)
    _append_list(lines, "Key Concepts", file.get("key_concepts", []), code=False)
    if file.get("usage_example"):
        lines += ["**Usage Example:**\n\n```text\n", file["usage_example"], "\n```\n\n"]


def _append_list(lines: list[str], title: str, items: list, code: bool) -> None:
    if not items:
        return
    lines.append(f"**{title}:**\n\n")
    for item in items:
        text = f"`{item}`" if code else str(item)
        lines.append(f"- {text}\n")
    lines.append("\n")


def _append_named_items(lines: list[str], title: str, items: list) -> None:
    if not items:
        return
    lines.append(f"**{title}:**\n\n")
    for item in items:
        if isinstance(item, dict):
            lines.append(f"- `{item.get('name', '')}`: {item.get('description', '')}\n")
        else:
            lines.append(f"- `{item}`\n")
    lines.append("\n")


def _top_folder(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[0] if len(parts) > 1 else "."
