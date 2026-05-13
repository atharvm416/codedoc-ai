"""Build clean public project documentation views from cached records."""

from __future__ import annotations

import json
import re
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
    dependency_catalog = _dependency_catalog(files)

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
        "dependency_catalog": dependency_catalog,
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

    catalog = view.get("dependency_catalog", [])
    if catalog:
        lines.append("## Dependency Catalog\n\n")
        for dependency in catalog:
            lines += [
                f"### {dependency['name']}\n\n",
                f"- Type: {dependency.get('type', 'unknown')}\n",
                f"- Used by: {dependency.get('file_count', 0)} file(s)\n",
            ]
            if dependency.get("used_for"):
                lines.append(f"- Used for: {dependency['used_for']}\n")
            lines.append("\n")

    files = view.get("files", [])
    if files:
        lines.append("## Files\n\n")
        for file in files:
            _append_file_markdown(lines, file)

    if error_summary and error_summary != "No errors.":
        lines += ["## Errors\n\n```text\n", error_summary, "\n```\n"]

    return "".join(lines)


def json_from_view(view: dict, error_summary: str = "") -> str:
    """Render a public project view as formatted JSON."""
    payload = dict(view)
    if error_summary and error_summary != "No errors.":
        payload["errors"] = error_summary
    return json.dumps(payload, indent=2, ensure_ascii=False)


def markdown_to_view(markdown: str) -> dict:
    """Parse codedoc Markdown output back into the public JSON view shape."""
    project = _parse_project_overview(markdown)
    run = _parse_run_summary(markdown)
    files = _parse_markdown_files(markdown)
    edges = _parse_dependency_edges(markdown)
    dependency_catalog = _parse_dependency_catalog(markdown)

    for file in files:
        path = file["path"]
        links = file.setdefault("links", {})
        links.setdefault(
            "internal_dependencies",
            sorted(edge["to"] for edge in edges if edge["from"] == path),
        )
        links.setdefault(
            "imported_by",
            sorted(edge["from"] for edge in edges if edge["to"] == path),
        )
        links.setdefault("external_dependencies", [])

    folders = _folder_view(files) if files else []
    languages = sorted({file["language"] for file in files if file.get("language")})
    entry_file = project.get("entry_file")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "entry_file": None if entry_file in (None, "not specified") else entry_file,
            "file_count": project.get("file_count", len(files)),
            "languages": project.get("languages") or languages,
            "folders": project.get("folders") or [folder["path"] for folder in folders],
        },
        "run": {
            "files_checked": run.get("files_checked", 0),
            "files_failed": run.get("files_failed", 0),
            "files_skipped": run.get("files_skipped", 0),
            "files_reused": run.get("files_reused", 0),
            "files_documented": len(files),
        },
        "tree": _tree([file["path"] for file in files]),
        "folders": folders,
        "dependency_catalog": dependency_catalog,
        "dependency_graph": edges,
        "files": files,
    }


def json_from_markdown(markdown: str) -> str:
    """Convert codedoc Markdown output to formatted JSON without using an LLM."""
    return json_from_view(markdown_to_view(markdown))


def markdown_from_json(data: str | dict, error_summary: str = "") -> str:
    """Convert codedoc JSON output to Markdown without using an LLM."""
    view = json.loads(data) if isinstance(data, str) else data
    return markdown_from_view(view, error_summary)


def _clean_file(record: dict) -> dict:
    result = record.get("documentation", {}) or {}
    dependencies = result.get("dependencies_analysis", {})
    external = dependencies.get("external", []) if isinstance(dependencies, dict) else []
    usage_notes = dependencies.get("usage_notes", []) if isinstance(dependencies, dict) else []
    dependency_refs = (
        dependencies.get("dependency_refs", []) if isinstance(dependencies, dict) else []
    )
    catalog_updates = (
        dependencies.get("catalog_updates", []) if isinstance(dependencies, dict) else []
    )

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
        "dependency_refs": dependency_refs,
        "dependency_usage": _dependency_usage_map(usage_notes),
        "dependency_catalog_updates": _clean_catalog_updates(catalog_updates),
    }
    return {key: value for key, value in file.items() if value not in (None, "", [], {})}


def _parse_project_overview(markdown: str) -> dict:
    section = _section(markdown, "Project Overview")
    return {
        "entry_file": _strip_code(_bullet_value(section, "Entry file")),
        "file_count": _parse_int(_bullet_value(section, "Files documented")),
        "languages": _parse_csv(_bullet_value(section, "Languages"), empty_values={"unknown"}),
        "folders": _parse_csv(_bullet_value(section, "Folders"), empty_values={"none"}),
    }


def _parse_run_summary(markdown: str) -> dict:
    section = _section(markdown, "Run Summary")
    return {
        "files_checked": _parse_int(_bullet_value(section, "Files checked")),
        "files_failed": _parse_int(_bullet_value(section, "Files failed")),
        "files_skipped": _parse_int(_bullet_value(section, "Files skipped")),
        "files_reused": _parse_int(_bullet_value(section, "Files reused from cache")),
    }


def _parse_dependency_edges(markdown: str) -> list[dict]:
    section = _section(markdown, "Dependency Map")
    edges: list[dict] = []
    for match in re.finditer(r"^- `(.+?)` -> `(.+?)`$", section, flags=re.MULTILINE):
        edges.append(
            {
                "from": match.group(1),
                "to": match.group(2),
                "type": "internal_import",
            }
        )
    return edges


def _parse_markdown_files(markdown: str) -> list[dict]:
    section = _section(markdown, "Files")
    if not section:
        return []

    chunks = re.split(r"(?m)^### ", section)
    files: list[dict] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        path, _, body = chunk.partition("\n")
        file: dict[str, Any] = {"path": path.strip()}
        _set_if_value(file, "id", _strip_code(_bold_value(body, "ID")))
        _set_if_value(file, "format", _bold_value(body, "Format"))
        _set_if_value(file, "language", _bold_value(body, "Language"))
        _set_if_value(file, "description", _paragraph_after_label(body, "Description"))
        _set_if_value(file, "role_in_system", _paragraph_after_label(body, "Role"))
        _set_if_value(file, "imports", _list_after_label(body, "Imports", code=True))
        _set_if_value(file, "functions", _named_items_after_label(body, "Functions"))
        _set_if_value(file, "classes", _named_items_after_label(body, "Classes"))
        _set_if_value(file, "exports", _list_after_label(body, "Exports", code=True))
        _set_if_value(file, "key_concepts", _list_after_label(body, "Key Concepts"))

        links = {
            "internal_dependencies": _list_after_label(
                body,
                "Internal Dependencies",
                code=True,
            ),
            "imported_by": _list_after_label(body, "Imported By", code=True),
            "external_dependencies": _list_after_label(
                body,
                "External Dependencies",
                code=True,
            ),
        }
        if any(links.values()):
            file["links"] = links

        usage = _fenced_text_after_label(body, "Usage Example")
        _set_if_value(file, "usage_example", usage)
        files.append(file)

    return files


def _parse_dependency_catalog(markdown: str) -> list[dict]:
    section = _section(markdown, "Dependency Catalog")
    if not section:
        return []

    dependencies: list[dict] = []
    chunks = re.split(r"(?m)^### ", section)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, body = chunk.partition("\n")
        dependency = {
            "name": name.strip(),
            "type": _bullet_value(body, "Type") or "unknown",
            "file_count": _parse_file_count(_bullet_value(body, "Used by")),
        }
        _set_if_value(dependency, "used_for", _bullet_value(body, "Used for"))
        dependencies.append(dependency)
    return dependencies


def _section(markdown: str, title: str) -> str:
    match = re.search(
        rf"(?ms)^## {re.escape(title)}\n\n(.*?)(?=^## |\Z)",
        markdown,
    )
    return match.group(1).strip() if match else ""


def _bullet_value(section: str, label: str) -> str:
    match = re.search(rf"(?m)^- {re.escape(label)}: (.*)$", section)
    return match.group(1).strip() if match else ""


def _bold_value(body: str, label: str) -> str:
    match = re.search(rf"(?m)^\*\*{re.escape(label)}:\*\*\s*(.*?)(?:  )?$", body)
    return match.group(1).strip() if match else ""


def _paragraph_after_label(body: str, label: str) -> str:
    match = re.search(
        rf"(?ms)^\*\*{re.escape(label)}:\*\*\s*(.*?)(?=\n\n\*\*|\Z)",
        body,
    )
    return match.group(1).strip() if match else ""


def _list_after_label(body: str, label: str, code: bool = False) -> list[str]:
    match = re.search(
        rf"(?ms)^\*\*{re.escape(label)}:\*\*\n\n(.*?)(?=\n\n\*\*|\Z)",
        body,
    )
    if not match:
        return []

    items = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        value = line[2:].strip()
        items.append(_strip_code(value) if code else value)
    return items


def _named_items_after_label(body: str, label: str) -> list[dict]:
    items = []
    for item in _list_after_label(body, label):
        match = re.match(r"`([^`]+)`(?::\s*(.*))?$", item)
        if match:
            items.append({"name": match.group(1), "description": match.group(2) or ""})
        else:
            items.append({"name": item, "description": ""})
    return items


def _fenced_text_after_label(body: str, label: str) -> str:
    match = re.search(
        rf"(?ms)^\*\*{re.escape(label)}:\*\*\n\n```text\n(.*?)\n```",
        body,
    )
    return match.group(1).strip() if match else ""


def _parse_csv(value: str, empty_values: set[str]) -> list[str]:
    cleaned = value.strip()
    if not cleaned or cleaned in empty_values:
        return []
    return [_strip_code(item.strip()) for item in cleaned.split(",") if item.strip()]


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_file_count(value: str) -> int:
    match = re.match(r"(\d+)", value.strip())
    return int(match.group(1)) if match else 0


def _strip_code(value: str) -> str:
    value = value.strip()
    if value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def _set_if_value(target: dict, key: str, value: Any) -> None:
    if value not in (None, "", [], {}):
        target[key] = value


def _dependency_usage_map(usage_notes: list) -> dict[str, str]:
    usage: dict[str, str] = {}
    for note in usage_notes:
        if not isinstance(note, dict):
            continue
        name = _normalize_dependency_name(note.get("import", ""))
        used_for = str(note.get("used_for", "")).strip()
        if name and used_for and name not in usage:
            usage[name] = used_for
    return usage


def _clean_catalog_updates(catalog_updates: list) -> list[dict]:
    updates = []
    for item in catalog_updates:
        if not isinstance(item, dict):
            continue
        name = _normalize_dependency_name(item.get("name", ""))
        if not name:
            continue
        update = {
            "name": name,
            "type": item.get("type") if item.get("type") in ("internal", "external") else _dependency_type(name),
            "used_for": str(item.get("used_for", "")).strip(),
        }
        updates.append({key: value for key, value in update.items() if value not in ("", None)})
    return updates


def _dependency_catalog(files: list[dict]) -> list[dict]:
    catalog: dict[str, dict] = {}

    for file in files:
        path = file.get("path", "")
        usage = file.pop("dependency_usage", {})
        catalog_updates = file.pop("dependency_catalog_updates", [])
        dependency_refs = [
            _normalize_dependency_name(name)
            for name in file.pop("dependency_refs", [])
            if _normalize_dependency_name(name)
        ]
        links = file.get("links", {})

        for update in catalog_updates:
            name = update["name"]
            item = catalog.setdefault(
                name,
                {
                    "name": name,
                    "type": update.get("type", _dependency_type(name)),
                    "used_for": update.get("used_for", ""),
                    "files": set(),
                },
            )
            item["files"].add(path)
            item["type"] = _merge_dependency_type(item.get("type"), update.get("type"))
            if _should_replace_catalog_text(item.get("used_for", ""), update.get("used_for", "")):
                item["used_for"] = update["used_for"]

        for dependency in dependency_refs:
            name = _normalize_dependency_name(dependency)
            if not name:
                continue
            item = catalog.get(name)
            if not item:
                if name not in usage:
                    continue
                item = catalog.setdefault(
                    name,
                    {
                        "name": name,
                        "type": _dependency_type(name),
                        "used_for": "",
                        "files": set(),
                    },
                )
            item["files"].add(path)
            if not item["used_for"] and usage.get(name):
                item["used_for"] = usage[name]

        for dependency in links.get("external_dependencies", []):
            name = _normalize_dependency_name(dependency)
            if not name:
                continue
            item = catalog.get(name)
            if not item:
                if name not in usage:
                    continue
                item = catalog.setdefault(
                    name,
                    {
                        "name": name,
                        "type": "external",
                        "used_for": "",
                        "files": set(),
                    },
                )
            item["files"].add(path)
            item["type"] = _merge_dependency_type(item.get("type"), "external")
            if not item["used_for"] and usage.get(name):
                item["used_for"] = usage[name]

        for dependency, used_for in usage.items():
            if not dependency:
                continue
            item = catalog.get(dependency)
            if not item:
                continue
            item["files"].add(path)
            if not item["used_for"]:
                item["used_for"] = used_for

    result = []
    for item in catalog.values():
        files_used = sorted(path for path in item.pop("files") if path)
        item["files"] = files_used
        item["file_count"] = len(files_used)
        result.append(item)

    return sorted(result, key=lambda item: (-item["file_count"], item["name"]))


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


def _merge_dependency_type(existing: str | None, candidate: str | None) -> str:
    if existing == "internal" or candidate == "internal":
        return "internal"
    if existing == "external" or candidate == "external":
        return "external"
    return candidate or existing or "unknown"


def _normalize_dependency_name(name: Any) -> str:
    value = str(name or "").strip()
    if value.startswith("package:"):
        value = value[len("package:") :]
    return value


def _dependency_type(name: str) -> str:
    if name.startswith(".") or "/" in name:
        return "internal"
    return "external"


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
