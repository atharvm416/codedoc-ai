"""Markdown serialization and parsing for codedoc project views.

Extracted from :mod:`codedoc.core.project_view` as part of the internal module
decomposition.  This module owns the Markdown side of the public view:

- rendering a built project view to Markdown (``markdown_from_view``), including
  the lightweight metadata comment and the lossless base64 embedded view;
- parsing Markdown back into the view shape (``markdown_to_view``), using the
  embedded view fast path and the legacy visible-text parser;
- the JSON⇄Markdown conversion helpers (``json_from_markdown``,
  ``markdown_from_json``);
- the embedded-view readers (``read_embedded_view`` /
  ``read_embedded_view_result``) and the comment builders.

Project-view *assembly* (``build_project_view``, ``json_from_view``,
``clean_file_record``, folder/tree/graph/catalog construction, pruning, and
usage-example sanitization) remains in :mod:`codedoc.core.project_view`.  This
module depends on a few pure assembly helpers from there; the dependency is
one-way (``markdown_view`` → ``project_view``).
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from codedoc.core.project_view import (
    SCHEMA_VERSION,
    _folder_view,
    _prune_empty,
    _tree,
    json_from_view,
)
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmbeddedViewResult:
    """Tri-state result of attempting to read a Markdown embedded view.

    ``state`` is one of:

    - ``"absent"``  — no ``codedoc-ai-view-base64`` block is present;
    - ``"valid"``   — a block is present and decoded to a usable view;
    - ``"invalid"`` — a block is present but corrupt / structurally invalid.

    The strict document reader rejects ``"invalid"`` (a corrupt embedded block
    means the document must not be silently trusted via visible prose), while
    the tolerant public ``read_embedded_view()`` collapses both ``"absent"``
    and ``"invalid"`` to ``None`` for conversion compatibility.
    """

    state: str  # "absent", "valid", or "invalid"
    view: dict | None


# Regex that matches the lightweight metadata comment embedded in Markdown output:
#   <!-- codedoc-ai: { ... } -->
_CODEDOC_META_COMMENT_RE = re.compile(
    r"<!--\s*codedoc-ai:\s*(\{.*?\})\s*-->", re.DOTALL
)

# Regex that matches the lossless base64-encoded full view comment:
#   <!-- codedoc-ai-view-base64
#   eyJ...
#   -->
_CODEDOC_VIEW_BASE64_RE = re.compile(
    r"<!--\s*codedoc-ai-view-base64\s*([\s\S]*?)\s*-->"
)


# ---------------------------------------------------------------------------
# Markdown serialisation
# ---------------------------------------------------------------------------

def markdown_from_view(view: dict, error_summary: str = "") -> str:
    """Render a public project view as Markdown.

    A lossless ``<!-- codedoc-ai-view-base64 ... -->`` block is written
    immediately after the lightweight metadata comment.  The visible Markdown
    sections are unchanged; they remain human-readable.  The hidden base64
    block allows :func:`markdown_to_view` to reconstruct the full public JSON
    view without any information loss on subsequent runs.
    """
    project = view.get("project", {})
    lines: list[str] = [
        _build_meta_comment(view, project),
        _build_full_view_comment(view),
        "# codedoc project documentation\n\n",
    ]
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


# ---------------------------------------------------------------------------
# Markdown → view (with lossless embedded-view fast path)
# ---------------------------------------------------------------------------

def markdown_to_view(markdown: str) -> dict:
    """Parse codedoc Markdown output back into the public JSON view shape.

    Fast path: when the Markdown contains a ``codedoc-ai-view-base64`` block
    (written by :func:`markdown_from_view`), the embedded view is decoded and
    returned directly — this is a lossless round-trip.

    Legacy fallback: for Markdown with no embedded block, the visible Markdown
    sections are parsed directly.  This path is lossy
    (dependency metadata and some internal fields are not recoverable from the
    visible text alone) but still produces a best-effort result.
    """
    # Fast path: lossless embedded view takes precedence over the visible parser.
    embedded = read_embedded_view(markdown)
    if embedded is not None:
        return embedded

    # Legacy visible-text parser (Markdown with no embedded block).
    project = _parse_project_overview(markdown)
    run = _parse_run_summary(markdown)
    files = _parse_markdown_files(markdown)
    edges = _parse_dependency_edges(markdown)
    dependency_catalog = _parse_dependency_catalog(markdown)

    for file in files:
        path = file["path"]
        existing_links = file.pop("links", {})
        links = {
            "internal_dependencies": existing_links.get("internal_dependencies")
            or sorted(edge["to"] for edge in edges if edge["from"] == path),
            "imported_by": existing_links.get("imported_by")
            or sorted(edge["from"] for edge in edges if edge["to"] == path),
            "external_dependencies": existing_links.get("external_dependencies", []),
            "sdk_dependencies": existing_links.get("sdk_dependencies", []),
        }
        links = _prune_empty(links)
        if links:
            file["links"] = links

    folders = _folder_view(files) if files else []
    languages = sorted({file["language"] for file in files if file.get("language")})
    entry_file = project.get("entry_file")

    view = {
        "schema_version": SCHEMA_VERSION,
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
    return _prune_empty(view)


def json_from_markdown(markdown: str) -> str:
    """Convert codedoc Markdown output to formatted JSON without using an LLM."""
    return json_from_view(markdown_to_view(markdown))


def markdown_from_json(data: str | dict, error_summary: str = "") -> str:
    """Convert codedoc JSON output to Markdown without using an LLM."""
    view = json.loads(data) if isinstance(data, str) else data
    return markdown_from_view(view, error_summary)


# ---------------------------------------------------------------------------
# Embedded lossless view helpers
# ---------------------------------------------------------------------------

def read_embedded_view_result(markdown: str) -> EmbeddedViewResult:
    """Tri-state read of the ``codedoc-ai-view-base64`` embedded view.

    Returns an :class:`EmbeddedViewResult` distinguishing ``"absent"`` (no
    block) from ``"invalid"`` (block present but corrupt / structurally bad)
    from ``"valid"`` (usable view).  The strict document reader uses this to
    reject ``"invalid"`` rather than silently falling back to visible prose.
    """
    match = _CODEDOC_VIEW_BASE64_RE.search(markdown)
    if not match:
        return EmbeddedViewResult(state="absent", view=None)

    raw = match.group(1).strip()
    try:
        json_bytes = base64.b64decode(raw, validate=True)
        data = json.loads(json_bytes.decode("utf-8"))
    except Exception as exc:
        logger.warning(
            "codedoc-ai-view-base64 block could not be decoded and will be ignored "
            "(falling back to visible Markdown parser): %s",
            exc,
        )
        return EmbeddedViewResult(state="invalid", view=None)

    if not isinstance(data, dict):
        logger.warning(
            "codedoc-ai-view-base64 decoded to %s, expected dict; "
            "falling back to visible Markdown parser.",
            type(data).__name__,
        )
        return EmbeddedViewResult(state="invalid", view=None)

    # Reject crash-safety or in-progress snapshots — these must never be
    # embedded, but guard against it defensively.
    if "_crash_safety" in data:
        logger.warning(
            "codedoc-ai-view-base64 contains a _crash_safety banner; "
            "treating as an incomplete snapshot and ignoring."
        )
        return EmbeddedViewResult(state="invalid", view=None)
    codedoc_meta = data.get("_codedoc", {})
    if isinstance(codedoc_meta, dict) and codedoc_meta.get("status") == "in_progress":
        logger.warning(
            "codedoc-ai-view-base64 is an in-progress snapshot; ignoring."
        )
        return EmbeddedViewResult(state="invalid", view=None)

    # Structural validation: require the three fields that make the view useful.
    required = {"schema_version", "project", "files"}
    missing = required - data.keys()
    if missing:
        logger.warning(
            "codedoc-ai-view-base64 is missing required fields %s; "
            "falling back to visible Markdown parser.",
            sorted(missing),
        )
        return EmbeddedViewResult(state="invalid", view=None)

    return EmbeddedViewResult(state="valid", view=data)


def read_embedded_view(markdown: str) -> dict | None:
    """Extract and decode the lossless embedded view from Markdown.

    Tolerant compatibility wrapper around :func:`read_embedded_view_result`:
    returns the decoded dict only when the block is present and valid, and
    ``None`` for both absent and invalid blocks so existing conversion callers
    fall back to the legacy visible-text parser unchanged.
    """
    result = read_embedded_view_result(markdown)
    return result.view if result.state == "valid" else None


def _public_view_for_embedding(view: dict) -> dict:
    """Return a sanitized copy of *view* safe to embed in Markdown.

    Strips crash-safety markers, the ``_codedoc`` wrapper (which is added by
    :func:`~codedoc.core.project_view.json_from_view` at JSON render time and
    must not appear in the embedded block), and the deprecated run-varying
    ``generated_at`` field (for run determinism).
    """
    excluded = {"_crash_safety", "_codedoc", "generated_at"}
    return {k: v for k, v in view.items() if k not in excluded}


def _build_full_view_comment(view: dict) -> str:
    """Encode *view* as a standard base64 UTF-8 JSON hidden comment block.

    The base64 encoding is necessary because raw generated summaries or
    dependency text can contain ``--`` or ``-->`` which would prematurely
    close an HTML comment and expose or corrupt the hidden metadata.

    The block format is::

        <!-- codedoc-ai-view-base64
        eyJzY2hlbWFfdmVyc2lvbiI6...
        -->
    """
    sanitized = _public_view_for_embedding(view)
    json_bytes = json.dumps(sanitized, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(json_bytes).decode("ascii")
    return f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"


# ---------------------------------------------------------------------------
# Internal helpers — metadata comment builder
# ---------------------------------------------------------------------------

def _build_meta_comment(view: dict, project: dict) -> str:
    """Return a one-line HTML comment embedding CodeDoc metadata for Markdown output.

    Includes ``file_hashes`` so that subsequent ``--format md`` runs can perform
    incremental hash checks without requiring a sibling JSON file.
    """
    files = view.get("files", [])
    file_hashes = {
        f["path"]: f["hash"]
        for f in files
        if isinstance(f, dict) and f.get("path") and f.get("hash")
    }
    meta = {
        "entry_file": project.get("entry_file"),
        "schema_version": view.get("schema_version", SCHEMA_VERSION),
        "file_hashes": file_hashes,
    }
    return f"<!-- codedoc-ai: {json.dumps(meta, ensure_ascii=False)} -->\n"


# ---------------------------------------------------------------------------
# Visible Markdown section parsers (legacy path)
# ---------------------------------------------------------------------------

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
            "sdk_dependencies": _list_after_label(
                body,
                "SDK / Standard Library",
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


# ---------------------------------------------------------------------------
# Visible Markdown parsing utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Markdown render helpers
# ---------------------------------------------------------------------------

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
        f"**Language:** {file.get('language', '')}  \n\n",
        "**Reachable from entry:** "
        f"{'Yes' if file.get('reachable_from_entry', True) else 'No'}  \n\n",
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
    _append_list(lines, "SDK / Standard Library", links.get("sdk_dependencies", []), code=True)
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
