"""
Directory scanner.

Walks a project root, finds all supported files, and detects the
language of each file deterministically from its extension.
No LLM is involved here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Map extension → language tag used by parser factory
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py":    "python",
    ".ts":    "typescript",
    ".tsx":   "tsx",
    ".js":    "javascript",
    ".jsx":   "jsx",
    ".dart":  "dart",
    ".java":  "java",
    ".cs":    "csharp",
    ".html":  "html",
    ".htm":   "html",
    ".kt":    "kotlin",
    ".swift": "swift",
    ".go":    "go",
    ".rb":    "ruby",
    ".rs":    "rust",
    ".cpp":   "cpp",
    ".c":     "c",
    ".h":     "c",
    ".hpp":   "cpp",
}

# Directories that should always be skipped
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "env", "myenv", ".env", ".hg", ".svn", "site-packages", "dist-packages",
    "dist", "build", ".next", ".nuxt", "target",
    "docs_output", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def scan_files(
    root: Path,
    supported_extensions: list[str],
    max_file_size_kb: int = 500,
    skip_dirs: list[str] | None = None,
    ignore_paths: list[str] | None = None,
) -> list[dict]:
    """
    Walk root recursively and return a list of file descriptors.

    Each descriptor:
        {
            "path": Path,           absolute path
            "rel_path": str,        relative to root (forward slashes)
            "language": str,        detected language tag
            "extension": str,       e.g. ".py"
        }
    """
    ext_set = {e.lower() for e in supported_extensions}
    skip_set = {d.lower() for d in SKIP_DIRS}
    if skip_dirs:
        skip_set.update(d.lower() for d in skip_dirs)
    ignore_prefixes = _normalise_ignore_paths(ignore_paths or [])
    results: list[dict] = []
    skipped_large = 0
    _walk.skipped_dirs = 0

    for file_path in _walk(root, skip_set, ignore_prefixes, root):
        ext = file_path.suffix.lower()
        if ext not in ext_set:
            continue

        # Skip files that are too large
        try:
            size_kb = file_path.stat().st_size / 1024
        except OSError:
            continue
        if size_kb > max_file_size_kb:
            logger.warning("Skipping large file (%dkb): %s", int(size_kb), file_path)
            skipped_large += 1
            continue

        language = EXTENSION_LANGUAGE_MAP.get(ext, "generic")
        rel = file_path.relative_to(root).as_posix()

        results.append({
            "path": file_path,
            "rel_path": rel,
            "language": language,
            "extension": ext,
        })

    skipped_dirs = getattr(_walk, "skipped_dirs", 0)
    logger.info(
        "Scanner found %d supported file(s) in %s (skipped %d directorie(s), %d large file(s))",
        len(results),
        root,
        skipped_dirs,
        skipped_large,
    )
    return results


def _walk(
    root: Path,
    skip_dirs: set[str],
    ignore_prefixes: set[str] | None = None,
    scan_root: Path | None = None,
) -> Iterator[Path]:
    """Yield all files under root, skipping ignored directories."""
    if ignore_prefixes is not None:
        _walk.ignore_prefixes = ignore_prefixes
    if scan_root is not None:
        _walk.scan_root = scan_root

    try:
        entries = list(root.iterdir())
    except OSError as exc:
        _walk.skipped_dirs += 1
        logger.warning("Skipping unreadable directory %s: %s", root, exc)
        return

    for item in entries:
        rel = item.relative_to(_walk.scan_root).as_posix()
        if item.is_dir():
            if (
                item.name.lower() in skip_dirs
                or item.name.startswith(".")
                or _is_ignored(rel, _walk.ignore_prefixes)
            ):
                _walk.skipped_dirs += 1
                continue
            yield from _walk(item, skip_dirs, _walk.ignore_prefixes, _walk.scan_root)
        elif item.is_file():
            if _is_ignored(rel, _walk.ignore_prefixes):
                continue
            yield item


def _normalise_ignore_paths(paths: list[str]) -> set[str]:
    prefixes: set[str] = set()
    for raw in paths:
        cleaned = str(raw).strip().replace("\\", "/")
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("/")
        cleaned = cleaned.rstrip("/")
        if cleaned:
            prefixes.add(cleaned.lower())
    return prefixes


def _is_ignored(rel_path: str, ignore_prefixes: set[str]) -> bool:
    rel = rel_path.replace("\\", "/").strip("/").lower()
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in ignore_prefixes)


def detect_entry_file(root: Path, hint: str | None) -> Path | None:
    """
    Resolve the entry file.
    If hint is given, resolve it relative to root.
    Otherwise try common entry file names in order.
    """
    common_entries = [
        "index.html", "main.tsx", "main.ts", "main.js",
        "main.py", "main.dart", "Main.java", "Program.cs",
    ]

    if hint:
        candidate = root / hint
        if candidate.exists():
            return candidate
        logger.warning("Specified entry file '%s' not found in %s", hint, root)
        return None

    for name in common_entries:
        candidate = root / name
        if candidate.exists():
            logger.info("Auto-detected entry file: %s", candidate)
            return candidate

    logger.warning("No entry file detected. Will process all files.")
    return None
