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
    "env", "dist", "build", ".next", ".nuxt", "target",
    "docs_output", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def scan_files(
    root: Path,
    supported_extensions: list[str],
    max_file_size_kb: int = 500,
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
    results: list[dict] = []

    for file_path in _walk(root):
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
            continue

        language = EXTENSION_LANGUAGE_MAP.get(ext, "generic")
        rel = file_path.relative_to(root).as_posix()

        results.append({
            "path": file_path,
            "rel_path": rel,
            "language": language,
            "extension": ext,
        })

    logger.info("Scanner found %d file(s) in %s", len(results), root)
    return results


def _walk(root: Path) -> Iterator[Path]:
    """Yield all files under root, skipping ignored directories."""
    for item in root.iterdir():
        if item.is_dir():
            if item.name in SKIP_DIRS or item.name.startswith("."):
                continue
            yield from _walk(item)
        elif item.is_file():
            yield item


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