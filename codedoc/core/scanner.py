"""
Directory scanner.

Walks a project root, finds all supported files, and detects the
language of each file deterministically from its extension.
No LLM is involved here.

0.8.1 changes
-------------
- ``SKIP_DIRS`` and ``EXTENSION_LANGUAGE_MAP`` are no longer hardcoded here.
  The scanner receives ``skip_dirs`` and ``extension_language_map`` from the
  caller (resolved by :func:`codedoc.core.loader.load_config`), making both
  fully configurable via ``codedoc.config.json`` or CLI flags.
- ``scan_files`` now accepts ``extension_language_map`` as its primary
  extension/language source.  The legacy ``supported_extensions`` keyword
  argument is kept for backward compatibility with direct callers.
- ``detect_entry_file`` accepts a ``candidates`` list so the auto-entry file
  search is driven by ``DEFAULTS["auto_entry_candidates"]`` rather than a
  hardcoded list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level fallbacks retained only for backward compatibility with direct
# callers that do not go through load_config().  Pipeline code uses the values
# from DEFAULTS["extension_language_map"] and DEFAULTS["auto_entry_candidates"]
# in loader.py.
# ---------------------------------------------------------------------------

_FALLBACK_LANGUAGE_MAP: dict[str, str] = {
    ".py":   "python",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".js":   "javascript",
    ".jsx":  "jsx",
    ".dart": "dart",
    ".java": "java",
    ".cs":   "csharp",
    ".html": "html",
    ".htm":  "html",
    ".kt":   "kotlin",
    ".swift":"swift",
    ".go":   "go",
    ".rb":   "ruby",
    ".rs":   "rust",
    ".cpp":  "cpp",
    ".c":    "c",
    ".h":    "c",
    ".hpp":  "cpp",
}

_FALLBACK_AUTO_ENTRY_CANDIDATES: list[str] = [
    "index.html", "main.tsx", "main.ts", "main.js",
    "main.py", "main.dart", "Main.java", "Program.cs",
]


def scan_files(
    root: Path,
    extension_language_map: dict[str, str] | None = None,
    max_file_size_kb: int = 500,
    skip_dirs: list[str] | None = None,
    ignore_paths: list[str] | None = None,
    *,
    # Deprecated keyword-only parameter kept for backward compatibility.
    # When provided without extension_language_map, a map is built from
    # _FALLBACK_LANGUAGE_MAP for the listed extensions.
    supported_extensions: list[str] | None = None,
) -> list[dict]:
    """
    Walk root recursively and return a list of file descriptors.

    Parameters
    ----------
    root:
        Project root directory to scan.
    extension_language_map:
        Maps file extensions (lower-case, with leading dot) to language tags.
        Extensions in the map are automatically supported — no separate
        ``supported_extensions`` list needed.  When *None* and
        ``supported_extensions`` is given, a compatibility map is built.
    max_file_size_kb:
        Files larger than this are skipped (default 500 KB).
    skip_dirs:
        Directory names to skip (case-insensitive).  Typically resolved from
        ``config["skip_dirs"]`` by the pipeline, which also auto-appends the
        output directory.  Directories whose names begin with ``.`` are always
        skipped regardless of this list.
    ignore_paths:
        Project-relative paths (files or directory subtrees) to exclude.
    supported_extensions:
        **Deprecated.** Kept for callers that do not supply
        ``extension_language_map``.  Language is set to the fallback map value
        or ``"generic"`` for unknown extensions.

    Returns
    -------
    list[dict]
        Each descriptor has keys: ``path`` (absolute :class:`Path`),
        ``rel_path`` (forward-slash string), ``language`` (str), ``extension``.
    """
    # Backward-compat guard: legacy positional callers may pass a list/tuple of
    # extensions as the second argument, e.g. scan_files(root, [".py", ".ts"]).
    # Since the old second parameter was `supported_extensions`, redirect to the
    # legacy path instead of crashing with AttributeError on list.items().
    if isinstance(extension_language_map, (list, tuple)):
        supported_extensions = list(extension_language_map)
        extension_language_map = None

    # Resolve the effective extension → language map.
    if extension_language_map is None:
        if supported_extensions is not None:
            extension_language_map = {
                ext.lower(): _FALLBACK_LANGUAGE_MAP.get(ext.lower(), "generic")
                for ext in supported_extensions
            }
        else:
            extension_language_map = {}

    ext_map = {e.lower(): lang for e, lang in extension_language_map.items()}
    skip_set = {d.lower() for d in (skip_dirs or [])}
    ignore_prefixes = _normalise_ignore_paths(ignore_paths or [])
    results: list[dict] = []
    skipped_large = 0
    _walk.skipped_dirs = 0

    for file_path in _walk(root, skip_set, ignore_prefixes, root):
        ext = file_path.suffix.lower()
        if ext not in ext_map:
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

        language = ext_map.get(ext, "generic")
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


def detect_entry_file(
    root: Path,
    hint: str | None,
    candidates: list[str] | None = None,
) -> Path | None:
    """
    Resolve the entry file.

    Parameters
    ----------
    root:
        Project root directory.
    hint:
        Entry file path relative to *root*, as provided via ``--entry``.
        When given, auto-detection is skipped.
    candidates:
        Ordered list of file names to try when no *hint* is given.
        Resolved from ``config["auto_entry_candidates"]`` by the pipeline.
        Falls back to :data:`_FALLBACK_AUTO_ENTRY_CANDIDATES` when ``None``.
    """
    auto_entries = candidates if candidates is not None else _FALLBACK_AUTO_ENTRY_CANDIDATES

    if hint:
        candidate = root / hint
        if candidate.exists():
            return candidate
        logger.warning("Specified entry file '%s' not found in %s", hint, root)
        return None

    for name in auto_entries:
        candidate = root / name
        if candidate.exists():
            logger.info("Auto-detected entry file: %s", candidate)
            return candidate

    logger.warning("No entry file detected. Will process all files.")
    return None
