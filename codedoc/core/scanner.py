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

import stat as stat_module
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
    # Safety control (0.9.6).  Keyword-only so existing positional callers stay
    # compatible.  When False (the default) every symlinked directory and file
    # is skipped, which prevents both symlink cycles and escapes outside the
    # project root.  When True, links are followed only when their resolved
    # target exists, has the expected type, and is contained by the resolved
    # project root.
    follow_symlinks: bool = False,
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
    follow_symlinks:
        When *False* (default) symlinked directories and files are skipped, so
        the scan never follows a link cycle and never escapes the project root.
        When *True* links are resolved strictly and followed only when the
        target exists, has the expected type, and resolves inside *root*.
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

    walker = _Walker(
        scan_root=root,
        skip_dirs=skip_set,
        ignore_prefixes=ignore_prefixes,
        follow_symlinks=follow_symlinks,
    )
    for file_path in walker.walk(root):
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

    skipped_dirs = walker.skipped_dirs
    logger.info(
        "Scanner found %d supported file(s) in %s (skipped %d directorie(s), %d large file(s))",
        len(results),
        root,
        skipped_dirs,
        skipped_large,
    )
    return results


class _Walker:
    """Iterative, symlink-safe directory walker with per-scan state.

    State (``scan_root``, ``skip_dirs``, ``ignore_prefixes``, ``skipped_dirs``,
    and the visited-identity sets) is held on the instance rather than on the
    function object, so the walk is re-entrant and two concurrent or sequential
    scans never share state.

    The walk uses an explicit stack instead of recursion, so a deep but acyclic
    tree cannot raise :class:`RecursionError`.  Every traversed directory's
    resolved identity is recorded, so a symlink/junction cycle or two aliases to
    the same real directory are visited at most once.  When ``follow_symlinks``
    is True, resolved file identities are tracked the same way so two aliases to
    one real file produce at most one descriptor.
    """

    def __init__(
        self,
        scan_root: Path,
        skip_dirs: set[str],
        ignore_prefixes: set[str],
        follow_symlinks: bool = False,
    ) -> None:
        self.scan_root = scan_root
        self.skip_dirs = skip_dirs
        self.ignore_prefixes = ignore_prefixes
        self.follow_symlinks = follow_symlinks
        self.skipped_dirs = 0
        self._resolved_root: Path = scan_root
        self._visited_dirs: set = set()
        self._visited_files: set = set()

    def walk(self, root: Path) -> Iterator[Path]:
        """Yield all files under *root*, skipping ignored/foreign directories.

        Directories are pushed onto the stack in reverse ``iterdir()`` order so
        that popping reproduces the depth-first encounter order of the previous
        recursive walk for normal trees.
        """
        try:
            self._resolved_root = Path(root).resolve(strict=False)
        except OSError:
            self._resolved_root = Path(root)

        root_identity = self._identity(root)
        if root_identity is not None:
            self._visited_dirs.add(root_identity)

        stack: list[Path] = self._expand(root)
        while stack:
            item = stack.pop()
            kind = _classify(item)
            is_link = _is_link_like(item)

            try:
                rel = item.relative_to(self.scan_root).as_posix()
            except ValueError:
                rel = item.name

            if kind == "dir":
                # Apply the lexical skip/dot/ignore rules to the in-root alias
                # *before* resolving or descending, so a link cannot smuggle in
                # an otherwise-ignored path.
                if (
                    item.name.lower() in self.skip_dirs
                    or item.name.startswith(".")
                    or _is_ignored(rel, self.ignore_prefixes)
                ):
                    self.skipped_dirs += 1
                    continue
                if is_link:
                    if not self.follow_symlinks:
                        self.skipped_dirs += 1
                        logger.debug("Skipping symlinked directory %s", item)
                        continue
                    target = _safe_resolve(item)
                    if (
                        target is None
                        or not target.is_dir()
                        or not self._within_root(target)
                    ):
                        self.skipped_dirs += 1
                        logger.debug(
                            "Skipping symlinked directory (broken, type-mismatched, "
                            "or out-of-root) %s",
                            item,
                        )
                        continue
                # Visited-identity guard for *every* directory (not only links):
                # stops cycles through symlinks/junctions and dedups aliases.
                identity = self._identity(item)
                if identity is not None:
                    if identity in self._visited_dirs:
                        logger.debug("Skipping already-visited directory %s", item)
                        continue
                    self._visited_dirs.add(identity)
                for child in self._expand(item):
                    stack.append(child)

            elif kind == "file":
                if _is_ignored(rel, self.ignore_prefixes):
                    continue
                if is_link:
                    if not self.follow_symlinks:
                        logger.debug("Skipping symlinked file %s", item)
                        continue
                    target = _safe_resolve(item)
                    if (
                        target is None
                        or not target.is_file()
                        or not self._within_root(target)
                    ):
                        logger.debug(
                            "Skipping symlinked file (broken, type-mismatched, "
                            "or out-of-root) %s",
                            item,
                        )
                        continue
                # The first in-root alias of a real file owns the descriptor;
                # later aliases resolve to the same identity and are skipped.
                if self.follow_symlinks:
                    file_identity = self._identity(item)
                    if file_identity is not None:
                        if file_identity in self._visited_files:
                            continue
                        self._visited_files.add(file_identity)
                yield item

            elif is_link:
                # Broken or inaccessible link: skip without aborting the scan.
                logger.debug("Skipping broken or inaccessible symlink %s", item)

    def _expand(self, directory: Path) -> list[Path]:
        """Return *directory*'s entries reversed, so popping restores order."""
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            self.skipped_dirs += 1
            logger.warning("Skipping unreadable directory %s: %s", directory, exc)
            return []
        entries.reverse()
        return entries

    def _within_root(self, resolved: Path) -> bool:
        """True when *resolved* (an already-resolved path) is inside the root."""
        try:
            resolved.relative_to(self._resolved_root)
            return True
        except ValueError:
            return False

    def _identity(self, path: Path):
        """Return a stable identity for *path* (follows links).

        Prefers ``(st_dev, st_ino)`` when meaningful and falls back to the
        normalized resolved path on platforms/filesystems where the inode is
        unavailable (e.g. some Windows configurations report ``st_ino == 0``).
        """
        try:
            st = path.stat()
        except OSError:
            return self._resolved_path_identity(path)
        if getattr(st, "st_ino", 0):
            return ("inode", st.st_dev, st.st_ino)
        return self._resolved_path_identity(path)

    @staticmethod
    def _resolved_path_identity(path: Path):
        try:
            return ("path", str(Path(path).resolve(strict=False)))
        except OSError:
            return None


def _classify(path: Path) -> str | None:
    """Classify *path* as ``"dir"``, ``"file"`` or ``None`` (following links)."""
    try:
        if path.is_dir():
            return "dir"
        if path.is_file():
            return "file"
    except OSError:
        return None
    return None


def _is_link_like(path: Path) -> bool:
    """True for symlinks and, on Windows, junctions / reparse points.

    Centralizes link detection so version/platform checks are not scattered
    through the traversal loop.  Uses :meth:`Path.is_symlink`, then
    :meth:`Path.is_junction` where available (Python 3.12+), and finally the
    reparse-point file attribute on older Windows runtimes.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        pass

    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            pass
    else:
        try:
            attrs = getattr(path.lstat(), "st_file_attributes", 0)
        except (OSError, AttributeError):
            attrs = 0
        reparse = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attrs & reparse:
            return True
    return False


def _safe_resolve(path: Path) -> Path | None:
    """Strictly resolve *path*; return ``None`` for broken/cyclic/missing links."""
    try:
        return Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return None


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
