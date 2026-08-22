"""Scan a project and detect file languages without using an LLM.

Callers provide skip rules, extension mappings, and entry candidates. The
``supported_extensions`` keyword remains compatible with direct callers.
"""

from __future__ import annotations

import os
import stat as stat_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Bounded per-scan logging (section 5.6): the first this-many unreadable
# files each get their own warning line; beyond that, one aggregate line.
MAX_UNREADABLE_FILE_WARNINGS = 20


def exclude_path_key(path: Path | str) -> str:
    """Shared normalization key for exact generated-target exclusion.

    Resolved non-strictly (a generated target need not exist yet) and
    OS-case-folded so both the caller building an ``exclude_paths`` set and
    the scanner checking each candidate file use the identical key. Equality
    only -- never used for basename or prefix matching, and deliberately not
    the lower-casing scheme ``_normalise_ignore_paths`` uses below (that
    scheme is for portable relative-path prefixes; this one is for exact
    resolved absolute paths).
    """
    return os.path.normcase(str(Path(path).resolve(strict=False)))


@dataclass
class ScanDiagnostics:
    """Optional keyword-only out-parameter for :func:`scan_files`.

    Mutated in place so ``scan_files``'s return type (``list[dict]``) never
    changes shape. Surfaced only in ``run_pipeline`` stats and the CLI
    summary -- never persisted to codedoc.json/codedoc.md schemas.

    One instance is shared across every ``scan_files`` call (and any other
    caller recording a read failure, e.g. planning) within a single
    ``run_pipeline`` invocation, including a rescan triggered by a detected
    stale source revision. ``record_large``/``record_unreadable`` dedup by
    ``exclude_path_key`` against this instance's own lifetime, not any one
    call's, so the same physical file is counted at most once per run no
    matter how many times it is rescanned.
    """

    files_skipped_large: int = 0
    files_skipped_unreadable: int = 0
    _large_seen: set = field(default_factory=set, repr=False, compare=False)
    _unreadable_seen: set = field(default_factory=set, repr=False, compare=False)

    def record_large(self, key: str) -> bool:
        """Count *key* as skipped-large; return True the first time this
        run sees it, False on a repeat (e.g. a rescan)."""
        if key in self._large_seen:
            return False
        self._large_seen.add(key)
        self.files_skipped_large += 1
        return True

    def record_unreadable(self, key: str) -> bool:
        """Count *key* as skipped-unreadable; return True the first time
        this run sees it, False on a repeat (e.g. a rescan)."""
        if key in self._unreadable_seen:
            return False
        self._unreadable_seen.add(key)
        self.files_skipped_unreadable += 1
        return True

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
    # Safety control.  Keyword-only so existing positional callers stay
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
    # Exact generated-target protection (section 5.6).  Keys built with
    # exclude_path_key() -- equality only, never basename or prefix
    # matching, so a source file that merely shares the output directory's
    # name is never excluded (co-located source/output stays supported).
    exclude_paths: "frozenset[str] | set[str] | None" = None,
    # Optional out-parameter, mutated in place; return type stays list[dict].
    diagnostics: "ScanDiagnostics | None" = None,
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
        ``config["skip_dirs"]`` by the pipeline; generated targets are
        protected separately through ``exclude_paths``. Directories whose
        names begin with ``.`` are always skipped regardless of this list.
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
    exclude_paths:
        Exact resolved generated-target keys (built with
        :func:`exclude_path_key`) to protect from ever being treated as
        source -- e.g. the active and opposite-format output files and the
        crash-recovery file.  Matched by equality only, so co-located
        source/output directories remain supported: a source file merely
        sharing the output directory's name is never excluded.
    diagnostics:
        Optional :class:`ScanDiagnostics` out-parameter, mutated in place
        with ``files_skipped_large`` and ``files_skipped_unreadable`` counts.

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
    exclude_keys = frozenset(exclude_paths or ())
    results: list[dict] = []
    # Dedup/count against the caller's own ScanDiagnostics when supplied, so
    # a rescan sharing the same instance (section 12.1 C5) never double-counts
    # a file this scan already recorded; a throwaway instance otherwise, so
    # the counting logic below is uniform either way.
    _diagnostics = diagnostics if diagnostics is not None else ScanDiagnostics()
    skipped_large_this_scan = 0
    unreadable_warned = 0
    unreadable_total_this_scan = 0

    def _record_unreadable(path: Path) -> None:
        nonlocal unreadable_warned, unreadable_total_this_scan
        key = exclude_path_key(path)
        # The shared diagnostics object owns run-wide deduplication. Its
        # boolean result must gate warnings as well as counters, otherwise a
        # rescan emits the same path again and can exceed the per-run warning
        # bound despite keeping the numeric counter stable.
        if not _diagnostics.record_unreadable(key):
            return
        unreadable_total_this_scan += 1
        if unreadable_warned < MAX_UNREADABLE_FILE_WARNINGS:
            logger.warning("Skipping unreadable file: %s", path)
            unreadable_warned += 1

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

        if exclude_keys and exclude_path_key(file_path) in exclude_keys:
            continue

        # Skip files that are too large
        try:
            size_kb = file_path.stat().st_size / 1024
        except OSError:
            _record_unreadable(file_path)
            continue
        if size_kb > max_file_size_kb:
            logger.warning("Skipping large file (%dkb): %s", int(size_kb), file_path)
            skipped_large_this_scan += 1
            _diagnostics.record_large(exclude_path_key(file_path))
            continue

        language = ext_map.get(ext, "generic")
        rel = file_path.relative_to(root).as_posix()

        results.append({
            "path": file_path,
            "rel_path": rel,
            "language": language,
            "extension": ext,
        })

    # Walker-level unreadable items (a genuine OSError classifying the path
    # itself, distinguished from "exists but is neither a file nor a
    # directory" -- see _classify) only count if their extension is one of
    # the supported ones; an unreadable file the scan would have ignored
    # anyway is not worth warning about.
    for path in walker.unreadable_files:
        if path.suffix.lower() in ext_map:
            _record_unreadable(path)

    if unreadable_total_this_scan > unreadable_warned:
        logger.warning(
            "%d more unreadable file(s) not shown",
            unreadable_total_this_scan - unreadable_warned,
        )

    skipped_dirs = walker.skipped_dirs
    logger.info(
        "Scanner found %d supported file(s) in %s (skipped %d directorie(s), "
        "%d large file(s), %d unreadable file(s))",
        len(results),
        root,
        skipped_dirs,
        skipped_large_this_scan,
        unreadable_total_this_scan,
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
        self.unreadable_files: list[Path] = []
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

            elif kind == "unreadable":
                # A genuine OSError (e.g. permission denied) probing this
                # path's type -- distinct from _classify's None case, which
                # means the path exists but is neither a file nor a
                # directory (a broken symlink target, socket, or device).
                self.unreadable_files.append(item)

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
    """Classify *path* as ``"dir"``, ``"file"``, ``"unreadable"``, or
    ``None`` (following links).

    ``Path.is_dir()``/``Path.is_file()`` only swallow the "doesn't
    exist"-shaped errno set (ENOENT/ENOTDIR) internally and re-raise
    everything else (notably EACCES, permission denied) -- so an ``OSError``
    reaching here is a genuine read failure, distinguished from the ``None``
    case, which means the path exists but is neither a file nor a directory
    (a broken symlink target, socket, or device) with no error at all.
    """
    try:
        if path.is_dir():
            return "dir"
    except OSError:
        return "unreadable"
    try:
        if path.is_file():
            return "file"
    except OSError:
        return "unreadable"
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
        # Section 5.6 / 12.1 C5: ``Path.exists()`` swallows only the
        # "doesn't exist"-shaped errno set (ENOENT/ENOTDIR/EBADF/ELOOP) and
        # re-raises everything else, notably EACCES.  An explicitly requested
        # entry whose own metadata cannot be read must fail as an actionable
        # ConfigError here -- before provider construction, recovery
        # initialization, or any output mutation -- never as a raw
        # PermissionError/OSError escaping the scan.  This is the
        # stat-inspection counterpart to the post-stat read failure that
        # ``codedoc.core.planning`` already reports.
        try:
            found = candidate.exists()
        except OSError as exc:
            raise ConfigError(
                f"Entry file '{hint}' could not be read: {exc}. Check the "
                "path and its file permissions."
            ) from exc
        if found:
            return candidate
        logger.warning("Specified entry file '%s' not found in %s", hint, root)
        return None

    for name in auto_entries:
        candidate = root / name
        # An auto-detection candidate is a guess, not a user request: an
        # unreadable one is skipped like a missing one so a single
        # permission-restricted file never aborts a run the user never
        # asked to centre on it.
        try:
            found = candidate.exists()
        except OSError:
            logger.debug("Skipping unreadable auto-entry candidate %s", candidate)
            continue
        if found:
            logger.info("Auto-detected entry file: %s", candidate)
            return candidate

    logger.warning("No entry file detected. Will process all files.")
    return None
