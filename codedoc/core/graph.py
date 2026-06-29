"""Dependency graph and import resolution helpers for codedoc."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path, PurePosixPath

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


class DependencyGraph:
    """Directed graph where edge A -> B means file A imports file B."""

    def __init__(self) -> None:
        self._edges: dict[str, set[str]] = defaultdict(set)
        self._reverse: dict[str, set[str]] = defaultdict(set)
        self._nodes: set[str] = set()
        # 0.11.1 (Workstream H): topological_order() is called several times per
        # run; emit the cycle warning at most once per graph instance (i.e. per
        # pipeline run) so a real cycle is not reported repeatedly.
        self._cycle_warning_emitted = False

    def add_file(self, rel_path: str) -> None:
        self._nodes.add(_to_posix(rel_path))

    def add_dependency(self, from_file: str, to_file: str) -> None:
        """Record that ``from_file`` imports ``to_file``."""
        from_rel = _to_posix(from_file)
        to_rel = _to_posix(to_file)
        self._nodes.add(from_rel)
        self._nodes.add(to_rel)
        self._edges[from_rel].add(to_rel)
        self._reverse[to_rel].add(from_rel)

    def dependencies_of(self, rel_path: str) -> set[str]:
        """Files that ``rel_path`` directly imports."""
        return set(self._edges.get(_to_posix(rel_path), set()))

    def dependents_of(self, rel_path: str) -> set[str]:
        """Files that import ``rel_path``."""
        return set(self._reverse.get(_to_posix(rel_path), set()))

    def all_files(self) -> set[str]:
        return set(self._nodes)

    def reachable_dependencies(self, entry_rel_path: str) -> set[str]:
        """Return the entry file plus every project file reachable from its imports."""
        entry = _to_posix(entry_rel_path)
        if entry not in self._nodes:
            return set()

        seen: set[str] = set()
        stack = [entry]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(sorted(self._edges.get(node, set()), reverse=True))
        return seen

    def affected_by_changes(self, changed_rel_paths: set[str]) -> set[str]:
        """Return changed files plus all files that depend on them, transitively."""
        seen: set[str] = set()
        queue: deque[str] = deque(_to_posix(p) for p in changed_rel_paths)
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            queue.extend(sorted(self._reverse.get(node, set())))
        return seen

    def topological_order(self) -> list[str]:
        """Return files in dependency-first order."""
        unresolved: dict[str, int] = {
            node: len(self._edges.get(node, set())) for node in self._nodes
        }
        queue: deque[str] = deque(sorted(n for n in self._nodes if unresolved[n] == 0))
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for dependent in sorted(self._reverse.get(node, set())):
                unresolved[dependent] -= 1
                if unresolved[dependent] == 0:
                    queue.append(dependent)

        remaining = sorted(n for n in self._nodes if n not in order)
        if remaining and not self._cycle_warning_emitted:
            # Count at WARNING once per run; full path list at DEBUG once per run.
            self._cycle_warning_emitted = True
            logger.warning(
                "Dependency cycle detected involving %d file(s). Processing them anyway. "
                "Run with --verbose to list the files involved.",
                len(remaining),
            )
            logger.debug("Dependency cycle members (%d): %s", len(remaining), remaining)
        return order + remaining

    def summary(self) -> str:
        lines = [
            f"DependencyGraph: {len(self._nodes)} nodes, "
            f"{sum(len(v) for v in self._edges.values())} edges"
        ]
        for node in sorted(self._nodes):
            deps = self._edges.get(node, set())
            if deps:
                lines.append(f"  {node} -> {sorted(deps)}")
        return "\n".join(lines)


def resolve_import(
    import_str: str,
    current_file: str,
    all_files: set[str],
    root: Path,
) -> str | None:
    """Resolve a parser import string to a known project-relative path.

    Resolution is purely lexical: candidates are matched, exact-case, against
    the project-relative paths in *all_files*.  It never probes the filesystem,
    installed packages, or the network, so the same repository resolves to the
    same dependency graph on case-sensitive and case-insensitive hosts alike.

    ``root`` is retained for backward compatibility only — it is no longer read.
    A case-mismatched import is left unresolved.
    """
    if not import_str:
        return None

    known = {_to_posix(path): path for path in all_files}
    candidates = _candidate_import_paths(import_str.strip(), current_file)
    for candidate in candidates:
        match = _match_candidate(candidate, known)
        if match:
            return match
    return None


# Must stay in sync with DEFAULTS["extension_language_map"] in loader.py.
_KNOWN_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".dart", ".java", ".cs", ".html", ".htm",
    ".kt", ".swift", ".go", ".rb", ".rs", ".cpp", ".c", ".h", ".hpp",
}


def _candidate_import_paths(import_str: str, current_file: str) -> list[str]:
    current = PurePosixPath(_to_posix(current_file))
    current_dir = current.parent
    candidates: list[str] = []
    suffix = PurePosixPath(import_str).suffix

    if import_str.startswith("package:"):
        package_path = import_str.split("/", 1)
        if len(package_path) == 2:
            candidates.append(f"lib/{package_path[1]}")
        return candidates

    if import_str.startswith(("./", "../")):
        candidates.append(_normalize_posix(str(current_dir / import_str)))
    elif import_str.startswith("."):
        candidates.append(_resolve_python_relative(import_str, current_dir))
    elif suffix and suffix in _KNOWN_EXTENSIONS:
        candidates.append(_normalize_posix(import_str))
        candidates.append(_normalize_posix(str(current_dir / import_str)))
    elif "/" in import_str or "\\" in import_str:
        candidates.append(_normalize_posix(import_str))
        candidates.append(_normalize_posix(str(current_dir / import_str)))
    elif "." in import_str:
        dotted = import_str.replace(".", "/")
        candidates.append(dotted)
        candidates.append(str(current_dir / dotted))
    else:
        candidates.append(import_str)
        candidates.append(str(current_dir / import_str))

    return [c for c in dict.fromkeys(candidates) if c and c != "."]


def _resolve_python_relative(import_str: str, current_dir: PurePosixPath) -> str:
    level = len(import_str) - len(import_str.lstrip("."))
    module = import_str[level:].replace(".", "/")
    base = current_dir
    for _ in range(max(level - 1, 0)):
        base = base.parent
    return _normalize_posix(str(base / module)) if module else _normalize_posix(str(base))


def _match_candidate(
    candidate: str,
    known: dict[str, str],
) -> str | None:
    """Match *candidate* exactly (no case folding) against the known paths."""
    candidate = _normalize_posix(candidate)
    for variant in _candidate_variants(candidate):
        if variant in known:
            return known[variant]
    return None


def _candidate_variants(candidate: str) -> list[str]:
    extensions = sorted(_KNOWN_EXTENSIONS)
    variants = [candidate]

    suffix = PurePosixPath(candidate).suffix
    if not suffix:
        variants.extend(candidate + ext for ext in extensions)
        variants.extend(f"{candidate}/index{ext}" for ext in extensions)
        variants.append(f"{candidate}/__init__.py")

    return [_normalize_posix(v) for v in dict.fromkeys(variants)]


def _normalize_posix(path: str) -> str:
    parts: list[str] = []
    for part in _to_posix(path).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _to_posix(path: str) -> str:
    return str(path).replace("\\", "/")
