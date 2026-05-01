"""
React / TypeScript / JavaScript import parser.

Uses regex (not a full AST) for speed and zero dependencies.
Handles ES6 import/export, dynamic import(), and require().
"""

from __future__ import annotations

import re
from pathlib import Path

from codedoc.utils.errors import ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

_PATTERNS: list[re.Pattern] = [
    # import ... from '...'
    re.compile(r'''(?:^|\n)\s*import\s+[^'"]*from\s+['"]([^'"]+)['"]'''),
    # import '...'  (side-effect import)
    re.compile(r'''(?:^|\n)\s*import\s+['"]([^'"]+)['"]'''),
    # export ... from '...'
    re.compile(r'''(?:^|\n)\s*export\s+[^'"]*from\s+['"]([^'"]+)['"]'''),
    # const x = require('...')
    re.compile(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)'''),
    # dynamic import('...')
    re.compile(r'''import\s*\(\s*['"]([^'"]+)['"]\s*\)'''),
]

# Skip node_modules / packages (no leading ./ or ../)
_IS_RELATIVE = re.compile(r'^\.{1,2}[/\\]|^\.{1,2}$')
_IS_PATH_ALIAS = re.compile(r'^@|^~')  # @components/... or ~/...


def parse(file_path: Path, language: str = "tsx") -> list[str]:
    """
    Extract import paths from a TS/TSX/JS/JSX file.
    Returns local/relative imports and path aliases; skips bare npm packages.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ParseError(str(file_path), f"Cannot read file: {exc}") from exc

    imports: list[str] = []
    seen: set[str] = set()

    for pattern in _PATTERNS:
        for match in pattern.finditer(content):
            imp = match.group(1).strip()
            if not imp or imp in seen:
                continue
            seen.add(imp)
            # Include relative paths and path aliases; skip npm packages
            if _IS_RELATIVE.match(imp) or _IS_PATH_ALIAS.match(imp):
                imports.append(imp)

    logger.debug("ReactParser found %d imports in %s", len(imports), file_path.name)
    return imports