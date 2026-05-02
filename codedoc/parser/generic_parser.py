"""
Generic import parser — regex-based fallback.

Handles languages not covered by a dedicated parser:
Java, Dart/Flutter, C#/.NET, Kotlin, Swift, Go, Ruby, Rust, C/C++.
No LLM involved. Pure deterministic regex extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

from codedoc.utils.errors import ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Language → list of (regex_pattern, group_index_for_import_path)
_PATTERNS: dict[str, list[tuple[re.Pattern, int]]] = {
    "java": [
        (re.compile(r'^\s*import\s+([\w.]+)\s*;', re.MULTILINE), 1),
    ],
    "kotlin": [
        (re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE), 1),
    ],
    "dart": [
        (re.compile(r'''^\s*import\s+['"]([^'"]+)['"]''', re.MULTILINE), 1),
        (re.compile(r'''^\s*part\s+['"]([^'"]+)['"]''', re.MULTILINE), 1),
    ],
    "csharp": [
        (re.compile(r'^\s*using\s+([\w.]+)\s*;', re.MULTILINE), 1),
    ],
    "swift": [
        (re.compile(r'^\s*import\s+(\w+)', re.MULTILINE), 1),
    ],
    "go": [
        (re.compile(r'"([\w./\-]+)"', re.MULTILINE), 1),
    ],
    "ruby": [
        (re.compile(r'''^\s*require(?:_relative)?\s+['"]([^'"]+)['"]''', re.MULTILINE), 1),
    ],
    "rust": [
        (re.compile(r'^\s*(?:use|mod|extern crate)\s+([\w:]+)', re.MULTILINE), 1),
    ],
    "c": [
        (re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.MULTILINE), 1),
    ],
    "cpp": [
        (re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]', re.MULTILINE), 1),
    ],
    "html": [
        (re.compile(r'<script[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE), 1),
        (re.compile(r'<link[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE), 1),
        (re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE), 1),
    ],
}

# Fallback for any unknown language: look for common import-like lines
_FALLBACK_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r'''(?:import|require|include|use|from)\s+['"]([^'"]+)['"]''', re.MULTILINE), 1),
    (re.compile(r'^\s*import\s+([\w./\-]+)', re.MULTILINE), 1),
]


def parse(file_path: Path, language: str) -> list[str]:
    """
    Extract import strings from a file using regex.
    Returns a list of raw import strings (not resolved to paths).
    """
    try:
        content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ParseError(str(file_path), f"Cannot read file: {exc}") from exc

    patterns = _PATTERNS.get(language, _FALLBACK_PATTERNS)
    imports: list[str] = []
    seen: set[str] = set()

    for pattern, group in patterns:
        for match in pattern.finditer(content):
            imp = match.group(group).strip()
            if imp and imp not in seen:
                seen.add(imp)
                imports.append(imp)

    logger.debug("GenericParser [%s] found %d imports in %s", language, len(imports), file_path.name)
    return imports
