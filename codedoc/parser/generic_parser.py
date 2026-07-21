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
    # Go imports are handled by the dedicated _parse_go() helper (see parse()),
    # not by this generic pattern table, because a naive "quoted string" regex
    # captures every string literal in the file (e.g. fmt.Println("hi")) as a
    # false import.  The entry is kept here only as documentation; parse()
    # intercepts "go" before this table is consulted.
    "go": [],
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
        # Real code dependencies only: <script src> and JS module imports.
        # <link href> (CSS, fonts, icons, preloads) is NOT a code import and was
        # producing false dependency edges, so it is intentionally omitted.
        (re.compile(r'<script[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE), 1),
        (re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE), 1),
    ],
}

# Fallback for any unknown language: look for common import-like lines
_FALLBACK_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r'''(?:import|require|include|use|from)\s+['"]([^'"]+)['"]''', re.MULTILINE), 1),
    (re.compile(r'^\s*import\s+([\w./\-]+)', re.MULTILINE), 1),
]


# Go import paths are string literals — interpreted ("...") or raw (`...`).
# Single-line form allows an optional alias: `import alias "path"` / `import _ "x"`.
_GO_SINGLE_IMPORT_RE = re.compile(
    r'^\s*import\s+(?:[\w.]+\s+)?(?:"([^"]+)"|`([^`]+)`)', re.MULTILINE
)
_GO_IMPORT_BLOCK_RE = re.compile(r'import\s*\(\s*(.*?)\)', re.DOTALL)
_GO_QUOTED_PATH_RE = re.compile(r'"([^"]+)"|`([^`]+)`')
_GO_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_GO_LINE_COMMENT_RE = re.compile(r'//[^\n]*')

_GO_SIMPLE_ESCAPE_BYTES = {
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
    "\\": 0x5C,
    "'": 0x27,
    '"': 0x22,
}


def _decode_go_interpreted(literal: str) -> str:
    """Decode the contents of a Go interpreted string literal.

    Go's ``\\xNN`` and three-digit octal escapes contribute individual bytes,
    while ``\\uNNNN`` and ``\\UNNNNNNNN`` contribute UTF-8 encoded Unicode code
    points. Building the complete byte sequence before decoding preserves that
    distinction, so ``caf\\xc3\\xa9`` and ``caf\\u00e9`` both become ``café``.

    Malformed or non-UTF-8 input is returned unchanged. Such text is not a valid
    project import path, but it must not make the best-effort parser fail.
    """
    if "\\" not in literal:
        return literal

    decoded = bytearray()
    index = 0

    try:
        while index < len(literal):
            char = literal[index]
            if char != "\\":
                decoded.extend(char.encode("utf-8"))
                index += 1
                continue

            if index + 1 >= len(literal):
                return literal

            escape = literal[index + 1]
            if escape in _GO_SIMPLE_ESCAPE_BYTES:
                decoded.append(_GO_SIMPLE_ESCAPE_BYTES[escape])
                index += 2
                continue

            if escape == "x":
                digits = literal[index + 2:index + 4]
                if len(digits) != 2 or not all(c in "0123456789abcdefABCDEF" for c in digits):
                    return literal
                decoded.append(int(digits, 16))
                index += 4
                continue

            if escape in "01234567":
                digits = literal[index + 1:index + 4]
                if len(digits) != 3 or not all(c in "01234567" for c in digits):
                    return literal
                value = int(digits, 8)
                if value > 0xFF:
                    return literal
                decoded.append(value)
                index += 4
                continue

            if escape in ("u", "U"):
                width = 4 if escape == "u" else 8
                digits = literal[index + 2:index + 2 + width]
                if (
                    len(digits) != width
                    or not all(c in "0123456789abcdefABCDEF" for c in digits)
                ):
                    return literal
                code_point = int(digits, 16)
                if code_point > 0x10FFFF or 0xD800 <= code_point <= 0xDFFF:
                    return literal
                decoded.extend(chr(code_point).encode("utf-8"))
                index += 2 + width
                continue

            return literal

        return decoded.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError, ValueError, OverflowError):
        return literal


def _strip_go_comments(text: str) -> str:
    """Remove Go block and line comments.

    Safe for import extraction: import paths never contain ``//`` (they use
    single slashes, e.g. ``github.com/x/y``), so stripping ``//`` to end-of-line
    cannot truncate a real import path. This prevents commented-out imports and
    quoted text inside comments from being read as imports.
    """
    text = _GO_BLOCK_COMMENT_RE.sub("", text)
    text = _GO_LINE_COMMENT_RE.sub("", text)
    return text


def _parse_go(content: str) -> list[str]:
    """Extract Go imports without matching arbitrary string literals or comments.

    Only string-literal paths in an ``import "..."`` statement or inside an
    ``import ( ... )`` block are returned. ``fmt.Println("hi")`` is not an
    import, commented-out imports are ignored, and raw-string (backtick) paths
    are supported.
    """
    cleaned = _strip_go_comments(content)
    imports: list[str] = []
    seen: set[str] = set()

    def _add(match: "re.Match") -> None:
        if match.group(1) is not None:          # interpreted "..." → decode escapes
            imp = _decode_go_interpreted(match.group(1)).strip()
        else:                                   # raw `...` → literal, no decoding
            imp = (match.group(2) or "").strip()
        if imp and imp not in seen:
            seen.add(imp)
            imports.append(imp)

    for block in _GO_IMPORT_BLOCK_RE.finditer(cleaned):
        for match in _GO_QUOTED_PATH_RE.finditer(block.group(1)):
            _add(match)

    for match in _GO_SINGLE_IMPORT_RE.finditer(cleaned):
        _add(match)

    return imports


def parse_text(source: str, language: str, *, file_path: str = "<source>") -> list[str]:
    """
    Extract import strings from already-decoded source using regex.
    Returns a list of raw import strings (not resolved to paths).

    ``file_path`` is used only for logging, so this and :func:`parse` produce
    literal-identical imports for the same decoded text.
    """
    if language == "go":
        imports = _parse_go(source)
        logger.debug("GenericParser [go] found %d imports in %s", len(imports), file_path)
        return imports

    patterns = _PATTERNS.get(language, _FALLBACK_PATTERNS)
    imports: list[str] = []
    seen: set[str] = set()

    for pattern, group in patterns:
        for match in pattern.finditer(source):
            imp = match.group(group).strip()
            if imp and imp not in seen:
                seen.add(imp)
                imports.append(imp)

    logger.debug("GenericParser [%s] found %d imports in %s", language, len(imports), file_path)
    return imports


def parse(file_path: Path, language: str) -> list[str]:
    """
    Extract import strings from a file using regex.
    Returns a list of raw import strings (not resolved to paths).
    """
    try:
        content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ParseError(str(file_path), f"Cannot read file: {exc}") from exc

    return parse_text(content, language, file_path=str(file_path))
