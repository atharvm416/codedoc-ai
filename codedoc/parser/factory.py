"""
Parser factory.

Given a file descriptor (from scanner), picks the right parser module
and returns the extracted import list. Fully deterministic — no LLM.
"""

from __future__ import annotations

from pathlib import Path

from codedoc.core.db import read_source_snapshot
from codedoc.utils.errors import ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# language tag → parser module function
_PARSER_MAP: dict[str, str] = {
    "python":     "python",
    "typescript": "react",
    "tsx":        "react",
    "javascript": "react",
    "jsx":        "react",
    "dart":       "generic",
    "java":       "generic",
    "kotlin":     "generic",
    "csharp":     "generic",
    "swift":      "generic",
    "go":         "generic",
    "ruby":       "generic",
    "rust":       "generic",
    "c":          "generic",
    "cpp":        "generic",
    "html":       "generic",
}


def parse_file(descriptor: dict) -> list[str]:
    """
    Parse a file and return its list of import strings.

    Args:
        descriptor: file descriptor from scanner
            {path: Path, rel_path: str, language: str, extension: str}

    Returns:
        list of import strings (raw, not resolved to paths)
    """
    file_path: Path = descriptor["path"]
    try:
        _, source_text = read_source_snapshot(file_path)
    except OSError as exc:
        raise ParseError(str(file_path), f"Cannot read file: {exc}") from exc
    return parse_source(descriptor, source_text)


def parse_source(descriptor: dict, source_text: str) -> list[str]:
    """
    Parse already-decoded source text and return its list of import strings.

    Snapshot-aware counterpart to :func:`parse_file`: *source_text* is the
    caller's already-read, already-decoded snapshot (see
    :func:`codedoc.core.db.read_source_snapshot`), so no file is reopened here.
    Returns literal-identical imports to :func:`parse_file` for an unchanged
    file, because both dispatch to the same per-language ``parse_text()``.

    Args:
        descriptor: file descriptor from scanner
            {path: Path, rel_path: str, language: str, extension: str}
        source_text: the decoded source snapshot to parse

    Returns:
        list of import strings (raw, not resolved to paths)
    """
    file_path: Path = descriptor["path"]
    language: str = descriptor.get("language", "generic")

    parser_type = _PARSER_MAP.get(language, "generic")

    try:
        if parser_type == "python":
            from codedoc.parser import python_parser
            return python_parser.parse_text(source_text, language, file_path=str(file_path))

        if parser_type == "react":
            from codedoc.parser import react_parser
            return react_parser.parse_text(source_text, language, file_path=str(file_path))

        # generic fallback
        from codedoc.parser import generic_parser
        return generic_parser.parse_text(source_text, language, file_path=str(file_path))

    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(str(file_path), f"Unexpected error in parser: {exc}") from exc
