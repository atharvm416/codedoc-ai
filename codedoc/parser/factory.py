"""
Parser factory.

Given a file descriptor (from scanner), picks the right parser module
and returns the extracted import list. Fully deterministic — no LLM.
"""

from __future__ import annotations

from pathlib import Path

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
    language: str = descriptor.get("language", "generic")

    parser_type = _PARSER_MAP.get(language, "generic")

    try:
        if parser_type == "python":
            from codedoc.parser import python_parser
            return python_parser.parse(file_path, language)

        if parser_type == "react":
            from codedoc.parser import react_parser
            return react_parser.parse(file_path, language)

        # generic fallback
        from codedoc.parser import generic_parser
        return generic_parser.parse(file_path, language)

    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(str(file_path), f"Unexpected error in parser: {exc}") from exc