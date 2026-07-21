"""
Python import parser — uses the built-in `ast` module.

Deterministic, no LLM. Extracts both `import X` and `from X import Y` forms.
"""

from __future__ import annotations

import ast
from pathlib import Path

from codedoc.utils.errors import ParseError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def parse_text(source: str, language: str = "python", *, file_path: str = "<source>") -> list[str]:
    """
    Extract import strings from already-decoded Python source using AST.
    Returns module names as strings, e.g. ["os", "pathlib", ".utils"].

    ``file_path`` is used only for error text and logging (the AST filename),
    so this and :func:`parse` produce literal-identical imports for the same
    decoded text.
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        raise ParseError(file_path, f"Python SyntaxError: {exc}") from exc

    imports: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name not in seen:
                    seen.add(name)
                    imports.append(name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level  # dots for relative imports
            prefix = "." * level
            full = f"{prefix}{module}" if module else prefix
            if full not in seen:
                seen.add(full)
                imports.append(full)

    logger.debug("PythonParser found %d imports in %s", len(imports), file_path)
    return imports


def parse(file_path: Path, language: str = "python") -> list[str]:
    """
    Extract import strings from a Python file using AST.
    Returns module names as strings, e.g. ["os", "pathlib", ".utils"].
    """
    try:
        source = file_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ParseError(str(file_path), f"Cannot read file: {exc}") from exc

    return parse_text(source, language, file_path=str(file_path))
