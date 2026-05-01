"""Runtime checks for codedoc.

The library should never install packages into a user's environment at runtime.
Installation belongs to pip/uv/poetry and explicit user commands.
"""

from __future__ import annotations

from pathlib import Path

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_codedoc_installed(project_root: str | Path) -> bool:
    """Check if codedoc is importable.

    ``project_root`` is accepted for backwards compatibility with older callers.
    """
    _ = project_root
    try:
        import codedoc  # noqa: F401
        logger.debug("codedoc package is already available.")
        return True
    except ImportError as exc:
        logger.error("codedoc is not importable in the current Python environment: %s", exc)
        return False
