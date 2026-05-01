"""Centralized logging for codedoc."""

from __future__ import annotations

import logging

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError:
    Console = None
    RichHandler = None

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler]
    if RichHandler is not None and Console is not None:
        handlers = [RichHandler(console=Console(), rich_tracebacks=True, show_path=False)]
        log_format = "%(message)s"
    else:
        handlers = [logging.StreamHandler()]
        log_format = "%(levelname)s:%(name)s:%(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="[%X]",
        handlers=handlers,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Change log level at runtime."""
    _configure()
    logging.getLogger().setLevel(str(level).upper())
