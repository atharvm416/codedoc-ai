"""Centralized logging for codedoc.

Namespace scoping (section 3A)
-------------------------------
Every CodeDoc logger is created by :func:`get_logger` from a module inside the
``codedoc`` package, so raising verbosity means raising ``codedoc`` alone.

- :func:`set_level` sets the level of ``logging.getLogger("codedoc")`` only.
  It never calls ``logging.getLogger().setLevel(...)``.  Records still reach
  the configured root handler through normal propagation, because the handler
  installed by ``logging.basicConfig(...)`` has no level of its own -- verbose
  CodeDoc output is therefore unchanged in content.
- Standalone (no pre-existing root handlers, the CLI case): ``_configure()``
  installs CodeDoc's own handler and leaves root at ``INFO``.  A dependency
  logger nobody enumerated then inherits ``INFO`` and cannot emit DEBUG even
  if a future dependency introduces one -- this design fails closed.
- Hosted (root already configured by an embedding application): ``basicConfig``
  is a no-op, so CodeDoc does not pretend it established root ``INFO``.  The
  host's root level, handler identities/levels, formatters, and filters are
  preserved exactly; only the ``codedoc`` logger level and the floor list below
  are touched.  If the host independently set root to DEBUG, an unlisted
  third-party namespace may emit DEBUG -- that is the host's decision, not a
  guarantee this module breaks.

Third-party floor (defense in depth)
-------------------------------------
An exact, closed list of provider/transport namespaces is held at ``WARNING``
or stricter during both normal and verbose execution, in both the standalone
and hosted cases -- in the hosted case this floor list is the *only* barrier.
A logger at ``NOTSET`` resolves to its nearest ancestor with a level set, so
flooring the parent namespace also covers every descendant (e.g. flooring
``httpcore`` covers ``httpcore.http11``; flooring ``openai`` covers
``openai.lib.parsing``).  ``google.genai`` and ``google_genai`` are both
required: ``google-genai`` uses the underscore form for
``google_genai._api_client`` and friends, which is not a child of ``google``.

Windows redirected-logging Unicode safety (section 3A, the P2 finding)
------------------------------------------------------------------------
A redirected Windows PowerShell run (``2>&1 | Tee-Object``) can leave a
process's own stdout/stderr bound to a legacy single-byte console code page
(e.g. CP1252) with the default strict error handler. Any bounded CodeDoc
diagnostic containing a character outside that code page then crashes
logging itself with a repeated ``UnicodeEncodeError`` -- ``--- Logging error
---`` -- rather than a leak of raw content. ``_configure()`` reconfigures
stdout/stderr to replace only the *error handler* (``backslashreplace``),
never the encoding, so a representable character still renders exactly as
before and an unrepresentable one renders as a deterministic escaped
sequence instead of raising. The Rich console is additionally constructed
with ``legacy_windows=False`` so it never routes through the legacy Win32
console API, which has the same failure mode independently of Python's own
stream encoding.
"""

from __future__ import annotations

import logging
import sys

try:
    from rich.console import Console
    from rich.logging import RichHandler
except ImportError:
    Console = None
    RichHandler = None

_configured = False

_CODEDOC_LOGGER_NAME = "codedoc"

# Closed, exact list of third-party provider/transport namespaces floored at
# WARNING-or-stricter. Do not widen this to "and other provider/transport
# loggers" or any other open-ended phrase -- add a namespace here explicitly.
_NOISY_LOGGERS = (
    "openai",
    "anthropic",
    "google",
    "google.genai",
    "google_genai",
    "google.auth",
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
)


def _apply_third_party_floor() -> None:
    """Reassert the exact reviewed floor for every listed namespace.

    A descendant with an explicitly lowered level does not inherit its
    parent's level, and a handler attached directly to a provider/transport
    logger can otherwise emit below the reviewed floor.  Normalize already
    configured descendants and their handlers as well as the named parents.
    Descendants left at ``NOTSET`` keep inheriting the nearest floored parent.
    """
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    configured = logging.Logger.manager.loggerDict
    for logger_name, candidate in tuple(configured.items()):
        if not isinstance(candidate, logging.Logger):
            continue
        if not any(
            logger_name == floor or logger_name.startswith(f"{floor}.")
            for floor in _NOISY_LOGGERS
        ):
            continue
        if candidate.level != logging.NOTSET and candidate.level < logging.WARNING:
            candidate.setLevel(logging.WARNING)
        for handler in candidate.handlers:
            if handler.level < logging.WARNING:
                handler.setLevel(logging.WARNING)


def _make_stream_encoding_safe(stream: object) -> None:
    """Best-effort: make *stream* tolerate characters the active console code
    page cannot represent, instead of raising ``UnicodeEncodeError``.

    Reproduces the exact P2 condition: a redirected Windows PowerShell run
    (``2>&1 | Tee-Object``) can leave a process's own stdout/stderr bound to a
    legacy single-byte code page (e.g. CP1252) with the default strict error
    handler, so any bounded CodeDoc diagnostic containing a character outside
    that code page crashes logging itself. Switching only the error handler
    (never the encoding) means a supported character still renders exactly as
    before; an unsupported one renders as a deterministic backslash-escaped
    sequence instead of raising. Silently does nothing for a stream that
    cannot be reconfigured (e.g. a test-capture stream) -- functionality is
    unaffected either way, since reconfiguration is a defense-in-depth safety
    net, not a behavior change on platforms/streams where it is unavailable.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass


def _configure() -> None:
    global _configured
    if _configured:
        return

    _make_stream_encoding_safe(sys.stdout)
    _make_stream_encoding_safe(sys.stderr)

    handlers: list[logging.Handler]
    if RichHandler is not None and Console is not None:
        handlers = [
            RichHandler(
                console=Console(legacy_windows=False),
                rich_tracebacks=True,
                show_path=False,
            )
        ]
        log_format = "%(message)s"
    else:
        handlers = [logging.StreamHandler()]
        log_format = "%(levelname)s:%(name)s:%(message)s"

    # A no-op when the root logger already has handlers (the hosted case):
    # CodeDoc then does not own root's level and must not pretend it does.
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="[%X]",
        handlers=handlers,
    )

    _apply_third_party_floor()

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Change CodeDoc's own log level at runtime.

    Scoped to ``logging.getLogger("codedoc")`` only -- this never raises the
    root logger, so an unlisted third-party namespace (NOTSET) keeps resolving
    to root's own level rather than inheriting CodeDoc's verbosity. The
    third-party floor list is reasserted unconditionally, in both directions,
    so it cannot be loosened by a normal-verbosity call after a verbose one.
    """
    _configure()
    normalized = str(level).upper()
    logging.getLogger(_CODEDOC_LOGGER_NAME).setLevel(normalized)
    _apply_third_party_floor()
