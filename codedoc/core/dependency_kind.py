"""Pure, deterministic non-project dependency classification (0.9.3).

This module is the single source of truth for deciding whether a
*non-project* import names a third-party package (``external``) or a
language SDK / standard-library module (``sdk``), and for canonicalizing the
import string to a stable package identifier.

Design constraints:

- pure, deterministic, and filesystem-free — never probes importability;
- never returns ``internal`` — internal dependencies come only from resolved
  project graph edges;
- committed static fallback sets, not environment-dependent probes.

Language tags are normalized to lowercase.  Behaviour groups:

- ``python``                                  -> Python behaviour
- ``javascript`` / ``jsx`` / ``typescript`` / ``tsx`` -> Node behaviour
- ``dart``                                    -> Dart behaviour
- anything else                               -> external
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

KIND_EXTERNAL = "external"
KIND_SDK = "sdk"

_NODE_LANGUAGES = frozenset({"javascript", "jsx", "typescript", "tsx"})


@dataclass(frozen=True)
class DependencyName:
    """Result of classifying a single non-project dependency string."""

    raw: str
    canonical: str
    kind: str  # "external" or "sdk"


# ---------------------------------------------------------------------------
# Committed fallback sets (static package data — never environment probes)
# ---------------------------------------------------------------------------

# Python 3.9 ``sys.stdlib_module_names``.  Used as the fallback when the
# running interpreter does not expose ``sys.stdlib_module_names`` (added in
# 3.10).  ``requires-python >= 3.9`` makes this fallback reachable.
_PYTHON_39_STDLIB: frozenset[str] = frozenset({
    "__future__", "_abc", "_aix_support", "_ast", "_asyncio", "_bisect",
    "_blake2", "_bootlocale", "_bootsubprocess", "_bz2", "_codecs",
    "_codecs_cn", "_codecs_hk", "_codecs_iso2022", "_codecs_jp", "_codecs_kr",
    "_codecs_tw", "_collections", "_collections_abc", "_compat_pickle",
    "_compression", "_contextvars", "_crypt", "_csv", "_ctypes", "_curses",
    "_curses_panel", "_datetime", "_dbm", "_decimal", "_elementtree",
    "_frozen_importlib", "_frozen_importlib_external", "_functools", "_gdbm",
    "_hashlib", "_heapq", "_imp", "_io", "_json", "_locale", "_lsprof",
    "_lzma", "_markupbase", "_md5", "_msi", "_multibytecodec",
    "_multiprocessing", "_opcode", "_operator", "_osx_support", "_overlapped",
    "_pickle", "_posixshmem", "_posixsubprocess", "_py_abc", "_pydecimal",
    "_pyio", "_queue", "_random", "_sha1", "_sha256", "_sha3", "_sha512",
    "_signal", "_sitebuiltins", "_socket", "_sqlite3", "_sre", "_ssl", "_stat",
    "_statistics", "_string", "_strptime", "_struct", "_symtable", "_thread",
    "_threading_local", "_tkinter", "_tracemalloc", "_uuid", "_warnings",
    "_weakref", "_weakrefset", "_winapi", "abc", "aifc", "antigravity",
    "argparse", "array", "ast", "asynchat", "asyncio", "asyncore", "atexit",
    "audioop", "base64", "bdb", "binascii", "binhex", "bisect", "builtins",
    "bz2", "cProfile", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
    "code", "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "crypt", "csv", "ctypes", "curses", "dataclasses", "datetime",
    "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "ensurepip", "enum", "errno", "faulthandler", "fcntl",
    "filecmp", "fileinput", "fnmatch", "formatter", "fractions", "ftplib",
    "functools", "gc", "genericpath", "getopt", "getpass", "gettext", "glob",
    "graphlib", "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
    "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "lib2to3", "linecache",
    "locale", "logging", "lzma", "mailbox", "mailcap", "marshal", "math",
    "mimetypes", "mmap", "modulefinder", "msilib", "msvcrt", "multiprocessing",
    "netrc", "nis", "nntplib", "nt", "ntpath", "nturl2path", "numbers",
    "opcode", "operator", "optparse", "os", "ossaudiodev", "parser", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform", "plistlib",
    "poplib", "posix", "posixpath", "pprint", "profile", "pstats", "pty",
    "pwd", "py_compile", "pyclbr", "pydoc", "pydoc_data", "pyexpat", "queue",
    "quopri", "random", "re", "readline", "reprlib", "resource", "rlcompleter",
    "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex",
    "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
    "socketserver", "spwd", "sqlite3", "sre_compile", "sre_constants",
    "sre_parse", "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symbol", "symtable", "sys", "sysconfig", "syslog",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "test",
    "textwrap", "this", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "trace", "traceback", "tracemalloc", "tty", "turtle",
    "turtledemo", "types", "typing", "unicodedata", "unittest", "urllib",
    "uu", "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
    "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
    "zipfile", "zipimport", "zlib", "zoneinfo",
})

# Node.js built-in modules (committed static set).
_NODE_BUILTINS: frozenset[str] = frozenset({
    "assert", "async_hooks", "buffer", "child_process", "cluster", "console",
    "constants", "crypto", "dgram", "diagnostics_channel", "dns", "domain",
    "events", "fs", "http", "http2", "https", "inspector", "module", "net",
    "os", "path", "perf_hooks", "process", "punycode", "querystring",
    "readline", "repl", "stream", "string_decoder", "sys", "timers", "tls",
    "trace_events", "tty", "url", "util", "v8", "vm", "wasi", "worker_threads",
    "zlib",
})


def _python_stdlib_names() -> frozenset[str]:
    """Return the active stdlib module-name set, preferring the interpreter's.

    Uses ``sys.stdlib_module_names`` (Python 3.10+) when available, falling
    back to the committed Python 3.9 set otherwise.  Importability is never
    used to classify modules.
    """
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return frozenset(names)
    return _PYTHON_39_STDLIB


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def classify_non_project_dependency(raw: str, language: str) -> DependencyName:
    """Classify and canonicalize a single non-project dependency string.

    Returns a :class:`DependencyName` with ``kind`` of ``"external"`` or
    ``"sdk"``.  Never returns ``internal``.  The empty / whitespace-only input
    yields an empty canonical value, which callers discard.
    """
    raw = "" if raw is None else str(raw)
    value = raw.strip()
    if not value:
        return DependencyName(raw=raw, canonical="", kind=KIND_EXTERNAL)

    lang = (language or "").strip().lower()

    if lang == "dart":
        return _classify_dart(raw, value)
    if lang == "python":
        return _classify_python(raw, value)
    if lang in _NODE_LANGUAGES:
        return _classify_node(raw, value)

    # Unknown language: best-effort external with a trimmed canonical.
    return DependencyName(raw=raw, canonical=value, kind=KIND_EXTERNAL)


# ---------------------------------------------------------------------------
# Per-language behaviour
# ---------------------------------------------------------------------------

def _classify_dart(raw: str, value: str) -> DependencyName:
    # Dart SDK libraries: dart:async, dart:core, dart:io, ... — retain full id.
    if value.startswith("dart:"):
        return DependencyName(raw=raw, canonical=value, kind=KIND_SDK)

    # Strip exactly one leading package: prefix.
    if value.startswith("package:"):
        value = value[len("package:"):]

    if value.startswith(("./", "../")):
        return DependencyName(
            raw=raw, canonical=_normalize_relative(value), kind=KIND_EXTERNAL
        )

    # Canonicalize package subpaths to the first package segment.
    canonical = value.split("/", 1)[0]
    return DependencyName(raw=raw, canonical=canonical, kind=KIND_EXTERNAL)


def _classify_python(raw: str, value: str) -> DependencyName:
    # Relative imports (leading dot) are resolved via the graph, not here;
    # their top-level module is empty, so callers discard them.
    if value.startswith(("./", "../")):
        return DependencyName(
            raw=raw, canonical=_normalize_relative(value), kind=KIND_EXTERNAL
        )

    # Canonicalize to the top-level module.
    top = value.split(".", 1)[0].strip()
    if not top:
        # e.g. ".utils" -> empty top-level module; discarded by callers.
        return DependencyName(raw=raw, canonical="", kind=KIND_EXTERNAL)

    kind = KIND_SDK if top in _python_stdlib_names() else KIND_EXTERNAL
    return DependencyName(raw=raw, canonical=top, kind=kind)


def _classify_node(raw: str, value: str) -> DependencyName:
    # node:fs, node:fs/promises -> SDK after stripping the node: prefix.
    if value.startswith("node:"):
        stripped = value[len("node:"):]
        canonical = stripped.split("/", 1)[0]
        return DependencyName(raw=raw, canonical=canonical, kind=KIND_SDK)

    if value.startswith(("./", "../")):
        return DependencyName(
            raw=raw, canonical=_normalize_relative(value), kind=KIND_EXTERNAL
        )

    # Scoped package: @scope/package(/subpath) -> @scope/package.
    if value.startswith("@"):
        parts = value.split("/")
        canonical = "/".join(parts[:2]) if len(parts) >= 2 else value
        return DependencyName(raw=raw, canonical=canonical, kind=KIND_EXTERNAL)

    # Unscoped: first segment is the package / built-in name.
    first = value.split("/", 1)[0]
    kind = KIND_SDK if first in _NODE_BUILTINS else KIND_EXTERNAL
    return DependencyName(raw=raw, canonical=first, kind=kind)


def _normalize_relative(value: str) -> str:
    """Return a relative import spelling with posix separators, prefix kept."""
    return value.replace("\\", "/")
