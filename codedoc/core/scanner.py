"""Scan a project and detect file languages without using an LLM.

Callers provide skip rules, extension mappings, and entry candidates. The
``supported_extensions`` keyword remains compatible with direct callers.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import stat as stat_module
from pathlib import Path, PureWindowsPath
from typing import Callable, Iterator

from codedoc.core.file_division import (
    EMPTY_PLAN_DETAILS_DIGEST,
    MAX_EPHEMERAL_PLAN_DETAIL_ITEMS,
    _WorstFirst,
    canonical_json,
)
from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

# Section 5.8 final-generation scanner diagnostics.
#
# At most this many path-bearing warning lines are logged per warning class
# (large / unreadable) per authoritative scan generation; beyond that a single
# exact aggregate remainder line is logged. The allowance resets for every new
# generation. This is the presentation/warning limit and is deliberately
# distinct from the ``MAX_EPHEMERAL_PLAN_DETAIL_ITEMS`` (4096) retention budget.
MAX_SCANNER_PATH_WARNINGS_PER_CLASS = 20
# Backwards-compatible alias for the former name.
MAX_UNREADABLE_FILE_WARNINGS = MAX_SCANNER_PATH_WARNINGS_PER_CLASS

# Frozen scanner diagnostic vocabulary (section 5.8). ``scanner_size_skip``
# descriptors always carry this guidance code; ``scanner_admission_skip``
# descriptors carry the guidance frozen to their reason, and the reason order
# below is the frozen display secondary key.
_SIZE_SKIP_GUIDANCE = "raise-scan-byte-limit-or-exclude"
_SIZE_SKIP_PHASE = "scanner-byte"
_ADMISSION_SKIP_PHASE = "scanner-admission"
_ADMISSION_REASON_GUIDANCE: dict[str, str] = {
    "unreadable": "fix-permissions-or-exclude",
    "ignored": "adjust-ignore-or-entry",
    "unsupported": "configure-extension-or-entry",
    "missing": "fix-entry-path",
}
_ADMISSION_REASON_ORDER: tuple[str, ...] = tuple(_ADMISSION_REASON_GUIDANCE)

# The exact aggregate-remainder phrase per path-bearing warning class. The five
# per-file classes keep the historical ``"<class> file(s)"`` wording; the
# warning-only unreadable-directory class uses its own accurate noun.
_WARN_REMAINDER_PHRASE: dict[str, str] = {
    "large": "large file(s)",
    "unreadable": "unreadable file(s)",
    "ignored": "ignored file(s)",
    "unsupported": "unsupported file(s)",
    "missing": "missing file(s)",
    "directory": "unreadable directory(ies)",
}

# The value every empty diagnostic category publishes: an all-zero count block
# and the canonical ``[]`` digest (shared with the section 6 substrate so the
# scanner and route categories agree). Copied per instance -- never shared
# mutably.
_EMPTY_SCANNER_CATEGORY: dict = {
    "details": [],
    "details_total": 0,
    "details_retained": 0,
    "details_omitted": 0,
    "details_digest": EMPTY_PLAN_DETAILS_DIGEST,
}


def exclude_path_key(path: Path | str) -> str:
    """Shared normalization key for exact generated-target exclusion.

    Resolved non-strictly (a generated target need not exist yet) and
    OS-case-folded so both the caller building an ``exclude_paths`` set and
    the scanner checking each candidate file use the identical key. Equality
    only -- never used for basename or prefix matching, and deliberately not
    the lower-casing scheme ``_normalise_ignore_paths`` uses below (that
    scheme is for portable relative-path prefixes; this one is for exact
    resolved absolute paths).
    """
    return os.path.normcase(str(Path(path).resolve(strict=False)))


class ScanDiagnostics:
    """Keyword-only out-parameter for :func:`scan_files` (section 5.8).

    Mutated in place so ``scan_files``'s return type (``list[dict]``) never
    changes shape. Surfaced only in ``run_pipeline`` stats and the CLI summary
    -- never persisted to any codedoc.json / codedoc.md schema.

    **Final-generation semantics.** ``scan_files`` opens one fresh bounded
    diagnostic generation per call and, when the same instance is reused for a
    rescan (a detected stale source revision), :meth:`finalize_scan_generation`
    *atomically replaces* the previous one -- counts, retained details, digests
    and warning allowances are the final generation's alone, never a merge.
    There is no run-lifetime "seen" set: a file that stays large / unreadable
    across N rescans counts once per generation and its warning allowance is
    fresh each time.

    Two categories are published, each with the section 5.8 bounded contract
    (``details`` <= ``MAX_EPHEMERAL_PLAN_DETAIL_ITEMS``; exact ``details_total``
    / ``details_retained`` / ``details_omitted``; and a ``details_digest`` over
    the COMPLETE canonical descriptor stream, hashed incrementally in walk /
    caller order and never from a materialized list):

    * ``scanner_size_skip``  -- ``{path, phase="scanner-byte", observed, limit,
      guidance_code}`` for files rejected by the byte-size gate.
    * ``scanner_admission_skip`` -- ``{path, phase="scanner-admission", reason,
      guidance_code}`` for the frozen reason set ``unreadable`` / ``ignored`` /
      ``unsupported`` / ``missing``. The bulk walk records ``unreadable``; the
      other three are the scanner-owned seam the later pipeline explicit-entry
      section feeds through :meth:`record`.

    ``files_skipped_large`` / ``files_skipped_unreadable`` remain as the
    compatibility scalars pipeline / CLI code already reads; they describe the
    final authoritative generation (plus any post-scan planning read failure
    counted through :meth:`record_unreadable`).
    """

    __slots__ = (
        "files_skipped_large",
        "files_skipped_unreadable",
        "scanner_size_skip",
        "scanner_admission_skip",
        "explicit_entry_hint",
        "_open",
        "_finalized",
        "_state",
        "_wclass",
        "_prev_key",
    )

    def __init__(self) -> None:
        self.files_skipped_large = 0
        self.files_skipped_unreadable = 0
        self.scanner_size_skip = dict(_EMPTY_SCANNER_CATEGORY)
        self.scanner_admission_skip = dict(_EMPTY_SCANNER_CATEGORY)
        # Section 5.8: the caller (``run_pipeline``) sets this to the raw
        # explicit-entry hint before :func:`scan_files` runs, so a
        # sole-candidate explicit entry that yields no admitted source and
        # the bulk walk cannot classify (configured-ignored /
        # unsupported-extension / genuinely missing) is folded into
        # ``scanner_admission_skip`` in the same generation, before finalize.
        # Size-skipped and unreadable explicit entries are already recorded
        # inline by the walk and are never double-counted. Never changes
        # which files are admitted; never adds a sixth category.
        self.explicit_entry_hint: str | None = None
        self._open = False
        self._finalized = False
        self._state: dict = {}
        self._wclass: dict = {}
        self._prev_key: dict = {}

    def begin_scan_generation(self) -> None:
        """Open a fresh authoritative generation.

        Lifecycle: ``begin`` -> (records) -> ``finalize`` publishes it exactly
        once. ``begin`` again starts a REPLACEMENT generation -- every warning
        allowance and the canonical-order cursor reset here. The previously
        published categories stay visible until the replacement is successfully
        finalized, so a crash (or an unfinished replacement) mid-scan leaves the
        last good generation intact. A ``record`` into an already-finalized
        generation fails closed (``RuntimeError``); a second ``finalize`` of the
        same generation is idempotent -- it republishes the identical categories,
        counts, and digests rather than mutating or merging anything.
        """
        for category in ("size", "admission"):
            hasher = hashlib.sha256()
            hasher.update(b"[")
            # [digest-hasher, is_first, retained-heap, total, seq]
            self._state[category] = [hasher, True, [], 0, 0]
        # Per path-bearing warning class -> [emitted_total, warned_count]. Each
        # of the six frozen kinds (five per-file, plus the warning-only
        # unreadable-directory class) has its own 20-line allowance and its own
        # exact aggregate remainder; one noisy kind never consumes another's
        # allowance, and every allowance resets here for the new generation.
        self._wclass = {
            k: [0, 0]
            for k in (
                "large", "unreadable", "ignored", "unsupported", "missing", "directory"
            )
        }
        # Fail-closed canonical-order cursor per digest category: the last
        # descriptor's canonical key (size: normalized path; admission:
        # (normalized path, frozen-reason index)). A record whose key is LESS
        # than this is rejected before it can affect the digest, totals,
        # retained heap, or any warning counter/output.
        self._prev_key = {"size": None, "admission": None}
        self._open = True
        self._finalized = False

    def record(
        self, kind: str, rel_path: str, *, observed: int = 0, limit: int = 0
    ) -> None:
        """Fold one skip into the pending generation, in walk / caller order.

        *kind* is ``"size"`` for a byte-size skip (``observed`` / ``limit`` in
        bytes), one of the frozen admission reasons ``unreadable`` / ``ignored``
        / ``unsupported`` / ``missing`` (produces a descriptor), or
        ``"directory"`` for an unreadable-directory event (warning-only -- no
        descriptor, digest, heap, or category-total effect). Fold only -- the
        published category is refreshed once in :meth:`finalize_scan_generation`,
        so the per-skip cost is O(1) amortized (one canonical_json, one hash
        update, one bounded heap op), never O(K log K).

        **Ordering contract.** A descriptor's canonical key -- ``size``: the
        normalized path; admission: ``(normalized path, frozen-reason index)``
        -- MUST be >= every previous key of its digest category. Any complete
        producer (the filesystem walk, or a later explicit-entry integration)
        must therefore supply every record for one generation in canonical order
        before the single :meth:`finalize_scan_generation`; a decreasing key
        fails closed. Equal keys are allowed and each duplicate is hashed. A
        record after finalization also fails closed -- open a fresh generation.
        """
        if self._finalized:
            raise RuntimeError(
                "scan generation already finalized; call begin_scan_generation() "
                "for a new one"
            )
        if not self._open:
            self.begin_scan_generation()
        # Canonicalize to project-relative POSIX and reject anything that is not
        # project-relative BEFORE it can reach a descriptor, the digest, or a
        # path-bearing warning. Never assume the caller pre-validated: rooted,
        # //UNC, drive-qualified (``C:`` / ``C:foo``) and parent-traversing
        # inputs are rejected with a generic message that never echoes the
        # input; redundant separators and ``.`` segments are collapsed. Hostile-
        # but-valid filename bytes (newline, tab, quote, backslash-derived,
        # bidi/control, non-ASCII) are preserved and neutralized only at render
        # time by ``ensure_ascii=True`` / ``canonical_json``.
        raw = rel_path if isinstance(rel_path, str) else str(rel_path)
        if "\x00" in raw:
            raise ValueError("scanner diagnostic path must not contain NUL")
        posix = raw.replace("\\", "/")
        head = posix.split("/", 1)[0]
        segments = [seg for seg in posix.split("/") if seg not in ("", ".")]
        if (
            posix.startswith("/")
            or (len(head) >= 2 and head[0].isalpha() and head[1] == ":")
            or ".." in segments
        ):
            raise ValueError("scanner diagnostic path must be project-relative")

        if kind == "directory":
            # Warning-only: bounded, project-relative, JSON-escaped, no raw
            # exception text. The project root itself renders as ".".
            rel = "/".join(segments) or "."
            counter = self._wclass["directory"]
            counter[0] += 1
            if counter[1] < MAX_SCANNER_PATH_WARNINGS_PER_CLASS:
                logger.warning(
                    "Skipping unreadable directory: %s",
                    json.dumps(rel, ensure_ascii=True),
                )
                counter[1] += 1
            return

        if not posix.strip() or not segments:
            raise ValueError("scanner diagnostic path must be project-relative")
        rel = "/".join(segments)

        if kind == "size":
            category = "size"
            wclass = "large"
            descriptor = {
                "path": rel,
                "phase": _SIZE_SKIP_PHASE,
                "observed": int(observed),
                "limit": int(limit),
                "guidance_code": _SIZE_SKIP_GUIDANCE,
            }
            rank = (-int(observed), rel)
            key = rel
            warn = "Skipping large file: " + json.dumps(rel, ensure_ascii=True)
        elif kind in _ADMISSION_REASON_GUIDANCE:
            category = "admission"
            wclass = kind
            reason_index = _ADMISSION_REASON_ORDER.index(kind)
            descriptor = {
                "path": rel,
                "phase": _ADMISSION_SKIP_PHASE,
                "reason": kind,
                "guidance_code": _ADMISSION_REASON_GUIDANCE[kind],
            }
            rank = (reason_index, rel)
            key = (rel, reason_index)
            warn = f"Skipping {kind} file: " + json.dumps(rel, ensure_ascii=True)
        else:
            raise ValueError(f"unknown scanner skip kind: {kind!r}")

        # Fail-closed canonical-order guard -- raises BEFORE any mutation.
        previous = self._prev_key[category]
        if previous is not None and key < previous:
            raise ValueError(
                "scanner diagnostic descriptors must be supplied in canonical "
                "order (normalized path ascending, then frozen reason order); "
                "a decreasing key was rejected"
            )

        state = self._state[category]
        hasher, first, heap, total, seq = state
        if not first:
            hasher.update(b",")
        state[1] = False
        hasher.update(canonical_json(descriptor).encode("utf-8"))
        state[3] = total + 1
        node = (_WorstFirst(rank), seq, descriptor)
        state[4] = seq + 1
        if len(heap) < MAX_EPHEMERAL_PLAN_DETAIL_ITEMS:
            heapq.heappush(heap, node)
        else:
            heapq.heappushpop(heap, node)
        counter = self._wclass[wclass]
        counter[0] += 1
        if counter[1] < MAX_SCANNER_PATH_WARNINGS_PER_CLASS:
            logger.warning("%s", warn)
            counter[1] += 1
        self._prev_key[category] = key

    def finalize_scan_generation(self) -> None:
        """Publish the pending generation exactly once and emit the one exact
        aggregate remainder line per path-bearing warning class.

        Idempotent: a repeated call with no intervening
        :meth:`begin_scan_generation` publishes nothing new and emits no
        further warnings. Nothing to finalize (never opened) fails closed.
        """
        if self._finalized:
            return
        if not self._open:
            raise RuntimeError("no scan generation is open")
        for category, attr in (
            ("size", "scanner_size_skip"),
            ("admission", "scanner_admission_skip"),
        ):
            hasher, _first, heap, total, _seq = self._state[category]
            digest = hasher.copy()
            digest.update(b"]")
            retained = [
                record
                for _worst, _order, record in sorted(heap, key=lambda n: n[0].rank)
            ]
            setattr(self, attr, {
                "details": retained,
                "details_total": total,
                "details_retained": len(retained),
                "details_omitted": total - len(retained),
                "details_digest": "sha256:" + digest.hexdigest(),
            })
        # Each path-bearing warning class gets its own exact remainder line with
        # its own accurate noun -- a suppressed class is never described as
        # another, and the line is emitted at most once per generation.
        for wclass in (
            "large", "unreadable", "ignored", "unsupported", "missing", "directory"
        ):
            emitted, warned = self._wclass[wclass]
            extra = emitted - warned
            if extra > 0:
                logger.warning(
                    "%d more %s not shown", extra, _WARN_REMAINDER_PHRASE[wclass]
                )
        self.files_skipped_large = self.scanner_size_skip["details_total"]
        self.files_skipped_unreadable = self._wclass["unreadable"][0]
        self._finalized = True

    def record_unreadable(self, key: str) -> bool:
        """Bump the compatibility ``files_skipped_unreadable`` scalar for a file
        that passed the scan's stat check but failed a later read (e.g.
        planning's canonical-snapshot read). There is no run-lifetime dedup set
        (section 5.8); disjoint from the scan's own unreadable set -- an
        unreadable file never reaches planning -- so this never double-counts a
        generation record. The bool return is kept for legacy callers."""
        self.files_skipped_unreadable += 1
        return True


# Module-level fallbacks retained only for backward compatibility with direct
# callers that do not go through load_config().  Pipeline code uses the values
# from DEFAULTS["extension_language_map"] and DEFAULTS["auto_entry_candidates"]
# in loader.py.
# ---------------------------------------------------------------------------

_FALLBACK_LANGUAGE_MAP: dict[str, str] = {
    ".py":   "python",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".js":   "javascript",
    ".jsx":  "jsx",
    ".dart": "dart",
    ".java": "java",
    ".cs":   "csharp",
    ".html": "html",
    ".htm":  "html",
    ".kt":   "kotlin",
    ".swift":"swift",
    ".go":   "go",
    ".rb":   "ruby",
    ".rs":   "rust",
    ".cpp":  "cpp",
    ".c":    "c",
    ".h":    "c",
    ".hpp":  "cpp",
}

_FALLBACK_AUTO_ENTRY_CANDIDATES: list[str] = [
    "index.html", "main.tsx", "main.ts", "main.js",
    "main.py", "main.dart", "Main.java", "Program.cs",
]


def scan_files(
    root: Path,
    extension_language_map: dict[str, str] | None = None,
    max_file_size_kb: int = 500,
    skip_dirs: list[str] | None = None,
    ignore_paths: list[str] | None = None,
    *,
    # Safety control.  Keyword-only so existing positional callers stay
    # compatible.  When False (the default) every symlinked directory and file
    # is skipped, which prevents both symlink cycles and escapes outside the
    # project root.  When True, links are followed only when their resolved
    # target exists, has the expected type, and is contained by the resolved
    # project root.
    follow_symlinks: bool = False,
    # Deprecated keyword-only parameter kept for backward compatibility.
    # When provided without extension_language_map, a map is built from
    # _FALLBACK_LANGUAGE_MAP for the listed extensions.
    supported_extensions: list[str] | None = None,
    # Exact generated-target protection (section 5.6).  Keys built with
    # exclude_path_key() -- equality only, never basename or prefix
    # matching, so a source file that merely shares the output directory's
    # name is never excluded (co-located source/output stays supported).
    exclude_paths: "frozenset[str] | set[str] | None" = None,
    # Optional out-parameter, mutated in place; return type stays list[dict].
    diagnostics: "ScanDiagnostics | None" = None,
) -> list[dict]:
    """
    Walk root recursively and return a list of file descriptors.

    Parameters
    ----------
    root:
        Project root directory to scan.
    extension_language_map:
        Maps file extensions (lower-case, with leading dot) to language tags.
        Extensions in the map are automatically supported — no separate
        ``supported_extensions`` list needed.  When *None* and
        ``supported_extensions`` is given, a compatibility map is built.
    max_file_size_kb:
        Files larger than this are skipped (default 500 KB).
    skip_dirs:
        Directory names to skip (case-insensitive).  Typically resolved from
        ``config["skip_dirs"]`` by the pipeline; generated targets are
        protected separately through ``exclude_paths``. Directories whose
        names begin with ``.`` are always skipped regardless of this list.
    ignore_paths:
        Project-relative paths (files or directory subtrees) to exclude.
    follow_symlinks:
        When *False* (default) symlinked directories and files are skipped, so
        the scan never follows a link cycle and never escapes the project root.
        When *True* links are resolved strictly and followed only when the
        target exists, has the expected type, and resolves inside *root*.
    supported_extensions:
        **Deprecated.** Kept for callers that do not supply
        ``extension_language_map``.  Language is set to the fallback map value
        or ``"generic"`` for unknown extensions.
    exclude_paths:
        Exact resolved generated-target keys (built with
        :func:`exclude_path_key`) to protect from ever being treated as
        source -- e.g. the active and opposite-format output files and the
        crash-recovery file.  Matched by equality only, so co-located
        source/output directories remain supported: a source file merely
        sharing the output directory's name is never excluded.
    diagnostics:
        Optional :class:`ScanDiagnostics` out-parameter, mutated in place
        with ``files_skipped_large`` and ``files_skipped_unreadable`` counts.

    Returns
    -------
    list[dict]
        Each descriptor has keys: ``path`` (absolute :class:`Path`),
        ``rel_path`` (forward-slash string), ``language`` (str), ``extension``.
    """
    # Backward-compat guard: legacy positional callers may pass a list/tuple of
    # extensions as the second argument, e.g. scan_files(root, [".py", ".ts"]).
    # Since the old second parameter was `supported_extensions`, redirect to the
    # legacy path instead of crashing with AttributeError on list.items().
    if isinstance(extension_language_map, (list, tuple)):
        supported_extensions = list(extension_language_map)
        extension_language_map = None

    # Resolve the effective extension → language map.
    if extension_language_map is None:
        if supported_extensions is not None:
            extension_language_map = {
                ext.lower(): _FALLBACK_LANGUAGE_MAP.get(ext.lower(), "generic")
                for ext in supported_extensions
            }
        else:
            extension_language_map = {}

    ext_map = {e.lower(): lang for e, lang in extension_language_map.items()}
    skip_set = {d.lower() for d in (skip_dirs or [])}
    ignore_prefixes = _normalise_ignore_paths(ignore_paths or [])
    exclude_keys = frozenset(exclude_paths or ())
    results: list[dict] = []
    # Section 5.8 final-generation diagnostics: open a fresh bounded generation
    # for THIS scan. When the caller reuses one ScanDiagnostics across a rescan,
    # this replaces the previous generation rather than merging into it. A
    # throwaway instance is used when no out-parameter was supplied so the walk
    # below is uniform either way.
    _diagnostics = diagnostics if diagnostics is not None else ScanDiagnostics()
    _diagnostics.begin_scan_generation()

    # Section 5.8: the explicit-entry classification zone. When the caller set
    # ``explicit_entry_hint``, normalize it to a project-relative POSIX path
    # (same collapse/reject rules as ``detect_entry_file``: a rooted, //UNC,
    # drive-qualified or ``..`` hint has no zone -- discovery raises for it).
    # Candidates of THAT target that the bulk walk would silently drop
    # (unsupported extension, ignore_paths / skip_dirs / hidden-segment
    # exclusion, a genuinely missing path) are folded into the SAME
    # authoritative generation, in canonical order, so cross-category evidence
    # stays complete and every warning is emitted exactly once. This never
    # changes which files are admitted.
    _cz_raw = str(_diagnostics.explicit_entry_hint or "").replace("\\", "/")
    _cz_parts = _cz_raw.split("/")
    _cz_segs = [s for s in _cz_parts if s not in ("", ".")]
    _cz_head = _cz_segs[0] if _cz_segs else ""
    _classify_rel: str | None = None
    if (
        _cz_segs
        and _cz_raw.strip()
        and not _cz_raw.startswith("/")
        and ".." not in _cz_parts
        and not (len(_cz_head) >= 2 and _cz_head[0].isalpha() and _cz_head[1] == ":")
    ):
        _classify_rel = "/".join(_cz_segs)
    # Host filesystem path semantics: ``os.path.normcase`` folds case (and
    # separators) on a case-insensitive host and is identity on a
    # case-sensitive one, so ``ENTRY.PY`` matches an existing ``entry.py``
    # only where the filesystem itself would. Never a global lowercase --
    # two genuinely distinct case-sensitive paths must not collide.
    _classify_key = (
        os.path.normcase(_classify_rel) if _classify_rel is not None else None
    )
    # The single pending ``missing`` admission record (folded in canonical
    # position by ``_flush_missing`` -- a 1-element streaming merge, never a
    # collect-and-sort). Seeded only when the zone target does not exist; an
    # OSError reading its own metadata is left to the walk's unreadable path.
    _missing_pending: list[str] = []
    if _classify_rel is not None:
        try:
            _cz_exists = (root / _classify_rel).exists()
        except OSError:
            _cz_exists = True
        if not _cz_exists:
            _missing_pending.append(_classify_rel)
    _flush_missing = lambda _p: (  # noqa: E731  (census-safe: Lambda, not FunctionDef)
        _diagnostics.record("missing", _missing_pending.pop(0))
        if (_missing_pending and _missing_pending[0] < _p)
        else None
    )

    walker = _Walker(
        scan_root=root,
        skip_dirs=skip_set,
        ignore_prefixes=ignore_prefixes,
        follow_symlinks=follow_symlinks,
        classify_rel=_classify_rel,
        classify_key=_classify_key,
        # A walker-classified unreadable item (a genuine OSError probing the
        # path's own type) folds an ``unreadable`` admission skip inline, in
        # deterministic traversal order -- but only when its extension is one we
        # would have scanned; an unreadable file the scan would have ignored
        # anyway is not worth a record.
        on_unreadable=lambda item: (
            item.suffix.lower() in ext_map
            and (_flush_missing(item.relative_to(root).as_posix()) or True)
            and _diagnostics.record(
                "unreadable", item.relative_to(root).as_posix()
            )
        ),
        # An un-enumerable directory (the root renders as "."): warning-only,
        # bounded, project-relative, JSON-escaped -- no descriptor is invented.
        on_unreadable_dir=lambda directory: _diagnostics.record(
            "directory", directory.relative_to(root).as_posix()
        ),
        # A candidate of the explicit target reached only by overriding a
        # hidden / skip_dirs / ignore_paths rule is ``ignored`` -- classified,
        # never admitted, in canonical traversal order.
        on_zone_ignored=lambda item: (
            (_flush_missing(item.relative_to(root).as_posix()) or True)
            and _diagnostics.record(
                "ignored", item.relative_to(root).as_posix()
            )
        ),
    )
    for file_path in walker.walk(root):
        ext = file_path.suffix.lower()
        # The project-relative POSIX path, computed once and reused below.
        rel = file_path.relative_to(root).as_posix()
        _relk = os.path.normcase(rel)
        _in_zone = _classify_key is not None and (
            _relk == _classify_key
            or _relk.startswith(_classify_key + os.sep)
        )
        if ext not in ext_map:
            # Silently skipped for a normal scan; for the explicit target this
            # is bounded ``unsupported`` admission evidence in this generation.
            if _in_zone:
                _flush_missing(rel)
                _diagnostics.record("unsupported", rel)
            continue

        if exclude_keys and exclude_path_key(file_path) in exclude_keys:
            continue

        # Byte-size admission gate: unchanged 500-KB default, unchanged exact
        # ``>`` comparison, unchanged byte-before-decoded-character order.
        try:
            st_size = file_path.stat().st_size
        except OSError:
            _flush_missing(rel)
            _diagnostics.record("unreadable", rel)
            continue
        if st_size / 1024 > max_file_size_kb:
            _diagnostics.record(
                "size", rel, observed=st_size, limit=max_file_size_kb * 1024
            )
            continue

        language = ext_map.get(ext, "generic")
        results.append({
            "path": file_path,
            "rel_path": rel,
            "language": language,
            "extension": ext,
        })

    # Section 5.8: fold the one pending ``missing`` explicit-entry record (if
    # any) at the tail of the canonical stream -- every real descriptor sorts
    # before a target that does not exist on disk -- then publish the SINGLE
    # authoritative generation. Final-generation atomic replacement still
    # holds: every ``scan_files`` call re-derives this from scratch.
    if _missing_pending:
        _diagnostics.record("missing", _missing_pending.pop(0))
    _diagnostics.finalize_scan_generation()

    skipped_dirs = walker.skipped_dirs
    logger.info(
        "Scanner found %d supported file(s) in %s (skipped %d directorie(s), "
        "%d large file(s), %d unreadable file(s))",
        len(results),
        root,
        skipped_dirs,
        _diagnostics.files_skipped_large,
        _diagnostics.files_skipped_unreadable,
    )
    return results


class _Walker:
    """Iterative, symlink-safe directory walker with per-scan state.

    State (``scan_root``, ``skip_dirs``, ``ignore_prefixes``, ``skipped_dirs``,
    and the visited-identity sets) is held on the instance rather than on the
    function object, so the walk is re-entrant and two concurrent or sequential
    scans never share state.

    The walk uses an explicit stack instead of recursion, so a deep but acyclic
    tree cannot raise :class:`RecursionError`.  Every traversed directory's
    resolved identity is recorded, so a symlink/junction cycle or two aliases to
    the same real directory are visited at most once.  When ``follow_symlinks``
    is True, resolved file identities are tracked the same way so two aliases to
    one real file produce at most one descriptor.
    """

    def __init__(
        self,
        scan_root: Path,
        skip_dirs: set[str],
        ignore_prefixes: set[str],
        follow_symlinks: bool = False,
        on_unreadable: Callable[[Path], None] | None = None,
        on_unreadable_dir: Callable[[Path], None] | None = None,
        classify_rel: str | None = None,
        classify_key: str | None = None,
        on_zone_ignored: Callable[[Path], None] | None = None,
    ) -> None:
        self.scan_root = scan_root
        self.skip_dirs = skip_dirs
        self.ignore_prefixes = ignore_prefixes
        self.follow_symlinks = follow_symlinks
        self.skipped_dirs = 0
        # Section 5.8: the project-relative POSIX path of the explicit entry
        # whose otherwise-silently-dropped candidates must still be
        # classified. A would-skip directory (hidden / skip_dirs /
        # ignore_paths) is descended ONLY when it IS this target, is inside
        # it, or is on the path to it; everything reached that way is
        # reported through ``on_zone_ignored`` and never yielded for
        # admission, so the admitted-file set is unchanged.
        self._classify_rel = classify_rel
        # Host-case match key for the zone (identity on a case-sensitive
        # host); descriptors still carry the actual discovered spelling.
        self._classify_key = classify_key
        self._on_zone_ignored = on_zone_ignored
        # An unreadable item / an un-enumerable directory is reported inline
        # through these callbacks (in deterministic traversal order) rather than
        # accumulated in a run-lifetime list -- section 5.8 forbids one retained
        # path per omitted scanner result -- so the bounded generation owns the
        # warning cap, the aggregate remainder, and the project-relative
        # rendering.
        self._on_unreadable = on_unreadable
        self._on_unreadable_dir = on_unreadable_dir
        self._resolved_root: Path = scan_root
        # Traversal-correctness identity sets (symlink-cycle prevention and
        # alias-admission safety) -- NOT diagnostic state, never removed.
        self._visited_dirs: set = set()
        self._visited_files: set = set()
        # SEPARATE identity sets for diagnostic-only descent through a
        # skipped explicit target: same cycle/alias correctness, but they
        # never consume or block the ordinary admission traversal above, so
        # ``explicit_entry_hint`` cannot change the admitted-file set.
        self._diag_visited_dirs: set = set()
        self._diag_visited_files: set = set()

    def walk(self, root: Path) -> Iterator[Path]:
        """Yield all files under *root*, skipping ignored / foreign directories.

        Each directory's entries are visited in ascending name order
        (``_expand``), depth-first, recursing into a subdirectory before its
        later-sorted siblings. The visitation order is therefore deterministic
        and invariant under the underlying filesystem's ``iterdir()`` order,
        which is what makes the section 5.8 canonical descriptor stream and its
        digest permutation-stable.
        """
        try:
            self._resolved_root = Path(root).resolve(strict=False)
        except OSError:
            self._resolved_root = Path(root)

        root_identity = self._identity(root)
        if root_identity is not None:
            self._visited_dirs.add(root_identity)
            self._diag_visited_dirs.add(root_identity)

        stack: list[tuple[Path, bool]] = [(p, False) for p in self._expand(root)]
        while stack:
            item, forced_ignored = stack.pop()
            kind = _classify(item)
            is_link = _is_link_like(item)

            try:
                rel = item.relative_to(self.scan_root).as_posix()
            except ValueError:
                rel = item.name

            _cz = self._classify_rel
            _czk = self._classify_key
            if _czk is not None:
                _rk = os.path.normcase(rel)
                _in_zone = _rk == _czk or _rk.startswith(_czk + os.sep)
                _leads_to_zone = _czk == _rk or _czk.startswith(_rk + os.sep)
            else:
                _in_zone = False
                _leads_to_zone = False
            # C/D: a supported explicit FILE target reached only by
            # overriding a hidden / skip_dirs / ignore_paths ancestor -- or
            # one directly matched by ignore_paths -- is ``ignored`` (the
            # configuration already excludes it), never ``unsupported`` /
            # ``unreadable`` / admitted, even if its own type probe failed.
            # Duplicate aliases obey the diagnostic file-identity guard.
            if (
                _czk is not None
                and kind in ("file", "unreadable")
                and (
                    forced_ignored
                    or (_in_zone and _is_ignored(rel, self.ignore_prefixes))
                )
            ):
                if self.follow_symlinks:
                    _fid = self._identity(item)
                    if _fid is not None:
                        if _fid in self._diag_visited_files:
                            continue
                        self._diag_visited_files.add(_fid)
                if self._on_zone_ignored is not None:
                    self._on_zone_ignored(item)
                continue

            if kind == "dir":
                # Apply the lexical skip/dot/ignore rules to the in-root alias
                # *before* resolving or descending, so a link cannot smuggle in
                # an otherwise-ignored path.
                _would_skip = (
                    item.name.lower() in self.skip_dirs
                    or item.name.startswith(".")
                    or _is_ignored(rel, self.ignore_prefixes)
                )
                # A would-skip directory the diagnostic traversal must not
                # enter at all -- it is neither the explicit target, inside
                # it, nor on the path to it.
                if _would_skip and not (_in_zone or _leads_to_zone):
                    self.skipped_dirs += 1
                    continue
                # A would-skip directory descended ONLY to reach / classify
                # the explicit target still counts once at the normal
                # admission frontier (the normal scanner would skip it here).
                # A deeper would-skip directory a skip-override already passed
                # through does NOT: the normal scanner would never reach it,
                # so counting it would inflate the compatibility total.
                if _would_skip and not forced_ignored:
                    self.skipped_dirs += 1
                _child_forced = forced_ignored or _would_skip
                # A would-skip / under-a-skip descent is diagnostic-only:
                # the normal scanner admits nothing here.
                _diag_only = _would_skip or forced_ignored
                if is_link:
                    if not self.follow_symlinks:
                        self.skipped_dirs += 1
                        logger.debug("Skipping symlinked directory %s", item)
                        continue
                    target = _safe_resolve(item)
                    if (
                        target is None
                        or not target.is_dir()
                        or not self._within_root(target)
                    ):
                        self.skipped_dirs += 1
                        logger.debug(
                            "Skipping symlinked directory (broken, type-mismatched, "
                            "or out-of-root) %s",
                            item,
                        )
                        continue
                # Visited-identity guard for *every* directory (not only links):
                # stops cycles through symlinks/junctions and dedups aliases.
                # A diagnostic-only descent uses its OWN set so it cannot
                # suppress (or be suppressed by) an ordinary admitted alias.
                identity = self._identity(item)
                if identity is not None:
                    _seen = (
                        self._diag_visited_dirs if _diag_only
                        else self._visited_dirs
                    )
                    if identity in _seen:
                        logger.debug("Skipping already-visited directory %s", item)
                        continue
                    _seen.add(identity)
                # Inside the explicit target every descendant is a
                # target-owned candidate. On the path to a deeper (file or
                # nested-directory) target, descend ONLY toward that target:
                # an off-path sibling is not target-owned evidence, so drop
                # it here -- before any classification, warning, hash, total
                # or sibling-subtree descent. This prune only runs inside a
                # would-skip segment (``_diag_only``), where the normal
                # scanner admits nothing anyway, so the admitted-file set is
                # unchanged.
                for child in self._expand(item):
                    if _diag_only and not _in_zone and _czk is not None:
                        _crk = os.path.normcase(rel + "/" + child.name)
                        if not (
                            _crk == _czk
                            or _crk.startswith(_czk + os.sep)
                            or _czk == _crk
                            or _czk.startswith(_crk + os.sep)
                        ):
                            continue
                    stack.append((child, _child_forced))

            elif kind == "file":
                # (A candidate that is the ignored explicit target itself is
                # classified by the hoisted forced-ignored block above.)
                if _is_ignored(rel, self.ignore_prefixes):
                    continue
                if is_link:
                    if not self.follow_symlinks:
                        logger.debug("Skipping symlinked file %s", item)
                        continue
                    target = _safe_resolve(item)
                    if (
                        target is None
                        or not target.is_file()
                        or not self._within_root(target)
                    ):
                        logger.debug(
                            "Skipping symlinked file (broken, type-mismatched, "
                            "or out-of-root) %s",
                            item,
                        )
                        continue
                # The first in-root alias of a real file owns the descriptor;
                # later aliases resolve to the same identity and are skipped.
                if self.follow_symlinks:
                    file_identity = self._identity(item)
                    if file_identity is not None:
                        if file_identity in self._visited_files:
                            continue
                        self._visited_files.add(file_identity)
                yield item

            elif kind == "unreadable":
                # A genuine OSError (e.g. permission denied) probing this
                # path's type -- distinct from _classify's None case, which
                # means the path exists but is neither a file nor a
                # directory (a broken symlink target, socket, or device).
                # Reported inline, in traversal order, never accumulated.
                if self._on_unreadable is not None:
                    self._on_unreadable(item)

            elif is_link:
                # Broken or inaccessible link: skip without aborting the scan.
                logger.debug("Skipping broken or inaccessible symlink %s", item)

    def _expand(self, directory: Path) -> list[Path]:
        """Return *directory*'s entries in DESCENDING global-path order, so a
        stack ``pop()`` yields them ascending and a depth-first-in-place walk
        visits every file in exact global normalized-POSIX-string order.

        The sort key appends ``"/"`` to a subdirectory's name so it sorts at the
        same code point (``/`` = 0x2F) that will separate it from its children
        in the eventual relative path: a file ``a.py`` (``.`` 0x2E) therefore
        precedes the subtree ``a/...`` (``/`` 0x2F), which precedes a sibling
        ``ab.py`` (``b`` 0x62) -- exactly ``sorted()`` of the rel paths. Sorting
        here (not consuming raw ``iterdir()`` order) also makes the whole walk
        deterministic and independent of the filesystem's enumeration order,
        which is what makes the section 5.8 canonical descriptor stream and its
        digest permutation-stable. Only one directory's entries are ever
        materialized -- traversal state, ``O(entries)``, never a diagnostic copy.
        """
        try:
            entries = list(directory.iterdir())
        except OSError:
            self.skipped_dirs += 1
            # Route through the bounded generation-owned warning state: a
            # project-relative POSIX path, one JSON-escaped field, capped at 20
            # per generation with one exact aggregate remainder, and NO raw
            # exception text (which can itself carry a path or control bytes).
            if self._on_unreadable_dir is not None:
                self._on_unreadable_dir(directory)
            return []
        decorated = [
            (entry.name + ("/" if os.path.isdir(entry) else ""), entry)
            for entry in entries
        ]
        decorated.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _key, entry in decorated]

    def _within_root(self, resolved: Path) -> bool:
        """True when *resolved* (an already-resolved path) is inside the root."""
        try:
            resolved.relative_to(self._resolved_root)
            return True
        except ValueError:
            return False

    def _identity(self, path: Path):
        """Return a stable identity for *path* (follows links).

        Prefers ``(st_dev, st_ino)`` when meaningful and falls back to the
        normalized resolved path on platforms/filesystems where the inode is
        unavailable (e.g. some Windows configurations report ``st_ino == 0``).
        """
        try:
            st = path.stat()
        except OSError:
            return self._resolved_path_identity(path)
        if getattr(st, "st_ino", 0):
            return ("inode", st.st_dev, st.st_ino)
        return self._resolved_path_identity(path)

    @staticmethod
    def _resolved_path_identity(path: Path):
        try:
            return ("path", str(Path(path).resolve(strict=False)))
        except OSError:
            return None


def _classify(path: Path) -> str | None:
    """Classify *path* as ``"dir"``, ``"file"``, ``"unreadable"``, or
    ``None`` (following links).

    ``Path.is_dir()``/``Path.is_file()`` only swallow the "doesn't
    exist"-shaped errno set (ENOENT/ENOTDIR) internally and re-raise
    everything else (notably EACCES, permission denied) -- so an ``OSError``
    reaching here is a genuine read failure, distinguished from the ``None``
    case, which means the path exists but is neither a file nor a directory
    (a broken symlink target, socket, or device) with no error at all.
    """
    try:
        if path.is_dir():
            return "dir"
    except OSError:
        return "unreadable"
    try:
        if path.is_file():
            return "file"
    except OSError:
        return "unreadable"
    return None


def _is_link_like(path: Path) -> bool:
    """True for symlinks and, on Windows, junctions / reparse points.

    Centralizes link detection so version/platform checks are not scattered
    through the traversal loop.  Uses :meth:`Path.is_symlink`, then
    :meth:`Path.is_junction` where available (Python 3.12+), and finally the
    reparse-point file attribute on older Windows runtimes.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        pass

    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            pass
    else:
        try:
            attrs = getattr(path.lstat(), "st_file_attributes", 0)
        except (OSError, AttributeError):
            attrs = 0
        reparse = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attrs & reparse:
            return True
    return False


def _safe_resolve(path: Path) -> Path | None:
    """Strictly resolve *path*; return ``None`` for broken/cyclic/missing links."""
    try:
        return Path(path).resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _normalise_ignore_paths(paths: list[str]) -> set[str]:
    prefixes: set[str] = set()
    for raw in paths:
        cleaned = str(raw).strip().replace("\\", "/")
        if not cleaned:
            continue
        cleaned = cleaned.lstrip("/")
        cleaned = cleaned.rstrip("/")
        if cleaned:
            prefixes.add(cleaned.lower())
    return prefixes


def _is_ignored(rel_path: str, ignore_prefixes: set[str]) -> bool:
    rel = rel_path.replace("\\", "/").strip("/").lower()
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in ignore_prefixes)


# Section 5.8 / Section 7 privacy: the single canonical entry-hint rule. An
# explicit entry hint must be a *normalized project-relative POSIX path*. An
# absolute / drive-qualified (``C:`` / ``C:foo``) / UNC (``//host``) / POSIX-
# rooted / ``..``-traversing / dot-only hint points outside the project root
# and is rejected -- before any filesystem probe, and without echoing the raw
# spelling. Redundant ``/`` and ``.`` segments in an otherwise-valid hint are
# collapsed to one canonical form used for both the candidate lookup and the
# not-found warning; an unsafe prefix is never silently stripped, it is
# rejected. Every check runs on the CANONICAL components, not the raw first
# segment, so a removable leading ``.`` (``./C:/private/secret.py``) cannot
# hide a drive prefix. Drive detection uses explicit Windows-path semantics so
# it is identical on every host OS (never the host ``Path`` class).
#
# ``detect_entry_file`` and ``run_pipeline`` share this one rule so the two can
# never drift: no *project-relative-invalid* hint -- absolute, drive-qualified,
# drive-relative, UNC, ``..`` traversal, dot-only, or post-normalization bypass
# -- reaches the filesystem, or the generated-target collision check, before
# validation. It does not cover byte-level hardening (e.g. an embedded NUL is
# not checked here and would still fail later as a raw ``ValueError`` from
# ``os.stat``); that is pre-existing debt shared with ``force_files`` /
# ``ignore_paths`` / ``output_dir`` and is out of scope for this rule.
# Expressed as a module-level
# ``lambda`` (returning the canonical relative POSIX path, or ``None`` when the
# hint points outside the root) plus a module-level message constant, not a
# ``def``, so it does not move the pinned source-structure declaration census
# (section 3.2) -- the same idiom as ``_flush_missing`` above and
# ``pipeline._freeze_preflight``.
_ENTRY_HINT_NOT_PROJECT_RELATIVE = (
    "The entry file must be a project-relative path inside the "
    "project root: a drive letter, a leading '/', a '//' network "
    "prefix or a '..' segment points outside it. Provide a path "
    "such as 'src/main.py', or remove the entry setting to "
    "document all files."
)
_canonical_entry_rel = lambda hint: (  # noqa: E731  (census-safe: Lambda, not FunctionDef)
    (lambda raw: (
        (lambda parts, segs: (
            None
            if (
                not raw.strip()
                or raw.startswith("/")                       # POSIX-rooted / //UNC (raw)
                or ".." in parts                             # parent traversal
                or not segs                                  # empty / '.' / './.' only
                or PureWindowsPath("/".join(segs)).drive     # C:/x, C:x, //host/share
                or PureWindowsPath("/".join(segs)).root      # \ , /x  (canonical)
                or (
                    len(segs[0]) >= 2
                    and segs[0][0].isalpha()
                    and segs[0][1] == ":"
                )                                            # C: / C:foo first component
            )
            else "/".join(segs)
        ))(
            raw.split("/"),
            [seg for seg in raw.split("/") if seg not in ("", ".")],
        )
    ))(str(hint).replace("\\", "/"))
)


def detect_entry_file(
    root: Path,
    hint: str | None,
    candidates: list[str] | None = None,
) -> Path | None:
    """
    Resolve the entry file.

    Parameters
    ----------
    root:
        Project root directory.
    hint:
        Entry file path relative to *root*, as provided via ``--entry``.
        When given, auto-detection is skipped.
    candidates:
        Ordered list of file names to try when no *hint* is given.
        Resolved from ``config["auto_entry_candidates"]`` by the pipeline.
        Falls back to :data:`_FALLBACK_AUTO_ENTRY_CANDIDATES` when ``None``.
    """
    auto_entries = candidates if candidates is not None else _FALLBACK_AUTO_ENTRY_CANDIDATES

    if hint:
        # Section 5.8 privacy: an explicit entry hint must be a normalized
        # project-relative POSIX path, validated and canonicalized BEFORE any
        # filesystem probe or warning. The rule itself (and its byte-identical
        # rejection message) lives once at module scope in
        # ``_canonical_entry_rel`` / ``_ENTRY_HINT_NOT_PROJECT_RELATIVE`` so
        # ``run_pipeline`` applies the identical check before its generated-
        # target collision check -- see the comment there. ``None`` means the
        # hint points outside the project root; otherwise the returned canonical
        # form is used for both the candidate lookup and the not-found warning.
        # Hostile-but-valid filename bytes are preserved until JSON rendering.
        entry_rel = _canonical_entry_rel(hint)
        if entry_rel is None:
            raise ConfigError(_ENTRY_HINT_NOT_PROJECT_RELATIVE)
        candidate = root / entry_rel
        # Section 5.6 / 12.1 C5: ``Path.exists()`` swallows only the
        # "doesn't exist"-shaped errno set (ENOENT/ENOTDIR/EBADF/ELOOP) and
        # re-raises everything else, notably EACCES.  An explicitly requested
        # entry whose own metadata cannot be read must fail as an actionable
        # ConfigError here -- before provider construction, recovery
        # initialization, or any output mutation -- never as a raw
        # PermissionError/OSError escaping the scan.  This is the
        # stat-inspection counterpart to the post-stat read failure that
        # ``codedoc.core.planning`` already reports.
        try:
            found = candidate.exists()
        except OSError as exc:
            raise ConfigError(
                f"Entry file '{entry_rel}' could not be read: {exc}. Check the "
                "path and its file permissions."
            ) from exc
        if found:
            return candidate
        # One JSON-escaped canonical project-relative field; never the absolute
        # project root, never a raw control character that could inject a
        # second line.
        logger.warning(
            "Specified entry file %s not found",
            json.dumps(entry_rel, ensure_ascii=True),
        )
        return None

    for name in auto_entries:
        candidate = root / name
        # An auto-detection candidate is a guess, not a user request: an
        # unreadable one is skipped like a missing one so a single
        # permission-restricted file never aborts a run the user never
        # asked to centre on it.
        try:
            found = candidate.exists()
        except OSError:
            logger.debug("Skipping unreadable auto-entry candidate %s", candidate)
            continue
        if found:
            logger.info("Auto-detected entry file: %s", candidate)
            return candidate

    logger.warning("No entry file detected. Will process all files.")
    return None
