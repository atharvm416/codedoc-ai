"""Main documentation pipeline for codedoc.

0.8.0 changes
-------------
- **Always-on live JSON backup** (Work Item 1): ``SafeWriter`` is now the
  default recorder for every run.  ``--safe-mode`` is a deprecated no-op.
  The live backup is created before AI work starts and finalized (banner
  removed) by ``write_project_outputs`` on clean completion.
- **Worker-thread recording** (Work Item 2): in parallel mode, ``record()``
  is called inside the worker via ``_process_and_record()``, so a Ctrl-C or
  crash never discards a file whose AI work already completed.
- **Live backup path helper** (Work Item 1): ``_resolve_live_backup_path()``
  centralises path logic so the MD-sibling case is handled consistently
  across writer creation, ownership checks, resume, stats, and cleanup.
- **error.log in output_dir** (Work Item 4): ``ErrorReporter`` is now
  initialised with ``output_dir / "error.log"`` instead of ``root / "error.log"``.
- **Error stats always set** (Work Item 4): ``stats["error_log"]``,
  ``stats["issues_recorded"]``, and ``stats["live_backup_path"]`` are
  populated on every return path.
- **Checkpoint migration** (Work Item 1): ``Checkpoint`` entries are loaded
  as a fallback when no live backup exists, then retired after the first
  successful run.
- **Rate-limit ladder** (Work Item 3): ``_is_rate_limit_error()``,
  ``_build_default_ladder()``, and ladder-based retry batches implemented in
  ``_process_agent_files()``.
"""

from __future__ import annotations

import concurrent.futures
import re
import sys
import time
from pathlib import Path
from typing import Any

from codedoc.agents.base_agent import truncate_for_llm
from codedoc.agents.orchestrator import Orchestrator
from codedoc.bootstrap import ensure_codedoc_installed
from codedoc.core.checkpoint import Checkpoint
from codedoc.core.db import compute_file_hash
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.record_meta import carry_private_keys
from codedoc.core.safe_writer import SafeWriter
from codedoc.core.graph import DependencyGraph, resolve_import
from codedoc.core.loader import load_config
from codedoc.core.output import (
    BUILD_FILENAME,
    inspect_output_ownership,
    read_existing_records,
    write_project_outputs,
)
from codedoc.core.planning import (
    PipelinePlan,
    build_pipeline_plan,
    normalize_force_files,
)
from codedoc.core.project_view import markdown_to_view, read_codedoc_meta
from codedoc.core.queue import ProcessingQueue
from codedoc.core.scanner import detect_entry_file, scan_files
from codedoc.core.usage import UsageAccumulator, estimate_tokens
from codedoc.llm.factory import create_provider
from codedoc.llm.rate_limit_profile import RateLimitProfile, get_rate_limit_profile
from codedoc.parser.factory import parse_file
from codedoc.utils.errors import AgentError, ConfigError, ErrorReporter, OutputError, ParseError
from codedoc.utils.logger import get_logger, set_level

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rate-limit detection helpers (Work Item 3)
# ---------------------------------------------------------------------------

_RATE_LIMIT_SIGNALS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "tokens per min",
    "tpm",
    "quota",
    "resource_exhausted",   # Gemini / Google AI
    "overloaded",           # Anthropic
    "529",                  # Anthropic overloaded
    "503",
)


def _is_rate_limit_error(
    exc: BaseException,
    profile: RateLimitProfile | None = None,
) -> bool:
    """Return True if *exc* or any cause in its chain is a rate-limit signal.

    Inspects ``str(exc)`` and walks ``__cause__`` / ``__context__`` so that
    provider signals are not hidden by wrapper exceptions.

    Parameters
    ----------
    exc:
        The exception to classify.
    profile:
        0.8.1 — when supplied, only ``profile.signals`` are used for detection,
        giving provider-specific accuracy.  When ``None`` (backward-compat),
        the module-level ``_RATE_LIMIT_SIGNALS`` tuple is used so that existing
        callers without a profile continue to work unchanged.
    """
    signals = profile.signals if profile is not None else _RATE_LIMIT_SIGNALS
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current).lower()
        if any(sig in msg for sig in signals):
            return True
        current = current.__cause__ or current.__context__
    return False


# Pre-compiled patterns for _detect_limit_type.  Word boundaries ensure that
# "tpm" inside "uptime" does not match, and parenthesised forms like "(TPM)"
# do match because ( and ) are non-word characters.
_DETECT_TPM_RE = re.compile(r"\btpm\b", re.IGNORECASE)
_DETECT_RPM_RE = re.compile(r"\brpm\b", re.IGNORECASE)


def _detect_limit_type(error_msg: str) -> str | None:
    """Classify the kind of rate limit from an error message string.

    Returns one of ``"tpm"``, ``"rpm"``, ``"quota"``, ``"overloaded"``, or
    ``None`` when the type cannot be determined.  Patterns are checked
    case-insensitively in priority order.

    Examples that must classify correctly:
    - ``"429 tokens per min exceeded"`` → ``"tpm"``
    - ``"limit exceeded (TPM)"``        → ``"tpm"``
    - ``"requests per min exceeded"``   → ``"rpm"``
    - ``"daily quota exhausted"``       → ``"quota"``
    - ``"529 overloaded"``              → ``"overloaded"``
    - ``"429 too many requests"``       → ``None``
    """
    msg = error_msg.lower()
    if "tokens per min" in msg or _DETECT_TPM_RE.search(error_msg):
        return "tpm"
    if "requests per min" in msg or _DETECT_RPM_RE.search(error_msg):
        return "rpm"
    if "daily" in msg or "quota" in msg or "resource_exhausted" in msg:
        return "quota"
    if "overloaded" in msg or "529" in msg:
        return "overloaded"
    return None


def _build_default_ladder(max_p: int) -> list[int]:
    """Build the default parallelism step-down ladder for *max_p* workers."""
    if max_p <= 1:
        return [1]
    if max_p == 2:
        return [2, 1]
    mid = max(2, max_p // 2)
    ladder: list[int] = [max_p]
    if mid < max_p:
        ladder.append(mid)
    if 1 not in ladder:
        ladder.append(1)
    return ladder


def _parse_retry_after(exc: BaseException) -> float | None:
    """Extract a Retry-After delay (seconds) from the exception message chain."""
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        msg = str(current)
        m = re.search(r"try again in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"retry.after[:\s]+([\d.]+)", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        current = current.__cause__ or current.__context__
    return None


# ---------------------------------------------------------------------------
# Live backup path helper (Work Item 1)
# ---------------------------------------------------------------------------

def _resolve_live_backup_path(
    output_dir: Path,
    output_format: str,
    json_filename: str,
    md_filename: str,
) -> Path:
    """Return the absolute live JSON backup path for the given output scenario.

    | output_format | md_filename    | json_filename | Live backup path         |
    |---------------|----------------|---------------|--------------------------|
    | json / both   | (any)          | codedoc.json  | output_dir/codedoc.json  |
    | md            | codedoc.md     | (ignored)     | output_dir/codedoc.json  |
    | md            | report.md      | (ignored)     | output_dir/report.json   |

    For MD-only runs the backup is a JSON sibling of the Markdown file,
    derived from ``md_filename`` stem (not from ``json_filename``, which
    ``_resolve_output_spec`` leaves as the default ``codedoc.json`` even when
    the user requested ``--output docs/report.md``).
    """
    if output_format == "md":
        md_stem = Path(md_filename).stem
        return output_dir / f"{md_stem}.json"
    return output_dir / json_filename


# ---------------------------------------------------------------------------
# Existing-docs loader
# ---------------------------------------------------------------------------

def _load_existing_file_docs(
    output_dir: Path,
    json_filename: str,
    md_filename: str = "codedoc.md",
    live_backup_path: Path | None = None,
    read_only: bool = False,
) -> dict[str, dict]:
    """Load per-file documentation from existing output files.

    With ``read_only=True`` (planning / dry-run) a stale 0.7.x build file is
    skipped without being unlinked; routing results are identical.  Real runs
    delete the stale file later, behind the mutation boundary, via
    :func:`_cleanup_stale_build_file`.

    Priority order
    --------------
    1. **Live backup path** (e.g. ``report.json`` for ``--output report.md``)
       when it differs from the default JSON path.  This handles the named-MD
       case where ``_resolve_output_spec`` leaves ``output_json_filename`` as
       ``codedoc.json`` even though the live backup is ``report.json``.
    2. **Final JSON output** (``json_filename``).  Covers the JSON/both case
       and the default-MD case where the live backup IS ``codedoc.json``.
    3. **Stale build file** (``.codedoc_build.json``).  Migration fallback for
       0.7.x interrupted MD runs.  Overlaid only when newer than the JSON.
    4. **Markdown fallback** — same-stem sibling or configured ``md_filename``.

    Returns a dict mapping rel_path → file record dict.
    """
    existing: dict[str, dict] = {}

    # 1. Live backup when it differs from the standard JSON path (named-MD case).
    json_path = output_dir / json_filename
    if live_backup_path and live_backup_path.resolve() != json_path.resolve():
        if live_backup_path.exists():
            try:
                existing = records_by_path(read_codedoc_document(live_backup_path))
                if existing:
                    logger.info(
                        "Loaded %d existing record(s) from live backup '%s'.",
                        len(existing),
                        live_backup_path.name,
                    )
                    return existing
            except (ConfigError, FileNotFoundError) as exc:
                logger.debug(
                    "Optional resume candidate '%s' rejected: %s",
                    live_backup_path.name,
                    exc,
                )

    # 2. Final JSON output (also the live backup for JSON/both and default-MD runs).
    if json_path.exists():
        try:
            existing = records_by_path(read_codedoc_document(json_path))
        except (ConfigError, FileNotFoundError) as exc:
            logger.debug(
                "Optional resume candidate '%s' rejected: %s",
                json_path.name,
                exc,
            )

    # 3. Stale 0.7.x build file migration fallback.
    build_path = output_dir / BUILD_FILENAME
    if build_path.exists():
        try:
            build_is_stale = (
                json_path.exists()
                and build_path.stat().st_mtime < json_path.stat().st_mtime
            )
            if build_is_stale:
                if read_only:
                    logger.info(
                        "Build file '%s' is older than '%s' — treating as stale "
                        "(read-only mode: not removed).",
                        BUILD_FILENAME,
                        json_filename,
                    )
                else:
                    logger.info(
                        "Build file '%s' is older than '%s' — treating as stale and removing it.",
                        BUILD_FILENAME,
                        json_filename,
                    )
                    try:
                        build_path.unlink()
                    except Exception:
                        pass
            else:
                # The reader parses only; age comparison, merge policy, and
                # deletion remain here in the pipeline (legacy stale-build role).
                build_files = records_by_path(
                    read_codedoc_document(build_path, legacy_role="stale_build")
                )
                if build_files:
                    logger.info(
                        "Build file '%s' found (%d record(s)) — merging (migration from 0.7.x).",
                        BUILD_FILENAME,
                        len(build_files),
                    )
                    existing.update(build_files)
        except Exception:
            pass

    if existing:
        return existing

    # 4. Markdown fallback.
    stem_sibling = output_dir / (Path(json_filename).stem + ".md")
    configured_md = output_dir / md_filename
    md_candidates: list[Path] = [stem_sibling]
    if configured_md != stem_sibling:
        md_candidates.append(configured_md)
    # Also check the live backup sibling when named-MD
    if live_backup_path:
        md_sibling = live_backup_path.with_suffix(".md")
        if md_sibling not in md_candidates:
            md_candidates.insert(0, md_sibling)

    for md_path in md_candidates:
        if md_path.exists():
            try:
                return _load_existing_file_docs_from_md(md_path)
            except Exception:
                pass

    return {}


def _load_existing_file_docs_from_md(md_path: Path) -> dict[str, dict]:
    """Parse an existing MD output file into a file-record dict."""
    content = md_path.read_text(encoding="utf-8-sig", errors="replace")

    file_hashes: dict[str, str] = {}
    try:
        meta = read_codedoc_meta(md_path)
        file_hashes = meta.get("file_hashes") or {}
    except ConfigError:
        pass

    view = markdown_to_view(content)
    result: dict[str, dict] = {}
    for file_record in view.get("files", []):
        path = file_record.get("path")
        if path:
            # Prefer the hash from the lightweight metadata comment (authoritative
            # for incremental reuse).  When the embedded view already contains a
            # hash (0.8.1+ Markdown), use that as the fallback so we never
            # overwrite a good hash with an empty string.
            file_hash = file_hashes.get(path) or file_record.get("hash", "")
            result[path] = {**file_record, "hash": file_hash}
    return result


def _public_record_to_doc(file_record: dict) -> dict:
    """Convert a public JSON file record back to a documentation dict."""
    links = file_record.get("links", {})
    deps = file_record.get("_deps") or {
        "external": links.get("external_dependencies", []),
    }
    doc = {
        "file_path": file_record.get("path", ""),
        "language": file_record.get("language", ""),
        "imports": file_record.get("imports", []),
        "description": file_record.get("description", ""),
        "role_in_system": file_record.get("role_in_system", ""),
        "key_concepts": file_record.get("key_concepts", []),
        "functions": file_record.get("functions", []),
        "classes": file_record.get("classes", []),
        "exports": file_record.get("exports", []),
        "usage_example": file_record.get("usage_example", ""),
        "dependencies_analysis": deps,
    }
    cleaned = {k: v for k, v in doc.items() if v not in (None, "", [], {}, {"external": [], "internal": []})}
    # 0.9.3: carry registered private keys from the public record into the
    # reconstructed flat documentation result so they survive resume/reuse.
    carry_private_keys(file_record, cleaned)
    return cleaned


def _build_documentation_records(
    rel_paths: set,
    file_map: dict,
    ordered_paths: list,
    existing_docs: dict,
    new_results: dict,
) -> list[dict]:
    """Build documentation records for write_project_outputs."""
    records = []
    for rel_path in ordered_paths:
        if rel_path not in rel_paths:
            continue

        if rel_path in new_results:
            result = new_results[rel_path]
            if isinstance(result, dict) and result.get("path") and not result.get("file_path"):
                doc = _public_record_to_doc(result)
            else:
                doc = dict(result)
        elif rel_path in existing_docs:
            doc = _public_record_to_doc(existing_docs[rel_path])
        else:
            continue

        if rel_path in new_results:
            descriptor = file_map.get(rel_path, {})
            try:
                file_hash = compute_file_hash(descriptor["path"]) if descriptor.get("path") else ""
            except Exception:
                file_hash = existing_docs.get(rel_path, {}).get("hash", "")
        else:
            file_hash = existing_docs.get(rel_path, {}).get("hash", "")

        records.append({
            "hash": file_hash,
            "file_path": rel_path,
            "language": doc.get("language", ""),
            "documentation": doc,
        })
    return records


def _cleanup_stale_build_file(output_dir: Path, json_filename: str) -> None:
    """Remove a stale 0.7.x build file — mutation-phase counterpart of the
    skip in ``_load_existing_file_docs(read_only=True)``."""
    build_path = output_dir / BUILD_FILENAME
    json_path = output_dir / json_filename
    try:
        if (
            build_path.exists()
            and json_path.exists()
            and build_path.stat().st_mtime < json_path.stat().st_mtime
        ):
            build_path.unlink()
            logger.info(
                "Removed stale build file '%s' (older than '%s').",
                BUILD_FILENAME,
                json_filename,
            )
    except Exception:
        pass


def _remove_legacy_db(output_dir: Path) -> None:
    """Remove codedoc_db.json left over from earlier versions."""
    legacy = output_dir / "codedoc_db.json"
    if legacy.exists():
        try:
            legacy.unlink()
            logger.info("Removed legacy codedoc_db.json (no longer used since 0.6.4)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker wrapper (Work Item 2)
# ---------------------------------------------------------------------------

def _process_and_record(
    descriptor: dict,
    orchestrator: Orchestrator,
    recorder: SafeWriter,
) -> dict:
    """Process one file and record it in the live backup from the worker thread.

    Recording happens here — inside the worker — so a Ctrl-C or crash that
    interrupts the main ``as_completed`` collection loop never discards a
    file whose AI work already completed.

    If ``_process_one_file`` raises for any reason (rate-limit, parse failure,
    model error), the exception propagates out of this function unchanged and
    ``recorder.record()`` is NOT called.  The batch-level code catches the
    future's exception and classifies it as rate-limit or non-rate-limit.
    """
    result = _process_one_file(descriptor, orchestrator)
    recorder.record(
        descriptor["rel_path"],
        result,
        _safe_file_hash(descriptor.get("path")),
    )
    return result


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    project_root: str | Path | dict | None = ".",
    config_overrides: dict | None = None,
) -> dict:
    """Run the full documentation pipeline on a project."""
    if isinstance(project_root, dict) and config_overrides is None:
        config_overrides = project_root
        project_root = "."
    if project_root is None:
        project_root = "."

    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root does not exist: {root}")

    if not ensure_codedoc_installed(root):
        raise RuntimeError("codedoc is not importable in the current Python environment.")

    if config_overrides is None:
        config_overrides = {}

    config = load_config(root, config_overrides)
    _resolve_entry_and_docs(root, config)

    set_level(config.get("log_level", "INFO"))
    logger.info("codedoc starting: root=%s", root)
    output_format = config.get("output_format", "json")
    logger.info("Output format: %s", output_format)

    dry_run = bool(config.get("dry_run", False))
    if dry_run:
        logger.info("Dry run: planning only — no writes, no provider, no LLM calls.")

    output_dir = root / config["output_dir"]

    json_filename = config.get("output_json_filename", "codedoc.json")
    md_filename = config.get("output_md_filename", "codedoc.md")
    live_backup_path = _resolve_live_backup_path(output_dir, output_format, json_filename, md_filename)

    # Read-only ownership inspection (0.9.2).  A real run fails fast before any
    # filesystem side effect, scanning, or LLM call when a final output target
    # is foreign-owned; a dry run records the conflicts and reports them.
    ownership_conflicts = inspect_output_ownership(
        output_dir, output_format, json_filename, md_filename, live_backup_path
    )
    if ownership_conflicts and not dry_run:
        raise ConfigError(ownership_conflicts[0]["message"])

    # 0.8.0: --safe-mode is deprecated; print at most one notice when it was
    # explicitly enabled, then continue normally.
    if config.get("safe_mode", False):
        print(
            "WARNING: --safe-mode is now the default behaviour in 0.8.0 and this "
            "flag has no additional effect.  It is kept for backwards compatibility "
            "and will be removed in a future release.",
            flush=True,
        )
        logger.info("--safe-mode flag noted; live backup is always on in 0.8.0.")

    # 0.8.0: error.log lives in the output directory, not the project root.
    # Construction is in-memory only; nothing is written until flush(), which
    # dry-run never calls.
    error_reporter = ErrorReporter(output_dir / "error.log")

    existing_docs = _load_existing_file_docs(
        output_dir, json_filename, md_filename, live_backup_path, read_only=True
    )

    # Build the scanner skip_dirs list.  Start from config["skip_dirs"] (already
    # resolved by load_config with _add/_remove applied), then unconditionally
    # append the output directory name so codedoc never scans its own output
    # even when the user removes "codedoc" via --remove-skip-dir.
    _scan_skip_dirs = list(config.get("skip_dirs", []))
    _raw_output_dir = str(config.get("output_dir", "codedoc"))
    if _raw_output_dir not in (".", ""):
        _output_dir_name = Path(_raw_output_dir).name
        if _output_dir_name and _output_dir_name not in _scan_skip_dirs:
            _scan_skip_dirs.append(_output_dir_name)

    all_files = scan_files(
        root,
        extension_language_map=config["extension_language_map"],
        max_file_size_kb=config["max_file_size_kb"],
        skip_dirs=_scan_skip_dirs,
        ignore_paths=config.get("ignore_paths"),
    )
    if not all_files:
        # A2: an explicitly specified entry cannot be honoured if nothing was
        # scanned — fail loudly rather than exit successfully having documented
        # nothing.
        if config.get("entry_file"):
            raise ConfigError(
                f"Entry file '{config['entry_file']}' was requested but no "
                f"supported source files were found in '{root}'. Check the path "
                "and your skip_dirs / ignore_paths / extension settings."
            )
        logger.warning("No supported files found in %s. Done.", root)
        if dry_run:
            return {
                "dry_run": True,
                "scanned": 0,
                "selected": 0,
                "entry_excluded": 0,
                "would_process": 0,
                "would_call_llm_for": 0,
                "unchanged": 0,
                "would_reuse": 0,
                "would_resume": 0,
                "forced": 0,
                "estimated_calls": 0,
                "estimated_input_tokens": 0,
                "estimate_is_lower_bound": True,
                "max_files": int(config.get("max_files", 0) or 0),
                "max_files_exceeded": False,
                "ownership_conflicts": ownership_conflicts,
                "output_dir": str(output_dir),
                "output_files": [],
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        _remove_legacy_db(output_dir)
        return {
            "checked": 0,
            "failed": 0,
            "skipped": 0,
            "output_dir": str(output_dir),
            "live_backup_path": None,
            "error_log": None,
            "issues_recorded": 0,
            "rate_limit_warnings": [],
        }

    graph, file_map = _build_graph(all_files, root, error_reporter)
    selected_rels, entry_rel = _select_files(root, config, graph, file_map)

    # Count of scanned files excluded by entry-reachability selection (A1).
    # Zero when no entry is in effect (selected_rels == all scanned files).
    # Surfaced in stats so the CLI can report it at the run summary.
    entry_excluded = len(file_map) - len(selected_rels)

    # Queue/topological order for the selected file set (all selected, not just agent files).
    ordered_selected = [p for p in graph.topological_order() if p in selected_rels]

    # 0.9.2: normalize forced paths against the project root (raises
    # ConfigError for paths outside the root).
    forced_paths = normalize_force_files(config.get("force_files") or [], root)

    # Migration eligibility: legacy Checkpoint entries may be used only when
    # the live backup contains no records — read-only equivalent of the
    # ``SafeWriter.size == 0`` check.
    live_records = read_existing_records(live_backup_path)
    checkpoint_results: dict[str, dict] = {}
    if not live_records:
        cp = Checkpoint(output_dir, entry_file=entry_rel)
        checkpoint_results = cp.load()
        if checkpoint_results:
            logger.info(
                "Migration: loaded %d entry(s) from legacy .codedoc_progress.json "
                "— they will be migrated into the live backup on first flush.",
                len(checkpoint_results),
            )

    # 0.9.2: one shared plan drives both dry-run and real execution.
    plan, materials = build_pipeline_plan(
        file_map=file_map,
        graph=graph,
        selected_rels=selected_rels,
        entry_rel=entry_rel,
        existing_docs=existing_docs,
        checkpoint_records=checkpoint_results,
        forced_paths=forced_paths,
        config=config,
    )

    if dry_run:
        return _build_dry_run_stats(
            plan, file_map, config, output_dir, ownership_conflicts
        )

    # Paid-file safety cap: enforced after the complete plan exists and before
    # any mutation, writer initialization, or provider creation.
    if plan.max_files_exceeded:
        raise ConfigError(
            f"This run would send {len(plan.agent_rels)} file(s) to the LLM, "
            f"which exceeds the configured max_files limit of {plan.max_files}. "
            "Inspect the plan first with --dry-run, and raise --max-files only "
            "after reviewing it."
        )

    # ------------------------------------------------------------------
    # Mutation boundary — everything below may write to the filesystem.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_db(output_dir)
    _cleanup_stale_build_file(output_dir, json_filename)

    # Always-on live backup writer (Work Item 1).
    recorder = SafeWriter(live_backup_path, output_format, entry_rel, file_map)
    recorder.set_queue_order(ordered_selected)

    # Ownership check + pre-load existing records (raises ConfigError for foreign files).
    recorder.load()

    skipped = len(plan.unchanged_rels)
    if skipped > 0:
        logger.info("Incremental mode: skipping %d unchanged file(s)", skipped)

    # Materialize reuse/resume records exactly as the plan routed them.
    new_results: dict[str, dict] = {}
    for rel_path in plan.identical_reuse_rels:
        source_doc = materials.identical_reuse_docs[rel_path]
        new_results[rel_path] = source_doc
        logger.info(
            "Reusing cached documentation for %s from identical content in %s",
            rel_path,
            source_doc.get("path", "unknown"),
        )
    reused = len(plan.identical_reuse_rels)

    for rel_path in plan.checkpoint_reuse_rels:
        new_results[rel_path] = materials.checkpoint_reuse_docs[rel_path]
        logger.info("Migrating from checkpoint: %s", rel_path)
    resumed = len(plan.checkpoint_reuse_rels)

    agent_rels = set(plan.agent_rels)
    usage = UsageAccumulator()

    rate_limit_warnings: list[dict] = []

    if not agent_rels:
        logger.info("All selected files are up-to-date or reused from cached content.")
        stats: dict = {
            "checked": 0,
            "failed": 0,
            "skipped": skipped,
            "reused": reused,
            "resumed": resumed,
            "entry_excluded": entry_excluded,
            "output_dir": str(output_dir),
            "rate_limit_warnings": rate_limit_warnings,
        }
        output_files = write_project_outputs(
            _build_documentation_records(
                selected_rels,
                file_map,
                graph.topological_order(),
                existing_docs,
                new_results,
            ),
            stats,
            output_dir,
            error_reporter.summary(),
            output_format,
            entry_rel,
            _graph_edges(graph, selected_rels),
            json_filename=json_filename,
            md_filename=md_filename,
        )
        stats["output_files"] = [str(path) for path in output_files if path]
        error_reporter.flush()
        recorder.delete()
        _set_issue_stats(stats, error_reporter, live_backup_path)
        _set_usage_stats(stats, usage, plan, config)
        return stats

    # Build the agent-file queue in topological order.
    queue = ProcessingQueue()
    for rel_path in graph.topological_order():
        if rel_path in agent_rels:
            queue.add(file_map[rel_path])

    # Create LLM provider AFTER the live backup is initialised.
    # initialize_empty() must be called before provider creation so the backup
    # exists even if provider init fails.
    recorder.initialize_empty()

    try:
        llm = create_provider(config)
        logger.info("LLM provider: %s", llm.provider_name)
    except Exception as exc:
        error_reporter.record(exc, context="LLM provider init")
        error_reporter.flush()
        # The exception propagates to the CLI which never sees stats, so print
        # the error.log path here so the user can always find diagnostics.
        if error_reporter.has_issues():
            print(
                f"\n  1 issue recorded. See {error_reporter.log_path.resolve()} for details.",
                file=sys.stderr,
                flush=True,
            )
        raise

    orchestrator = Orchestrator(
        llm,
        parallel=config.get("parallel_agents", True),
        max_content_chars=config.get("max_content_chars", 12000),
        usage=usage,
    )
    stats = {
        "checked": 0,
        "failed": 0,
        "skipped": skipped,
        "reused": reused,
        "resumed": resumed,
        "entry_excluded": entry_excluded,
        "rate_limit_warnings": rate_limit_warnings,
    }

    max_workers = min(config.get("max_parallel_files", 5), len(agent_rels)) or 1
    retry_attempts = config.get("file_retry_attempts", 1)
    max_consecutive_failures = config.get("max_consecutive_failures", 5)
    logger.info(
        "File processing plan: %d file(s), up to %d file(s) in parallel, "
        "provider=%s, retry_attempts=%d, max_consecutive_failures=%d",
        len(agent_rels),
        max_workers,
        llm.provider_name,
        retry_attempts,
        max_consecutive_failures,
    )

    _process_agent_files(
        queue=queue,
        orchestrator=orchestrator,
        stats=stats,
        error_reporter=error_reporter,
        max_workers=max_workers,
        retry_attempts=retry_attempts,
        max_consecutive_failures=max_consecutive_failures,
        new_results=new_results,
        recorder=recorder,
        config=config,
    )

    stats["output_dir"] = str(output_dir)
    output_files = write_project_outputs(
        _build_documentation_records(
            selected_rels,
            file_map,
            graph.topological_order(),
            existing_docs,
            new_results,
        ),
        stats,
        output_dir,
        error_reporter.summary(),
        output_format,
        entry_rel,
        _graph_edges(graph, selected_rels),
        json_filename=json_filename,
        md_filename=md_filename,
    )
    stats["output_files"] = [str(path) for path in output_files if path]
    error_reporter.flush()
    # For MD-only runs, remove the live JSON backup after clean MD conversion.
    recorder.delete()
    _set_issue_stats(stats, error_reporter, live_backup_path)
    _set_usage_stats(stats, usage, plan, config)

    logger.info(
        "Done. checked=%d failed=%d skipped=%d output=%s",
        stats["checked"],
        stats["failed"],
        stats["skipped"],
        output_dir,
    )

    if error_reporter.has_issues():
        logger.info(
            "%d issue(s) recorded. See %s for details.",
            error_reporter.issue_count(),
            error_reporter.log_path,
        )

    return stats


def _set_issue_stats(
    stats: dict,
    error_reporter: ErrorReporter,
    live_backup_path: Path,
) -> None:
    """Populate error/issue stats keys on *stats* in-place."""
    if error_reporter.has_issues():
        stats["error_log"] = str(error_reporter.log_path.resolve())
        stats["issues_recorded"] = error_reporter.issue_count()
    else:
        stats["error_log"] = None
        stats["issues_recorded"] = 0
    if live_backup_path is not None:
        stats["live_backup_path"] = str(live_backup_path.resolve())
    else:
        stats["live_backup_path"] = None


def _set_usage_stats(
    stats: dict,
    usage: UsageAccumulator,
    plan: PipelinePlan,
    config: dict,
) -> None:
    """Populate planned/actual usage keys on *stats* in-place (0.9.2).

    Token figures are character-heuristic estimates, not tokenizer counts.
    """
    stats.update(usage.snapshot())
    stats["planned_calls"] = len(plan.agent_rels) * 3
    stats["planned_files"] = len(plan.agent_rels)
    # Files the plan routed to the LLM that were neither completed nor failed
    # (e.g. a run aborted early by the consecutive-failure health check).
    stats["unattempted_files"] = max(
        0, len(plan.agent_rels) - stats.get("checked", 0) - stats.get("failed", 0)
    )
    # Resolved from config so env/config-enabled partial mode reaches the CLI.
    stats["allow_partial"] = bool(config.get("allow_partial", False))


def _build_dry_run_stats(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    config: dict,
    output_dir: Path,
    ownership_conflicts: list[dict],
) -> dict:
    """Build the read-only dry-run stats dict from the shared plan."""
    return {
        "dry_run": True,
        "scanned": len(plan.scanned_rels),
        "selected": len(plan.selected_rels),
        "entry_excluded": len(plan.scanned_rels - plan.selected_rels),
        "would_process": len(plan.process_rels),
        "would_call_llm_for": len(plan.agent_rels),
        "unchanged": len(plan.unchanged_rels),
        "would_reuse": len(plan.identical_reuse_rels),
        "would_resume": len(plan.checkpoint_reuse_rels),
        "forced": len(plan.forced_rels),
        "estimated_calls": len(plan.agent_rels) * 3,
        "estimated_input_tokens": _estimate_planned_input_tokens(plan, file_map, config),
        "estimate_is_lower_bound": True,
        "max_files": plan.max_files,
        "max_files_exceeded": plan.max_files_exceeded,
        "ownership_conflicts": ownership_conflicts,
        "output_dir": str(output_dir),
        "output_files": [],
    }


def _estimate_planned_input_tokens(
    plan: PipelinePlan,
    file_map: dict[str, dict],
    config: dict,
) -> int:
    """Estimate input tokens for the planned LLM calls — a lower bound.

    The structure and dependency prompts are built from known inputs.  The
    documentation prompt embeds the other agents' responses, which do not
    exist yet, so it is estimated with empty analysis objects.  Uses the same
    centralized truncation helper as real execution so the estimated source
    size matches what would actually be sent.
    """
    from codedoc.agents import dependency_agent, documentation_agent, structure_agent

    max_chars = config.get("max_content_chars", 12000)
    total = 0
    for rel_path in plan.agent_rels:
        descriptor = file_map[rel_path]
        try:
            content = descriptor["path"].read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            content = ""
        try:
            imports = parse_file(descriptor)
        except Exception:
            imports = []
        language = descriptor.get("language", "generic")
        content = truncate_for_llm(content, max_chars)
        prompts = (
            structure_agent.build_prompt(rel_path, content, imports, language),
            dependency_agent.build_prompt(rel_path, content, imports, language),
            documentation_agent.build_prompt(rel_path, content, language, {}, {}),
        )
        for system, prompt in prompts:
            total += estimate_tokens(system) + estimate_tokens(prompt)
    return total


# ---------------------------------------------------------------------------
# Parallel / sequential file processing (Work Items 2 & 3)
# ---------------------------------------------------------------------------

def _process_agent_files(
    queue: ProcessingQueue,
    orchestrator: Orchestrator,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
    recorder: SafeWriter,
    config: dict,
) -> None:
    descriptors: list[dict] = []
    while True:
        descriptor = queue.next()
        if descriptor is None:
            break
        descriptors.append(descriptor)

    if max_workers <= 1 or len(descriptors) <= 1:
        _process_files_sequentially(
            descriptors,
            orchestrator,
            queue,
            stats,
            error_reporter,
            retry_attempts,
            max_consecutive_failures,
            new_results,
            recorder,
            config,
        )
        return

    # Build the parallelism ladder.
    rate_limit_adaptive = config.get("rate_limit_adaptive", True)
    custom_ladder = config.get("parallel_ladder")
    if custom_ladder:
        ladder = list(custom_ladder)
    else:
        ladder = _build_default_ladder(max_workers)

    provider_name = orchestrator.llm.provider_name
    original_max_workers = max_workers
    remaining = list(descriptors)
    respect_retry_after = config.get("respect_retry_after", True)
    retry_after_cap = config.get("retry_after_cap_s", 30)

    # 0.8.1: get the provider-aware rate-limit profile (with config overrides).
    profile = get_rate_limit_profile(provider_name, config)
    event_number = 0  # cumulative step-down event count across all rungs

    for level_index, level in enumerate(ladder):
        if not remaining:
            break

        level = min(level, len(remaining)) or 1
        succeeded, retry_rate_limited, failed_non_rate_limited = _process_descriptor_batch(
            remaining,
            orchestrator,
            queue,
            stats,
            error_reporter,
            max_workers=level,
            max_consecutive_failures=max_consecutive_failures,
            recorder=recorder,
            profile=profile,
        )
        new_results.update(succeeded)

        # Non-rate-limit failures are retried sequentially immediately so errors
        # are clearly diagnosed and stats["failed"] is correctly incremented.
        if failed_non_rate_limited:
            logger.info(
                "Retrying %d non-rate-limit failed file(s) sequentially for clearer diagnostics.",
                len(failed_non_rate_limited),
            )
            _process_files_sequentially(
                failed_non_rate_limited,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                config,
            )

        if not retry_rate_limited or not rate_limit_adaptive:
            # No rate-limited files remain, or adaptive mode is off.
            if retry_rate_limited and not rate_limit_adaptive:
                # Treat remaining rate-limited files as sequential retry.
                remaining_descs = [d for d, _e in retry_rate_limited]
                _process_files_sequentially(
                    remaining_descs,
                    orchestrator,
                    queue,
                    stats,
                    error_reporter,
                    retry_attempts,
                    max_consecutive_failures,
                    new_results,
                    recorder,
                    config,
                )
            break

        # --- Step down to next ladder rung ---
        next_level = ladder[level_index + 1] if level_index + 1 < len(ladder) else 1

        # 0.8.1: unpack descriptors and exceptions from the rate-limited list.
        remaining_descs = [d for d, _e in retry_rate_limited]
        exceptions = [e for _d, e in retry_rate_limited]

        # 0.8.1: compute inter-rung sleep duration.
        retry_afters = [
            ra for ra in (_parse_retry_after(e) for e in exceptions)
            if ra is not None
        ]
        retry_after_s = max(retry_afters) if retry_afters else None

        if respect_retry_after and retry_after_s is not None:
            sleep_s = min(retry_after_s, retry_after_cap)
        elif profile.min_backoff_s > 0:
            sleep_s = min(
                profile.min_backoff_s * (profile.backoff_scale ** level_index),
                retry_after_cap,
            )
        else:
            sleep_s = 0.0

        # 0.8.1: derive error sample and limit type from the first exception.
        error_sample = str(exceptions[0])[:200] if exceptions else ""
        limit_type = _detect_limit_type(error_sample) if error_sample else None

        event_number += 1
        warning = {
            "provider": provider_name,
            "original_max_parallel": original_max_workers,
            "current_level": level,
            "new_level": next_level,
            "retried_count": len(remaining_descs),
            "retry_after_s": retry_after_s,
            "sleep_s": sleep_s,
            "error_sample": error_sample,
            "limit_type": limit_type,
            "event_number": event_number,
            "rung_index": level_index,
        }
        stats.setdefault("rate_limit_warnings", []).append(warning)

        warn_msg = (
            f"[{provider_name}] Rate limit detected - your configured "
            f"max_parallel_files ({original_max_workers}) has been reduced to "
            f"{next_level}. Retrying {len(remaining_descs)} remaining file(s) "
            f"at lower concurrency."
        )
        if sleep_s > 0:
            warn_msg += f" Sleeping {sleep_s:.1f}s before retry."
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        error_reporter.record(RuntimeError(warn_msg), context="rate limit step-down", level="warning")

        # 0.8.1: apply inter-rung backoff sleep before the next ladder level.
        if sleep_s > 0:
            logger.info(
                "Rate-limit backoff: sleeping %.1fs before level %d (rung %d, event %d)",
                sleep_s, next_level, level_index, event_number,
            )
            time.sleep(sleep_s)

        remaining = remaining_descs

        # If we have exhausted the ladder, fall through to sequential.
        if level_index + 1 >= len(ladder):
            if remaining:
                still_limited_msg = (
                    f"[{provider_name}] Still rate-limited at max_parallel_files=1. "
                    f"Processing sequentially. You may want to lower your default "
                    f"max_parallel_files in config."
                )
                print(still_limited_msg, flush=True)
                logger.warning(still_limited_msg)
            _process_files_sequentially(
                remaining,
                orchestrator,
                queue,
                stats,
                error_reporter,
                retry_attempts,
                max_consecutive_failures,
                new_results,
                recorder,
                config,
            )
            break


def _process_descriptor_batch(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    max_workers: int,
    recorder: SafeWriter,
    max_consecutive_failures: int = 5,
    profile: RateLimitProfile | None = None,
) -> tuple[dict[str, dict], list[tuple[dict, Exception]], list[dict]]:
    """Process a batch of descriptors in parallel at *max_workers* concurrency.

    Returns
    -------
    succeeded : dict[str, dict]
        rel_path → result for files that completed without error.  These have
        already been recorded in the live backup by the worker thread.
    retry_rate_limited : list[tuple[dict, Exception]]
        (descriptor, causing_exception) pairs for files that hit a rate-limit
        signal.  The exception is preserved so the caller can parse
        ``Retry-After`` hints and compute appropriate inter-rung backoff.
    failed_non_rate_limited : list[dict]
        Descriptors that failed for non-rate-limit reasons.
    """
    succeeded: dict[str, dict] = {}
    retry_rate_limited: list[tuple[dict, Exception]] = []
    failed_non_rate_limited: list[dict] = []

    consecutive_failures = 0
    health_reported = False
    total = len(descriptors)
    completed = 0

    logger.info(
        "Parallel batch: %d file(s) at max_workers=%d, provider=%s",
        total,
        max_workers,
        orchestrator.llm.provider_name,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map: dict[concurrent.futures.Future, dict] = {
            pool.submit(_process_and_record, descriptor, orchestrator, recorder): descriptor
            for descriptor in descriptors
        }

        for future in concurrent.futures.as_completed(future_map):
            descriptor = future_map[future]
            rel_path = descriptor["rel_path"]
            try:
                result = future.result()
                # Already recorded in the worker — just update new_results and stats.
                succeeded[rel_path] = result
                queue.mark_checked(rel_path)
                stats["checked"] += 1
                consecutive_failures = 0
                completed += 1
                _log_file_progress("OK", rel_path, completed, total)
            except Exception as exc:
                completed += 1
                if _is_rate_limit_error(exc, profile):
                    # Only treat as done if a worker recorded it THIS run (it may
                    # have succeeded before the future was cancelled).  A file
                    # that is only *preloaded* from a prior output (stale) must be
                    # retried, never restored from old documentation (A4).
                    if not recorder.recorded_this_run(rel_path):
                        # 0.8.1: preserve the causing exception alongside the
                        # descriptor so the caller can parse Retry-After hints.
                        retry_rate_limited.append((descriptor, exc))
                        _log_file_progress("RATE-LIMIT", rel_path, completed, total, str(exc))
                    else:
                        # Already recorded — treat as succeeded.  Recover the
                        # real persisted record (A4) so the final output is not
                        # overwritten with an empty placeholder.
                        recovered = recorder.get_record(rel_path)
                        succeeded[rel_path] = (
                            _public_record_to_doc(recovered) if recovered else {}
                        )
                        queue.mark_checked(rel_path)
                        stats["checked"] += 1
                        _log_file_progress("OK(late)", rel_path, completed, total)
                else:
                    failed_non_rate_limited.append(descriptor)
                    consecutive_failures += 1
                    _log_file_progress("RETRY", rel_path, completed, total, str(exc))

                if (
                    consecutive_failures >= max_consecutive_failures
                    and not health_reported
                ):
                    health_reported = True
                    _cancel_pending(future_map)
                    error_reporter.record(
                        RuntimeError(
                            f"Parallel processing saw {consecutive_failures} consecutive "
                            "non-rate-limit file failures. "
                            "Failed files will be retried sequentially."
                        ),
                        context="parallel processing health check",
                        level="warning",
                    )

    return succeeded, retry_rate_limited, failed_non_rate_limited


def _process_files_sequentially(
    descriptors: list[dict],
    orchestrator: Orchestrator,
    queue: ProcessingQueue,
    stats: dict,
    error_reporter: ErrorReporter,
    retry_attempts: int,
    max_consecutive_failures: int,
    new_results: dict,
    recorder: SafeWriter,
    config: dict,
) -> None:
    consecutive_failures = 0
    total = len(descriptors)
    respect_retry_after = config.get("respect_retry_after", True)
    retry_after_cap = config.get("retry_after_cap_s", 30)

    logger.info(
        "Starting sequential documentation: %d file(s), provider=%s",
        total,
        orchestrator.llm.provider_name,
    )

    for index, descriptor in enumerate(descriptors, start=1):
        rel_path = descriptor["rel_path"]
        try:
            result = _process_one_file_with_retries(
                descriptor,
                orchestrator,
                retry_attempts,
                respect_retry_after=respect_retry_after,
                retry_after_cap=retry_after_cap,
            )
            new_results[rel_path] = result
            recorder.record(rel_path, result, _safe_file_hash(descriptor.get("path")))
            queue.mark_checked(rel_path)
            stats["checked"] += 1
            consecutive_failures = 0
            _log_file_progress("OK", rel_path, index, total)
        except (ParseError, OutputError, AgentError) as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, str(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            _log_file_progress("FAIL", rel_path, index, total, str(exc))
        except Exception as exc:
            error_reporter.record(exc, context=rel_path)
            queue.mark_failed(rel_path, str(exc))
            stats["failed"] += 1
            consecutive_failures += 1
            _log_file_progress("FAIL", rel_path, index, total, str(exc))

        if consecutive_failures >= max_consecutive_failures:
            error_reporter.record(
                RuntimeError(
                    "Stopping sequential processing after "
                    f"{consecutive_failures} consecutive file failures. "
                    "Check API credentials, provider availability, model name, "
                    "rate limits, and network connectivity."
                ),
                context="sequential processing health check",
            )
            break


def _process_one_file_with_retries(
    descriptor: dict,
    orchestrator: Orchestrator,
    retry_attempts: int,
    respect_retry_after: bool = True,
    retry_after_cap: int = 30,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retry_attempts + 1):
        try:
            return _process_one_file(descriptor, orchestrator)
        except Exception as exc:
            last_error = exc
            if attempt < retry_attempts:
                # Apply Retry-After sleep for rate-limit errors in sequential mode.
                if respect_retry_after and _is_rate_limit_error(exc):
                    delay = _parse_retry_after(exc)
                    if delay is not None:
                        sleep_s = min(delay, retry_after_cap)
                        logger.info(
                            "Retry-After: sleeping %.1fs before retrying %s",
                            sleep_s,
                            descriptor["rel_path"],
                        )
                        time.sleep(sleep_s)
                logger.info(
                    "Retrying %s (%d/%d): %s",
                    descriptor["rel_path"],
                    attempt + 1,
                    retry_attempts,
                    exc,
                )
    raise last_error or RuntimeError("Unknown processing failure")


def _log_file_progress(
    status: str,
    rel_path: str,
    completed: int,
    total: int,
    detail: str | None = None,
) -> None:
    percent = int((completed / total) * 100) if total else 100
    remaining = max(total - completed, 0)
    message = "[%s] %s | %d/%d complete (%d%%), %d remaining"
    if detail:
        logger.warning(
            message + " | %s",
            status, rel_path, completed, total, percent, remaining, detail,
        )
    else:
        logger.info(message, status, rel_path, completed, total, percent, remaining)


def _process_one_file(descriptor: dict, orchestrator: Orchestrator) -> dict:
    rel_path = descriptor["rel_path"]
    logger.info("[START] %s | provider=%s", rel_path, orchestrator.llm.provider_name)
    file_path: Path = descriptor["path"]
    content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    imports = parse_file(descriptor)
    result = orchestrator.process(descriptor, content, imports)
    errors = _agent_errors(result)
    if errors:
        raise AgentError(orchestrator.__class__.__name__, descriptor["rel_path"], "; ".join(errors))
    return result


def _agent_errors(result: dict) -> list[str]:
    errors = []
    for key in ("structure", "dependencies_analysis", "documentation"):
        value: Any = result.get(key, {})
        if isinstance(value, dict) and value.get("error"):
            agent = value.get("agent", key)
            errors.append(f"{agent}: {value['error']}")
    return errors


def _safe_file_hash(file_path: Path | None) -> str:
    if not file_path:
        return ""
    try:
        return compute_file_hash(file_path)
    except Exception:
        return ""


def _cancel_pending(future_map: dict) -> None:
    for future in future_map:
        if not future.done():
            future.cancel()


# ---------------------------------------------------------------------------
# Config / graph helpers
# ---------------------------------------------------------------------------

def _resolve_entry_and_docs(root: Path, config: dict) -> None:
    """Auto-discover the entry point from a previously generated CodeDoc file."""
    if config.get("entry_file"):
        return

    raw_output = config.get("output_dir", "codedoc")
    p = Path(raw_output)

    if p.suffix.lower() in (".json", ".md"):
        candidate = (root / p) if not p.is_absolute() else p
        candidates: list[Path] = [candidate]
    else:
        out_dir = (root / p) if not p.is_absolute() else p
        json_filename = config.get("output_json_filename", "codedoc.json")
        md_filename = config.get("output_md_filename", "codedoc.md")
        output_format = config.get("output_format", "json")

        json_cand = out_dir / json_filename
        # 0.8.0: also probe the live backup sibling for MD-only format.
        live_backup_cand = _resolve_live_backup_path(out_dir, output_format, json_filename, md_filename)
        md_cand = out_dir / md_filename
        candidates = [json_cand]
        if live_backup_cand.resolve() != json_cand.resolve():
            candidates.append(live_backup_cand)
        if output_format in ("md", "both"):
            candidates.append(md_cand)

    for candidate in candidates:
        if candidate.exists():
            try:
                meta = read_codedoc_meta(candidate)
            except ConfigError:
                continue
            entry = meta.get("entry_file")
            if entry:
                logger.info(
                    "Resuming: entry '%s' read from '%s'", entry, candidate.name
                )
                config["entry_file"] = entry
            return

        if candidate.suffix.lower() == ".json":
            md_sibling = candidate.with_suffix(".md")
            if md_sibling.exists():
                try:
                    meta = read_codedoc_meta(md_sibling)
                    entry = meta.get("entry_file")
                    if entry:
                        logger.info(
                            "Cross-format resume: entry '%s' read from '%s'",
                            entry,
                            md_sibling.name,
                        )
                        config["entry_file"] = entry
                    return
                except ConfigError:
                    pass

    # No existing output was found and no --entry was provided.
    # Leave config["entry_file"] unset so _select_files() can call
    # detect_entry_file() with the configured auto-entry candidates.
    # If auto-detection also finds nothing, the existing "process all
    # supported files" behaviour from detect_entry_file() applies.
    logger.info(
        "No existing CodeDoc output found and no --entry provided. "
        "Auto-detection via detect_entry_file() will be attempted."
    )


def _build_graph(
    all_files: list[dict],
    root: Path,
    error_reporter: ErrorReporter,
) -> tuple[DependencyGraph, dict[str, dict]]:
    graph = DependencyGraph()
    all_rel_paths = {d["rel_path"] for d in all_files}
    file_map = {d["rel_path"]: d for d in all_files}

    for descriptor in all_files:
        graph.add_file(descriptor["rel_path"])
        try:
            imports = parse_file(descriptor)
            for imp in imports:
                resolved = resolve_import(imp, descriptor["rel_path"], all_rel_paths, root)
                if resolved:
                    graph.add_dependency(descriptor["rel_path"], resolved)
        except ParseError as exc:
            error_reporter.record(exc, context=descriptor["rel_path"])

    return graph, file_map


def _select_files(
    root: Path,
    config: dict,
    graph: DependencyGraph,
    file_map: dict[str, dict],
) -> tuple[set[str], str | None]:
    all_rel_paths = set(file_map)
    # A2: distinguish an *explicitly specified* entry (via --entry or read from
    # existing docs) from auto-detection.  An explicit entry that cannot be
    # resolved or is absent from the scanned set must be a hard error — silently
    # falling back to documenting the whole repo turns a typo into an expensive
    # full-repo LLM run.  Auto-detection finding nothing keeps the original
    # "document all files" behaviour.
    explicit_entry = config.get("entry_file")
    entry = detect_entry_file(
        root,
        explicit_entry,
        config.get("auto_entry_candidates"),
    )
    if not entry:
        if explicit_entry:
            raise ConfigError(
                f"Entry file '{explicit_entry}' was not found in '{root}'. "
                "Provide an entry path inside the project root, or remove the "
                "entry setting to document all files."
            )
        return all_rel_paths, None

    # An explicit entry may resolve to a path outside the project root (e.g. an
    # absolute path or one with '..').  relative_to() would raise ValueError;
    # surface it as an actionable ConfigError instead.
    try:
        entry_rel = entry.relative_to(root).as_posix()
    except ValueError:
        if explicit_entry:
            raise ConfigError(
                f"Entry file '{explicit_entry}' resolves outside the project "
                f"root '{root}'. Provide an entry path inside the project."
            )
        return all_rel_paths, None

    if entry_rel not in file_map:
        if explicit_entry:
            raise ConfigError(
                f"Entry file '{entry_rel}' exists but was not in the scanned file "
                "set. It may be excluded by skip_dirs, ignore_paths, an "
                "unsupported extension, or max_file_size_kb. Adjust your "
                "configuration, or remove the entry setting to document all files."
            )
        logger.warning(
            "Entry file '%s' was not found in the scanned file set — "
            "documenting all %d file(s) instead.",
            entry_rel,
            len(all_rel_paths),
        )
        return all_rel_paths, None

    reachable = graph.reachable_dependencies(entry_rel) or {entry_rel}

    # Visibility for the known entry-reachability limitation (A1): files that are
    # not transitively imported from the entry are NOT documented.  Previously
    # this exclusion was silent.  We now warn loudly and list a sample so the
    # omission is never invisible.  The structural fix (how selection should
    # behave) is deferred to 0.10.0; this only surfaces the current behaviour.
    excluded = all_rel_paths - reachable
    if excluded:
        sample = ", ".join(sorted(excluded)[:10])
        more = "" if len(excluded) <= 10 else f" (+{len(excluded) - 10} more)"
        logger.warning(
            "Entry-based selection: %d of %d scanned file(s) are NOT reachable "
            "from entry '%s' and will NOT be documented: %s%s. "
            "Files not imported (transitively) from the entry are skipped — this "
            "is a known limitation. To document every scanned file, run without "
            "--entry. Full reachability handling is planned for 0.10.0.",
            len(excluded),
            len(all_rel_paths),
            entry_rel,
            sample,
            more,
        )
    else:
        logger.info(
            "Entry file: %s (%d reachable project file(s); all scanned files reachable)",
            entry_rel,
            len(reachable),
        )
    return reachable, entry_rel


def _graph_edges(graph: DependencyGraph, selected_rels: set[str]) -> list[dict]:
    edges: list[dict] = []
    for rel_path in sorted(selected_rels):
        for dependency in sorted(graph.dependencies_of(rel_path)):
            if dependency in selected_rels:
                edges.append({"from": rel_path, "to": dependency, "type": "internal_import"})
    return edges
