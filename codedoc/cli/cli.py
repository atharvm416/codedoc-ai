"""
codedoc CLI entry point.

Behaviour notes
---------------
- Removed compatibility flags such as ``--safe-mode`` are rejected with
  migration guidance; crash recovery is always active.
- Issues are reported from bounded in-memory diagnostics; no issue-log path is
  persisted or printed.
- Rate-limit step-down warnings from ``stats["rate_limit_warnings"]`` are
  printed to stdout.
- On interrupt, the dedicated crash-recovery file path attached by the pipeline
  (``KeyboardInterrupt.recovery_path``) is named so the user knows the stable
  output was preserved, together with the ordinary-versus-fresh reuse boundary;
  if no recovery file was confirmed, a truthful generic message is printed.
- When an entry is excluded by reachability, ``stats["entry_excluded"]`` is
  reported in the run summary.

First run:
    codedoc --entry src/main.py              # document from entry; save to codedoc/
    codedoc --entry src/main.py --output docs/report.json

Subsequent runs (entry read from the exact selected output when available):
    codedoc                                  # resumes from codedoc/ folder
    codedoc --output codedoc/codedoc.json    # explicit path to previous output
    codedoc --format md                      # reuse MD, or convert exact JSON sibling
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codedoc.utils.errors import CodeDocError, bounded_exception_summary

_RECOVERY_REUSE_BOUNDARY = (
    "Compatible completed ordinary and split records may be reused, and "
    "compatible current schema-4 split node checkpoints may resume. Forced, "
    "stale, identity-mismatched, legacy, foreign, or unsupported state is "
    "rerun or preserved and blocked according to the documented remedy."
)


def _bounded_reason(exc: BaseException) -> str:
    """Rendering-boundary two-tier check shared by every CLI error branch."""
    return str(exc) if isinstance(exc, CodeDocError) else bounded_exception_summary(exc)


def _bounded_traceback(exc: BaseException) -> str:
    """Render a ``--verbose`` diagnostic trace bounded to CodeDoc frames and
    exception categories only.

    Never renders an exception message, chained-cause text, request/response
    body, prompt, source, credential, or local-variable value -- only each
    frame's file/line/function location (filtered to CodeDoc's own package)
    and bounded exception categories.  This replaces
    ``traceback.print_exc()``, whose default rendering includes the full
    chained exception message text.
    """
    import traceback as _traceback

    import codedoc as _codedoc_pkg

    codedoc_dir = Path(_codedoc_pkg.__file__).resolve().parent
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    lines = [
        "Bounded diagnostic trace (--verbose): CodeDoc frames and exception "
        "categories only."
    ]
    for level, node in enumerate(reversed(chain)):
        indent = "  " * level
        category = (
            type(node).__name__
            if isinstance(node, CodeDocError)
            else bounded_exception_summary(node).split(" (", 1)[0]
        )
        lines.append(f"{indent}[{category}]")
        for frame in _traceback.extract_tb(node.__traceback__):
            try:
                Path(frame.filename).resolve().relative_to(codedoc_dir)
                frame_in_codedoc = True
            except (OSError, ValueError):
                frame_in_codedoc = False
            if frame_in_codedoc:
                lines.append(f"{indent}  {Path(frame.filename).name}:{frame.lineno} in {frame.name}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codedoc",
        description="AI-powered codebase documentation — structured, incremental, LLM-agnostic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # --- First run ---
  codedoc --entry src/main.py                          document from entry; save to codedoc/
  codedoc --entry src/main.py --output ./docs          save to custom directory
  codedoc --entry src/main.py --output docs/api.json   save as a named JSON file
  codedoc --entry src/main.py --format md              write only codedoc.md

  # --- Subsequent runs: entry read from exact selected docs ---
  codedoc                                              resume from codedoc/codedoc.json
  codedoc --output codedoc/codedoc.json                resume from explicit file path
  codedoc --format md                                  reuse MD, or convert exact JSON sibling
  codedoc --format both                                generate JSON + Markdown

  # --- Provider / model overrides ---
  codedoc --provider gemini --entry src/main.py
  codedoc --provider anthropic --model claude-haiku-4-5-20251001 --entry src/main.py
  codedoc --ignore /myenv --entry src/main.py          ignore a project-root path

large-file split execution:
  split supports dry-run planning, paid execution, same-path completed split reuse,
  and node recovery in analysis-mode single. triple plus split is rejected
  during configuration validation before scanning or other side effects.
  Oversized files are divided at local semantic or lexical boundaries with
  complete source coverage and a bounded reduction topology.

  max_content_chars bounds each planned leaf, reducer manifest, and complete
  final manifest. Planning never truncates split source or silently falls back
  to truncate. It reports the first named provider-free capacity reason:
  atom-cap, symbol-cap, unit-cap, chunk-cap, reduction-envelope-cap,
  reduction-fan-in-cap, reduction-depth-cap, or
  final-synthesis-envelope-cap. Exactly compatible same-path completed split
  records are reused with zero calls; current schema-4 checkpoints resume only
  unpaid nodes. Forced, stale, legacy, foreign, or unsupported state is rerun
  or preserved and blocked. Under-threshold files retain ordinary execution.
        """,
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        metavar="PATH",
        help="Path to the project root (default: current directory)",
    )
    parser.add_argument(
        "--entry",
        metavar="FILE",
        default=None,
        help=(
            "Optional entry file relative to the project root. An exact selected "
            "output may supply it; otherwise configured candidates are auto-detected. "
            "If none is found, all scanned files are documented."
        ),
    )
    parser.add_argument(
        "--documentation-scope",
        choices=["entry", "all"],
        default=None,
        help=(
            "Documentation coverage: entry follows files reachable from the entry; "
            "all includes every scanned source file (default: entry)."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic", "gemini"],
        default=None,
        help="API provider: auto, openai, anthropic, or gemini (default: auto)",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help=(
            "Model name to use — e.g. gpt-4o-mini, claude-haiku-4-5-20251001, "
            "gemini-2.5-flash. The provider is auto-detected from this name only "
            "when --provider is auto (the default); an explicit --provider always wins."
        ),
    )
    parser.add_argument(
        "--trust-api-base-url",
        metavar="URL",
        default=None,
        dest="trust_api_base_url",
        help=(
            "Runtime approval for a custom api_base_url configured in "
            "codedoc.config.json — required before codedoc will send your API "
            "key, source, or prompts to it. Must canonicalize to exactly the "
            "configured api_base_url (scheme, host, port, path; no username, "
            "password, query string, or fragment). Never settable through "
            "codedoc.config.json or config_overrides. The "
            "CODEDOC_TRUST_API_BASE_URL environment variable is the other "
            "accepted source; this option wins when both are set."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Output path — a directory (e.g. my_docs) or a specific file "
            "(e.g. docs/report.json or docs/report.md). "
            "Defaults to codedoc/ in the project root. "
            "On subsequent runs, pointing to an existing CodeDoc file resumes "
            "documentation from the entry point stored in that file. "
            "When a file path is given, format is inferred from the extension "
            "and overrides --format. Unsupported extensions stop the run with an error."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["json", "md", "both"],
        default=None,
        help="Output format: json, md, or both (default: json)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Project-relative path to ignore. "
            "Can be passed multiple times: --ignore /myenv --ignore generated"
        ),
    )
    parser.add_argument(
        "--skip-dirs",
        nargs="+",
        metavar="DIR",
        default=None,
        dest="skip_dirs",
        help=(
            "Replace the default skip-dirs list entirely with the given names. "
            "Use --add-skip-dir / --remove-skip-dir to extend or reduce instead."
        ),
    )
    parser.add_argument(
        "--add-skip-dir",
        action="append",
        default=[],
        metavar="DIR",
        dest="add_skip_dirs",
        help=(
            "Add a directory name to the skip list (repeatable). "
            "Example: --add-skip-dir generated"
        ),
    )
    parser.add_argument(
        "--remove-skip-dir",
        action="append",
        default=[],
        metavar="DIR",
        dest="remove_skip_dirs",
        help=(
            "Remove a directory name from the default skip list (repeatable). "
            "Example: --remove-skip-dir codedoc  (allows scanning the package source)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Plan only: report what would be scanned, skipped, reused, and sent "
            "to the LLM — with approximate call/token estimates — without writing "
            "any file or contacting any provider. Works without an API key."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Safety cap on the number of files allowed to make LLM calls. "
            "The run stops with an error before any write or API call when the "
            "plan exceeds N. 0 means unlimited (default: 0)."
        ),
    )
    parser.add_argument(
        "--max-planned-calls",
        type=int,
        default=None,
        metavar="N",
        dest="max_planned_calls",
        help=(
            "Safety cap on initially planned LLM calls, including "
            "prompt-customization reviews and initial documentation calls "
            "(0 = unlimited). Checked before provider creation; retries and "
            "corrections are excluded. (default: 0)"
        ),
    )
    parser.add_argument(
        "--force-files",
        action="append",
        default=[],
        metavar="FILE",
        dest="force_files",
        help=(
            "Project-relative path to reprocess even if unchanged (repeatable): "
            "--force-files src/a.py --force-files src/b.py"
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=False,
        help=(
            "Exit 0 even when some files failed, as long as the run completed "
            "and produced output. Setup, ownership, cap, provider, and write "
            "errors still fail."
        ),
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help=(
            "Disable parallel agent execution within each file. "
            "Only affects 'triple' analysis mode (StructureAgent/DependencyAgent "
            "concurrency); 'single' mode makes one call per file regardless."
        ),
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["single", "triple"],
        default=None,
        dest="analysis_mode",
        help=(
            "Per-file analysis mode: 'single' makes one combined provider call "
            "per file (default); 'triple' runs the three-agent path "
            "(structure + dependency + documentation)."
        ),
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help=(
            "Write a complete editable codedoc.config.json with all public "
            "defaults and editable single/triple instructions, then exit."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --init-config, refresh only prompt_profiles in an existing config.",
    )
    parser.add_argument(
        "--max-parallel-files",
        type=int,
        default=None,
        metavar="N",
        help="Maximum files to process at once (default: 5)",
    )
    parser.add_argument(
        "--truncation-head-ratio",
        type=float,
        default=None,
        metavar="FLOAT",
        dest="truncation_head_ratio",
        help=(
            "Head fraction (0.0–1.0 exclusive) for the head-plus-tail source "
            "truncation split (default: 0.70). Lower values send more of the "
            "file tail to the LLM; raise this for files where definitions live "
            "near the top."
        ),
    )
    parser.add_argument(
        "--provider-request-timeout-s",
        type=str,
        default=None,
        metavar="SECONDS",
        dest="provider_request_timeout_s",
        help=(
            "Per connect/read/write/pool phase transport timeout for a single "
            "provider request, in seconds (1-600 inclusive, default: 120). "
            "Not a wall-clock deadline for the whole call. Must be a plain "
            "ASCII decimal number (no sign, exponent, or digit grouping)."
        ),
    )
    parser.add_argument(
        "--large-file-strategy",
        choices=["truncate", "split"],
        default=None,
        dest="large_file_strategy",
        help=(
            "Oversized readable source handling: 'truncate' keeps the legacy "
            "head/tail behavior (default); 'split' enables provider-free "
            "planning, paid execution, same-path completed split reuse, and current "
            "schema-4 node recovery in single mode."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    from codedoc import __version__

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def _has_resolved_prompt_profile(stats: dict) -> bool:
    """Whether an inline profile source won resolution for this run.

    ``inline`` is the only resolved custom source; ``absent`` means
    developer defaults and prints no profile section.
    """
    return stats.get("prompt_profile_source") == "inline"


def _print_feasibility_advisories(stats: dict | None, *, file=None) -> None:
    """Print bounded, non-blocking prompt feasibility notes when present."""
    if not isinstance(stats, dict):
        return
    notes = stats.get("prompt_customization_feasibility_advisories", ())
    if not notes:
        return
    print("Feasibility advisory (non-blocking):", file=file)
    for note in notes:
        print(f"- {note}", file=file)


def _print_prompt_profile_dry_run(stats: dict) -> None:
    """Print projected profile/review costs without changing no-profile output."""
    if not _has_resolved_prompt_profile(stats):
        return
    documentation = stats.get("documentation_calls_planned", 0)
    review = stats.get("prompt_customization_security_review_calls_planned", 0)
    print("\n  Prompt profile:")
    print(f"    Source                : {stats.get('prompt_profile_source')}")
    print(
        f"    Active                : {'yes' if stats.get('prompt_profile_active') else 'no'}"
    )
    print(
        f"    Affected files        : {stats.get('prompt_profile_affected_files', 0)}"
    )
    print(f"    Documentation calls   : {documentation} planned")
    print(f"    Security-review calls : {review} planned")
    print(f"    Total paid calls      : {documentation + review} planned")
    _print_feasibility_advisories(stats)


def _print_prompt_profile_run(stats: dict) -> None:
    """Print actual profile/review category costs for an ordinary real run."""
    if not _has_resolved_prompt_profile(stats):
        return
    documentation = stats.get("documentation_calls_attempted", 0)
    review = stats.get("prompt_customization_security_review_calls_attempted", 0)
    print("\n  Prompt profile:")
    print(f"    Source                : {stats.get('prompt_profile_source')}")
    print(
        f"    Active                : {'yes' if stats.get('prompt_profile_active') else 'no'}"
    )
    print(
        f"    Affected files        : {stats.get('prompt_profile_affected_files', 0)}"
    )
    print(
        "    Security review       : "
        f"{stats.get('prompt_customization_security_review', 'not-required')} "
        f"({review} attempted, "
        f"{stats.get('prompt_customization_security_review_calls_completed', 0)} completed)"
    )
    print(f"    Documentation calls   : {documentation} attempted")
    print(f"    Total attempted calls : {documentation + review}")
    _print_feasibility_advisories(stats)


def _print_split_observability(
    stats: dict, *, include_synthesis_estimate: bool = False
) -> None:
    """Print split-only counts without changing default truncate output."""
    if stats.get("large_file_strategy") != "split":
        return
    print("\n  Large-file split plan:")
    print(f"    Ordinary files          : {stats.get('split_ordinary_files', 0)}")
    print(f"    Syntax-divided files    : {stats.get('split_syntax_files', 0)}")
    print(f"    Lexical-divided files   : {stats.get('split_lexical_files', 0)}")
    print(f"    Blocked files           : {stats.get('split_blocked_files', 0)}")
    reasons = stats.get("split_blocked_by_reason", {})
    if isinstance(reasons, dict) and reasons:
        print(
            "    Blocked reasons         : "
            + ", ".join(
                f"{reason}={count}" for reason, count in sorted(reasons.items())
            )
        )
    pairs = stats.get("split_blocked_pairs", ())
    if isinstance(pairs, (list, tuple)) and pairs:
        print("    Blocked path/reason pairs:")
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                print(f"      {pair[0]} ({pair[1]})")
    print(
        f"    Semantic units / chunks : {stats.get('split_units', 0)} / "
        f"{stats.get('split_chunks', 0)}"
    )
    print(f"    Continuation groups     : {stats.get('split_continuation_groups', 0)}")
    print(
        "    Unit-consolidation levels/calls: "
        f"{stats.get('split_unit_consolidation_levels', 0)} / "
        f"{stats.get('split_unit_consolidation_calls_planned', 0)}"
    )
    print(
        "    General reduction levels/calls : "
        f"{stats.get('split_general_reduction_levels', 0)} / "
        f"{stats.get('split_general_reduction_calls_planned', 0)}"
    )
    print(
        "    Final synthesis calls   : "
        f"{stats.get('split_final_synthesis_calls_planned', 0)}"
    )
    print(
        "    Completed reused / partial resumed: "
        f"{stats.get('split_completed_files_reused', 0)} / "
        f"{stats.get('split_partial_files_resumed', 0)}"
    )
    print(
        "    Unpaid/reexecuted/quarantined nodes: "
        f"{stats.get('split_unpaid_nodes', 0)} / "
        f"{stats.get('split_reexecuted_nodes', 0)} / "
        f"{stats.get('split_quarantined_nodes', 0)}"
    )
    print(
        "    Recovery conflict files : "
        f"{stats.get('split_recovery_conflict_files', 0)}"
    )
    print(
        "    Planned call categories: "
        f"{stats.get('file_documentation_calls_planned', 0)} file / "
        f"{stats.get('unit_documentation_calls_planned', 0)} leaf / "
        f"{stats.get('file_reduction_calls_planned', 0)} reduction / "
        f"{stats.get('synthesis_calls_planned', 0)} synthesis"
    )
    if include_synthesis_estimate and stats.get("split_synthesis_input_estimate"):
        print(
            "    Synthesis input estimate: "
            f"{stats['split_synthesis_input_estimate']} from configured ceiling"
        )
    advisory = stats.get("split_complexity_advisory")
    if include_synthesis_estimate and advisory:
        print(f"\n  Advisory (non-blocking): {advisory}")


def _print_dry_run_summary(stats: dict) -> None:
    """Print the planning summary for a --dry-run invocation."""
    print("\ncodedoc dry run — no files were written, no provider was contacted.")
    print(f"  Files scanned          : {stats.get('scanned', 0)}")
    print(f"  Files selected         : {stats.get('selected', 0)}")
    if stats.get("files_skipped_large", 0):
        print(f"  Files skipped (too large): {stats['files_skipped_large']}")
    if stats.get("files_skipped_unreadable", 0):
        print(f"  Files skipped (unreadable): {stats['files_skipped_unreadable']}")
    scope = stats.get("documentation_scope", "entry")
    print(f"  Documentation scope    : {scope}")
    print(f"  Analysis mode          : {stats.get('analysis_mode', 'single')}")
    if stats.get("large_file_strategy") == "split":
        print(
            "  Calls per under-threshold file: "
            f"{stats.get('initial_calls_per_file', 1)}"
        )
    else:
        print(f"  Initial calls per file : {stats.get('initial_calls_per_file', 1)}")
    print(f"  Entry reachable        : {stats.get('entry_reachable', 0)}")
    print(f"  Entry disconnected     : {stats.get('entry_disconnected', 0)}")
    # Derive the excluded count from the clearer reachable/disconnected
    # counts.  Under scope 'entry' the disconnected files are exactly the ones
    # excluded from documentation; scope 'all' documents them so none are
    # excluded.  The compatibility ``entry_excluded`` stat is still returned.
    disconnected = stats.get("entry_disconnected", 0)
    excluded = disconnected if scope == "entry" else 0
    if excluded:
        print(
            f"  Files excluded         : {excluded} disconnected file(s); "
            "use --documentation-scope all for complete coverage"
        )
    elif disconnected:
        print("  Disconnected status    : included by documentation_scope='all'")
    print(f"  Would process          : {stats.get('would_process', 0)}")
    print(f"  Unchanged (skipped)    : {stats.get('unchanged', 0)}")
    print(f"  Would reuse (identical): {stats.get('would_reuse', 0)}")
    if stats.get("would_resume", 0):
        print(f"  Would resume           : {stats['would_resume']}")
    if stats.get("forced", 0):
        print(f"  Forced                 : {stats['forced']}")
    if stats.get("would_skip_insufficient_source", 0):
        print(
            "  Would skip (insufficient): "
            f"{stats.get('would_skip_insufficient_source', 0)} file(s) — "
            "empty or whitespace-only; no documentation call."
        )
    print(f"  Would call LLM for     : {stats.get('would_call_llm_for', 0)} file(s)")
    print(f"  Estimated LLM calls    : {stats.get('estimated_calls', 0)}")
    correction_enabled = stats.get("response_correction_enabled", False)
    if correction_enabled:
        print(
            f"  Response correction    : enabled "
            f"(up to {stats.get('response_correction_calls_possible_max', 0)} extra call(s))"
        )
        print(
            f"  Worst-case LLM calls   : {stats.get('estimated_calls_max_with_correction', 0)} "
            "(baseline + one correction per documentation call; worst case, not an "
            "expected charge)"
        )
    else:
        print("  Response correction    : disabled (0 possible extra calls)")
    if stats.get("disconnected_paid_files", 0):
        print(
            "  Disconnected candidates: "
            f"{stats['disconnected_paid_files']} "
            f"({stats.get('disconnected_planned_calls', 0)} planned initial calls)"
        )
    if stats.get("large_file_strategy") == "split":
        estimate_qualifier = (
            "approximate mixed bound; synthesis uses the configured ceiling"
        )
    else:
        estimate_qualifier = (
            "approximate lower bound"
            if stats.get("estimate_is_lower_bound", False)
            else "approximate"
        )
    print(
        f"  Estimated input tokens : ~{stats.get('estimated_input_tokens', 0)} "
        f"({estimate_qualifier} — character heuristic, not a tokenizer)"
    )
    print(f"  Output directory       : {stats.get('output_dir', '')}")
    _print_split_observability(stats, include_synthesis_estimate=True)
    _print_prompt_profile_dry_run(stats)

    if stats.get("max_files_exceeded"):
        print(
            "\n  WARNING: the plan "
            f"({stats.get('max_files_candidate_files', 0)} "
            "documentation-call candidate(s)) exceeds "
            f"--max-files {stats.get('max_files', 0)}; "
            f"{stats.get('would_call_llm_for', 0)} would actually be sent after "
            "source gating. "
            "The corresponding real run would stop with exit code 2 before "
            "writing anything or calling any provider."
        )

    if stats.get("max_planned_calls_exceeded"):
        if stats.get("large_file_strategy") == "split":
            category_detail = (
                f"{stats.get('file_documentation_calls_planned', 0)} "
                "file documentation, "
                f"{stats.get('unit_documentation_calls_planned', 0)} "
                "leaf documentation, "
                f"{stats.get('file_reduction_calls_planned', 0)} "
                "file reduction, "
                f"{stats.get('synthesis_calls_planned', 0)} file synthesis"
            )
        else:
            category_detail = (
                f"{stats.get('documentation_calls_planned', 0)} "
                "file documentation"
            )
        print(
            "\n  WARNING: the plan has "
            f"{stats.get('total_calls_planned', 0)} initially planned LLM call(s) "
            f"({stats.get('prompt_customization_security_review_calls_planned', 0)} "
            "prompt-customization review, "
            f"{category_detail}), "
            f"exceeding --max-planned-calls {stats.get('max_planned_calls', 0)}. "
            "The corresponding real run would stop with exit code 2 before "
            "writing anything or calling any provider."
        )

    conflicts = stats.get("ownership_conflicts") or []
    if conflicts:
        print(f"\n  WARNING: {len(conflicts)} output ownership conflict(s) found:")
        for conflict in conflicts:
            print(f"    - {conflict.get('path', '')}")
        print(
            "  The corresponding real run would stop with exit code 2 before "
            "writing anything."
        )


def _print_run_summary(stats: dict) -> None:
    """Print the completion summary for a real run."""
    print("\ncodedoc complete.")
    print(f"  Files documented by LLM       : {stats['checked']}")
    print(f"  Files reused (unchanged)      : {stats.get('skipped', 0)}")
    print(f"  Files reused (identical content): {stats.get('reused', 0)}")
    if stats.get("skipped_insufficient_source", 0):
        print(
            "Skipped (insufficient source): "
            f"{stats.get('skipped_insufficient_source', 0)} file(s) — "
            "empty or whitespace-only, not sent to the provider."
        )
    if stats.get("resumed", 0):
        print(f"  Files resumed from recovery   : {stats['resumed']}")
    if stats.get("files_skipped_large", 0):
        print(f"  Files skipped (too large)     : {stats['files_skipped_large']}")
    if stats.get("files_skipped_unreadable", 0):
        print(f"  Files skipped (unreadable)    : {stats['files_skipped_unreadable']}")
    print(f"  Files failed     : {stats['failed']}")
    scope = stats.get("documentation_scope", "entry")
    print(f"  Scope            : {scope}")
    print(f"  Analysis mode    : {stats.get('analysis_mode', 'single')}")
    if stats.get("large_file_strategy") == "split":
        print(
            "  Calls/under-threshold file: "
            f"{stats.get('initial_calls_per_file', 1)}"
        )
    else:
        print(f"  Initial calls/file: {stats.get('initial_calls_per_file', 1)}")
    print(f"  Entry reachable  : {stats.get('entry_reachable', 0)}")
    print(f"  Disconnected     : {stats.get('entry_disconnected', 0)}")
    # Derive the excluded count from the clearer reachable/disconnected
    # counts (see _print_dry_run_summary).  ``entry_excluded`` is still returned
    # in stats for compatibility.
    disconnected = stats.get("entry_disconnected", 0)
    excluded = disconnected if scope == "entry" else 0
    if excluded:
        print(
            f"  Files excluded   : {excluded} disconnected file(s) under "
            "documentation_scope='entry'; use --documentation-scope all for "
            "complete coverage."
        )
    elif disconnected:
        print("  Disconnected set : included by documentation_scope='all'")
    if stats.get("disconnected_paid_files", 0):
        print(
            "  Disconnected candidates: "
            f"{stats['disconnected_paid_files']} file(s), "
            f"{stats.get('disconnected_planned_calls', 0)} planned initial calls"
        )
    print(f"  Output directory : {stats['output_dir']}")
    for output_file in stats.get("output_files", []):
        print(f"  Output file      : {output_file}")
    _print_split_observability(stats)

    # Approximate usage accounting — only when LLM work was planned.
    if stats.get("planned_calls", 0) or stats.get("attempted_calls", 0):
        initially_planned = stats.get(
            "total_calls_planned", stats.get("planned_calls", 0)
        )
        print(
            f"  LLM calls        : {stats.get('attempted_calls', 0)} attempted "
            f"({stats.get('successful_calls', 0)} ok, "
            f"{stats.get('failed_calls', 0)} failed; "
            f"{initially_planned} initially planned)"
        )
        print(
            "  Logical calls    : "
            f"{stats.get('attempted_logical_calls', 0)} attempted, "
            f"{stats.get('planned_calls_not_attempted', 0)} planned-not-attempted; "
            f"{stats.get('additional_attempts', 0)} additional attempt(s)"
        )
        print(
            f"  Tokens (approx.) : ~{stats.get('estimated_input_tokens', 0)} in / "
            f"~{stats.get('estimated_output_tokens', 0)} out "
            "(character estimate, not a tokenizer)"
        )
    # Correction summary: one line only when a targeted correction was attempted.
    corrections = stats.get("response_correction_calls_attempted", 0)
    if corrections > 0:
        print(
            f"  Response corrections: {corrections} attempted "
            f"({stats.get('response_correction_calls_succeeded', 0)} succeeded, "
            f"{stats.get('response_correction_calls_failed', 0)} failed)"
        )
    _print_prompt_profile_run(stats)

    # Compact rate-limit summary — only shown when events occurred.
    # Per-event messages were already printed in real time during the run.
    rate_limit_warnings = stats.get("rate_limit_warnings", [])
    if rate_limit_warnings:
        event_count = len(rate_limit_warnings)
        providers = sorted({w["provider"] for w in rate_limit_warnings})
        total_sleep = sum(w.get("sleep_s", 0) or 0 for w in rate_limit_warnings)
        sleep_note = f", {total_sleep:.1f}s total backoff" if total_sleep > 0 else ""
        print(
            f"\n  Rate limits: {event_count} step-down event(s) "
            f"[{', '.join(providers)}]{sleep_note}."
        )

    # Note how many issues were recorded (details were printed during the run;
    # hard-error summaries are embedded in the final output).
    issues = stats.get("issues_recorded", 0)
    if issues:
        failed = stats.get("failed", 0)
        if failed > 0:
            print(f"\n  {failed} file(s) failed; {issues} issue(s) recorded.")
        else:
            print(f"\n  {issues} issue(s) recorded (all recovered).")


def _confirm_risky_prompt_customization(_warnings: tuple[str, ...]) -> bool:
    """Ask for explicit medium-risk consent only on an interactive terminal."""
    if not sys.stdin.isatty():
        return False
    try:
        answer = input("Proceed with this medium-risk customization? [y/N] ")
    except (EOFError, OSError):
        return False
    return answer.strip().casefold() in {"y", "yes"}


def run_cli(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Exit-code contract:
      0   — complete success, dry-run success, or --allow-partial
      1   — file-processing failures, output/write failure, unexpected fatal error
      2   — invalid path/input/config, ownership conflict, cap exceeded,
            or provider initialization failure
      130 — keyboard interrupt
    """
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if argv and argv[0] in {"run", "execute"}:
        argv = argv[1:]

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse uses SystemExit for both help/version (0) and invalid input
        # (2). Keep help/version behavior, but make invalid input follow the
        # testable run_cli() integer-return contract.
        if exc.code == 0:
            raise
        return int(exc.code or 2)

    if args.init_config:
        # Initialization operates on the current working directory and accepts
        # no documentation-run option or positional project path.
        unrelated = any(
            (
                args.root != ".",
                args.entry is not None,
                args.documentation_scope is not None,
                args.provider is not None,
                args.model is not None,
                args.trust_api_base_url is not None,
                args.output is not None,
                args.format is not None,
                bool(args.ignore),
                args.skip_dirs is not None,
                bool(args.add_skip_dirs),
                bool(args.remove_skip_dirs),
                args.dry_run,
                args.max_files is not None,
                args.max_planned_calls is not None,
                bool(args.force_files),
                args.allow_partial,
                args.no_parallel,
                args.analysis_mode is not None,
                args.large_file_strategy is not None,
                args.max_parallel_files is not None,
                args.truncation_head_ratio is not None,
                args.provider_request_timeout_s is not None,
                args.verbose,
            )
        )
        if unrelated:
            print(
                "Error: --init-config can be combined only with --force; project "
                "paths and documentation-run options are not accepted.",
                file=sys.stderr,
            )
            return 2
        try:
            from codedoc.core.config_template import init_config

            result = init_config(Path.cwd(), args.force)
            print(result.message)
            return 0
        except SystemExit as exc:
            return int(exc.code or 2)
        except Exception as exc:
            print(f"Error: {_bounded_reason(exc)}", file=sys.stderr)
            return 2

    if args.force:
        print(
            "Error: --force requires --init-config.",
            file=sys.stderr,
        )
        return 2

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: project root is not a directory: {root}", file=sys.stderr)
        return 2

    overrides: dict = {}
    if args.entry:
        overrides["entry_file"] = args.entry
    if args.documentation_scope is not None:
        overrides["documentation_scope"] = args.documentation_scope
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.model:
        overrides["model_name"] = args.model
    if args.output:
        overrides["output_dir"] = args.output
    if args.format:
        overrides["output_format"] = args.format
    if args.ignore:
        overrides["ignore_paths"] = args.ignore
    if args.skip_dirs is not None:
        overrides["skip_dirs"] = args.skip_dirs
    if args.add_skip_dirs:
        overrides["skip_dirs_add"] = args.add_skip_dirs
    if args.remove_skip_dirs:
        overrides["skip_dirs_remove"] = args.remove_skip_dirs
    if args.no_parallel:
        overrides["parallel_agents"] = False
    if args.analysis_mode is not None:
        overrides["analysis_mode"] = args.analysis_mode
    if args.large_file_strategy is not None:
        overrides["large_file_strategy"] = args.large_file_strategy
    if args.max_parallel_files is not None:
        overrides["max_parallel_files"] = args.max_parallel_files
    if args.truncation_head_ratio is not None:
        overrides["truncation_head_ratio"] = args.truncation_head_ratio
    if args.provider_request_timeout_s is not None:
        overrides["provider_request_timeout_s"] = args.provider_request_timeout_s
    if args.verbose:
        overrides["log_level"] = "DEBUG"
    if args.dry_run:
        overrides["dry_run"] = True
    if args.max_files is not None:
        overrides["max_files"] = args.max_files
    if args.max_planned_calls is not None:
        overrides["max_planned_calls"] = args.max_planned_calls
    if args.force_files:
        overrides["force_files"] = args.force_files
    if args.allow_partial:
        overrides["allow_partial"] = True

    try:
        from codedoc.pipeline import run_pipeline

        stats = run_pipeline(
            root,
            config_overrides=overrides,
            confirm_risky=_confirm_risky_prompt_customization,
            trust_api_base_url=args.trust_api_base_url,
        )

        if stats.get("dry_run"):
            _print_dry_run_summary(stats)
            return 0

        _print_run_summary(stats)

        failed = stats.get("failed", 0)
        if failed > 0:
            # --allow-partial may also be enabled via config/env; the pipeline
            # surfaces the resolved value in stats.
            if args.allow_partial or stats.get("allow_partial"):
                unattempted = stats.get("unattempted_files", 0)
                never_attempted_note = (
                    f" and {unattempted} file(s) were never attempted "
                    "(run aborted early by the failure health check)"
                    if unattempted
                    else ""
                )
                print(
                    f"\nWARNING: output is INCOMPLETE — {failed} file(s) "
                    f"failed{never_attempted_note}. Exiting 0 because "
                    "--allow-partial is enabled.",
                    flush=True,
                )
                return 0
            return 1
        return 0

    except FileNotFoundError as exc:
        print(f"Error: {_bounded_reason(exc)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt as exc:
        # The pipeline attaches the exact selected crash-recovery path to
        # the interrupt as ``recovery_path`` only when that file exists on disk.
        # The stable output is never touched mid-run, so it is always preserved;
        # we report the recovery path and the release-specific reuse boundary.
        recovery_path = getattr(exc, "recovery_path", None)
        if recovery_path:
            print(
                "\nRun interrupted. Your previous stable output was left untouched. "
                "Files completed before the interrupt are saved in the crash-recovery "
                f"file:\n  {recovery_path}\n"
                f"Re-run the same command. {_RECOVERY_REUSE_BOUNDARY}",
                file=sys.stderr,
            )
        else:
            print(
                "\nRun interrupted before any crash-recovery file was created or "
                "confirmed. Your previous stable output (if any) was left untouched. "
                f"Re-run the same command to start. {_RECOVERY_REUSE_BOUNDARY}",
                file=sys.stderr,
            )
        return 130
    except Exception as exc:
        from codedoc.utils.errors import (
            ConfigError,
            OutputError,
            UnrecoverableProviderError,
        )

        if isinstance(exc, UnrecoverableProviderError):
            # A doomed-run safe stop — not an unexpected crash. Completed
            # file-level results are in recovery, subject to the release-specific
            # reuse boundary below. A *terminal*
            # abort (billing/credentials/model/access) is a setup/credentials
            # class problem → exit 2 (consistent
            # with ConfigError/ProviderInitError).  A *bounded rate-limit / quota*
            # stop is a transient "retry later" condition, not a credentials
            # fault → exit 1 so automation does not read it as "fix credentials".
            print(f"Error: {exc}", file=sys.stderr)
            print(
                "\nCompleted files are saved in crash_recovery.json in your output "
                f"directory. Re-run the same command. {_RECOVERY_REUSE_BOUNDARY}",
                file=sys.stderr,
            )
            if args.verbose:
                print(_bounded_traceback(exc), file=sys.stderr)
            return 2 if getattr(exc, "category", None) == "terminal" else 1
        if isinstance(exc, ConfigError):
            # Includes ProviderInitError (provider initialization failures),
            # ownership conflicts, the max_files cap, and the prompt-customization
            # fail-closed / TOO_RISKY review errors.  A fail-closed review carries
            # bounded numeric attempt statistics (never profile text) before the
            # ordinary setup error, so the paid cost of the aborted review is
            # visible.
            err_stats = getattr(exc, "stats", None)
            _print_feasibility_advisories(err_stats, file=sys.stderr)
            if isinstance(err_stats, dict) and (
                "prompt_customization_security_review_calls_attempted" in err_stats
            ):
                print(
                    "  Paid calls before stop: review "
                    f"{err_stats.get('prompt_customization_security_review_calls_attempted', 0)} "
                    f"attempted/"
                    f"{err_stats.get('prompt_customization_security_review_calls_completed', 0)} "
                    "completed, documentation 0.",
                    file=sys.stderr,
                )
            print(f"Error: {exc}", file=sys.stderr)
            if args.verbose:
                print(_bounded_traceback(exc), file=sys.stderr)
            return 2
        if isinstance(exc, OutputError):
            # The OutputError message already carries the sanitized OS
            # cause/category and the affected path.  Add concrete next-step
            # guidance keyed on whether this was a transient lock or a persistent
            # accessibility failure.  CodeDoc never names the locking process
            # unless the OS supplied it in the message above.
            from codedoc.core.io_diagnostics import classify_os_error

            print(f"Error: {exc}", file=sys.stderr)
            category = classify_os_error(exc)
            if category == "locked":
                print(
                    "\nThis looks like a transient file lock. Any crash-recovery "
                    "file already created by this run remains preserved; a lock "
                    "can also occur before one is created. Close any program that "
                    "may be viewing the file and rerun the same command to "
                    f"continue. {_RECOVERY_REUSE_BOUNDARY} CodeDoc cannot identify "
                    "the locking process unless the "
                    "operating system reports it.",
                    file=sys.stderr,
                )
            else:
                print(
                    "\nChoose a writable output directory or correct local "
                    "permissions, then rerun the same command. Any crash-recovery "
                    "file already created remains preserved; the failure can also "
                    f"occur before one exists. {_RECOVERY_REUSE_BOUNDARY}",
                    file=sys.stderr,
                )
            if args.verbose:
                print(_bounded_traceback(exc), file=sys.stderr)
            return 1
        print(f"Fatal error: {_bounded_reason(exc)}", file=sys.stderr)
        if args.verbose:
            print(_bounded_traceback(exc), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point: exit nonzero via SystemExit, return on success."""
    code = run_cli(argv)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
