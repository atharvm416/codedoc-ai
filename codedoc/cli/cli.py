"""
codedoc CLI entry point.

Behaviour notes
---------------
- ``--safe-mode`` is deprecated (kept for backwards compatibility; prints a
  no-op warning at runtime).
- The error / issue log path is printed whenever any issue is recorded, not
  only when ``failed > 0``.
- Rate-limit step-down warnings from ``stats["rate_limit_warnings"]`` are
  printed to stdout.
- On interrupt, the dedicated crash-recovery file path attached by the pipeline
  (``KeyboardInterrupt.recovery_path``) is named so the user knows the stable
  output was preserved and which file enables resume; if no recovery file was
  confirmed, a truthful generic message is printed instead.
- When an entry is excluded by reachability, ``stats["entry_excluded"]`` is
  reported in the run summary.

First run:
    codedoc run --entry src/main.py              # document from entry; save to codedoc/
    codedoc run --entry src/main.py --output docs/report.json

Subsequent runs (entry auto-read from previous docs when available):
    codedoc run                                  # resumes from codedoc/ folder
    codedoc run --output codedoc/codedoc.json    # explicit path to previous output
    codedoc run --format md                      # convert existing JSON to Markdown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codedoc",
        description="AI-powered codebase documentation — structured, incremental, LLM-agnostic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # --- First run ---
  codedoc run --entry src/main.py                          document from entry; save to codedoc/
  codedoc run --entry src/main.py --output ./docs          save to custom directory
  codedoc run --entry src/main.py --output docs/api.json   save as a named JSON file
  codedoc run --entry src/main.py --format md              write only codedoc.md

  # --- Subsequent runs: entry read from existing docs ---
  codedoc run                                              resume from codedoc/ (auto-detected)
  codedoc run --output codedoc/codedoc.json                resume from explicit file path
  codedoc run --format md                                  convert cached JSON to Markdown
  codedoc run --format both                                generate JSON + Markdown

  # --- Provider / model overrides ---
  codedoc run --provider gemini --entry src/main.py
  codedoc run --provider anthropic --model claude-haiku-4-5-20251001 --entry src/main.py
  codedoc run --ignore /myenv --entry src/main.py          ignore a project-root path
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
            "Entry file relative to project root (e.g. src/main.py). "
            "Required for the first run. On subsequent runs the entry point is "
            "read automatically from the previously generated documentation file."
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
    ignore_group = parser.add_mutually_exclusive_group()
    ignore_group.add_argument(
        "--manage-output-gitignore",
        dest="manage_output_gitignore",
        action="store_true",
        default=None,
        help="Manage a codedoc-owned block in the output directory .gitignore.",
    )
    ignore_group.add_argument(
        "--no-manage-output-gitignore",
        dest="manage_output_gitignore",
        action="store_false",
        help="Disable managed output .gitignore updates.",
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
            "gemini-2.5-flash. When set, provider is auto-detected from the model name."
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
    # [DEPRECATED] Live JSON backup is always on since 0.8.0.  The flag is
    # still accepted for backwards compatibility but hidden from --help
    # (0.9.2); the pipeline prints one compatibility warning when enabled.
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
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
        "--max-parallel-files",
        type=int,
        default=None,
        metavar="N",
        help="Maximum files to process at once (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v",
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


def _print_ignore_status(stats: dict, dry_run: bool) -> None:
    if not stats.get("output_gitignore_enabled", False):
        return
    if stats.get("output_gitignore_warning"):
        print(f"  Managed ignore warning: {stats['output_gitignore_warning']}")
    elif stats.get("output_gitignore_updated"):
        print(f"  Managed ignore updated: {stats.get('output_gitignore_path', '')}")
    elif dry_run:
        print("  Managed ignore         : enabled (dry run; no change)")
    else:
        print("  Managed ignore         : enabled; no eligible artifact change")


def _print_dry_run_summary(stats: dict) -> None:
    """Print the planning summary for a --dry-run invocation."""
    print("\ncodedoc dry run — no files were written, no provider was contacted.")
    print(f"  Files scanned          : {stats.get('scanned', 0)}")
    print(f"  Files selected         : {stats.get('selected', 0)}")
    scope = stats.get("documentation_scope", "entry")
    print(f"  Documentation scope    : {scope}")
    print(f"  Analysis mode          : {stats.get('analysis_mode', 'single')}")
    print(
        "  Initial calls per file : "
        f"{stats.get('initial_calls_per_file', 1)}"
    )
    print(f"  Entry reachable        : {stats.get('entry_reachable', 0)}")
    print(f"  Entry disconnected     : {stats.get('entry_disconnected', 0)}")
    # 0.10.0: derive the excluded count from the clearer reachable/disconnected
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
    print(f"  Would call LLM for     : {stats.get('would_call_llm_for', 0)} file(s)")
    print(f"  Estimated LLM calls    : {stats.get('estimated_calls', 0)}")
    if stats.get("disconnected_paid_files", 0):
        print(
            "  Disconnected paid files: "
            f"{stats['disconnected_paid_files']} "
            f"({stats.get('disconnected_planned_calls', 0)} planned initial calls)"
        )
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
    _print_ignore_status(stats, dry_run=True)

    if stats.get("max_files_exceeded"):
        print(
            f"\n  WARNING: the plan ({stats.get('would_call_llm_for', 0)} paid "
            f"file(s)) exceeds --max-files {stats.get('max_files', 0)}. "
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
    print(f"  Files documented : {stats['checked']}")
    print(f"  Files reused     : {stats.get('reused', 0)}")
    if stats.get("resumed", 0):
        print(f"  Files resumed    : {stats['resumed']}")
    print(f"  Files failed     : {stats['failed']}")
    scope = stats.get("documentation_scope", "entry")
    print(f"  Scope            : {scope}")
    print(f"  Analysis mode    : {stats.get('analysis_mode', 'single')}")
    print(
        "  Initial calls/file: "
        f"{stats.get('initial_calls_per_file', 1)}"
    )
    print(f"  Entry reachable  : {stats.get('entry_reachable', 0)}")
    print(f"  Disconnected     : {stats.get('entry_disconnected', 0)}")
    # 0.10.0: derive the excluded count from the clearer reachable/disconnected
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
            f"  Disconnected paid: {stats['disconnected_paid_files']} file(s), "
            f"{stats.get('disconnected_planned_calls', 0)} planned initial calls"
        )
    print(f"  Output directory : {stats['output_dir']}")
    for output_file in stats.get("output_files", []):
        print(f"  Output file      : {output_file}")
    _print_ignore_status(stats, dry_run=False)

    # 0.9.2: approximate usage accounting — only when LLM work was planned.
    if stats.get("planned_calls", 0) or stats.get("attempted_calls", 0):
        print(
            f"  LLM calls        : {stats.get('attempted_calls', 0)} attempted "
            f"({stats.get('successful_calls', 0)} ok, "
            f"{stats.get('failed_calls', 0)} failed; "
            f"{stats.get('planned_calls', 0)} planned)"
        )
        print(
            f"  Tokens (approx.) : ~{stats.get('estimated_input_tokens', 0)} in / "
            f"~{stats.get('estimated_output_tokens', 0)} out "
            "(character estimate, not a tokenizer)"
        )

    # 0.8.1: compact rate-limit summary — only shown when events occurred.
    # Per-event messages were already printed in real time during the run.
    rate_limit_warnings = stats.get("rate_limit_warnings", [])
    if rate_limit_warnings:
        event_count = len(rate_limit_warnings)
        providers = sorted({w["provider"] for w in rate_limit_warnings})
        total_sleep = sum(w.get("sleep_s", 0) or 0 for w in rate_limit_warnings)
        sleep_note = f", {total_sleep:.1f}s total backoff" if total_sleep > 0 else ""
        print(
            f"\n  Rate limits: {event_count} step-down event(s) "
            f"[{', '.join(providers)}]{sleep_note}. "
            "Details in error.log."
        )

    # Always print issue log path when any issue was recorded (Work Item 4).
    issues = stats.get("issues_recorded", 0)
    error_log = stats.get("error_log")
    if issues and error_log:
        failed = stats.get("failed", 0)
        if failed > 0:
            print(f"\n  {failed} file(s) failed. See {error_log} for details.")
        else:
            print(f"\n  {issues} issue(s) recorded (all recovered). See {error_log} for details.")


def run_cli(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Exit-code contract (0.9.2):
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

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: project root is not a directory: {root}", file=sys.stderr)
        return 2

    overrides: dict = {}
    if args.entry:
        overrides["entry_file"] = args.entry
    if args.documentation_scope is not None:
        overrides["documentation_scope"] = args.documentation_scope
    if args.manage_output_gitignore is not None:
        overrides["manage_output_gitignore"] = args.manage_output_gitignore
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
    if args.safe_mode:
        overrides["safe_mode"] = True
    if args.no_parallel:
        overrides["parallel_agents"] = False
    if args.analysis_mode is not None:
        overrides["analysis_mode"] = args.analysis_mode
    if args.max_parallel_files is not None:
        overrides["max_parallel_files"] = args.max_parallel_files
    if args.verbose:
        overrides["log_level"] = "DEBUG"
    if args.dry_run:
        overrides["dry_run"] = True
    if args.max_files is not None:
        overrides["max_files"] = args.max_files
    if args.force_files:
        overrides["force_files"] = args.force_files
    if args.allow_partial:
        overrides["allow_partial"] = True

    try:
        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(root, config_overrides=overrides)

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
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt as exc:
        # 0.9.8: the pipeline attaches the exact selected crash-recovery path to
        # the interrupt as ``recovery_path`` only when that file exists on disk.
        # The stable output is never touched mid-run, so it is always preserved;
        # we just report which file enables resume (or that none was created).
        recovery_path = getattr(exc, "recovery_path", None)
        if recovery_path:
            print(
                "\nRun interrupted. Your previous stable output was left untouched. "
                "Files completed before the interrupt are saved in the crash-recovery "
                f"file:\n  {recovery_path}\n"
                "Re-run the same command to resume from it.",
                file=sys.stderr,
            )
        else:
            print(
                "\nRun interrupted before any crash-recovery file was created or "
                "confirmed. Your previous stable output (if any) was left untouched. "
                "Re-run the same command to start or resume.",
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
            # 0.9.7: a doomed-run safe stop — not an unexpected crash.  The
            # pipeline already recorded and flushed it to error.log; here we only
            # present it.  Completed files are in the live JSON backup and
            # re-running resumes.  A *terminal* abort (billing/credentials/model/
            # access) is a setup/credentials class problem → exit 2 (consistent
            # with ConfigError/ProviderInitError).  A *bounded rate-limit / quota*
            # stop is a transient "retry later" condition, not a credentials
            # fault → exit 1 so automation does not read it as "fix credentials".
            print(f"Error: {exc}", file=sys.stderr)
            print(
                "\nCompleted files are saved in the live JSON backup in your "
                "output directory. Re-run the same command to resume — only the "
                "unfinished files will be re-documented.",
                file=sys.stderr,
            )
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 2 if getattr(exc, "category", None) == "terminal" else 1
        if isinstance(exc, ConfigError):
            # Includes ProviderInitError (provider initialization failures),
            # ownership conflicts, and the max_files cap.
            print(f"Error: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 2
        if isinstance(exc, OutputError):
            print(f"Error: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        print(f"Fatal error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point: exit nonzero via SystemExit, return on success."""
    code = run_cli(argv)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
