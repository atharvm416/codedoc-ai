"""
codedoc CLI entry point.

0.8.1 changes
-------------
- Version bumped to 0.8.1.
- ``--safe-mode`` is marked as deprecated in the help text (the flag is kept
  for backwards compatibility and printed as a no-op warning at runtime).
- Error / issue log path is always printed when any issue is recorded, not
  only when ``failed > 0``.
- Rate-limit step-down warnings from ``stats["rate_limit_warnings"]`` are
  printed to stdout.
- Interrupt message includes the live backup path from
  ``stats["live_backup_path"]`` when available.

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
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=False,
        help=(
            "[DEPRECATED] Live JSON backup is now always on in 0.8.0 — this flag "
            "has no additional effect and is kept only for backwards compatibility. "
            "It will be removed in a future release."
        ),
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help=(
            "Disable parallel agent execution within each file. "
            "Useful when an API has strict concurrency limits."
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
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.9.0",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if argv and argv[0] in {"run", "execute"}:
        argv = argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    overrides: dict = {}
    if args.entry:
        overrides["entry_file"] = args.entry
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
    if args.max_parallel_files is not None:
        overrides["max_parallel_files"] = args.max_parallel_files
    if args.verbose:
        overrides["log_level"] = "DEBUG"

    try:
        from codedoc.pipeline import run_pipeline
        stats = run_pipeline(root, config_overrides=overrides)

        print(f"\ncodedoc complete.")
        print(f"  Files documented : {stats['checked']}")
        print(f"  Files reused     : {stats.get('reused', 0)}")
        if stats.get("resumed", 0):
            print(f"  Files resumed    : {stats['resumed']}")
        print(f"  Files failed     : {stats['failed']}")
        print(f"  Output directory : {stats['output_dir']}")
        for output_file in stats.get("output_files", []):
            print(f"  Output file      : {output_file}")

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

        if stats.get("failed", 0) > 0:
            sys.exit(1)

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        # Try to report the live backup path if available in stats.
        backup_msg = ""
        try:
            from codedoc.pipeline import _resolve_live_backup_path
        except Exception:
            pass
        print(
            "\nRun interrupted. Progress has been saved to the live JSON backup — "
            "re-run the same command to resume from where it stopped." + backup_msg,
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as exc:
        from codedoc.utils.errors import ConfigError
        if isinstance(exc, ConfigError):
            print(f"Error: {exc}", file=sys.stderr)
        else:
            print(f"Fatal error: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
