"""
codedoc CLI entry point.

Usage:
    codedoc run                        # document current directory
    codedoc execute                    # alias for codedoc run
    codedoc .                          # document current directory
    codedoc /path/to/project           # document a specific directory
    codedoc run --entry src/main.py    # specify entry file
    codedoc run --llm local            # use local LLM (Ollama)
    codedoc run --provider gemini      # use Google Gemini
    codedoc run --model gpt-4o         # override model
    codedoc run --output ./my_docs     # override output directory
    codedoc run --format md            # write Markdown instead of JSON
    codedoc run --verbose              # debug logging
    python -m codedoc run              # same as above, alternative invocation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codedoc",
        description="AI-powered codebase documentation — local-first, LLM-agnostic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  codedoc run                        document the current directory
  codedoc execute                    alias for codedoc run
  codedoc .                          document the current directory (legacy style)
  codedoc /path/to/project           document a specific project
  codedoc run src                    document ./src
  codedoc run --entry src/main.py    start from a specific entry file
  codedoc run --llm local            use a local LLM (Ollama / LM Studio)
  codedoc run --provider gemini      use Google Gemini
  codedoc run --model claude-haiku-4-5-20251001 --llm api   use Claude
  codedoc run --output ./docs        write output to ./docs
  codedoc run --format md            write one combined Markdown file
  codedoc run --ignore /myenv        ignore a project-root path
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
        help="Entry file relative to project root (e.g. src/main.py)",
    )
    parser.add_argument(
        "--llm",
        choices=["api", "local"],
        default=None,
        help="LLM mode: 'api' (OpenAI/Claude) or 'local' (Ollama/LM Studio)",
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
        help="Model name to use (e.g. gpt-4o-mini, claude-haiku-4-5-20251001, qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Output directory for generated docs (default: ./docs_output)",
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
            "Project-relative file or directory path to ignore. "
            "Can be passed multiple times, e.g. --ignore /myenv --ignore generated."
        ),
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        default=False,
        help="Disable parallel agent execution (useful for local LLMs with limited VRAM)",
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
        version="%(prog)s 0.5.0",
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

    # Build config overrides from CLI flags
    overrides: dict = {}
    if args.entry:
        overrides["entry_file"] = args.entry
    if args.llm:
        overrides["llm_mode"] = args.llm
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
        print(f"  Files failed     : {stats['failed']}")
        print(f"  Output directory : {stats['output_dir']}")
        for output_file in stats.get("output_files", []):
            print(f"  Output file      : {output_file}")

        if stats["failed"] > 0:
            print(f"\n  See error.log in {root} for details.")
            sys.exit(1)

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
