# codedoc-ai

`codedoc-ai` generates structured, incrementally reusable documentation for source
repositories. It scans source locally, builds a deterministic dependency graph,
sends only files that need analysis to a configured LLM, and writes JSON,
Markdown, or both.

## Highlights

- Explicit or auto-detected entry files, with `entry` or `all` documentation scope.
- One combined provider call per file by default; optional triple-agent analysis.
- Exact-output incremental reuse based on source hashes and analysis identity.
- One fixed crash-recovery file that preserves completed work after interruption.
- Config-only, validated instruction customization with mandatory semantic review.
- OpenAI, Anthropic, Gemini, and OpenAI-compatible endpoint support.
- Read-only dry runs, paid-file caps, deterministic ownership guards, and stable
  CI-oriented exit codes.

## Installation

```bash
pip install codedoc-ai
```

## Quick start

Set a provider credential in the process environment, then run CodeDoc:

```bash
export OPENAI_API_KEY="your-key"
codedoc --entry src/main.py
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
codedoc --entry src/main.py
```

The default output is `codedoc/codedoc.json`. Common alternatives:

```bash
codedoc --format md
codedoc --format both
codedoc --output docs/report.json
codedoc --documentation-scope all
codedoc --dry-run --max-files 25
```

The optional leading verbs `run` and `execute` are accepted aliases; examples
use the shorter canonical spelling. An entry is optional: CodeDoc can recover it
from the selected output, auto-detect a configured candidate, or document all
scanned files when no candidate exists.

On later runs, use the same output selection. CodeDoc reads only the exact
selected target(s), reuses unchanged owned records, and reprocesses changed files.

## File contract

CodeDoc automatically manages a deliberately small set of persistent files:

| Phase | Exact file | Purpose |
| --- | --- | --- |
| Configuration | `<project>/codedoc.config.json` | Optional runtime configuration and inline instructions. |
| Active run | `<output>/crash_recovery.json` | In-progress recovery state. |
| Final output | Exact selected `.json`, `.md`, or both | Stable CodeDoc-owned result. |

There is no alternate-config search, external prompt-profile search, prompt
directory, `.env` loading, checkpoint/build/database migration, sibling-output
discovery, persistent issue log, or managed `.gitignore` behavior. Stray legacy
files are not opened, migrated, renamed, or deleted.

Temporary atomic-write siblings and writability probes are short-lived
implementation details. They use unique names in the target directory and are
cleaned up best-effort.

## Configuration

Generate a complete, valid, editable configuration from the canonical defaults:

```bash
codedoc --init-config
```

This writes `codedoc.config.json` in the current directory. It includes every
public setting, `api_key: null`, and editable schema-v2 single/triple instruction
defaults. Credentials are never copied into the file.

Existing targets are refused unless `--force` is supplied. Forced regeneration
validates the existing file and atomically replaces only `prompt_profiles`; every
other top-level setting and value is preserved, and no backup is created. CodeDoc
reads subsequent edits from this exact active file.

Useful defaults include:

| Setting | Default |
| --- | --- |
| `llm_provider` | `auto` |
| `model_name` | provider default |
| `documentation_scope` | `entry` |
| `analysis_mode` | `single` |
| `output_dir` | `codedoc` |
| `output_format` | `json` |
| `max_parallel_files` | `5` |
| `file_retry_attempts` | `1` |
| `max_file_size_kb` | `500` |
| `max_content_chars` | `12000` |
| `follow_symlinks` | `false` |
| `propagate_changes` | `true` |
| `rate_limit_adaptive` | `true` |

Run `codedoc --init-config` rather than copying a partial example when you need
the complete current key set.

## Command-line options

| Flag | Purpose |
| --- | --- |
| `--entry FILE` | Select an entry file; otherwise recover or auto-detect one. |
| `--documentation-scope {entry,all}` | Document entry-reachable files or all scanned files. |
| `--provider NAME` | Select `auto`, `openai`, `anthropic`, or `gemini`. |
| `--model MODEL` | Override the provider model. |
| `--output PATH` | Select an output directory or exact `.json`/`.md` file. |
| `--format {json,md,both}` | Select output format. |
| `--ignore PATH` | Add a project-relative ignored path; repeatable. |
| `--skip-dirs DIRS` | Replace default skipped directory names with a comma-separated list. |
| `--add-skip-dir DIR` | Add a skipped directory name; repeatable. |
| `--remove-skip-dir DIR` | Remove a default skipped directory name; repeatable. |
| `--dry-run` | Plan without writes or provider calls. |
| `--max-files N` | Cap files allowed to make documentation calls (`0` is unlimited). |
| `--force-files FILE` | Reprocess a selected path even when unchanged; repeatable. |
| `--allow-partial` | Exit zero after a completed run with file failures. |
| `--no-parallel` | Disable within-file parallel agents in triple mode. |
| `--analysis-mode {single,triple}` | Select one combined call or the three-agent path. |
| `--init-config` | Create the complete active config and exit. |
| `--force` | With `--init-config`, refresh only editable profiles. |
| `--max-parallel-files N` | Set concurrent file processing (default `5`). |
| `--truncation-head-ratio FLOAT` | Set the head/tail source truncation split. |
| `--verbose`, `-v` | Enable debug logging. |
| `--version` | Print the installed version and exit. |

Ignore rules are resolved from `--ignore`/`ignore_paths` and the
`--skip-dirs`/`--add-skip-dir`/`--remove-skip-dir` family. Paths are
project-relative; skip-directory values are directory names.

## Providers and environment variables

Credentials and ordinary scalar overrides may come from operating-system
environment variables. CodeDoc does not read `.env` files.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI credential. |
| `ANTHROPIC_API_KEY` | Anthropic credential. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini credential. |
| `LLM_API_KEY` | Generic fallback credential. |
| `LLM_PROVIDER` | `auto`, `openai`, `anthropic`, or `gemini`. |
| `MODEL_NAME` | Provider model name. |
| `API_BASE_URL` | OpenAI-compatible endpoint base URL. |
| `OUTPUT_DIR` | Output directory or exact output file. |
| `CODEDOC_OUTPUT_FORMAT` | `json`, `md`, or `both`. |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |
| `CODEDOC_IGNORE_PATHS` | Semicolon-separated project-relative paths to ignore. |
| `CODEDOC_MAX_PARALLEL_FILES` | File concurrency. |
| `CODEDOC_FILE_RETRY_ATTEMPTS` | Per-file retry attempts. |
| `CODEDOC_MAX_CONSECUTIVE_FAILURES` | Consecutive-failure abort threshold. |
| `CODEDOC_MAX_CONTENT_CHARS` | Per-file prompt content ceiling. |
| `CODEDOC_DRY_RUN` | Planning-only mode. |
| `CODEDOC_MAX_FILES` | Paid-file cap (`0` means unlimited). |
| `CODEDOC_FORCE_FILES` | Semicolon-separated project-relative paths. |
| `CODEDOC_ALLOW_PARTIAL` | Allow a completed partial run to exit zero. |
| `CODEDOC_ANALYSIS_MODE` | `single` or `triple`. |
| `CODEDOC_TRUNCATION_HEAD_RATIO` | Head fraction for source truncation. |

Provider defaults are OpenAI `gpt-4o-mini`, Anthropic
`claude-haiku-4-5-20251001`, and Gemini `gemini-2.5-flash`. Select explicitly with
`--provider` and `--model`, or use `auto` with a recognized model prefix.

## Inline instructions

The only runtime instruction source is `prompt_profiles` inside the exact
`codedoc.config.json` (or an in-memory Python override). Both schema versions are
readable:

- schema v1 uses `fields`;
- schema v2 uses `requested_shape` and is the generated/recommended form.

Every present mode uses a required `common` envelope and an optional
`per_language` complete replacement:

```json
{
  "prompt_profiles": {
    "schema_version": 2,
    "single": {
      "common": {
        "requested_shape": {
          "description": "Explain what this file does.",
          "role_in_system": "Explain its architectural role."
        }
      },
      "per_language": {}
    }
  }
}
```

The old flat mode layout is rejected with migration guidance. `per_extension` is
reserved for a future additive design and is currently rejected. Its documented
future precedence is `per_extension > per_language > common`, with full-block
replacement rather than field merging.

`analysis_mode: single` exposes one combined editable instruction JSON at
`single.common`. Triple mode exposes three independently editable instruction
JSON blocks at `triple.common.structure`, `.dependency`, and `.documentation`.
Supported field order, optional fields, and bounded instruction text are editable;
fixed system, safety, factuality, scanning, retry, cache, and serialization rules
are not. `per_language` remains a complete-block replacement.

An effective non-default instruction is reviewed only when it will reach a planned
LLM documentation call. `SAFE` continues, `RISKY` requires explicit per-run
confirmation, and `TOO_RISKY` always stops. Initialization, unedited defaults,
dry runs, cache-only work, and deterministic JSON↔Markdown conversion make no
security-review call. There is no stored bypass.
Fixed system roles, factuality/safety rules, parser facts, cleaners, provider
selection, scanning, retry, cache, ownership, and artifact serialization are not
customizable.

Use `codedoc --init-config` as the registry-backed reference for exact fields,
types, and complete defaults.

## Output, incremental reuse, and ownership

JSON mode reads only its exact JSON target. Markdown mode reads only its exact
Markdown target, including the lossless embedded project view. Both mode reads
only its exact two targets and blocks before provider contact if schema, entry,
path set, hashes, or cache identity disagree.

An unrelated sibling is never used. For example, selecting `docs/report.json`
does not inspect `docs/report.md`. When no selected output supplies an entry,
ordinary source entry auto-detection runs.

CodeDoc refuses to overwrite foreign, empty, or malformed final targets. Custom
output names remain supported when supplied explicitly:

```bash
codedoc --output docs/report.json
codedoc --output docs/report.md
```

`--format both` requires a directory because it writes two files.

## Crash recovery

Every real run uses exactly `<output_dir>/crash_recovery.json`. The file is
created only after ownership/path checks, deterministic validation, read-only
planning, paid caps, and any mandatory semantic review have succeeded. It is
updated atomically after each completed file.

The recovery file includes a versioned identity covering project root, exact
selected targets, entry, documentation scope, analysis mode/revision, and sorted
per-language profile digests. A compatible in-progress file is resumed. A
foreign, malformed, completed, unsupported, or identity-mismatched file blocks
without mutation. To start fresh, delete `crash_recovery.json` in the output
directory; to resume, restore the prior run configuration.

On success, final output is atomically replaced first and recovery is removed
second. Interruptions, provider failures, and final-output failures preserve the
stable prior output and the recovery file. Dry-run may inspect the exact recovery
path but never creates, changes, or deletes it.

## Planning and diagnostics

`--dry-run` performs scanning and planning without persistent mutation, provider
creation, or API calls. `--max-files N` counts only files that would make
documentation calls. `--force-files PATH` bypasses reuse for a selected file.

Issues are bounded in memory, printed to the terminal, and included in permitted
final/recovery metadata. CodeDoc does not write `error.log`.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success, dry-run success, or explicitly allowed completed partial output. |
| `1` | Processing/output failure or bounded rate-limit stop. |
| `2` | Invalid input/config/path, ownership/recovery conflict, cap failure, or terminal provider failure. |
| `130` | Keyboard interrupt. |

## Python API

```python
from codedoc import run_pipeline

stats = run_pipeline({
    "entry_file": "src/main.py",
    "output_format": "json",
    "max_parallel_files": 3,
})

stats = run_pipeline("/path/to/project", {"output_format": "both"})
```

In-memory overrides are supported but do not create another persistent config
source. Unsupported settings such as external prompt paths, risky-review bypass,
safe mode, and managed output ignore files raise targeted configuration errors.

## Troubleshooting

- Missing credential: set the matching provider environment variable.
- Unexpected paid work: run the identical command with `--dry-run` and inspect
  the exact output selection and analysis mode.
- Recovery conflict: follow the error’s expected/found field, then restore the
  prior configuration or delete the exact `crash_recovery.json` to start fresh.
- Missing files: check entry selection, `documentation_scope`, `skip_dirs`,
  `ignore_paths`, `extension_language_map`, and `max_file_size_kb`.
- Rate limits: lower `max_parallel_files`; adaptive stepping is enabled by default.

## License

See [LICENSE](LICENSE).
