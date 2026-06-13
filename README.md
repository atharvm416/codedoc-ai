# codedoc-ai

`codedoc-ai` is a Python library and CLI that generates structured, reusable documentation memory for source codebases. It is built for AI coding agents, human maintainers, and teams that want a stable map of a project before making changes.

The tool scans source files, resolves project-local imports into a dependency graph, sends only files that need analysis to an LLM, and writes one combined, structured documentation artifact designed for both humans and AI. By default that artifact is JSON.

Current release: `0.9.3`.

## What It Does

- Finds supported source files in a project.
- Starts from an explicit entry file when provided.
- Otherwise auto-detects common entry files such as `main.py`, `main.tsx`, `index.html`, `Main.java`, and related names.
- If an entry file is found, documents that file and its reachable project dependencies.
- If no entry file is found, documents all supported project files.
- Parses imports locally before calling an LLM.
- Processes dependencies before dependent files where possible.
- Processes up to 5 files at a time by default.
- Retries failed parallel files sequentially for clearer diagnostics.
- Stops early with actionable provider/API health messages when many files fail consecutively.
- Uses SHA-256 content hashes as smart file IDs.
- Reuses existing documentation for unchanged files.
- Reuses existing documentation when another file has identical content.
- Embeds metadata (entry point, schema version, and per-file hashes) in every output file so the next run can resume incrementally without re-specifying the entry.
- Survives interruptions: writes a live JSON backup before any AI work starts, then updates it after every completed file. A Ctrl-C or crash always leaves a readable partial output file — no results are lost, and re-running the same command resumes automatically from where it stopped.
- Adaptive rate-limit parallelism: when a provider signals 429 / rate-limit, file concurrency is stepped down (`5 → 2 → 1`) and a provider-specific warning is printed to the terminal. No manual intervention needed.
- Refuses to overwrite any file it did not create (ownership guard), protecting your data from accidental output collisions.
- Provides a filesystem-read-only `--dry-run` with approximate lower-bound call and token estimates.
- Supports a pre-call `--max-files` cap and repeatable `--force-files` reprocessing.
- Reports stable CI-oriented exit codes and optional `--allow-partial` behavior.
- Writes a clean, structured public project view to `codedoc/codedoc.json` by default, or Markdown when requested.
- Public output includes project overview, file tree, folder map, dependency graph, dependency catalog, and flattened file summaries.
- Converts public JSON to Markdown without another AI call.
- Parses generated Markdown back into the public JSON shape when needed.

## Defaults

If the user runs:

```bash
codedoc run
```

`codedoc` uses these defaults:

| Setting | Default |
| --- | --- |
| LLM provider | `auto` (OpenAI) |
| API model | provider default (OpenAI/auto → `gpt-4o-mini`) |
| Output directory | `codedoc` |
| Output format | `json` |
| Output file | `codedoc/codedoc.json` |
| Parallel agents | `true` |
| Max parallel files | `5` |
| File retry attempts | `1` |
| Max consecutive failures | `5` |
| Change propagation | `true` |
| Live JSON backup | always on (0.8.0 default) |
| Rate-limit adaptive | `true` |
| Max file size | `500 KB` |
| Max content chars | `12000` |
| Dry run | `false` |
| Maximum paid files | `0` (unlimited) |
| Forced files | `[]` |
| Allow partial output | `false` |

Because the default provider uses the OpenAI API, a user must supply an API key unless they select a different provider.

If no model is specified (neither `--model` nor `model_name` in config), each provider falls back to its own default:

| Provider | Default model |
| --- | --- |
| OpenAI / `auto` | `gpt-4o-mini` |
| Anthropic | `claude-haiku-4-5-20251001` |
| Gemini | `gemini-2.5-flash` |

## Installation

Install from PyPI:

```bash
pip install codedoc-ai
```

The package installs the hosted-provider SDKs needed for OpenAI, Anthropic, and Gemini:

```text
openai
anthropic
google-genai
```

## Quick Start

### First Run

Provide an entry point when you want CodeDoc to document only the reachable project dependencies from that file, then save the result to the `codedoc/` folder:

```bash
codedoc run --entry src/main.py
```

`codedoc/codedoc.json` is written by default. The entry point is embedded as metadata in the output file so you never need to specify it again.

Write to a custom location:

```bash
codedoc run --entry src/main.py --output docs/report.json
```

Write only Markdown:

```bash
codedoc run --entry src/main.py --format md
```

### Subsequent Runs

After the first run, just run:

```bash
codedoc run
```

CodeDoc finds `codedoc/codedoc.json` automatically, reads the entry point from its metadata, and only reprocesses files that have changed.

Point to a specific previously generated file:

```bash
codedoc run --output docs/report.json
```

Convert format without any AI calls (served entirely from the cache):

```bash
codedoc run --format md
codedoc run --format both
```

Limit file-level concurrency (useful with strict API rate limits):

```bash
codedoc run --max-parallel-files 3
```

## CLI Help

Use `--help` to see every CLI option supported by the installed version:

```bash
codedoc --help
```

The recommended command is `codedoc run`. The CLI also accepts a project path after `run`; omitting the path means "document the current working directory":

```bash
codedoc run
codedoc run /path/to/project
codedoc run --entry src/main.py --format both --max-parallel-files 5
```

For backward compatibility, `codedoc .` and `codedoc /path/to/project` still work.

Common commands:

| Command | Purpose |
| --- | --- |
| `codedoc run --entry src/main.py` | First run — specify entry file; output to `codedoc/`. |
| `codedoc run` | Subsequent run — entry read from `codedoc/codedoc.json` metadata. |
| `codedoc execute` | Alias for `codedoc run`. |
| `codedoc run --format json` | Write only `codedoc/codedoc.json`. |
| `codedoc run --format md` | Write only `codedoc/codedoc.md`. |
| `codedoc run --format both` | Write both JSON and Markdown. |
| `codedoc run --output docs` | Write output to `docs/` directory. |
| `codedoc run --output docs/report.json` | Write a single named JSON file. |
| `codedoc run --output docs/report.md` | Write a single named Markdown file. |
| `codedoc run --provider gemini --model gemini-2.5-flash` | Use Google Gemini. |
| `codedoc run --provider anthropic --model claude-haiku-4-5-20251001` | Use Anthropic Claude. |
| `codedoc run --ignore /myenv --ignore generated` | Ignore project paths. |
| `codedoc run --dry-run --max-files 25` | Inspect the plan without writes, provider creation, or API calls. |
| `codedoc run --max-files 25` | Stop before mutation or API calls if more than 25 files need LLM work. |
| `codedoc run --force-files src/a.py --force-files src/b.py` | Explicitly reprocess selected files. |
| `codedoc run --allow-partial` | Exit 0 for completed partial runs, with a prominent warning. |
| `codedoc run --max-parallel-files 3` | Limit concurrent file processing. |
| `codedoc .` | Legacy shorthand for documenting the current directory. |
| `codedoc --version` | Print the installed version. |

## Choosing a Provider

| Use case | Recommended provider |
| --- | --- |
| Best default quality with minimal setup | OpenAI (`gpt-4o-mini` or `gpt-4o`) |
| Claude-specific documentation style or Anthropic account | Anthropic Claude |
| Google AI Studio / Gemini account | Google Gemini |
| OpenAI-compatible gateway such as LiteLLM or a custom endpoint | OpenAI mode with `api_base_url` |

Provider selection is deterministic:

- `llm_provider = "openai"` uses OpenAI or any OpenAI-compatible API.
- `llm_provider = "anthropic"` uses Anthropic Claude.
- `llm_provider = "gemini"` uses Google Gemini through the official `google-genai` SDK.
- `llm_provider = "auto"` with a model name starting with `claude` uses Anthropic.
- `llm_provider = "auto"` with a model name starting with `gemini` uses Gemini.
- `llm_provider = "auto"` with any other model uses OpenAI or an OpenAI-compatible API.
- If OpenAI/`auto` is selected and no model is provided, `gpt-4o-mini` is used.
- If Gemini is selected and no model is provided, `gemini-2.5-flash` is used.
- If Anthropic is selected and no model is provided, `claude-haiku-4-5-20251001` is used.

## OpenAI API Setup

Use OpenAI when you want the default hosted API path.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-your-openai-key"
codedoc run --model gpt-4o-mini
```

Windows Command Prompt:

```bat
set OPENAI_API_KEY=sk-your-openai-key
codedoc run --model gpt-4o-mini
```

macOS/Linux:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
codedoc run --model gpt-4o-mini
```

OpenAI-compatible API example:

```bash
codedoc run --model your-model-name
```

For compatible APIs, set `api_base_url` in `codedoc.config.json` or `API_BASE_URL` in `.env`.

## Anthropic API Setup

Use Anthropic by selecting the `anthropic` provider or using a model name that starts with `claude`.

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
codedoc run --provider anthropic --model claude-haiku-4-5-20251001
```

Windows Command Prompt:

```bat
set ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
codedoc run --provider anthropic --model claude-haiku-4-5-20251001
```

macOS/Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
codedoc run --provider anthropic --model claude-haiku-4-5-20251001
```

## Gemini API Setup

Use Gemini when you want Google's hosted Gemini models. Set `llm_provider` to `gemini`, or use a Gemini model name with `llm_provider` left as `auto`.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
codedoc run --provider gemini --model gemini-2.5-flash
```

Windows Command Prompt:

```bat
set GEMINI_API_KEY=your-gemini-api-key
codedoc run --provider gemini --model gemini-2.5-flash
```

macOS/Linux:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
codedoc run --provider gemini --model gemini-2.5-flash
```

`GOOGLE_API_KEY` is also supported as an alias for `GEMINI_API_KEY`.

## Configuration

Create `codedoc.config.json` in the project being documented:

```json
{
  "llm_provider": "auto",
  "model_name": "gpt-4o-mini",
  "api_base_url": null,
  "entry_file": null,
  "output_dir": "codedoc",
  "output_format": "json",
  "supported_extensions": [".py", ".ts", ".tsx", ".js", ".jsx", ".dart", ".java", ".cs", ".html"],
  "parallel_agents": true,
  "max_parallel_files": 5,
  "file_retry_attempts": 1,
  "max_consecutive_failures": 5,
  "log_level": "INFO",
  "max_file_size_kb": 500,
  "propagate_changes": true,
  "rate_limit_adaptive": true,
  "parallel_ladder": null,
  "respect_retry_after": true,
  "retry_after_cap_s": 30,
  "rate_limit_backoff_s": null,
  "rate_limit_backoff_scale": null,
  "rate_limit_signals_add": [],
  "rate_limit_signals_remove": [],
  "skip_dirs": ["myenv", ".venv", "venv", "env", "node_modules", "__pycache__", "codedoc"],
  "skip_dirs_add": [],
  "skip_dirs_remove": [],
  "max_content_chars": 12000,
  "extension_language_map": {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".dart": "dart",
    ".java": "java",
    ".cs": "csharp",
    ".html": "html",
    ".htm": "html",
    ".kt": "kotlin",
    ".swift": "swift",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp"
  },
  "extension_language_map_add": {},
  "extension_language_map_remove": [],
  "auto_entry_candidates": ["index.html", "main.tsx", "main.ts", "main.js", "main.py", "main.dart", "Main.java", "Program.cs"],
  "auto_entry_candidates_add": [],
  "auto_entry_candidates_remove": [],
  "provider_prefixes": {
    "anthropic": ["claude"],
    "gemini": ["gemini"],
    "openai": ["gpt-", "o1", "o3", "text-"]
  },
  "provider_prefixes_add": {},
  "provider_prefixes_remove": {},
  "ignore_paths": ["/myenv", "services/generated"]
}
```

Configuration precedence, from strongest to weakest:

1. CLI flags, such as `--model`, `--provider`, `--format`, and `--output`.
2. Environment variables and values loaded from `.env`.
3. `codedoc.config.json` or `config.json`.
4. Built-in defaults.

Supported output formats:

| Value | Result |
| --- | --- |
| `json` | Writes only `codedoc/codedoc.json`. |
| `md` | Writes only `codedoc/codedoc.md`. |
| `both` | Writes both combined files. |

Parallelism settings:

| Setting | Purpose |
| --- | --- |
| `parallel_agents` | Runs the structure and dependency agents for a single file in parallel. |
| `max_parallel_files` | Maximum number of files processed at the same time. Default: `5`. |
| `file_retry_attempts` | Number of sequential retries for a failed file. Default: `1`. |
| `max_consecutive_failures` | Stops the run after repeated failures so provider/API problems are visible quickly. Default: `5`. |

Configurable defaults added in 0.8.1:

| Setting | Purpose |
| --- | --- |
| `skip_dirs`, `skip_dirs_add`, `skip_dirs_remove` | Replace, extend, or reduce directory names skipped anywhere in the tree. Use `--remove-skip-dir codedoc` to document this package source while codedoc still skips its output directory. |
| `extension_language_map`, `extension_language_map_add`, `extension_language_map_remove` | Control which extensions are scanned and what language label each gets. Any extension in the resolved map is supported. |
| `auto_entry_candidates`, `auto_entry_candidates_add`, `auto_entry_candidates_remove` | Control first-run entry auto-detection when `--entry` is omitted. |
| `provider_prefixes`, `provider_prefixes_add`, `provider_prefixes_remove` | Control model-name based provider auto-detection and matching API-key lookup. |

Configurable settings added in 0.9.0:

| Setting | Default | Purpose |
| --- | --- | --- |
| `max_content_chars` | `12000` | Maximum characters sent to the LLM per file. Long files are truncated once, one WARNING reports the path and counts, and the marker stays inside the ceiling. Must be at least `1000`. |

Planning and CI settings added in 0.9.2:

| Setting | Default | Purpose |
| --- | --- | --- |
| `dry_run` | `false` | Compute the real routing plan without filesystem mutation or provider/API interaction. |
| `max_files` | `0` | Maximum files allowed to make LLM calls after reuse and resume decisions. `0` is unlimited. |
| `force_files` | `[]` | Selected project paths to reprocess explicitly before dependency propagation. |
| `allow_partial` | `false` | Exit 0 only for completed runs that produced partial output after file failures. |

## Environment Variables

Secrets should live in environment variables or a local `.env` file that is ignored by Git. Use [.env.example](.env.example) as the template.

Supported variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key. |
| `ANTHROPIC_API_KEY` | Anthropic API key. |
| `GEMINI_API_KEY` | Google Gemini API key. |
| `GOOGLE_API_KEY` | Google API key alias for Gemini. |
| `LLM_API_KEY` | Generic fallback API key. |
| `LLM_PROVIDER` | `auto`, `openai`, `anthropic`, or `gemini`. |
| `MODEL_NAME` | Model name to use. |
| `API_BASE_URL` | OpenAI-compatible base URL for custom or gateway endpoints. |
| `OUTPUT_DIR` | Output directory. |
| `CODEDOC_OUTPUT_FORMAT` | `json`, `md`, or `both`. |
| `CODEDOC_SAFE_MODE` | Deprecated — live backup is always on since 0.8.0. |
| `CODEDOC_MAX_PARALLEL_FILES` | Maximum files processed at once. |
| `CODEDOC_FILE_RETRY_ATTEMPTS` | Sequential retry attempts for a failed file. |
| `CODEDOC_MAX_CONSECUTIVE_FAILURES` | Consecutive failure threshold before stopping. |
| `LOG_LEVEL` | `INFO`, `DEBUG`, etc. |
| `CODEDOC_IGNORE_PATHS` | Semicolon-separated ignore paths. |
| `CODEDOC_MAX_CONTENT_CHARS` | Maximum characters of file content sent to the LLM. Equivalent to `max_content_chars` in config. |
| `CODEDOC_DRY_RUN` | Boolean planning-only mode. |
| `CODEDOC_MAX_FILES` | Non-negative paid-file cap; `0` is unlimited. |
| `CODEDOC_FORCE_FILES` | Semicolon-separated forced project paths. |
| `CODEDOC_ALLOW_PARTIAL` | Boolean partial-output exit-code override. |

Example `.env` for OpenAI:

```text
OPENAI_API_KEY=sk-your-openai-key
MODEL_NAME=gpt-4o-mini
CODEDOC_OUTPUT_FORMAT=json
```

Example `.env` for Anthropic:

```text
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
LLM_PROVIDER=anthropic
MODEL_NAME=claude-haiku-4-5-20251001
CODEDOC_OUTPUT_FORMAT=json
```

Example `.env` for Gemini:

```text
GEMINI_API_KEY=your-gemini-api-key
LLM_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
CODEDOC_OUTPUT_FORMAT=json
```

## Ignore Rules

Use `skip_dirs` for directory names that should be skipped anywhere in the tree.

Use `ignore_paths` for strict project-relative paths. A leading slash means "from the project root", so `/myenv` ignores only the root `myenv` directory.

CLI example:

```bash
codedoc run --entry main.py --ignore /myenv --ignore services/generated
```

Environment variable example:

Windows PowerShell:

```powershell
$env:CODEDOC_IGNORE_PATHS="/myenv;services/generated"
```

macOS/Linux:

```bash
export CODEDOC_IGNORE_PATHS="/myenv;services/generated"
```

## Output and Cache

`codedoc` writes all output to the configured output directory. The project root is never written to.

Default output:

```text
codedoc/codedoc.json
```

JSON only:

```bash
codedoc run --format json
```

```text
codedoc/codedoc.json
```

Markdown only:

```bash
codedoc run --format md
```

```text
codedoc/codedoc.md
```

Custom output file name and location:

```bash
codedoc run --entry src/main.py --output project_docs/analysis.json
codedoc run --entry src/main.py --output project_docs/analysis.md
```

When a file path is passed to `--output`, the format is inferred from the extension — no need to also pass `--format`. Passing an unsupported extension (anything other than `.json` or `.md`) stops the run with a clear error. `--format both` requires a directory, not a named file.

### Metadata and Resume

Every generated file embeds a small metadata block that stores the entry point and schema version. This is how CodeDoc resumes documentation runs without asking for `--entry` a second time.

In JSON files the block is the first key in the document:

```json
{
  "_codedoc": {
    "entry_file": "src/main.py",
    "schema_version": "1.4"
  },
  ...
}
```

Since 0.9.3 the completed output contains no run-varying timestamp: two runs over identical sources, documentation, configuration, and stats produce byte-identical JSON and Markdown. Older outputs that still contain a `generated_at` field remain fully readable. (Live crash-safety backups keep `created_at` / `updated_at` diagnostics.)

In Markdown files it is an HTML comment at the very top. It also embeds `file_hashes` so that subsequent Markdown-only runs can perform incremental hash checks without requiring a sibling JSON file:

```text
<!-- codedoc-ai: {"entry_file": "src/main.py", "schema_version": "1.4", "file_hashes": {"src/main.py": "abc123...", ...}} -->
```

If this metadata is missing or corrupted, `codedoc` raises a clear error rather than silently failing. To recover, re-run with `--entry` to generate a fresh document.

If a JSON output file is missing but an identically-named Markdown file is present (e.g. `codedoc/claude.md` when `codedoc/claude.json` is expected), `codedoc` reads the entry point from the Markdown metadata and resumes from there.

### Incremental Cache Behaviour

Incremental state lives inside the output file itself — there is no separate cache database. On each run, `codedoc` reads the existing output file, extracts per-file hashes and documentation records, and compares them against current file content. Only files whose content has changed are sent to the LLM.

The CLI logs the selected output format and the exact output file path during execution for better visibility.

The public `codedoc.json` and `codedoc.md` are structured, human- and AI-readable output files. They include:

- Project overview (entry file, file count, languages).
- File tree representation.
- Folder-based grouping with summaries.
- Internal dependency graph between project files.
- Project-level dependency catalog with deduplicated dependency purpose.
- Flattened file summaries (no nested duplication).
- Imports, exports, functions, classes.
- Internal, external, SDK/standard-library, and reverse dependencies (`imported_by`).

Since 0.9.3, third-party packages and language standard-library / SDK modules are separated: each file's `links` carry `external_dependencies` (third-party) and `sdk_dependencies` (e.g. Python stdlib, Dart `dart:*`, Node built-ins). The `SDK / Standard Library` Markdown section is rendered only when non-empty, and `internal_dependencies` / `imported_by` are derived **only** from resolved project-graph edges — unresolved agent text can never become an internal link. Missing `sdk_dependencies` loads as an empty list for older outputs.

They exclude internal processing data such as raw LLM responses and per-file history.

### Dependency Catalog

`codedoc-ai` keeps dependency details useful without repeating the same explanation in every file. The AI may suggest internal `catalog_updates` while processing individual files. The public output consumes those updates and emits one merged `dependency_catalog`.

Example public JSON:

```json
{
  "dependency_catalog": [
    {
      "name": "pydantic",
      "type": "external",
      "used_for": "Defines validated schema models for API data.",
      "files": ["schemas/userschema.py", "schemas/projectschema.py"],
      "file_count": 2
    }
  ],
  "files": [
    {
      "path": "schemas/userschema.py",
      "links": {
        "external_dependencies": ["pydantic"],
        "sdk_dependencies": ["typing"]
      }
    }
  ]
}
```

The catalog is grouped by `(type, canonical_name)`, so the same package seen across files merges into one entry, while `external` and `sdk` entries stay distinct. An `internal` catalog hint from the model is kept only when it exactly matches a resolved internal path for that file; otherwise it is reclassified as a third-party / SDK dependency.

The file still says what it uses. The shared explanation lives once in the catalog. This keeps JSON smaller, Markdown cleaner, and later agent analysis less noisy.

### JSON and Markdown Conversion

The LLM is asked for structured JSON-like analysis. Final output formatting is handled by Python code:

```text
AI/cache records
  -> public project view
  -> codedoc.json or codedoc.md
```

That means `--format md` does not require a separate Markdown-generating AI call. Markdown is rendered from the same project view as JSON. The library also provides internal helpers to convert public JSON to Markdown and parse generated Markdown back into the public JSON shape.

## Incremental Processing

On each run, `codedoc` follows this process:

1. Load config and environment.
2. Resolve the entry point — from `--entry` if given, otherwise from metadata in the existing output file or legacy auto-detection.
3. Scan supported files while respecting `skip_dirs` and `ignore_paths`.
4. Build a dependency graph from parsed imports.
5. Select files reachable from the entry point.
6. Normalize forced paths and add valid forced files before dependency propagation.
7. Compute one immutable plan covering changed, unchanged, reused, resumed, and paid-agent files.
8. In `--dry-run`, return that plan and approximate lower-bound usage without writing or creating a provider.
9. In a real run, enforce ownership and `max_files` before creating directories, writers, logs, or providers.
10. Materialize identical-content and checkpoint reuse exactly as planned.
11. Send only paid-agent files to the LLM, retry failures, and write final output.
12. Report actual call attempts and approximate input/output token totals.

This means repeated runs should only send new or changed code to the LLM. Unchanged code and exact duplicate content are reused.

## Crash Recovery and Safe Mode

`codedoc` is built so that interrupting a run — Ctrl-C, a crash, or a dropped
network connection — never forces you to repeat work that already completed.

### Default: always-on live JSON backup

Every run creates a visible live JSON backup in the output directory **before
the first AI call**, then updates it atomically after each completed file.
You do not need to enable anything — `--safe-mode` is deprecated since 0.8.0.

- `codedoc/codedoc.json` is written immediately with a `_crash_safety` banner
  and an empty `files` array, before any LLM request is made.
- After every completed file the backup is updated (`.tmp` rename — atomic).
- If a run is interrupted, the backup stays on disk with `_crash_safety` clearly
  marking it as partial output.
- Re-run the same command — files already in the backup are verified by content
  hash and skipped; only the remaining files are sent to the LLM.
- If a file was edited between the interruption and the re-run, its hash no
  longer matches and it is re-documented, so you never restore stale docs.

The live backup is written atomically (to a temporary sibling, then renamed) so a
crash mid-write can never corrupt it, and writes are thread-safe under parallel
processing.

**Files array ordering.** The `files` array in the live backup follows the
topological (dependency-first) processing order, not completion order or
alphabetical order. An interrupted backup is therefore structured consistently
with the final clean output.

**MD-only and named-MD runs.**
- `--format md`: live backup is `codedoc/codedoc.json`; removed automatically
  after clean Markdown conversion. If interrupted, `codedoc.json` remains as the
  resume source and the next run converts without any LLM calls.
- `--output docs/report.md`: live backup is `docs/report.json` (sibling derived
  from the Markdown stem). Same lifecycle as above.

**`--safe-mode` (deprecated).** This flag is kept for backwards compatibility
and now has no effect — live backup is always on. Passing it prints a deprecation
notice. It will be removed in a future release.

### Adaptive rate-limit parallelism (0.8.1)

When a provider signals 429 / rate-limit / quota-exceeded, codedoc automatically
steps down file-level concurrency instead of hammering the API:

```
[OpenAI] Rate limit detected - your configured max_parallel_files (5) has been
reduced to 2. Retrying 4 remaining file(s) at lower concurrency.
```

The default step-down ladder for `max_parallel_files = 5` is `[5, 2, 1]`.
Customize it in config:

```json
{
  "rate_limit_adaptive": true,
  "parallel_ladder": [5, 2, 1],
  "respect_retry_after": true,
  "retry_after_cap_s": 30
}
```

Provider-specific rate-limit signals are recognised for OpenAI (`429`, `rate limit`,
`rate_limit`, `too many requests`, `tokens per min`, `tpm`, `quota`), Anthropic
(`529`, `overloaded`, `rate_limit`, `429`), and Gemini (`resource_exhausted`,
`quota`, `429`, `503`). Non-rate-limit errors never trigger a step-down.

In 0.8.1, codedoc sleeps between parallel step-down rungs using provider-aware
backoff. You can tune this in config:

```json
{
  "rate_limit_backoff_s": null,
  "rate_limit_backoff_scale": null,
  "rate_limit_signals_add": ["capacity exceeded", "throttled"],
  "rate_limit_signals_remove": ["503"]
}
```

Set `rate_limit_backoff_s` to `0` to disable computed inter-rung backoff.
`Retry-After` hints are still honored when `respect_retry_after` is true.

### Lossless Markdown regeneration (0.8.1)

Markdown output remains human-readable, but codedoc now embeds a hidden
base64-encoded public JSON view in a `<!-- codedoc-ai-view-base64 ... -->`
comment. This lets later Markdown-to-JSON conversion and incremental re-runs
recover dependency catalogs, per-file dependency metadata, links, and hashes
without another LLM call. Legacy Markdown without the embedded view still uses
the best-effort visible Markdown parser.

### Issue log (`error.log`)

When any issue is recorded during a run, codedoc writes `error.log` inside the
**output directory** (e.g. `codedoc/error.log`), not in the project root.  The
absolute path is printed at the end of the run:

```
1 issue(s) recorded (all recovered). See /path/to/codedoc/error.log for details.
```

Recovered rate-limit step-downs are recorded as warnings in `error.log` but
**do not** appear as errors in the final `codedoc.json` or Markdown output.
Only hard file failures are surfaced there.

### Ownership guard

`codedoc` checks that any existing file at the target path was produced by
codedoc (a `_codedoc` metadata block in JSON, or a `<!-- codedoc-ai: -->` comment
in Markdown). If the file is foreign, malformed, or empty, the run stops with a
clear `ConfigError`. Choose a different `--output` directory or remove the
conflicting file to proceed.

**Preflight (0.9.0).** The ownership check now runs *before* any filesystem
changes, directory creation, scanning, or LLM calls. A foreign target that would
block the final write is caught immediately — no tokens are spent and no output
directory is created.

## Planning, Cost Guardrails, and CI

Use `codedoc run --dry-run --max-files 25` to inspect a run safely. Dry-run
uses the same routing plan as real execution. It may read source, existing
outputs, live backups, and legacy checkpoints, but it does not create an output
directory, write `error.log`, initialize `SafeWriter`, create a provider, or
call an API. It works without an API key.

Token figures use a simple character heuristic. Dry-run input totals are
explicitly lower bounds because the documentation prompt includes earlier
agent responses that do not exist during planning. No monetary estimate is
provided.

`--max-files N` counts only files that would actually make LLM calls after
unchanged skipping, identical-content reuse, and eligible checkpoint reuse. A
real run exceeding the cap exits `2` before persistent mutation or provider
creation. Dry-run still exits `0` and reports that the equivalent real run
would fail.

Force selected files with repeatable options:

```bash
codedoc run --force-files src/a.py --force-files src/b.py
```

Explicitly forced files bypass unchanged, identical-content, and checkpoint
reuse. They are added before normal dependency propagation; propagated
dependents retain normal reuse behavior.

CLI exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success, dry-run success, or explicitly allowed partial output. |
| `1` | File-processing failure, output/write failure, or unexpected fatal error. |
| `2` | Invalid input/config/path, ownership conflict, cap exceeded, or provider initialization failure. |
| `130` | Keyboard interrupt. |

`--allow-partial` changes only completed runs with file-level failures. Setup,
ownership, cap, provider initialization, write, and unexpected fatal errors
remain nonzero.

A packaged manual-only GitHub Actions example is installed at
`codedoc/templates/github-actions-codedoc.yml`. It performs a dry-run before
the paid run, applies the same cap to both, uploads documentation as an
artifact, uses `contents: read`, and never commits or pushes. Selected source
is sent to an external provider and API usage may cost money.

### More detail

[`RUN_FLOW.md`](RUN_FLOW.md) documents the full end-to-end pipeline and every
success, interrupt/resume, and failure scenario across OpenAI, Anthropic, and
Gemini.

## Python API

The CLI is not required. You can run the same workflow from Python with `run_pipeline(...)`.

For the current working directory, pass only the config dict:

```python
from codedoc import run_pipeline

stats = run_pipeline({
    "entry_file": "src/main.py",
    "llm_provider": "auto",
    "model_name": "gpt-4o-mini",
    "parallel_agents": True,
    "max_parallel_files": 5,
    "file_retry_attempts": 1,
    "output_dir": "codedoc",
    "output_format": "json",
    "ignore_paths": ["/myenv", "services/generated"],
})

print(stats)
```

You can also pass a project root when you want to document another directory:

```python
from codedoc import run_pipeline

run_pipeline(r"D:\projects\my_app", {"output_format": "both"})
```

These forms are equivalent:

```python
run_pipeline()
run_pipeline(".")
run_pipeline({})
```

Equivalent examples:

```python
from codedoc import run_pipeline

# Same idea as: codedoc run --format md
run_pipeline({"output_format": "md"})

# Same idea as: codedoc run D:\projects\my_app --format both
run_pipeline(r"D:\projects\my_app", {"output_format": "both"})

# Same idea as: codedoc run --max-parallel-files 3 --ignore /myenv
run_pipeline({
    "max_parallel_files": 3,
    "ignore_paths": ["/myenv"],
})
```

CLI flags map directly to config keys:

| CLI option | Python config key |
| --- | --- |
| `PATH` | Optional first `run_pipeline(project_root, ...)` argument |
| `--entry` | `entry_file` |
| `--provider` | `llm_provider` |
| `--model` | `model_name` |
| `--output` | `output_dir` |
| `--format` | `output_format` |
| `--ignore` | `ignore_paths` |
| `--dry-run` | `dry_run: True` |
| `--max-files` | `max_files` |
| `--force-files` | `force_files` |
| `--allow-partial` | `allow_partial: True` |
| `--no-parallel` | `parallel_agents: False` |
| `--max-parallel-files` | `max_parallel_files` |
| `--verbose` | `log_level: "DEBUG"` |

## Troubleshooting

If API mode fails with an API key error:

- Set `OPENAI_API_KEY` for OpenAI models.
- Set `ANTHROPIC_API_KEY` for Claude models. Make sure model names start with `claude`.
- Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini models. Make sure model names start with `gemini`, or pass `--provider gemini`.

If many files fail quickly:

- Check `error.log` in the output directory (e.g. `codedoc/error.log`); `codedoc` records the file and failure context.
- Verify API credentials and model name.
- Check provider rate limits and network connectivity.
- Lower `max_parallel_files`.
- Increase `file_retry_attempts` if failures are temporary.

If files are missing from output:

- Check `entry_file` or `--entry`; only reachable dependencies are selected when an entry file is used.
- Check `skip_dirs` and `ignore_paths`.
- Check `supported_extensions`.
- Check `max_file_size_kb`.

## License

This project is released under the MIT License. See [LICENSE](https://github.com/atharvm416/codedoc-ai/blob/main/LICENSE).
