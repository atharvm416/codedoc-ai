# codedoc-ai

`codedoc-ai` is a Python library and CLI that generates structured, reusable documentation memory for source codebases. It is built for AI coding agents, human maintainers, and teams that want a stable map of a project before making changes.

The tool scans source files, resolves project-local imports into a dependency graph, sends only files that need analysis to an LLM, and writes one combined, structured documentation artifact designed for both humans and AI. By default that artifact is JSON.

Current release: `0.6.1`.

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
- Stores incremental memory in `codedoc_db.json`.
- Uses SHA-256 content hashes as smart file IDs.
- Reuses cached analysis for unchanged files.
- Reuses cached analysis when another file has identical content.
- Recreates the selected output file from cache if the user deletes it.
- Embeds metadata (entry point, schema version) in every output file so the next run can resume without re-specifying the entry.
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
| API model | `gpt-4o-mini` |
| Output directory | `codedoc` |
| Output format | `json` |
| Output file | `codedoc/codedoc.json` |
| Parallel agents | `true` |
| Max parallel files | `5` |
| File retry attempts | `1` |
| Max consecutive failures | `5` |
| Change propagation | `true` |
| Max file size | `500 KB` |

Because the default provider uses the OpenAI API, a user must supply an API key unless they select a different provider.

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
- If no model is provided, `gpt-4o-mini` is used.
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

## A Word From Codex

From the perspective of a coding agent, `codedoc-ai` acts like a compact project memory layer. I still need to read the actual code before editing, but a current `codedoc.json` gives me a much better starting map: what files exist, what they own, how they depend on each other, what has changed, and where the risky parts probably are.

That matters because most agent time is spent on discovery before implementation. Without a project memory file, an agent has to repeatedly search, open, infer, and cross-check files. With `codedoc-ai`, the agent can first inspect a structured registry, then open only the files that are likely to matter.

Estimated impact for Codex, Claude, Copilot-style agents, and other AI coding tools:

| Area | Without `codedoc-ai` | With fresh `codedoc.json` | Practical improvement |
| --- | --- | --- | --- |
| Initial codebase understanding | Search-heavy and partial | Project map available immediately | 25-35% better first-pass understanding |
| Discovery token usage | High | Lower, because the agent reads a compact index first | 25-50% fewer discovery tokens on medium projects |
| Relevant file selection | Manual search and inference | Uses file roles, tree, folders, and dependency links | 30-60% faster targeting |
| Complex cross-file edits | More fragile | Better impact map through dependencies and `imported_by` links | 20-40% better edit reliability |
| Hallucination risk | Higher when file roles are guessed | Lower when roles, imports, and relationships are explicit | 20-35% lower risk with fresh output |
| Refactor safety | Depends on repeated manual tracing | Dependency graph and catalog guide impact checks | 25-45% better safety |
| New session onboarding | Starts nearly cold | Starts from reusable project memory | 40-70% faster orientation |

The most important caveat is freshness. `codedoc.json` should guide the agent, not replace code reading. The best workflow is:

1. Read `codedoc.json` first.
2. Use it to identify likely files, folders, symbols, and dependency paths.
3. Open the real source files before editing.
4. Make the change.
5. Run focused tests.
6. Regenerate `codedoc` output so the memory reflects the new code.

This is why the library favors structured JSON internally. AI-generated text can vary, but a stable machine-readable project view lets future agents and developers reason over the codebase with less repeated discovery.

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
  "skip_dirs": ["myenv", ".venv", "venv", "env", "node_modules", "__pycache__", "codedoc"],
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
| `CODEDOC_MAX_PARALLEL_FILES` | Maximum files processed at once. |
| `CODEDOC_FILE_RETRY_ATTEMPTS` | Sequential retry attempts for a failed file. |
| `CODEDOC_MAX_CONSECUTIVE_FAILURES` | Consecutive failure threshold before stopping. |
| `LOG_LEVEL` | `INFO`, `DEBUG`, etc. |
| `CODEDOC_IGNORE_PATHS` | Semicolon-separated ignore paths. |

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

`codedoc` writes public documentation and its incremental cache to the same output directory. Everything stays in one place — the project root is never written to.

Default output:

```text
codedoc/codedoc.json
codedoc/codedoc_db.json
```

JSON only:

```bash
codedoc run --format json
```

```text
codedoc/codedoc.json
codedoc/codedoc_db.json
```

Markdown only:

```bash
codedoc run --format md
```

```text
codedoc/codedoc.md
codedoc/codedoc_db.json
```

Custom output file name and location:

```bash
codedoc run --entry src/main.py --output project_docs/analysis.json
codedoc run --entry src/main.py --output project_docs/analysis.md
```

When a file path is passed to `--output`, the format is inferred from the extension — no need to also pass `--format`. Passing an unsupported extension (anything other than `.json` or `.md`) stops the run with a clear error.

### Metadata and Resume

Every generated file embeds a small metadata block that stores the entry point and schema version. This is how CodeDoc resumes documentation runs without asking for `--entry` a second time.

In JSON files the block is the first key in the document:

```json
{
  "_codedoc": {
    "entry_file": "src/main.py",
    "schema_version": "1.3",
    "generated_at": "2025-..."
  },
  ...
}
```

In Markdown files it is an HTML comment at the very top:

```text
<!-- codedoc-ai: {"entry_file": "src/main.py", "schema_version": "1.3", ...} -->
```

If this metadata is missing or corrupted, `codedoc` raises a clear error rather than silently failing. To recover, re-run with `--entry` to generate a fresh document.

### Cache Behaviour

All generated files — including `codedoc_db.json` — are stored in the output directory so the project root stays clean. On the first run after upgrading from an older version, any existing `codedoc_db.json` at the project root is automatically moved into the output directory.

The selected output format is authoritative for the default filenames (`codedoc.json` and `codedoc.md`). If a previous run wrote `codedoc.json` and the next run selects `md` only, the old `codedoc.json` is removed. Custom-named output files (e.g. `analysis.json`) are never automatically deleted between runs. If the selected output file is deleted, `codedoc` recreates it from `codedoc_db.json` when the cache is still valid.

The CLI logs the selected output format and the exact output file path during execution for better visibility.

`codedoc_db.json` stores:

- File path.
- File format.
- SHA-256 content hash.
- Last processed timestamp.
- Git commit and author when available (stored only in internal cache, not public output by default).
- Imports.
- Generated description and structure.
- Full cached documentation result.
- Processing history.

Keep `codedoc_db.json` ignored unless the team intentionally wants to version generated project memory.

The public `codedoc.json` and `codedoc.md` are cleaner than the cache. They include:

- Project overview (entry file, file count, languages).
- File tree representation.
- Folder-based grouping with summaries.
- Internal dependency graph between project files.
- Project-level dependency catalog with deduplicated dependency purpose.
- Flattened file summaries (no nested duplication).
- Imports, exports, functions, classes.
- Internal, external, and reverse dependencies (`imported_by`).

They exclude cache-specific data such as history, raw LLM responses, and author metadata by default.

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
        "external_dependencies": ["pydantic"]
      }
    }
  ]
}
```

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
6. Compute each selected file's SHA-256 hash.
7. Skip files whose path and hash already match the cache.
8. Reuse cached analysis if another file has the same content hash.
9. If `propagate_changes` is true, reprocess files that depend on changed files.
10. Send only remaining files to the selected LLM, up to `max_parallel_files` at a time.
11. Retry failed parallel files sequentially so errors are easier to diagnose.
12. Stop early if repeated failures suggest the API or provider is unavailable.
13. Update `codedoc_db.json` from the main pipeline path.
14. Rebuild the selected output file from cached records, embedding metadata for the next run.

This means repeated runs should only send new or changed code to the LLM. Unchanged code and exact duplicate content are reused.

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
| `--no-parallel` | `parallel_agents: False` |
| `--max-parallel-files` | `max_parallel_files` |
| `--verbose` | `log_level: "DEBUG"` |

## Troubleshooting

If API mode fails with an API key error:

- Set `OPENAI_API_KEY` for OpenAI models.
- Set `ANTHROPIC_API_KEY` for Claude models. Make sure model names start with `claude`.
- Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini models. Make sure model names start with `gemini`, or pass `--provider gemini`.

If many files fail quickly:

- Check `error.log`; `codedoc` records the file and failure context.
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
