# codedoc-ai

`codedoc-ai` is a local-first Python library and CLI that generates structured, reusable documentation memory for source codebases. It is built for AI coding agents, human maintainers, and teams that want a stable map of a project before making changes.

The tool scans source files, resolves project-local imports into a dependency graph, sends only files that need analysis to an LLM, and writes one combined, structured documentation artifact designed for both humans and AI. By default that artifact is JSON.

Current release: `0.5.0`.

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
- Writes a clean, structured public project view to `docs_output/codedoc.json` by default, or Markdown when requested.
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
| LLM mode | `api` |
| API model | `gpt-4o-mini` |
| Output directory | `docs_output` |
| Output format | `json` |
| Output file | `docs_output/codedoc.json` |
| Parallel agents | `true` |
| Max parallel files | `5` |
| File retry attempts | `1` |
| Max consecutive failures | `5` |
| Change propagation | `true` |
| Max file size | `500 KB` |

Because default `llm_mode` is `api`, a user must provide an API key unless they choose local mode.

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

Document the current project using the default API model and JSON output:

```bash
codedoc run
```

Document from a known entry file:

```bash
codedoc run --entry src/main.py
```

Write output to a custom directory:

```bash
codedoc run --output docs_output
```

Write Markdown instead of JSON:

```bash
codedoc run --format md
```

Write both JSON and Markdown:

```bash
codedoc run --format both
```

Limit file-level concurrency:

```bash
codedoc run --max-parallel-files 3
```

Use this when an API has strict rate limits, or when a local model cannot comfortably handle 5 files in flight.

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
| `codedoc run` | Document the current directory with default JSON output. |
| `codedoc execute` | Alias for `codedoc run`. |
| `codedoc run --format md` | Write only `docs_output/codedoc.md`. |
| `codedoc run --format both` | Write both JSON and Markdown. |
| `codedoc run --entry src/main.py` | Start from a known entry file. |
| `codedoc run --output docs` | Write public output to `docs`. |
| `codedoc run --llm local --model qwen2.5-coder:7b` | Use a local OpenAI-compatible model. |
| `codedoc run --provider gemini --model gemini-2.5-flash` | Use Google Gemini. |
| `codedoc run --ignore /myenv --ignore generated` | Ignore project paths. |
| `codedoc run --max-parallel-files 3` | Limit concurrent file processing. |
| `codedoc .` | Legacy shorthand for documenting the current directory. |
| `codedoc --version` | Print the installed version. |

## Choosing an LLM

Use this rule of thumb:

| Use case | Recommended mode |
| --- | --- |
| Best default quality with minimal setup | OpenAI API |
| Claude-specific documentation style or Anthropic account | Anthropic API |
| Google AI Studio / Gemini account | Gemini API |
| No cloud calls, private code, or offline workflows | Local LLM |
| OpenAI-compatible gateway such as LM Studio, Ollama, LiteLLM, or a custom endpoint | Local mode or API mode with `api_base_url` |

Provider selection is deterministic:

- `llm_mode = "local"` always uses the local OpenAI-compatible provider.
- `llm_provider = "openai"` uses OpenAI/OpenAI-compatible APIs.
- `llm_provider = "anthropic"` uses Anthropic Claude.
- `llm_provider = "gemini"` uses Google Gemini through the official `google-genai` SDK.
- `llm_provider = "auto"` with a model name starting with `claude` uses Anthropic.
- `llm_provider = "auto"` with a model name starting with `gemini` uses Gemini.
- `llm_provider = "auto"` with any other model uses OpenAI/OpenAI-compatible APIs.
- If no model is provided in API mode, `gpt-4o-mini` is used.
- If Gemini is selected and no model is provided, `gemini-2.5-flash` is used.
- If no model is provided in local mode, `qwen2.5-coder:7b` is used.

## OpenAI API Setup

Use OpenAI when you want the default hosted API path.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-your-openai-key"
codedoc run --llm api --model gpt-4o-mini
```

Windows Command Prompt:

```bat
set OPENAI_API_KEY=sk-your-openai-key
codedoc run --llm api --model gpt-4o-mini
```

macOS/Linux:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
codedoc run --llm api --model gpt-4o-mini
```

OpenAI-compatible API example:

```bash
codedoc run --llm api --model your-model-name
```

For compatible APIs, set `api_base_url` in `codedoc.config.json` or `API_BASE_URL` in `.env`.

## Anthropic API Setup

Use Anthropic by choosing a Claude model name. The model name must start with `claude` so `codedoc` can select the Anthropic provider.

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
codedoc run --llm api --model claude-haiku-4-5-20251001
```

Windows Command Prompt:

```bat
set ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
codedoc run --llm api --model claude-haiku-4-5-20251001
```

macOS/Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
codedoc run --llm api --model claude-haiku-4-5-20251001
```

## Gemini API Setup

Use Gemini when you want Google's hosted Gemini models. Set `llm_provider` to `gemini`, or use a Gemini model name with `llm_provider` left as `auto`.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
codedoc run --llm api --provider gemini --model gemini-2.5-flash
```

Windows Command Prompt:

```bat
set GEMINI_API_KEY=your-gemini-api-key
codedoc run --llm api --provider gemini --model gemini-2.5-flash
```

macOS/Linux:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
codedoc run --llm api --provider gemini --model gemini-2.5-flash
```

`GOOGLE_API_KEY` is also supported as an alias for `GEMINI_API_KEY`.

## Local LLM Setup

Use local mode when code should stay on the machine or when the user is running Ollama, LM Studio, llama.cpp server, or another OpenAI-compatible local server.

### Ollama

Start Ollama and pull a coding model.


```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

In another terminal:

```bash
codedoc run --llm local --model qwen2.5-coder:7b
```

Default Ollama URL:

```text
http://localhost:11434/v1
```

### LM Studio

In LM Studio, start the local server with an OpenAI-compatible endpoint. The common base URL is:

```text
http://localhost:1234/v1
```

Then run:

```bash
codedoc run --llm local --model your-loaded-model
```

Set the base URL in config:

```json
{
  "llm_mode": "local",
  "model_name": "your-loaded-model",
  "api_base_url": "http://localhost:1234/v1"
}
```

For local LLMs, set `parallel_agents` to `false` if the model or GPU has limited memory.

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
  "llm_mode": "api",
  "llm_provider": "auto",
  "model_name": "gpt-4o-mini",
  "api_base_url": null,
  "entry_file": null,
  "output_dir": "docs_output",
  "output_format": "json",
  "supported_extensions": [".py", ".ts", ".tsx", ".js", ".jsx", ".dart", ".java", ".cs", ".html"],
  "parallel_agents": true,
  "max_parallel_files": 5,
  "file_retry_attempts": 1,
  "max_consecutive_failures": 5,
  "log_level": "INFO",
  "max_file_size_kb": 500,
  "propagate_changes": true,
  "skip_dirs": ["myenv", ".venv", "venv", "env", "node_modules", "__pycache__", "docs_output"],
  "ignore_paths": ["/myenv", "services/generated"]
}
```

Configuration precedence, from strongest to weakest:

1. CLI flags, such as `--model`, `--llm`, `--format`, and `--output`.
2. Environment variables and values loaded from `.env`.
3. `codedoc.config.json` or `config.json`.
4. Built-in defaults.

Supported output formats:

| Value | Result |
| --- | --- |
| `json` | Writes only `docs_output/codedoc.json`. This is the default. |
| `md` | Writes only `docs_output/codedoc.md`. |
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
| `LLM_MODE` | `api` or `local`. |
| `LLM_PROVIDER` | `auto`, `openai`, `anthropic`, or `gemini`. |
| `MODEL_NAME` | Model name to use. |
| `API_BASE_URL` | OpenAI-compatible base URL. |
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
LLM_MODE=api
MODEL_NAME=gpt-4o-mini
CODEDOC_OUTPUT_FORMAT=json
```

Example `.env` for Anthropic:

```text
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
LLM_MODE=api
LLM_PROVIDER=anthropic
MODEL_NAME=claude-haiku-4-5-20251001
CODEDOC_OUTPUT_FORMAT=json
```

Example `.env` for Gemini:

```text
GEMINI_API_KEY=your-gemini-api-key
LLM_MODE=api
LLM_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
CODEDOC_OUTPUT_FORMAT=json
```

Example `.env` for Ollama:

```text
LLM_MODE=local
MODEL_NAME=qwen2.5-coder:7b
API_BASE_URL=http://localhost:11434/v1
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

`codedoc` writes public documentation to the selected output directory and private incremental memory to the project root.

Default output:

```text
docs_output/codedoc.json
codedoc_db.json
```

Markdown output:

```bash
codedoc run --format md
```

```text
docs_output/codedoc.md
codedoc_db.json
```

Both formats:

```bash
codedoc run --format both
```

```text
docs_output/codedoc.json
docs_output/codedoc.md
codedoc_db.json
```

The selected output format is authoritative. If a previous run wrote Markdown and the next run selects JSON, the old Markdown output is removed. If the selected output file is deleted, `codedoc` recreates it from `codedoc_db.json` when the cache is still valid.

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
2. Scan supported files while respecting `skip_dirs` and `ignore_paths`.
3. Build a dependency graph from parsed imports.
4. Select files from `--entry`, `entry_file`, auto-detected entry, or all files.
5. Compute each selected file's SHA-256 hash.
6. Skip files whose path and hash already match the cache.
7. Reuse cached analysis if another file has the same content hash.
8. If `propagate_changes` is true, reprocess files that depend on changed files.
9. Send only remaining files to the selected LLM, up to `max_parallel_files` at a time.
10. Retry failed parallel files sequentially so errors are easier to diagnose.
11. Stop early if repeated failures suggest the API or provider is unavailable.
12. Update `codedoc_db.json` from the main pipeline path.
13. Rebuild the selected output file from cached records.

This means repeated runs should only send new or changed code to the LLM. Unchanged code and exact duplicate content are reused.

## Python API

The CLI is not required. You can run the same workflow from Python with `run_pipeline(...)`.

For the current working directory, pass only the config dict:

```python
from codedoc import run_pipeline

stats = run_pipeline({
    "entry_file": "src/main.py",
    "llm_mode": "local",
    "llm_provider": "auto",
    "model_name": "qwen2.5-coder:7b",
    "api_base_url": "http://localhost:11434/v1",
    "parallel_agents": False,
    "max_parallel_files": 2,
    "file_retry_attempts": 1,
    "output_dir": "docs_output",
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
| `--llm` | `llm_mode` |
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
- Set `ANTHROPIC_API_KEY` for Claude models.
- Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini models.
- Make sure Claude model names start with `claude`.
- Make sure Gemini model names start with `gemini`, or pass `--provider gemini`.

If local mode fails:

- Confirm the local server is running.
- Confirm the `api_base_url` points to an OpenAI-compatible `/v1` endpoint.
- For Ollama, use `http://localhost:11434/v1`.
- For LM Studio, commonly use `http://localhost:1234/v1`.
- Try `parallel_agents: false` for smaller local models.
- Lower `max_parallel_files` if the model or server cannot handle concurrent files.

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
