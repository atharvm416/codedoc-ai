# codedoc-ai

`codedoc-ai` is a local-first Python library and CLI that generates structured, reusable documentation memory for source codebases. It is built for AI coding agents, human maintainers, and teams that want a stable map of a project before making changes.

The tool scans source files, resolves project-local imports into a dependency graph, sends only files that need analysis to an LLM, and writes one combined, structured documentation artifact designed for both humans and AI. By default that artifact is JSON.

## What It Does

- Finds supported source files in a project.
- Starts from an explicit entry file when provided.
- Otherwise auto-detects common entry files such as `main.py`, `main.tsx`, `index.html`, `Main.java`, and related names.
- If an entry file is found, documents that file and its reachable project dependencies.
- If no entry file is found, documents all supported project files.
- Parses imports locally before calling an LLM.
- Processes dependencies before dependent files where possible.
- Stores incremental memory in `codedoc_db.json`.
- Uses SHA-256 content hashes as smart file IDs.
- Reuses cached analysis for unchanged files.
- Reuses cached analysis when another file has identical content.
- Recreates the selected output file from cache if the user deletes it.
- Writes a clean, structured public project view to `docs_output/codedoc.json` by default, or Markdown when requested.
- Public output includes project overview, file tree, folder map, dependency graph, and flattened file summaries.

## Defaults

If the user runs:

```bash
codedoc .
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
| Change propagation | `true` |
| Max file size | `500 KB` |

Because default `llm_mode` is `api`, a user must provide an API key unless they choose local mode.

## Installation

Install from PyPI:

```bash
pip install codedoc-ai
```

## Quick Start

Document the current project using the default API model and JSON output:

```bash
codedoc .
```

Document from a known entry file:

```bash
codedoc . --entry src/main.py
```

Write output to a custom directory:

```bash
codedoc . --output docs_output
```

Write Markdown instead of JSON:

```bash
codedoc . --format md
```

Write both JSON and Markdown:

```bash
codedoc . --format both
```

## Choosing an LLM

Use this rule of thumb:

| Use case | Recommended mode |
| --- | --- |
| Best default quality with minimal setup | OpenAI API |
| Claude-specific documentation style or Anthropic account | Anthropic API |
| No cloud calls, private code, or offline workflows | Local LLM |
| OpenAI-compatible gateway such as LM Studio, Ollama, LiteLLM, or a custom endpoint | Local mode or API mode with `api_base_url` |

Provider selection is deterministic:

- `llm_mode = "local"` always uses the local OpenAI-compatible provider.
- `llm_mode = "api"` with a model name starting with `claude` uses Anthropic.
- `llm_mode = "api"` with any other model uses OpenAI/OpenAI-compatible APIs.
- If no model is provided in API mode, `gpt-4o-mini` is used.
- If no model is provided in local mode, `qwen2.5-coder:7b` is used.

## OpenAI API Setup

Use OpenAI when you want the default hosted API path.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-your-openai-key"
codedoc . --llm api --model gpt-4o-mini
```

Windows Command Prompt:

```bat
set OPENAI_API_KEY=sk-your-openai-key
codedoc . --llm api --model gpt-4o-mini
```

macOS/Linux:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
codedoc . --llm api --model gpt-4o-mini
```

OpenAI-compatible API example:

```bash
codedoc . --llm api --model your-model-name
```

For compatible APIs, set `api_base_url` in `codedoc.config.json` or `API_BASE_URL` in `.env`.

## Anthropic API Setup

Use Anthropic by choosing a Claude model name. The model name must start with `claude` so `codedoc` can select the Anthropic provider.

Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
codedoc . --llm api --model claude-haiku-4-5-20251001
```

Windows Command Prompt:

```bat
set ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
codedoc . --llm api --model claude-haiku-4-5-20251001
```

macOS/Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
codedoc . --llm api --model claude-haiku-4-5-20251001
```

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
codedoc . --llm local --model qwen2.5-coder:7b
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
codedoc . --llm local --model your-loaded-model
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

## Configuration

Create `codedoc.config.json` in the project being documented:

```json
{
  "llm_mode": "api",
  "model_name": "gpt-4o-mini",
  "api_base_url": null,
  "entry_file": null,
  "output_dir": "docs_output",
  "output_format": "json",
  "supported_extensions": [".py", ".ts", ".tsx", ".js", ".jsx", ".dart", ".java", ".cs", ".html"],
  "parallel_agents": true,
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

## Environment Variables

Secrets should live in environment variables or a local `.env` file that is ignored by Git. Use [.env.example](.env.example) as the template.

Supported variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key. |
| `ANTHROPIC_API_KEY` | Anthropic API key. |
| `LLM_API_KEY` | Generic fallback API key. |
| `LLM_MODE` | `api` or `local`. |
| `MODEL_NAME` | Model name to use. |
| `API_BASE_URL` | OpenAI-compatible base URL. |
| `OUTPUT_DIR` | Output directory. |
| `CODEDOC_OUTPUT_FORMAT` | `json`, `md`, or `both`. |
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
MODEL_NAME=claude-haiku-4-5-20251001
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
codedoc . --entry main.py --ignore /myenv --ignore services/generated
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
codedoc . --format md
```

```text
docs_output/codedoc.md
codedoc_db.json
```

Both formats:

```bash
codedoc . --format both
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
- Flattened file summaries (no nested duplication).
- Imports, exports, functions, classes.
- Internal, external, and reverse dependencies (`imported_by`).

They exclude cache-specific data such as history, raw LLM responses, and author metadata by default.

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
9. Send only remaining files to the selected LLM.
10. Update `codedoc_db.json`.
11. Rebuild the selected output file from cached records.

This means repeated runs should only send new or changed code to the LLM. Unchanged code and exact duplicate content are reused.

## Python API

```python
from codedoc import run_pipeline

stats = run_pipeline(".", {
    "entry_file": "src/main.py",
    "llm_mode": "local",
    "model_name": "qwen2.5-coder:7b",
    "api_base_url": "http://localhost:11434/v1",
    "parallel_agents": False,
    "output_dir": "docs_output",
    "output_format": "json",
    "ignore_paths": ["/myenv", "services/generated"],
})

print(stats)
```

## Troubleshooting

If API mode fails with an API key error:

- Set `OPENAI_API_KEY` for OpenAI models.
- Set `ANTHROPIC_API_KEY` for Claude models.
- Make sure Claude model names start with `claude`.

If local mode fails:

- Confirm the local server is running.
- Confirm the `api_base_url` points to an OpenAI-compatible `/v1` endpoint.
- For Ollama, use `http://localhost:11434/v1`.
- For LM Studio, commonly use `http://localhost:1234/v1`.
- Try `parallel_agents: false` for smaller local models.

If files are missing from output:

- Check `entry_file` or `--entry`; only reachable dependencies are selected when an entry file is used.
- Check `skip_dirs` and `ignore_paths`.
- Check `supported_extensions`.
- Check `max_file_size_kb`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
