# codedoc-ai

`codedoc-ai` is a local-first Python library and CLI that generates linked documentation for source codebases. It is designed for AI coding agents, human maintainers, and teams that want a reusable project memory before making changes.

The tool can start from a specific entry file, follow project-local imports into a dependency tree, analyze each reachable file with a configurable LLM, and write both human-readable Markdown and machine-readable JSON.

## Why This Exists

AI coding tools work better when they receive a clear map of the codebase before they edit it. `codedoc-ai` creates that map in a provider-neutral format so it can be used with Codex, Claude, GitHub Copilot, local LLMs, or custom automation.

## Features

- Entry-file traversal with dependency-first processing.
- Project-wide scanning when no entry file is supplied.
- Parsers for Python, React/TypeScript, JavaScript, Dart/Flutter, Java, C#, HTML, and common fallback languages.
- LLM provider abstraction for OpenAI-compatible APIs, Anthropic Claude, Ollama, LM Studio, and other local OpenAI-compatible servers.
- Incremental memory through `codedoc_db.json`, including file hashes, processing history, author, Git commit, imports, and generated summaries.
- One Markdown file and one JSON file per source file.
- `_index.json` for agent consumption and `_summary.md` for run summaries.
- Offline test-friendly architecture with deterministic parsers and mockable LLM providers.

## Installation

```bash
pip install codedoc-ai
```

For local development:

```bash
git clone https://github.com/atharvm416/codedoc-ai.git
cd codedoc-ai
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS/Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Quick Start

From the project you want to document:

```bash
codedoc . --entry src/main.py --output docs_output
```

Document the whole project:

```bash
codedoc .
```

Use a local LLM:

```bash
codedoc . --llm local --model qwen2.5-coder:7b
```

Use an API model:

```bash
set OPENAI_API_KEY=your-key
codedoc . --llm api --model gpt-4o-mini
```

Use Claude:

```bash
set ANTHROPIC_API_KEY=your-key
codedoc . --llm api --model claude-haiku-4-5-20251001
```

PowerShell users can set environment variables like this:

```powershell
$env:OPENAI_API_KEY="your-key"
```

## Configuration

Create `codedoc.config.json` in the project being documented:

```json
{
  "llm_mode": "local",
  "model_name": "qwen2.5-coder:7b",
  "api_base_url": "http://localhost:11434/v1",
  "entry_file": "src/main.py",
  "output_dir": "docs_output",
  "supported_extensions": [".py", ".ts", ".tsx", ".js", ".jsx", ".dart", ".java", ".cs", ".html"],
  "parallel_agents": false,
  "max_file_size_kb": 500,
  "propagate_changes": true
}
```

Secrets should live in environment variables or a local `.env` file that is ignored by Git. Use [.env.example](.env.example) as the template.

## Output

`codedoc` writes:

- `docs_output/<file>.md`: readable documentation for each file.
- `docs_output/<file>.json`: structured metadata for AI tools.
- `docs_output/_index.json`: machine-readable index linking generated docs.
- `docs_output/_summary.md`: run summary and errors.
- `codedoc_db.json`: local incremental memory. Keep it ignored unless your team intentionally wants to version it.

## Python API

```python
from codedoc import run_pipeline

stats = run_pipeline(".", {
    "entry_file": "src/main.py",
    "llm_mode": "local",
    "model_name": "qwen2.5-coder:7b",
})

print(stats)
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Publishing Your Public Repo

Before publishing your own repository:

- Confirm the `Homepage`, `Issues`, and `authors` fields in `pyproject.toml`.
- Confirm the package name `codedoc-ai` is still available to you on PyPI.
- Run:

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
