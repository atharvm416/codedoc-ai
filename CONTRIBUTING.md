# Contributing

Thanks for helping improve `codedoc-ai`.

## How Contributions Work

1. Fork the repository.
2. Create a branch from `main`.
3. Make your changes.
4. Add or update tests when behavior changes.
5. Run the verification commands.
6. Open a pull request with a clear description.

## Development Setup

```bash
git clone https://github.com/atharvm416/codedoc-ai.git
cd codedoc-ai
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Verification

Run these before opening a pull request:

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

## Good First Contributions

- Add parser fixtures for more frameworks.
- Improve import resolution for a specific language.
- Add tests for edge cases in dependency traversal.
- Improve documentation examples.
- Add provider support for another OpenAI-compatible local server.

## Pull Request Guidelines

- Keep changes focused.
- Explain why the change is needed.
- Include before/after behavior when fixing a bug.
- Do not commit `.env`, API keys, `docs_output`, `codedoc_db.json`, `dist`, or virtual environments.
- Keep generated artifacts out of pull requests unless a maintainer asks for them.

## Reporting Bugs

Please include:

- Your operating system.
- Python version.
- `codedoc-ai` version.
- The command you ran.
- Relevant config without secrets.
- A small reproduction project or fixture when possible.
