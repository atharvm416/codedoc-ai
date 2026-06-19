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

## Supported Python versions

`codedoc-ai` supports Python 3.10, 3.11, and 3.12. The declared
`requires-python` range, the package classifiers, and the CI test matrix are
kept in agreement; CI tests every interpreter the package claims to support.

## Verification

Run these before opening a pull request:

```bash
python -m pytest
python -m ruff check codedoc
python -m ruff check tests/test_095_*.py tests/test_093_dependency_view.py tests/test_graph.py tests/test_agents.py tests/test_080_features.py tests/test_097_*.py
python -m build
python -m twine check dist/*
```

Continuous integration (`.github/workflows/ci.yml`) runs the full test suite on
Python 3.10–3.12, Ruff over production and the release-touched tests, and a
packaging job that builds the sdist and wheel, runs `twine check`, installs the
wheel into a clean environment, and smoke-tests `import codedoc`,
`codedoc --version`, and `codedoc --help`. The older test files still have known
Ruff findings, so the lint gate is intentionally scoped until those are cleaned
in a dedicated behavior-free change. CI never publishes, makes paid provider
calls, or requires provider secrets.

## Run lifecycle

The verified phase ordering of a run — read-only preflight, scan/plan, the
mutation boundary, live-backup initialization before provider creation,
execution, atomic finalization, diagnostics, and cleanup — is documented in
[`RUN_FLOW.md`](RUN_FLOW.md).

## Good First Contributions

- Add parser fixtures for more frameworks.
- Improve import resolution for a specific language.
- Add tests for edge cases in dependency traversal.
- Improve documentation examples.
- Add provider support for another OpenAI-compatible local server.

## Module Size and Cohesion

File size is a review *signal*, not a hard gate. Production modules should
normally stay cohesive and under roughly 700 lines; a module that grows well
past that is usually doing too many jobs and is a good candidate for
extraction into single-responsibility modules (as the pipeline and the
project-view serializer were split in 0.9.4).

There is no CI line-count rule, and you should never split code merely to hit
a number. Generated files, large data tables, parsers, and tightly coupled
serializers may legitimately justify more lines. Prefer splitting along clear
responsibilities (and one-way import boundaries) rather than by line count.

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
