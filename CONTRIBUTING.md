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
python -m ruff check .
python -m build
python -m twine check dist/*
```

Continuous integration (`.github/workflows/ci.yml`) runs the full test suite on
Python 3.10–3.12, Ruff over the complete repository, and a
packaging job that builds the sdist and wheel, runs `twine check`, installs the
wheel into a clean environment, and smoke-tests `import codedoc`,
`codedoc --version`, and `codedoc --help`. CI never publishes, makes paid
provider calls, or requires provider secrets.

## Test Suite Conventions

- **No imports between collected test modules.** A file matching
  `tests/test_*.py` must never import another `tests/test_*.py` module.
  Reusable fakes, profile constants, and clock helpers live in
  `tests/support/`, which is never collected by pytest.
- **A helper needs a real importer before it moves to `tests/support/`.** A
  helper used by only one caller belongs in that caller's own test module.
  `tests/conftest.py` stays limited to pytest hooks and genuinely shared
  fixtures, not generic importable helpers.
- **Retry, backoff, and rate-limit tests use `tests/support/clocks.py`.** Its
  `capture_sleeps(monkeypatch, target)` helper replaces the named sleep
  callable with a recorder and asserts the requested delay sequence instead of
  waiting for it. Direct `time.sleep` in a test is reserved for the three
  concurrency-coordination sites the structural guard allowlists.
- **The `platform` marker** flags filesystem- or OS-sensitive tests (for
  example `tests/test_096_scan_symlinks.py`). Run
  `python -m pytest -m "not platform"` to skip them on a restricted
  filesystem.
- `tests/test_suite_architecture.py` enforces all of the above structurally
  (no test-to-test imports, no collected file or `Test*` class under
  `tests/support/`, the `time.sleep` allowlist, and that every registered
  marker is documented here).

## Run lifecycle

The verified phase ordering of a run — read-only preflight, scan/plan, the
mutation boundary, crash-recovery initialization before provider creation,
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
project-view serializer were split into separate modules).

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
