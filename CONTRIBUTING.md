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

Continuous integration (`.github/workflows/ci.yml`) runs three review-local test
selections on Python 3.10-3.12: unit plus meta, integration, and contract plus
end-to-end. A separate platform matrix covers Linux on all supported Python
versions and Windows and macOS on Python 3.12. Successful Python 3.12 test rows
contribute data to one combined coverage report. Ruff still checks the complete
repository, and the packaging job builds and validates distributions and
smoke-tests an installed wheel. CI never publishes, makes paid provider calls,
or requires provider secrets.

## Test Suite Conventions

The target test tree is organized by ownership and execution environment:

```text
tests/
+-- unit/          # isolated behavior of one component
+-- integration/   # collaboration among repository components
+-- contract/      # stable public, configuration, output, and provider contracts
+-- e2e/           # complete user-visible workflows
+-- platform/      # filesystem- or OS-sensitive behavior
+-- meta/          # structural checks over the suite itself
+-- support/       # behavior-named helpers shared by two or more test modules
`-- fixtures/      # inert test data, including immutable goldens
```

Every directory containing a test module carries an `__init__.py`. This gives
each test module a unique import identity, prevents same-basename collisions,
and makes collection behavior consistent across pytest import modes (T1).

Run the same targeted selections used by CI with:

```bash
python -m pytest tests/unit tests/meta
python -m pytest tests/integration
python -m pytest tests/contract tests/e2e
python -m pytest tests/platform
```

`python -m pytest` remains the canonical local full-suite command.

- No collected test module imports another collected `test_*.py` module.
- A helper used by one test module stays local to that module. A behavior-named
  module moves to `tests/support/` only when at least two collected test modules
  import it; `tests/conftest.py` remains limited to pytest hooks and genuinely
  shared fixtures.
- Golden files live in `tests/fixtures/documents/golden/` and are never
  regenerated merely to make a failing test pass (T4). Review an intentional
  contract change and its golden diff together.
- Existing logical pytest node identities remain stable even though their full
  node paths move to the mapped ownership targets. New tests use descriptive
  feature-oriented module and test names, never release-number filenames.
- Retry, backoff, and rate-limit tests use `tests/support/clocks.py` to record
  requested delays. Direct `time.sleep` is limited to the three explicitly
  guarded concurrency-coordination modules.
- The registered `platform` marker belongs at module level on every test module
  under `tests/platform/` and nowhere else. Use
  `python -m pytest -m "not platform"` on a restricted filesystem.

`tests/meta/test_suite_architecture.py` enforces these rules structurally from
paths and ASTs without importing collected test modules.

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
