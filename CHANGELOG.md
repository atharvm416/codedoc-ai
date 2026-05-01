# Changelog

## 0.1.1 - 2026-05-01

- Added safer default scanning for virtual environments such as `myenv`.
- Added configurable `skip_dirs`.
- Added strict project-relative ignore paths through CLI, config, environment, and Python API.
- Added `--ignore PATH` CLI option.
- Added scanner tests for virtual environment and strict path ignores.
- Fixed misleading API key warning when CLI overrides select local LLM mode.

## 0.1.0 - 2026-05-01

- Initial alpha release.
- Added entry-file dependency traversal.
- Added local and API LLM provider support.
- Added per-file Markdown and JSON output.
- Added `_index.json`, `_summary.md`, and incremental `codedoc_db.json` memory.
- Added CLI and Python API entry points.
