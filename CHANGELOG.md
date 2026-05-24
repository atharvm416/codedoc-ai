# Changelog

## 0.7.0 - 2026-05-24

### MD-only incremental now works (Issue 1)
- `_build_meta_comment` now embeds a `file_hashes` dict inside the `<!-- codedoc-ai: ... -->` metadata comment written at the top of every `codedoc.md`. Each entry maps a relative file path to its SHA-256 hash.
- `_load_existing_file_docs` now falls back to the MD file when no JSON exists. It reads hashes from the metadata comment and file records from the parsed MD content. Users who only ever run `--format md` no longer pay full LLM cost on every run.
- MD files generated before 0.7.0 have no `file_hashes`; the first 0.7.0 run re-processes everything once, then subsequent runs are incremental.
- Zero extra files: MD-only output remains a single file.

### Cross-format resume (Issue 2)
- `_resolve_entry_and_docs` now checks for a same-stem `.md` sibling when a `.json` candidate does not exist (e.g. `--output codedoc/claude.json` after a previous run wrote `codedoc/claude.md`).
- `_load_existing_file_docs` checks the same-stem MD sibling before falling back to the configured MD filename.

### Warning when entry file not in scanned set (Issue 3)
- `_select_files` now logs a `WARNING` when the entry file exists on disk but is absent from the scanner's file map (unsupported extension, too large, in a skip directory).

### Removed dead `write_outputs` function (Issue 4)
- `codedoc/core/output.py`: removed the never-called `write_outputs()` backward-compat wrapper that still referenced removed fields (`id`, `format`, `last_processed`, `git_commit`, `author`). Unused `datetime`/`timezone` imports also removed.

### `--format both` with a named file is now a hard error (Issue 5)
- `_resolve_output_spec` raises `ConfigError` when `output_format` is `"both"` and a named file path is given. Previously this silently downgraded to a single format. The error message directs developers to use a directory path instead.

### Tests
- Added 5 regression tests covering all fixes above.

## 0.6.4 - 2026-05-24

- Removed `codedoc_db.json` entirely — the public `codedoc.json` output already stores `hash` per file, which is sufficient for incremental processing.
- Hash-based incremental check now compares `compute_file_hash(path)` against `existing_docs[rel].get("hash")` from the public JSON, replacing the DB lookup.
- Added `_deps` field per file in the public JSON: stores the raw `dependencies_analysis` dict so the dependency catalog can be fully rebuilt from unchanged files on the next incremental run without an LLM call. Not rendered in Markdown output.
- `_public_record_to_doc` now reads `_deps` back and sets it as `dependencies_analysis`; falls back to `links.external_dependencies` for old-format JSON files.
- No per-file checkpoint writes during a run — crash recovery now means re-running the affected files.
- Legacy cleanup: if `codedoc_db.json` exists in the output directory at run time, it is deleted and a log message is emitted.
- `codedoc/core/db.py` stripped to just the `compute_file_hash` utility; `CodeDocDB` class removed.

## 0.6.3 - 2026-05-24

- Trimmed `codedoc_db.json` to the minimum needed for incremental runs:
  - Removed `history` array entirely — every field it contained (`file_path`, `processed_at`, `hash`, `author`) was already present in the `files` section, making it pure duplication. It was also never read anywhere in the pipeline.
  - Removed `author` and `git_commit` fields from per-file DB entries — no longer stored in any output since 0.6.2, so they served no purpose in the cache.
  - Removed git subprocess calls (`git rev-parse`, `git config user.name`) from the DB write path — nothing reads their output anymore, so there is no reason to shell out on every file write.
- Each DB entry now contains only: `hash`, `last_processed`, and (when present) `dependencies_analysis`.
- Existing `codedoc_db.json` files with the old format are migrated transparently on the next run (history is silently dropped).

## 0.6.2 - 2026-05-23

- Cleaned public output for better AI scannability (schema version 1.4):
  - Removed `id` field per file (always identical to `hash` — pure duplication).
  - Removed `last_processed` field per file (internal processing timestamp, not documentation content).
  - Removed `state` field per file (always `"checked"` in public output — carries no signal).
  - Removed `format` field per file (file extension is already in `path`; `language` covers the language name).
  - Result: each file record is smaller and contains only documentation-relevant content.
- Markdown output no longer renders `**ID:**` or `**Format:**` header lines per file.

## 0.6.1 - 2026-05-23

- Improved run logging:
  - Replaced animated file progress bars with stable log lines.
  - Logs now show provider/model, configured file concurrency, file start events, completion percentage, and remaining file count.
  - Format switches now log when an unselected public output file is removed.
  - Parallel file processing is now visible in log output.
  - Internal agent processing events demoted to debug level to reduce noise.

## 0.6.0 - 2026-05-23

- Added metadata-backed reruns:
  - JSON output now includes a top-level `_codedoc` metadata block.
  - Markdown output now includes a hidden `codedoc-ai` metadata comment.
  - Stored metadata includes the entry file, schema version, and generation time.
  - Subsequent runs can recover the entry file from a previously generated `.json` or `.md` documentation file.
- Changed first-run/resume behavior:
  - First runs require an explicit entry file when no valid previous CodeDoc output is available.
  - If no output path is provided, CodeDoc checks the default `codedoc/` folder for previous docs.
  - Invalid or metadata-free documentation files now fail clearly instead of being treated as valid resume sources.
- Changed default generated output location from `docs_output/` to `codedoc/`.
- Kept JSON as the default public output format.
- Added support for output file paths:
  - `--output docs/report.json` writes a named JSON file.
  - `--output docs/report.md` writes a named Markdown file.
  - File extension now determines the selected output format for explicit file paths.
  - Unsupported output file extensions now raise a configuration error.
- Moved the incremental cache into the selected output directory:
  - `codedoc_db.json` is now stored beside generated docs.
  - Existing root-level `codedoc_db.json` files are migrated into the output directory when possible.
- Improved output cleanup:
  - Default managed files (`codedoc.json`, `codedoc.md`) are removed when switching formats.
  - Legacy per-file outputs such as `main.py.json` and `main.py.md` are cleaned up.
  - Custom-named output files are preserved across runs.
- Simplified provider mode support for this release:
  - Active providers are OpenAI/OpenAI-compatible, Anthropic, and Gemini.
  - Local provider code remains in the package but is not exposed through the CLI/factory in 0.6.0.
  - Removed `--llm` / `LLM_MODE` from the documented public workflow.
- Improved provider implementations:
  - Reused Anthropic clients instead of creating a client per request.
  - Added native JSON-mode handling for OpenAI and Gemini where available.
  - Improved Gemini system-instruction handling.
- Updated CLI help, README, and version metadata for the 0.6.0 workflow.
- Added regression coverage for:
  - Missing entry plus missing docs raising a clear configuration error.
  - Resuming from existing JSON metadata.
  - Custom output filename behavior.
  - JSON remaining the default format.
  - Cache/output cleanup and metadata preservation.

## 0.5.2 - 2026-05-13

- Fixed cache structure duplication issues in generated documentation output.
- Improved dependency/import resolution to prevent incorrect file mappings and false dependency relationships.
- Cleaned and normalized public dependency output generation.
- Reduced noisy dependency cycles in generated Markdown and JSON outputs.
- Added regression coverage for cache structure and dependency resolution behavior.

## 0.5.1 - 2026-05-13

- Cleaned generated cache and public JSON by pruning empty arrays, empty objects, nulls, and duplicate nested fields.
- Removed the top-level cache `version` field from newly written `codedoc_db.json`.
- Improved Markdown-to-JSON conversion so it no longer recreates empty default sections.
- Tightened agent prompts to avoid placeholder package names and empty output fields.

## 0.5.0 - 2026-05-13

- Promoted `codedoc-ai` to the 0.5.0 feature line.
- Added bounded file-level parallelism:
  - Processes up to 5 files at a time by default.
  - Adds `--max-parallel-files N` for CLI control.
  - Adds `max_parallel_files`, `file_retry_attempts`, and `max_consecutive_failures` config options.
- Added sequential retry fallback for files that fail during parallel execution.
- Added provider/API health diagnostics when repeated file processing failures suggest bad credentials, rate limits, model errors, network issues, or provider downtime.
- Kept cache writes ordered and centralized so `codedoc_db.json` remains structured even when files are processed concurrently.
- Added AI-friendly dependency cataloging:
  - File-level dependencies remain on each file.
  - AI can suggest `catalog_updates` internally.
  - Public output receives a merged `dependency_catalog`.
  - Repeated dependency explanations are deduplicated across JSON and Markdown.
- Added deterministic JSON/Markdown conversion helpers so public JSON can become Markdown without another AI call, and generated Markdown can be parsed back into the public JSON shape.
- Clarified DependencyAgent output so generic import notes stay out of repeated file records unless they are file-specific.
- Added Google Gemini support through the official `google-genai` SDK.
- Added `llm_provider` config and `--provider auto|openai|anthropic|gemini` CLI selection.
- Expanded README with Codex/AI-agent analysis covering token savings, hallucination reduction, complex edit safety, and recommended workflows.
- Added tests for:
  - File-level parallel processing.
  - Retry behavior.
  - Dependency catalog output.
  - JSON/Markdown conversion.
  - Format switching from cache.

## 0.1.4 - 2026-05-02

- Redesigned **public output structure** for cleaner, AI-friendly documentation.
- Separated **internal cache (`codedoc_db.json`)** from **public output (`codedoc.json` / `codedoc.md`)**.
- Added **project-level overview** including entry file, file count, languages, and folder summary.
- Added **project tree visualization** in both JSON and Markdown outputs.
- Added **folder-based grouping** with summarized purpose and file listings.
- Introduced **dependency graph** with internal file relationships and external dependencies.
- Flattened file structure in public output:
  - Removed nested and duplicated `result` / `documentation` blocks.
  - Consolidated descriptions, roles, functions, classes, and exports into a single clean structure.
- Added **file-level linking metadata**:
  - `internal_dependencies`
  - `external_dependencies`
  - `imported_by`
- Removed **author and git metadata** from public output by default.
- Improved **Markdown output (`--format md`)**:
  - Added Project Overview, Tree, Folder Map, Dependency Map, and structured file summaries.
- Ensured **format-specific output behavior**:
  - `--format md` → only `codedoc.md`
  - `--format json` → only `codedoc.json`
  - `--format both` → both files
- Added **clear CLI and pipeline logging**:
  - Displays selected output format
  - Displays exact output file path
- Added **BOM-safe file reading (`utf-8-sig`)** across Python, JS/TS, and generic parsers.
- Ensured **language-agnostic processing** (no Python-only assumptions).
- Added tests for:
  - New public output structure
  - Markdown generation
  - Dependency graph presence
  - Cross-language compatibility (including TS/TSX)
- Cleaned up public output by removing:
  - Cache history
  - Raw agent responses
  - Redundant description fields
  
## 0.1.3 - 2026-05-02

- Changed generated docs to one combined JSON file by default.
- Added `--format json|md|both` output selection.
- Added smart content-hash reuse for unchanged and duplicate files.
- Added cache-based output regeneration when selected docs are missing.
- Redesigned public output with project overview, tree, folder map, dependency graph, and flattened file summaries.
- Removed local author metadata and raw agent result duplication from public output.
- Expanded public README with provider setup, defaults, config, output, and cache behavior.

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
