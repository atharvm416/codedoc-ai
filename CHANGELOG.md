# Changelog

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
