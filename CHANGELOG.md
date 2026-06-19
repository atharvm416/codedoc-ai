# Changelog

## 0.9.7 - 2026-06-19

Release candidate updated: 2026-06-19 23:34 IST (UTC+05:30).

### Token- and time-safe failure handling (corrective patch)

A corrective patch release. It stops spending provider calls — and your tokens,
money, and wall-clock time — on a run that cannot succeed, while never losing
work already completed. No new documentation scopes, prompts, providers,
response-schema changes, output artifacts, configuration keys, environment
variables, or CLI flags. Successful runs make exactly the same provider calls as
before.

Previously the only error classifier in the pipeline was rate-limit detection,
so every other provider error was treated as a generic transient failure and
retried per file. A run started with an invalid key, unknown model, or forbidden
access walked through several files before the consecutive-failure breaker
stopped it; a mid-run budget/credit exhaustion matched the rate-limit signals and
slept through the entire backoff ladder for nothing; an input-too-large error
re-sent the identical oversized prompt on every retry. The existing live JSON
backup already made every stop *safe*; this release makes the *decision to stop*
smart and bounded.

- **Terminal-error fast stop.** Two deterministic, network-free classifiers now
  inspect only the text already present in the raised exception chain. A
  confirmed billing/credit exhaustion, invalid credentials, unknown model, or
  forbidden/permission error stops the run on its first occurrence with an
  actionable message and the configuration/credentials exit code (`2`) — instead
  of retrying per file and sleeping through the backoff schedule. Classification
  is conservative by design: bare numeric HTTP codes (`401`/`402`/`403`/`404`/
  `413`) never trigger an abort on their own, and a bare `quota` /
  `resource_exhausted` / `429` remains a retryable rate limit. When in doubt, an
  error stays retryable.
- **Input-too-large is recorded, not retried.** A request/context-too-large error
  is recorded as a failed file without any retry (re-sending the identical prompt
  cannot succeed); the rest of the run proceeds.
- **Bounded rate-limit retrying.** A persistent ambiguous rate limit / quota
  exhaustion (no terminal-billing phrase, e.g. Gemini `RESOURCE_EXHAUSTED`) now
  stops after a bounded amount of retrying — at most one full step-down ladder
  traversal plus one lowest-concurrency pass — instead of grinding to the
  consecutive-failure breaker. This is a transient "retry later" condition, so it
  exits `1`, not `2`. Normal transient rate limits where files keep succeeding
  between step-downs are unaffected.
- **Operator-facing safe-stop reporting.** Both stop paths record the abort to
  `error.log` from the pipeline before it reaches the CLI, and the CLI prints a
  clear, non-`Fatal error:` message naming the cause class and confirming that
  completed files are saved in the live JSON backup and that re-running the same
  command resumes. Every stop preserves the live backup; re-running re-documents
  only the unfinished files.
- **Release-gate classifier corrections.** Context-size messages that use the
  ambiguous phrase `hard limit` are now kept as input-specific failures instead
  of being mistaken for billing exhaustion and aborting the whole run. The
  resolved provider rate-limit profile (including configured signal additions
  and removals) is now used consistently by parallel processing, sequential
  retries, and the sequential zero-progress bound.

The rate-limit signal set, step-down ladder, backoff math, `retry_after_cap_s`
per-sleep cap, `retry_attempts`, concurrency, the three-agent orchestration,
prompts, schema, and provider behavior are all unchanged. Exit codes for existing
error types are unchanged; the only new surface is the unrecoverable-provider
stop (exit `2` for a terminal abort, exit `1` for the bounded rate-limit stop).

## 0.9.6 - 2026-06-17

### Scan robustness and resolution precision (corrective patch)

A corrective patch release. It fixes correctness and robustness defects found
in a code audit and makes the existing CI green by fixing two non-portable
tests. No new documentation scopes, prompts, providers, response-schema
changes, or output artifacts. The only new configuration key is the safety
control `follow_symlinks` (default `False`).

- **Symlink-safe, iterative scanner.** The directory walk is now an explicit
  stack instead of recursion, so a deeply nested acyclic tree can no longer
  raise `RecursionError`. Every traversed directory's resolved identity
  (`(st_dev, st_ino)` where meaningful, else the normalized resolved path) is
  tracked, so symlink/junction cycles and multiple aliases to one real
  directory are visited at most once. By default (`follow_symlinks=False`) all
  symlinked directories and files are skipped — preventing both link cycles and
  escapes outside the project root. With `follow_symlinks=True`, links are
  followed only when their target exists, has the expected type, and resolves
  inside the project root; broken, inaccessible, type-mismatched, and
  out-of-root links are skipped. Lexical skip/dot/ignore rules are applied to a
  link's in-root alias before it is resolved, so a link cannot bypass an ignored
  path, and only project-relative paths are ever emitted as `rel_path`.
- **Deterministic, exact-case import resolution.** Import resolution no longer
  probes the filesystem, so the same repository resolves to the same dependency
  graph on case-sensitive and case-insensitive hosts. The filesystem-dependent
  case-folded matching and the standalone bare final-segment candidate (which
  could link `collections.abc` or `com.example.Bar` to an unrelated root-level
  file) are removed. Dotted imports resolve only through their directory-anchored
  forms; relative, Python dotted-relative, and Dart `package:` imports are
  unchanged. A case-mismatched or otherwise unresolved import stays in the
  per-file `imports` list but creates no internal graph edge, so the dependency
  graph, entry reachability, and catalog reflect only real resolved edges. The
  now-unused `_filesystem_is_case_insensitive` / `_swap_case_letter` helpers were
  removed from `graph.py`; `resolve_import()`'s `root` parameter is retained for
  compatibility only and is no longer read.
- **Atomic legacy summary writer.** The backward-compatible `write_summary()`
  helper now routes through `atomic_write_text` like every other final writer,
  so no completed public artifact is ever written via truncate-in-place.
- **Stricter configuration bounds.** `max_file_size_kb` must be a positive
  integer (`>= 1`); `0`, negatives, and booleans are rejected before scanning
  instead of silently skipping every file. `retry_after_cap_s` must be `>= 0`
  (zero still disables the cap); negatives and booleans are rejected.
- **Portable CI tests.** Two environment-coupled tests were made
  platform- and checkout-name independent (force-path normalization is asserted
  per platform; the `run`-alias test compares the captured root to the actual
  working directory) so the existing 3.10/3.11/3.12 CI matrix passes unchanged.
  No product code, CI workflow, configuration default, prompt, schema, or output
  artifact changed for this work.

## 0.9.5 - 2026-06-15

### Correctness and reliability (behavior changes are bounded and listed below)

A corrective patch release. The only intentional change to successful output is
the dependency-catalog correction; all other changes harden persistence,
packaging, and CI without altering successful serialized contents. No schema
bump, no new configuration, CLI feature, prompt, provider, or output artifact.

- **Evidence-based dependency catalog.** A catalog entry is now admitted only
  when its `(type, canonical_name)` key is authorized by a file's finalized
  links (graph-resolved `internal_dependencies`, or deterministically classified
  `external_dependencies` / `sdk_dependencies`). Model `catalog_updates`,
  `dependency_refs`, and `usage_notes` may enrich a proven dependency with
  `used_for` text but can no longer create or retype one; unresolved hints are
  discarded rather than reclassified. A Python external whose canonical root is
  the project's own package and is resolved internally by the graph is dropped as
  a false external. Every emitted entry now carries non-empty `used_for` text and
  a backing file. Deterministic output remains byte-identical except where an
  entry lacked authoritative evidence.
- **Atomic completed output.** Final JSON and Markdown are written through a
  single canonical `atomic_write_text` helper (unique temp sibling, flush +
  fsync, rename) so a completed artifact can never be truncated in place. In
  `both` mode both payloads are rendered before any target is mutated, Markdown
  is replaced first and JSON last (the JSON path is also the live backup), giving
  per-artifact atomicity.
- **Fatal live-backup persistence.** A failed live-backup write now raises
  `LiveBackupWriteError` (an `OutputError`) instead of being silently swallowed.
  `SafeWriter.record()` rolls back all in-memory markers on failure, and the
  execution layer treats persistence failure as fatal on both the sequential and
  parallel paths — no retry, no rate-limit reclassification, pending work
  cancelled — so the run never continues under a false crash-safety guarantee.
- **Artifact-path collision rejection.** Distinct generated artifacts that would
  target the same normalized path are rejected before scanning or mutation, while
  the intentional final-JSON / live-backup phase alias is accepted.
- **Three-provider contract matrix.** OpenAI, Anthropic, and Gemini are verified
  through one shared contract using injected fake SDK clients (no network or
  credentials in normal tests).
- **Active CI and metadata honesty.** Added a least-privilege CI workflow
  (tests, lint, build, `twine check`, clean-wheel smoke). The declared
  `requires-python` is now `>=3.10,<3.13`, the classifiers drop Python 3.9, and
  the CI matrix tests exactly 3.10, 3.11, and 3.12.
- **Release hygiene.** Import resolution now follows the actual target
  filesystem's case
  semantics instead of folding case unconditionally. Documentation-agent
  fallback handling is defined on the agent class rather than installed by a
  runtime monkey-patch. The dormant local-provider compatibility module now
  uses the standard library for its optional liveness check, so `requests` is
  no longer a runtime dependency, and repository tests are no longer bundled
  into the source distribution.

## 0.9.4 - 2026-06-14

### Internal decomposition (structural only — no behavior change)

This release reorganizes two oversized modules into cohesive,
single-responsibility units. It does **not** change file selection, provider
calls, prompts, retries, the output schema, output contents, the dependency
catalog, configuration defaults, or the CLI. For the same inputs the run
behaves identically and the serialized JSON/Markdown is byte-identical to
0.9.3 output.

- **Pipeline decomposition.** `codedoc/pipeline.py` is now a thin lifecycle
  coordinator. Its internals moved into three modules behind the unchanged
  `run_pipeline()` facade and phase ordering:
  - `codedoc/core/resume.py` — live-backup path resolution, existing JSON/MD
    record loading, public→internal record reconstruction, final
    documentation-record construction, and stale-build / legacy-db cleanup.
  - `codedoc/core/discovery.py` — entry recovery from existing CodeDoc
    metadata, dependency-graph construction, entry-reachability selection, and
    graph-edge serialization (selection behavior moved unchanged).
  - `codedoc/core/execution.py` — rate-limit / retry-after classification, the
    adaptive-parallelism ladder, and sequential/parallel processing behind a
    new `ExecutionContext` / `ExecutionOptions` boundary and the
    `execute_agent_files()` entry point. The provider-aware `RateLimitProfile`
    and execution policy are built by the pipeline and passed in; execution no
    longer reads the configuration dictionary.
- **Serializer extraction.** Markdown serialization/parsing moved from
  `codedoc/core/project_view.py` into `codedoc/core/markdown_view.py`
  (`markdown_from_view`, `markdown_to_view`, `json_from_markdown`,
  `markdown_from_json`, the embedded-view readers, and the visible-Markdown
  parsers/render helpers). `project_view.py` retains view assembly, the
  dependency catalog, pruning, usage-example sanitization, and
  `read_codedoc_meta`.
- **Compatibility.** The moved private helpers remain importable from their
  previous modules for one release: pipeline helpers from `codedoc.pipeline`,
  and the serializer helpers from `codedoc.core.project_view` (forwarded
  lazily to `codedoc.core.markdown_view`). These re-exports are deprecated and
  emit no runtime warning. No schema change; `SCHEMA_VERSION` stays `1.4`.
- **Tests.** Added `tests/test_094_pipeline_boundaries.py` and
  `tests/test_094_project_view_split.py`, including byte-identical
  golden-output fixtures (`tests/fixtures/golden_094_*`). One existing
  monkeypatch target (`codedoc.pipeline.time.sleep`) was retargeted to the
  defining module (`codedoc.core.execution.time.sleep`).

## 0.9.3 - 2026-06-13

### Deterministic output, dependency categories, and centralized reading

- **SDK/standard-library separation.** A new pure, deterministic, language-aware
  classifier (`codedoc/core/dependency_kind.py`) splits non-project imports into
  third-party `external_dependencies` and standard-library / SDK
  `sdk_dependencies` (additive field). Dart `dart:*`, Python stdlib (via
  `sys.stdlib_module_names` with a committed Python 3.9 fallback), and Node
  built-ins / `node:*` are recognized as SDK; package subpaths and scoped npm
  packages are canonicalized to their package root. Importability is never used
  to classify modules.
- **Internal links only from the graph.** `links.internal_dependencies` /
  `links.imported_by` now come exclusively from resolved dependency-graph edges.
  Unresolved agent text can no longer create an internal link, and an `internal`
  catalog hint is accepted only when it exactly matches a resolved internal path
  for that file; otherwise it is reclassified as non-project data. The catalog is
  grouped by `(type, canonical_name)`.
- **Centralized document reader.** A single read-only parser
  (`codedoc/core/document.py`, `read_codedoc_document`) owns CodeDoc JSON /
  Markdown parsing and structural ownership. Output ownership, `SafeWriter`,
  metadata reads, existing-record reads, resume candidates, and stale-build
  migration all route through it while keeping their own missing/malformed
  policy. It reads UTF-8 with optional BOM, rejects invalid UTF-8, validates
  collection types, rejects duplicate paths, prefers a valid embedded view, and
  fails closed on unknown/missing-schema completed output and unsupported
  extensions.
- **Ownership tightening (intentional).** Markdown that merely contains a
  `<!-- codedoc-ai:` marker but whose metadata is malformed (and which has no
  valid embedded view) is now treated as foreign and is never overwritten. Valid
  legacy Markdown remains accepted.
- **Deterministic, timestamp-free completed output.** `generated_at` is removed
  from the completed JSON `_codedoc` block, the Markdown metadata comment, and
  the embedded lossless view. Two runs with identical sources, documentation,
  configuration, and stats now produce byte-identical JSON and Markdown. Old
  outputs containing `generated_at` remain readable. Live backups keep
  `created_at` / `updated_at` diagnostics (new backups write `created_at`).
- **Private record metadata plumbing.** A registry
  (`codedoc/core/record_meta.py`) preserves explicitly registered private keys
  through JSON, Markdown (embedded view only — never visible prose), live backup,
  and resume reconstruction. The production registry is empty in this release;
  arbitrary underscore-prefixed model output is not preserved.
- No schema bump: `sdk_dependencies` and private keys are additive; missing
  `sdk_dependencies` loads as an empty list.

## 0.9.2 - 2026-06-12

### Safe planning and CI ergonomics

- Added a filesystem-read-only, provider-free `--dry-run` driven by the same
  immutable routing plan as real execution.
- Added `--max-files`, repeatable `--force-files`, and `--allow-partial`, with
  matching config and environment-variable support.
- Added stable CLI exit codes for success, file/output failures, setup errors,
  and interrupts.
- Added approximate planned and actual LLM call/token reporting. Dry-run totals
  are explicitly lower bounds and no monetary accuracy is claimed.
- Centralized per-file truncation so all three agents receive the same bounded
  source string and only one warning is emitted.
- Added read-only ownership inspection and moved the paid-file cap ahead of
  filesystem mutation, writer initialization, and provider creation.
- Added a packaged, manual-only GitHub Actions workflow with a dry-run, paid
  cap, least-privilege permissions, and artifact upload.
- Kept `--safe-mode` accepted but hidden for backward compatibility.
- Added focused 0.9.2 regression coverage and synchronized release identity.

## 0.9.1 - 2026-06-08

### Bug-fix stabilization patch (first PyPI release)

Corrective-only patch. No new features or output-shape changes.

- **A1 — entry-reachability is no longer silent.** When an entry is given,
  files not reachable from it were dropped without notice. `_select_files` now
  logs a clear WARNING listing the excluded files, records `stats["entry_excluded"]`,
  and the CLI prints an excluded-files line. (The structural selection fix is
  tracked for a later minor; this patch only removes the silent failure.)
- **A2 — a wrong `--entry` no longer silently documents the whole repo.** An
  explicitly specified entry that cannot be resolved, is not in the scanned set,
  resolves outside the project root, or is given when **no** supported files are
  scanned, now raises `ConfigError` instead of falling back to all files or
  exiting successfully. Auto-detection with no entry still documents everything.
- **A3 — parser false imports fixed.** The Go parser no longer treats arbitrary
  string literals (e.g. `fmt.Println("hi")`) as imports — only string-literal
  paths in `import "..."` statements and `import ( ... )` blocks are read,
  comments are ignored, and raw-string (backtick) paths are supported.
  Interpreted literals use Go's byte-accurate escape semantics, including
  multi-byte UTF-8 `\xNN` / octal sequences and Unicode escapes. The HTML parser
  no longer treats CSS `<link href>` as a code import (kept `<script src>` and
  JS imports).
- **A4 — no stale/empty record substituted for a real one.** In the parallel
  batch, a rate-limited file was treated as "already recorded" using state that
  also included records **preloaded** from a prior run, so a *changed* file could
  be restored from stale documentation instead of retried. `SafeWriter` now
  tracks records written *this run* (`recorded_this_run()`); a changed,
  rate-limited file is retried, and a file genuinely recorded this run recovers
  its real record via `get_record()` (never an empty `{}`).
- **A5 — honest interrupt message.** Removed dead code; the Ctrl-C message is now
  conditional ("…if the run reached file processing") so it never falsely claims
  progress was saved when interrupted before any file was processed.
- **A6 — scanner is re-entrant.** The directory walker no longer stores state on
  the function object; state lives on a per-scan `_Walker` instance.
- **Version identity.** `pyproject.toml`, `codedoc.__version__`, the CLI
  `--version`, and the README all report `0.9.1`, and the automated test
  (`test_version_identity_consistent`) enforces agreement across **all four**,
  including the README "Current release" line.
- **Reliable tests.** `tests/conftest.py` redirects the temp root into the repo
  (`.pyt_tmp`) so a locked system temp dir does not make the suite unrunnable.
  (This addresses the observed locked-system-temp failure; it is not a guarantee
  for every environment.)

## 0.9.0 - 2026-06-04

### Output preflight safety, clean INFO logs, extension list fix, configurable content truncation

---

#### G0 — Output Preflight Safety

Foreign output targets now fail immediately with a `ConfigError` before the
scanner runs, the provider initialises, or any LLM API call is made. Previously
a foreign file at the target path would only be detected inside
`write_project_outputs`, after all tokens had already been spent.

- **`codedoc/core/output.py`**: Added `preflight_output_targets()` which calls
  `_check_file_ownership()` for all final public targets (JSON, MD, both) and a
  new `_check_md_live_backup_ownership()` for the MD live-backup JSON sibling.
- **`codedoc/pipeline.py`**: Calls `preflight_output_targets()` immediately after
  output spec resolution, before `scan_files()` and `create_provider()`.
- **`codedoc/core/loader.py`**: `_resolve_output_spec()` now only emits the
  format-conflict warning when `--format` was explicitly passed by the user (not
  when the default `"json"` value from DEFAULTS triggers a mismatch).

#### G1 — Clean Log Output

Third-party HTTP libraries (`httpx`, `httpcore`, `openai`, `anthropic`,
`google.auth`) are now silenced at WARNING level by default. At `--verbose` /
DEBUG the HTTP diagnostics are restored. Per-agent progress lines appear at INFO
so users can see what codedoc is doing at each step.

- **`codedoc/utils/logger.py`**: `_NOISY_LOGGERS` constant defines the list;
  `_configure()` sets those loggers to WARNING; `set_level()` lowers them to
  DEBUG when the root logger is set to DEBUG.
- **`codedoc/agents/orchestrator.py`**: Added timing via `time.monotonic()` and
  INFO/WARNING log lines after each agent: `[FILE] path | structure ok  0.8s`,
  `[FILE] path | dependencies ok  0.9s`, `[FILE] path | documentation ok  1.2s`.
  Fallbacks emit WARNING with `"fallback"` in the message.

#### G5 — Extension List Consistency

`_candidate_variants()` in `graph.py` used a hardcoded 9-extension list that
was out of sync with `_KNOWN_EXTENSIONS` and `DEFAULTS["extension_language_map"]`.
Import resolution for Go, Kotlin, Swift, Rust, Ruby, and C-family files silently
produced no candidates.

- **`codedoc/core/graph.py`**: `_KNOWN_EXTENSIONS` expanded to all 19 extensions
  in `DEFAULTS["extension_language_map"]`. `_candidate_variants()` now uses
  `sorted(_KNOWN_EXTENSIONS)` instead of a separate hardcoded list. A comment
  notes the sync requirement with `loader.py`.

#### G6 — Configurable Content Truncation

Files above 12,000 characters were silently truncated with a DEBUG-only log.
Users saw degraded documentation for large files with no indication why.

- **`codedoc/core/loader.py`**: `max_content_chars` added to `DEFAULTS` (12000)
  and `_ENV_KEY_MAP` (`CODEDOC_MAX_CONTENT_CHARS`). Validation requires a positive
  integer ≥ 1000.
- **`codedoc/agents/base_agent.py`**: Removed module-level `_MAX_CONTENT_CHARS`
  constant. `BaseAgent.__init__` now accepts `max_content_chars: int = 12000`.
  `_truncate()` uses `self._max_content_chars` and logs at INFO with the file
  path and original / truncated character counts.
- **`codedoc/agents/orchestrator.py`**: `Orchestrator.__init__` accepts
  `max_content_chars: int = 12000` and forwards it to each agent.
- **`codedoc/pipeline.py`**: Passes `config.get("max_content_chars", 12000)` to
  the `Orchestrator` constructor.
- All three agent subclasses pass `file_path` to `_truncate()` for accurate logs.

---

## 0.8.1 - 2026-06-02

### Lossless Markdown, placeholder sanitization, configurable defaults, provider-aware rate-limit backoff

---

#### Workstream A — Lossless Markdown View

Markdown output now embeds the complete public JSON view as a hidden base64
comment so `json_from_markdown()` (and incremental re-runs that read a `.md`
file) recover the full dependency catalog, per-file hashes, and all dependency
metadata without any information loss.

- **`codedoc/core/project_view.py`**:
  - `markdown_from_view()` writes a `<!-- codedoc-ai-view-base64 ... -->` block
    immediately after the legacy `<!-- codedoc-ai: ... -->` metadata comment.
    The block is standard base64-encoded UTF-8 JSON, which avoids comment-safety
    issues with raw `--` or `-->` sequences in generated text.
  - `markdown_to_view()` now tries the embedded view first (fast, lossless path);
    falls back to the existing visible Markdown parser for pre-0.8.1 files.
  - New public helper `read_embedded_view(markdown)` decodes and validates the
    embedded block; returns `None` on any failure so callers fall back safely.
  - `read_codedoc_meta()` no longer raises `ConfigError` when `entry_file` is
    `null`; a valid CodeDoc file with no entry point is now correctly identified
    as owned rather than foreign.
- **`codedoc/pipeline.py`**:
  - `_load_existing_file_docs_from_md()` preserves file hashes from the embedded
    view when the lightweight metadata comment has no hash for a path.
  - `_resolve_entry_and_docs()` no longer raises unconditionally when no existing
    output is found; first runs without `--entry` now reach `detect_entry_file()`
    for auto-detection instead of failing immediately.

#### Workstream B — Placeholder Usage Example Sanitization

LLM-generated usage examples that contain placeholder package names (e.g.
`import 'package:your_package/...'`) are now removed before any output is
written or cached.

- **`codedoc/core/project_view.py`**: `_clean_file()` calls the new
  `_sanitize_usage_example()` helper, which checks against `_PLACEHOLDER_PATTERN`
  (a compiled `re.IGNORECASE` regex with word-boundary guards).  Covered
  placeholders: `your_package_name`, `your_package`, `your_project`, `your_app`,
  `example_package`, `my_package`, and Dart-style `package:example/`.
  Sanitization is idempotent and applies to both freshly generated records and
  cached/reused records loaded from prior output files.

#### Workstream C — Configurable Hardcoded Defaults

All previously hardcoded scanner and provider defaults are now driven by a
single source of truth in `DEFAULTS` (`loader.py`) and support `_add` / `_remove`
override keys.

- **`codedoc/core/loader.py`**:
  - `DEFAULTS` gains eleven new keys: `skip_dirs_add`, `skip_dirs_remove`,
    `extension_language_map` (full 18-entry map), `extension_language_map_add`,
    `extension_language_map_remove`, `auto_entry_candidates`,
    `auto_entry_candidates_add`, `auto_entry_candidates_remove`,
    `provider_prefixes`, `provider_prefixes_add`, `provider_prefixes_remove`.
  - Three resolver helpers implement the resolution order (replace → `_add` →
    `_remove`): `_resolve_list_override`, `_resolve_dict_override`,
    `_resolve_nested_list_dict_override`.
  - `_apply_config_overrides()` is called after all config sources are merged;
    it resolves all four configurable keys and derives `supported_extensions`
    from the resolved `extension_language_map`.
  - Backward-compat bridge: if `supported_extensions` was explicitly set to a
    value different from the defaults, it is used as a filter on
    `extension_language_map` so old configs continue to restrict scanning as
    intended.
- **`codedoc/core/scanner.py`**:
  - Hardcoded `SKIP_DIRS` and `EXTENSION_LANGUAGE_MAP` removed.
  - `scan_files()` receives `extension_language_map` (primary) instead of
    `supported_extensions`.  A positional-list guard handles legacy callers
    that pass a list as the second argument.
  - `detect_entry_file()` receives the resolved `auto_entry_candidates` list;
    falls back to a module-level default for direct callers.
- **`codedoc/pipeline.py`**: passes `extension_language_map` and
  `auto_entry_candidates` to the scanner; always appends the output directory
  name to the scan skip list (even when the user removed it via
  `--remove-skip-dir`) to prevent codedoc from documenting its own output.
- **`codedoc/cli/cli.py`**: three new flags: `--skip-dirs DIR [...]`,
  `--add-skip-dir DIR` (repeatable), `--remove-skip-dir DIR` (repeatable).
- **`codedoc/llm/factory.py`**: `create_provider()`, `_make_api()`,
  `_resolve_api_provider()`, and `_provider_api_key()` all accept and use
  `provider_prefixes` from config; module-level tuples kept as fallbacks.

#### Workstream D — Provider-Aware Rate-Limit Backoff

Parallel ladder step-downs now sleep between rungs using provider-aware
exponential backoff, with optional `Retry-After` hint parsing.

- **`codedoc/llm/rate_limit_profile.py`** *(new)*:
  - `RateLimitProfile` dataclass — `provider`, `signals`, `min_backoff_s`,
    `backoff_scale`.
  - `PROVIDER_PROFILES` — preconfigured profiles for `openai`, `anthropic`,
    `gemini`, and `default`.
  - `get_rate_limit_profile(provider_name, config)` — returns the resolved
    profile with `rate_limit_backoff_s`, `rate_limit_backoff_scale`,
    `rate_limit_signals_add`, and `rate_limit_signals_remove` applied without
    mutating module defaults.
- **`codedoc/pipeline.py`**:
  - `_is_rate_limit_error(exc, profile=None)` — when a `profile` is supplied,
    checks only `profile.signals`; falls back to `_RATE_LIMIT_SIGNALS` for
    backward compatibility with callers without a profile.
  - `_detect_limit_type(error_msg)` — classifies errors as `"tpm"`, `"rpm"`,
    `"quota"`, `"overloaded"`, or `None`.
  - `_process_descriptor_batch()` return type changed:
    `retry_rate_limited` is now `list[tuple[dict, Exception]]` so the causing
    exception is preserved for `Retry-After` parsing and error sampling.
  - `_process_agent_files()`: fetches the provider profile, passes it to
    `_process_descriptor_batch()`, and sleeps between rungs using:
    - `min(Retry-After, retry_after_cap_s)` when a hint is present and
      `respect_retry_after = True`,
    - `min(min_backoff_s × backoff_scale ^ rung, retry_after_cap_s)` otherwise,
    - no sleep when `rate_limit_backoff_s = 0`.
  - Rate-limit warning dicts now include: `retry_after_s`, `sleep_s`,
    `error_sample`, `limit_type`, `event_number`, `rung_index`.
- **`codedoc/core/loader.py`**: four new `DEFAULTS` keys:
  `rate_limit_backoff_s`, `rate_limit_backoff_scale`, `rate_limit_signals_add`,
  `rate_limit_signals_remove`.
- **`codedoc/cli/cli.py`**: compact rate-limit summary line printed only when
  step-down events occurred; shows event count, providers, and total sleep time.

#### Version

- `codedoc/__init__.py`, `pyproject.toml`, `cli.py`: `0.8.0` → `0.8.1`.

#### Validation

- Added regression coverage for lossless Markdown regeneration, placeholder
  sanitization, configurable defaults, provider-aware rate-limit backoff, and
  rate-limit edge cases.
- Full test suite passes.
- Built sdist/wheel and verified release metadata with `twine check`.

---

## 0.8.0 - 2026-05-31

### Always-on live JSON crash backup, parallel crash-safety, rate-limit adaptive parallelism, error.log overhaul

0.8.0 closes the full known crash-safety/output-safety gap end to end.

---

#### Work Item 1 — Always-on live JSON backup (replaces hidden checkpoint)

Every run now writes a visible live JSON backup that is updated after each completed file.
`--safe-mode` is deprecated and kept only for backwards compatibility — it now prints a
deprecation notice and has no additional effect.

- **`codedoc/core/safe_writer.py`** (overhauled): `SafeWriter` is now the default recorder.
  Constructor now accepts a pre-computed `backup_path: Path` directly.  The live backup
  always starts with a `_crash_safety` banner as the first JSON key so interrupted files are
  immediately recognisable as crash-recovery backups.  Three new methods:
  `initialize_empty()` — writes the banner before any AI call;
  `set_queue_order()` — controls the `files` array order (topological / queue order, not
  alphabetical);  `has_record()` — deduplication check for retry logic.
  `delete()` removes the live backup for MD-only runs after a clean Markdown conversion.
  If deletion fails (Windows file-lock) a warning is logged and the path is reported so the
  user knows the leftover file is safe to remove manually.

- **`codedoc/pipeline.py`** — `_resolve_live_backup_path()` helper centralises all backup
  path logic, including the named-MD sibling case (`--output docs/report.md` → live backup
  at `docs/report.json`).  `SafeWriter` is always created regardless of `--safe-mode`.
  `initialize_empty()` is called before `create_provider()` so the backup exists even if
  provider initialisation fails.  The topological order is passed to `set_queue_order()`.
  Old `.codedoc_progress.json` checkpoints are migrated on the first run that finds no live
  backup and deleted from the rotation afterwards.  New stats keys returned:
  `live_backup_path` (absolute path to live backup), `error_log` (absolute path, set when
  any issue is recorded), `issues_recorded` (total count), `rate_limit_warnings` (list of
  step-down events).

- **`codedoc/core/output.py`**: removed the intermediate `.codedoc_build.json` write for
  `--format md` runs.  Markdown is written directly from the in-memory view; crash safety
  is provided by the live JSON backup.  `BUILD_FILENAME` is kept only for reading/migrating
  stale 0.7.x build files.

- **`codedoc/core/loader.py`**: updated `_load_existing_file_docs()` to accept
  `live_backup_path` so the named-MD sibling (`report.json`) is probed before the default
  `json_filename`.

#### Work Item 2 — Parallel crash-safety: record in worker thread

Previously a Ctrl-C or crash during parallel processing could discard a completed file's
result because `recorder.record()` was called in the main `as_completed` loop.

- **`codedoc/pipeline.py`** — `_process_and_record()` wrapper calls `recorder.record()`
  inside the worker thread before returning, so a crash between worker completion and main
  collection never loses a result.  The main loop no longer calls `recorder.record()` in the
  parallel path.  `has_record()` is checked before adding a descriptor to the retry list so
  a file that already recorded before batch cancellation is not submitted twice.

#### Work Item 3 — Adaptive parallelism on rate limits

When a provider signals 429 / rate-limit / too-many-requests, file concurrency is stepped
down through a ladder instead of hammering the API at the original concurrency.

- **`codedoc/pipeline.py`**:
  - `_is_rate_limit_error()` — walks the full `__cause__`/`__context__` chain; covers
    OpenAI (`429`, `rate_limit_exceeded`, `tpm`), Anthropic (`529`, `overloaded`), and
    Gemini (`RESOURCE_EXHAUSTED`, `quota`).
  - `_build_default_ladder()` — generates the step-down ladder for any
    `max_parallel_files` value (e.g. `5 → [5, 2, 1]`, `10 → [10, 5, 1]`).
  - `_process_descriptor_batch()` — processes one ladder level and classifies results as
    succeeded / retry-rate-limited / failed-non-rate-limit.
  - `_process_agent_files()` — iterates the ladder, collects step-down events into
    `stats["rate_limit_warnings"]`, prints a provider-specific WARNING to stdout on each
    step-down with the provider name and original `max_parallel_files` value.
  - `_parse_retry_after()` — extracts `Retry-After` sleep delays from error messages;
    applied in sequential mode too when `respect_retry_after = True`.
- **`codedoc/core/loader.py`**: added `rate_limit_adaptive`, `parallel_ladder`,
  `respect_retry_after`, `retry_after_cap_s` to `DEFAULTS`; full `parallel_ladder`
  validation in `_validate()` (strictly decreasing, clamped to `max_parallel_files`,
  trailing `1` appended if missing).

#### Work Item 4 — `error.log` discoverability and `ErrorReporter` severity

- **`codedoc/utils/errors.py`**: `ErrorReporter.record()` gains a `level` parameter
  (`"error"` / `"warning"`).  `has_errors()` and `error_count()` count only error-level
  entries.  `has_issues()` and `issue_count()` count all entries.  `summary()` returns `""`
  for warning-only runs so recovered rate-limits never appear in the final `codedoc.json`
  `errors` field or the Markdown `## Errors` section.  Log header changed from `error(s)` to
  `issue(s)`.
- **`codedoc/pipeline.py`**: `ErrorReporter` is now initialised with
  `output_dir / "error.log"` instead of `root / "error.log"`.  `stats["error_log"]` and
  `stats["issues_recorded"]` are set on every return path (not only when `failed > 0`).
  Rate-limit health-check notes are recorded as `level="warning"` so they appear in
  `error.log` for diagnostics but do not alarm the final output.
- **`codedoc/cli/cli.py`**: the error log path is always printed when
  `stats["issues_recorded"] > 0`; message distinguishes "file(s) failed" from "issue(s)
  recorded (all recovered)".  Rate-limit step-down warnings are printed to stdout.
  `--safe-mode` help updated to `[DEPRECATED]`.

#### Version

- `codedoc/__init__.py`, `pyproject.toml`, `cli.py`: `0.7.2` → `0.8.0`.

#### Tests

- `tests/test_scenarios.py`: updated 3 `SafeWriter` constructor calls to new `backup_path`
  signature.
- `tests/test_080_features.py` *(new, 38 tests)*: covers live backup creation, banner
  presence, queue order, parallel crash-safety, ownership guard, resume, hash-change
  reprocess, checkpoint migration, rate-limit ladder, signal detector (OpenAI/Anthropic/
  Gemini/false-positives/cause-chain), provider notifications, error.log location and stats,
  deprecation notice, `--format both` behaviour, stats keys, ladder validation,
  no-files early return, and warning exclusion from final output.

**All 163 tests pass** (125 existing + 38 new).

---

**Behaviour on interrupt and resume (0.8.0 default — always-on live backup):**
1. User runs `codedoc run --entry src/main.py` on a 100-file project.
2. Before the first LLM call, `codedoc/codedoc.json` is created with a `_crash_safety`
   banner and an empty `files` array.
3. After every completed file, `codedoc/codedoc.json` is updated atomically (`.tmp` rename).
4. Run is interrupted (Ctrl-C, crash) after 60 files.  `codedoc/codedoc.json` contains 60
   complete file records in topological order, clearly marked with `_crash_safety` as
   partial output.
5. User re-runs; `codedoc.json` is read (including in-progress entries), 60 unchanged files
   are skipped, only the remaining 40 are sent to the LLM.
6. On clean completion, `write_project_outputs` overwrites `codedoc.json` with a final
   clean output (no `_crash_safety`, no `status = "in_progress"`).

**MD-only and named-MD runs:**
- `--format md`: live backup is `codedoc/codedoc.json`; removed automatically on clean
  Markdown write.  On interrupt, the JSON sibling remains as the resume source.
- `--output docs/report.md`: live backup is `docs/report.json` (sibling derived from the
  Markdown stem); removed on clean success.

**Rate-limit step-down example:**
```
[OpenAI] Rate limit detected - your configured max_parallel_files (5) has been
reduced to 2. Retrying 4 remaining file(s) at lower concurrency.
```

---

## 0.7.2 - 2026-05-30

### Added: incremental progress checkpoint + `--safe-mode` live output + MD intermediate + ownership guard

This release fully solves the data-loss-on-interrupt problem for every output format and run
mode.  It also adds the first line of defence against codedoc accidentally overwriting files
it did not create.

---

#### Checkpoint (always-on, default behaviour)

Reverses the 0.6.4 decision ("no per-file checkpoint writes during a run") by introducing a
lightweight, thread-safe checkpoint file that persists each file result to disk the moment it
completes, for all output formats (JSON, MD, and both).

- `codedoc/core/checkpoint.py` *(new)*: `Checkpoint` class — writes `.codedoc_progress.json`
  to the output directory after every file.  Writes are atomic: content is serialised to a
  `.tmp` sibling first, then renamed into place so a crash mid-write never leaves a corrupt
  file.  Thread-safe via a per-instance lock; safe to call from parallel worker threads.
- `codedoc/core/__init__.py`: exported `Checkpoint` in `__all__` and the lazy `__getattr__`
  dispatcher, consistent with all other public core exports.

#### `--safe-mode` (opt-in, visible partial output)

Adds a `--safe-mode` CLI flag and matching `safe_mode` config key / `CODEDOC_SAFE_MODE`
environment variable.  When active, `Checkpoint` is replaced by `SafeWriter`, which writes
directly to the real output file after every completed file — so the output always contains
whatever has been documented so far, even if the run is interrupted.

- `codedoc/core/safe_writer.py` *(new)*: `SafeWriter` class — same thread-safe, atomic-write
  design as `Checkpoint`, but the target is the real output file rather than a hidden
  intermediate.  The partial JSON embeds `_codedoc.status = "in_progress"` so subsequent runs
  can distinguish it from a completed output and resume correctly.
  - **JSON / both format**: target is `codedoc.json`.  The final `write_project_outputs` call
    overwrites it with the complete, polished output — no separate cleanup required.
  - **MD-only format**: target is `.codedoc_build.json` (internal build file, see below).
    After a successful MD write, `SafeWriter.delete()` removes it.  On failure it is
    preserved so the user still has partial output and a re-run resumes automatically.
- `codedoc/core/project_view.py`: added public `clean_file_record()` wrapper around the
  internal `_clean_file()` so `SafeWriter` can produce structurally identical file entries to
  what `build_project_view` would produce.
- `codedoc/core/__init__.py`: exported `SafeWriter`.
- `codedoc/core/loader.py`: added `"safe_mode": False` to `DEFAULTS`, `"CODEDOC_SAFE_MODE"`
  to `_ENV_KEY_MAP`, and bool-coercion in `_validate()` (env vars arrive as strings).
- `codedoc/pipeline.py`:
  - `run_pipeline`: creates either `SafeWriter` or `Checkpoint` depending on `safe_mode`;
    both are referred to via the `recorder` variable.  Calls `recorder.record()` /
    `recorder.delete()` uniformly — the recorder type determines the behaviour.
  - `_process_agent_files` / `_process_files_sequentially`: parameter renamed
    `checkpoint` → `recorder`; type annotation updated to `Checkpoint | SafeWriter`.
  - `_resolve_entry_and_docs`: always probes the JSON candidate and build file before MD,
    regardless of the current `--format` setting, enabling cross-format and build-file resume.
- `codedoc/cli/cli.py`: added `--safe-mode` flag; `KeyboardInterrupt` message updated;
  `Files resumed` summary line added.

#### MD-only runs now always produce a JSON intermediate before converting

Previously a `--format md` run held all results in RAM and wrote one file at the end — a
crash before that point lost everything.  Now `write_project_outputs` for MD format writes
the full result to `.codedoc_build.json` **before** starting the Markdown conversion.

- On successful MD write → `.codedoc_build.json` is deleted automatically.
- On failure (exception, crash during conversion) → `.codedoc_build.json` is preserved;
  codedoc logs its location.  Re-running the same command loads it via the incremental hash
  check and re-attempts the conversion without any LLM calls.

`--format both` is unaffected: the JSON output itself serves as the durable intermediate.

#### Internal build file (`.codedoc_build.json`)

`BUILD_FILENAME = ".codedoc_build.json"` (exported from `codedoc.core.output`) names the
internal intermediate file used by both `write_project_outputs` (MD-only runs) and
`SafeWriter` (safe-mode MD runs).  The dot-prefix marks it as a system-managed file — not a
final output, not user-editable.

- `codedoc/pipeline.py` — `_load_existing_file_docs`: loads from both `codedoc.json`
  (baseline) and `.codedoc_build.json` (newer-run overlay) and **merges** them.  Build-file
  records take priority per-file so that LLM work completed in an interrupted newer run is
  never discarded just because an older `codedoc.json` already exists.
- `codedoc/pipeline.py` — `_resolve_entry_and_docs`: adds `.codedoc_build.json` to the
  candidate list so the entry file is recoverable from a partial build file.

#### Ownership guard before writing output files

`write_project_outputs` and `SafeWriter` now verify that any existing file at the target path
was produced by codedoc before allowing an overwrite.  If the file does **not** carry a
`_codedoc` metadata block (JSON) or `<!-- codedoc-ai: -->` comment (Markdown), a
`ConfigError` is raised — codedoc refuses to overwrite data it did not create.

- `codedoc/core/output.py`: `_check_file_ownership(path)` — raises `ConfigError` for
  non-codedoc files; passes silently for new files or files codedoc owns.  The check now
  covers `json_path`, `md_path`, **and** `build_path` (`.codedoc_build.json`).
- `codedoc/core/safe_writer.py`: `load()` now raises `ConfigError` at startup when the
  target file exists but has no `_codedoc` block, preventing SafeWriter from ever flushing
  over a foreign file during the run.
- `codedoc/cli/cli.py`: `ConfigError` is surfaced with an `"Error: ..."` prefix (matching
  `FileNotFoundError`) rather than `"Fatal error: ..."`, giving the user a clean actionable
  message without a traceback.

#### Fixed: modified files are re-documented when resuming from a checkpoint

When a run is interrupted and a file is edited before the user re-runs, the checkpoint entry
for that file is discarded and the file is re-documented rather than silently restoring stale
documentation.

- `codedoc/core/checkpoint.py`: `record()` now accepts an optional `file_hash` parameter.
  When provided, the hash is stored inside the checkpoint entry under the reserved key
  ``"_checkpoint_hash"``.
- `codedoc/core/safe_writer.py`: `record()` updated with the same optional `file_hash`
  parameter for interface consistency.
- `codedoc/pipeline.py`:
  - Added `_safe_file_hash()` helper.
  - Both `_process_agent_files` (parallel path) and `_process_files_sequentially` compute
    and forward the file hash to `recorder.record()`.
  - The routing loop uses three explicit branches:
    1. **No hash stored** (`stored_hash == ""`): checkpoint was written by code older than
       0.7.2 and cannot be verified — reprocess to avoid silently restoring potentially
       stale documentation.
    2. **Hash mismatch** (`content_hash != stored_hash`): file was modified after it was
       checkpointed — discard entry, reprocess.
    3. **Hash matches**: checkpoint entry is current — restore it and skip the LLM.
  - The ``"_checkpoint_hash"`` key is stripped before the entry is stored in
    ``new_results``, so it never surfaces in the final output.

#### Fixed: hardening of the recovery / ownership work (review follow-ups)

Follow-up fixes to the recovery and ownership features above, found while
reviewing the release.

- `codedoc/core/safe_writer.py` — `SafeWriter.load()`:
  - **No longer erases prior work on a safe-mode interrupt.**  When a *completed*
    `codedoc.json` already exists, its records are now pre-loaded into memory, so
    the first per-file flush preserves them.  Previously the first flush wrote
    only the files processed in the current run, erasing previously completed
    records if the run was then interrupted — making `--safe-mode` worse than the
    default checkpoint.  Records are now pre-loaded for both `in_progress`
    intermediates and completed outputs.
  - **Refuses to overwrite malformed / unreadable target files.**  `load()` now
    raises `ConfigError` when the target file cannot be parsed as JSON or is not a
    JSON object with a `_codedoc` block, instead of logging a warning and starting
    fresh (which would overwrite the foreign file on the first flush).  This brings
    `SafeWriter` in line with `_check_file_ownership` in `output.py`, which already
    treated malformed files as foreign.
  - The stale module docstring describing `codedoc.json` as the MD-only
    intermediate was corrected to `.codedoc_build.json`.
- `codedoc/pipeline.py` — `_load_existing_file_docs()`: the `.codedoc_build.json`
  overlay is now **freshness-gated**.  A build file is only overlaid onto
  `codedoc.json` when it is at least as new (by modification time).  A build file
  left behind by an earlier crashed MD run, after a later `--format json` run
  rewrote `codedoc.json`, is now detected as stale, skipped, and removed — so older
  build-file records can no longer silently replace newer JSON documentation (the
  inverse of the merge case the overlay was added for).
- `codedoc/__init__.py`: `__version__` corrected from `0.7.0` to `0.7.2` to match
  the CLI `--version` output and `pyproject.toml`.
- `OPENAI_RUN_FLOW.md` → `RUN_FLOW.md`: the run-flow / scenario reference was
  renamed and generalised from OpenAI-only to cover all three providers (OpenAI,
  Anthropic, Gemini) — correcting the API-key resolution and JSON-mode sections —
  and four scenarios were added: newer vs. stale build-file overlay, safe-mode
  resume with a completed output present, and malformed/foreign target files.
- `README.md`: documented the checkpoint recovery, `--safe-mode`, the
  `.codedoc_build.json` intermediate, the ownership guard, and the
  `CODEDOC_SAFE_MODE` environment variable; bumped the documented release to
  `0.7.2`.

---

**Behaviour on interrupt and resume (default — Checkpoint):**
1. User runs `codedoc run --entry src/main.py` on a 100-file project.
2. Run is interrupted (Ctrl-C, crash) after 60 files complete.
3. `.codedoc_progress.json` in the output directory holds all 60 results.
4. User re-runs the same command; 60 files are restored from the checkpoint (hash-verified),
   only the remaining 40 are sent to the LLM.
5. On clean completion the checkpoint file is deleted automatically.

**Behaviour on interrupt and resume (`--safe-mode`):**
1. User runs `codedoc run --safe-mode --entry src/main.py` on a 100-file project.
2. After every file, the output file is updated with the results so far.
3. Run is interrupted after 60 files; the output contains 60 complete file records.
4. User re-runs; the existing hash-based incremental logic detects all 60 files as unchanged
   and skips them automatically — only the remaining 40 are sent to the LLM.
5. On clean completion `write_project_outputs` overwrites the output with the final polished
   result (and `SafeWriter.delete()` removes the intermediate for MD-only runs).

## 0.7.1 - 2026-05-25

### Fixed: provider-specific default models not applied when `--model` is omitted (GitHub Issue #2)

- `codedoc/core/loader.py`: changed `DEFAULTS["model_name"]` from `"gpt-4o-mini"` to `""`.
- Previously, the global default `"gpt-4o-mini"` was a truthy string that short-circuited the `or` fallbacks in the provider factory for every provider. Running `--provider gemini` without `--model` would silently send requests to Gemini using the OpenAI model name `gpt-4o-mini`, causing a 404 from the Gemini API. The same bug applied to `--provider anthropic` without `--model`, which would have called Anthropic with `gpt-4o-mini` and failed.
- With an empty string default, the factory's per-provider fallbacks now activate correctly:
  - Gemini with no model → `gemini-2.5-flash`
  - Anthropic with no model → `claude-haiku-4-5-20251001`
  - OpenAI / auto with no model → `gpt-4o-mini` (unchanged)
- Behaviour when `--model` is explicitly passed is unchanged.

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
