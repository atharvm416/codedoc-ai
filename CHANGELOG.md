# Changelog

## 0.12.1 - Unreleased

### Response diagnostics and targeted correction

- **Structured response diagnostics.** Every rejected provider response now
  produces a bounded, structured diagnostic with a stable top-level reason code
  (`no_json_object`, `json_parse_error`, `top_level_not_object`,
  `no_usable_fields`, `missing_required`) and bounded per-field removal reasons
  (`unknown_field`, `wrong_type`, `empty_value`, `invalid_value`, `duplicate`,
  `item_limit`, `response_cap`, `not_requested`). Normal logs emit one concise
  rejection line; `--verbose` adds only bounded structural metadata. No raw
  response, source, prompt, or credential text is ever logged or attached.
- **Stronger exact-JSON prompt rules.** One shared clause block
  (`EXACT_JSON_RESPONSE_RULES`) is interpolated into all four documentation
  prompts so they cannot drift: return one JSON object only; no fences or prose;
  exactly the requested keys; no renamed keys; preserve each field's type; every
  required field non-empty and valid; omit optional data only as the schema
  allows; invent no facts.
- **Registry-driven acceptance.** A response is rejected when any effective
  registry-required field is missing or empty, or when none of the effective
  requested fields survive cleaning and profile filtering — enforced identically
  in single and triple modes.
- **Targeted response correction (opt-in, disabled by default).** New config key
  `response_correction_enabled` (strict boolean, default `false`, config-only —
  no CLI flag or environment variable). When enabled, an eligible response-contract
  failure receives at most one targeted correction call through the shared
  usage-counted provider path; the corrected text is revalidated through the
  identical acceptance path. Correction is invoked per agent per file and never
  reruns a successful sibling agent.
- **No duplicate whole-file retry for contract failures.** A response-contract
  rejection is classified `response_contract_final` and is non-retryable in both
  the parallel and sequential paths, whether correction is disabled or has been
  attempted. A correction call that fails on a rate-limit/transport fault ends
  correction for that file; a billing/credential/model correction fault stays a
  run-level abort with crash recovery left resumable.
- **Run-level correction statistics.** A thread-safe correction ledger reports
  `response_contract_failures` and correction call attempts/successes/failures in
  run statistics and the console summary. Each correction call is one paid attempt
  already counted in `attempted_calls`, so it stays a subset of
  `documentation_calls_attempted`; the existing call-category invariant is
  unchanged. `--dry-run` reports whether correction is enabled and a bounded
  worst-case call ceiling. These stay internal run metadata, absent from completed
  JSON/Markdown output.
- **Cache identity advances to `file-doc-v3`.** The rendered requested-shape block
  and its `_prompt_profile_digest` are unchanged, but the strengthened prompt
  semantics and stricter acceptance change generation strategy, so matching
  `file-doc-v2` records are reprocessed once.

## 0.12.0 - 2026-07-11

### Extension-scoped prompt profiles

- **Added extension-scoped prompt profiles.** The per-file override scope is
  `per_extension`, keyed by file extension.
  This is the only user-visible surface change.
- Each file's effective requested-shape block is resolved by
  `longest matching per_extension > common > built-in default`. Matching is on the
  file's lowercased basename, so multi-part suffixes work and the longest match
  wins (`.d.ts` beats `.ts`), matching is case-insensitive, and a file named
  exactly `.ts` is not treated as a `.ts`-suffixed file. An override is a complete
  replacement of the block, never a field-by-field merge.
- Each `per_extension` key must be a lowercase dotted suffix (at most 32
  characters) whose final segment is one of the project's configured extensions
  (`extension_language_map`); a silently dead override such as `.pyy` when only
  `.py` is configured is rejected. At most 64 overrides per mode. In triple mode
  each override carries all three agent keys, and a non-empty `triple.per_extension`
  requires `triple.common.documentation`.
- Cache invalidation is confined to files whose effective block changed: editing a
  used override reprocesses only the files that resolve to it, an unused override
  costs zero provider calls and invalidates nothing, and a profile with no
  `per_extension` renders byte-identical prompts and the exact prior
  `_prompt_profile_digest` (the digest scheme stays `pp-v1`). Identical-content
  reuse honours the destination file's extension scope.
- The mandatory standards/safety review now covers each profile block reachable
  by a planned file (the common block and each reachable `per_extension` override),
  deduplicating byte-identical blocks so equivalent scopes are reviewed once.
  `SAFE`/`RISKY`/`TOO_RISKY` semantics and the confirmation contract are unchanged.
- **`crash_recovery.json` keeps recovery identity version 1** while dropping its
  profile-wide compared field. Narrowing the compared identity field set is
  backward-compatible, so no existing recovery file is invalidated by the upgrade.
  Each recovered completed record is instead re-validated individually against the
  current per-file `_prompt_profile_digest`, so an unrelated profile edit or a
  newly added file no longer discards a resumable run.
- The generated config (`--init-config`) and the regenerated `codedoc.config.json`
  now emit `per_extension: {}` in each mode section. No
  new CLI flag, environment variable, or top-level config key. Completed JSON /
  Markdown output shape and `schema_version` are unchanged. OpenAI, Anthropic, and
  Gemini preserve equivalent prompt-profile and mandatory-review behaviour.

## 0.11.9 - 2026-07-09

### Completed JSON contract cleanup

- Removed transitional top-level `run`, `_codedoc`, and `project` blocks from
  newly written completed `codedoc.json` output. New completed JSON uses
  `last_run` as the single run/project summary, with `last_run.entry_file`
  carrying entry identity.
- Kept backward-compatible readers for older completed JSON and Markdown that
  still contain `_codedoc`, `project`, and `run`, so existing outputs remain
  reusable for incremental runs and safe overwrites.
- Kept internal `_codedoc` recovery metadata in `crash_recovery.json` and kept
  the hidden Markdown ownership comment. Those are internal ownership/recovery
  markers, not completed JSON user-facing blocks.

## 0.11.8 - 2026-07-09

### Truthful run and project metadata

- Added a canonical `last_run` block whose field names match what CodeDoc
  actually counted: LLM-documented files, failed files, unchanged cache reuse,
  identical-content reuse, unattempted files, recovery-resumed files, selected
  files, scanned files, entry source, documentation scope, and analysis mode.
- Retained the legacy `run` block unchanged for one compatibility window, so
  older CodeDoc builds can still recognize and safely overwrite newer output.
  New consumers should read `last_run` and fall back to `run` only when parsing
  documents produced before 0.11.8.
- Corrected newly rendered Markdown and legacy summary labels. The old
  `Files reused from cache` label was attached to identical-content reuse, not
  unchanged cache reuse; new output renders `Files reused (unchanged)` and
  `Files reused (identical content)` instead. The legacy parser still accepts
  the old labels when reading older Markdown.
- Documented the `_codedoc` ownership envelope and private underscore-prefixed
  file-record keys.

## 0.11.7 - 2026-07-09

### CRLF-compatible legacy Markdown reading

- Reading a CodeDoc Markdown document now canonicalizes CRLF and bare-CR line
  endings to LF in memory before the visible-text parser runs. A legacy
  Markdown file with Windows line endings — produced by any `core.autocrlf`
  checkout or a Windows-written `codedoc.md` — is now parsed identically to its
  LF form, so ownership checks, incremental reuse, and JSON/Markdown conversion
  no longer silently return zero files on such input. The normalization is
  in-memory only: no file on disk is rewritten.
- The change is confined to the reader. Valid embedded base64 views remain
  authoritative, absent embedded views still use the legacy visible parser,
  invalid embedded views keep their fail-closed rejection, and the base64
  payload bytes are untouched. Normalization only changes behaviour for
  documents whose LF-equivalent form is already accepted as CodeDoc-owned;
  malformed metadata, invalid embedded views, schema-less visible-only
  documents, and marker-less foreign Markdown remain rejected. No prompt,
  provider call, configuration, CLI surface, schema, cache identity, recovery
  identity, or generated output format changes.
- Test fixtures are now pinned to their committed bytes via a repository-root
  `.gitattributes` (`tests/fixtures/** -text`), making the suite deterministic
  across platforms and `core.autocrlf` settings.
- Completed output and crash-recovery writes now use LF line endings on every
  platform. This makes generated text bytes deterministic across Windows,
  macOS, and Linux while the reader remains newline-agnostic for existing CRLF
  documents.
- The legacy `codedoc.core.output.write_summary()` helper now writes a minimal
  owned CodeDoc Markdown document, so its `codedoc.md` can be safely overwritten
  by a later normal pipeline run.

## 0.11.6 - 2026-07-05

### Safe cross-format incremental reuse

- A single-format run now reuses the exact opposite-format CodeDoc sibling when
  the requested target does not yet exist. Switching from Markdown to JSON, or
  JSON to Markdown, converts unchanged records without repeating provider calls;
  only changed or cache-incompatible files are documented again.
- An existing requested target remains authoritative. Fallback probes only the
  expected sibling in the selected output directory and validates ownership and
  structure before reuse; foreign or malformed fallbacks stop before paid work.
- Named outputs use the same stem (`report.json` / `report.md`), while directory
  outputs use their configured JSON/Markdown filename pair. No directory walk,
  unrelated default filename, or modification-time selection is performed.
- Compatible partial recovery state overlays the stable sibling, and
  `crash_recovery.json` is removed only after the requested output commits
  successfully. Both-format conflict checks, versionless output, legacy readers,
  and ordinary cache-identity rules remain unchanged.

## 0.11.5 - 2026-07-04

### Versionless user-facing data

- Newly generated prompt-profile configuration, final JSON and Markdown
  documentation, embedded Markdown project views, and crash-recovery files no
  longer expose `schema_version`.
- Existing supported versioned prompt profiles and documents remain readable;
  current versionless documents use the CodeDoc ownership envelope plus strict
  structural validation.
- Provider-requested shapes and cleaned responses remain versionless, while
  project identity, incremental reuse, recovery matching, and lossless
  JSON/Markdown conversion retain their deterministic non-version checks.

## 0.11.4 - 2026-07-04

### Fail-closed configuration and public-surface hardening

- Unknown, malformed, non-finite, or incorrectly typed configuration now fails
  before scanning, provider creation, mutation, or paid documentation work.
- Configuration authoring is consolidated to `codedoc --init-config [--force]`.
  The schema-description, instruction-only, and named-template forms are removed;
  forced regeneration validates the active file and replaces only
  `prompt_profiles`, preserving every unrelated top-level value.
- Generated profiles expose one editable combined block for single mode and
  independently editable structure, dependency, and documentation blocks for
  triple mode. Security review runs only for effective non-default instructions
  headed to planned LLM calls; medium risk now requires explicit per-run
  confirmation, while high risk remains non-overridable.
- Documentation and help consistently teach the shorter `codedoc …` spelling;
  the accepted `run` and `execute` aliases remain supported. README now covers the
  complete CLI and environment-variable surfaces, and release-tethered prose was
  removed from runtime code and public guides.
- Recovery messages name `crash_recovery.json`, entry-selection help reflects
  auto-detection/all-files behavior, and the bundled Actions template uses the
  canonical command form.
- Documentation output shape, prompt bytes, document schema, and cache/recovery
  identity are unchanged outside the explicit validation, initialization, and
  medium-risk confirmation behavior above.
- This is the first official-PyPI release in the 0.11 line. Users upgrading from
  0.10.x should review the 0.11.0–0.11.3 entries below for cumulative breaking
  changes.

## 0.11.3 - 2026-07-02

### Config-only instructions and exact file lifecycle

This release consolidates the unpublished 0.11 instruction feature into a small,
predictable surface. It intentionally favors the clean contract over
compatibility with the internal 0.11.0/0.11.1 experiments.

- **One config file.** CodeDoc automatically reads exactly
  `<project_root>/codedoc.config.json`. The `config.json` fallback, `.env` file
  loading, external prompt-profile files (`codedoc-prompt-profiles.json`),
  auto-detection, and `--prompt-profile` / `--no-prompt-profile` are removed. OS
  environment variables remain supported for credentials and scalar overrides.
  `.env.example` is deleted and the `python-dotenv` dependency is dropped.
- **`common` instruction envelope.** Every mode section now uses
  `<mode>: { "common": {...}, "per_language": {...} }` for both schema v1
  (`fields`) and v2 (`requested_shape`). The flat 0.11.0/0.11.1 layout produces a
  deterministic migration error. The `per_extension` scope is reserved
  (precedence `per_extension > per_language > common`) but rejected as unsupported
  in 0.11.3.
- **`codedoc --init-config [NAME] [--force]`** writes a complete, editable
  `codedoc.config.json` with every public default plus single/triple instructions;
  a `NAME` writes a non-active help template. **`codedoc --init-instructions
  [single|triple|both] [--force]`** writes/replaces only the inline
  `prompt_profiles`. Neither keeps a backup; `--force` replaces atomically.
  `--export-prompt-profile` is removed (covered by the initializers and
  `--describe-prompt-schema`).
- **Deterministic documentation fallback.** A customized single-only profile in
  triple mode resolves its documentation block by projecting the compatible
  `single.common` fields (`description`, `role_in_system`, `key_concepts`,
  `usage_example`) — missing triple documentation resolves as explicit → projected
  single → built-in. The paid single-to-triple routing conversion is removed.
- **Mandatory, non-overridable security review.** `TOO_RISKY` always blocks;
  `--allow-risky-prompt-customization` / `prompt_customization_allow_risky` are
  removed.
- **Exact output and one recovery file.** Incremental state comes only from the
  exact selected `codedoc.json` / `codedoc.md` target(s); no sibling, opposite-
  format, or directory discovery. `both` mode fails with an actionable error when
  the two documents' identities conflict. Crash recovery is exactly
  `<output_dir>/crash_recovery.json` with a versioned run identity — no candidate
  walk or numbered suffixes. An owned in-progress recovery whose identity does not
  match the current run blocks with a "delete crash_recovery.json to start fresh,
  or restore the prior configuration" message instead of silently resuming.
- **Removed auxiliary artifacts.** The persistent `error.log`, the managed output
  `.gitignore` (`manage_output_gitignore` / `output_gitignore_filename`), and all
  probing/migration of legacy `.codedoc_progress.json`, `.codedoc_build.json`, and
  `codedoc_db.json` are removed. Diagnostics are printed and embedded in the final
  output; issues are kept in memory only. The deprecated `--safe-mode` /
  `CODEDOC_SAFE_MODE` no-op is removed. A stale config that still sets any removed
  key now fails with a targeted error naming the key and its replacement behavior.
- No change to the public document schema (`SCHEMA_VERSION` `1.4`), cache identity
  (`ANALYSIS_REVISION` `file-doc-v2`, `pp-v1`, `truncate-v1`,
  `no-prompt-profile-v1`), default prompt bytes, or OpenAI/Anthropic/Gemini parity.

## 0.11.2 - 2026-06-30

### Version-reference decluttering

- Removed historical, changelog-style version tags from source comments and
  docstrings — leading `# X.Y.Z:` markers, `(Workstream …)` / `(Work Item …)`
  provenance labels, parenthetical `(0.X.Y …)` module/section tags, and inline
  "since / before / as of X.Y" references — so the code reads as a description of
  current behavior. This CHANGELOG remains the single record of version history.
- Removed "available since `<version>`" framing from `README.md` prose: settings
  and feature section headers, the run-determinism and one-time-reprocessing
  notes, and similar dated phrasing, keeping all descriptive content.
- Documentation/comment-only release. No executable logic, public document schema
  (`SCHEMA_VERSION` `1.4`), cache identity (`ANALYSIS_REVISION` `file-doc-v2`,
  `pp-v1`, `truncate-v1`, `no-prompt-profile-v1`), prompt bytes, model ids, or
  supported Python versions changed. The full test suite, Ruff, `compileall`, and
  the README/default-block drift guards pass with the same pass/skip counts as
  0.11.1.

## 0.11.1 - 2026-06-29

### Config-embedded AI output structures

- Added a literal version-2 `requested_shape` prompt-profile format whose JSON
  keys and containers resemble the desired output, so output customization can
  live entirely inside `codedoc.config.json` without a second profile file. The
  legacy version-1 `fields` format, external profile files, auto-detection, and
  `--prompt-profile` remain fully compatible. `schema_version` is optional and
  inferred from the block syntax (version 1 from `fields`, version 2 from
  `requested_shape`); version 2 is accepted only inline. Equivalent version-1 and
  version-2 profiles render an identical prompt and share the same
  `_prompt_profile_digest`, so existing caches are reused unchanged.
- Added a deterministic strict-JSON loader (`codedoc/utils/json_utils.py`) shared
  by the config loader, external-profile reader, and the security/routing control
  responses; it rejects duplicate object keys at every nesting depth.
- Added an opt-in single-to-triple conversion proposal: selecting `triple` mode
  with only a *customized* `single` structure runs the ordinary paid security
  review plus one separately disclosed paid routing call (new
  `PromptProfileRoutingAgent`), prints a config-ready `triple` proposal, and stops
  without generating documentation. The proposal is bounded, deterministically
  validated, fail-closed, and never written to config automatically. A
  developer-standard-equivalent single structure in triple mode resolves to the
  built-in defaults with no paid call.
- Added separate documentation/security-review/routing call-attempt accounting
  that reconciles exactly to the aggregate attempt total, plus dedicated CLI
  conversion summaries and a config-ready stdout export
  (`--export-prompt-profile`); `--describe-prompt-schema` now documents both
  formats and rejects `--format both`. Path export still emits a version-1
  external profile.
- Reduced duplicate dependency-cycle warning noise: the cycle count is logged at
  WARNING at most once per run and the full path list at DEBUG, with no change to
  topological ordering or cycle handling. The public document `SCHEMA_VERSION`
  (`1.4`), `ANALYSIS_REVISION` (`file-doc-v2`), output vocabulary, and no-profile
  prompt/cache behavior are unchanged.


## 0.11.0 - 2026-06-28

### Mode-based JSON prompt profiles

- Added one canonical registry for the requested JSON shapes used by `single`
  and `triple` analysis. Profiles may reorder registered fields, rewrite their
  bounded instruction strings, omit optional fields, and select per-language
  overrides without changing system prompts, fixed rules, cleaner types, parser
  facts, provider configuration, or output paths.
- Added inline, explicit-file, and root auto-detected profile resolution plus
  `--prompt-profile`, `--no-prompt-profile`, strict environment booleans, schema
  description, and safe default-profile export utilities.
- Active customization that will reach planned documentation calls receives a
  mandatory, bounded, paid semantic standards/safety review through the configured
  provider. `SAFE` proceeds, `RISKY` proceeds with warnings, and `TOO_RISKY` blocks
  unless the explicit risky-customization override is set. Deterministic validation
  and strict response cleaning are never overridable.
- Added a shared post-clean filter and `_prompt_profile_digest` private cache
  identity so omitted fields cannot reappear and only affected files are
  invalidated. The public schema remains `1.4`; default/no-profile prompts and
  legacy cache reuse remain compatible.


## 0.10.3 - 2026-06-25

### Truncation parameters now participate in cache identity

A patch release that fixes a cache-invalidation gap left by the configurable truncation
controls. Before 0.10.3, changing `max_content_chars` or `truncation_head_ratio` altered the
truncated prompt sent for an oversized file but did **not** invalidate that file's cached
record, so an incremental re-run silently reused stale documentation — the exact remedy the
truncation warning recommends ("raise `max_content_chars`") had no effect on a cached run.

- **`_max_context_revision` cache-identity key (`codedoc/core/record_meta.py`).** A new private
  per-file cache-identity key encodes the effective `max_content_chars` ceiling and
  `truncation_head_ratio` (e.g. `truncate-v1:max=12000:head=0.7000`) for any file large enough
  to be truncated. A file that fits within the ceiling carries no value and stays reusable
  across ceiling/ratio changes — only files whose prompt is actually truncated are affected.
  The key joins the single centralized reuse predicate alongside `_analysis_revision` and
  `_analysis_mode`; no second reuse path is introduced.

- **Precise, minimal invalidation.** Changing `max_content_chars` or `truncation_head_ratio`
  reprocesses exactly the files large enough to be truncated and no others; files that fit the
  ceiling remain byte-for-byte reusable. On the first run after upgrade, legacy oversized
  records (which lack the key) are reprocessed once; every other record is reused.

- **Consistent character counting.** Planning computes each file's character count exactly as
  the orchestrator does (`utf-8-sig`, `errors="replace"`, via a shared `read_source_text`
  helper in `codedoc/core/db.py`), with a cheap byte-size short-circuit so files within the
  ceiling are never re-read. `_analysis_revision` and `SCHEMA_VERSION` are not bumped; the
  public document schema is unchanged (`schema_version` stays `1.4`).

Existing configuration, API, CLI, and output formats are fully backward-compatible.

## 0.10.2 - 2026-06-24

### Dependency projection fix, configurable truncation ratio, and error classifier extraction

A patch release that completes three targeted improvements identified during 0.10.1 review.
The public document schema is unchanged (`schema_version` stays `1.4`). `SCHEMA_VERSION`
is not bumped. `_analysis_revision` is not bumped — the truncation ratio is user-controlled
and explicit; the dependency projection fix only changes which deterministic source is used,
not prompts, cleaned record shape, or provider-facing analysis semantics.

- **Parser-authoritative dependency projection for non-React languages (Workstream C).**
  0.10.1 made Python parser-authoritative but kept JS/TS/Dart/Java/etc. on model-provided
  external dependencies. The global rule is now: every parser that emits complete imports is
  authoritative for public external/SDK links, with an explicit exception for the React/Node
  family (`js`, `jsx`, `ts`, `tsx`), whose parser deliberately omits bare npm packages. For
  Python and generic-parser languages (Dart, Java, Kotlin, C#, Swift, Go, Ruby, Rust, C/C++,
  HTML), public links are derived from the imports that did not resolve to an internal project
  file via graph resolution — so relative Dart imports and same-directory file references never
  leak as bogus external dependencies. Dart SDK imports (`dart:io`) classify as SDK; Dart
  package imports (`package:record/record.dart`) canonicalize to the package name (`record`).
  React/Node bare npm dependencies continue to come from model `_deps.external`. Single and
  triple modes now produce byte-identical external/SDK links for identical source for Python
  and generic-parser languages; React/Node remains model-assisted for bare npm packages.

- **Configurable truncation head ratio (Workstream A).** The 70/30 head-plus-tail split
  introduced in 0.10.1 is now user-configurable via `truncation_head_ratio` in the config
  file (or `CODEDOC_TRUNCATION_HEAD_RATIO` environment variable, or `--truncation-head-ratio`
  CLI flag). The default stays 0.70, producing byte-identical output to 0.10.1 when not set.
  Invalid values (0.0, 1.0, outside (0,1), booleans, non-numeric strings) are rejected before
  provider creation. The dry-run token estimate uses the same configured ratio so the planning
  output reflects the actual truncated input.

- **Error classifier extraction (Workstream B).** Signal constants and pure classification
  functions (`_classify_failure`, `_build_terminal_abort`, `_parse_retry_after`, etc.) are
  now in `codedoc/core/error_classifier.py`. `execution.py` is reduced from ~1150 to ~770
  lines. Deprecated compat re-exports remain in `execution.py` for one release. No behavior
  change; this is a structural-only decomposition following the 0.9.4 pattern.

Existing configuration, API, CLI, and output formats are fully backward-compatible. All
compat re-exports added in this release will be removed in a future release.

## 0.10.1 - 2026-06-23

### Output diagnostics, Windows write resilience, and deterministic enrichment

A patch release that makes local output failures actionable and less likely to
interrupt a paid run, and corrects verified cross-mode enrichment inconsistencies,
without weakening CodeDoc's atomic-write, crash-recovery, ownership, factuality, or
credential-safety guarantees. The public document schema is unchanged
(`schema_version` stays `1.4`).

- **Actionable local I/O diagnostics.** A new private `codedoc/core/io_diagnostics.py`
  classifies a local write failure into a stable category (`locked`, `permission`,
  `missing_parent`, `is_directory`, `no_space`, `read_only`, `io`, `serialization`)
  and formats a concise, secret-free cause from OS metadata only (exception class,
  Windows `winerror`, portable `errno`, OS reason text) plus the affected local path.
  No API keys, prompts, source contents, or provider responses ever appear in a
  message. `LiveBackupWriteError` now carries this cause and a resume hint.
- **Bounded transient-lock retry on atomic replace.** `atomic_write_text()` retries
  only the final `Path.replace` step, and only for a Windows sharing/lock violation
  (`winerror` 32/33), using a fixed sub-second delay sequence
  (`ATOMIC_REPLACE_RETRY_DELAYS_S = (0.05, 0.15, 0.30)`). It reuses the already
  flushed/fsynced temporary file, never recreates provider output or re-calls the
  provider, and never retries `ENOSPC`, read-only media, missing-parent, directory
  collisions, serialization failures, or non-lock permission denials. A successful
  retry leaves no temporary sibling; an exhausted one removes the temp file and
  raises the original failure with its cause intact.
- **Output accessibility preflight.** A real run validates that the output directory
  can be created, written (UTF-8), flushed, fsynced, atomically renamed, and cleaned
  up — using uniquely named probe files — after planning and the paid-file cap but
  **before** any provider is created. It never overwrites a user file, leaves no
  probe artifacts, and is skipped entirely for `--dry-run`. The authoritative
  recovery-target check remains `SafeWriter.initialize_empty()`.
- **Current-run error-log lifecycle.** `error.log` now begins with a stable ASCII
  ownership marker (`# codedoc-ai issue log`; the legacy `codedoc issue log` header is
  still recognized). A clean, issue-free run removes a stale CodeDoc-owned log so a
  historical failure no longer looks current; a foreign file at that path is left
  byte-identical and never deleted, truncated, or overwritten. `ErrorReporter.flush()`
  is now routed through the canonical atomic writer. A fatal `LiveBackupWriteError`
  records best-effort diagnostics and prints the recovery and error-log paths without
  letting a log-write failure mask the primary error.
- **Deterministic dependency projection.** Public dependency links no longer come
  from model type labels, so `single` and `triple` modes produce identical links for
  identical source. For Python the projection is fully parser-authoritative (parser
  imports + finalized graph edges; project imports that resolve to a graph edge are
  never mislabeled external; model output can never add, remove, or reclassify a
  link). For languages whose parser intentionally omits third-party package
  specifiers (e.g. JS/TS), the external/SDK set is taken from the model's reported
  dependencies and canonicalized by the same deterministic classifier; this preserves
  real non-Python dependency information rather than dropping it. Model
  `catalog_updates` / `dependency_refs` / `usage_notes` remain enrichment-only.
- **Shared strict enrichment and aligned prompts.** Single-mode response cleaners
  moved to `codedoc/agents/response_cleaning.py` and now also clean the triple-mode
  `StructureAgent` / `DependencyAgent` / `DocumentationAgent` subresponses, so both
  modes enforce the same documented keys, bounds, de-duplication, and boolean/malformed
  rejection. Prompts in both families share precise definitions: `functions`/`classes`
  are symbols *defined in* the file, `exports` are deliberately exposed names
  (including package re-exports), imported names are never relabeled as local symbols,
  and `usage_example` is included only when supported by the file's real public API —
  never a placeholder path.
- **Head-plus-tail source context.** Files larger than `max_content_chars` now send a
  leading *and* a trailing slice (~70/30) with the truncation marker between them,
  within the same character ceiling, so late class/function definitions and entry
  points are no longer invisible. The single shared helper is used by both modes and
  dry-run estimation, and at most one truncation warning is logged per file.
- **Cache identity advances to `file-doc-v2`.** Because prompt and cleaning semantics
  changed, `ANALYSIS_REVISION` advances from `file-doc-v1` to `file-doc-v2`. Existing
  0.10.0 outputs and recovery files remain readable, but matching `file-doc-v1`
  records are regenerated **once** under the corrected contract before reuse, so the
  first 0.10.1 run over an existing project re-documents previously cached files and
  incurs a one-time provider cost. Mode identity (`single`/`triple`) is still required.

Provider selection, call counts, retries, rate limits, usage accounting, the public
JSON/Markdown schema, and crash-recovery semantics are otherwise unchanged. Windows
improvements do not regress Linux or macOS behavior.

## 0.10.0 - 2026-06-22

### Selectable per-file call mode (default one call)

A feature release that makes normal analysis default to **one** validated provider
call per processed file, while keeping the legacy three-call path available as an
opt-in. The public record shape, pipeline scheduling, incremental reuse, rate-limit
handling, and output behavior are unchanged.

- **`analysis_mode` (`single` | `triple`).** New configuration key,
  `CODEDOC_ANALYSIS_MODE` environment variable, and `--analysis-mode {single,triple}`
  CLI flag. `single` (default) runs one combined `FileDocumentationAgent` call per
  file; `triple` runs the legacy StructureAgent/DependencyAgent/DocumentationAgent
  path (three calls). Validated at the loader; the CLI flag defaults to `None` so an
  absent flag never overwrites a config-resolved value, and absence from every source
  resolves to `single`.
- **Default call count goes from three to one.** `estimated_calls`, `planned_calls`,
  `disconnected_planned_calls`, and usage accounting all reflect the resolved mode
  via a single `initial_calls_per_file()` helper (one for `single`, three for
  `triple`), in both dry-run and real paths. Two new stats — `analysis_mode` and
  `initial_calls_per_file` — are reported on every stats path, and CLI summaries show
  them so the default one-call behavior is not silent.
- **The combined agent is provider-neutral.** OpenAI, Anthropic, and Gemini all run
  the one-call contract through their existing JSON mechanisms and the same
  `LLMError` / retry / rate-limit / usage-accounting / file-failure paths. Response
  cleaning is strict (unknown keys removed, malformed items dropped, booleans
  rejected, order-preserving de-duplication, named item/length caps, and a global
  response cap with a fixed lower-priority-first trim order).
- **Cache identity is revision- and mode-aware.** Records carry private
  `_analysis_revision` and `_analysis_mode` keys, registered in a dedicated
  `CACHE_IDENTITY_KEYS` registry. A single centralized predicate governs every reuse
  source (same-path, identical-content, live-backup, legacy checkpoint): reuse
  requires a matching content hash **and** every cache-identity key. Pre-0.10.0
  records (no revision) are reprocessed exactly once; a mode switch invalidates reuse.
- **Documentation-quality scope (deterministic only).** Deterministic fixture
  assertions verify that a correct combined response maps losslessly into the public
  record shape and the catalog/graph for Python, TypeScript/TSX, Dart, and Java
  fixtures. This checks plumbing, not live-model prose: the default change to
  `single` is an explicit cost/latency decision, and one-call live prose quality is
  **not** claimed to equal or exceed the three-call path.
- **Compatibility.** The three agent classes remain importable and power `triple`
  mode. `PipelinePlan.documented_rels` is now the canonical dataclass field, with
  `selected_rels` retained as a read-only delegating alias; the `entry_excluded`
  statistic is retained while CLI wording derives the excluded count from the
  reachable/disconnected counts. No `SCHEMA_VERSION` bump — the new keys are private
  and additive. `parallel_agents` / `--no-parallel` affects only `triple` mode and
  has no per-file effect in `single` mode.

## 0.9.9 - 2026-06-20

### Complete coverage and managed output

A feature release that adds an explicit documentation-scope choice, deterministic
entry-reachability data in the public output, and opt-in management of a
codedoc-owned block in the output directory's `.gitignore`. Defaults stay
behaviorally conservative: provider, prompts, response schema, retries,
concurrency, content ceilings, and cache identity are unchanged, and a default run
makes exactly the same provider calls as before.

- **`documentation_scope` (`entry` | `all`).** New configuration key and
  `--documentation-scope` CLI flag selecting coverage. `entry` (default) documents
  only files reachable from the entry, preserving current selection and cost; `all`
  is an explicit cost-bearing choice that documents every scanned source file,
  including disconnected ones. Validated at the loader (the safety net for values
  arriving from a config file) and again defensively at the discovery boundary. The
  CLI flag defaults to `None` and only enters the override dict when supplied, so an
  absent flag never overwrites a config-resolved value.
- **Scope is run configuration, not resume metadata.** It is never recovered from a
  prior output file: a later run with no override returns to the conservative
  `entry` default, so a previous `all` run can never silently make a later default
  run incur full-repository provider work. Switching `all` back to `entry` also
  drops stale disconnected records from the final output while retaining all
  eligible cache reuse. For repeatable full coverage, keep
  `documentation_scope: "all"` in config or pass `--documentation-scope all` each
  run.
- **`reachable_from_entry` on every public file record.** An additive boolean
  (`true` for files reachable from the entry, or all files when there is no entry;
  `false` for disconnected files included only by `all`). Recomputed at view-build
  time without provider calls, present in JSON and the lossless Markdown embed, and
  rendered as exactly one `**Reachable from entry:** Yes|No` line per file section.
  No schema bump — the field is purely additive.
- **New scope statistics.** `documentation_scope`, `entry_reachable`,
  `entry_disconnected`, `disconnected_paid_files`, and `disconnected_planned_calls`
  are populated on every stats path (dry-run, real, and early returns).
  `disconnected_paid_files` counts disconnected files initially routed to providers
  under `all`; `disconnected_planned_calls` reports planned initial calls (retries
  can increase actual calls). The compatibility `entry_excluded` statistic is
  retained. CLI dry-run and run summaries report scope, reachable/disconnected
  counts, and paid-file versus planned-call units distinctly.
- **Opt-in managed output `.gitignore` (`manage_output_gitignore`, default off).**
  When enabled, maintains a codedoc-owned block (`# >>> codedoc-managed … >>>` …
  `# <<< codedoc-managed <<<`) in the configured output ignore file
  (`output_gitignore_filename`, default `.gitignore`), listing only the stable
  final artifacts and diagnostics confirmed to exist after finalization — never a
  transient recovery/checkpoint file. Entries are root-anchored literal patterns
  with Git metacharacters (and trailing spaces) escaped, sorted and de-duplicated.
  User content outside the owned block is preserved; LF/CRLF is preserved. The
  target is validated for a portable filename, containment beneath the output
  directory, and non-symlink/non-directory, and is added to the existing
  artifact-path collision check before any scan or mutation. When disabled, the
  ignore file is never read for write, created, or modified.
- **Ignore management fails closed and never affects documentation.** A malformed or
  unsafe existing block, or any filesystem error, leaves the target byte-identical,
  is surfaced only as an auxiliary warning (`output_gitignore_warning`), and never
  marks the documentation run failed. New stable status keys
  (`output_gitignore_enabled`, `output_gitignore_updated`, `output_gitignore_path`,
  `output_gitignore_warning`) appear on every CLI-consumed stats dict.
- **Provider-neutral.** OpenAI, Anthropic, and Gemini receive the same file set for
  the same scope; scope is computed before provider dispatch and managed-ignore
  behavior never depends on provider or model.
- Internal: `_select_files` now returns `(reachable_rels, documented_rels,
  entry_rel)`; `PipelinePlan` gains a read-only `documented_rels` property over the
  preserved `selected_rels` storage field (the legacy name is unchanged);
  `build_project_view` and `write_project_outputs` gain an optional
  `reachable_rels` parameter that defaults to marking every file reachable for
  direct callers. New module `codedoc/core/ignore_manager.py` and generic
  owned-block helpers (`merge_managed_block`, `write_owned_block`, `BlockError`) in
  `codedoc/core/block_manager.py`, both delegating final mutation to the existing
  `atomic_write_text()`.

## 0.9.8 - 2026-06-20

### Dedicated crash-recovery file (corrective robustness patch)

A corrective robustness patch. It changes *where* in-progress work is staged on
disk and *when* the stable output is written; it does not change documentation
content, prompts, provider behaviour, concurrency, the response schema, file
selection, or the shape of the completed output. Successful runs make exactly the
same provider calls as before. No new configuration key, environment variable, or
CLI flag is added.

Previously, in `json` and `both` modes the live backup path *was* the final JSON
path: a new run wrote incremental in-progress records (with the `_crash_safety`
banner) straight into `codedoc.json`, so the moment a run started it overwrote the
user's last stable completed output — and an interruption left only a partial
in-progress document where the clean one had been.

- **In-progress records now go to a dedicated file**, `crash_recovery_<stem>.json`
  (derived from the final output stem), for **every** format — never the stable
  output. The stable completed output (`codedoc.json`, a named `--output` JSON, or
  the Markdown) is not opened, truncated, or mutated while a run is in progress.
- **The stable output is written once, on clean completion**, and only then is the
  recovery file deleted (write-stable-then-delete-recovery). If the stable write
  fails, the recovery file is preserved so the run stays resumable. A failure to
  delete the recovery file raises `OutputError`, leaves both the completed stable
  output and the recovery file intact, and is not reported as success.
- **Resume** combines the stable completed output (reuse baseline) with the active
  recovery file (in-progress overlay), and — for migrated Markdown runs — a legacy
  in-progress JSON sibling, in a fixed oldest-to-newest, whole-record precedence.
  Unchanged files are still skipped by content hash.
- **A present-but-unreadable/foreign recovery file is preserved**, not overwritten:
  the run advances to `crash_recovery_<stem>(2).json`, `(3).json`, … (bounded at
  1000 candidates → `OutputError`).
- **Backward migration is automatic.** A stable output left as an in-progress
  `_crash_safety` document by an earlier version is detected, used as a resume
  source, and migrated into the new layout with new writes going to a separate
  recovery file. No manual file deletion is required.
- **`--output` may not target a `crash_recovery_*` name** (`.json`/`.md`, including
  `(<n>)` forms); such a name is rejected with a `ConfigError` before any scan or
  mutation. The reserved prefix is a fixed internal constant.
- **CLI interrupt messaging** names the exact dedicated recovery file that enables
  resume (or truthfully reports that none was confirmed). Exit code `130` is
  unchanged.
- Internal: `validate_distinct_artifact_paths` now treats the recovery file as its
  own `live_backup` artifact and the `json_live_backup` self-alias is removed; no
  public API or output path is renamed. `--safe-mode` remains the accepted,
  deprecated no-op.

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
