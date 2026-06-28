# codedoc run lifecycle

This document describes the verified phase ordering that `run_pipeline`
(`codedoc/pipeline.py`) follows for a real (non-dry-run) run. It reflects the
behavior implemented in the current source; it is not a roadmap and makes no
promises about future work.

The deterministic backbone (scan, dependency graph, selection) is the source of
truth. LLM prose is bounded enrichment layered on top. The lifecycle is ordered
so that no provider call and no filesystem mutation happens until the run is
known to be safe.

## Phases

1. **Configuration and path resolution.** Resolve the output directory, the JSON
   and Markdown filenames, and the **dedicated crash-recovery file**. The base
   recovery path is `crash_recovery_<stem>.json` (`_resolve_live_backup_path`,
   derived from the final output stem), and `select_active_recovery_path` then
   walks candidates (`crash_recovery_<stem>.json`, `…(2).json`, `…(3).json`, …)
   read-only to choose the active one: an absent name is a fresh run; a valid
   in-progress recovery document is resumed from; a completed / malformed /
   foreign file at a recovery name is preserved and skipped to the next suffix
   (bounded at 1000 candidates → `OutputError`). A `--output` filename whose own
   stem begins with `crash_recovery_` is rejected with a `ConfigError`.

2. **Artifact-path collision check (read-only).** `validate_distinct_artifact_paths`
   rejects two distinct generated artifacts that would target the same
   normalized path, before any scan or mutation. The selected recovery file is
   its own `live_backup` artifact, distinct from `json`, `markdown`, and
   `error_log`, in every format. When `manage_output_gitignore` is enabled, the
   resolved managed-ignore target (validated for portable filename, containment
   beneath the output directory, and non-symlink/non-directory) is added to this
   collision set so it cannot alias any other artifact.

3. **Read-only preflight.** `inspect_output_ownership` checks that every final
   output target that already exists was produced by codedoc. A real run stops
   with a `ConfigError` on the first foreign-owned target; a dry run records the
   conflicts and reports them. The `ErrorReporter` is constructed in memory only
   (nothing is written until `flush()`).

4. **Prompt-profile validation, then scan and plan.** Any inline, explicit, or
   auto-detected mode-based prompt profile is resolved and deterministically
   validated before scanning. `scan_files` walks the project (skipping the output
   directory) with an iterative, symlink-safe walk: deep trees cannot raise
   `RecursionError`, every directory's resolved identity is tracked so cycles
   and aliases are visited once, and — with the default `follow_symlinks=false`
   — symlinked directories and files are skipped so the scan never follows a
   link cycle or escapes the project root. The dependency graph is then built
   from purely lexical, exact-case import resolution (no filesystem probe, so
   the same repository yields the same graph on every OS). `_select_files` then
   computes two distinct sets from this precise graph: the **reachable** set
   (files transitively imported from the entry, or every scanned file when there
   is no entry) and the **documented** set (the reachable set under the default
   `documentation_scope="entry"`, or every scanned file under `all`).
   `build_pipeline_plan` plans over the documented set, while the reachable set
   feeds the additive `reachable_from_entry` field and the scope statistics
   (`entry_reachable`, `entry_disconnected`, `disconnected_paid_files`,
   `disconnected_planned_calls`). `documentation_scope` is validated at the
   loader and again defensively here; it is run configuration only and is never
   recovered from prior output. Planned-call statistics (`planned_calls`,
   `estimated_calls`, `disconnected_planned_calls`) are mode-aware: they multiply
   the agent-file count by `initial_calls_per_file(analysis_mode)` — one for the
   default `single` mode, three for `triple`. Reuse eligibility is decided by a
   single predicate over the content hash plus the cache-identity keys
   (`_analysis_revision`, `_analysis_mode`, `_max_context_revision`, and
   `_prompt_profile_digest`), normalized through one shared absent-default map so an
   omitted key and an explicit no-profile sentinel compare equal. A record whose
   revision, mode, truncation identity, or active prompt-profile digest no longer
   matches is reprocessed once. A dry run returns its projected statistics here and
   performs no mutation.

5. **Paid-file safety cap.** After the full plan exists and before any mutation,
   writer initialization, or provider creation, a plan that exceeds the
   configured `max_files` limit stops with a `ConfigError`.

   When active customization will reach planned provider calls, the plan also
   constructs complete deterministic review batches and reports their exact paid
   call count. Dry-run reports them as pending and makes no call. A real run
   completes every semantic standards/safety review batch here, before mutation.
   `SAFE` proceeds, `RISKY` proceeds with warnings, and `TOO_RISKY` blocks unless
   explicitly overridden. The verdict is probabilistic; deterministic validation
   and strict cleaners remain non-overridable.

6. **Mutation boundary.** Everything below may write to the filesystem. The
   output directory is created, then a **provider-free output accessibility
   preflight** (0.10.1) validates create → UTF-8 write → flush → fsync → atomic
   rename → cleanup using uniquely named probe files. It runs for every real
   finalization path (including all-reused runs), never for `--dry-run`, never
   overwrites a user file, and raises a classified `OutputError` before any
   provider is created if the directory is not writable. Legacy artifacts are
   then cleaned, and the
   crash-recovery `SafeWriter` (targeting the dedicated recovery file) is
   constructed and given the topological queue order. `SafeWriter.load()` is
   seeded with the merged reuse set computed by the canonical resume boundary
   (`_load_existing_file_docs`: stable completed output as the baseline, then the
   legacy in-progress overlay, then the active recovery overlay — whole-record,
   oldest-to-newest), so planning and the writer consume the same records and
   every partial flush of the recovery file is a self-contained resumable
   snapshot. The stable output is **not** opened or mutated here.

7. **Reuse / resume materialization.** Records routed by the plan as identical
   reuse or checkpoint reuse are materialized in memory. If no files need agent
   work, the run finalizes immediately (phase 10) and returns.

8. **Recovery initialization and provider reuse/creation.** `initialize_empty()`
   flushes the in-progress banner to the dedicated recovery file. When semantic
   review was not required, this still occurs before provider creation and a
   recovery-write failure makes zero provider calls. When review was required,
   the already-reviewed provider is reused; recovery initialization still occurs
   before any documentation call. The orchestrator is then built and given
   the resolved `analysis_mode` once: `single` (default) dispatches one combined
   `FileDocumentationAgent` call per file, while `triple` runs the three legacy
   agents (three calls); both modes produce the identical flat record and route
   provider exceptions through the same paths. If a `KeyboardInterrupt`
   propagates from here on, the pipeline attaches the exact selected recovery
   path to the exception (when the file exists) so the CLI can name it.

9. **Execution.** `execute_agent_files` processes the queue (sequential or the
   parallel rate-limit ladder). Each completed file is persisted to the dedicated
   recovery file from the worker via `SafeWriter.record()`. A recovery-write
   failure is fatal: it is never retried, never reclassified as a rate-limit or
   ordinary failure, and propagates after pending work is cancelled and running
   workers settle. Recoverable per-file failures (`ParseError`, `AgentError`,
   etc.) are handled by the retry logic and do not stop the run.

   **Unrecoverable-provider stop (0.9.7).** At every failure-handling site the
   same fixed precedence is applied to each error's message chain: a terminal
   billing/credit/credentials/model/access fault, or a bounded zero-progress
   rate limit, raises `UnrecoverableProviderError` and leaves execution; a
   request/context-too-large error is recorded as a failed file without a retry;
   everything else keeps the existing rate-limit or transient handling. The
   `UnrecoverableProviderError` carries a `category` (`"terminal"` or
   `"rate_limit_exhausted"`). Like the persistence-failure path, the abort
   cancels pending parallel work and propagates after running workers settle — it
   never writes final output, so the stable output stays untouched and the
   recovery file stays intact and resumable.

10. **Finalization.** `write_project_outputs` renders the complete payload(s) and
    atomically replaces the stable target(s) — the **first** time the stable
    output is written this run. See *Both-mode finalization* below. On an
    `UnrecoverableProviderError` this step is skipped entirely: the pipeline
    records and flushes the abort to `error.log`, then re-raises so the CLI can
    present a safe-stop message (exit `2` for `"terminal"`, exit `1` for
    `"rate_limit_exhausted"`) and the recovery file is preserved for resume.

11. **Diagnostics.** `ErrorReporter.flush()` atomically writes `error.log` in the
    output directory when any issue was recorded; the log begins with the
    `# codedoc-ai issue log` ownership marker. On a clean, issue-free run a stale
    CodeDoc-owned `error.log` left by a prior failure is removed so it no longer
    looks current (0.10.1); a foreign file at that path is left byte-identical and
    never deleted, truncated, or overwritten, and a removal failure is surfaced as
    an auxiliary `stale_log_warning` rather than failing the run. A fatal
    `LiveBackupWriteError` that escapes execution records best-effort diagnostics
    (target path, classified OS cause, traceback) and prints the recovery and
    error-log paths, without a secondary log-write failure masking the primary
    persistence error.

12. **Cleanup.** Only after the stable output is written, `SafeWriter.delete()`
    removes the dedicated recovery file — for **every** format. Order matters: if
    the stable write fails the recovery file must remain so the run is still
    resumable. A deletion `OSError` raises `OutputError` naming the recovery path
    and leaves **both** the completed stable output and the recovery file intact;
    the run is reported unsuccessful and the next invocation finalizes again. For
    a migrated pre-0.9.8 Markdown run, the leftover legacy in-progress JSON
    sibling is also removed here so only the Markdown remains.

13. **Managed output `.gitignore` (auxiliary, opt-in).** Only after every
    required stable output is written, diagnostics are flushed, and the recovery
    file is deleted, and only when `manage_output_gitignore` is enabled, the
    pipeline collects the basenames of the stable artifacts plus the diagnostic
    log confirmed to exist (never a transient recovery/checkpoint file) and
    rewrites a codedoc-owned block in the configured ignore file via
    `write_owned_block`. This is strictly auxiliary: a `BlockError`/`OSError`
    here is recorded as a warning (surfaced in `output_gitignore_warning`) and
    never marks the documentation run failed, never mutates content outside the
    owned block, and leaves malformed ownership byte-identical.

## The dedicated crash-recovery file (0.9.8)

In-progress (crash-recovery) records are staged in a dedicated
`crash_recovery_<stem>.json` (or a `(<n>)`-suffixed sibling), **never** the
stable output. For every format the stable completed output — the final JSON for
`json`/`both`, the Markdown for `md`/`both` — is not opened, truncated, or
mutated while a run is in progress; it is written once at clean completion, after
which the recovery file is deleted. An interrupted or failed run therefore leaves
the last stable output intact **and** a resumable recovery file.

The selected recovery file is its own `live_backup` artifact in the phase-2
collision check, distinct from `json`, `markdown`, and `error_log`. A foreign
file sitting at a recovery name is preserved and skipped by the candidate walk
(phase 1), not treated as a run-blocking conflict.

**Resume** combines, by project-relative path in a fixed oldest-to-newest
precedence: (1) the clean stable completed output as the reuse baseline; (2) a
legacy in-progress stable-path document (for Markdown runs, the pre-0.9.8 JSON
sibling); (3) the active dedicated recovery records. A path present in a later
source replaces the whole earlier record. A pre-0.9.8 stable output left as an
in-progress `_crash_safety` document is detected via
`read_codedoc_document(...).in_progress`, used as a resume source, and migrated
into the new layout automatically with new writes going to the separate recovery
file — no manual file deletion is required.

## Both-mode finalization (per-artifact atomicity)

`both` mode guarantees per-artifact atomicity, not a cross-file transaction:

1. The project view is built and **both** payload strings (JSON and Markdown) are
   rendered before either final target is mutated.
2. Markdown is replaced first (atomically).
3. JSON is replaced last (atomically).

Both stable targets are distinct from the dedicated recovery file, which is
deleted only after both stable writes succeed.

Consequences:

- If Markdown replacement fails, neither stable artifact reflects the new run and
  the recovery file is preserved for resume.
- If the final JSON replacement fails after Markdown has succeeded, Markdown
  holds the new complete document, JSON keeps its previous complete content, and
  the recovery file is preserved (not deleted).
- No target is ever truncated in place: each replacement writes a uniquely named
  temporary sibling, flushes and closes it, then renames it over the target via
  the canonical `atomic_write_text` helper.
