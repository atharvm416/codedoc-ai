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
   and Markdown filenames, and the live-backup path
   (`_resolve_live_backup_path`).

2. **Artifact-path collision check (read-only).** `validate_distinct_artifact_paths`
   rejects two distinct generated artifacts that would target the same
   normalized path, before any scan or mutation. See *Path aliasing* below for
   how the intentional JSON / live-backup alias is represented.

3. **Read-only preflight.** `inspect_output_ownership` checks that every final
   output target that already exists was produced by codedoc. A real run stops
   with a `ConfigError` on the first foreign-owned target; a dry run records the
   conflicts and reports them. The `ErrorReporter` is constructed in memory only
   (nothing is written until `flush()`).

4. **Scan and plan.** `scan_files` walks the project (skipping the output
   directory), the dependency graph and selection are built, and
   `build_pipeline_plan` produces the complete plan. A dry run returns its
   projected statistics here and performs no mutation.

5. **Paid-file safety cap.** After the full plan exists and before any mutation,
   writer initialization, or provider creation, a plan that exceeds the
   configured `max_files` limit stops with a `ConfigError`.

6. **Mutation boundary.** Everything below may write to the filesystem. The
   output directory is created, legacy artifacts are cleaned, and the live-backup
   `SafeWriter` is constructed and given the topological queue order.
   `SafeWriter.load()` performs the ownership check and pre-loads any existing
   records (raising `ConfigError` for a foreign file).

7. **Reuse / resume materialization.** Records routed by the plan as identical
   reuse or checkpoint reuse are materialized in memory. If no files need agent
   work, the run finalizes immediately (phase 10) and returns.

8. **Live-backup initialization, then provider creation.** `initialize_empty()`
   flushes the in-progress backup banner to disk *before* the LLM provider is
   created. A live-backup write failure here raises `LiveBackupWriteError` before
   any provider exists, so initialization failure makes **zero** provider calls.
   Only after the backup is initialized is the provider created and the
   orchestrator built.

9. **Execution.** `execute_agent_files` processes the queue (sequential or the
   parallel rate-limit ladder). Each completed file is persisted to the live
   backup from the worker via `SafeWriter.record()`. A live-backup persistence
   failure is fatal: it is never retried, never reclassified as a rate-limit or
   ordinary failure, and propagates after pending work is cancelled and running
   workers settle. Recoverable per-file failures (`ParseError`, `AgentError`,
   etc.) are handled by the retry logic and do not stop the run.

10. **Finalization.** `write_project_outputs` renders the complete payload(s) and
    atomically replaces the final target(s). See *Both-mode finalization* below.

11. **Diagnostics.** `ErrorReporter.flush()` writes `error.log` in the output
    directory when any issue was recorded.

12. **Cleanup.** For Markdown-only runs the live JSON backup sibling is removed
    after a clean Markdown write (`SafeWriter.delete()`). For JSON and both
    modes the live backup *is* the final JSON, so there is nothing to remove.

## Path aliasing (JSON and the live backup)

For `--format json` and `--format both`, the live backup and the final JSON are
the **same path**: the run writes the in-progress JSON throughout and the
finalization step overwrites it with the clean payload. This is intentional, not
a collision. The collision check in phase 2 therefore submits that single path
once under one logical artifact name, `json_live_backup`, so the alias is never
mistaken for two artifacts targeting one path.

For `--format md`, the live backup is a JSON **sibling** of the Markdown file
(e.g. `codedoc.json` next to `codedoc.md`). Markdown-only mode submits separate
`markdown` and `live_backup` artifacts to the collision check. The diagnostic
log is always submitted as `error_log`.

## Both-mode finalization (per-artifact atomicity)

`both` mode guarantees per-artifact atomicity, not a cross-file transaction:

1. The project view is built and **both** payload strings (JSON and Markdown) are
   rendered before either final target is mutated.
2. Markdown is replaced first (atomically).
3. JSON is replaced last (atomically), because the JSON path is also the live
   backup.

Consequences:

- If Markdown replacement fails, the previous JSON live backup is left intact.
- If the final JSON replacement fails after Markdown has succeeded, Markdown
  holds the new complete document while JSON remains the previous complete live
  backup.
- No target is ever truncated in place: each replacement writes a uniquely named
  temporary sibling, flushes and closes it, then renames it over the target via
  the canonical `atomic_write_text` helper.
