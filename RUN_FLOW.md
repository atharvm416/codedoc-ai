# CodeDoc run lifecycle

This document describes the phase ordering CodeDoc uses for a real run.
The deterministic scanner, parser, graph, and output cleaners remain authoritative;
LLM output is bounded enrichment.

## Persistent-file allowlist

CodeDoc automatically reads or writes only:

1. `<project_root>/codedoc.config.json` for optional configuration and inline instructions;
2. `<output_dir>/crash_recovery.json` while a run is in progress;
3. the exact selected CodeDoc-owned JSON and/or Markdown final target, plus its
   deterministic opposite-format counterpart only when the selected target is
   missing.

There is no alternate-config, `.env`, external-profile, directory-wide output,
checkpoint, build, database, legacy-recovery, issue-log, or `.gitignore`
discovery. The counterpart uses the configured filename pair, or the same stem
for a named output; no directory walk or unrelated sibling is permitted.

## Ordered phases

1. **Load configuration.** Read only the exact `codedoc.config.json`, merge
   supported environment/in-memory scalar overrides, reject removed keys, and
   resolve exact final target paths. `crash_recovery.json` is a reserved final
   filename.

2. **Read-only output preflight.** Validate distinct artifact paths and ownership
   of every existing selected target. Load incremental records from the selected
   target, or strictly validate its exact opposite-format counterpart when the
   selected target is missing. A foreign or malformed fallback blocks before
   provider use. In both mode, compare schema, entry, exact path set, hashes, and
   normalized cache identity; any mismatch blocks before mutation/provider use.

3. **Instruction resolution.** Resolve `prompt_profiles` as inline or absent,
   validate schema v1/v2 under the required `common` envelope, choose single or
   triple mode, and build deterministic documentation projection when a
   single-only customization is selected in triple mode. No routing conversion
   exists.

4. **Scan and plan.** Scan source, construct the dependency graph, select entry
   reachability/documentation scope, calculate per-language profile digests, and
   build the versioned recovery identity.

5. **Exact recovery inspection.** Inspect only
   `<output_dir>/crash_recovery.json`. Missing means fresh state. A compatible
   owned in-progress document overlays selected-target or cross-format fallback
   records. Foreign, malformed, completed, unsupported, or identity-mismatched
   recovery blocks with guidance to restore the prior configuration or delete the
   exact recovery file.

6. **Final read-only gates.** Build the final plan from stable plus compatible
   recovery records, enforce `max_files`, and perform mandatory semantic review
   only for active customization that will reach documentation calls. SAFE
   continues, RISKY warns, TOO_RISKY blocks. Dry-run returns here and mutates
   nothing.

7. **Mutation boundary.** Create the output directory and run the provider-free
   create/write/fsync/atomic-rename/delete accessibility probe. Initialize
   `SafeWriter` with the exact recovery path, queue order, compatible records, and
   recovery identity. The empty in-progress recovery snapshot is written before
   provider creation/documentation calls.

8. **Execution.** Process dependencies before dependents where possible. Single
   mode makes one combined documentation call per file; triple mode runs
   structure, dependency, and documentation agents. Each completed record is
   atomically persisted to the fixed recovery file. Recoverable failures follow
   bounded retry/rate-limit rules; terminal/provider/persistence failures stop and
   preserve recovery.

9. **Finalization.** Render all selected payloads before replacement. Markdown is
   atomically replaced before JSON in both mode; this is per-artifact atomicity,
   not a cross-file transaction. Only after every selected final target succeeds
   is `crash_recovery.json` deleted. A failed final write or recovery deletion
   reports failure and leaves recovery available.

10. **Diagnostics.** Bounded issues remain in memory/terminal and in permitted
    final or recovery metadata. No persistent `error.log` is created.

## Cache and recovery identity

Per-file reuse uses the single centralized predicate over content hash and:

- `_analysis_revision`;
- `_analysis_mode`;
- `_max_context_revision`;
- `_prompt_profile_digest`.

The run-level recovery identity additionally binds project root, exact selected
targets, entry, documentation scope, analysis mode/revision, and sorted effective
profile digests by language. The run identity gates whether recovery may be
overlaid; it does not replace per-file reuse checks.

## Failure invariants

- Stable output is never mutated during analysis.
- Recovery is initialized only after all read-only gates and semantic review.
- Recovery is preserved on interruption, provider failure, and final-output failure.
- No unrelated sibling or legacy file is opened or deleted; an exact validated
  opposite-format counterpart is read-only.
- Dry-run performs no persistent mutation and contacts no provider.
