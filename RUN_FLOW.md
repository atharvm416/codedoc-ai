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
   validate schema v1/v2 under the required `common` envelope and the optional
   `per_extension` overrides, choose single or triple mode, and build
   deterministic documentation projection when a single-only customization is
   selected in triple mode. Each file's effective block is chosen by
   `longest matching per_extension > common > built-in default` on the lowercased
   basename. No routing conversion exists.

4. **Scan and plan.** Scan source, construct the dependency graph, select entry
   reachability/documentation scope, compute per-file profile digests by
   extension scope, and build the versioned recovery identity (which no longer
   binds a profile-wide digest).

5. **Exact recovery inspection.** Inspect only
   `<output_dir>/crash_recovery.json`. Missing means fresh state. A compatible
   owned in-progress document overlays selected-target or cross-format fallback
   records. Foreign, malformed, completed, unsupported, or identity-mismatched
   recovery blocks with guidance to restore the prior configuration or delete the
   exact recovery file.

6. **Final read-only gates.** Build the final plan from stable plus compatible
   recovery records, enforce `max_files`, and perform mandatory semantic review
   only for active customization that will reach documentation calls. SAFE
   continues, RISKY warns, TOO_RISKY blocks. Separately, derive deterministic
   non-blocking feasibility advisories for customized fields that appear to need
   cross-file context; these add no provider call and never alter the safety
   verdict. Dry-run returns here and mutates nothing.

7. **Mutation boundary.** Create the output directory and run the provider-free
   create/write/fsync/atomic-rename/delete accessibility probe. Initialize
   `SafeWriter` with the exact recovery path, queue order, compatible records, and
   recovery identity. The empty in-progress recovery snapshot is written before
   provider creation/documentation calls.

8. **Execution.** Process dependencies before dependents where possible. Single
   mode makes one combined documentation call per file; triple mode runs
   structure, dependency, and documentation agents. Immediately after a source
   file is read, a deterministic pre-check skips empty or whitespace-only content
   before parsing or any per-file agent call. The skip is not a failure, persists
   no placeholder record, and transactionally removes any stale preloaded record.
   Each non-skipped agent response passes one canonical validation path: JSON
   candidate extraction, parse, top-level object check, strict cleaning, profile
   filtering, a check that at least one requested field survives, and
   registry-required-field validation. A response that fails this contract is
   rejected with a bounded, structured diagnostic (a stable top-level reason code
   plus bounded per-field removal reasons) that carries no raw response, source,
   prompt, or credential text.

   When response correction is enabled (`response_correction_enabled: true`), each
   agent's response boundary makes at most one targeted correction call for its own
   eligible failure, revalidated through the identical path; a successful sibling
   agent is never rerun. When correction is disabled, the eligible failure is
   final at the initial call. Either way the failure is classified
   `response_contract_final` and is non-retryable, so a response-contract rejection
   never becomes a duplicate whole-file call. A correction call that fails on a
   rate-limit/transport fault ends correction for that file (still
   `response_contract_final`); a billing/credential/model correction fault stays a
   run-level terminal abort. Each completed record is atomically persisted to the
   fixed recovery file. Other recoverable failures follow the bounded
   retry/rate-limit rules governed by `file_retry_attempts`;
   terminal/provider/persistence failures stop and preserve recovery.

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
targets, entry, documentation scope, and analysis mode/revision. It no longer
binds a profile-wide digest — narrowing the compared field set is
backward-compatible and keeps recovery identity version 1. The run identity gates
whether recovery may be overlaid; each overlaid record is still re-validated by
the per-file reuse checks above, so `_prompt_profile_digest` selectively filters
recovered records by extension scope.

## Failure invariants

- Stable output is never mutated during analysis.
- Recovery is initialized only after all read-only gates and semantic review.
- Recovery is preserved on interruption, provider failure, and final-output failure.
- No unrelated sibling or legacy file is opened or deleted; an exact validated
  opposite-format counterpart is read-only.
- Dry-run performs no persistent mutation and contacts no provider.
