# CodeDoc run lifecycle

This document describes the active phase ordering for ordinary real runs and
the separate provider-free large-file planning preview. Local scanning,
parsing, graph construction, output cleaning, and serialization are
deterministic; provider output is bounded enrichment.

## Availability boundary

The default `large_file_strategy: truncate` supports real `single` and `triple`
runs. `large_file_strategy: split` is accepted only when `dry_run: true` and
`analysis_mode: single`.

A real split request fails during configuration validation before scanning,
recovery inspection, output-directory creation, prompt-customization review,
provider construction, or provider calls. `triple + split` fails at the same
boundary. Neither request silently falls back to truncate.

A valid split dry-run may scan and construct a complete local division,
reduction topology, capacity result, and initial-call estimate. It never accepts
or reuses split completed state or partial recovery, never writes output or a
checkpoint, and never contacts a provider.

## Persistent-file allowlist

CodeDoc automatically reads or writes only:

1. `<project_root>/codedoc.config.json` for optional configuration and inline
   instructions;
2. `<output_dir>/crash_recovery.json` while an ordinary real run is in
   progress; and
3. the exact selected CodeDoc-owned JSON and/or Markdown final target, plus its
   deterministic opposite-format counterpart only when the selected target is
   missing.

There is no alternate-config, `.env`, external-profile, directory-wide output,
database, issue-log, or `.gitignore` discovery. A named output uses only its
configured counterpart; CodeDoc does not scan unrelated siblings.

## Provider-free split planning

When `large_file_strategy` resolves to `split` in a valid dry-run, CodeDoc reads
one canonical decoded snapshot per selected source file. A file at or below
`max_content_chars` remains one planned whole-file call. An oversized file is
divided at deterministic syntax boundaries when available, otherwise at
complete lexical line boundaries. An individually oversized semantic unit or
physical line receives deterministic continuation chunks.

Every source character belongs to exactly one planned leaf. Adjacent fitting
semantic units may share a planned leaf call while retaining their own
identities. Continuations for one semantic unit consolidate before general
reduction. General reduction continues only until the complete final manifest
fits; several ordered roots may feed the planned final synthesis.

Structure extraction is runtime-offline: it never downloads a grammar or
writes a grammar cache. Without the optional structure package, a matching
grammar, or a usable parse, planning uses one lexical atom per physical line.
The lexical path has a 4,096 lexical-atom ceiling and reports `atom-cap` when
that limit is exceeded. Because the cap counts atoms rather than characters,
raising `max_content_chars` cannot clear it. For a supported language currently
using lexical fallback, syntax-aware extraction may reduce the atom count:

```bash
pip install "codedoc-ai[structure]"
```

The package is a reason-specific option, not a promise to repair malformed or
error-dominated source. Refactoring is the reliable remedy when a usable syntax
parse is unavailable.

Planning measures the exact canonical JSON representation used by bounded leaf,
reducer, and final manifests, including quotes, backslashes, controls,
newlines, and Unicode. It evaluates these provider-free capacity reasons in
fixed order:

1. `atom-cap`;
2. `symbol-cap`;
3. `unit-cap`;
4. `chunk-cap`;
5. `reduction-envelope-cap`;
6. `reduction-fan-in-cap`;
7. `reduction-depth-cap`; and
8. `final-synthesis-envelope-cap`.

A blocked file contributes no provider call. Dry-run reports each blocked path
and reason and exits without mutation. A genuine planning invariant failure is
not converted into a capacity result; it aborts planning while prior output
remains untouched.

The split dry-run manifest reports ordinary-file, leaf,
unit-consolidation/general-reduction, and final-synthesis call categories. It
also reports a deterministic worst-case final-input estimate that reserves the
complete 3,000-character canonical ledger-synopsis allowance rather than using
one concrete trimmed ledger as a proxy. It is a character-based estimate rather
than a tokenizer-exact prediction. These are planning estimates only: split
execution, completed split output, reuse, checkpointing, and recovery are
unavailable.

## Ordered phases for an ordinary real run

1. **Load configuration.** Read only the exact `codedoc.config.json`, merge
   supported environment and in-memory overrides, normalize strict values, and
   reject unavailable strategy/mode combinations.

2. **Read-only output preflight.** Resolve exact targets and verify ownership of
   every existing selected artifact. If a selected format is absent, inspect
   only its exact opposite-format counterpart. Foreign, malformed, conflicting,
   or unsupported ownership blocks before provider use or mutation.

3. **Instruction resolution.** Validate `prompt_profiles`, resolve the effective
   single/triple shape for each extension, and build deterministic projection
   when a single-only customization is selected in triple mode.

4. **Scan and select.** Scan supported source, construct the dependency graph,
   determine entry reachability and documentation scope, and freeze each
   provider-bound file from one source snapshot. Content, byte hash, effective
   language, and derived imports describe that same snapshot. A detected
   concurrent source change causes one complete source-dependent rebuild; a
   second change fails before accounting or provider construction.

5. **Recovery inspection.** Inspect only the exact
   `<output_dir>/crash_recovery.json`. Compatible ordinary completed records may
   overlay stable output. Foreign, completed, unsupported, malformed, or
   run-identity-mismatched recovery blocks without mutation. Restore the prior
   configuration to resume, or move the recovery file aside to start fresh;
   deletion explicitly discards that state.

6. **Final read-only plan and caps.** Apply same-path and identical-content reuse
   only when the current hash, effective language, analysis identity, and prompt
   profile agree. Reject empty or whitespace-only source locally. Build the
   canonical review and documentation call manifest. Enforce `max_files` and
   `max_planned_calls` before usage accounting, confirmation, writer
   initialization, or provider construction.

7. **Prompt-customization review.** If active non-default instructions will
   reach unpaid provider work, construct the provider and run the mandatory
   semantic review. A later output-accessibility failure must state that this
   review may already have been billed; it must not claim the run was
   provider-free.

8. **Mutation preflight.** Create and probe the output directory through the
   classified create/write/fsync/atomic-rename/delete boundary. Permission and
   space failures become stable output errors rather than raw filesystem
   exceptions.

9. **Execution and recovery.** Initialize the one recovery file only when
   ordinary provider work remains. Process dependencies before dependents where
   possible. Single mode makes one combined call per ordinary file; triple mode
   makes its three bounded agent calls. Response cleaning, optional correction,
   retry limits, terminal-provider handling, and usage accounting share the
   canonical call boundary. Completed ordinary records are checkpointed
   atomically.

10. **Finalization.** Project every completed record through the public schema
    before deriving JSON, visible Markdown, embedded views, or lightweight
    metadata. Render selected payloads before replacement. Each artifact is
    replaced atomically; `both` mode is per-artifact atomic, not a cross-file
    transaction. Remove recovery only after every selected write succeeds.

11. **Diagnostics.** Keep bounded issues in memory and terminal output.
    Permitted hard-error summaries may appear in final output. CodeDoc does not
    create a persistent `error.log`.

## Cache and recovery identity

Ordinary per-file reuse uses one centralized predicate over content hash,
effective language, and the registered cache identity:

- `_analysis_revision`;
- `_analysis_mode`;
- `_max_context_revision`; and
- `_prompt_profile_digest`.

The run-level ordinary recovery identity additionally binds project root, exact
targets, entry, documentation scope, analysis mode/revision, and the effective
large-file strategy. Every recovered record is still revalidated by the
per-file predicate.

The split preview constructs private planning identities for deterministic
topology and tests, but it does not accept or publish a completed split identity
and does not inspect split partial recovery.

## Call authorization and accounting

Dry-run and ordinary execution derive counts from one canonical call manifest.
`max_planned_calls` is checked before provider construction and covers initial
logical calls, including mandatory prompt-review calls. Retries and corrections
are additional attempts attached to an existing logical call.

An ordinary real run reconciles planned logical calls with attempted calls.
`planned_calls_not_attempted` may be non-zero after a bounded stop;
`additional_attempts` records retries and corrections. A clean completion,
allowed partial completion, and terminal abort use the same accounting model.

## Failure invariants

- Real split and `triple + split` fail before every side effect.
- Split dry-run performs no provider call and no persistent mutation.
- Stable output is not mutated during analysis.
- Recovery is initialized only after read-only gates, caps, and any semantic
  review.
- Recovery is preserved on interruption, provider failure, and final-output
  failure.
- After interruption is observed, no new initial, retry, or correction call
  begins.
- Empty and whitespace-only files make no provider call and publish no
  placeholder.
- No unrelated sibling or legacy file is opened or deleted.
- Output replacement is ownership-guarded and atomic per artifact.
- Provider errors and public output never expose credentials, raw prompts,
  source text, or raw provider responses.
