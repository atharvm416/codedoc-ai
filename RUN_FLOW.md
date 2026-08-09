# CodeDoc run lifecycle

This document describes the active phase ordering for ordinary and large-file
runs. Local scanning, parsing, graph construction, split planning, output
cleaning, and serialization are deterministic; provider output is bounded
enrichment.

## Availability boundary

The default `large_file_strategy: truncate` supports real `single` and `triple`
runs. Beginning in `0.14.2`, `large_file_strategy: split` supports provider-free
dry-run planning, paid execution, same-path completed reuse, and dependency-valid
node recovery with `analysis_mode: single`. As of `0.14.3`, `single + split`
execution, completed-record reuse, and node-level recovery are fully
supported; `triple + split` remains unavailable. Split leaf signatures stay
private, internal matching metadata bounded to the parser-aligned
600-character ceiling; split never silently truncates an over-bound response.

`triple + split` fails during configuration validation before scanning,
recovery inspection, output-directory creation, prompt-customization review,
provider construction, or provider calls. It never silently falls back to
truncate.

An exactly compatible same-path completed split record authorizes zero-call
reuse. Cross-path identical-content split reuse remains unavailable because
split identities are path-bound. Accepted leaf, reducer, and final nodes are
checkpointed only after cleaning and validation; a compatible interrupted run
resumes only unpaid nodes. An explicit force bypasses reuse/recovery for that
path while old stable output and recovery remain preserved until replacement.

The predecessor `0.14.1` `fresh-only-v1` split record is stale by default under
`0.14.2` and reruns once. Schema-1 and schema-2 partials are preserved but not
resumed. Rolling back to `0.14.1` reruns `large-file-v3` split output fresh and
blocks schema-4 recovery preserve-first.

The current node-keyed partial-recovery generation is schema 4 (`0.14.3`,
bound to the current `leaf-capsule-v6` leaf identity). Released schema 3
(`0.14.2`, `leaf-capsule-v5`) is now an unsupported predecessor generation,
preserved and blocked before any node is read, before planning, `SafeWriter`,
or provider construction — the recovery artifact stays byte-identical.
Nothing from a released schema-3 partial is carried forward; a fresh
`0.14.3` run performs complete v6 re-execution. Finish an unfinished `0.14.2`
split run with `0.14.2`, or move `crash_recovery.json` aside (or delete it as
a deliberate discard) to start fresh under `0.14.3`.

## Persistent-file allowlist

CodeDoc automatically reads or writes only:

1. `<project_root>/codedoc.config.json` for optional configuration and inline
   instructions;
2. `<output_dir>/crash_recovery.json` while a real run is in
   progress; and
3. the exact selected CodeDoc-owned JSON and/or Markdown final target, plus its
   deterministic opposite-format counterpart only when the selected target is
   missing.

There is no alternate-config, `.env`, external-profile, directory-wide output,
database, issue-log, or `.gitignore` discovery. A named output uses only its
configured counterpart; CodeDoc does not scan unrelated siblings.

## Split planning, reuse, and recovery

When `large_file_strategy` resolves to `split`, CodeDoc reads one canonical
decoded snapshot per selected source file. A file at or below
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
and reason and exits without mutation; a real run stops before provider creation
or persistent mutation. A genuine planning invariant failure is not converted
into a capacity result; it aborts planning while prior output remains untouched.

The split dry-run manifest reports ordinary-file, leaf,
unit-consolidation/general-reduction, and final-synthesis call categories. It
also reports a deterministic worst-case final-input estimate that reserves the
complete 3,000-character canonical ledger-synopsis allowance rather than using
one concrete trimmed ledger as a proxy. It is a character-based estimate rather
than a tokenizer-exact prediction. Dry-run stops after this provider-free plan
and does not consume checkpoints; a real run executes the same authorized
topology after completed reuse and dependency-valid recovery remove already-paid
work.

## Ordered phases for a real run

1. **Load configuration and instructions.** Read only the exact
   `codedoc.config.json`, merge supported environment and in-memory overrides,
   normalize strict values, reject unavailable strategy/mode combinations, and
   resolve/classify the effective prompt profile. Instruction resolution occurs
   before entry recovery, ownership inspection, stable-output reads, or source
   scanning.

2. **Resolve exact artifacts and inspect ownership.** Resolve entry information
   from configuration or the exact selected documentation, validate that JSON,
   Markdown, and recovery targets do not collide, then inspect ownership of the
   selected targets. If a selected format is absent, read only its exact
   opposite-format counterpart. Foreign, malformed, conflicting, or unsupported
   ownership blocks before provider use or mutation.

3. **Scan and select.** Scan supported source, construct the dependency graph,
   determine entry reachability and documentation scope, and freeze each
   provider-bound file from one source snapshot. Content, byte hash, effective
   language, and derived imports describe that same snapshot. A detected
   concurrent source change causes one complete source-dependent rebuild; a
   second change fails before accounting or provider construction.

4. **Inspect recovery and build the final read-only plan.** Inspect only the
   exact `<output_dir>/crash_recovery.json`. Compatible ordinary completed
   records may overlay stable output. Foreign, completed, unsupported,
   malformed, or run-identity-mismatched recovery blocks without mutation.
   Current schema-4 split nodes are validated topologically; valid siblings are
   retained, rejected nodes are quarantined, and their ancestors are pruned.
   Planning applies completed reuse, rejects insufficient source locally,
   divides oversized split files, and blocks any capacity failure before calls.

5. **Build the canonical manifest and enforce caps.** Derive review scopes only
   from files with unpaid work, then build one review/documentation call
   manifest. Enforce `max_files` and `max_planned_calls` before usage accounting,
   confirmation, writer initialization, or provider construction.

6. **Prompt-customization review.** If active non-default instructions will
   reach unpaid provider work, construct the provider and run the mandatory
   semantic review. A later output-accessibility failure states that this review
   may already have been billed; it does not claim the run was provider-free.

7. **Mutation preflight.** Create and probe the output directory through the
   classified create/write/fsync/atomic-rename/delete boundary. Permission and
   space failures become stable output errors rather than raw filesystem
   exceptions.

8. **Execution and recovery.** Initialize the one recovery file only when
   provider work remains, before documentation-provider construction. Process
   dependencies before dependents where possible. Ordinary single mode makes one
   combined call per file; triple mode makes three bounded calls. An oversized
   split file runs only unpaid nodes in its planned leaf/reduction/final tree.
   Every returned node is cleaned, schema-validated, and transactionally
   checkpointed before a dependent is scheduled. A fully recovered final is
   restored locally with no provider or review. Only completed file-level output
   reaches the public record. Response cleaning, optional correction, retry limits,
   terminal-provider handling, cancellation, and usage accounting share the
   canonical call boundary.

9. **Finalization.** Project every completed record through the public schema
   before deriving JSON, visible Markdown, embedded views, or lightweight
   metadata. Render selected payloads before replacement. Each artifact is
   replaced atomically; `both` mode is per-artifact atomic, not a cross-file
   transaction. Remove recovery only after every selected write succeeds.

10. **Diagnostics.** Keep bounded issues in memory and terminal output.
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

An oversized split result additionally carries the private current
`large-file-v3` topology/imports identity. The retired `_split_reuse_contract`
key is no longer stamped, but remains registered so literal `0.14.1`
`fresh-only-v1` records round-trip unchanged and compare stale. Completed cache
reuse is provider-agnostic. Partial node identity additionally binds provider,
model, and effective endpoint; imports-only changes preserve leaves/reducers and
rerun final synthesis. Cross-path split reuse remains unavailable.

Schema-4 recovery stores container provenance, ordered node state, exact
stage-local input digests, and bounded non-executable quarantine. Released
schema-3, schema-1, schema-2, unknown, foreign, duplicate, aliased, or unsafe
container state is preserved and blocked. The remedies are ordered
preserve-first: restore the matching version/configuration or move the file
aside; deletion is only an explicit discard.

## Call authorization and accounting

Dry-run and real execution derive counts from one canonical call manifest.
`max_planned_calls` is checked before provider construction and covers initial
logical calls, including mandatory prompt-review calls. Retries and corrections
are additional attempts attached to an existing logical call.

Every real run reconciles planned logical calls with attempted calls.
`planned_calls_not_attempted` may be non-zero after a bounded stop;
`additional_attempts` records retries and corrections. A clean completion,
allowed partial completion, and terminal abort use the same accounting model.

## Failure invariants

- `triple + split` fails before every side effect.
- Split dry-run performs no provider call and no persistent mutation.
- Exactly compatible same-path completed split records authorize zero-call
  reuse; cross-path split records do not.
- Current dependency-valid schema-4 nodes authorize only their own paid work;
  invalid nodes never authorize an ancestor and remain quarantined until valid
  replacement or completed output succeeds.
- Released schema-3, schema-1/schema-2, foreign, future, malformed, duplicate,
  or aliased recovery is preserved and blocked before provider construction or
  mutation.
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
- CodeDoc verbosity never raises the root logger or lowers reviewed provider,
  authentication, HTTP-client, or transport floors. Provider errors and public
  output never expose credentials, raw prompts, source text, or raw provider
  responses.
