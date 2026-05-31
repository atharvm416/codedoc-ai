# codedoc-ai — Full Run Flow & Scenarios

This document describes exactly how `codedoc` runs end-to-end across all three
supported providers — **OpenAI**, **Anthropic**, and **Gemini** — covering every
meaningful success, interrupt/resume, and failure scenario.

It is provider-agnostic: the pipeline, live-backup crash-safety, ownership
guard, and incremental logic are identical regardless of which LLM you use. Only
the per-file API call and the JSON-enforcement mechanism differ per provider, as
noted where relevant.

---

## 1. How a Provider Is Selected

### Explicit selection

```bash
codedoc run --provider openai    --entry src/main.py
codedoc run --provider anthropic --model claude-haiku-4-5-20251001 --entry src/main.py
codedoc run --provider gemini    --model gemini-2.5-flash --entry src/main.py
```

### Auto-detection rules (`factory.py → _resolve_api_provider`)

When `--provider` is omitted (or `auto`), the provider is inferred from the
model name prefix:

| Model prefix | Provider selected |
|---|---|
| `gpt-`, `o1`, `o3`, `text-` | OpenAI |
| `claude` | Anthropic |
| `gemini` | Gemini |
| (none / empty) | OpenAI (default) |

### API key resolution order (`factory.py → _provider_api_key`)

The key is resolved for whichever provider was selected. `LLM_API_KEY` is a
generic fallback that works for any provider.

| Provider | Environment variables checked, in order |
|---|---|
| OpenAI | `OPENAI_API_KEY` → `LLM_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` → `LLM_API_KEY` |
| Gemini | `GEMINI_API_KEY` → `GOOGLE_API_KEY` → `LLM_API_KEY` |

`config["api_key"]` (set in `codedoc.config.json`) takes precedence over all
environment variables. If no key is found → `ConfigError` is raised immediately
and the run is aborted before any file is touched.

### Default model per provider

If neither `--model` nor `config["model_name"]` is set, the provider's default
is used:

| Provider | Default model |
|---|---|
| OpenAI / auto | `gpt-4o-mini` |
| Anthropic | `claude-haiku-4-5-20251001` |
| Gemini | `gemini-2.5-flash` |

---

## 2. Full Pipeline Flow (Happy Path — First Run)

```
codedoc run --entry src/main.py
```

```
CLI (cli.py)
  │
  ├─ parse args → build overrides dict
  ├─ call run_pipeline(root, config_overrides)
  │
pipeline.py — run_pipeline()
  │
  ├─ load_config()               read codedoc.config.json + .env + env vars + overrides
  ├─ _resolve_entry_and_docs()   no prior docs found → entry_file set from --entry
  ├─ set_level(INFO)
  ├─ scan_files()                find all .py/.ts/.js/etc. files under root
  ├─ _build_graph()              parse imports from each file → DependencyGraph
  ├─ _select_files()             traverse graph from entry → reachable set
  │
  ├─ _load_existing_file_docs()  no codedoc.json → returns {}
  ├─ recorder = SafeWriter(live_backup_path)   (always, since 0.8.0)
  │     recorder.load()          no prior backup → ownership check passes
  │
  ├─ changed_rels = ALL files    (no existing docs → everything has hash mismatch)
  ├─ agent_rels = ALL files
  │
  ├─ recorder.set_queue_order(topological order)
  ├─ recorder.initialize_empty() → writes codedoc/codedoc.json with _crash_safety banner
  │                                  BEFORE the LLM provider is created
  │
  ├─ create_provider(config)     → OpenAIProvider / AnthropicProvider / GeminiProvider
  │
  ├─ _process_agent_files()      up to 5 files in parallel (default max_parallel_files)
  │     └── for each file (via _process_and_record in worker thread):
  │           _process_one_file()
  │             ├─ read file content
  │             ├─ parse imports
  │             └─ Orchestrator.process()
  │                   ├─ [Thread 1] StructureAgent   → LLM call (functions, classes, exports)
  │                   ├─ [Thread 2] DependencyAgent  → LLM call (imports, external deps)
  │                   │   (both run in parallel)
  │                   └─ DocumentationAgent          → LLM call (description, concepts, usage)
  │                                                    receives structure + deps as context
  │           → result dict assembled by Orchestrator._merge()
  │           → recorder.record(rel_path, result, file_hash)
  │               updates codedoc/codedoc.json atomically after every file
  │               (called inside worker thread — safe against Ctrl-C)
  │     └── on rate-limit: step down concurrency ladder, print WARNING to stdout
  │
  ├─ write_project_outputs()
  │     ├─ _check_file_ownership(codedoc.json)  → passes (codedoc owns it)
  │     ├─ build_project_view()
  │     └─ writes codedoc/codedoc.json (final clean — no _crash_safety banner)
  │
  ├─ recorder.delete()           no-op for JSON format (file already overwritten above)
  │                              removes JSON sibling for MD-only runs
  │
  └─ returns stats dict
       Files documented : N
       Files reused     : 0
       Files failed     : 0
       Output file      : codedoc/codedoc.json
       live_backup_path : .../codedoc/codedoc.json
```

**3 LLM calls per file.** A 20-file project makes 60 API requests total (the 3
agents run in parallel within each file; files themselves are processed in
batches of up to 5).

---

## 3. Per-File Agent Detail

For each file the `Orchestrator` fires three agents. Each agent requests
structured JSON. The **mechanism used to enforce JSON differs per provider**,
but the pipeline treats all three identically — the result is always a JSON
object parsed by `BaseAgent._parse_json`.

| Provider | JSON enforcement |
|---|---|
| OpenAI | `response_format={"type": "json_object"}` (native JSON mode) |
| Gemini | `response_mime_type="application/json"` (native JSON mode) |
| Anthropic | No native JSON-mode parameter — relies on strong text instructions plus the agent's JSON-extraction parsing |

```
File: src/auth/login.py
  │
  ├─ [Thread 1] StructureAgent.run()
  │     prompt: "Analyze the structure of this Python file..."
  │     LLM call → returns: { functions: [...], classes: [...], exports: [...], description: "..." }
  │
  ├─ [Thread 2] DependencyAgent.run()
  │     prompt: "Analyze the imports and dependencies..."
  │     LLM call → returns: { dependencies_analysis: { external: [...], internal: [...] } }
  │
  └─ [after both complete] DocumentationAgent.run_with_context()
        prompt: "Document this file. Context: [structure] [dependencies]..."
        LLM call → returns: { key_concepts: [...], usage_example: "...", role_in_system: "..." }

Result merged → recorder.record() (inside worker thread) → live JSON backup updated atomically
```

---

## 4. Recovery Mechanisms (0.8.0)

Since 0.8.0, codedoc uses a single always-on **live JSON backup** for crash
recovery.  The old hidden checkpoint (`Checkpoint` / `.codedoc_progress.json`)
is no longer created for new runs.  `--safe-mode` is deprecated and has no
additional effect.

| Scenario | Live backup path | Final output | On interrupt |
|---|---|---|---|
| `--format json` (default) | `codedoc/codedoc.json` | Same file (banner removed) | Backup kept with `_crash_safety` banner |
| `--format both` | `codedoc/codedoc.json` | `codedoc.json` + `codedoc.md` | Backup kept |
| `--format md` | `codedoc/codedoc.json` | `codedoc/codedoc.md` (backup removed on success) | Backup kept |
| `--output docs/report.md` | `docs/report.json` | `docs/report.md` (backup removed on success) | `report.json` kept |
| `--output docs/report.json` | `docs/report.json` | Same file (banner removed) | Backup kept |

**Lifecycle:**
1. Before any AI call: backup created with `_crash_safety` banner and empty `files[]`.
2. After each file: backup updated atomically (`.tmp` rename). Files are in topological order.
3. On clean finish (JSON): `write_project_outputs` overwrites the backup without banner.
4. On clean finish (MD): Markdown written, then backup deleted. If deletion fails (file lock),
   a warning is printed and the leftover JSON can be deleted manually.
5. On interrupt: backup remains with `_crash_safety` as the first key and `files[]` containing
   only completed work. Re-running the same command resumes from it.

**In parallel mode:** each worker calls `recorder.record()` before returning to the
main thread, so a Ctrl-C or crash after a worker completes never discards that result.

**Key guarantees:**

- **Hash-verified resume.** On re-run, files whose hash matches the live backup are skipped;
  files with changed hashes or no stored hash are re-documented.
- **Queue order.** The `files` array follows topological processing order (not completion
  order or path sorting) in both the live backup and the final output.
- **Ownership guard.** `codedoc` refuses to overwrite a file it did not create (no `_codedoc`
  metadata block) — including the JSON backup sibling for named-MD runs.

**Rate-limit step-down (0.8.0):**
When a rate-limit signal is detected during parallel processing, codedoc steps down the
file concurrency ladder and prints a provider-specific notice to the terminal:

```
[OpenAI] Rate limit detected - your configured max_parallel_files (5) has been
reduced to 2. Retrying 4 remaining file(s) at lower concurrency.
```

Recovered rate-limit events appear in `error.log` (located in the output directory,
not the project root) as warnings, and do not alarm the final output.

---

## 5. Scenarios

---

### Scenario A — Clean First Run, All Files Succeed ✅

**Command:** `codedoc run --entry src/main.py`

**What happens:**
- `codedoc/codedoc.json` created immediately with `_crash_safety` banner (before LLM starts).
- All files sent to the LLM (nothing cached). Each file: 3 parallel LLM calls → result
  merged → written to `codedoc.json` atomically inside the worker thread.
- All files complete → `write_project_outputs` overwrites `codedoc.json` with final clean
  output (no `_crash_safety`, no `status = "in_progress"`).

**Output:**
```
codedoc complete.
  Files documented : 20
  Files reused     : 0
  Files failed     : 0
  Output file      : codedoc/codedoc.json
```

---

### Scenario B — Subsequent Run, No Files Changed ✅

**Command:** `codedoc run` (re-run after Scenario A)

**What happens:**
- `_load_existing_file_docs` loads `codedoc/codedoc.json` → 20 file records with hashes.
- `changed_rels` = empty set (all hashes match current files).
- `agent_rels` = empty set.
- `write_project_outputs` called immediately with existing records (no LLM calls).
- Checkpoint created then immediately deleted.

**Output:**
```
codedoc complete.
  Files documented : 0
  Files reused     : 0
  Files failed     : 0
  Output file      : codedoc/codedoc.json
```

---

### Scenario C — Subsequent Run, 3 Files Changed ✅

**Command:** `codedoc run` (after editing 3 files)

**What happens:**
- `changed_rels` = 3 files (hashes differ from stored).
- `propagate_changes=True` (default): the dependency graph is checked — any file
  that imports a changed file is also added to `process_rels`.
- Only changed + dependent files sent to the LLM.
- Unchanged files reused from `existing_docs`.
- Final JSON written with all 20 files (17 reused + 3 or more re-documented).

**Output:**
```
codedoc complete.
  Files documented : 5      ← 3 changed + 2 dependents
  Files reused     : 0
  Files failed     : 0
  Output file      : codedoc/codedoc.json
```

---

### Scenario D — Run Interrupted Mid-Way (Ctrl-C) 🔄

**Command:** `codedoc run --entry src/main.py` on a 20-file project, Ctrl-C after 8 files.

**What happens during the run (0.8.0):**
- Before any LLM call: `codedoc/codedoc.json` created with `_crash_safety` banner and empty `files[]`.
- After each completed file: worker thread calls `recorder.record()` → `codedoc.json` updated atomically.
- After 8 files complete, Ctrl-C → `KeyboardInterrupt` caught in `cli.py`.
- Message printed: "Run interrupted. Progress has been saved to the live JSON backup — re-run to resume."
- Exit code 130.
- `codedoc/codedoc.json` remains on disk with 8 file records in topological order, `_crash_safety` banner present.

**On re-run:**
- `_load_existing_file_docs` reads `codedoc.json` (including the in-progress backup) → `existing_docs` = 8 entries with hashes.
- Changed files: computed → 0 (nothing changed) → `process_rels` = 12 remaining files.
- 12 remaining files → `agent_rels` → sent to the LLM.
- `write_project_outputs` overwrites `codedoc.json` with the final clean output (no `_crash_safety`).

**Output:**
```
codedoc complete.
  Files documented : 12
  Files reused     : 0
  Files skipped    : 8
  Files failed     : 0
  Output file      : codedoc/codedoc.json
```

---

### Scenario E — File Edited Between Interrupt and Resume 🔄

**Command:** Run interrupted after 8 files. User edits `src/auth/login.py`. Then re-runs.

**What happens (0.8.0):**
- `existing_docs` loaded from `codedoc/codedoc.json` live backup; `src/auth/login.py` entry has hash `"abc123"`.
- Current hash of `src/auth/login.py` = `"def456"` (different → in `changed_rels`).
- File added to `agent_rels` → sent to the LLM; the new result replaces the old record in the same queue-order slot.
- The other 7 completed files (unchanged) are skipped.

**Result:** 13 files sent to the LLM (12 new + 1 re-documented due to the edit).

---

### Scenario F — Migration from Legacy Checkpoint (0.7.x → 0.8.0) 🔄

**Situation:** Upgrading from 0.7.x — a `.codedoc_progress.json` exists from an interrupted run but no live JSON backup exists yet.

**What happens:**
- `recorder.load()` — live backup (`codedoc.json`) not found → records are empty.
- `recorder.size == 0` → pipeline loads `Checkpoint(output_dir).load()` as migration fallback.
- `checkpoint_results` = prior entries with `_checkpoint_hash`.
- Routing loop checks each checkpoint entry:
  - Hash matches current file → `resumed += 1` (no LLM call).
  - Hash missing or mismatched → `agent_rels.add(rel_path)` (reprocessed).
- On clean finish, the new live backup (`codedoc.json`) contains all results. `.codedoc_progress.json` is NOT recreated.

**Result:** One-time migration. After this run, all crash-recovery uses the live JSON backup.

---

### Scenario G — Single File API Failure (Retried, Then Failed) ❌

**What happens:**
- `_process_one_file_with_retries` tries `retry_attempts + 1` times (default: 2 total).
- Each attempt: the provider's `complete_json()` raises `LLMError(provider, "...")`.
- After all retries are exhausted → the exception propagates out.
- `_process_files_sequentially` catches it:
  - `error_reporter.record(exc, context=rel_path)`
  - `queue.mark_failed(rel_path, str(exc))`
  - `stats["failed"] += 1`
- `consecutive_failures += 1`.

**Other files continue** — one file failing does not stop the run.

**Output:**
```
codedoc complete.
  Files documented : 19
  Files reused     : 0
  Files failed     : 1
  Output file      : codedoc/codedoc.json

  1 file(s) failed. See /path/to/project/codedoc/error.log for details.
```

`error.log` (located in the **output directory**, not the project root) contains the full traceback for the failed file. Exit code 1.

---

### Scenario H — Consecutive File Failures (Health Check Triggers) ❌

**Default:** `max_consecutive_failures = 5`

**What happens (parallel processing path):**
- After 5 consecutive failures in `_process_agent_files`:
  - `_cancel_pending(future_map)` cancels all remaining parallel futures.
  - `error_reporter.record(RuntimeError("Parallel processing saw 5 consecutive file failures..."))`.
  - All failed files collected in `failed_descriptors`.
- `_process_files_sequentially` retries each failed file individually for clearer errors.
- During the sequential retry: if consecutive failures hit 5 again:
  - `error_reporter.record(RuntimeError("Stopping sequential processing after 5 consecutive file failures..."))`.
  - Processing stops early. Remaining files in the queue are never attempted.

**Why this matters:** If the provider's API is down or the key is revoked, every
file fails. The health check stops the run instead of spending N pointless API calls.

**Log messages:**
```
WARNING  [RETRY] src/file1.py | 1/20 complete (5%), 19 remaining | LLMError [OpenAI]: ...
WARNING  [RETRY] src/file2.py | 2/20 complete (10%), 18 remaining | LLMError [OpenAI]: ...
...
WARNING  Parallel processing saw 5 consecutive failures. API may be unavailable...
INFO     Retrying 5 failed file(s) sequentially for clearer errors.
WARNING  Stopping sequential processing after 5 consecutive file failures. Check API credentials...
```

---

### Scenario I — Missing API Key ❌

**Command:** `codedoc run --entry src/main.py` with no provider key set.

**What happens:**
- `create_provider(config)` calls `_provider_api_key(provider, model)` for the
  selected provider and finds nothing in the environment.
- `api_key = ""` → `_make_api` raises:
  ```
  ConfigError: API mode requires an API key. Set OPENAI_API_KEY, ANTHROPIC_API_KEY,
  or GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file, or pass LLM_API_KEY as a
  generic fallback.
  ```
- `cli.py` catches `ConfigError` → prints `"Error: ..."` → exit code 1.
- **No files are processed. No output is written.**

---

### Scenario J — Invalid or Non-Existent Model Name ❌

**Command (example):** `codedoc run --model gpt-99-turbo --entry src/main.py`
(equivalent failures occur with a bad `claude-*` or `gemini-*` model name).

**What happens:**
- The provider is created successfully (model name is not validated at init time).
- First file processing: `complete_json()` sends the request to the API.
- The API returns a 404 / "model not found" error.
- `LLMError(provider, "The model '...' does not exist...")` is raised.
- The file is marked as failed. With consecutive failures the health check
  eventually stops the run.

**Note:** The error is per-file, not a startup error. All files will fail with
the same error. Check `error.log` for the model-not-found message.

---

### Scenario K — Live Backup Interrupted (JSON, 0.8.0 default) 🔄

**Command:** `codedoc run --entry src/main.py`  *(Note: `--safe-mode` is deprecated and a no-op since 0.8.0; this is now the default behaviour for every run.)*

**What happens during the run:**
- `SafeWriter(backup_path, "json", ...)` is always created — no flag needed.
- Before any LLM call: `codedoc.json` is written with `_crash_safety` banner and empty `files[]`.
- `SafeWriter.load()` — if `codedoc.json` exists:
  - Valid codedoc file (has `_codedoc` block, including `in_progress` backup) → pre-loaded.
  - Unreadable / malformed / no `_codedoc` block → `ConfigError` immediately (foreign file).
  - Does not exist → proceed fresh.
- After every file: `recorder.record()` inside the worker thread → `codedoc.json` updated atomically.
- Interrupted → `codedoc.json` remains on disk with `_crash_safety` banner and partial `files[]`.

**On re-run:**
- `_load_existing_file_docs` reads `codedoc.json` (in-progress backup is valid — has `_codedoc` block).
- Hash comparison: files already in backup that are unchanged → skipped; changed or missing → sent to the LLM.
- Final `write_project_outputs` overwrites `codedoc.json` with clean final output (no `_crash_safety`).

---

### Scenario L — Live Backup Interrupted (MD-only, 0.8.0) 🔄

**Command:** `codedoc run --format md --entry src/main.py`

**What happens:**
- `_resolve_live_backup_path(output_dir, "md", ...)` → backup is `codedoc/codedoc.json` (JSON sibling of `codedoc.md`).
- Before any LLM call: `codedoc/codedoc.json` created with `_crash_safety` banner.
- After every file: `codedoc/codedoc.json` updated atomically (inside worker thread).
- Interrupted → `codedoc/codedoc.json` on disk with `_crash_safety` and partial records.

**On re-run:**
- `_load_existing_file_docs` reads the live backup `codedoc/codedoc.json` → `existing_docs` populated.
- Unchanged files skipped. Remaining files sent to the LLM.
- `write_project_outputs` writes `codedoc.md` (directly, no intermediate build file).
- On clean success → `recorder.delete()` removes `codedoc/codedoc.json` → only `codedoc.md` remains.

---

### Scenario M — MD Run, Markdown Conversion Crashes 🔄

**Command:** `codedoc run --format md --entry src/main.py` — all 20 LLM calls
succeed, but `markdown_from_view()` throws an exception during conversion.

**What happens in `write_project_outputs` (0.8.0):**
```
Step 1: _check_file_ownership(md_path)     → passes (new file)
Step 2: build_project_view()               → succeeds
Step 3: _write_project_markdown(view, md_path) → EXCEPTION ❌
→ raises OutputError
→ codedoc/codedoc.json (live backup) remains on disk with all 20 file records
```

Note: in 0.8.0 there is no intermediate `.codedoc_build.json` written by `write_project_outputs`.
The live backup (`codedoc.json`) written by `SafeWriter` throughout the run is the crash source.

**On re-run:**
- `_load_existing_file_docs` reads `codedoc/codedoc.json` → found, valid, 20 records.
- All 20 files unchanged → `agent_rels` = empty.
- `write_project_outputs` called immediately (no LLM calls) → re-attempts MD conversion.
- If conversion succeeds → `codedoc/codedoc.json` deleted → `codedoc.md` written.

**No LLM cost on retry.** Only the broken conversion step is retried.

---

### Scenario N — Stale Legacy `.codedoc_build.json` After a Later JSON Run 🔄

**Situation (migration from 0.7.x):** A `--format md` run under 0.7.x crashed and left a
`.codedoc_build.json`. Later, a `--format json` run under 0.8.0 rewrote `codedoc.json`.
The build file is now **older** than `codedoc.json`.

```
codedoc/codedoc.json         ← NEWER, from the 0.8.0 JSON run (authoritative)
codedoc/.codedoc_build.json  ← OLDER, leftover from the 0.7.x crashed MD run
```

**What `_load_existing_file_docs` does:**
1. Loads `codedoc.json` → `existing` = records from the newer JSON run.
2. Detects build file is **older** than `codedoc.json` → stale; removes it.
3. Returns records from `codedoc.json` only.

**Result:** The newer `codedoc.json` documentation is preserved; the stale build file is cleaned up.

---

### Scenario O — Ownership Conflict (Final Output) ❌

**Situation:** The output directory already contains a `codedoc.json` that is NOT
a codedoc output (e.g. a package manifest, an API spec) — but is still valid JSON.

**What happens in `write_project_outputs`:**
- `_check_file_ownership(json_path)` reads `codedoc.json`.
- Parses JSON → no top-level `"_codedoc"` key.
- Raises:
  ```
  ConfigError: 'codedoc.json' already exists but does not appear to be a codedoc output file.
  codedoc will not overwrite it to protect your data.

  To resolve this, choose one of:
    • Use a different output directory:   codedoc run --output my_docs/
    • Delete or rename the conflicting file:  /path/to/codedoc/codedoc.json
  ```
- `cli.py` catches `ConfigError` → prints `"Error: ..."` → exit code 1.
- **The user's file is never touched.**

**Same protection applies to:** `codedoc.md` (checks for the `<!-- codedoc-ai: -->`
comment) and the named-MD JSON sibling (e.g. `docs/report.json` — checked before any AI call).

---

### Scenario P — Foreign File at the Live Backup Target Path ❌

**Situation:** `codedoc run --entry src/main.py` and there is a pre-existing
foreign `codedoc.json` — **valid JSON but with no `_codedoc` metadata** (e.g. a
config or manifest a user happened to name `codedoc.json`).

**What happens:**
- `SafeWriter.load()` is called before any file processing begins.
- Reads `codedoc.json` → parses JSON → `data.get("_codedoc")` is not a dict.
- Raises `ConfigError` immediately.
- **Run aborted before the first API call is made.**

This is earlier protection than `write_project_outputs` — the foreign file is
protected even before any LLM work starts, not just at the write step.

---

### Scenario Q — Resume with a Completed `codedoc.json` Plus New Files 🔄

**Situation:** A previous clean run produced a `codedoc.json` (20 files).
The user adds 5 new files and re-runs, but interrupts after only 3 of the new files complete.

**What happens (and why prior work is NOT lost):**
- `SafeWriter.load()` sees the existing codedoc JSON. Even though its status is not
  `in_progress` (it is a completed output), **all 20 existing records are pre-loaded into memory.**
- `initialize_empty()` writes the in-progress banner — the first flush includes all 20 old records.
- As each new file is processed, `SafeWriter` flushes 20 original + new ones.
- Interrupt after 3 new files → `codedoc.json` holds 23 records (20 original + 3 new).

**On re-run:**
- `_load_existing_file_docs` reads all 23 records.
- 20 original + 3 new are unchanged → skipped. Only the remaining 2 new files sent to the LLM.

**Why this matters:** Without the pre-load, the first live-backup flush would overwrite
`codedoc.json` with only the newly processed files, erasing all prior completed records on
interrupt. The pre-load guarantees the live backup never loses prior completed work.

---

### Scenario R — Malformed or Empty File at the Target Path ❌

**Situation:** A `codedoc.json` exists at the target path but is **not parseable**
— truncated JSON, an empty file, or binary content.

**What happens:**
- **Live backup check:** `SafeWriter.load()` fails to parse it, treats it as foreign,
  and raises `ConfigError` immediately — before any LLM work begins.
- **Final write:** `_check_file_ownership` (in `output.py`) also raises `ConfigError`
  if the file is not a valid codedoc output.

`codedoc` refuses to overwrite an unreadable file it cannot confirm it created.
To proceed, delete/rename the file or choose a different `--output` directory.

---

### Scenario S — (see Scenario N — now merged with the stale build-file migration case)

---

## 6. Error Hierarchy

| Exception | Where raised | What it means |
|---|---|---|
| `ConfigError` | factory, loader, safe_writer, output | User-fixable problem — shown as `"Error: ..."` |
| `LLMError` | api_provider | A provider API call failed — caught per-file |
| `AgentError` | pipeline (from orchestrator result) | An agent returned `{"error": ...}` — caught per-file |
| `ParseError` | parser, graph builder | A file could not be parsed — caught per-file |
| `OutputError` | output.py | Writing the final output failed — fatal |
| `KeyboardInterrupt` | cli.py | Ctrl-C — live JSON backup preserved, exit 130 |

---

## 7. File State After Every Outcome

| What happened | Files on disk |
|---|---|
| Clean run (JSON) | `codedoc/codedoc.json` |
| Clean run (MD) | `codedoc/codedoc.md` |
| Clean run (both) | `codedoc/codedoc.json` + `codedoc/codedoc.md` |
| Interrupted (any mode) | `codedoc/codedoc.json` (partial, `_crash_safety` banner present) |
| Interrupted (MD-only, `--format md`) | `codedoc/codedoc.json` (live backup sibling, partial) |
| Interrupted (`--output docs/report.md`) | `docs/report.json` (live backup sibling, partial) |
| MD conversion crashed | `codedoc/codedoc.json` (live backup, all records — no MD written) |
| Any file failed | `codedoc/error.log` (output directory, not project root) |
| Any issue recorded (recovered warnings) | `codedoc/error.log` (path printed to terminal) |
| Aborted (ConfigError, missing key, foreign/malformed file) | Live backup may exist if abort was after `initialize_empty()` |
