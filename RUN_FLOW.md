# codedoc-ai — Full Run Flow & Scenarios

This document describes exactly how `codedoc` runs end-to-end across all three
supported providers — **OpenAI**, **Anthropic**, and **Gemini** — covering every
meaningful success, interrupt/resume, and failure scenario.

It is provider-agnostic: the pipeline, checkpointing, safe-mode, ownership
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

Result merged → recorder.record() → .codedoc_progress.json updated
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

**What happens during the run:**
- After each of the first 8 files: `recorder.record()` → `.codedoc_progress.json` written.
- Ctrl-C → `KeyboardInterrupt` caught in `cli.py`.
- Message printed: "Run interrupted. Progress has been saved — re-run to resume."
- Exit code 130.
- `.codedoc_progress.json` remains on disk with 8 results.

**On re-run:**
- `recorder.load()` reads `.codedoc_progress.json` → `recorder_results` = 8 entries.
- Routing loop for each file in `process_rels`:
  - If `rel_path in recorder_results`:
    - `stored_hash` is read from `_checkpoint_hash`.
    - If the hash matches the current file → `resumed += 1` (no LLM call).
    - If the hash differs (file was edited) → `agent_rels.add(rel_path)` (re-sent to the LLM).
  - Remaining 12 files → `agent_rels` → sent to the LLM.
- Only 12 (or fewer if some were edited) LLM calls made.

**Output:**
```
codedoc complete.
  Files documented : 12
  Files reused     : 0
  Files resumed    : 8
  Files failed     : 0
  Output file      : codedoc/codedoc.json
```

---

### Scenario E — File Edited Between Interrupt and Resume 🔄

**Command:** Run interrupted after 8 files. User edits `src/auth/login.py`. Then re-runs.

**What happens:**
- `recorder_results` has an entry for `src/auth/login.py` with `_checkpoint_hash = "abc123"`.
- Current hash of `src/auth/login.py` = `"def456"` (different).
- Routing loop: `stored_hash` exists AND `content_hash != stored_hash`.
- Log: `"File 'src/auth/login.py' was modified after it was checkpointed — reprocessing."`
- File added to `agent_rels` → sent to the LLM.
- The other 7 checkpointed files (unchanged) are restored without LLM calls.

**Result:** 13 files sent to the LLM (12 new + 1 re-documented due to the edit).

---

### Scenario F — Old Checkpoint (No Hash Stored) 🔄

**Situation:** Checkpoint was written by a version older than 0.7.2 — no `_checkpoint_hash` key.

**What happens:**
- `stored_hash = checkpoint_entry.get("_checkpoint_hash", "")` → `""`
- `not stored_hash` is True.
- Log: `"Checkpoint entry for 'src/auth/login.py' has no hash (written by an older version) — reprocessing to ensure correctness."`
- File added to `agent_rels` → re-sent to the LLM regardless.

**Result:** All files from the old checkpoint are reprocessed once (a one-time cost
on upgrade). After this run, all new checkpoints carry hashes and subsequent
interrupts resume correctly.

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

  See error.log in /path/to/project for details.
```

`error.log` contains the full traceback for the failed file. Exit code 1.

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

### Scenario K — `--safe-mode` Run Interrupted (JSON) 🔄

**Command:** `codedoc run --safe-mode --entry src/main.py`

**What happens during the run:**
- `SafeWriter(output_dir, "codedoc.json", "json", ...)` is created.
- `SafeWriter.load()` — if `codedoc.json` exists:
  - If it is a valid codedoc file (has a `_codedoc` block) → its records are
    pre-loaded so they are preserved in every partial write (see Scenario Q).
  - If it is unreadable / malformed / has no `_codedoc` block → `ConfigError`
    raised immediately (foreign file, run aborted — see Scenarios P and R).
  - If it doesn't exist → proceed fresh.
- After every file: `SafeWriter.record()` → `codedoc.json` updated with partial
  results (`status = "in_progress"`). You can open `codedoc.json` and see work so far.
- Interrupted → `codedoc.json` stays on disk with partial results.

**On re-run:**
- `_load_existing_file_docs` reads `codedoc.json` (partial `in_progress` is fine —
  it has a valid `_codedoc` block and `files` array).
- Hash comparison: files already in `codedoc.json` that are unchanged → skipped.
- Files not yet in `codedoc.json` → sent to the LLM.
- Final `write_project_outputs` overwrites `codedoc.json` with the complete polished output.

---

### Scenario L — `--safe-mode` + `--format md` Run Interrupted 🔄

**Command:** `codedoc run --safe-mode --format md --entry src/main.py`

**What happens:**
- `SafeWriter(output_dir, "codedoc.json", "md", ...)` → `self._path = .codedoc_build.json`.
- After every file: `.codedoc_build.json` is updated (not `codedoc.json`).
- Interrupted → `.codedoc_build.json` on disk with partial records.

**On re-run:**
- `_load_existing_file_docs`:
  1. Tries `codedoc.json` → not found (never written in MD mode).
  2. Tries `.codedoc_build.json` → found, valid `_codedoc` block → loads records.
  3. Merges (only the build file here) → `existing_docs` populated.
- Unchanged files skipped. Remaining files sent to the LLM.
- `write_project_outputs` writes the `.codedoc_build.json` backup → then `codedoc.md`.
- On success → `.codedoc_build.json` deleted. User sees only `codedoc.md`.

---

### Scenario M — MD Run, Markdown Conversion Crashes 🔄

**Command:** `codedoc run --format md --entry src/main.py` — all 20 LLM calls
succeed, but `markdown_from_view()` throws an exception during conversion.

**What happens in `write_project_outputs`:**
```
Step 1: _check_file_ownership(build_path)  → passes (new file)
Step 2: _check_file_ownership(md_path)     → passes (new file)
Step 3: build_project_view()               → succeeds
Step 4: _write_project_json(view, build_path)  → .codedoc_build.json written ✅
Step 5: _write_project_markdown(view, md_path) → EXCEPTION ❌
→ except block: logs "Output write failed — preserved at '.codedoc_build.json'"
→ raises OutputError
→ .codedoc_build.json remains on disk (complete JSON of all 20 files)
```

**On re-run:**
- `_load_existing_file_docs`: `codedoc.json` not found → tries `.codedoc_build.json`
  → found, valid, 20 records.
- All 20 files unchanged (same hashes) → `agent_rels` = empty.
- `write_project_outputs` called immediately (no LLM calls) → re-attempts MD conversion.
- If conversion succeeds → `.codedoc_build.json` deleted → `codedoc.md` written.

**No LLM cost on retry.** Only the broken conversion step is retried.

---

### Scenario N — Newer `.codedoc_build.json` Overlays Older `codedoc.json` 🔄

**Situation:** User ran `--format json` first (created `codedoc.json` with 5
files), then ran `--format md`, which processed 8 files before crashing (left a
**newer** `.codedoc_build.json`).

```
codedoc/codedoc.json         ← 5 files, from an OLDER JSON run
codedoc/.codedoc_build.json  ← 8 files, from a NEWER interrupted MD run
```

**What `_load_existing_file_docs` does:**
1. Loads `codedoc.json` → `existing` = 5 records.
2. Compares modification times → the build file is **newer**, so it is authoritative.
3. Loads `.codedoc_build.json` → `build_files` = 8 records.
4. `existing.update(build_files)` → `existing` = 8 records (build file takes
   priority per-file).
5. Returns 8 records.

**Result:** The 3 files that were processed in the interrupted MD run but NOT in
the older JSON run are correctly reused. No unnecessary LLM calls.

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
comment) and `.codedoc_build.json` (checks for the `_codedoc` block).

---

### Scenario P — `--safe-mode` with a Foreign File at the Target Path ❌

**Situation:** `codedoc run --safe-mode --entry src/main.py` and there is a
pre-existing foreign `codedoc.json` — **valid JSON but with no `_codedoc`
metadata** (e.g. a config or manifest a user happened to name `codedoc.json`).

**What happens:**
- `SafeWriter.load()` is called before any file processing begins.
- Reads `codedoc.json` → parses JSON → `data.get("_codedoc")` is not a dict.
- Raises `ConfigError` immediately.
- **Run aborted before the first API call is made.**

This is earlier protection than `write_project_outputs` — the foreign file is
protected even before any LLM work starts, not just at the write step.

---

### Scenario Q — `--safe-mode` Resume with a Completed `codedoc.json` Present 🔄

**Situation:** A previous run produced a **complete** `codedoc.json` (20 files).
The user then adds 5 new files and re-runs with `--safe-mode`, but interrupts
after only 3 of the new files are processed.

**What happens (and why prior work is NOT lost):**
- `SafeWriter.load()` sees a valid codedoc file. Even though its status is not
  `in_progress` (it is a completed output), **all 20 existing records are
  pre-loaded into memory.**
- As each new file is processed, `SafeWriter` flushes the full set — the 20
  original records **plus** the new ones — to `codedoc.json`.
- Interrupt after 3 new files → `codedoc.json` holds 23 records (20 original + 3 new).

**On re-run:**
- `_load_existing_file_docs` reads all 23 records.
- The 20 original + 3 new are unchanged → skipped. Only the remaining 2 new files
  are sent to the LLM.

**Why this scenario matters:** Without the pre-load, the first safe-mode flush
would overwrite `codedoc.json` with only the newly processed files, erasing the
20 previously completed records on interrupt — making `--safe-mode` *worse* than
the default. The pre-load guarantees safe mode never loses work that was already
on disk. (Fixed in 0.7.2.)

---

### Scenario R — Malformed or Empty File at the Target Path ❌

**Situation:** A `codedoc.json` exists at the target path but is **not parseable**
— truncated JSON, an empty file, or binary content.

**What happens:**
- **Default mode / final write:** `_check_file_ownership` (in `output.py`) fails
  to parse it, treats it as foreign, and raises `ConfigError` — the file is not
  overwritten.
- **`--safe-mode`:** `SafeWriter.load()` fails to parse it, treats it as foreign,
  and raises `ConfigError` immediately — before any LLM work begins.

Either way, `codedoc` refuses to overwrite an unreadable file it cannot confirm
it created. To proceed, delete/rename the file or choose a different `--output`
directory. (Malformed-file protection in safe mode was added in 0.7.2; prior to
that, safe mode would start fresh and overwrite it.)

---

### Scenario S — Stale `.codedoc_build.json` After a Later JSON Run 🔄

**Situation:** A `--format md` run crashed and left a `.codedoc_build.json`.
Later, a `--format json` run rewrote `codedoc.json` (which does **not** clean up
the build file). The build file is now **older** than `codedoc.json`.

```
codedoc/codedoc.json         ← NEWER, from the latest JSON run (authoritative)
codedoc/.codedoc_build.json  ← OLDER, leftover from the earlier crashed MD run
```

**What `_load_existing_file_docs` does:**
1. Loads `codedoc.json` → `existing` records.
2. Compares modification times → the build file is **older**, so it is **stale**.
3. The stale build file is **not** overlaid and is **removed** so it cannot
   interfere with future runs.
4. Log: `"Build file '.codedoc_build.json' is older than 'codedoc.json' — treating it as stale and removing it; the newer JSON takes priority."`

**Result:** The newer `codedoc.json` documentation is preserved. Without this
freshness check, the older build-file records would silently replace the newer
JSON records. (Fixed in 0.7.2 — the inverse of Scenario N.)

---

## 6. Error Hierarchy

| Exception | Where raised | What it means |
|---|---|---|
| `ConfigError` | factory, loader, safe_writer, output | User-fixable problem — shown as `"Error: ..."` |
| `LLMError` | api_provider | A provider API call failed — caught per-file |
| `AgentError` | pipeline (from orchestrator result) | An agent returned `{"error": ...}` — caught per-file |
| `ParseError` | parser, graph builder | A file could not be parsed — caught per-file |
| `OutputError` | output.py | Writing the final output failed — fatal |
| `KeyboardInterrupt` | cli.py | Ctrl-C — checkpoint preserved, exit 130 |

---

## 7. File State After Every Outcome

| What happened | Files on disk |
|---|---|
| Clean run (JSON) | `codedoc/codedoc.json` |
| Clean run (MD) | `codedoc/codedoc.md` |
| Clean run (both) | `codedoc/codedoc.json` + `codedoc/codedoc.md` |
| Interrupted (default mode) | `codedoc/.codedoc_progress.json` |
| Interrupted (`--safe-mode` JSON) | `codedoc/codedoc.json` (partial, `in_progress`) |
| Interrupted (`--safe-mode` MD) | `codedoc/.codedoc_build.json` (partial, `in_progress`) |
| MD conversion crashed | `codedoc/.codedoc_build.json` (complete JSON, no MD) |
| Any file failed | `error.log` in project root |
| Aborted (ConfigError, missing key, foreign/malformed file) | Nothing written |
