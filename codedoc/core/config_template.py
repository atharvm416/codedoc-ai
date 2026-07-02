"""Canonical full-config generator and config-only instruction initializer.

This module owns the two safe, provider-free config utilities:

- ``--init-config [NAME]`` writes a complete editable ``codedoc.config.json``
  (or a help-template file under a custom NAME) containing every public default
  plus editable single/triple prompt instructions;
- ``--init-instructions [single|triple|both]`` writes or replaces only the
  inline ``prompt_profiles`` value inside the exact ``codedoc.config.json``.

Both utilities generate their defaults from the canonical registries
(:data:`codedoc.core.loader.DEFAULTS` and
:func:`codedoc.core.prompt_profiles.default_prompt_profiles`) so there is never a
second hand-maintained template.  Neither utility scans source, detects an entry,
contacts a provider, runs an output preflight, initializes recovery, or mutates a
final-output target.

Public-default ownership
------------------------
:data:`PUBLIC_CONFIG_KEYS` is the stable, ordered list of user-facing
``DEFAULTS`` keys (with brief descriptions for README/help generation).
:data:`GENERATOR_EXCLUDED_KEYS` is the set never emitted (derived or removed
keys).  Together they must classify every ``DEFAULTS`` key; the drift test in
``tests/test_0113_config_generation.py`` fails if a newly added public default
lacks generator ownership or if an internal/removed key leaks into generated JSON.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from codedoc.core.block_manager import (
    create_text_exclusive,
    replace_text_atomic_no_backup,
)
from codedoc.core.loader import DEFAULTS, _CONFIG_FILENAME, _validate_portable_filename
from codedoc.core.prompt_profiles import default_prompt_profiles
from codedoc.utils.errors import ConfigError
from codedoc.utils.json_utils import DuplicateJSONKeyError, loads_no_duplicate_keys

# Keys never emitted into generated config.  ``supported_extensions`` is derived
# (read-only after load) from ``extension_language_map`` and is deliberately
# omitted (settled decision 5).  The remaining names were removed in 0.11.3 and
# are already absent from ``DEFAULTS``; they are listed here as a belt-and-braces
# guard and as the drift-test anchor so re-adding one to ``DEFAULTS`` cannot leak
# it into generated JSON without also failing the classification test.
GENERATOR_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        "supported_extensions",
        "safe_mode",
        "manage_output_gitignore",
        "output_gitignore_filename",
        "prompt_profile_file",
        "prompt_profile_auto_detect",
        "prompt_profile_disabled",
        "prompt_customization_allow_risky",
    }
)

# Ordered, user-facing config keys with short descriptions.  Every ``DEFAULTS``
# key that is not in ``GENERATOR_EXCLUDED_KEYS`` must appear here exactly once.
# ``api_key`` is always emitted as ``null`` and ``prompt_profiles`` as the full
# ``common``-envelope schema-v2 object; every other key is emitted as its
# resolved default value.
PUBLIC_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("llm_mode", "LLM mode. Only 'api' is supported."),
    ("llm_provider", "Provider: 'auto', 'openai', 'anthropic', or 'gemini'."),
    ("model_name", "Model identifier; empty selects the provider default."),
    ("api_base_url", "Custom OpenAI-compatible base URL, or null."),
    ("api_key", "Never stored here; set an API key via environment variable."),
    ("entry_file", "Entry file relative to the project root, or null to auto-detect."),
    ("documentation_scope", "'entry' (reachable from entry) or 'all' scanned files."),
    ("output_dir", "Output directory (or a file path ending in .json/.md)."),
    ("output_format", "'json', 'md', or 'both'."),
    ("output_json_filename", "JSON output filename."),
    ("output_md_filename", "Markdown output filename."),
    ("parallel_agents", "Run structure/dependency agents in parallel (triple mode)."),
    ("max_parallel_files", "Maximum files processed concurrently."),
    ("file_retry_attempts", "Per-file retry attempts on a recoverable failure."),
    ("max_consecutive_failures", "Abort after this many consecutive file failures."),
    ("log_level", "Logging level (DEBUG, INFO, WARNING, ERROR)."),
    ("max_file_size_kb", "Skip files larger than this many KB."),
    ("follow_symlinks", "Follow symlinked files/directories when scanning."),
    ("propagate_changes", "Re-document dependents of a changed file."),
    ("rate_limit_adaptive", "Adaptively step down parallelism on rate limits."),
    ("parallel_ladder", "Explicit parallelism step-down ladder, or null."),
    ("respect_retry_after", "Honor a provider Retry-After hint."),
    ("retry_after_cap_s", "Upper bound (seconds) on an honored Retry-After."),
    ("skip_dirs", "Directory names skipped while scanning (replaces the default list)."),
    ("skip_dirs_add", "Directory names to add to the skip list."),
    ("skip_dirs_remove", "Directory names to remove from the skip list."),
    ("extension_language_map", "Extension -> language map (source of truth for scanning)."),
    ("extension_language_map_add", "Extension -> language entries to add."),
    ("extension_language_map_remove", "Extensions to remove from the map."),
    ("auto_entry_candidates", "Filenames tried during entry auto-detection."),
    ("auto_entry_candidates_add", "Entry-candidate filenames to add."),
    ("auto_entry_candidates_remove", "Entry-candidate filenames to remove."),
    ("provider_prefixes", "Provider -> model-name prefixes for auto-detection."),
    ("provider_prefixes_add", "Provider -> prefixes to add."),
    ("provider_prefixes_remove", "Provider -> prefixes to remove."),
    ("rate_limit_backoff_s", "Global min backoff seconds override, or null."),
    ("rate_limit_backoff_scale", "Global backoff scale override, or null."),
    ("rate_limit_signals_add", "Extra rate-limit signal strings to add."),
    ("rate_limit_signals_remove", "Rate-limit signal strings to remove."),
    ("ignore_paths", "Project-relative paths to ignore."),
    ("max_content_chars", "Maximum source characters sent to the LLM per file."),
    ("dry_run", "Plan only: no writes and no provider calls."),
    ("max_files", "Safety cap on files that may make LLM calls (0 = unlimited)."),
    ("force_files", "Project-relative paths to reprocess even when unchanged."),
    ("allow_partial", "Exit 0 even when some files failed (completed runs only)."),
    ("analysis_mode", "'single' (one combined call) or 'triple' (three agents)."),
    ("truncation_head_ratio", "Head fraction (0<r<1) of the head+tail truncation split."),
    ("prompt_profiles", "Inline single/triple instructions under the 'common' envelope."),
)

_CREDENTIAL_NOTE = (
    'Credentials are never written to the config: "api_key" stays null. Set an '
    "API key as an environment variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, "
    "GEMINI_API_KEY, GOOGLE_API_KEY, or the generic LLM_API_KEY)."
)

_VALID_INSTRUCTION_MODES = ("single", "triple", "both")


@dataclass(frozen=True)
class InitResult:
    """Outcome of an ``--init-config`` / ``--init-instructions`` invocation."""

    path: Path
    message: str


# ---------------------------------------------------------------------------
# Full-config generation (Workstream A)
# ---------------------------------------------------------------------------

def build_default_config() -> dict:
    """Return the complete, editable default config dict.

    Emits every :data:`PUBLIC_CONFIG_KEYS` entry in order with its resolved
    default value.  ``api_key`` is emitted as ``null``; ``prompt_profiles`` is the
    full ``common``-envelope schema-v2 object for both single and triple modes.
    The result round-trips through ``load_config`` and is developer-standard
    equivalent (inert until edited).
    """
    out: dict = {}
    for key, _description in PUBLIC_CONFIG_KEYS:
        if key == "api_key":
            out[key] = None
        elif key == "prompt_profiles":
            out[key] = default_prompt_profiles()
        else:
            out[key] = copy.deepcopy(DEFAULTS[key])
    return out


def render_default_config_json() -> str:
    """Return the generated default config as pretty UTF-8 JSON text."""
    return json.dumps(build_default_config(), indent=2, ensure_ascii=False) + "\n"


def init_config(project_root: Path | str, name: str | None, force: bool) -> InitResult:
    """Write a full default ``codedoc.config.json`` (or a custom-name template).

    With ``name=None`` the target is ``<project_root>/codedoc.config.json`` (the
    active config).  With a portable ``name`` the target is that file — a
    non-active *help template* that must be renamed/copied to
    ``codedoc.config.json`` to take effect.  Never scans source, detects an entry,
    contacts a provider, or mutates any output/recovery file.
    """
    root = Path(project_root)
    if name is None:
        target = root / _CONFIG_FILENAME
        is_template = False
    else:
        filename = _normalize_config_filename(name)
        target = root / filename
        is_template = filename.lower() != _CONFIG_FILENAME.lower()

    rerun = "codedoc --init-config" + (f" {name}" if name is not None else "")
    _write_generated_file(target, render_default_config_json(), force, rerun)

    if is_template:
        message = (
            f"Template created: {target}\n"
            "CodeDoc reads only codedoc.config.json; rename or copy this file to "
            "codedoc.config.json to activate it.\n" + _CREDENTIAL_NOTE
        )
    else:
        message = (
            f"Created {target} with every public default and editable single/triple "
            "prompt instructions. Edit it, then run codedoc.\n" + _CREDENTIAL_NOTE
        )
    return InitResult(path=target, message=message)


# ---------------------------------------------------------------------------
# Config-only instruction initialization (Workstream B)
# ---------------------------------------------------------------------------

def init_instructions(
    project_root: Path | str, mode: str | None, force: bool
) -> InitResult:
    """Create/insert/replace only the inline ``prompt_profiles`` in the config.

    Operates only on ``<project_root>/codedoc.config.json`` (no filename option).
    *mode* is ``single``, ``triple``, or ``both`` (default ``both``).  Preserves
    every other semantic setting; makes no provider/review/documentation call and
    creates no output/recovery file.
    """
    resolved_mode = mode or "both"
    if resolved_mode not in _VALID_INSTRUCTION_MODES:
        raise ConfigError(
            "instruction mode must be 'single', 'triple', or 'both'; got "
            f"{resolved_mode!r}."
        )
    section = None if resolved_mode == "both" else resolved_mode
    profiles = default_prompt_profiles(section)

    root = Path(project_root)
    target = root / _CONFIG_FILENAME

    if not target.exists():
        text = json.dumps({"prompt_profiles": profiles}, indent=2, ensure_ascii=False) + "\n"
        create_text_exclusive(target, text)
        return InitResult(
            path=target,
            message=(
                f"Created {target} with {resolved_mode} prompt instructions.\n"
                + _CREDENTIAL_NOTE
            ),
        )

    if target.is_symlink():
        raise ConfigError(f"refusing to modify symlink '{target}'.")
    if target.is_dir():
        raise ConfigError(f"'{target}' is a directory, not a config file.")

    original_bytes = target.read_bytes()
    try:
        text = original_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"'{target.name}' is not valid UTF-8.") from exc
    try:
        data = loads_no_duplicate_keys(text)
    except DuplicateJSONKeyError as exc:
        raise ConfigError(
            f"'{target.name}' is not valid JSON: duplicate key {exc.key!r}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"'{target.name}' is not valid JSON: {exc}.") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"'{target.name}' must be a JSON object.")

    existing = data.get("prompt_profiles")
    replacing = existing is not None
    if replacing and not force:
        raise ConfigError(
            f"'{target.name}' already defines a non-null 'prompt_profiles'. Re-run "
            "with --force to replace only that value (all other settings are "
            "preserved; JSON formatting may be canonicalized):\n"
            f"  codedoc --init-instructions {resolved_mode} --force"
        )

    data["prompt_profiles"] = profiles
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    # End-to-end concurrent-change guard against the bytes we parsed.
    if target.read_bytes() != original_bytes:
        raise ConfigError(f"'{target.name}' changed during update; aborting.")
    replace_text_atomic_no_backup(target, new_text)

    if replacing:
        message = (
            f"Replaced prompt_profiles in {target} with {resolved_mode} instructions "
            "(all other settings preserved; JSON formatting canonicalized)."
        )
    else:
        message = (
            f"Added {resolved_mode} prompt instructions to {target} "
            "(all other settings preserved)."
        )
    return InitResult(path=target, message=message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_config_filename(name: str) -> str:
    """Normalize + validate a custom ``--init-config NAME`` into a portable file.

    Appends ``.json`` (case-insensitively) when missing — never twice — then
    validates the final basename through the shared portable-filename validator so
    paths, drives, traversal, reserved device names, and control characters are
    rejected.
    """
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("--init-config NAME must be a non-empty portable filename.")
    if name.lower().endswith(".json.json"):
        raise ConfigError(
            "--init-config NAME must not repeat the .json extension."
        )
    candidate = name if name.lower().endswith(".json") else name + ".json"
    _validate_portable_filename(candidate, "--init-config NAME")
    return candidate


def _write_generated_file(target: Path, text: str, force: bool, rerun: str) -> None:
    """Create *target* with *text*, refusing an existing file unless *force*.

    Without ``--force`` an existing target is refused with the exact rerun command
    (no persistent backup is ever written — ``--force`` is the user's explicit
    permission to discard the old bytes).  A failed atomic replacement leaves the
    original byte-identical.
    """
    if force:
        replace_text_atomic_no_backup(target, text)
        return
    if target.exists():
        raise ConfigError(
            f"'{target.name}' already exists. Re-run with --force to replace its "
            f"contents (the current bytes are discarded, no backup is kept):\n"
            f"  {rerun} --force"
        )
    create_text_exclusive(target, text)
