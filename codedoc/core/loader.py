"""Config loader for codedoc."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

from codedoc.utils.errors import ConfigError
from codedoc.utils.json_utils import (
    DuplicateJSONKeyError,
    NonFiniteJSONNumberError,
    loads_no_duplicate_keys,
)
from codedoc.utils.logger import get_logger
from codedoc.core.release_policy import current_split_release_policy
from codedoc.llm.factory import (
    EndpointTrustAttestation,
    effective_endpoint_identity,
)

logger = get_logger(__name__)


class ResolvedConfig(dict):
    """Dict-compatible configuration mapping returned by :func:`load_config`.

    Behaves exactly as the plain configuration ``dict`` every current reader
    already expects -- construction, ``dict()`` conversion, ``.get()``,
    iteration, membership, and JSON serialization are all unchanged, so no
    call site needs to change shape. The sole addition is the ``endpoint_trust``
    attribute, an ``EndpointTrustAttestation | None`` recording the canonical
    endpoint digest authorized for this config's ``api_base_url`` (if any) and
    the runtime mechanism that authorized it.

    ``endpoint_trust`` is deliberately a plain instance attribute, never a dict
    key: ``__slots__`` below is the only place it is stored, so it is excluded
    from iteration, ``dict()`` conversion, JSON serialization, persistence,
    cache identity, recovery identity, and provider identity by construction --
    it can be neither forged through configuration nor leaked into an
    artifact. :func:`codedoc.llm.factory.create_provider` reads it via
    ``getattr(config, "endpoint_trust", None)`` so a plain ``dict`` (which has
    no such attribute) is always treated as unattested.
    """

    __slots__ = ("endpoint_trust",)

    def __init__(self, *args: Any, endpoint_trust: EndpointTrustAttestation | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.endpoint_trust = endpoint_trust


# Environment variable naming the runtime endpoint-trust approval URL. Read
# directly by the authorization gate inside load_config() -- deliberately not
# part of _ENV_KEY_MAP, because it never becomes a config key: it is compared
# against api_base_url and then discarded, never merged into the returned
# config.
_TRUST_API_BASE_URL_ENV = "CODEDOC_TRUST_API_BASE_URL"

# Keys that can only ever express endpoint-trust approval at runtime (CLI
# option or environment variable above). Rejected wherever ordinary
# configuration is read, with a message naming the two actual mechanisms,
# because a project-controlled codedoc.config.json or a programmatic
# config_overrides dict must never be able to satisfy this gate.
_ENDPOINT_TRUST_KEYS: tuple[str, ...] = (
    "trust_api_base_url",
    "trusted_api_base_url",
    "_endpoint_trust",
)


def _reject_endpoint_trust_keys(data: dict[str, Any], *, source: str) -> None:
    """Raise :class:`ConfigError` when *data* sets an endpoint-trust key.

    Detected before generic unknown-key rejection so the error names the two
    actual runtime approval mechanisms instead of a generic "unknown key"
    message.
    """
    if not isinstance(data, dict):
        return
    offending = [key for key in _ENDPOINT_TRUST_KEYS if key in data]
    if not offending:
        return
    details = "\n".join(f"  - '{key}'" for key in offending)
    plural = "keys" if len(offending) > 1 else "key"
    raise ConfigError(
        f"{source} sets {len(offending)} endpoint-trust configuration {plural}:\n"
        f"{details}\n"
        "Endpoint-trust approval for a custom api_base_url can never be granted "
        "through codedoc.config.json or config_overrides. Approve the exact "
        "endpoint at runtime instead, using the --trust-api-base-url CLI option "
        f"or the {_TRUST_API_BASE_URL_ENV} environment variable."
    )


DEFAULTS: dict[str, Any] = {
    "llm_mode": "api",
    "llm_provider": "auto",
    "model_name": "",
    "api_base_url": None,
    "api_key": None,
    "entry_file": None,
    "documentation_scope": "entry",
    "output_dir": "codedoc",
    "output_format": "json",
    "output_json_filename": "codedoc.json",
    "output_md_filename": "codedoc.md",
    # supported_extensions: read-only after load_config() — always derived from
    # the resolved extension_language_map.  The value listed here is the legacy
    # default set and acts as the detection baseline: if a caller passes a
    # *different* list, _apply_config_overrides() treats it as a filter on the
    # extension_language_map (backward-compat bridge for older configs).
    "supported_extensions": [
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".dart",
        ".java",
        ".cs",
        ".html",
    ],
    "parallel_agents": True,
    "max_parallel_files": 5,
    "file_retry_attempts": 1,
    "max_consecutive_failures": 5,
    "log_level": "INFO",
    "max_file_size_kb": 500,
    # Scanner safety: when False (default) symlinked directories and files
    # are skipped, preventing symlink cycles and escapes outside the project
    # root.  Settable via JSON config or the Python API only (no CLI flag/env).
    "follow_symlinks": False,
    "propagate_changes": True,
    # Rate-limit adaptive parallelism
    "rate_limit_adaptive": True,
    "parallel_ladder": None,
    "respect_retry_after": True,
    "retry_after_cap_s": 30,
    # -----------------------------------------------------------------------
    # skip_dirs — single source of truth (was split across loader + scanner)
    # -----------------------------------------------------------------------
    "skip_dirs": [
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "myenv",
        ".env",
        "node_modules",
        "site-packages",
        "dist-packages",
        "dist",
        "build",
        ".next",
        ".nuxt",
        "target",
        "codedoc",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    ],
    # Extend skip_dirs without replacing the full list.
    "skip_dirs_add": [],
    # Remove entries from skip_dirs.  Use to allow scanning a package whose
    # directory name appears in the default list (e.g. "codedoc").
    "skip_dirs_remove": [],
    # -----------------------------------------------------------------------
    # extension_language_map — replaces the hardcoded EXTENSION_LANGUAGE_MAP
    # in scanner.py.  Any extension in the resolved map is automatically
    # supported — no need to edit both this and supported_extensions.
    # -----------------------------------------------------------------------
    "extension_language_map": {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "jsx",
        ".dart": "dart",
        ".java": "java",
        ".cs": "csharp",
        ".html": "html",
        ".htm": "html",
        ".kt": "kotlin",
        ".swift": "swift",
        ".go": "go",
        ".rb": "ruby",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
    },
    # Add new extension → language entries (merged with extension_language_map).
    "extension_language_map_add": {},
    # Remove extensions from the map (list of extension strings, e.g. [".htm"]).
    "extension_language_map_remove": [],
    # -----------------------------------------------------------------------
    # auto_entry_candidates — replaces the hardcoded common_entries list
    # in scanner.detect_entry_file().
    # -----------------------------------------------------------------------
    "auto_entry_candidates": [
        "index.html",
        "main.tsx",
        "main.ts",
        "main.js",
        "main.py",
        "main.dart",
        "Main.java",
        "Program.cs",
    ],
    "auto_entry_candidates_add": [],
    "auto_entry_candidates_remove": [],
    # -----------------------------------------------------------------------
    # provider_prefixes — replaces the hardcoded _*_PREFIXES tuples in
    # factory.py.  Used by provider auto-detection and API-key lookup.
    # -----------------------------------------------------------------------
    "provider_prefixes": {
        "anthropic": ["claude"],
        "gemini": ["gemini"],
        "openai": ["gpt-", "o1", "o3", "text-"],
    },
    # Add prefixes per provider: {"anthropic": ["claude2"], "custom": ["mymodel-"]}.
    "provider_prefixes_add": {},
    # Remove prefixes per provider: {"openai": ["o1"]}.
    "provider_prefixes_remove": {},
    # -----------------------------------------------------------------------
    # Rate-limit profile config overrides
    # -----------------------------------------------------------------------
    # Override min_backoff_s for all providers globally (float or None).
    # Set to 0 to disable computed inter-rung backoff entirely.
    "rate_limit_backoff_s": None,
    # Override backoff_scale for all providers globally (float or None).
    "rate_limit_backoff_scale": None,
    # Extra signal strings appended to the resolved provider profile.
    # Useful for custom API gateways that return non-standard error messages.
    "rate_limit_signals_add": [],
    # Signal strings to remove from the resolved provider profile.
    "rate_limit_signals_remove": [],
    # -----------------------------------------------------------------------
    "ignore_paths": [],
    # Configurable per-file content limit sent to the LLM.
    "max_content_chars": 12000,
    # Large-file handling strategy. "truncate" is the byte-compatible default;
    # "split" opts into deterministic local source division before provider calls.
    "large_file_strategy": "truncate",
    # -----------------------------------------------------------------------
    # Planning / CI safety
    # -----------------------------------------------------------------------
    # Read-only planning run: no filesystem mutation, no provider creation.
    "dry_run": False,
    # Maximum number of files allowed to make LLM calls. 0 means unlimited.
    "max_files": 0,
    # Safety cap on initially planned LLM calls, including prompt-customization
    # reviews and initial documentation calls (0 = unlimited). Checked before
    # provider creation; retries and corrections are excluded.
    "max_planned_calls": 0,
    # Project-relative paths to reprocess even when their hash is unchanged.
    "force_files": [],
    # Exit 0 even when some files failed (completed runs only).
    "allow_partial": False,
    # -----------------------------------------------------------------------
    # Selectable per-file analysis mode
    # -----------------------------------------------------------------------
    # "single" — one combined provider call per file (default).
    # "triple" — the legacy StructureAgent/DependencyAgent/DocumentationAgent
    #            three-call path.
    "analysis_mode": "single",
    # -----------------------------------------------------------------------
    # Configurable truncation head ratio
    # -----------------------------------------------------------------------
    # Head fraction of the head-plus-tail truncation split.  The default 0.70
    # produces a ~70/30 head/tail split.
    # Must be a float strictly between 0.0 and 1.0 (exclusive).
    "truncation_head_ratio": 0.70,
    # -----------------------------------------------------------------------
    # Provider request transport timeout
    # -----------------------------------------------------------------------
    # Per connect/read/write/pool phase transport timeout (seconds) passed to
    # the provider SDK client, not a wall-clock deadline for the whole call.
    # Must be a number in the inclusive range 1-600.
    "provider_request_timeout_s": 120,
    # -----------------------------------------------------------------------
    # Targeted response correction (opt-in, disabled by default)
    # -----------------------------------------------------------------------
    # When true, a single targeted corrective provider call is made for an
    # eligible response-contract failure (malformed shape, missing/empty required
    # field, or a response that retains none of its requested fields).  Disabled
    # by default so the standard path never adds a paid repair call.  A
    # response-contract rejection is never converted into a whole-file retry.
    "response_correction_enabled": False,
    # -----------------------------------------------------------------------
    # Mode-based JSON prompt profiles
    # -----------------------------------------------------------------------
    # Inline profile object (single and/or triple sections) customizing the
    # requested JSON shape block. ``None`` means no inline profile.  This is the
    # only prompt-customization source; external profile files,
    # auto-detection, and the disable flag were removed.
    "prompt_profiles": None,
}

# CodeDoc automatically reads exactly one configuration file at the project
# root.  There is no candidate list, no ``config.json`` fallback, and no
# ``--config FILE`` runtime selector.
_CONFIG_FILENAME = "codedoc.config.json"

# Unsupported runtime keys. Detected in the exact config file and in
# in-memory overrides *before* the loader filters unknown/default keys, so a
# stale config that still sets one fails loudly instead of looking active while
# CodeDoc silently ignores it. Each value explains the current behavior.
_REMOVED_CONFIG_KEYS: dict[str, str] = {
    "safe_mode": (
        "crash recovery is always active; there is no safe_mode "
        "setting and no replacement is required."
    ),
    "manage_output_gitignore": (
        "CodeDoc no longer manages an output .gitignore; manage your "
        "version-control policy yourself."
    ),
    "output_gitignore_filename": (
        "CodeDoc no longer manages an output .gitignore; manage your "
        "version-control policy yourself."
    ),
    "prompt_profile_file": (
        "external prompt-profile files were removed; move the profile inline "
        "under 'prompt_profiles' in codedoc.config.json."
    ),
    "prompt_profile_auto_detect": (
        "prompt-profile auto-detection was removed; the only profile source is "
        "the inline 'prompt_profiles' value in codedoc.config.json."
    ),
    "prompt_profile_disabled": (
        "the prompt-profile disable flag was removed; omit 'prompt_profiles' "
        "(or set it to null) to run with developer defaults."
    ),
    "prompt_customization_allow_risky": (
        "the risky-customization override was removed; a TOO_RISKY semantic "
        "review always blocks and cannot be bypassed."
    ),
}

_ENV_KEY_MAP = {
    "LLM_PROVIDER": "llm_provider",
    "MODEL_NAME": "model_name",
    "API_BASE_URL": "api_base_url",
    "LLM_API_KEY": "api_key",
    "OUTPUT_DIR": "output_dir",
    "CODEDOC_OUTPUT_FORMAT": "output_format",
    "LOG_LEVEL": "log_level",
    "CODEDOC_IGNORE_PATHS": "ignore_paths",
    "CODEDOC_MAX_PARALLEL_FILES": "max_parallel_files",
    "CODEDOC_FILE_RETRY_ATTEMPTS": "file_retry_attempts",
    "CODEDOC_MAX_CONSECUTIVE_FAILURES": "max_consecutive_failures",
    "CODEDOC_MAX_CONTENT_CHARS": "max_content_chars",
    "CODEDOC_LARGE_FILE_STRATEGY": "large_file_strategy",
    "CODEDOC_DRY_RUN": "dry_run",
    "CODEDOC_MAX_FILES": "max_files",
    "CODEDOC_MAX_PLANNED_CALLS": "max_planned_calls",
    "CODEDOC_FORCE_FILES": "force_files",
    "CODEDOC_ALLOW_PARTIAL": "allow_partial",
    "CODEDOC_ANALYSIS_MODE": "analysis_mode",
    "CODEDOC_TRUNCATION_HEAD_RATIO": "truncation_head_ratio",
    "CODEDOC_PROVIDER_REQUEST_TIMEOUT_S": "provider_request_timeout_s",
}

# Allowed values for the selectable per-file analysis mode.
VALID_ANALYSIS_MODES = ("single", "triple")

# Allowed values for oversized readable source handling.
VALID_LARGE_FILE_STRATEGIES = ("truncate", "split")

# Config keys whose environment values are parsed as semicolon-separated lists.
_ENV_LIST_KEYS = {"ignore_paths", "force_files"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_endpoint_trust_approval(
    trust_api_base_url: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(approval_url, mechanism)`` from the two runtime-only sources.

    *trust_api_base_url* (the CLI's ``--trust-api-base-url``) wins when both it
    and ``CODEDOC_TRUST_API_BASE_URL`` are present. Selection is on presence
    (``is not None``), not truthiness: once the caller has explicitly supplied
    *trust_api_base_url*, the environment variable is never consulted, even if
    that value turns out to be blank or the wrong type -- a present higher-
    precedence candidate must fail on its own terms, never silently defer to a
    lower one. Neither source is ever read from configuration.
    """
    if trust_api_base_url is not None:
        if not _non_empty_string(trust_api_base_url):
            # Never render the supplied value: a non-string container may hold
            # an approval URL with embedded credentials, and every
            # authorization failure must emit no approval URL and no
            # credential.  Name the type only.
            raise ConfigError(
                "--trust-api-base-url must be a non-empty string when supplied "
                f"(received a value of type {type(trust_api_base_url).__name__})."
            )
        return trust_api_base_url.strip(), "--trust-api-base-url"
    env_value = os.environ.get(_TRUST_API_BASE_URL_ENV)
    if env_value and env_value.strip():
        return env_value.strip(), _TRUST_API_BASE_URL_ENV
    return None, None


def load_config(
    root: Path,
    overrides: dict[str, Any] | None = None,
    *,
    trust_api_base_url: str | None = None,
) -> ResolvedConfig:
    """Load and merge config from the exact config file, environment, and defaults.

    CodeDoc automatically reads exactly one persistent configuration file,
    ``<root>/codedoc.config.json``.  No other filename is probed, no ``.env`` is
    loaded, and there is no ``--config FILE`` selector.  Programmatic callers may
    still pass in-memory *overrides*; that does not create a second persistent
    source.

    A non-empty ``api_base_url`` additionally requires runtime endpoint-trust
    approval from exactly two sources -- the *trust_api_base_url* keyword (the
    CLI's ``--trust-api-base-url``) and the ``CODEDOC_TRUST_API_BASE_URL``
    environment variable -- and can never be satisfied by
    ``codedoc.config.json`` or *overrides* (see ``_reject_endpoint_trust_keys``).
    The gate is resolved, and any credential environment variable is read, only
    after that decision (two-phase credential resolution, below), and applies
    identically to a dry run.
    """
    config: dict[str, Any] = dict(DEFAULTS)

    # Phase 1: merge every ordinary key from every source. ``api_key`` is
    # deliberately excluded from these merges -- its file/overrides candidates
    # are retained in local variables only, and no credential environment
    # variable is read, until the endpoint-authorization gate below resolves.
    file_api_key_candidate: Any = None
    candidate = root / _CONFIG_FILENAME
    if candidate.exists():
        try:
            # Parse through the shared strict loader so a duplicate object key
            # anywhere in the file (including nested inside ``prompt_profiles``)
            # is rejected instead of silently last-key-wins.
            data = loads_no_duplicate_keys(candidate.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ConfigError(
                    f"'{_CONFIG_FILENAME}' must be a JSON object, got {type(data).__name__}"
                )
            _reject_removed_keys(data, source=_CONFIG_FILENAME)
            _reject_endpoint_trust_keys(data, source=_CONFIG_FILENAME)
            _reject_unknown_keys(data, source=_CONFIG_FILENAME)
            file_api_key_candidate = data.get("api_key")
            config.update({k: v for k, v in data.items() if k != "api_key"})
            logger.info("Config loaded from %s", candidate)
        except DuplicateJSONKeyError as exc:
            raise ConfigError(
                f"Invalid JSON in '{_CONFIG_FILENAME}': duplicate key {exc.key!r}."
            ) from exc
        except NonFiniteJSONNumberError as exc:
            raise ConfigError(f"Invalid JSON in '{_CONFIG_FILENAME}': {exc}.") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in '{_CONFIG_FILENAME}': {exc}") from exc
    else:
        logger.info("No %s found in %s; using defaults.", _CONFIG_FILENAME, root)

    for env_key, config_key in _ENV_KEY_MAP.items():
        if env_key == "LLM_API_KEY":
            continue  # credential env var: deferred to phase 2, below the gate
        val = os.environ.get(env_key)
        if val:
            if config_key in _ENV_LIST_KEYS:
                config[config_key] = [p.strip() for p in val.split(";") if p.strip()]
            else:
                config[config_key] = val
            logger.debug("Config override from env: %s", env_key)

    overrides_api_key_candidate: Any = None
    if overrides:
        _reject_removed_keys(overrides, source="config_overrides")
        _reject_endpoint_trust_keys(overrides, source="config_overrides")
        _reject_unknown_keys(overrides, source="config_overrides")
        overrides_api_key_candidate = overrides.get("api_key")
        config.update({k: v for k, v in overrides.items() if k != "api_key"})

    # Validate values before merge helpers iterate or coerce them. This ensures
    # malformed collections and scalar paths produce ConfigError rather than a
    # raw TypeError/AttributeError. api_key's own shape is validated in phase 2
    # below, once its value is resolved.
    _validate_pre_resolution(config)

    # --- Endpoint-authorization gate --------------------------------------
    # Resolved before any credential is read or selected, and applied on the
    # same terms to a dry run. An empty api_base_url is the default provider
    # endpoint and needs no approval; a non-empty one requires the approval URL
    # to canonicalize to exactly the same digest.
    approval_url, approval_mechanism = _resolve_endpoint_trust_approval(trust_api_base_url)
    configured_base_url = config.get("api_base_url") or None
    endpoint_trust: EndpointTrustAttestation | None = None
    if approval_url and not configured_base_url:
        raise ConfigError(
            "An endpoint-trust approval was supplied (via --trust-api-base-url "
            f"or {_TRUST_API_BASE_URL_ENV}) but no api_base_url is configured. "
            "Remove the approval, or set api_base_url to the endpoint it approves."
        )
    if configured_base_url:
        configured_digest = effective_endpoint_identity(configured_base_url)
        if not approval_url:
            raise ConfigError(
                "api_base_url is set to a custom endpoint (digest "
                f"{configured_digest}) but no runtime approval was supplied. "
                "Approve the exact endpoint using --trust-api-base-url or "
                f"{_TRUST_API_BASE_URL_ENV}, or remove api_base_url to use the "
                "default provider endpoint."
            )
        approval_digest = effective_endpoint_identity(approval_url)
        if approval_digest != configured_digest:
            raise ConfigError(
                "The supplied endpoint-trust approval does not match the "
                f"configured api_base_url (digest {configured_digest}). Approve "
                "exactly this endpoint using --trust-api-base-url or "
                f"{_TRUST_API_BASE_URL_ENV}."
            )
        endpoint_trust = EndpointTrustAttestation(
            digest=configured_digest, mechanism=approval_mechanism
        )

    # --- Phase 2: credential resolution, now that authorization is settled.
    # Precedence, last present source wins: config-file api_key (lowest),
    # LLM_API_KEY (overrides it), programmatic config_overrides api_key
    # (overrides both). Provider-specific fallbacks (OPENAI_API_KEY, etc.) stay
    # in create_provider(), applied only when this resolves to nothing.
    llm_api_key_env = os.environ.get("LLM_API_KEY")
    resolved_api_key: Any = None
    for source_value in (file_api_key_candidate, llm_api_key_env, overrides_api_key_candidate):
        if source_value is not None:
            resolved_api_key = source_value
    if resolved_api_key is not None and not _non_empty_string(resolved_api_key):
        raise ConfigError("api_key must be a non-empty string or null.")
    config["api_key"] = resolved_api_key

    # Resolve <key> / <key>_add / <key>_remove overrides for configurable
    # default keys.  Must run after all sources are merged so the final
    # _add / _remove values are available.
    _apply_config_overrides(config)

    # Resolve output path — must run after all sources are merged so the
    # final output_dir value is available.
    _resolve_output_spec(config, overrides or {})

    _validate(config)
    return ResolvedConfig(config, endpoint_trust=endpoint_trust)


def validate_config_data(data: dict[str, Any], *, source: str = "configuration") -> None:
    """Validate one config object's syntax without reading environment variables
    or files, and without performing runtime endpoint authorization.

    Checks that ``api_base_url`` (if present) is syntactically a valid
    HTTP/HTTPS URL carrying no username, password, query string, or fragment --
    the same syntax rule ``load_config()``'s authorization gate applies -- and
    rejects the endpoint-trust keys under the same message as ``load_config()``.
    It never resolves or requires an endpoint-trust approval and can never by
    itself authorize a run: only ``load_config()``'s runtime gate does that.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must be a JSON object.")
    _reject_removed_keys(data, source=source)
    _reject_endpoint_trust_keys(data, source=source)
    _reject_unknown_keys(data, source=source)
    config = dict(DEFAULTS)
    config.update(data)
    _validate_pre_resolution(config)
    effective_endpoint_identity(config.get("api_base_url"))
    _apply_config_overrides(config)
    _resolve_output_spec(config, data)
    _validate(config, warn_missing_api_key=False)


# ---------------------------------------------------------------------------
# Removed-key detection
# ---------------------------------------------------------------------------


def _reject_removed_keys(data: dict[str, Any], *, source: str) -> None:
    """Raise :class:`ConfigError` when *data* sets an unsupported key.

    Detects the removed keys *before* the loader filters unknown/default keys so a
    stale config or override that still sets one fails loudly with the replacement
    behavior, instead of looking active while CodeDoc silently ignores it.  Names
    every offending key so a config carrying several is fixed in one pass.
    """
    if not isinstance(data, dict):
        return
    offending = [key for key in _REMOVED_CONFIG_KEYS if key in data]
    if not offending:
        return
    details = "\n".join(
        f"  - '{key}': {_REMOVED_CONFIG_KEYS[key]}" for key in offending
    )
    plural = "keys" if len(offending) > 1 else "key"
    raise ConfigError(
        f"{source} sets {len(offending)} removed configuration {plural}:\n{details}\n"
        "Remove the listed key(s) to continue."
    )


def _reject_unknown_keys(data: dict[str, Any], *, source: str) -> None:
    """Reject all unknown top-level configuration keys deterministically."""
    unknown = sorted(str(key) for key in data if key not in DEFAULTS)
    if not unknown:
        return
    details = "\n".join(f"  - {key}" for key in unknown)
    plural = "keys" if len(unknown) != 1 else "key"
    raise ConfigError(
        f"{source} contains {len(unknown)} unknown configuration {plural}:\n"
        f"{details}\nCorrect or remove the listed key(s)."
    )


# ---------------------------------------------------------------------------
# Override-resolution helpers
# ---------------------------------------------------------------------------


def _resolve_list_override(
    key: str,
    raw_config: dict[str, Any],
    defaults: dict[str, Any],
) -> list:
    """Resolve a list config key with optional ``<key>_add`` / ``<key>_remove``.

    Resolution order:
    1. ``<key>`` — replaces the default list entirely when explicitly set.
    2. ``<key>_add`` — appends new items (duplicates suppressed, order preserved).
    3. ``<key>_remove`` — removes specified items.
    """
    base = list(raw_config.get(key, defaults.get(key, [])))

    add = raw_config.get(f"{key}_add") or []
    seen = set(base)
    for item in add:
        if item not in seen:
            base.append(item)
            seen.add(item)

    remove_set = set(raw_config.get(f"{key}_remove") or [])
    if remove_set:
        base = [item for item in base if item not in remove_set]

    return base


def _resolve_dict_override(
    key: str,
    raw_config: dict[str, Any],
    defaults: dict[str, Any],
) -> dict:
    """Resolve a flat dict config key with optional ``<key>_add`` / ``<key>_remove``.

    Resolution order:
    1. ``<key>`` — replaces the default dict entirely when explicitly set.
    2. ``<key>_add`` — updates with new key→value pairs (overwrites existing).
    3. ``<key>_remove`` — removes keys listed in the value (list of strings).
    """
    base = dict(raw_config.get(key, defaults.get(key, {})))

    add = raw_config.get(f"{key}_add") or {}
    if isinstance(add, dict):
        base.update(add)

    remove = raw_config.get(f"{key}_remove") or []
    for k in remove:
        base.pop(k, None)

    return base


def _resolve_nested_list_dict_override(
    key: str,
    raw_config: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, list[str]]:
    """Resolve a ``dict[str, list[str]]`` config key with ``<key>_add`` / ``<key>_remove``.

    Used for ``provider_prefixes`` where each value is a list of model-name
    prefix strings.

    Resolution order:
    1. ``<key>`` — replaces defaults when explicitly set.
    2. ``<key>_add`` — a ``dict[str, list[str]]``; appends prefixes per provider.
    3. ``<key>_remove`` — a ``dict[str, list[str]]``; removes prefixes per provider.
    """
    base: dict[str, list[str]] = {
        k: list(v) for k, v in raw_config.get(key, defaults.get(key, {})).items()
    }

    add = raw_config.get(f"{key}_add") or {}
    if isinstance(add, dict):
        for provider, prefixes in add.items():
            existing = base.setdefault(provider, [])
            seen = set(existing)
            for prefix in prefixes or []:
                if prefix not in seen:
                    existing.append(prefix)
                    seen.add(prefix)

    remove = raw_config.get(f"{key}_remove") or {}
    if isinstance(remove, dict):
        for provider, prefixes in remove.items():
            if provider in base:
                remove_set = set(prefixes or [])
                base[provider] = [p for p in base[provider] if p not in remove_set]

    return base


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_string_list(config: dict[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, list) or not all(
        _non_empty_string(item) for item in value
    ):
        raise ConfigError(f"{key} must be a list of non-empty strings.")


def _validate_extension(value: Any, key: str) -> None:
    if not _non_empty_string(value) or not value.startswith(".") or len(value) < 2:
        raise ConfigError(f"{key} entries must be extensions such as '.py'.")


def _reject_non_finite(value: Any, key: str = "configuration") -> None:
    """Reject NaN and infinities in in-memory configuration recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{key} contains a non-finite number.")
    if isinstance(value, dict):
        for child_key, child in value.items():
            _reject_non_finite(child, f"{key}.{child_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{key}[{index}]")


def _validate_pre_resolution(config: dict[str, Any]) -> None:
    """Validate shapes that merge/path resolution assumes are well formed."""
    _reject_non_finite(config)

    required_strings = (
        "llm_mode",
        "llm_provider",
        "model_name",
        "documentation_scope",
        "output_dir",
        "output_format",
        "output_json_filename",
        "output_md_filename",
        "log_level",
        "analysis_mode",
    )
    for key in required_strings:
        value = config.get(key)
        if not isinstance(value, str) or (key != "model_name" and not value.strip()):
            raise ConfigError(
                f"{key} must be a string{'' if key == 'model_name' else ' and must not be empty'}."
            )
    for key in ("entry_file", "api_base_url", "api_key"):
        value = config.get(key)
        if value is not None and not _non_empty_string(value):
            raise ConfigError(f"{key} must be a non-empty string or null.")

    json_name = _validate_portable_filename(
        config["output_json_filename"], "output_json_filename"
    )
    md_name = _validate_portable_filename(
        config["output_md_filename"], "output_md_filename"
    )
    if not json_name.lower().endswith(".json"):
        raise ConfigError("output_json_filename must end in '.json'.")
    if not md_name.lower().endswith(".md"):
        raise ConfigError("output_md_filename must end in '.md'.")

    for key in (
        "skip_dirs",
        "skip_dirs_add",
        "skip_dirs_remove",
        "ignore_paths",
        "auto_entry_candidates",
        "auto_entry_candidates_add",
        "auto_entry_candidates_remove",
        "rate_limit_signals_add",
        "rate_limit_signals_remove",
        "force_files",
    ):
        _validate_string_list(config, key)

    for key in ("supported_extensions", "extension_language_map_remove"):
        value = config.get(key)
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be a list of file extensions.")
        for item in value:
            _validate_extension(item, key)

    for key in ("extension_language_map", "extension_language_map_add"):
        value = config.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must map file extensions to language tags.")
        for extension, language in value.items():
            _validate_extension(extension, key)
            if not _non_empty_string(language):
                raise ConfigError(f"{key} language tags must be non-empty strings.")

    for key in (
        "provider_prefixes",
        "provider_prefixes_add",
        "provider_prefixes_remove",
    ):
        value = config.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must map providers to lists of model prefixes.")
        for provider, prefixes in value.items():
            if (
                not _non_empty_string(provider)
                or not isinstance(prefixes, list)
                or not all(_non_empty_string(prefix) for prefix in prefixes)
            ):
                raise ConfigError(
                    f"{key} must map non-empty provider names to lists of non-empty strings."
                )

    ladder = config.get("parallel_ladder")
    if ladder is not None and not isinstance(ladder, list):
        raise ConfigError(
            "parallel_ladder must be null or a list of positive integers."
        )


def _apply_config_overrides(config: dict[str, Any]) -> None:
    """Apply ``<key>_add`` / ``<key>_remove`` resolutions for all configurable keys.

    Modifies *config* in-place.  Must be called after all sources (JSON, env,
    CLI overrides) have been merged into *config*.
    """
    config["skip_dirs"] = _resolve_list_override("skip_dirs", config, DEFAULTS)
    config["extension_language_map"] = _resolve_dict_override(
        "extension_language_map", config, DEFAULTS
    )
    config["auto_entry_candidates"] = _resolve_list_override(
        "auto_entry_candidates", config, DEFAULTS
    )
    config["provider_prefixes"] = _resolve_nested_list_dict_override(
        "provider_prefixes", config, DEFAULTS
    )
    # Backward-compat bridge for explicit supported_extensions overrides.
    #
    # Older configs may set "supported_extensions": [".py", ".ts"] to restrict
    # scanning.  extension_language_map is the single source of truth, but we
    # honour an *explicit* supported_extensions override by applying it as a
    # filter on the resolved map — so old configs keep working without migration.
    #
    # Detection rule: if config["supported_extensions"] differs from
    # DEFAULTS["supported_extensions"] it was explicitly set by the user
    # (JSON / env / CLI).  In that case intersect the resolved map with the
    # user's list, giving them the restrictive behaviour they expected.
    _raw_supported = config.get("supported_extensions")
    _default_supported = set(DEFAULTS.get("supported_extensions", []))
    if (
        _raw_supported is not None
        and isinstance(_raw_supported, list)
        and set(_raw_supported) != _default_supported
    ):
        _user_exts = {e.lower() for e in _raw_supported}
        config["extension_language_map"] = {
            k: v
            for k, v in config["extension_language_map"].items()
            if k.lower() in _user_exts
        }

    # Derive supported_extensions from the final resolved map.
    # extension_language_map is now the single source of truth for which
    # extensions are scanned and what language they are labelled as.
    config["supported_extensions"] = sorted(config["extension_language_map"].keys())


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------


def _resolve_output_spec(config: dict, overrides: dict) -> None:
    """
    Parse output_dir for an optional file extension and resolve format + filename.

    Supported forms
    ---------------
    Directory (no extension):
        "codedoc"              → unchanged; filenames stay codedoc.json / codedoc.md
        "path/to/my_docs"      → unchanged
        "."                    → project root

    File path with supported extension:
        "docs/report.json"     → output_dir="docs", format="json",
                                  json_filename="report.json"
        "report.md"            → output_dir=".", format="md",
                                  md_filename="report.md"

    File path with unsupported extension:
        "report.txt"           → ConfigError (stops execution)
    """
    raw = str(config.get("output_dir", "codedoc"))
    p = Path(raw)
    suffix = p.suffix.lower()

    if not suffix:
        # Plain directory path — nothing more to do.
        return

    if suffix not in (".json", ".md"):
        raise ConfigError(
            f"Unsupported output file extension '{suffix}'. "
            "Provide a directory name (e.g. 'docs_output') or a file path "
            "ending in '.json' or '.md' (e.g. 'docs/report.json')."
        )

    # ``crash_recovery.json`` is the single fixed name codedoc keeps for its
    # dedicated crash-recovery file and must never be a user output target.  Check
    # the user-supplied filename exactly (case-insensitively, matching the
    # collision guard applied to the resolved paths).  The same constant guards the
    # recovery-path writer so the two cannot drift.  Imported locally to avoid an
    # import cycle with resume.py.
    from codedoc.core.resume import RECOVERY_FILENAME

    if p.name.casefold() == RECOVERY_FILENAME.casefold():
        raise ConfigError(
            f"'{p.name}' is the reserved crash-recovery filename, which codedoc "
            "keeps separate from the stable output and cannot be used as an output "
            "target.\n"
            "Choose a different output name — e.g. 'docs/report.json' or a "
            "directory like 'docs_output'."
        )

    # --- File path with a valid extension ---
    inferred_format = "json" if suffix == ".json" else "md"
    parent_str = str(p.parent)  # "." when no directory component

    # Check for format conflicts with the file extension.
    # config["output_format"] holds the merged value from all sources at this point.
    current_format = config.get("output_format")
    if current_format and current_format != inferred_format:
        if current_format == "both":
            raise ConfigError(
                f"'--format both' cannot be combined with a named output file ('{p.name}'). "
                f"Provide a directory instead — e.g. '--output {parent_str or '.'}' — "
                "and codedoc will write both codedoc.json and codedoc.md there."
            )
        # Only warn when the user explicitly set --format; the default value "json"
        # from DEFAULTS does not warrant a warning.
        if "output_format" in overrides:
            logger.warning(
                "Output file '%s' implies format '%s', but --format '%s' was also "
                "specified. The file extension takes precedence — '%s' will be used.",
                p.name,
                inferred_format,
                current_format,
                inferred_format,
            )

    config["output_dir"] = parent_str
    config["output_format"] = inferred_format

    # A named single-format target has one deterministic read-only conversion
    # sibling: the same stem with the opposite supported extension.  Only the
    # selected format is written; resolving the sibling name here prevents a
    # later ``report.json`` run from probing an unrelated default ``codedoc.md``.
    if inferred_format == "json":
        config["output_json_filename"] = p.name
        config["output_md_filename"] = p.with_suffix(".md").name
    else:
        config["output_md_filename"] = p.name
        config["output_json_filename"] = p.with_suffix(".json").name

    logger.info(
        "Output path resolved: dir='%s'  file='%s'  format='%s'",
        parent_str,
        p.name,
        inferred_format,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TRUE_STRINGS = ("true", "1", "yes")
_FALSE_STRINGS = ("false", "0", "no")


def _coerce_strict_bool(value: Any, key: str) -> bool:
    """Coerce a boolean config value, rejecting unrecognized strings.

    Accepts real booleans and the repository's documented boolean string forms
    (``"true"``/``"1"``/``"yes"`` and ``"false"``/``"0"``/``"no"``).  Any other
    string raises :class:`ConfigError` rather than being silently coerced.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise ConfigError(f"{key} must be a boolean (true/false); got {value!r}.")
    raise ConfigError(f"{key} must be a boolean (true/false); got {value!r}.")


def _coerce_strict_int(value: Any, key: str) -> int:
    """Accept integers and integer strings while rejecting bools/floats."""
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ConfigError(f"{key} must be an integer.")


def _validate(config: dict[str, Any], *, warn_missing_api_key: bool = True) -> None:
    """Raise ConfigError for invalid values."""
    if config.get("llm_mode", "api") != "api":
        raise ConfigError(
            f"Unsupported llm_mode '{config['llm_mode']}'. "
            "The only supported mode is 'api'."
        )

    if config.get("llm_provider") not in ("auto", "openai", "anthropic", "gemini"):
        raise ConfigError(
            "llm_provider must be one of: 'auto', 'openai', 'anthropic', or 'gemini'."
        )

    if config.get("documentation_scope", "entry") not in ("entry", "all"):
        raise ConfigError("documentation_scope must be 'entry' or 'all'.")

    # Selectable per-file analysis mode — reject unknown values before
    # provider creation.
    if config.get("analysis_mode", "single") not in VALID_ANALYSIS_MODES:
        raise ConfigError(
            "analysis_mode must be 'single' (one combined call per file) or "
            "'triple' (the three-agent path)."
        )

    if config.get("large_file_strategy", "truncate") not in VALID_LARGE_FILE_STRATEGIES:
        raise ConfigError("large_file_strategy must be exactly 'truncate' or 'split'.")

    config["log_level"] = config["log_level"].upper()
    if config["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ConfigError("log_level must be DEBUG, INFO, WARNING, or ERROR.")

    # Normalize every public boolean consistently. Integers are rejected even
    # though bool is an int subclass in Python; documented environment strings
    # remain accepted.
    for key in (
        "parallel_agents",
        "follow_symlinks",
        "propagate_changes",
        "rate_limit_adaptive",
        "respect_retry_after",
        "dry_run",
        "allow_partial",
        "response_correction_enabled",
    ):
        config[key] = _coerce_strict_bool(config[key], key)

    if (
        config.get("analysis_mode", "single") == "triple"
        and config.get("large_file_strategy", "truncate") == "split"
    ):
        raise ConfigError(
            "large_file_strategy 'split' is unavailable with analysis_mode "
            "'triple'; use analysis_mode 'single' for split planning, or use "
            "large_file_strategy 'truncate' for triple mode."
        )

    if (
        config.get("large_file_strategy", "truncate") == "split"
        and not config["dry_run"]
        and not current_split_release_policy().execution
    ):
        raise ConfigError(
            "Real large_file_strategy 'split' execution is disabled by this "
            "build's release policy. Set dry_run=true or pass --dry-run to "
            "inspect the split plan; use large_file_strategy 'truncate' for a "
            "real run."
        )

    if (
        warn_missing_api_key
        and config["llm_mode"] == "api"
        and not config["dry_run"]
        and not (config.get("api_key") or _has_provider_api_key())
    ):
        logger.warning(
            "llm_mode is 'api' but no API key was found. Set LLM_API_KEY, "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY "
            "as an environment variable."
        )

    if not isinstance(config.get("supported_extensions"), list):
        raise ConfigError("supported_extensions must be a list of file extensions.")

    if not isinstance(config.get("skip_dirs"), list):
        raise ConfigError("skip_dirs must be a list of directory names.")

    if not isinstance(config.get("ignore_paths"), list):
        raise ConfigError("ignore_paths must be a list of project-relative paths.")

    if not isinstance(config.get("extension_language_map"), dict):
        raise ConfigError(
            "extension_language_map must be a dict mapping file extensions to language tags."
        )

    if not isinstance(config.get("auto_entry_candidates"), list):
        raise ConfigError("auto_entry_candidates must be a list of file names.")

    if not isinstance(config.get("provider_prefixes"), dict):
        raise ConfigError(
            "provider_prefixes must be a dict mapping provider names to lists of model prefixes."
        )

    # Validate rate-limit profile override keys
    _rls = config.get("rate_limit_backoff_s")
    if _rls is not None:
        try:
            if isinstance(_rls, bool):
                raise ValueError
            _v = float(_rls)
            if not math.isfinite(_v) or _v < 0:
                raise ValueError
            config["rate_limit_backoff_s"] = _v
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "rate_limit_backoff_s must be a non-negative number or null."
            ) from exc

    _rls = config.get("rate_limit_backoff_scale")
    if _rls is not None:
        try:
            if isinstance(_rls, bool):
                raise ValueError
            _v = float(_rls)
            if not math.isfinite(_v) or _v <= 0:
                raise ValueError
            config["rate_limit_backoff_scale"] = _v
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "rate_limit_backoff_scale must be a positive number or null."
            ) from exc

    if not isinstance(config.get("rate_limit_signals_add", []), list):
        raise ConfigError("rate_limit_signals_add must be a list of strings.")

    if not isinstance(config.get("rate_limit_signals_remove", []), list):
        raise ConfigError("rate_limit_signals_remove must be a list of strings.")

    if config.get("output_format") not in ("json", "md", "both"):
        raise ConfigError("output_format must be one of: 'json', 'md', or 'both'.")

    config["max_file_size_kb"] = _coerce_strict_int(
        config["max_file_size_kb"], "max_file_size_kb"
    )
    if config["max_file_size_kb"] < 1:
        raise ConfigError(
            "max_file_size_kb must be at least 1; a non-positive value would "
            "silently skip every file."
        )

    for key in (
        "max_parallel_files",
        "file_retry_attempts",
        "max_consecutive_failures",
    ):
        config[key] = _coerce_strict_int(config[key], key)

    if config["max_parallel_files"] < 1:
        raise ConfigError("max_parallel_files must be at least 1.")

    if config["file_retry_attempts"] < 0:
        raise ConfigError("file_retry_attempts must be 0 or greater.")

    if config["max_consecutive_failures"] < 1:
        raise ConfigError("max_consecutive_failures must be at least 1.")

    # Validate and normalise parallel_ladder
    ladder = config.get("parallel_ladder")
    if ladder is not None:
        if not isinstance(ladder, list) or not all(
            isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in ladder
        ):
            raise ConfigError(
                "parallel_ladder must be a list of positive integers, e.g. [5, 2, 1]."
            )
        if ladder != sorted(ladder, reverse=True):
            raise ConfigError(
                "parallel_ladder values must be strictly decreasing (highest first), "
                "e.g. [5, 2, 1]."
            )
        # Clamp values exceeding max_parallel_files
        max_p = config["max_parallel_files"]
        if any(x > max_p for x in ladder):
            logger.warning(
                "parallel_ladder contains value(s) exceeding max_parallel_files (%d); "
                "clamping ladder values.",
                max_p,
            )
            seen: set[int] = set()
            clamped: list[int] = []
            for x in (min(v, max_p) for v in ladder):
                if x not in seen:
                    seen.add(x)
                    clamped.append(x)
            ladder = clamped
            config["parallel_ladder"] = ladder
        # Ensure 1 is always the last rung
        if ladder[-1] != 1:
            config["parallel_ladder"] = ladder + [1]

    # Coerce retry_after_cap_s to int.  Zero is valid and intentionally
    # disables the backoff cap; negative values are invalid.  Booleans rejected.
    config["retry_after_cap_s"] = _coerce_strict_int(
        config.get("retry_after_cap_s", 30), "retry_after_cap_s"
    )
    if config["retry_after_cap_s"] < 0:
        raise ConfigError("retry_after_cap_s must be 0 or greater.")

    config["max_content_chars"] = _coerce_strict_int(
        config["max_content_chars"], "max_content_chars"
    )
    if config["max_content_chars"] < 1000:
        raise ConfigError("max_content_chars must be at least 1000.")

    # max_files — integer >= 0; 0 means unlimited; booleans rejected.
    config["max_files"] = _coerce_strict_int(config.get("max_files", 0), "max_files")
    if config["max_files"] < 0:
        raise ConfigError("max_files must be an integer greater than or equal to 0.")

    # max_planned_calls — integer >= 0; 0 means unlimited; booleans rejected.
    config["max_planned_calls"] = _coerce_strict_int(
        config.get("max_planned_calls", 0), "max_planned_calls"
    )
    if config["max_planned_calls"] < 0:
        raise ConfigError(
            "max_planned_calls must be an integer greater than or equal to 0."
        )

    # force_files — a list of non-empty path strings.
    force_files = config.get("force_files", [])
    if not isinstance(force_files, list) or not all(
        isinstance(p, str) and p.strip() for p in force_files
    ):
        raise ConfigError("force_files must be a list of non-empty path strings.")
    config["force_files"] = [p.strip() for p in force_files]

    # truncation_head_ratio — float strictly between 0.0 and 1.0.
    raw_ratio = config.get("truncation_head_ratio", 0.70)
    if isinstance(raw_ratio, bool):
        raise ConfigError(
            "truncation_head_ratio must be a number strictly between 0.0 and 1.0 "
            "(exclusive); got a boolean."
        )
    try:
        ratio_val = float(raw_ratio)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "truncation_head_ratio must be a number strictly between 0.0 and 1.0 "
            f"(exclusive); got {raw_ratio!r}."
        ) from exc
    if not math.isfinite(ratio_val) or not (0.0 < ratio_val < 1.0):
        raise ConfigError(
            "truncation_head_ratio must be strictly between 0.0 and 1.0 "
            f"(exclusive); got {ratio_val!r}."
        )
    config["truncation_head_ratio"] = ratio_val

    # provider_request_timeout_s — a per connect/read/write/pool phase
    # transport timeout (seconds), inclusive range 1-600, default 120.
    # Strictly ASCII decimal strings or non-boolean numeric values only:
    # no sign, no exponent, no underscore digit grouping, no non-ASCII
    # digits, no NaN/infinity. Never echoes the rejected input (section
    # 5.5). Section 12.1 C3: the key is absent only when nothing overwrote
    # DEFAULTS' own 120, which this dict.get default matches -- an explicit
    # JSON/Python null (config-file or config_overrides) or an empty string
    # (config-file or CLI text) must fail boundedly instead of silently
    # defaulting; an empty environment variable remains absent, handled
    # generically upstream by the "only truthy env values override" rule.
    raw_timeout = config.get("provider_request_timeout_s", 120)
    _timeout_error = ConfigError(
        "provider_request_timeout_s must be a number between 1 and 600 (inclusive)."
    )
    if raw_timeout is None:
        raise _timeout_error
    if isinstance(raw_timeout, bool):
        raise _timeout_error
    if isinstance(raw_timeout, (int, float)):
        timeout_val = float(raw_timeout)
    elif isinstance(raw_timeout, str):
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", raw_timeout):
            raise _timeout_error
        timeout_val = float(raw_timeout)
    else:
        raise _timeout_error
    if not math.isfinite(timeout_val) or not (1 <= timeout_val <= 600):
        raise _timeout_error
    config["provider_request_timeout_s"] = timeout_val

    # Inline prompt-customization profile. Structural profile validation happens
    # later in prompt_profiles.resolve_profile_source; here we only enforce that
    # the config-level value is an object or null.  ``prompt_profiles`` is the only
    # profile source.
    inline_profiles = config.get("prompt_profiles")
    if inline_profiles is not None and not isinstance(inline_profiles, dict):
        raise ConfigError("prompt_profiles must be an inline JSON object or null.")


_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _validate_portable_filename(value: Any, key: str) -> str:
    """Validate one portable basename without reading or writing its target."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ConfigError(f"{key} must be one non-empty portable filename.")
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or "/" in value
        or "\\" in value
        or value[-1] in {" ", "."}
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ConfigError(f"{key} must be one non-empty portable filename.")
    stem = re.split(r"\.", value, maxsplit=1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise ConfigError(f"{key} uses reserved device name '{stem}'.")
    return value


def _has_provider_api_key() -> bool:
    return any(
        os.environ.get(key)
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "LLM_API_KEY",
        )
    )
