"""Config loader for codedoc."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path):
        return False

from codedoc.utils.errors import ConfigError
from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULTS: dict[str, Any] = {
    "llm_mode": "api",
    "llm_provider": "auto",
    "model_name": "",
    "api_base_url": None,
    "api_key": None,
    "entry_file": None,
    "output_dir": "codedoc",
    "output_format": "json",
    "output_json_filename": "codedoc.json",
    "output_md_filename": "codedoc.md",
    # supported_extensions: read-only after load_config() — always derived from
    # the resolved extension_language_map.  The value listed here is the legacy
    # default set and acts as the detection baseline: if a caller passes a
    # *different* list, _apply_config_overrides() treats it as a filter on the
    # extension_language_map (backward-compat bridge for pre-0.8.1 configs).
    "supported_extensions": [
        ".py", ".ts", ".tsx", ".js", ".jsx", ".dart",
        ".java", ".cs", ".html",
    ],
    "safe_mode": False,
    "parallel_agents": True,
    "max_parallel_files": 5,
    "file_retry_attempts": 1,
    "max_consecutive_failures": 5,
    "log_level": "INFO",
    "max_file_size_kb": 500,
    "propagate_changes": True,
    # 0.8.0 rate-limit adaptive parallelism
    "rate_limit_adaptive": True,
    "parallel_ladder": None,
    "respect_retry_after": True,
    "retry_after_cap_s": 30,
    # -----------------------------------------------------------------------
    # 0.8.1 skip_dirs — single source of truth (was split across loader + scanner)
    # -----------------------------------------------------------------------
    "skip_dirs": [
        "__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env", "myenv",
        ".env", "node_modules", "site-packages", "dist-packages", "dist", "build",
        ".next", ".nuxt", "target", "codedoc", ".mypy_cache", ".pytest_cache",
        ".ruff_cache",
    ],
    # Extend skip_dirs without replacing the full list.
    "skip_dirs_add": [],
    # Remove entries from skip_dirs.  Use to allow scanning a package whose
    # directory name appears in the default list (e.g. "codedoc").
    "skip_dirs_remove": [],
    # -----------------------------------------------------------------------
    # 0.8.1 extension_language_map — replaces the hardcoded EXTENSION_LANGUAGE_MAP
    # in scanner.py.  Any extension in the resolved map is automatically
    # supported — no need to edit both this and supported_extensions.
    # -----------------------------------------------------------------------
    "extension_language_map": {
        ".py":   "python",
        ".ts":   "typescript",
        ".tsx":  "tsx",
        ".js":   "javascript",
        ".jsx":  "jsx",
        ".dart": "dart",
        ".java": "java",
        ".cs":   "csharp",
        ".html": "html",
        ".htm":  "html",
        ".kt":   "kotlin",
        ".swift":"swift",
        ".go":   "go",
        ".rb":   "ruby",
        ".rs":   "rust",
        ".cpp":  "cpp",
        ".c":    "c",
        ".h":    "c",
        ".hpp":  "cpp",
    },
    # Add new extension → language entries (merged with extension_language_map).
    "extension_language_map_add": {},
    # Remove extensions from the map (list of extension strings, e.g. [".htm"]).
    "extension_language_map_remove": [],
    # -----------------------------------------------------------------------
    # 0.8.1 auto_entry_candidates — replaces the hardcoded common_entries list
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
    # 0.8.1 provider_prefixes — replaces the hardcoded _*_PREFIXES tuples in
    # factory.py.  Used by provider auto-detection and API-key lookup.
    # -----------------------------------------------------------------------
    "provider_prefixes": {
        "anthropic": ["claude"],
        "gemini":    ["gemini"],
        "openai":    ["gpt-", "o1", "o3", "text-"],
    },
    # Add prefixes per provider: {"anthropic": ["claude2"], "custom": ["mymodel-"]}.
    "provider_prefixes_add": {},
    # Remove prefixes per provider: {"openai": ["o1"]}.
    "provider_prefixes_remove": {},
    # -----------------------------------------------------------------------
    # 0.8.1 rate-limit profile config overrides
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
}

_CONFIG_FILENAMES = ["codedoc.config.json", "config.json"]
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
    "CODEDOC_SAFE_MODE": "safe_mode",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_config(root: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load and merge config from JSON, .env, environment, and defaults."""
    config: dict[str, Any] = dict(DEFAULTS)

    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.debug("Loaded .env from %s", env_file)

    json_loaded = False
    for filename in _CONFIG_FILENAMES:
        candidate = root / filename
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ConfigError(
                        f"'{filename}' must be a JSON object, got {type(data).__name__}"
                    )
                config.update({k: v for k, v in data.items() if k in DEFAULTS})
                logger.info("Config loaded from %s", candidate)
                json_loaded = True
                break
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSON in '{filename}': {exc}") from exc

    if not json_loaded:
        logger.info("No codedoc.config.json or config.json found in %s; using defaults.", root)

    for env_key, config_key in _ENV_KEY_MAP.items():
        val = os.environ.get(env_key)
        if val:
            if config_key == "ignore_paths":
                config[config_key] = [p.strip() for p in val.split(";") if p.strip()]
            else:
                config[config_key] = val
            logger.debug("Config override from env: %s", env_key)

    if overrides:
        config.update({k: v for k, v in overrides.items() if k in DEFAULTS})

    # Resolve <key> / <key>_add / <key>_remove overrides for configurable
    # default keys.  Must run after all sources are merged so the final
    # _add / _remove values are available.
    _apply_config_overrides(config)

    # Resolve output path — must run after all sources are merged so the
    # final output_dir value is available.
    _resolve_output_spec(config, overrides or {})

    _validate(config)
    return config


# ---------------------------------------------------------------------------
# Override-resolution helpers (0.8.1)
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
        k: list(v)
        for k, v in raw_config.get(key, defaults.get(key, {})).items()
    }

    add = raw_config.get(f"{key}_add") or {}
    if isinstance(add, dict):
        for provider, prefixes in add.items():
            existing = base.setdefault(provider, [])
            seen = set(existing)
            for prefix in (prefixes or []):
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
    # Pre-0.8.1 users may have "supported_extensions": [".py", ".ts"] in their
    # config file to restrict scanning.  After 0.8.1, extension_language_map is
    # the single source of truth, but we honour an *explicit* supported_extensions
    # override by applying it as a filter on the resolved map — so old configs
    # keep working without migration.
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

    if inferred_format == "json":
        config["output_json_filename"] = p.name
    else:
        config["output_md_filename"] = p.name

    logger.info(
        "Output path resolved: dir='%s'  file='%s'  format='%s'",
        parent_str,
        p.name,
        inferred_format,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(config: dict[str, Any]) -> None:
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

    if config["llm_mode"] == "api" and not (
        config.get("api_key") or _has_provider_api_key()
    ):
        logger.warning(
            "llm_mode is 'api' but no API key was found. Set LLM_API_KEY, "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY "
            "in your environment or .env file."
        )

    # Coerce safe_mode to bool (env vars arrive as strings).
    raw_safe = config.get("safe_mode", False)
    if isinstance(raw_safe, str):
        config["safe_mode"] = raw_safe.strip().lower() in ("true", "1", "yes")
    else:
        config["safe_mode"] = bool(raw_safe)

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

    # Validate 0.8.1 rate-limit profile override keys
    _rls = config.get("rate_limit_backoff_s")
    if _rls is not None:
        try:
            _v = float(_rls)
            if _v < 0:
                raise ValueError
            config["rate_limit_backoff_s"] = _v
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "rate_limit_backoff_s must be a non-negative number or null."
            ) from exc

    _rls = config.get("rate_limit_backoff_scale")
    if _rls is not None:
        try:
            _v = float(_rls)
            if _v <= 0:
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
        raise ConfigError(
            "output_format must be one of: 'json', 'md', or 'both'."
        )

    try:
        config["max_file_size_kb"] = int(config["max_file_size_kb"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("max_file_size_kb must be an integer.") from exc

    for key in ("max_parallel_files", "file_retry_attempts", "max_consecutive_failures"):
        try:
            config[key] = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key} must be an integer.") from exc

    if config["max_parallel_files"] < 1:
        raise ConfigError("max_parallel_files must be at least 1.")

    if config["file_retry_attempts"] < 0:
        raise ConfigError("file_retry_attempts must be 0 or greater.")

    if config["max_consecutive_failures"] < 1:
        raise ConfigError("max_consecutive_failures must be at least 1.")

    # Validate and normalise parallel_ladder (0.8.0)
    ladder = config.get("parallel_ladder")
    if ladder is not None:
        if not isinstance(ladder, list) or not all(
            isinstance(x, int) and x > 0 for x in ladder
        ):
            raise ConfigError(
                "parallel_ladder must be a list of positive integers, "
                "e.g. [5, 2, 1]."
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

    # Coerce retry_after_cap_s to int
    try:
        config["retry_after_cap_s"] = int(config.get("retry_after_cap_s", 30))
    except (TypeError, ValueError) as exc:
        raise ConfigError("retry_after_cap_s must be an integer.") from exc


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
