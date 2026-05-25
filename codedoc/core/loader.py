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
    "propagate_changes": True,
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
}


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

    # Resolve output path — must run after all sources are merged so the
    # final output_dir value is available.
    _resolve_output_spec(config, overrides or {})

    _validate(config)
    return config


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

    if not isinstance(config.get("supported_extensions"), list):
        raise ConfigError("supported_extensions must be a list of file extensions.")

    if not isinstance(config.get("skip_dirs"), list):
        raise ConfigError("skip_dirs must be a list of directory names.")

    if not isinstance(config.get("ignore_paths"), list):
        raise ConfigError("ignore_paths must be a list of project-relative paths.")

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
