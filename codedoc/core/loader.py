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
    "model_name": "gpt-4o-mini",
    "api_base_url": None,
    "api_key": None,
    "entry_file": None,
    "output_dir": "docs_output",
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
    "log_level": "INFO",
    "max_file_size_kb": 500,
    "propagate_changes": True,
}

_CONFIG_FILENAMES = ["codedoc.config.json", "config.json"]
_ENV_KEY_MAP = {
    "LLM_MODE": "llm_mode",
    "MODEL_NAME": "model_name",
    "API_BASE_URL": "api_base_url",
    "LLM_API_KEY": "api_key",
    "OPENAI_API_KEY": "api_key",
    "ANTHROPIC_API_KEY": "api_key",
    "OUTPUT_DIR": "output_dir",
    "LOG_LEVEL": "log_level",
}


def load_config(root: Path) -> dict[str, Any]:
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
            config[config_key] = val
            logger.debug("Config override from env: %s", env_key)

    _validate(config)
    return config


def _validate(config: dict[str, Any]) -> None:
    """Raise ConfigError for invalid values."""
    if config["llm_mode"] not in ("local", "api"):
        raise ConfigError(f"llm_mode must be 'local' or 'api', got '{config['llm_mode']}'")

    if config["llm_mode"] == "api" and not config.get("api_key"):
        logger.warning(
            "llm_mode is 'api' but no API key was found. Set LLM_API_KEY, "
            "OPENAI_API_KEY, or ANTHROPIC_API_KEY in your environment or .env file."
        )

    if config["llm_mode"] == "local" and not config.get("api_base_url"):
        config["api_base_url"] = "http://localhost:11434/v1"

    if not isinstance(config.get("supported_extensions"), list):
        raise ConfigError("supported_extensions must be a list of file extensions")

    try:
        config["max_file_size_kb"] = int(config["max_file_size_kb"])
    except (TypeError, ValueError) as exc:
        raise ConfigError("max_file_size_kb must be an integer") from exc
