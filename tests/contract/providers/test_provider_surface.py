"""Tests organized by feature ownership."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

from codedoc.utils.errors import ConfigError

_PROVIDER_CHOICES = ("auto", "openai", "anthropic", "gemini")


def test_cli_provider_choices_are_exactly_the_supported_set():
    from codedoc.cli.cli import build_parser

    parser = build_parser()
    action = parser._option_string_actions["--provider"]
    assert tuple(action.choices) == _PROVIDER_CHOICES


def test_config_llm_provider_validation_accepts_exactly_the_supported_set(tmp_path):
    from codedoc.core.loader import load_config

    for provider in _PROVIDER_CHOICES:
        load_config(tmp_path, {"llm_provider": provider, "dry_run": True})

    with pytest.raises(ConfigError):
        load_config(tmp_path, {"llm_provider": "local", "dry_run": True})
    with pytest.raises(ConfigError):
        load_config(tmp_path, {"llm_provider": "", "dry_run": True})


def test_factory_resolution_never_returns_local():
    from codedoc.core.loader import DEFAULTS
    from codedoc.llm.factory import _resolve_api_provider

    prefixes = DEFAULTS["provider_prefixes"]
    for model in ("local-model", "llama3", "qwen2.5-coder:7b", ""):
        assert _resolve_api_provider("auto", model.lower(), prefixes) in (
            "openai",
            "anthropic",
            "gemini",
        )

    with pytest.raises(ConfigError):
        _resolve_api_provider("local", "any-model", prefixes)


def test_missing_or_invalid_provider_raises_config_error():
    from codedoc.llm.factory import create_provider

    with pytest.raises(ConfigError):
        create_provider(
            {"llm_mode": "api", "llm_provider": "local", "model_name": "x", "api_key": "k"}
        )

    with pytest.raises(ConfigError):
        create_provider(
            {"llm_mode": "api", "llm_provider": "auto", "model_name": "x", "api_key": ""}
        )


def test_openai_compatible_base_url_reaches_openai_provider(monkeypatch):
    from codedoc.llm.factory import create_provider

    captured = {}

    class FakeOpenAIProvider:
        def __init__(self, api_key, model, base_url=None):
            captured.update({"api_key": api_key, "model": model, "base_url": base_url})

    monkeypatch.setattr("codedoc.llm.api_provider.OpenAIProvider", FakeOpenAIProvider)

    provider = create_provider(
        {
            "llm_mode": "api",
            "llm_provider": "openai",
            "model_name": "local-model",
            "api_key": "sk-local",
            "api_base_url": "http://localhost:11434/v1",
        }
    )

    assert isinstance(provider, FakeOpenAIProvider)
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["model"] == "local-model"


def test_local_provider_module_absent_from_installed_package_tree():
    """Deterministic package-tree inspection, not an arbitrary environment probe."""
    import codedoc.llm as llm_package

    package_dir = Path(llm_package.__file__).resolve().parent
    assert not (package_dir / "local_provider.py").exists()

    module_names = {name for _, name, _ in pkgutil.iter_modules([str(package_dir)])}
    assert "local_provider" not in module_names

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("codedoc.llm.local_provider")
