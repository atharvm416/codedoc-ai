"""Tests organized by feature ownership."""

from __future__ import annotations

import re
from pathlib import Path
from codedoc import __version__
from codedoc.cli.cli import build_parser
from codedoc.core.loader import _ENV_KEY_MAP

ROOT = Path(__file__).resolve().parents[3]

def test_public_docs_and_help_use_canonical_command_spelling():
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        assert "codedoc run" not in path.read_text(encoding="utf-8")
    assert "codedoc run" not in build_parser().format_help()
    template = ROOT / "codedoc" / "templates" / "github-actions-codedoc.yml"
    assert "\n            run\n" not in template.read_text(encoding="utf-8")

def test_release_literals_and_narrative_are_allowlisted():
    version_pattern = re.compile(r"\b\d+\.\d+\.\d+\b")
    phrase_pattern = re.compile(
        r"this release|current release|removed in 0\.\d+|not accepted in 0\.\d+",
        re.IGNORECASE,
    )
    paths = [ROOT / "README.md", ROOT / "RUN_FLOW.md"]
    paths.extend((ROOT / "codedoc").rglob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path == ROOT / "codedoc" / "__init__.py":
            text = text.replace(__version__, "")
        assert not version_pattern.search(text), path
        assert not phrase_pattern.search(text), path

def test_readme_documents_every_long_flag_and_environment_variable():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    assert not {option for option in options if option not in readme}
    env_vars = set(_ENV_KEY_MAP) | {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_API_KEY",
    }
    assert not {name for name in env_vars if name not in readme}

def test_entry_help_matches_optional_resolution_and_all_files_fallback():
    parser = build_parser()
    assert parser.parse_args([]).entry is None
    entry_action = next(
        action for action in parser._actions if "--entry" in action.option_strings
    )
    help_text = entry_action.help.lower()
    assert "optional entry" in help_text
    assert "exact selected output may supply it" in help_text
    assert "configured candidates are auto-detected" in help_text
    assert "all scanned files are documented" in help_text
