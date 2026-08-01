"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import re
from pathlib import Path

import codedoc
import codedoc.core
from codedoc import __version__
from codedoc.cli.cli import build_parser
from codedoc.core.config_template import PUBLIC_CONFIG_KEYS
from codedoc.core.file_division import (
    EXECUTION_IDENTITY_SCHEMA_REVISION,
    FINAL_SYNTHESIS_REVISION,
    LEAF_CAPSULE_SCHEMA_REVISION,
    LEDGER_SCHEMA_REVISION,
    PACKER_SCHEMA_REVISION,
    REDUCER_PROMPT_REVISION,
    REDUCTION_CAPSULE_SCHEMA_REVISION,
    REDUCTION_PACKING_REVISION,
    STRUCTURE_SCHEMA_REVISION,
    UNIT_SCHEMA_REVISION,
)
from codedoc.core.loader import _ENV_KEY_MAP
from codedoc.core.record_meta import ANALYSIS_REVISION

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


def test_public_docs_explain_fresh_split_execution_and_optional_structure_extra():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    run_flow = (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8")
    large_files = readme.split("### Large files", 1)[1].split(
        "### Response correction", 1
    )[0]
    normalized_large_files = " ".join(large_files.split())

    assert "--large-file-strategy split" in readme
    assert readme.count('pip install "codedoc-ai[structure]"') == 1
    assert 'pip install "codedoc-ai[structure]"' in large_files
    assert "This extra is optional." in normalized_large_files
    assert "does not download grammars" in normalized_large_files
    assert "create a grammar cache" in normalized_large_files
    assert "4,096 planned lexical atoms" in normalized_large_files
    assert "`atom-cap` before any provider call" in normalized_large_files
    assert "optional package" in normalized_large_files
    assert "fresh split execution" in normalized_large_files
    assert "again on every real run" in normalized_large_files
    assert "completed split reuse" in normalized_large_files
    assert "partial split recovery are not active" in normalized_large_files
    assert "Files at or below `max_content_chars`" in normalized_large_files
    assert "held only in memory" in normalized_large_files
    assert "repeats the failed logical call" in normalized_large_files
    assert "never written to recovery" in normalized_large_files
    active_checks = large_files.split(
        "#### Active split accounting, identity, and provider checks", 1
    )[1].split("#### Later completed split reuse and node recovery", 1)[0]
    normalized_active_checks = " ".join(active_checks.split())
    assert "P = R + O + C + U + G + F" in normalized_active_checks
    assert "private `_large_file_identity`" in normalized_active_checks
    assert "machine-readable JSON" in normalized_active_checks
    assert "Provider construction must attest" in normalized_active_checks
    assert "malformed HTTP(S) URL" in normalized_active_checks
    future_recovery = large_files.split(
        "#### Later completed split reuse and node recovery", 1
    )[1]
    assert "P = R + O + (C - Hc)" in future_recovery
    assert "Recovery is dependency-closed" in future_recovery
    normalized_recovery = " ".join(readme.split("## Crash recovery", 1)[1].split())
    assert "whether ordinary or fresh split" in normalized_recovery
    assert "Fresh split node progress remains process-local" in normalized_recovery
    assert "completed ordinary-file records are resumed" in normalized_recovery
    assert "completed fresh-split records are deliberately rerun" in normalized_recovery
    assert "default `large_file_strategy: truncate`" in run_flow
    assert "resolves to `split`" in run_flow

    # RUN_FLOW is a named public surface in the documentation contract, so it
    # must carry the same lexical-atom ceiling, offline guarantee, and
    # conditional structure-extra remedy as README rather than only the two
    # generic split phrases above.
    normalized_run_flow = " ".join(run_flow.split())
    assert "runtime-offline" in normalized_run_flow
    assert "never downloads a grammar" in normalized_run_flow
    assert "writes a grammar cache" in normalized_run_flow
    assert "4,096 lexical-atom ceiling" in normalized_run_flow
    assert "reports `atom-cap`" in normalized_run_flow
    # The remedy must stay reason-specific: a higher ceiling cannot clear a
    # line-counted atom cap, so the extra is what applies to that reason.
    assert "raising `max_content_chars` cannot clear it" in normalized_run_flow
    assert 'pip install "codedoc-ai[structure]"' in normalized_run_flow
    assert "fails during configuration validation before scanning" in normalized_run_flow
    assert "always executes its full leaf" in normalized_run_flow
    assert "does not reuse same-path or identical-content" in normalized_run_flow
    assert "preserved and rejected" in normalized_run_flow
    assert "process-local memory" in normalized_run_flow
    assert "only the failed logical call is repeated" in normalized_run_flow


def test_model_help_scopes_provider_auto_detection_to_auto():
    """`--model` must not claim auto-detection when --provider is explicit."""
    model_help = next(
        action.help
        for action in build_parser()._actions
        if "--model" in action.option_strings
    )
    normalized = " ".join(model_help.split())
    assert "only when --provider is auto" in normalized
    assert "explicit --provider always wins" in normalized


def test_release_documents_name_every_active_split_identity():
    # The current release's own changelog section is the release manifest for
    # this contract.  Internal planning documents are deliberately not part of
    # the repository and must never be a source of current identity truth.
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    current_sections = (
        changelog.split("## 0.14.1", 1)[1].split("## 0.14.0", 1)[0],
    )
    active = (
        STRUCTURE_SCHEMA_REVISION,
        UNIT_SCHEMA_REVISION,
        PACKER_SCHEMA_REVISION,
        LEAF_CAPSULE_SCHEMA_REVISION,
        LEDGER_SCHEMA_REVISION,
        REDUCTION_CAPSULE_SCHEMA_REVISION,
        REDUCTION_PACKING_REVISION,
        REDUCER_PROMPT_REVISION,
        FINAL_SYNTHESIS_REVISION,
        EXECUTION_IDENTITY_SCHEMA_REVISION,
        "large-file-v2",
        ANALYSIS_REVISION,
    )

    for section in current_sections:
        assert all(f"`{revision}`" in section for revision in active)


def test_readme_configuration_reference_matches_generated_public_keys():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Configuration reference", 1)[1].split(
        "### Large files", 1
    )[0]
    documented = re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE)
    assert documented == [name for name, _description in PUBLIC_CONFIG_KEYS]


def test_readme_export_table_matches_intentional_python_exports():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Exported Python surface", 1)[1].split(
        "## Troubleshooting", 1
    )[0]
    documented = re.findall(r"^\| `([^`]+)` \|", section, flags=re.MULTILINE)
    expected = [
        *(f"codedoc.{name}" for name in codedoc.__all__),
        *(f"codedoc.core.{name}" for name in codedoc.core.__all__),
    ]
    assert documented == expected


def test_readme_json_examples_are_valid_json():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"```json\n(.*?)\n```", readme, flags=re.DOTALL)
    assert examples
    for example in examples:
        json.loads(example)
