"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import re
from pathlib import Path

import codedoc
import codedoc.core
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


def _locate_paragraph(text: str, anchor: str) -> str:
    """Return the single Markdown paragraph starting with *anchor*.

    A paragraph is bounded by a blank line on either side. Gate 17.13's
    contextual assertions (section 20A item 5) must prove that a set of
    facts appears *together*, in one located paragraph or tightly bounded
    section -- not merely that each fact appears somewhere in the whole
    document, which a scattered, unrelated mention would also satisfy.
    """
    start = text.index(anchor)
    end = text.find("\n\n", start)
    return text[start : end if end != -1 else len(text)]


def test_public_docs_and_help_use_canonical_command_spelling():
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        assert "codedoc run" not in path.read_text(encoding="utf-8")
    assert "codedoc run" not in build_parser().format_help()
    template = ROOT / "codedoc" / "templates" / "github-actions-codedoc.yml"
    assert "\n            run\n" not in template.read_text(encoding="utf-8")

def test_release_docs_frame_current_and_predecessor_versions_explicitly():
    """Gate 17.13, assertion 3: `0.14.0`, `0.14.1`, and `0.14.2` references
    must be framed as historical or downgrade guidance, not merely present
    anywhere in the document. Located via the shared "predecessor ...
    fresh-only-v1" paragraph opening both files' downgrade guidance, rather
    than a whole-document substring search that a same-page but unrelated
    mention of the same version number would also satisfy."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    run_flow = (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8")
    readme_paragraph = _locate_paragraph(
        readme, "The `0.14.1` `fresh-only-v1` completed split contract is stale"
    )
    run_flow_paragraph = _locate_paragraph(
        run_flow, "The predecessor `0.14.1` `fresh-only-v1` split record is stale"
    )
    for paragraph in (readme_paragraph, run_flow_paragraph):
        assert "0.14.2" in paragraph
        assert "0.14.1" in paragraph
        assert "fresh-only-v1" in paragraph
        assert "stale" in paragraph
        assert "Rolling back to `0.14.1`" in paragraph

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


def test_public_docs_explain_split_reuse_recovery_and_optional_structure_extra():
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
    assert "Completed split reuse and node-level partial recovery begin in `0.14.2`" in normalized_large_files
    assert "exactly compatible same-path completed split record" in normalized_large_files
    assert "Cross-path identical-content split reuse remains unavailable" in normalized_large_files
    assert "Schema-1 and schema-2 partial state" in normalized_large_files
    assert "preserved but not resumed" in normalized_large_files
    assert "Imports-only changes" in normalized_large_files
    assert "Provider, model, or effective-endpoint changes invalidate partial nodes" in normalized_large_files
    assert "completed cache reuse remains provider-agnostic" in normalized_large_files
    assert "Files at or below `max_content_chars`" in normalized_large_files
    active_checks = large_files.split(
        "#### Split accounting, identity, and provider checks", 1
    )[1].split("#### Completed split reuse and node recovery", 1)[0]
    normalized_active_checks = " ".join(active_checks.split())
    assert "P = R + O + (C - Hc) + (U - Hu) + (G - Hg) + (F - Hf)" in normalized_active_checks
    assert "private `_large_file_identity`" in normalized_active_checks
    assert "machine-readable JSON" in normalized_active_checks
    assert "Provider construction must attest" in normalized_active_checks
    assert "malformed HTTP(S) URL" in normalized_active_checks
    current_recovery = large_files.split(
        "#### Completed split reuse and node recovery", 1
    )[1]
    assert "Recovery is dependency-closed" in current_recovery
    normalized_recovery = " ".join(readme.split("## Crash recovery", 1)[1].split())
    assert "current schema-4 split container is validated in plan order" in normalized_recovery
    assert "bounded non-executable quarantine" in normalized_recovery
    assert "deletion is an explicit choice to discard" in normalized_recovery
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
    assert "same-path completed reuse" in normalized_run_flow
    assert "dependency-valid node recovery" in normalized_run_flow
    assert "Cross-path identical-content split reuse remains unavailable" in normalized_run_flow
    assert "Schema-1 and schema-2 partials are preserved but not resumed" in normalized_run_flow
    assert "preserved and blocked" in normalized_run_flow


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
    # Locate the newest release section structurally -- it is whatever comes
    # first after the "# Changelog" title -- rather than matching one
    # version's exact heading text, so this test survives every version bump
    # without needing to rewrite an earlier release's historical heading
    # (e.g. the dateless-separator "## 0.14.2 2026-08-04").
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_sections = changelog.split("\n## ")[1:]
    assert release_sections, "CHANGELOG.md has no '## <version>' release heading."
    current_sections = (release_sections[0],)
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
        "large-file-v3",
        ANALYSIS_REVISION,
    )

    for section in current_sections:
        assert all(f"`{revision}`" in section for revision in active)


_DISQUALIFYING_SUPPORT_QUALIFIERS = (
    "preview", "fresh-only", "dormant", "recovery-preview", "candidate", "beta",
)


def _sentences_containing(text: str, needle: str) -> list[str]:
    """Split *text* into rough sentences and return those containing *needle*."""
    return [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if needle in sentence
    ]


def test_readme_and_run_flow_declare_split_fully_supported_without_qualifiers():
    """Gate 17.13, assertions 1-2: README and RUN_FLOW must state that
    `single + split` execution, completed-record reuse, and node-level
    recovery are fully supported, and the sentence making that claim must
    carry none of the disqualifying preview/fresh-only/dormant/
    recovery-preview/candidate/beta qualifiers. Checked sentence-by-sentence
    so an unrelated historical sentence elsewhere in the document (e.g.
    describing what `0.14.1` used to do) cannot fail this test."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        text = path.read_text(encoding="utf-8")
        support_sentences = _sentences_containing(text, "fully supported")
        assert support_sentences, f"{path.name} makes no 'fully supported' claim."
        assert any(
            "single + split" in sentence
            and "completed-record reuse" in sentence
            and "node-level" in sentence
            and "recovery" in sentence
            for sentence in support_sentences
        ), f"{path.name}'s support claim omits execution/reuse/recovery."
        for sentence in support_sentences:
            lowered = sentence.lower()
            assert not any(
                qualifier in lowered for qualifier in _DISQUALIFYING_SUPPORT_QUALIFIERS
            ), f"{path.name} support claim carries a disqualifying qualifier: {sentence!r}"


def test_security_md_recovery_sensitivity_wording_is_present_and_unchanged():
    """Gate 17.13, assertion 4: the inherited `0.14.2` SECURITY.md wording
    about `crash_recovery.json` sensitivity is present and untouched --
    `0.14.3` declares no new security contract and must not become a second
    owner of this text (preserved, not edited, by section 4A)."""
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "crash_recovery.json" in security
    assert "sensitive derived project information" in security
    assert "Current verbose mode emits bounded CodeDoc diagnostics" in security


def test_readme_and_run_flow_name_schema4_current_and_schema3_predecessor():
    """Gate 17.13, assertion 5: README and RUN_FLOW must name schema 4 as
    the current partial generation and schema 3 as unsupported predecessor
    recovery, state complete v6 re-execution, and explain both supported
    remedies for an unfinished `0.14.2` split run -- all inside the one
    located paragraph that actually states the schema boundary, not
    scattered anywhere in the document. Whole-document token presence does
    not prove these facts belong together; a "schema 4" mention in one
    unrelated sentence and a "schema 3" mention in another would satisfy a
    bare substring search without ever stating the boundary between them."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        text = path.read_text(encoding="utf-8")
        paragraph = _locate_paragraph(text, "The current node-keyed")
        assert "schema 4" in paragraph
        assert "schema 3" in paragraph
        assert "unsupported predecessor" in paragraph
        assert "leaf-capsule-v6" in paragraph
        assert "v6 re-execution" in paragraph
        assert "finish" in paragraph.lower() and "0.14.2" in paragraph
        assert "move" in paragraph.lower() and "crash_recovery.json" in paragraph
        assert "deliberate discard" in paragraph or "explicit discard" in paragraph


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
