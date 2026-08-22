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
    MAX_LEAF_EXPORT_ITEM_CHARS,
    MAX_LEAF_EXPORT_ITEMS,
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

#: Vocabulary a reader of README/RUN_FLOW cannot act on. Release numbers date
#: the documents and internal revision names are invisible outside the source
#: tree, so both belong in CHANGELOG.md instead. Kept as one list so the ban is
#: enforced identically everywhere it is applied.
_INTERNAL_DOC_VOCABULARY = (
    "leaf-capsule", "file-reduction-v", "large-file-v", "fresh-only",
    "source-structure-v", "semantic-unit-v", "division-packer",
    "fact-ledger", "reduction-capsule-v", "reduction-packing",
    "file-synthesis-v", "division-execution-v", "file-doc-v",
    "ordinary-path-v", "truncate-v", "MAX_QUARANTINE_ENTRIES_PER_FILE",
    "schema 1", "schema 2", "schema 3", "schema 4",
    "schema-1", "schema-2", "schema-3", "schema-4",
)

#: Matches a bare or `v`-prefixed release number. `\b\d` alone misses
#: `v0.14.6`, because there is no word boundary between `v` and `0`.
_RELEASE_NUMBER = re.compile(r"(?<![\w.])v?\d+\.\d+\.\d+(?![\w.])")


def test_public_docs_carry_no_release_numbers_or_internal_revision_names():
    """README and RUN_FLOW document what CodeDoc offers and how to use it.

    They are not a release history. A reader cannot act on "advances to
    `leaf-capsule-v7`" or on "a fresh `0.14.3` run" -- the first names an
    internal identity revision that exists only inside the source tree, and
    the second dates the document the moment the next release ships. Both
    belong in CHANGELOG.md, which has its own contract test
    (`test_release_documents_name_every_active_split_identity`) requiring the
    full identity roster there.

    The behaviour those sentences used to carry is not dropped; it is restated
    in user terms and asserted by the siblings below."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        text = path.read_text(encoding="utf-8")
        found = _RELEASE_NUMBER.findall(text)
        assert not found, f"{path.name} names release(s) {sorted(set(found))}"
        for token in _INTERNAL_DOC_VOCABULARY:
            assert token not in text, f"{path.name} names internal `{token}`"


def test_public_docs_state_the_incompatible_recovery_remedies_in_user_terms():
    """The fact the retired version paragraphs actually carried: recovery this
    build cannot resume is preserved, never rewritten or silently discarded,
    and the user has exactly two supported remedies. Asserted without naming
    which release wrote the file, since the answer is "whichever one did"."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        assert "cannot resume is recognized and preserved" in normalized, path.name
        assert "the CodeDoc version that wrote the file" in normalized, path.name
        assert "move `crash_recovery.json` aside" in normalized, path.name
        assert "explicit discard" in normalized, path.name
        # Preserve-first is only meaningful if nothing is paid or mutated first.
        assert "before any node is read" in normalized, path.name
        # The retired schema-generation tests also owned this claim: every
        # other rejection stops the run rather than guessing. Scoped to the
        # sentence that actually states it, and asserting all five cases --
        # a whole-document search would pass on an unrelated mention, and
        # checking only three left the other two deletable with the suite
        # green.
        assert "stays fail-closed" in normalized, path.name
        fail_closed = normalized.split("stays fail-closed", 1)[1].split(". ", 1)[0]
        for case in (
            "malformed container",
            "foreign owner",
            "unsupported container version",
            "unplanned or duplicate node ID",
            "set-aside map that exceeds its bound",
        ):
            assert case in fail_closed, f"{path.name} drops {case!r}"
        assert "raise and stop the run" in fail_closed, path.name


def test_public_docs_state_the_upgrade_reprocessing_cost_in_user_terms():
    """The other fact those paragraphs carried: an upgrade that changes the
    internal fragment contract makes earlier split work stale, both unfinished
    and completed, and that costs one extra pass. Stated without naming the
    revision that changed, which the user has no way to look up."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        assert "versions the internal contract each fragment is documented under" in (
            normalized
        ), path.name
        assert "stale" in normalized, path.name
        assert "re-executed" in normalized, path.name
        assert "reprocessed in full" in normalized, path.name
        # Whole-plan invalidation must be documented as recoverable, not fatal.
        assert "instead of aborting the run" in normalized or (
            "rather than aborting the run" in normalized
        ), path.name
        assert "Rolling back to an older version" in normalized, path.name

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
        "CODEDOC_TRUST_API_BASE_URL",
    }
    assert not {name for name in env_vars if name not in readme}


def test_generated_target_exclusion_is_not_documented_as_directory_wide():
    """Section 5.7: only exact generated targets are excluded automatically.

    Public documentation must not broaden that production contract into a
    claim that the whole output directory is excluded, because supported
    source files can intentionally be co-located there after removing an
    ordinary skip-dir.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "--remove-skip-dir codedoc --output docs_output" in readme
    assert (
        "exact generated JSON, Markdown, and recovery target files are excluded"
        in normalized_readme
    )
    assert "output directory itself is not automatically excluded" in normalized_readme
    assert "co-located there remain scannable" in normalized_readme

    directory_wide_claim = re.compile(
        r"\boutput directory is (?:always )?excluded\b", re.IGNORECASE
    )
    for public_path in (ROOT / "README.md", ROOT / "CHANGELOG.md"):
        public_text = " ".join(public_path.read_text(encoding="utf-8").split())
        assert directory_wide_claim.search(public_text) is None

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
    assert "exactly compatible same-path completed split record" in normalized_large_files
    assert "Cross-path identical-content split reuse remains unavailable" in normalized_large_files
    assert "cannot resume is recognized and preserved" in normalized_large_files
    assert "never resumed, rewritten, or silently discarded" in normalized_large_files
    assert "Imports-only changes" in normalized_large_files
    assert "Provider, model, or effective-endpoint changes invalidate partial nodes" in normalized_large_files
    assert "completed cache reuse remains provider-agnostic" in normalized_large_files
    assert "Files at or below `max_content_chars`" in normalized_large_files
    assert "up to 32 functions and up to 32 classes" in normalized_large_files
    assert "reduction narrative is capped at 300 characters" in normalized_large_files
    assert (
        "reducer prompt states that bound explicitly" in normalized_large_files
    )
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
    assert "compatible split container is validated in plan order" in normalized_recovery
    assert "bounded, non-executable set-aside map" in normalized_recovery
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
    assert "cannot resume is recognized and preserved" in normalized_run_flow
    assert "preserved and blocked" in normalized_run_flow
    assert "up to 32 functions and up to 32 classes" in normalized_run_flow
    assert "reduction narrative is capped at 300 characters" in normalized_run_flow
    assert "reducer prompt states that bound explicitly" in normalized_run_flow


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


def test_readme_and_run_flow_declare_ordinary_reuse_same_path_only():
    """0.14.4: ordinary identical-content reuse (`files_reused_identical_content`)
    is same-path only -- CodeDoc never copies documentation across paths, even
    for byte-identical content. README must no longer claim reuse "from
    another path"; RUN_FLOW's existing cross-path *split* sentence is a
    separate, unrelated claim and must stay exactly as it is."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    run_flow = (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8")

    assert "another path with identical content" not in readme

    readme_paragraph = _locate_paragraph(
        readme, "Ordinary identical-content reuse (`files_reused_identical_content`)"
    )
    normalized_readme_paragraph = " ".join(readme_paragraph.split())
    assert "same-path only" in normalized_readme_paragraph
    assert "never copies documentation from one path" in normalized_readme_paragraph

    run_flow_paragraph = _locate_paragraph(
        run_flow, "Ordinary identical-content reuse is same-path only"
    )
    normalized_run_flow_paragraph = " ".join(run_flow_paragraph.split())
    assert "same-path only" in normalized_run_flow_paragraph
    assert "never copies documentation from one path" in normalized_run_flow_paragraph
    assert "_ordinary_path_identity" in normalized_run_flow_paragraph

    # The unrelated split cross-path sentence is untouched by this correction.
    assert "Cross-path split reuse remains unavailable." in run_flow


def test_readme_and_run_flow_state_reason_codes_and_endpoint_authorization():
    """0.14.4: a final response-contract failure names its closed reason
    code, and a custom api_base_url requires runtime endpoint-trust approval
    that configuration can never satisfy -- both facts must be publicly
    documented, not just true internally."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    run_flow = (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8")

    response_correction = readme.split("### Response correction", 1)[1].split(
        "## Command-line options", 1
    )[0]
    normalized_response_correction = " ".join(response_correction.split())
    assert "names its closed reason code" in normalized_response_correction
    assert "no source text, prompt text, raw or truncated" in normalized_response_correction

    normalized_run_flow = " ".join(run_flow.split())
    assert "names its closed reason code" in normalized_run_flow
    assert "runtime endpoint-trust approval bound to" in normalized_run_flow
    assert "can never satisfy this gate" in normalized_run_flow
    assert "resolved before any credential is read" in normalized_run_flow


def test_security_md_recovery_sensitivity_wording_is_present_and_unchanged():
    """Gate 17.13, assertion 4: the inherited `0.14.2` SECURITY.md wording
    about `crash_recovery.json` sensitivity is present and untouched --
    `0.14.3` declares no new security contract and must not become a second
    owner of this text (preserved, not edited, by section 4A)."""
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "crash_recovery.json" in security
    assert "sensitive derived project information" in security
    assert "Current verbose mode emits bounded CodeDoc diagnostics" in security


_EXPORT_SECTION_HEADING = "Module exports in a split file"


def _export_section(text: str) -> str:
    """Return the whitespace-normalized body of the shared export section.

    Sliced by heading rather than by paragraph: the section is deliberately
    structured (capability, then usage) and spans several blocks, so a
    single-paragraph locator would silently check only the first one.
    """
    start = text.index(_EXPORT_SECTION_HEADING) + len(_EXPORT_SECTION_HEADING)
    rest = text[start:]
    end = rest.index("\n#")
    return " ".join(rest[:end].split())


def test_readme_and_run_flow_document_split_export_behavior_not_release_history():
    """README and RUN_FLOW describe what CodeDoc offers and how to use it.

    Behavior is documented as a named capability section, not as a
    `0.14.x advances ...` release narrative -- version numbering and internal
    revision names belong in CHANGELOG.md, which has its own contract test
    (`test_release_documents_name_every_active_split_identity`). The release
    archaeology these documents used to carry was removed, not preserved; the
    facts it carried are asserted version-free by the siblings above.

    Both documents must carry the same section, so the two cannot drift into
    describing different products."""
    assert MAX_LEAF_EXPORT_ITEMS == 32
    assert MAX_LEAF_EXPORT_ITEM_CHARS == 256

    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        text = path.read_text(encoding="utf-8")
        assert _EXPORT_SECTION_HEADING in text, path.name
        section = _export_section(text)

        # Structured as capability, then usage -- not one undifferentiated wall.
        assert "**What CodeDoc reports.**" in section, path.name
        assert "**How you use it.**" in section, path.name

        # What is available: the rule itself.
        assert "declaration or re-export" in section
        assert "declaration visibility decides this" in section
        assert (
            "not an export merely because the value containing it is exported"
            in section
        )
        # The carve-out. Stating the exclusion absolutely would misdocument
        # `__all__`, `module.exports`, and brace-enclosed export lists as
        # non-exports, contradicting the shipped prompt. The unqualified
        # sentence is forbidden outright, so a future reword cannot satisfy
        # this test by adding the carve-out while leaving the wrong claim
        # standing beside it.
        assert "exported-names manifest" in section
        assert "the module's export table" in section
        assert "those entries are the exported names" in section
        assert "are data, not exports" not in section
        # Bounds, and that they are enforced losslessly.
        assert "at most 32 export names" in section
        assert "256 characters" in section
        assert "never silently" in section
        # The correction route carries the same definition.
        assert "the same definition is sent with the optional single repair call" in (
            section
        )

        # How you use it: rerun-to-resume, and the one-off reprocessing cost.
        assert "rerun the identical command" in section
        assert "resume" in section
        assert "redoes each large file's split work once" in section
        assert "whether that file was finished or still in progress" in section
        assert "--dry-run" in section
        assert "response_correction_enabled" in section

        # House convention: no release numbering in these two documents' own
        # description of current behavior.
        assert "0.14." not in section, f"{path.name} export section names a release"
        assert "leaf-capsule" not in section, (
            f"{path.name} export section names an internal revision"
        )


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
