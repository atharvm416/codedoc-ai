"""Tests organized by feature ownership."""

from __future__ import annotations

import ast
import io
import json
import re
import tokenize
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

#: Matches a bare or `v`-prefixed release number. The right boundary permits
#: sentence punctuation but refuses a partial match inside a longer numeric
#: component sequence.
_RELEASE_NUMBER = re.compile(r"(?<![\w.])v?\d+\.\d+\.\d+(?!\w|\.\d)")

#: Temporal wording that turns an evergreen guide or implementation comment
#: into release history even when it does not spell out a release number.
_RELEASE_HISTORY_WORDING = re.compile(
    r"\b(?:before\s+this|this|current|previous|older)\s+release\b|"
    r"\brelease-specific\b",
    re.IGNORECASE,
)

#: Default-off language is incorrect for response correction. Keep this narrow
#: so accurate opt-in/default-off descriptions of unrelated features remain
#: valid implementation commentary.
_STALE_CORRECTION_DEFAULT_WORDING = re.compile(
    r"(?:\b(?:response\s+)?correction\b.{0,160}"
    r"\b(?:opt[- ]in|disabled\s+by\s+default)\b)|"
    r"(?:\b(?:opt[- ]in|disabled\s+by\s+default)\b.{0,160}"
    r"\b(?:response\s+)?correction\b)",
    re.IGNORECASE | re.DOTALL,
)

#: Market-superlative and product-category claim *shapes* README/RUN_FLOW must
#: never make about CodeDoc. Every alternative is a phrase, not a token: the
#: bare words are legitimate in their ordinary senses and stay allowed --
#: "cleaned up best-effort", a "Best use" table column, "AI assistants" /
#: "coding assistants" as the audience, "triple-agent" / "three-agent"
#: analysis, `parallel_agents`, "package indexes", "indexed by", "semantic
#: boundaries" / "semantic review" / "semantic units". What is banned is a
#: superlative assertion about the product ("the best ...", "best-in-class",
#: "world-class", "industry-leading", "state-of-the-art", ...) or a claim that
#: CodeDoc is an "autonomous [coding] agent", a "semantic index", or an
#: "interactive assistant" -- categories CodeDoc is documented as *not* being.
_UNSUPPORTED_CLAIM = re.compile(
    r"""
      \bthe\s+best\b
    | \b(?:world|industry|market|best)[\s-]+(?:class|leading)\b
    | \bbest[\s-]+in[\s-]+class\b
    | \bworld['’]s\s+(?:best|leading|fastest)\b
    | \bbest[\s-]+of[\s-]+breed\b
    | \bstate[\s-]of[\s-]the[\s-]art\b
    | \bcutting[\s-]edge\b
    | \bsecond\s+to\s+none\b
    | \bunrivall?ed\b
    | \bautonomous(?:ly)?\b
    | \bsemantic\s+index\b
    | \binteractive\s+assistant\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def test_public_docs_carry_no_release_numbers_or_internal_revision_names():
    """README and RUN_FLOW document what CodeDoc offers and how to use it.

    They are not a release history. A reader cannot act on an internal
    identity-revision advance or on a fresh package-version run -- the first
    names an implementation detail that exists only inside the source tree,
    and the second dates the document the moment the next release ships. Both
    belong in CHANGELOG.md, which has its own contract test
    (`test_release_documents_name_every_active_split_identity`) requiring the
    full identity roster there.

    The behaviour those sentences used to carry is not dropped; it is restated
    in user terms and asserted by the siblings below."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        text = path.read_text(encoding="utf-8")
        found = _RELEASE_NUMBER.findall(text)
        assert not found, f"{path.name} names release(s) {sorted(set(found))}"
        historical = _RELEASE_HISTORY_WORDING.findall(text)
        assert not historical, (
            f"{path.name} contains release-history wording {sorted(set(historical))}"
        )
        for token in _INTERNAL_DOC_VOCABULARY:
            assert token not in text, f"{path.name} names internal `{token}`"


def test_release_number_guard_handles_sentence_punctuation_without_partial_matches():
    for text in (
        "Current release: 0.14.7.",
        "Current release: v0.14.7.",
        "Current release: 0.14.7",
    ):
        assert _RELEASE_NUMBER.search(text), text
    assert not _RELEASE_NUMBER.search("Compatibility schema: 0.14.7.4")


def _comment_and_docstring_records(path: Path):
    """Yield source line and text for comment blocks and true docstrings."""
    with tokenize.open(path) as source:
        text = source.read()
    source_lines = text.splitlines()
    comment_records = []
    block_line = None
    block_end = None
    block_parts = []

    def flush_comment_block():
        nonlocal block_line, block_end, block_parts
        if block_line is not None:
            comment_records.append((block_line, "\n".join(block_parts)))
        block_line = None
        block_end = None
        block_parts = []

    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            line, column = token.start
            is_full_line = not source_lines[line - 1][:column].strip()
            if is_full_line and block_end is not None and line == block_end + 1:
                block_parts.append(token.string)
                block_end = line
            elif is_full_line:
                flush_comment_block()
                block_line = line
                block_end = line
                block_parts = [token.string]
            else:
                flush_comment_block()
                comment_records.append((line, token.string))
    flush_comment_block()
    yield from comment_records

    tree = ast.parse(text, filename=str(path))
    docstring_owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, docstring_owners):
            docstring = ast.get_docstring(node, clean=False)
            if docstring is not None:
                yield node.body[0].lineno, docstring


def test_production_comments_and_docstrings_are_release_agnostic():
    """Implementation explanations stay behavioral instead of dating changes."""
    violations = []
    for path in sorted((ROOT / "codedoc").rglob("*.py")):
        for line, text in _comment_and_docstring_records(path):
            normalized = " ".join(text.split())
            releases = _RELEASE_NUMBER.findall(normalized)
            history = _RELEASE_HISTORY_WORDING.findall(normalized)
            if releases or history:
                violations.append(
                    f"{path.relative_to(ROOT)}:{line}: releases={releases!r}, "
                    f"history={history!r}"
                )
    assert not violations, "Release-specific production commentary:\n" + "\n".join(
        violations
    )


def test_production_response_correction_commentary_matches_default_on_policy():
    """Production explanations cannot describe correction as default-off."""
    assert _STALE_CORRECTION_DEFAULT_WORDING.search("Correction is opt-in.")
    assert _STALE_CORRECTION_DEFAULT_WORDING.search(
        "Response correction is disabled by default."
    )
    assert _STALE_CORRECTION_DEFAULT_WORDING.search(
        "Response correction is\ndisabled by default."
    )

    violations = []
    for path in sorted((ROOT / "codedoc").rglob("*.py")):
        for line, text in _comment_and_docstring_records(path):
            normalized = " ".join(text.split())
            if _STALE_CORRECTION_DEFAULT_WORDING.search(normalized):
                violations.append(f"{path.relative_to(ROOT)}:{line}: {normalized}")

    assert not violations, "Stale correction-default commentary:\n" + "\n".join(
        violations
    )


def test_public_docs_make_no_unsupported_superlative_or_product_category_claims():
    """README and RUN_FLOW position CodeDoc as an incremental, reusable-state
    documentation engine, described only through current, supported behaviour.

    They therefore may not reach for an unsupported market superlative -- "the
    best ...", "best-in-class", "world-class", "industry-leading",
    "state-of-the-art" -- nor claim CodeDoc is a product category it is not: an
    "autonomous [coding] agent", a "semantic index", or an "interactive
    assistant". CodeDoc plans deterministically, sends only remaining work, and
    stops; it does not act on its own initiative, it does not build or serve a
    searchable index, and it holds no conversation. Such a claim would mislead a
    reader about what they are getting, and it would date the moment it proved
    untrue.

    The ban is on the claim *shape*, not the words: "cleaned up best-effort", a
    "Best use" table column, "AI assistants" / "coding assistants" as the
    audience for the output, "triple-agent" / "three-agent" analysis,
    `parallel_agents`, "package indexes", "indexed by", and "semantic
    boundaries" / "semantic review" / "semantic units" are all current, correct
    usage and stay allowed."""
    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        # Whitespace-normalized so a claim wrapped across a line break is still
        # one contiguous phrase, matching the sibling recovery-remedy test.
        normalized = " ".join(path.read_text(encoding="utf-8").split())
        hits = sorted({m.group(0).strip() for m in _UNSUPPORTED_CLAIM.finditer(normalized)})
        assert not hits, (
            f"{path.name} makes an unsupported market-superlative or "
            f"product-category claim: {hits}"
        )


def test_readme_and_run_flow_position_codedoc_as_an_incremental_reusable_state_engine():
    """The load-bearing claim of the product: CodeDoc treats documentation as
    durable, reusable state and pays only for the remainder, rather than
    regenerating every record from scratch on each run. A reader who loses that
    framing cannot tell what CodeDoc is *for* -- it reads as just another
    one-shot generator -- so it must be stated where a reader meets it, not left
    implicit. Section 11 mutation check 28 requires that removing or gutting it
    fail this contract; section 13's handoff requires confirming both public
    documents present CodeDoc this way using only current behaviour.

    README/RUN_FLOW asymmetry (deliberate, asserted here so the next auditor
    reads the reasoning rather than re-deriving it): README is the product
    document and states the positioning outright in its lede. RUN_FLOW's own
    opening line scopes it to "the active phase ordering" -- it is a lifecycle
    reference, not a pitch, and it never makes the positioning *statement*.
    What it must instead show is that reuse and recovery are a first-class,
    centrally-governed phase of every run, not an afterthought: a dedicated
    "Split planning, reuse, and recovery" phase, a single "Cache and recovery
    identity" predicate, zero-call reuse of a compatible record, and a resume
    (not a restart) of an interrupted one. The pair collectively carries the
    requirement; demanding the identical marketing vocabulary in a phase-order
    document would be the substring-ban mistake in a new form. If a future
    reviewer concludes RUN_FLOW must itself carry the positioning wording, that
    is a documentation change for them and the independent auditor to make --
    not something this contract should force by lowering the bar to a bare
    grep."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    run_flow = (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8")

    # --- README: the lede states the positioning outright. Located by anchor
    # and asserted as one contiguous clause inside that one paragraph, so the
    # word "incremental" scattered elsewhere in README (the Highlights list,
    # the reuse section, ...) cannot satisfy this if the lede is gutted.
    lede_anchor = "> **codedoc-ai is an incremental documentation engine"
    assert lede_anchor in readme, "README.md lost its positioning lede"
    # The lede is a Markdown blockquote; drop the "> " line prefixes before
    # whitespace-normalizing so a clause spanning two quoted lines stays one
    # contiguous phrase.
    normalized_lede = " ".join(
        _locate_paragraph(readme, lede_anchor).replace("> ", "").split()
    )
    assert "incremental documentation engine" in normalized_lede
    assert (
        "treats documentation as reusable state instead of regenerating it on "
        "every run" in normalized_lede
    ), "README lede no longer states the reusable-state-vs-regenerate positioning"
    assert (
        "reuses compatible completed records and recovery checkpoints"
        in normalized_lede
    )
    assert "sends only the remaining provider-bound work" in normalized_lede

    # --- RUN_FLOW: reuse/recovery documented as a first-class run phase. ---
    assert (
        "This document describes the active phase ordering" in run_flow
    ), "RUN_FLOW.md lost its lifecycle-document framing"
    assert "## Split planning, reuse, and recovery" in run_flow, (
        "RUN_FLOW.md no longer gives reuse/recovery its own phase heading"
    )
    assert "## Cache and recovery identity" in run_flow, (
        "RUN_FLOW.md no longer treats cache/reuse identity as its own phase"
    )
    normalized_predicate = " ".join(
        _locate_paragraph(
            run_flow,
            "Ordinary per-file reuse uses one centralized predicate over content hash",
        ).split()
    )
    assert "one centralized predicate over content hash" in normalized_predicate
    assert "the registered cache identity" in normalized_predicate

    normalized_reuse_boundary = " ".join(
        _locate_paragraph(
            run_flow,
            "An exactly compatible same-path completed split record authorizes",
        ).split()
    )
    assert (
        "zero-call" in normalized_reuse_boundary
        and "reuse" in normalized_reuse_boundary
    ), "RUN_FLOW no longer states zero-call reuse of a compatible completed record"
    assert "resumes only unpaid nodes" in normalized_reuse_boundary, (
        "RUN_FLOW no longer states an interrupted run resumes rather than restarts"
    )


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
    # The newest changelog section is the release manifest for this contract.
    # Internal planning documents are deliberately not part of
    # the repository and must never be a source of current identity truth.
    # Locate the newest release section structurally -- it is whatever comes
    # first after the "# Changelog" title -- rather than matching one
    # version's exact heading text, so this test survives every version bump
    # without needing to rewrite an earlier historical heading.
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
    so an unrelated historical sentence elsewhere in the document cannot fail
    this test."""
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
    """Ordinary identical-content reuse (`files_reused_identical_content`) is
    same-path only -- CodeDoc never copies documentation across paths, even
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
    """A final response-contract failure names its closed reason code, and a
    custom api_base_url requires runtime endpoint-trust approval
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
    """Gate 17.13, assertion 4: the inherited SECURITY.md wording about
    `crash_recovery.json` sensitivity remains present and untouched. This patch
    declares no new security contract and must not become a second owner of
    that text (preserved, not edited, by section 4A)."""
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


def _github_heading_anchor(heading_text: str) -> str:
    """Approximate GitHub's heading-to-anchor slug: lowercase, drop every
    character that is not a word character, whitespace, or hyphen, then
    hyphenate whitespace runs."""
    slug = heading_text.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


def test_readme_table_of_contents_links_resolve_to_headings():
    """Every in-document `](#...)` link in README's Contents list must point at
    a heading that exists in the file.

    Written against the whole Contents list rather than one heading, so a future
    section rename that leaves its table-of-contents entry pointing at a dead
    anchor is caught without adding a one-off assertion for each rename."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    contents = readme.split("## Contents", 1)[1].split("\n## ", 1)[0]
    toc_anchors = re.findall(r"\]\(#([^)]+)\)", contents)
    assert toc_anchors, "README Contents section exposes no anchor links to check."

    heading_anchors = {
        _github_heading_anchor(text)
        for text in re.findall(r"^#{1,6}\s+(.*?)\s*$", readme, flags=re.MULTILINE)
    }

    unresolved = sorted(a for a in toc_anchors if a not in heading_anchors)
    assert not unresolved, (
        f"README Contents links with no matching heading anchor: {unresolved}"
    )


# ---------------------------------------------------------------------------
# Section 11 mutation check 27 / section 7.1: source exactness is defined
# against the canonical decoded snapshot, never the raw file. The paragraphs
# that carry that -- the canonical-snapshot / BOM / replacement-decode /
# newline-normalization statement, the separate original-byte content hash,
# the explicit "no raw round-trip" disclaimer, the three local-subdivision
# worked examples, and the zero CRLF-atomicity delta for a normal
# filesystem-loaded file -- were previously unpinned, so mutation check 27
# could not be falsified through README/RUN_FLOW. These contracts close that.
# Located by section heading and whitespace-normalized (matching the sibling
# recovery-remedy and positioning tests), never a whole-file snapshot.
# ---------------------------------------------------------------------------

_SEMANTIC_DIVISION_HEADING = "#### Semantic division and synthesis"
_SPLIT_PLANNING_HEADING = "## Split planning, reuse, and recovery"


def _doc_section(text: str, heading: str) -> str:
    """Whitespace-normalized body between *heading* and the next heading of the
    same or higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    start = text.index(heading) + len(heading)
    rest = text[start:]
    ends = [
        rest.index(marker)
        for marker in ("\n" + "#" * k + " " for k in range(1, level + 1))
        if marker in rest
    ]
    return " ".join(rest[: min(ends) if ends else len(rest)].split())


def test_public_docs_define_source_exactness_against_the_canonical_decoded_snapshot():
    """README's `Semantic division and synthesis` section must keep defining
    split planning/reconstruction against the canonical decoded snapshot -- BOM
    stripped, invalid UTF-8 replaced, CRLF and lone CR normalized to LF -- with
    a separate content hash still binding the original bytes and an explicit
    statement that the raw file is not preserved or round-tripped. RUN_FLOW's
    split phase must at least carry the canonical-snapshot framing, exactly-once
    leaf coverage, and the same newline normalization.

    Mutation check 27: replacing this with a raw-file-preservation claim, or
    deleting the BOM/replacement/newline sentence, must fail here."""
    readme_section = _doc_section(
        (ROOT / "README.md").read_text(encoding="utf-8"), _SEMANTIC_DIVISION_HEADING
    )
    run_flow_section = _doc_section(
        (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8"), _SPLIT_PLANNING_HEADING
    )

    # Claim 1: canonical decoded snapshot is the reconstruction authority.
    assert (
        "reconstruction are defined against the **canonical decoded snapshot**"
        in readme_section
    )
    assert "not its raw bytes" in readme_section
    assert "measured in Unicode code points of that snapshot" in readme_section
    assert "pieces of a divided file reconstruct it exactly" in readme_section
    # Claim 2: UTF-8 BOM stripped.
    assert "a leading byte-order mark is stripped" in readme_section
    # Claim 3: invalid UTF-8 decoded with replacement.
    assert (
        "undecodable bytes become the Unicode replacement character"
        in readme_section
    )
    # Claim 4: CRLF and lone CR normalized to LF before split planning.
    assert r"`\r\n` and lone `\r` are normalized to `\n`" in readme_section
    # Claim 5: the separate raw content hash still binds the original bytes.
    assert (
        "records a separate content hash over the file's original bytes"
        in readme_section
    )
    # Claim 6: no promise to preserve or round-trip the raw source file.
    assert (
        "does not promise to preserve or round-trip the raw file itself"
        in readme_section
    )

    # RUN_FLOW carries the framing and the same newline normalization.
    assert (
        "CodeDoc reads one canonical decoded snapshot per selected source file"
        in run_flow_section
    )
    assert (
        "Every source character belongs to exactly one planned leaf"
        in run_flow_section
    )
    assert r"normalizes `\r\n` and lone `\r` to `\n` first" in run_flow_section


def test_public_docs_state_the_local_subdivision_worked_examples():
    """Both README and RUN_FLOW must keep all three section 5.6 worked
    examples, stated against their anti-patterns so a reader cannot mistake the
    behavior:

    * a 2,010-character unit at B=1,000 becomes about 670 + 670 + 670, not
      1,000 + 1,000 + 10;
    * an 8,292-character span at B=2,000 becomes five locally balanced pieces,
      not a power-of-two recursive halving into eight;
    * given natural units of 1,243 / 482 / 285, only the 1,243 unit is
      subdivided near 622 + 621 while the fitting 482 and 285 units keep their
      exact bytes, ranges, and identities (and, README adds, may co-pack into a
      767-character leaf call).

    Mutation check 27: deleting these examples from either document must fail
    here."""
    readme_section = _doc_section(
        (ROOT / "README.md").read_text(encoding="utf-8"), _SEMANTIC_DIVISION_HEADING
    )
    run_flow_section = _doc_section(
        (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8"), _SPLIT_PLANNING_HEADING
    )

    for section, name in (
        (readme_section, "README.md"),
        (run_flow_section, "RUN_FLOW.md"),
    ):
        # 2,010 @ B=1,000 -> ~670 x 3, contrasted with 1,000 + 1,000 + 10.
        assert "2,010-character unit at a 1,000-character ceiling" in section, name
        assert "670 character" in section, name
        assert "not 1,000 + 1,000 + 10" in section, name
        # 8,292 @ B=2,000 -> five pieces, not eight by halving.
        assert "8,292-character span at a 2,000-character ceiling" in section, name
        assert "becomes five pieces" in section, name
        assert "power-of-two" in section and "halving into eight" in section, name
        # 1,243 / 482 / 285 -> only 1,243 subdivided (~622 + 621); others kept.
        assert (
            "natural units of 1,243, 482, and 285 characters" in section
        ), name
        assert "1,243-character unit is subdivided (about 622 + 621)" in section, name
        assert (
            "482 and 285 units keep their exact bytes, ranges, and identities"
            in section
        ), name

    # README additionally states the fitting units may co-pack into one call.
    assert (
        "482 and 285 units keep their exact bytes, ranges, and identities and "
        "may still share one 767-character leaf call" in readme_section
    )


def test_public_docs_state_zero_crlf_atomicity_delta_for_filesystem_loaded_files():
    """Both documents must keep saying that the CRLF-atomicity extra-piece case
    is a defensive, direct/internal-only path: a normal filesystem-loaded file
    has already had CRLF and lone CR normalized to LF, so its reported
    CRLF-atomicity delta is always zero.

    Mutation check 27: changing that reported delta from "always zero" to
    "potentially nonzero", or deleting RUN_FLOW's equivalent explanation, must
    fail here."""
    readme_section = _doc_section(
        (ROOT / "README.md").read_text(encoding="utf-8"), _SEMANTIC_DIVISION_HEADING
    )
    run_flow_section = _doc_section(
        (ROOT / "RUN_FLOW.md").read_text(encoding="utf-8"), _SPLIT_PLANNING_HEADING
    )

    # The extra piece is described as defensive and CRLF-pair driven, not a
    # normal outcome.
    assert r"a `\r\n` pair is treated as one" in readme_section
    assert "defensive subdivision may" in readme_section
    assert "one extra piece" in readme_section and "one extra call" in readme_section
    assert (
        r"normal filesystem pipeline normalizes `\r\n` and lone `\r` to `\n` "
        r"before planning, so that case cannot arise there and its reported "
        r"count is always zero" in readme_section
    )

    assert r"A `\r\n` pair is indivisible for cut placement" in run_flow_section
    assert (
        "defensive subdivision may use one extra piece and one extra call"
        in run_flow_section
    )
    assert (
        r"normal filesystem pipeline normalizes `\r\n` and lone `\r` to `\n` "
        r"first, so that count is always zero there" in run_flow_section
    )


#: An *affirmative* claim that CodeDoc preserves or round-trips the raw source
#: bytes/file -- the shape mutation check 27 forbids. Every legitimate current
#: use is either a different object (the generated JSON/Markdown "round trips",
#: a completed cache record "round-trips unchanged") or an explicit negation
#: ("does not promise to preserve or round-trip the raw file"), so the detector
#: is scoped to a sentence that names the raw/original source file-or-bytes,
#: makes a preservation/round-trip claim about it, and carries no negation.
_RAW_SOURCE_TOKEN = re.compile(
    r"\braw(?:[- ]source)?[- ](?:file|bytes?|source)\b"
    r"|\boriginal(?:[- ]source)?[- ](?:file|bytes?)\b",
    re.IGNORECASE,
)
_PRESERVATION_TOKEN = re.compile(
    r"\b(?:preserv\w*|round[- ]?trip\w*|byte[- ]for[- ]byte|reproduc\w*|"
    r"kept\s+intact|retained\s+verbatim|losslessly)\b",
    re.IGNORECASE,
)
_PRESERVATION_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|does\s+not|is\s+not|are\s+not|without|"
    r"rather\s+than|instead\s+of)\b|n't\b",
    re.IGNORECASE,
)


def _affirmative_raw_preservation_sentences(text: str) -> list[str]:
    return [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if _RAW_SOURCE_TOKEN.search(sentence)
        and _PRESERVATION_TOKEN.search(sentence)
        and not _PRESERVATION_NEGATION.search(sentence)
    ]


def test_public_docs_make_no_affirmative_raw_source_preservation_claim():
    """Negative contract for mutation check 27: neither README nor RUN_FLOW may
    claim CodeDoc preserves or round-trips the raw source bytes/file. The one
    sentence that mentions the raw file must stay a negation."""
    # Detector fires on the forbidden shape ...
    assert _affirmative_raw_preservation_sentences(
        "CodeDoc preserves the raw source file byte-for-byte across a run."
    )
    assert _affirmative_raw_preservation_sentences(
        "The raw file round-trips exactly and the original bytes are reproduced."
    )
    # ... and stays silent on the real negated sentence and on unrelated
    # round-trip uses (generated Markdown, completed cache records).
    assert not _affirmative_raw_preservation_sentences(
        "CodeDoc also records a separate content hash over the file's original "
        "bytes for change detection; it does not promise to preserve or "
        "round-trip the raw file itself."
    )
    assert not _affirmative_raw_preservation_sentences(
        "The JSON has an embedded Markdown view for safe round trips."
    )
    assert not _affirmative_raw_preservation_sentences(
        "A completed cache record round-trips unchanged and compares stale."
    )

    for path in (ROOT / "README.md", ROOT / "RUN_FLOW.md"):
        offenders = _affirmative_raw_preservation_sentences(
            path.read_text(encoding="utf-8")
        )
        assert not offenders, (
            f"{path.name} affirmatively claims raw-source preservation: {offenders}"
        )
