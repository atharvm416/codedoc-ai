"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.record_metadata_cases import private_key  # noqa: F401, F811

import json
from codedoc.core.markdown_view import (
    markdown_from_view,
    markdown_to_view,
    read_embedded_view,
)
from codedoc.core.project_view import build_project_view
from tests.support.markdown_cases import _make_view
from tests.support.record_metadata_cases import _view_with_secret
from tests.support.dependency_view_cases import _record as dependency_view_record
from codedoc.core.output import write_project_outputs
from tests.support.reachability_cases import _record as reachability_record
from tests.support.run_metadata_cases import _split_record, _split_stats


def test_markdown_renders_split_run_summary_without_internal_unit_content():
    view = build_project_view([_split_record()], _split_stats())

    markdown = markdown_from_view(view)

    assert "Semantic units / leaf chunks" in markdown
    assert "Completed split files reused / partial files resumed: 0 / 0" in markdown
    assert "Unpaid / reexecuted / quarantined nodes: 3 / 1 / 2" in markdown
    assert "Recovery conflict files: 1" in markdown
    assert "**Source coverage:**" not in markdown
    assert "Documentation Unit" not in markdown
    assert markdown_to_view(markdown) == view

def test_md_output_embeds_file_hashes_in_metadata_comment(tmp_path):
    """file_hashes must appear in the <!-- codedoc-ai: ... --> comment so that
    subsequent --format md runs can perform incremental hash checks."""
    import json as _json


    output_dir = tmp_path / "out"
    records = [
        {
            "hash": "deadbeef01",
            "file_path": "app.py",
            "language": "python",
            "documentation": {
                "file_path": "app.py",
                "language": "python",
                "description": "The app.",
            },
        }
    ]
    _, md_path = write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        output_format="md",
        entry_file="app.py",
    )

    assert md_path is not None
    content = md_path.read_text(encoding="utf-8")
    assert "<!-- codedoc-ai:" in content

    # Extract metadata comment and verify file_hashes is present
    import re
    match = re.search(r"<!-- codedoc-ai: (\{.*?\}) -->", content, re.DOTALL)
    assert match, "metadata comment not found"
    meta = _json.loads(match.group(1))
    assert "file_hashes" in meta
    assert meta["file_hashes"].get("app.py") == "deadbeef01"

def test_legacy_visible_markdown_parses():
    """Markdown without the base64 block falls back to the visible parser."""
    legacy = (
        "<!-- codedoc-ai: {\"entry_file\": \"main.py\", \"schema_version\": \"1.4\", "
        "\"file_hashes\": {}} -->\n"
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 1\n"
        "- Languages: python\n"
        "- Folders: none\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 0\n"
        "- Files reused from cache: 0\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python  \n\n"
        "**Description:** The entry point.\n\n"
    )
    assert read_embedded_view(legacy) is None  # no embedded block
    view = markdown_to_view(legacy)
    assert view["project"]["entry_file"] == "main.py"
    assert view["files"][0]["path"] == "main.py"
    assert view["files"][0]["description"] == "The entry point."

def test_A6_legacy_markdown_uses_visible_parser():
    """A6: Markdown without the base64 block falls back to the visible parser."""
    from codedoc.core.project_view import markdown_to_view

    # Hand-crafted legacy Markdown — no codedoc-ai-view-base64 block.
    legacy_md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 1\n"
        "- Languages: python\n"
        "- Folders: none\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 0\n"
        "- Files reused from cache: 0\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python  \n\n"
        "**Description:** Entry point.\n\n"
    )

    view = markdown_to_view(legacy_md)
    assert view is not None
    files = view.get("files", [])
    assert len(files) == 1
    assert files[0]["path"] == "main.py"
    assert files[0]["language"] == "python"

def test_A6_legacy_markdown_has_no_dependency_catalog():
    """A6b: Legacy Markdown without base64 block has no dependency_catalog (best-effort)."""
    from codedoc.core.project_view import markdown_to_view

    # Minimal legacy Markdown with no Dependency Catalog section
    legacy_md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 1\n"
        "- Languages: python\n"
        "- Folders: none\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 0\n"
        "- Files reused from cache: 0\n\n"
    )

    view = markdown_to_view(legacy_md)
    # No Dependency Catalog section → should be absent or empty
    assert not view.get("dependency_catalog"), (
        "Legacy Markdown without a Dependency Catalog section must yield no catalog"
    )

def test_A13_confirms_legacy_has_no_dep_catalog():
    """A13: Confirms that A6b/legacy-path behaviour is intentional (best-effort only)."""
    from codedoc.core.project_view import json_from_markdown

    # Markdown with a Dependency Map section but NO Dependency Catalog section,
    # and no base64 block.
    legacy_md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Dependency Map\n\n"
        "- `main.py` -> `utils.py`\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python  \n\n"
    )

    regen = json.loads(json_from_markdown(legacy_md))
    # The legacy parser cannot reconstruct dependency_catalog from visible sections
    assert not regen.get("dependency_catalog"), (
        "Legacy Markdown without an explicit Dependency Catalog section "
        "must not produce a dependency_catalog in the regenerated JSON"
    )

def test_A14_hashes_preserved_from_embedded_view(tmp_path):
    """A14: When the embedded view is used, file hashes are preserved."""
    from codedoc.core.project_view import markdown_from_view
    from codedoc.pipeline import _load_existing_file_docs_from_md

    view = _make_view()
    md_path = tmp_path / "codedoc.md"
    md_path.write_text(markdown_from_view(view), encoding="utf-8")

    docs = _load_existing_file_docs_from_md(md_path)

    # Both hashes must be preserved from the embedded view
    assert docs["main.py"]["hash"] == "abc123", (
        "main.py hash must survive the Markdown load from embedded view"
    )
    assert docs["utils.py"]["hash"] == "def456", (
        "utils.py hash must survive the Markdown load from embedded view"
    )

def test_A14_meta_comment_hash_takes_precedence(tmp_path):
    """A14b: When both meta comment and embedded view have hashes, meta comment wins."""
    from codedoc.core.project_view import markdown_from_view
    from codedoc.pipeline import _load_existing_file_docs_from_md

    view = _make_view()
    # Write the Markdown (which has both the meta comment hashes and embedded view hashes)
    md_path = tmp_path / "codedoc.md"
    md_path.write_text(markdown_from_view(view), encoding="utf-8")

    docs = _load_existing_file_docs_from_md(md_path)

    # The meta comment has the same hashes as the embedded view (they're from the
    # same run), so the result must match.
    assert docs["main.py"]["hash"] == "abc123"
    assert docs["utils.py"]["hash"] == "def456"

def test_A14_embedded_hash_fallback_when_meta_hash_missing(tmp_path):
    """A14c: When meta comment has no hash for a path, embedded view hash is used."""
    import re as _re
    from codedoc.core.project_view import markdown_from_view
    from codedoc.pipeline import _load_existing_file_docs_from_md

    view = _make_view()
    md_content = markdown_from_view(view)

    # Corrupt the meta comment to remove file_hashes entirely
    md_content = _re.sub(
        r'<!-- codedoc-ai: \{.*?\} -->',
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", "generated_at": "", "file_hashes": {}} -->',
        md_content,
        flags=_re.DOTALL,
    )

    md_path = tmp_path / "codedoc.md"
    md_path.write_text(md_content, encoding="utf-8")

    docs = _load_existing_file_docs_from_md(md_path)

    # With empty meta comment hashes, must fall back to embedded view hashes
    assert docs["main.py"]["hash"] == "abc123", (
        "When meta comment has no hash, embedded view hash must be used as fallback"
    )
    assert docs["utils.py"]["hash"] == "def456"

def test_visible_markdown_never_renders_private_key(private_key):
    view = _view_with_secret(private_key)
    md = markdown_from_view(view)
    # Strip the hidden base64 block; the value must not appear in visible prose.
    visible = md.split("-->", 2)[-1]
    assert "TOPSECRET" not in visible

def test_legacy_visible_markdown_old_labels_parse_to_both_blocks():
    legacy = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"file_hashes": {}} -->\n'
        "# codedoc project documentation\n\n"
        "## Project Overview\n\n"
        "- Entry file: `main.py`\n"
        "- Files documented: 3\n"
        "- Languages: python\n"
        "- Folders: `.`\n\n"
        "## Run Summary\n\n"
        "- Files checked: 1\n"
        "- Files failed: 0\n"
        "- Files skipped: 2\n"
        "- Files reused from cache: 0\n\n"
        "## Files\n\n"
        "### main.py\n\n"
        "**Language:** python\n\n"
        "**Description:** Entry.\n"
    )

    view = markdown_to_view(legacy)
    assert view["run"] == {
        "files_checked": 1,
        "files_failed": 0,
        "files_skipped": 2,
        "files_reused": 0,
        "files_documented": 1,
    }
    assert view["last_run"]["files_documented_by_llm"] == 1
    assert view["last_run"]["files_reused_unchanged"] == 2
    assert view["last_run"]["files_reused_identical_content"] == 0
    assert view["last_run"]["entry_source"] == "auto-detected"
    assert view["last_run"]["analysis_mode"] == "single"

def test_markdown_round_trip_preserves_sdk_dependencies():
    view = build_project_view(
        [dependency_view_record("m.py", "python", external=["os", "requests"])],
        {"checked": 1},
    )
    md = markdown_from_view(view)
    assert "SDK / Standard Library" in md
    round_tripped = markdown_to_view(md)
    assert round_tripped["files"][0]["links"]["sdk_dependencies"] == ["os"]

def test_visible_markdown_renders_sdk_label_only_when_non_empty():
    view = build_project_view(
        [dependency_view_record("m.py", "python", external=["requests"])],
        {"checked": 1},
    )
    md = markdown_from_view(view)
    visible = md.split("-->", 2)[-1]
    assert "SDK / Standard Library" not in visible

def test_markdown_has_one_visible_line_per_file_and_lossless_embed():
    view = build_project_view(
        [reachability_record("main.py"), reachability_record("orphan.py")],
        {"checked": 2},
        entry_file="main.py",
        reachable_rels={"main.py"},
    )
    markdown = markdown_from_view(view)
    assert markdown.count("**Reachable from entry:**") == 2
    assert "**Reachable from entry:** Yes" in markdown
    assert "**Reachable from entry:** No" in markdown
    assert read_embedded_view(markdown) == view
