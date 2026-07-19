"""Tests organized by feature ownership."""

from __future__ import annotations

import base64
from pathlib import Path
import pytest
from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.markdown_view import (
    _CODEDOC_VIEW_BASE64_RE,
    json_from_markdown,
    markdown_to_view,
    read_embedded_view_result,
)
from codedoc.core.project_view import build_project_view, markdown_from_view
from codedoc.core.resume import _load_existing_file_docs_from_md
from codedoc.utils.errors import ConfigError

VARIANTS = ("lf", "crlf", "cr")

def _variant(lf: str, kind: str) -> str:
    """Return *lf* re-expressed in the requested newline form."""
    if kind == "lf":
        return lf
    if kind == "crlf":
        return lf.replace("\n", "\r\n")
    if kind == "cr":
        return lf.replace("\n", "\r")
    raise AssertionError(f"unknown variant {kind!r}")

def _write_variant(path: Path, lf: str, kind: str) -> Path:
    """Write *lf* in the requested newline form as exact bytes (no retranslation)."""
    path.write_bytes(_variant(lf, kind).encode("utf-8"))
    return path

def _decoded_payload(text: str) -> bytes:
    """Return the raw base64-decoded embedded-view bytes from *text*."""
    match = _CODEDOC_VIEW_BASE64_RE.search(text)
    assert match is not None, "expected an embedded base64 block"
    return base64.b64decode(match.group(1).strip(), validate=True)

VISIBLE_LEGACY_LF = (
    '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
    '"file_hashes": {"main.py": "h-main", "utils.py": "h-utils"}} -->\n'
    "# codedoc project documentation\n"
    "\n"
    "## Project Overview\n"
    "\n"
    "- Entry file: `main.py`\n"
    "- Files documented: 2\n"
    "- Languages: python\n"
    "- Folders: `.`\n"
    "\n"
    "## Run Summary\n"
    "\n"
    "- Files checked: 2\n"
    "- Files failed: 0\n"
    "- Files skipped: 0\n"
    "- Files reused from cache: 0\n"
    "\n"
    "## Project Tree\n"
    "\n"
    "```text\n"
    "main.py\n"
    "utils.py\n"
    "```\n"
    "\n"
    "## Dependency Map\n"
    "\n"
    "- `main.py` -> `utils.py`\n"
    "\n"
    "## Files\n"
    "\n"
    "### main.py\n"
    "\n"
    "**Language:** python\n"
    "\n"
    "**Description:** Entry point of the app.\n"
    "\n"
    "**Imports:**\n"
    "\n"
    "- `utils`\n"
    "\n"
    "**Usage Example:**\n"
    "\n"
    "```text\n"
    "python main.py\n"
    "```\n"
    "\n"
    "### utils.py\n"
    "\n"
    "**Language:** python\n"
    "\n"
    "**Description:** Helper utilities.\n"
)

SCHEMALESS_VISIBLE_LF = (
    '<!-- codedoc-ai: {"entry_file": "main.py", '
    '"file_hashes": {"main.py": "h-main"}} -->\n'
    "# codedoc project documentation\n"
    "\n"
    "## Files\n"
    "\n"
    "### main.py\n"
    "\n"
    "**Language:** python\n"
    "\n"
    "**Description:** Entry.\n"
)

FOREIGN_LF = (
    "# Just some notes\n"
    "\n"
    "This file has no CodeDoc marker at all.\n"
    "\n"
    "## Files\n"
    "\n"
    "### main.py\n"
    "\n"
    "**Language:** python\n"
)

CORRUPT_EMBEDDED_LF = (
    '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4"} -->\n'
    "# codedoc project documentation\n"
    "\n"
    "<!-- codedoc-ai-view-base64\n"
    "!!! not valid base64 !!!\n"
    "-->\n"
    "\n"
    "## Files\n"
    "\n"
    "### main.py\n"
    "\n"
    "**Language:** python\n"
)

def _embedded_lf() -> str:
    """Render a current lossless embedded-view Markdown document (all LF)."""
    view = build_project_view(
        [
            {
                "hash": "h1",
                "file_path": "main.py",
                "language": "python",
                "documentation": {
                    "file_path": "main.py",
                    "language": "python",
                    "description": "Entry point.",
                    "imports": ["utils"],
                },
            }
        ],
        {"checked": 1},
        entry_file="main.py",
    )
    return markdown_from_view(view)

@pytest.mark.parametrize("kind", VARIANTS)
def test_visible_markdown_to_view_is_newline_agnostic(kind):
    baseline = markdown_to_view(VISIBLE_LEGACY_LF)
    got = markdown_to_view(_variant(VISIBLE_LEGACY_LF, kind))
    # Whole-view equality covers project, run, files, imports, descriptions,
    # links, and usage_example simultaneously.
    assert got == baseline

    # Sanity-pin what the baseline actually parsed, so the equality above is
    # comparing a fully-populated view (not two empty ones).
    assert [f["path"] for f in baseline["files"]] == ["main.py", "utils.py"]
    main = baseline["files"][0]
    assert main["description"] == "Entry point of the app."
    assert main["imports"] == ["utils"]
    assert main["usage_example"] == "python main.py"
    assert main["links"]["internal_dependencies"] == ["utils.py"]
    assert baseline["files"][1]["links"]["imported_by"] == ["main.py"]

@pytest.mark.parametrize("kind", VARIANTS)
def test_read_codedoc_document_visible_is_newline_agnostic(tmp_path, kind):
    # This is the exact path the two baseline tests exercise: read_bytes +
    # decode (no universal-newline translation).  CRLF/CR fail before the fix.
    base = read_codedoc_document(_write_variant(tmp_path / "base.md", VISIBLE_LEGACY_LF, "lf"))
    doc = read_codedoc_document(_write_variant(tmp_path / f"doc_{kind}.md", VISIBLE_LEGACY_LF, kind))

    assert [f["path"] for f in doc.files] == [f["path"] for f in base.files] == ["main.py", "utils.py"]
    assert doc.file_hashes == base.file_hashes == {"main.py": "h-main", "utils.py": "h-utils"}
    assert doc.entry_file == base.entry_file == "main.py"
    assert doc.schema_version == base.schema_version == "1.4"

@pytest.mark.parametrize("kind", VARIANTS)
def test_resume_and_json_agree_across_variants(tmp_path, kind):
    baseline_view = markdown_to_view(VISIBLE_LEGACY_LF)
    baseline_json = json_from_markdown(VISIBLE_LEGACY_LF)

    text = _variant(VISIBLE_LEGACY_LF, kind)
    assert markdown_to_view(text) == baseline_view
    assert json_from_markdown(text) == baseline_json

    # _load_existing_file_docs_from_md reads via Path.read_text(), whose
    # universal-newline translation already normalizes — so it agrees today.
    # Pin the invariant anyway in case that reader ever switches to read_bytes().
    p = _write_variant(tmp_path / f"resume_{kind}.md", VISIBLE_LEGACY_LF, kind)
    records = _load_existing_file_docs_from_md(p)
    assert set(records) == {f["path"] for f in baseline_view["files"]} == {"main.py", "utils.py"}
    assert records["main.py"]["hash"] == "h-main"

@pytest.mark.parametrize("kind", VARIANTS)
def test_embedded_view_and_payload_are_newline_agnostic(tmp_path, kind):
    lf = _embedded_lf()
    text = _variant(lf, kind)

    # The decoded view is identical across variants ...
    assert markdown_to_view(text) == markdown_to_view(lf)
    assert read_embedded_view_result(text).view == read_embedded_view_result(lf).view
    # ... and the raw base64-decoded payload bytes are byte-identical, proving
    # normalization never reaches inside the embedded block.
    assert _decoded_payload(text) == _decoded_payload(lf)

    doc = read_codedoc_document(_write_variant(tmp_path / f"emb_{kind}.md", lf, kind))
    assert doc.entry_file == "main.py"
    assert records_by_path(doc)["main.py"]["imports"] == ["utils"]

@pytest.mark.parametrize("kind", VARIANTS)
def test_corrupt_embedded_block_stays_invalid(tmp_path, kind):
    text = _variant(CORRUPT_EMBEDDED_LF, kind)
    assert read_embedded_view_result(text).state == "invalid"
    with pytest.raises(ConfigError):
        read_codedoc_document(_write_variant(tmp_path / f"corrupt_{kind}.md", CORRUPT_EMBEDDED_LF, kind))

@pytest.mark.parametrize("kind", VARIANTS)
def test_schemaless_visible_only_still_rejected(tmp_path, kind):
    with pytest.raises(ConfigError):
        read_codedoc_document(
            _write_variant(tmp_path / f"schemaless_{kind}.md", SCHEMALESS_VISIBLE_LF, kind)
        )

@pytest.mark.parametrize("kind", VARIANTS)
def test_foreign_markdown_still_rejected(tmp_path, kind):
    with pytest.raises(ConfigError):
        read_codedoc_document(_write_variant(tmp_path / f"foreign_{kind}.md", FOREIGN_LF, kind))

@pytest.mark.parametrize("kind", VARIANTS)
def test_read_does_not_mutate_file(tmp_path, kind):
    p = _write_variant(tmp_path / f"nomutate_{kind}.md", VISIBLE_LEGACY_LF, kind)
    before_bytes = p.read_bytes()
    before_mtime = p.stat().st_mtime_ns

    read_codedoc_document(p)

    assert p.read_bytes() == before_bytes
    assert p.stat().st_mtime_ns == before_mtime
