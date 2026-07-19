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
from tests.support.project_view_cases import _build_view
import base64
from tests.support.markdown_cases import _make_view
from tests.support.record_metadata_cases import _view_with_secret
from codedoc.core.document import (
    read_codedoc_document,
)
from tests.support.run_metadata_cases import _records as run_metadata_records
from tests.support.run_metadata_cases import _stats as run_metadata_stats
from tests.support.run_metadata_cases import _partition_sum
from tests.support.json_document_cases import _view

def test_embedded_view_round_trip_lossless():
    view = _build_view()
    md = markdown_from_view(view)
    assert markdown_to_view(md) == view

def test_private_keys_survive_embedded_absent_from_visible(monkeypatch):
    monkeypatch.setattr(
        "codedoc.core.record_meta.PRIVATE_RECORD_KEYS", frozenset({"_secret_marker"})
    )
    records = [
        {
            "hash": "h",
            "file_path": "a.py",
            "language": "python",
            "_secret_marker": "KEEPME",
            "documentation": {"description": "x", "dependencies_analysis": {}},
        }
    ]
    view = build_project_view(records, {"checked": 1}, entry_file="a.py")
    assert view["files"][0]["_secret_marker"] == "KEEPME"

    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert embedded["files"][0]["_secret_marker"] == "KEEPME"
    # The visible Markdown (with the hidden base64 block removed) must not leak it.
    import re

    visible = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    assert "_secret_marker" not in visible
    assert "KEEPME" not in visible

def test_A1_legacy_meta_comment_present():
    """A1: Generated Markdown always starts with the legacy <!-- codedoc-ai: ... --> comment."""
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    assert "<!-- codedoc-ai:" in md, "Legacy metadata comment must be present"
    assert '"entry_file"' in md
    assert '"file_hashes"' in md

def test_A2_base64_block_present():
    """A2: Generated Markdown contains <!-- codedoc-ai-view-base64 ... -->."""
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    assert "<!-- codedoc-ai-view-base64" in md, "Base64 view block must be present"
    assert "-->" in md  # block is properly closed

def test_A2_base64_block_comes_after_meta_comment():
    """A2b: The base64 block appears after the legacy metadata comment."""
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    meta_pos = md.index("<!-- codedoc-ai:")
    b64_pos = md.index("<!-- codedoc-ai-view-base64")
    assert b64_pos > meta_pos, "Base64 block must appear after the legacy comment"

def test_A3_embedded_view_decodes_to_valid_json():
    """A3: The base64 payload decodes and parses as a valid JSON dict."""
    import re
    from codedoc.core.project_view import markdown_from_view

    md = markdown_from_view(_make_view())
    match = re.search(r"<!-- codedoc-ai-view-base64\s*([\s\S]*?)\s*-->", md)
    assert match, "Base64 block must be present"

    raw = match.group(1).strip()
    decoded = base64.b64decode(raw)
    data = json.loads(decoded.decode("utf-8"))

    assert isinstance(data, dict), "Decoded payload must be a dict"
    assert "schema_version" not in data
    assert "last_run" in data
    assert "project" not in data
    assert "run" not in data
    assert "files" in data

def test_A3_embedded_view_contains_expected_fields():
    """A3b: The decoded view contains all major fields."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    view = _make_view()
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)

    assert embedded is not None
    assert "schema_version" not in embedded
    assert embedded["last_run"]["entry_file"] == "main.py"
    assert "project" not in embedded
    assert "run" not in embedded
    assert len(embedded["files"]) == 2
    assert embedded["dependency_graph"] == view["dependency_graph"]
    assert embedded["dependency_catalog"] == view["dependency_catalog"]

def test_A4_embedded_view_has_no_crash_safety():
    """A4: The embedded view must not contain _crash_safety."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    md = markdown_from_view(_make_view())
    embedded = read_embedded_view(md)

    assert embedded is not None
    assert "_crash_safety" not in embedded

def test_A4_embedded_view_has_no_in_progress_status():
    """A4b: The embedded view must not have _codedoc.status = 'in_progress'."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    md = markdown_from_view(_make_view())
    embedded = read_embedded_view(md)

    assert embedded is not None
    codedoc_meta = embedded.get("_codedoc", {})
    assert not (isinstance(codedoc_meta, dict) and codedoc_meta.get("status") == "in_progress")

def test_A4_read_embedded_view_rejects_crash_safety_block():
    """A4c: read_embedded_view rejects payloads containing _crash_safety."""
    from codedoc.core.project_view import read_embedded_view

    bad_view = {"schema_version": "1.4", "project": {}, "files": [], "_crash_safety": "INCOMPLETE"}
    b64 = base64.b64encode(json.dumps(bad_view).encode()).decode()
    md = f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"

    result = read_embedded_view(md)
    assert result is None, "Crash-safety block must be rejected"

def test_A4_read_embedded_view_rejects_in_progress_snapshot():
    """A4d: read_embedded_view rejects in-progress snapshots."""
    from codedoc.core.project_view import read_embedded_view

    bad_view = {
        "schema_version": "1.4",
        "project": {},
        "files": [],
        "_codedoc": {"status": "in_progress"},
    }
    b64 = base64.b64encode(json.dumps(bad_view).encode()).decode()
    md = f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"

    result = read_embedded_view(md)
    assert result is None, "In-progress snapshot must be rejected"

def test_A7_corrupted_base64_falls_back_to_visible_parser():
    """A7: A corrupt base64 block triggers a warning and falls back to visible parser."""
    from codedoc.core.project_view import markdown_to_view

    md = (
        '<!-- codedoc-ai: {"entry_file": "main.py", "schema_version": "1.4", '
        '"generated_at": "2026-01-01T00:00:00+00:00", "file_hashes": {}} -->\n'
        "<!-- codedoc-ai-view-base64\n"
        "THIS_IS_NOT_VALID_BASE64!!!\n"
        "-->\n"
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
        "**Description:** Fallback description.\n\n"
    )

    # Must not raise — falls back to visible parser
    view = markdown_to_view(md)
    assert view is not None
    files = view.get("files", [])
    assert any(f["path"] == "main.py" for f in files), (
        "Fallback to visible parser must still recover the file record"
    )

def test_A7_valid_json_but_missing_required_fields_falls_back():
    """A7b: Valid base64 JSON but missing required fields falls back to visible parser."""
    from codedoc.core.project_view import read_embedded_view

    # Valid JSON but missing 'files'
    incomplete = {"schema_version": "1.4", "project": {"entry_file": "main.py"}}
    b64 = base64.b64encode(json.dumps(incomplete).encode()).decode()
    md = f"<!-- codedoc-ai-view-base64\n{b64}\n-->\n"

    result = read_embedded_view(md)
    assert result is None, "Incomplete view (missing 'files') must be rejected"

def test_A8_dangerous_chars_in_text_safe_in_base64():
    """A8: Usage examples containing '--' or '-->' are stored safely in base64."""
    from codedoc.core.project_view import markdown_from_view, read_embedded_view

    dangerous_view = _make_view(files=[
        {
            "hash": "aaa",
            "path": "main.py",
            "language": "python",
            "description": "Has --> and -- in description.",
            "usage_example": "cmd --flag --> output\n<!-- not a comment -->",
        }
    ])

    md = markdown_from_view(dangerous_view)
    embedded = read_embedded_view(md)

    assert embedded is not None, (
        "Embedded view must decode correctly even when payload contains '-->' or '--'"
    )
    file = embedded["files"][0]
    assert "-->" in file["usage_example"]
    assert "--flag" in file["usage_example"]
    assert "<!-- not a comment -->" in file["usage_example"]

def test_read_embedded_view_returns_none_for_plain_markdown():
    """Sanity: read_embedded_view returns None when there is no base64 block."""
    from codedoc.core.project_view import read_embedded_view

    plain = "# Hello\n\nNo codedoc block here."
    assert read_embedded_view(plain) is None

def test_read_embedded_view_returns_none_for_empty_string():
    """Sanity: read_embedded_view returns None for an empty string."""
    from codedoc.core.project_view import read_embedded_view

    assert read_embedded_view("") is None

def test_embedded_markdown_preserves_private_key(private_key):
    view = _view_with_secret(private_key)
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert embedded is not None
    assert embedded["files"][0]["_secret"] == "TOPSECRET"

def test_nonzero_skip_counter_round_trips_only_through_embedded_markdown_view():
    stats = {
        **run_metadata_stats(),
        "files_selected": 7,
        "skipped_insufficient_source": 1,
    }
    view = build_project_view(run_metadata_records(), stats, entry_file="main.py")
    markdown = markdown_from_view(view)

    assert markdown_to_view(markdown) == view
    assert view["last_run"]["files_skipped_insufficient_source"] == 1
    assert "Files skipped insufficient source" not in markdown
    assert _partition_sum(view["last_run"]) == view["last_run"]["files_selected"]

def test_markdown_embeds_new_shape_without_legacy_wrappers(tmp_path):
    md = markdown_from_view(_view())
    path = tmp_path / "codedoc.md"
    path.write_text(md, encoding="utf-8")

    embedded = markdown_to_view(md)
    assert "last_run" in embedded
    assert "project" not in embedded
    assert "run" not in embedded

    doc = read_codedoc_document(path)
    assert doc.entry_file == "main.py"
    assert doc.view["last_run"]["entry_file"] == "main.py"
