"""Tests organized by feature ownership."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from codedoc.core.markdown_view import (
    markdown_from_view,
    read_embedded_view,
)
from codedoc.core.project_view import build_project_view, json_from_view
from tests.support.fixture_paths import GOLDEN_DOCUMENT_FIXTURES
from tests.support.project_view_cases import _build_view
from tests.support.deterministic_records import _records as deterministic_records

_REPO_ROOT = Path(__file__).resolve().parents[3]

CANONICAL_PRIVATE_KEYS = [
    "_analysis_revision",
    "_analysis_mode",
    "_max_context_revision",
    "_prompt_profile_digest",
]

# Fixed seeds, each of which produces a distinct non-canonical frozenset order
# on the pre-fix carrier.  They make the determinism proof reproducible instead
# of depending on the parent interpreter's ambient hash seed.
_HASH_SEEDS = ("0", "1", "2", "7", "12345")

def _scrambled_private_key_record() -> dict:
    """One record whose private keys are deliberately NOT in canonical order.

    A canonical-order source would pass even if the carrier simply followed the
    source dictionary, so the ordering contract must be driven from a scrambled
    one.
    """
    return {
        "hash": "h-main",
        "file_path": "main.py",
        "language": "python",
        "_prompt_profile_digest": "profile-digest",
        "_max_context_revision": "truncate-v1:max=10:head=0.7000",
        "_analysis_mode": "single",
        "_analysis_revision": "file-doc-v3",
        "documentation": {
            "description": "Entry point.",
            "dependencies_analysis": {},
        },
    }

def _private_keys_of(file_record: dict) -> list:
    return [
        key
        for key in file_record
        if isinstance(key, str) and key.startswith("_") and key != "_deps"
    ]

# Runs in a child interpreter under an explicit PYTHONHASHSEED.  It imports only
# from ``codedoc`` and builds its own record, so the child never needs the
# ``tests`` package to be importable.
_DETERMINISM_CHILD = """
import json
from codedoc.core.markdown_view import markdown_from_view
from codedoc.core.project_view import build_project_view, json_from_view

record = {
    "hash": "h-main",
    "file_path": "main.py",
    "language": "python",
    "_prompt_profile_digest": "profile-digest",
    "_max_context_revision": "truncate-v1:max=10:head=0.7000",
    "_analysis_mode": "single",
    "_analysis_revision": "file-doc-v3",
    "documentation": {"description": "Entry point.", "dependencies_analysis": {}},
}
view = build_project_view([record], {"checked": 1}, entry_file="main.py")
text = json_from_view(view)
markdown = markdown_from_view(view)
keys = [
    key
    for key in json.loads(text)["files"][0]
    if key.startswith("_") and key != "_deps"
]
print(json.dumps({"keys": keys, "json": text, "markdown": markdown}))
"""

def _emit_under_seed(seed: str) -> dict:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_CHILD],
            cwd=str(_REPO_ROOT),
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None)
        stderr = getattr(exc, "stderr", None)
        raise AssertionError(
            f"determinism child failed for PYTHONHASHSEED={seed!r}; "
            f"stdout={stdout!r}; stderr={stderr!r}"
        ) from exc
    return json.loads(completed.stdout)

def _read_golden(name: str) -> str:
    return (GOLDEN_DOCUMENT_FIXTURES / name).read_text(encoding="utf-8")

def _without_reachability(view: dict) -> dict:
    return {
        **view,
        "files": [
            {key: value for key, value in file.items() if key != "reachable_from_entry"}
            for file in view.get("files", [])
        ],
    }

def _without_visible_reachability(markdown: str) -> str:
    return markdown.replace("**Reachable from entry:** Yes  \n\n", "").replace(
        "**Reachable from entry:** No  \n\n", ""
    )

def test_view_assembly_byte_identical():
    view = _without_reachability(_build_view())
    assert json.dumps(view, indent=2, ensure_ascii=False) == _read_golden(
        "project_view.json"
    )

def test_json_serialization_byte_identical():
    view = _without_reachability(_build_view())
    assert json_from_view(view, "No errors.") == _read_golden("completed_output.json")

def test_markdown_serialization_byte_identical():
    view = _without_reachability(_build_view())
    rendered = markdown_from_view(view, "Sample error\nsecond line")
    assert _without_visible_reachability(rendered) == _read_golden(
        "completed_output.md"
    )


def test_legacy_golden_bytes_remain_free_of_optional_split_fields():
    for name in (
        "project_view.json",
        "completed_output.json",
        "completed_output.md",
    ):
        golden = _read_golden(name)
        assert "documentation_units" not in golden
        assert '"division"' not in golden
        assert "_large_file_identity" not in golden

def test_empty_view_json_byte_identical():
    empty = build_project_view([], {"checked": 0}, entry_file=None, graph_edges=[])
    assert json_from_view(empty) == _read_golden("empty_output.json")

def test_empty_view_markdown_byte_identical():
    empty = build_project_view([], {"checked": 0}, entry_file=None, graph_edges=[])
    assert markdown_from_view(empty) == _read_golden("empty_output.md")

def test_completed_view_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    assert "generated_at" not in view

def test_completed_json_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    text = json_from_view(view)
    assert "generated_at" not in text

def test_completed_markdown_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    md = markdown_from_view(view)
    assert "generated_at" not in md

def test_embedded_view_has_no_generated_at():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    md = markdown_from_view(view)
    embedded = read_embedded_view(md)
    assert "generated_at" not in embedded

def test_json_omits_generated_at_even_from_legacy_view():
    view = build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py")
    view["generated_at"] = "2026-01-01T00:00:00+00:00"  # simulate a legacy caller view
    text = json_from_view(view)
    payload = json.loads(text)
    assert "generated_at" not in payload
    assert "generated_at" not in payload.get("_codedoc", {})

def test_two_runs_produce_byte_identical_json():
    a = json_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    b = json_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    assert a == b

def test_two_runs_produce_byte_identical_markdown():
    a = markdown_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    b = markdown_from_view(build_project_view(deterministic_records(), {"checked": 2}, entry_file="main.py"))
    assert a == b

def test_json_private_keys_use_canonical_order():
    view = build_project_view(
        [_scrambled_private_key_record()], {"checked": 1}, entry_file="main.py"
    )
    payload = json.loads(json_from_view(view))
    assert _private_keys_of(payload["files"][0]) == CANONICAL_PRIVATE_KEYS

def test_embedded_view_private_keys_use_canonical_order():
    view = build_project_view(
        [_scrambled_private_key_record()], {"checked": 1}, entry_file="main.py"
    )
    embedded = read_embedded_view(markdown_from_view(view))
    assert _private_keys_of(embedded["files"][0]) == CANONICAL_PRIVATE_KEYS

def test_private_keys_are_canonically_ordered_across_hash_seeds():
    emitted = {seed: _emit_under_seed(seed) for seed in _HASH_SEEDS}

    for seed, result in emitted.items():
        assert result["keys"] == CANONICAL_PRIVATE_KEYS, (
            f"PYTHONHASHSEED={seed} emitted {result['keys']}"
        )

    first = emitted[_HASH_SEEDS[0]]
    for seed in _HASH_SEEDS[1:]:
        assert emitted[seed]["json"] == first["json"]
        assert emitted[seed]["markdown"] == first["markdown"]
