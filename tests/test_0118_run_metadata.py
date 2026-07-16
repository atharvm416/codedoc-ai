"""0.11.8 run metadata accuracy and compatibility coverage."""

from __future__ import annotations

import copy
import json

import pytest

from codedoc.core.discovery import _resolve_entry_and_docs
from codedoc.core.document import (
    _LAST_RUN_INTEGER_FIELDS,
    _LAST_RUN_OPTIONAL_INTEGER_FIELDS,
    read_codedoc_document,
)
from codedoc.core.markdown_view import markdown_from_view, markdown_to_view
from codedoc.core.output import _is_codedoc_owned, write_summary
from codedoc.core.planning import PipelinePlan
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.core.record_meta import PRIVATE_RECORD_KEYS
from codedoc.pipeline import _final_entry_source, _set_plan_counters
from codedoc.utils.errors import ConfigError

_REQUIRED_LAST_RUN_INTEGER_FIELDS = {
    "files_scanned",
    "files_selected",
    "files_documented_by_llm",
    "files_failed",
    "files_unattempted",
    "files_reused_unchanged",
    "files_reused_identical_content",
    "files_resumed_from_recovery",
}


def _records() -> list[dict]:
    return [
        {
            "hash": "h-main",
            "file_path": "main.py",
            "language": "python",
            "_analysis_revision": "file-doc-v3",
            "_analysis_mode": "single",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_prompt_profile_digest": "no-profile",
            "documentation": {
                "description": "Entry point.",
                "dependencies_analysis": {"external": ["requests"]},
            },
        },
        {
            "hash": "h-utils",
            "file_path": "utils.py",
            "language": "python",
            "documentation": {"description": "Utilities."},
        },
    ]


def _stats() -> dict:
    return {
        "checked": 1,
        "failed": 1,
        "skipped": 2,
        "reused": 1,
        "resumed": 1,
        "analysis_mode": "single",
        "entry_source": "explicit",
        "documentation_scope": "entry",
        "files_scanned": 7,
        "files_selected": 6,
        "unattempted_files": 1,
    }


def _view() -> dict:
    return build_project_view(_records(), _stats(), entry_file="main.py")


def _legacy_data() -> dict:
    data = json.loads(json_from_view(_view()))
    data["_codedoc"] = {"entry_file": "main.py"}
    data["project"] = {
        "entry_file": "main.py",
        "file_count": 2,
        "languages": ["python"],
        "folders": ["."],
    }
    data["run"] = {
        "files_checked": 1,
        "files_failed": 1,
        "files_skipped": 2,
        "files_reused": 1,
        "files_documented": 2,
    }
    data["last_run"].pop("entry_file", None)
    return data


def test_last_run_is_truthful_and_legacy_wrappers_are_removed():
    view = _view()

    assert "run" not in view
    assert "project" not in view
    assert view["last_run"] == {
        "entry_file": "main.py",
        "entry_source": "explicit",
        "documentation_scope": "entry",
        "analysis_mode": "single",
        "files_scanned": 7,
        "files_selected": 6,
        "files_documented_by_llm": 1,
        "files_failed": 1,
        "files_unattempted": 1,
        "files_skipped_insufficient_source": 0,
        "files_reused_unchanged": 2,
        "files_reused_identical_content": 1,
        "files_resumed_from_recovery": 1,
    }
    assert (
        view["last_run"]["files_selected"]
        == view["last_run"]["files_reused_unchanged"]
        + view["last_run"]["files_reused_identical_content"]
        + view["last_run"]["files_documented_by_llm"]
        + view["last_run"]["files_failed"]
        + view["last_run"]["files_unattempted"]
        + view["last_run"]["files_skipped_insufficient_source"]
    )
    assert len(view["files"]) < view["last_run"]["files_selected"]
    assert view["last_run"]["files_resumed_from_recovery"] <= view["last_run"][
        "files_reused_unchanged"
    ]


def test_markdown_and_summary_render_truthful_labels(tmp_path):
    md = markdown_from_view(_view())
    summary = write_summary(_stats(), tmp_path).read_text(encoding="utf-8")

    for text in (md, summary):
        assert "Files reused from cache" not in text
        assert "Files documented by LLM: 1" in text
        assert "Files reused (unchanged): 2" in text
        assert "Files reused (identical content): 1" in text
        assert "Files resumed from recovery: 1" in text


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


def test_json_reader_accepts_valid_last_run(tmp_path):
    path = tmp_path / "codedoc.json"
    path.write_text(json_from_view(_view()), encoding="utf-8")

    doc = read_codedoc_document(path)

    assert doc.view["last_run"]["files_selected"] == 6
    assert doc.entry_file == "main.py"
    assert "run" not in doc.view
    assert "project" not in doc.view
    assert "_codedoc" not in doc.view


def test_skip_counter_is_additive_optional_on_read(tmp_path):
    assert _LAST_RUN_INTEGER_FIELDS == _REQUIRED_LAST_RUN_INTEGER_FIELDS
    assert "files_skipped_insufficient_source" not in _LAST_RUN_INTEGER_FIELDS
    assert _LAST_RUN_OPTIONAL_INTEGER_FIELDS == {
        "files_skipped_insufficient_source"
    }

    data = json.loads(json_from_view(_view()))
    data["last_run"].pop("files_skipped_insufficient_source")
    path = tmp_path / "without-additive-counter.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    document = read_codedoc_document(path)
    assert "files_skipped_insufficient_source" not in document.view["last_run"]
    assert _partition_sum(document.view["last_run"]) == document.view["last_run"][
        "files_selected"
    ]


def test_nonzero_skip_counter_round_trips_only_through_embedded_markdown_view():
    stats = {
        **_stats(),
        "files_selected": 7,
        "skipped_insufficient_source": 1,
    }
    view = build_project_view(_records(), stats, entry_file="main.py")
    markdown = markdown_from_view(view)

    assert markdown_to_view(markdown) == view
    assert view["last_run"]["files_skipped_insufficient_source"] == 1
    assert "Files skipped insufficient source" not in markdown
    assert _partition_sum(view["last_run"]) == view["last_run"]["files_selected"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_json_reader_rejects_malformed_optional_skip_counter(tmp_path, value):
    data = json.loads(json_from_view(_view()))
    data["last_run"]["files_skipped_insufficient_source"] = value
    path = tmp_path / "malformed-optional-counter.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError):
        read_codedoc_document(path)


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("negative", lambda lr: lr.update({"files_failed": -1})),
        ("bool", lambda lr: lr.update({"files_failed": True})),
        ("missing", lambda lr: lr.pop("files_selected")),
        ("entry_source", lambda lr: lr.update({"entry_source": "mystery"})),
        ("scope", lambda lr: lr.update({"documentation_scope": "wide"})),
        ("mode", lambda lr: lr.update({"analysis_mode": "quad"})),
        ("partition", lambda lr: lr.update({"files_selected": 99})),
    ],
)
def test_json_reader_rejects_malformed_last_run(tmp_path, label, mutate):
    data = json.loads(json_from_view(_view()))
    mutate(data["last_run"])
    path = tmp_path / f"{label}.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError):
        read_codedoc_document(path)


def test_underscore_file_keys_are_limited_to_registered_private_keys_and_deps():
    view = _view()
    underscored = {
        key
        for file in view["files"]
        for key in file
        if isinstance(key, str) and key.startswith("_")
    }

    assert underscored == set(PRIVATE_RECORD_KEYS) | {"_deps"}


def test_unknown_envelope_keys_are_tolerated(tmp_path):
    data = _legacy_data()
    data["_codedoc"]["future_key"] = {"ok": True}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert read_codedoc_document(path).metadata["future_key"] == {"ok": True}


def test_envelope_entry_file_cross_check_fails_closed(tmp_path):
    data = _legacy_data()
    broken = copy.deepcopy(data)
    broken["_codedoc"]["entry_file"] = "other.py"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")

    with pytest.raises(ConfigError):
        read_codedoc_document(path)


# ---------------------------------------------------------------------------
# Partition invariant across every run shape (D3/D7).
# ---------------------------------------------------------------------------

def _partition_sum(last_run: dict) -> int:
    return (
        last_run["files_reused_unchanged"]
        + last_run["files_reused_identical_content"]
        + last_run["files_documented_by_llm"]
        + last_run["files_failed"]
        + last_run["files_unattempted"]
        + last_run.get("files_skipped_insufficient_source", 0)
    )


@pytest.mark.parametrize(
    ("shape", "counts", "selected"),
    [
        # checked, failed, skipped, reused, unattempted
        ("all_reused", (0, 0, 5, 0, 0), 5),
        ("fresh_full", (5, 0, 0, 0, 0), 5),
        ("mixed", (3, 0, 2, 1, 0), 6),
        ("with_failures", (2, 2, 0, 0, 0), 4),
        ("health_check_abort", (2, 1, 0, 0, 2), 5),
    ],
)
def test_partition_invariant_holds_for_every_run_shape(shape, counts, selected):
    checked, failed, skipped, reused, unattempted = counts
    stats = {
        "checked": checked,
        "failed": failed,
        "skipped": skipped,
        "reused": reused,
        "resumed": 0,
        "files_scanned": selected,
        "files_selected": selected,
        "unattempted_files": unattempted,
    }
    view = build_project_view(_records(), stats, entry_file="main.py")
    assert view["last_run"]["files_selected"] == _partition_sum(view["last_run"]) == selected


def _plan(scanned, selected, agent):
    return PipelinePlan(
        scanned_rels=frozenset(scanned),
        documented_rels=frozenset(selected),
        changed_rels=frozenset(),
        forced_rels=frozenset(),
        process_rels=frozenset(agent),
        unchanged_rels=frozenset(),
        identical_reuse_rels=frozenset(),
        agent_rels=frozenset(agent),
        entry_rel=None,
        max_files=0,
        max_files_exceeded=False,
    )


def test_set_plan_counters_computes_unattempted_and_keeps_partition(tmp_path):
    # 10 scanned, 8 selected, 5 routed to the agent; 3 checked + 1 failed leaves
    # 1 unattempted (the health-check / early-abort case D7 targets).
    plan = _plan(
        scanned=[f"f{i}.py" for i in range(10)],
        selected=[f"f{i}.py" for i in range(8)],
        agent=[f"f{i}.py" for i in range(5)],
    )
    stats = {"checked": 3, "failed": 1, "skipped": 2, "reused": 1, "resumed": 0}
    _set_plan_counters(stats, plan)

    assert stats["files_scanned"] == 10
    assert stats["files_selected"] == 8
    assert stats["unattempted_files"] == 1

    view = build_project_view(_records(), stats, entry_file="main.py")
    assert view["last_run"]["files_unattempted"] == 1
    assert view["last_run"]["files_selected"] == _partition_sum(view["last_run"]) == 8


def test_resumed_is_a_subset_and_must_not_be_summed_into_the_partition():
    # _stats(): resumed=1 is already inside reused_unchanged=2.  The true sum equals
    # files_selected; naively adding resumed overcounts — this pins the invariant so
    # nobody "fixes" it by summing resumed in (D5).
    lr = _view()["last_run"]
    assert lr["files_resumed_from_recovery"] <= lr["files_reused_unchanged"]
    assert lr["files_selected"] == _partition_sum(lr)
    assert _partition_sum(lr) + lr["files_resumed_from_recovery"] > lr["files_selected"]


# ---------------------------------------------------------------------------
# entry_source coverage: all four values, and the None <-> "none" equivalence (D10).
# ---------------------------------------------------------------------------

def test_entry_source_none_exactly_when_no_project_entry():
    stats = {
        "checked": 2,
        "failed": 0,
        "skipped": 0,
        "reused": 0,
        "resumed": 0,
        "files_scanned": 2,
        "files_selected": 2,
    }
    view = build_project_view(_records(), stats, entry_file=None)
    assert view["last_run"]["entry_file"] is None
    assert view["last_run"]["entry_source"] == "none"
    assert view["last_run"]["documentation_scope"] == "all"


def test_resolve_entry_source_recovered_from_prior_document(tmp_path):
    # A prior completed codedoc.json carrying an entry, and no --entry supplied,
    # resolves as "recovered" and writes the entry back into config.
    (tmp_path / "codedoc.json").write_text(
        json_from_view(build_project_view(_records(), _stats(), entry_file="main.py")),
        encoding="utf-8",
    )
    config = {"output_dir": str(tmp_path), "output_format": "json"}

    source = _resolve_entry_and_docs(tmp_path, config)

    assert source == "recovered"
    assert config["entry_file"] == "main.py"
    # The pipeline keeps "recovered" verbatim regardless of later auto-detection.
    assert _final_entry_source(source, "main.py") == "recovered"


def test_final_entry_source_maps_pending_to_auto_detected_or_none():
    assert _final_entry_source("pending", "main.py") == "auto-detected"
    assert _final_entry_source("pending", None) == "none"
    assert _final_entry_source("explicit", None) == "explicit"


# ---------------------------------------------------------------------------
# Round trip and legacy read (compatibility window).
# ---------------------------------------------------------------------------

def test_last_run_round_trips_through_markdown_and_json():
    view = _view()

    # Markdown -> view is lossless through the embedded base64 block.
    back_md = markdown_to_view(markdown_from_view(view))
    assert back_md["last_run"] == view["last_run"]

    # JSON serialise/parse preserves last_run byte-for-byte in the payload.
    assert json.loads(json_from_view(view))["last_run"] == view["last_run"]


def test_legacy_json_without_last_run_is_accepted_and_owned(tmp_path):
    # A 0.11.6 document: has `run`, has no `last_run`.  It must still be owned and
    # readable so a newer build can safely overwrite it.
    data = _legacy_data()
    data.pop("last_run")
    path = tmp_path / "codedoc.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert _is_codedoc_owned(path)
    doc = read_codedoc_document(path)
    assert "last_run" not in doc.view
    assert doc.view["run"]["files_documented"] == 2


def test_current_json_without_envelope_is_owned(tmp_path):
    data = json.loads(json_from_view(_view()))
    path = tmp_path / "current.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert _is_codedoc_owned(path)
    assert read_codedoc_document(path).entry_file == "main.py"


def test_empty_legacy_envelope_fails_closed(tmp_path):
    data = json.loads(json_from_view(_view()))
    data["_codedoc"] = {}
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert not _is_codedoc_owned(path)
    with pytest.raises(ConfigError):
        read_codedoc_document(path)
