from __future__ import annotations

import json

from codedoc.core.markdown_view import markdown_from_view, read_embedded_view
from codedoc.core.output import write_project_outputs
from codedoc.core.project_view import build_project_view, json_from_view


def _record(path: str) -> dict:
    return {
        "hash": path,
        "file_path": path,
        "language": "python",
        "documentation": {
            "file_path": path,
            "language": "python",
            "description": path,
        },
    }


def test_reachability_is_present_for_true_and_false_records():
    view = build_project_view(
        [_record("main.py"), _record("orphan.py")],
        {"checked": 2},
        entry_file="main.py",
        reachable_rels={"main.py"},
    )
    by_path = {file["path"]: file for file in view["files"]}
    assert by_path["main.py"]["reachable_from_entry"] is True
    assert by_path["orphan.py"]["reachable_from_entry"] is False
    assert '"reachable_from_entry": false' in json_from_view(view)


def test_markdown_has_one_visible_line_per_file_and_lossless_embed():
    view = build_project_view(
        [_record("main.py"), _record("orphan.py")],
        {"checked": 2},
        entry_file="main.py",
        reachable_rels={"main.py"},
    )
    markdown = markdown_from_view(view)
    assert markdown.count("**Reachable from entry:**") == 2
    assert "**Reachable from entry:** Yes" in markdown
    assert "**Reachable from entry:** No" in markdown
    assert read_embedded_view(markdown) == view


def test_direct_view_call_defaults_all_records_to_reachable():
    view = build_project_view([_record("main.py")], {"checked": 1})
    assert view["files"][0]["reachable_from_entry"] is True


def test_write_project_outputs_propagates_reachable_rels(tmp_path):
    json_path, _ = write_project_outputs(
        [_record("main.py"), _record("orphan.py")],
        {"checked": 2},
        tmp_path,
        output_format="json",
        entry_file="main.py",
        reachable_rels={"main.py"},
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    by_path = {file["path"]: file for file in data["files"]}
    assert by_path["main.py"]["reachable_from_entry"] is True
    assert by_path["orphan.py"]["reachable_from_entry"] is False


def test_write_project_outputs_defaults_to_reachable_for_direct_callers(tmp_path):
    json_path, _ = write_project_outputs(
        [_record("main.py")],
        {"checked": 1},
        tmp_path,
        output_format="json",
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["files"][0]["reachable_from_entry"] is True


def test_write_project_outputs_builds_project_view_once(tmp_path, monkeypatch):
    import codedoc.core.output as output_module

    calls = []
    real_builder = output_module.build_project_view

    def counting_builder(*args, **kwargs):
        calls.append((args, kwargs))
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(output_module, "build_project_view", counting_builder)
    write_project_outputs(
        [_record("main.py")],
        {"checked": 1},
        tmp_path,
        output_format="both",
        reachable_rels={"main.py"},
    )
    assert len(calls) == 1
