"""0.11.5 regression coverage for versionless user-facing data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codedoc.agents.response_cleaning import clean_combined_response
from codedoc.agents.dependency_agent import build_prompt as build_dependency_prompt
from codedoc.agents.documentation_agent import build_prompt as build_documentation_prompt
from codedoc.agents.file_documentation_agent import build_prompt as build_combined_prompt
from codedoc.agents.structure_agent import build_prompt as build_structure_prompt
from codedoc.core.config_template import build_default_config, init_config
from codedoc.core.document import read_codedoc_document
from codedoc.core.markdown_view import (
    json_from_markdown,
    markdown_from_view,
    read_embedded_view,
)
from codedoc.core.project_view import build_project_view, json_from_view
from codedoc.core.prompt_profiles import default_prompt_profiles, validate_profile
from codedoc.core.safe_writer import SafeWriter
from codedoc.utils.errors import ConfigError


def _assert_versionless(value: object) -> None:
    if isinstance(value, dict):
        assert "schema_version" not in value
        for item in value.values():
            _assert_versionless(item)
    elif isinstance(value, list):
        for item in value:
            _assert_versionless(item)


def _view() -> dict:
    record = {
        "hash": "h1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {
            "description": "Entry point.",
            "language": "python",
        },
    }
    return build_project_view(
        [record], {"checked": 1}, entry_file="main.py", graph_edges=[]
    )


def _validate(raw: dict):
    return validate_profile(
        raw,
        active_mode="single",
        known_extensions=frozenset({".py"}),
        source="inline",
        source_path=None,
    )


def test_generated_and_forced_config_are_versionless_and_round_trip(tmp_path):
    generated = build_default_config()
    _assert_versionless(generated["prompt_profiles"])
    _validate(generated["prompt_profiles"])

    target = tmp_path / "codedoc.config.json"
    target.write_text(
        json.dumps({**generated, "output_format": "md", "prompt_profiles": None}),
        encoding="utf-8",
    )
    init_config(tmp_path, force=True)
    refreshed = json.loads(target.read_text(encoding="utf-8"))
    assert refreshed["output_format"] == "md"
    _assert_versionless(refreshed["prompt_profiles"])


@pytest.mark.parametrize("version", [1, 2])
def test_explicit_legacy_profile_versions_remain_readable(version):
    raw = default_prompt_profiles("single", schema_version=version)
    _assert_versionless(raw)
    raw["schema_version"] = version
    assert raw["schema_version"] == version
    assert _validate(raw).schema_version == version


def test_mixed_versionless_profile_syntax_fails_closed():
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["fields"] = []
    with pytest.raises(ConfigError, match="both 'fields'.*'requested_shape'"):
        _validate(raw)


def test_final_json_markdown_and_round_trip_are_recursively_versionless(tmp_path):
    view = _view()
    json_data = json.loads(json_from_view(view))
    markdown = markdown_from_view(view)
    embedded = read_embedded_view(markdown)
    regenerated = json.loads(json_from_markdown(markdown))

    _assert_versionless(view)
    _assert_versionless(json_data)
    _assert_versionless(embedded)
    _assert_versionless(regenerated)
    assert "schema_version" not in markdown
    assert {k: v for k, v in json_data.items() if k != "_codedoc"} == {
        k: v for k, v in regenerated.items() if k != "_codedoc"
    }

    json_path = tmp_path / "codedoc.json"
    md_path = tmp_path / "codedoc.md"
    json_path.write_text(json_from_view(view), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    assert read_codedoc_document(json_path).schema_version is None
    assert read_codedoc_document(md_path).schema_version is None


def test_recovery_and_cleaned_provider_response_are_versionless(tmp_path):
    recovery = tmp_path / "crash_recovery.json"
    writer = SafeWriter(recovery, "json", "main.py", {})
    writer.initialize_empty()
    recovery_data = json.loads(recovery.read_text(encoding="utf-8"))
    _assert_versionless(recovery_data)
    assert read_codedoc_document(recovery).in_progress is True

    cleaned = clean_combined_response(
        {"description": "Useful.", "schema_version": "unsolicited"}, "main.py"
    )
    assert cleaned == {"description": "Useful."}


def test_provider_requested_shapes_are_versionless():
    prompts = [
        build_combined_prompt("main.py", "x = 1", [], "python"),
        build_structure_prompt("main.py", "x = 1", [], "python"),
        build_dependency_prompt("main.py", "x = 1", [], "python"),
        build_documentation_prompt("main.py", "x = 1", "python", {}, {}),
    ]
    for system, prompt in prompts:
        assert "schema_version" not in system
        assert "schema_version" not in prompt


def test_supported_versioned_legacy_documents_remain_readable():
    fixtures = Path(__file__).parent / "fixtures"
    assert read_codedoc_document(fixtures / "legacy_codedoc_13.json").schema_version == "1.3"
    assert read_codedoc_document(fixtures / "legacy_codedoc_14.md").schema_version == "1.4"


def test_versionless_foreign_or_contradictory_json_is_rejected(tmp_path):
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"_codedoc": {}, "files": []}), encoding="utf-8")
    with pytest.raises(ConfigError, match="canonical"):
        read_codedoc_document(foreign)

    data = json.loads(json_from_view(_view()))
    data["_codedoc"] = {"entry_file": "main.py"}
    data["last_run"]["entry_file"] = "other.py"
    foreign.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="contradictory"):
        read_codedoc_document(foreign)
