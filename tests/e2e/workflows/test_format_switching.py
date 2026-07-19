"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
import json
from tests.support.pipeline_scenarios import no_llm
from tests.support.pipeline_scenarios import write_existing_json
from tests.support.pipeline_scenarios import write_existing_md
from tests.support.markdown_cases import _fake_provider

def test_pipeline_cached_run_honors_markdown_format(tmp_path, monkeypatch):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    docs_output = tmp_path / "docs_output"
    docs_output.mkdir()

    file_hash = compute_file_hash(main)

    write_project_outputs(
        [{
            "file_path": "main.py",
            "hash": file_hash,
            "language": "python",
            "documentation": {"description": "Main entry point."},
            **_PRIOR_RUN_IDENTITY,
        }],
        {"checked": 1, "failed": 0, "skipped": 0},
        docs_output,
        output_format="md",
        entry_file="main.py",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached markdown output")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "output_dir": "docs_output",
            "output_format": "md",
        },
    )

    md_path = tmp_path / "docs_output" / "codedoc.md"
    assert stats["checked"] == 0
    assert stats["output_files"] == [str(md_path)]
    assert md_path.exists()
    assert "Main entry point." in md_path.read_text(encoding="utf-8")

def test_pipeline_cached_run_can_switch_back_to_json(tmp_path, monkeypatch):

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    docs = tmp_path / "docs_output"
    docs.mkdir()

    file_hash = compute_file_hash(main)

    # Pre-write a valid public JSON (JSON format) so docs can be recovered.
    (docs / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "main.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "main.py",
                    "hash": file_hash,
                    "description": "Main entry point.",
                    "language": "python",
                    "format": "py",
                    **_PRIOR_RUN_IDENTITY,
                }
            ],
        }),
        encoding="utf-8",
    )
    (docs / "codedoc.md").write_text("# old", encoding="utf-8")

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached json output")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "output_dir": "docs_output",
            "output_format": "json",
        },
    )

    json_path = tmp_path / "docs_output" / "codedoc.json"
    assert stats["checked"] == 0
    assert stats["output_files"] == [str(json_path)]
    assert json_path.exists()
    assert "Main entry point." in json_path.read_text(encoding="utf-8")

def test_C1_switch_json_to_md(tmp_path, monkeypatch):
    """An unchanged JSON sibling converts to Markdown without paid work."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "JSON switch.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert (tmp_path / "codedoc" / "codedoc.json").exists()  # NOT deleted

def test_C2_switch_md_to_json(tmp_path, monkeypatch):
    """An unchanged Markdown sibling converts to JSON without paid work."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_md(tmp_path / "codedoc" / "codedoc.md",
                      compute_file_hash(src), "MD switch.")
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()  # NOT deleted

def test_C3_switch_both_to_json(tmp_path, monkeypatch):
    """C3: have both codedoc.json + codedoc.md, run --format json → reads JSON, MD stays."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Both to JSON.")
    write_existing_md(tmp_path / "codedoc" / "codedoc.md",
                      compute_file_hash(src), "Both to JSON.")
    no_llm(monkeypatch)
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                             "propagate_changes": False})
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()

def test_C4_switch_both_to_md(tmp_path, monkeypatch):
    """A missing Markdown target reuses the existing JSON counterpart."""
    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline
    src = tmp_path / "main.py"
    src.write_text("x=1\n")
    write_existing_json(tmp_path / "codedoc" / "codedoc.json",
                        compute_file_hash(src), "Both to MD.")
    no_llm(monkeypatch)
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False})
    assert stats["checked"] == 0
    assert (tmp_path / "codedoc" / "codedoc.md").exists()

def _run_pipeline(tmp_path, monkeypatch, **cfg):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: _fake_provider())
    from codedoc.pipeline import run_pipeline
    defaults = {"parallel_agents": False, "propagate_changes": False}
    defaults.update(cfg)
    return run_pipeline(tmp_path, defaults)

def test_A15_clean_md_run_writes_markdown_and_removes_json(tmp_path, monkeypatch):
    """A15: Clean MD-only run produces Markdown with embedded view and no JSON sibling."""
    (tmp_path / "main.py").write_text("x=1\n")

    _run_pipeline(tmp_path, monkeypatch, entry_file="main.py", output_format="md")

    md_path = tmp_path / "codedoc" / "codedoc.md"
    json_sibling = tmp_path / "codedoc" / "codedoc.json"

    assert md_path.exists(), "codedoc.md must be written"
    assert not json_sibling.exists(), "JSON sibling must be removed after clean MD run"

    # Markdown must have both the legacy comment and the new base64 block
    md_content = md_path.read_text(encoding="utf-8")
    assert "<!-- codedoc-ai:" in md_content
    assert "<!-- codedoc-ai-view-base64" in md_content

def test_A15_md_run_embedded_view_passes_all_validations(tmp_path, monkeypatch):
    """A15b: Embedded view in a pipeline-generated Markdown is valid and decodable."""
    from codedoc.core.project_view import read_embedded_view

    (tmp_path / "main.py").write_text("x=1\n")
    _run_pipeline(tmp_path, monkeypatch, entry_file="main.py", output_format="md")

    md_path = tmp_path / "codedoc" / "codedoc.md"
    md_content = md_path.read_text(encoding="utf-8")

    embedded = read_embedded_view(md_content)
    assert embedded is not None, "Embedded view must be present and valid after pipeline run"
    assert "_crash_safety" not in embedded
    assert "files" in embedded
    assert len(embedded["files"]) >= 1
