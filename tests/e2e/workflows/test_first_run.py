"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_scenarios import patch_provider

def test_json_is_the_default_combined_output(tmp_path):
    from codedoc.core.output import write_project_outputs

    output_dir = tmp_path / "docs_output"
    output_dir.mkdir()
    (output_dir / "main.py.json").write_text("{}", encoding="utf-8")
    (output_dir / "main.py.md").write_text("# old", encoding="utf-8")
    (output_dir / "codedoc.md").write_text("# old combined", encoding="utf-8")

    records = [
        {
            "id": "abc123",
            "hash": "abc123",
            "file_path": "main.py",
            "format": "py",
            "language": "python",
            "last_processed": "2026-05-02T00:00:00+00:00",
            "git_commit": None,
            "author": None,
            "documentation": {
                "file_path": "main.py",
                "language": "python",
                "extension": ".py",
                "imports": ["utils"],
                "description": "Main entry point.",
                "role_in_system": "Starts the app.",
                "functions": [{"name": "main", "description": "Runs the app."}],
                "classes": [],
                "exports": [],
                "dependencies_analysis": {},
                "key_concepts": ["entry point"],
                "usage_example": "python main.py",
                "structure": {},
                "documentation": {},
                "state": "checked",
            },
        }
    ]

    json_path, md_path = write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
    )

    assert json_path == output_dir / "codedoc.json"
    assert md_path is None
    output = json_path.read_text(encoding="utf-8")
    assert '"schema_version"' not in output
    assert '"path": "main.py"' in output
    assert '"author"' not in output
    assert '"result"' not in output
    assert '"tree"' in output
    assert '"folders"' in output
    assert '"dependency_graph"' not in output
    assert '"exports": []' not in output
    assert '"dependencies_analysis": {}' not in output

def test_markdown_format_writes_only_combined_markdown(tmp_path):
    from codedoc.core.output import write_project_outputs

    output_dir = tmp_path / "docs_output"
    output_dir.mkdir()
    (output_dir / "codedoc.json").write_text("{}", encoding="utf-8")

    records = [
        {
            "id": "abc123",
            "hash": "abc123",
            "file_path": "main.py",
            "format": "py",
            "language": "python",
            "last_processed": "2026-05-02T00:00:00+00:00",
            "git_commit": None,
            "author": None,
            "documentation": {
                "file_path": "main.py",
                "language": "python",
                "description": "Main entry point.",
                "state": "checked",
            },
        }
    ]

    json_path, md_path = write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        output_format="md",
    )

    assert json_path is None
    assert md_path == output_dir / "codedoc.md"
    output = md_path.read_text(encoding="utf-8")
    assert "## Project Tree" in output
    assert "## Folder Map" in output
    assert "### main.py" in output

def test_A1_first_run_default_output(tmp_path, monkeypatch):
    """A1: first run, no flags → writes codedoc/codedoc.json only."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "propagate_changes": False,
                                     "parallel_agents": False})
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert not (tmp_path / "codedoc" / "codedoc.md").exists()
    assert stats["output_files"] == [str(tmp_path / "codedoc" / "codedoc.json")]

def test_A2_first_run_format_md(tmp_path, monkeypatch):
    """A2: first run with --format md → writes codedoc/codedoc.md only."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "md",
                                     "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert not (tmp_path / "codedoc" / "codedoc.json").exists()
    assert stats["output_files"] == [str(tmp_path / "codedoc" / "codedoc.md")]

def test_A3_first_run_format_both(tmp_path, monkeypatch):
    """A3: first run with --format both → writes codedoc.json AND codedoc.md."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    stats = run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "both",
                                     "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "codedoc" / "codedoc.json").exists()
    assert (tmp_path / "codedoc" / "codedoc.md").exists()
    assert len(stats["output_files"]) == 2

def test_A4_first_run_custom_directory(tmp_path, monkeypatch):
    """A4: --output custom_dir → writes custom_dir/codedoc.json."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "my_docs",
                             "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "my_docs" / "codedoc.json").exists()
    assert not (tmp_path / "my_docs" / "codedoc.md").exists()

def test_A5_first_run_named_json_file(tmp_path, monkeypatch):
    """A5: --output docs/report.json → writes docs/report.json."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "docs/report.json",
                             "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "docs" / "report.json").exists()
    assert not (tmp_path / "docs" / "codedoc.json").exists()

def test_A6_first_run_named_md_file(tmp_path, monkeypatch):
    """A6: --output docs/report.md → writes docs/report.md only."""
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    run_pipeline(tmp_path, {"entry_file": "main.py", "output_dir": "docs/report.md",
                             "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "docs" / "report.md").exists()
    assert not (tmp_path / "docs" / "report.json").exists()

def test_A11_conflicting_format_extension_warns_uses_extension(tmp_path, monkeypatch, caplog):
    """A11: --output report.json --format md → warning logged, JSON is written (extension wins)."""
    import logging
    patch_provider(monkeypatch)
    (tmp_path / "main.py").write_text("x=1\n")
    from codedoc.pipeline import run_pipeline
    with caplog.at_level(logging.WARNING):
        run_pipeline(tmp_path, {"entry_file": "main.py",
                                 "output_dir": "docs/report.json",
                                 "output_format": "md",
                                 "propagate_changes": False, "parallel_agents": False})
    assert (tmp_path / "docs" / "report.json").exists()
    assert any("implies format" in r.message or "takes precedence" in r.message
               for r in caplog.records)
