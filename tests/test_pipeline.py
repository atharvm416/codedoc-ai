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
    assert not (output_dir / "main.py.json").exists()
    assert not (output_dir / "main.py.md").exists()
    assert not (output_dir / "codedoc.md").exists()
    output = json_path.read_text(encoding="utf-8")
    assert '"schema_version": "1.3"' in output
    assert '"path": "main.py"' in output
    assert '"author"' not in output
    assert '"result"' not in output
    assert '"tree"' in output
    assert '"folders"' in output
    assert '"dependency_graph"' in output


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
    assert not (output_dir / "codedoc.json").exists()
    output = md_path.read_text(encoding="utf-8")
    assert "## Project Tree" in output
    assert "## Folder Map" in output
    assert "### main.py" in output


def test_db_reuses_cached_documentation_for_unchanged_files(tmp_path):
    from codedoc.core.db import CodeDocDB

    source = tmp_path / "main.py"
    source.write_text("def main():\n    pass\n", encoding="utf-8")

    db = CodeDocDB(tmp_path)
    result = {
        "file_path": "main.py",
        "language": "python",
        "extension": ".py",
        "imports": [],
        "description": "Main entry point.",
        "state": "checked",
    }
    db.mark_processed("main.py", source, result)

    assert db.needs_processing("main.py", source) is False

    records = db.documentation_records(
        {"main.py"},
        {"main.py": {"extension": ".py", "language": "python"}},
        ["main.py"],
    )
    assert records[0]["file_path"] == "main.py"
    assert records[0]["format"] == "py"
    assert records[0]["documentation"]["description"] == "Main entry point."

    source.write_text("def main():\n    return 1\n", encoding="utf-8")
    assert db.needs_processing("main.py", source) is True


def test_pipeline_reuses_identical_file_content_without_llm(tmp_path, monkeypatch):
    from codedoc.core.db import CodeDocDB
    from codedoc.pipeline import run_pipeline

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    content = "def shared():\n    return 1\n"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    db = CodeDocDB(tmp_path)
    db.mark_processed(
        "first.py",
        first,
        {
            "file_path": "first.py",
            "language": "python",
            "extension": ".py",
            "imports": [],
            "description": "Shared helper.",
            "state": "checked",
        },
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for identical cached content")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["reused"] == 1
    output = (tmp_path / "docs_output" / "codedoc.json").read_text(encoding="utf-8")
    assert '"path": "first.py"' in output
    assert '"path": "second.py"' in output
    assert '"description": "Shared helper."' in output

    (tmp_path / "docs_output" / "codedoc.json").unlink()
    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["reused"] == 0
    assert (tmp_path / "docs_output" / "codedoc.json").exists()


def test_pipeline_cached_run_honors_markdown_format(tmp_path, monkeypatch):
    from codedoc.core.db import CodeDocDB
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    db = CodeDocDB(tmp_path)
    db.mark_processed(
        "main.py",
        main,
        {
            "file_path": "main.py",
            "language": "python",
            "extension": ".py",
            "imports": [],
            "description": "Main entry point.",
            "state": "checked",
        },
    )

    (tmp_path / "docs_output").mkdir()
    (tmp_path / "docs_output" / "codedoc.json").write_text("{}", encoding="utf-8")

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
    assert not (tmp_path / "docs_output" / "codedoc.json").exists()
    assert "Main entry point." in md_path.read_text(encoding="utf-8")


def test_public_output_contains_tree_folders_and_dependency_graph(tmp_path):
    import json

    from codedoc.core.output import write_project_outputs

    output_dir = tmp_path / "docs_output"
    records = [
        {
            "id": "main-hash",
            "hash": "main-hash",
            "file_path": "src/main.tsx",
            "format": "tsx",
            "language": "tsx",
            "last_processed": "2026-05-02T00:00:00+00:00",
            "author": "Should Not Leak",
            "documentation": {
                "file_path": "src/main.tsx",
                "language": "tsx",
                "imports": ["./router"],
                "description": "Starts the frontend app.",
                "role_in_system": "Application entry.",
                "functions": [],
                "classes": [],
                "exports": ["App"],
                "dependencies_analysis": {"external": ["react"]},
                "key_concepts": ["rendering"],
                "state": "checked",
            },
        },
        {
            "id": "router-hash",
            "hash": "router-hash",
            "file_path": "src/router.tsx",
            "format": "tsx",
            "language": "tsx",
            "last_processed": "2026-05-02T00:00:00+00:00",
            "documentation": {
                "file_path": "src/router.tsx",
                "language": "tsx",
                "imports": [],
                "description": "Defines routes.",
                "role_in_system": "Routes application screens.",
                "functions": [],
                "classes": [],
                "exports": ["Router"],
                "dependencies_analysis": {"external": ["react-router-dom"]},
                "key_concepts": ["routing"],
                "state": "checked",
            },
        },
    ]

    json_path, md_path = write_project_outputs(
        records,
        {"checked": 2, "failed": 0, "skipped": 0, "reused": 0},
        output_dir,
        output_format="both",
        entry_file="src/main.tsx",
        graph_edges=[
            {
                "from": "src/main.tsx",
                "to": "src/router.tsx",
                "type": "internal_import",
            }
        ],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["project"]["entry_file"] == "src/main.tsx"
    assert payload["tree"]["src"]["main.tsx"]["type"] == "file"
    assert payload["folders"][0]["path"] == "src"
    assert payload["dependency_graph"] == [
        {
            "from": "src/main.tsx",
            "to": "src/router.tsx",
            "type": "internal_import",
        }
    ]
    assert payload["files"][0]["links"]["internal_dependencies"] == ["src/router.tsx"]
    assert payload["files"][1]["links"]["imported_by"] == ["src/main.tsx"]
    assert "author" not in json_path.read_text(encoding="utf-8")
    assert "result" not in json_path.read_text(encoding="utf-8")

    markdown = md_path.read_text(encoding="utf-8")
    assert "## Project Tree" in markdown
    assert "src/" in markdown
    assert "`src/main.tsx` -> `src/router.tsx`" in markdown
