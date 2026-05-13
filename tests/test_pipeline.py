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
    assert not (output_dir / "codedoc.json").exists()
    output = md_path.read_text(encoding="utf-8")
    assert "## Project Tree" in output
    assert "## Folder Map" in output
    assert "### main.py" in output


def test_public_output_converts_json_and_markdown_without_llm():
    import json

    from codedoc.core.project_view import json_from_markdown, markdown_from_json

    view = {
        "schema_version": "1.3",
        "generated_at": "2026-05-02T00:00:00+00:00",
        "project": {
            "entry_file": "main.py",
            "file_count": 1,
            "languages": ["python"],
            "folders": ["."],
        },
        "run": {
            "files_checked": 1,
            "files_failed": 0,
            "files_skipped": 0,
            "files_reused": 0,
            "files_documented": 1,
        },
        "tree": {"main.py": {"type": "file", "path": "main.py"}},
        "folders": [
            {
                "path": ".",
                "summary": "Root-level python files (1 file(s)). Common concepts: entry point.",
                "file_count": 1,
                "languages": ["python"],
                "files": ["main.py"],
                "key_concepts": ["entry point"],
            }
        ],
        "dependency_graph": [],
        "files": [
            {
                "id": "abc123",
                "path": "main.py",
                "format": "py",
                "language": "python",
                "description": "Main entry point.",
                "role_in_system": "Starts the app.",
                "imports": ["utils"],
                "functions": [{"name": "main", "description": "Runs the app."}],
                "key_concepts": ["entry point"],
                "usage_example": "python main.py",
                "links": {
                    "internal_dependencies": [],
                    "imported_by": [],
                    "external_dependencies": ["click"],
                },
            }
        ],
    }

    markdown = markdown_from_json(view)
    converted = json.loads(json_from_markdown(markdown))

    assert "## Project Overview" in markdown
    assert converted["project"]["entry_file"] == "main.py"
    assert converted["project"]["languages"] == ["python"]
    assert converted["files"][0]["path"] == "main.py"
    assert converted["files"][0]["description"] == "Main entry point."
    assert converted["files"][0]["functions"] == [
        {"name": "main", "description": "Runs the app."}
    ]
    assert converted["files"][0]["links"]["external_dependencies"] == ["click"]


def test_markdown_to_json_does_not_create_empty_default_sections():
    import json

    from codedoc.core.project_view import json_from_markdown

    markdown = """# codedoc project documentation

## Project Overview

- Entry file: `main.py`
- Files documented: 1
- Languages: python
- Folders: `.`

## Run Summary

- Files checked: 1
- Files failed: 0
- Files skipped: 0
- Files reused from cache: 0

## Project Tree

```text
main.py
```

## Files

### main.py

**ID:** `abc123`  
**Format:** py  
**Language:** python  

**Description:** Main entry point.

"""

    converted = json.loads(json_from_markdown(markdown))

    assert "dependency_graph" not in converted
    assert "dependency_catalog" not in converted
    assert "links" not in converted["files"][0]


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
    assert records[0]["documentation"]["file_path"] == "main.py"

    import json

    cache = json.loads(db.db_path.read_text(encoding="utf-8"))
    assert "version" not in cache
    cached_result = cache["files"]["main.py"].get("result", {})
    assert "file_path" not in cached_result
    assert "state" not in cached_result

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


def test_pipeline_cached_run_can_switch_back_to_json(tmp_path, monkeypatch):
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

    docs = tmp_path / "docs_output"
    docs.mkdir()
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
    assert not (tmp_path / "docs_output" / "codedoc.md").exists()
    assert "Main entry point." in json_path.read_text(encoding="utf-8")


def test_python_api_accepts_config_as_first_argument(tmp_path, monkeypatch):
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
            "description": "Current directory API.",
            "state": "checked",
        },
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached API output")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)
    monkeypatch.chdir(tmp_path)

    stats = run_pipeline({"output_dir": "docs_output", "output_format": "md"})

    assert stats["checked"] == 0
    assert (tmp_path / "docs_output" / "codedoc.md").exists()
    assert "Current directory API." in (
        tmp_path / "docs_output" / "codedoc.md"
    ).read_text(encoding="utf-8")


def test_cli_run_alias_passes_current_directory_and_overrides(monkeypatch):
    from codedoc.cli.cli import main

    captured = {}

    def fake_run_pipeline(root, config_overrides=None):
        captured["root"] = root
        captured["config"] = config_overrides
        return {
            "checked": 0,
            "failed": 0,
            "reused": 0,
            "output_dir": "docs_output",
            "output_files": [],
        }

    monkeypatch.setattr("codedoc.pipeline.run_pipeline", fake_run_pipeline)

    main(["run", "--format", "md", "--max-parallel-files", "3"])

    assert captured["root"].name == "codedoc"
    assert captured["config"]["output_format"] == "md"
    assert captured["config"]["max_parallel_files"] == 3


def test_pipeline_processes_files_with_bounded_parallelism(tmp_path, monkeypatch):
    import json
    import threading
    import time

    from codedoc.pipeline import run_pipeline

    for index in range(6):
        (tmp_path / f"file_{index}.py").write_text(
            f"def func_{index}():\n    return {index}\n",
            encoding="utf-8",
        )

    class SlowProvider:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        @property
        def provider_name(self):
            return "SlowProvider"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                if "key_concepts" in prompt:
                    return json.dumps(
                        {
                            "description": "Documented file.",
                            "role_in_system": "Test role.",
                            "key_concepts": [],
                            "usage_example": "",
                        }
                    )
                if "dependencies_analysis" in prompt:
                    return json.dumps(
                        {
                            "dependencies_analysis": {
                                "internal": [],
                                "external": [],
                                "dependency_refs": [],
                                "catalog_updates": [],
                                "usage_notes": [],
                                "warnings": [],
                            }
                        }
                    )
                return json.dumps(
                    {
                        "description": "Structured file.",
                        "role_in_system": "Test role.",
                        "functions": [],
                        "classes": [],
                        "exports": [],
                    }
                )
            finally:
                with self.lock:
                    self.active -= 1

    provider = SlowProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "parallel_agents": False,
            "max_parallel_files": 3,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 6
    assert stats["failed"] == 0
    assert 1 < provider.max_active <= 3
    assert (tmp_path / "docs_output" / "codedoc.json").exists()


def test_pipeline_retries_failed_file_before_marking_failed(tmp_path, monkeypatch):
    import json

    from codedoc.pipeline import run_pipeline

    source = tmp_path / "main.py"
    source.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    class FlakyProvider:
        def __init__(self):
            self.calls = 0

        @property
        def provider_name(self):
            return "FlakyProvider"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider outage")
            if "key_concepts" in prompt:
                return json.dumps(
                    {
                        "description": "Recovered file.",
                        "role_in_system": "Recovered role.",
                        "key_concepts": [],
                        "usage_example": "",
                    }
                )
            if "dependencies_analysis" in prompt:
                return json.dumps(
                    {
                        "dependencies_analysis": {
                            "internal": [],
                            "external": [],
                            "dependency_refs": [],
                            "catalog_updates": [],
                            "usage_notes": [],
                            "warnings": [],
                        }
                    }
                )
            return json.dumps(
                {
                    "description": "Recovered structure.",
                    "role_in_system": "Recovered role.",
                    "functions": [],
                    "classes": [],
                    "exports": [],
                }
            )

    provider = FlakyProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert provider.calls > 1
    assert "Recovered file." in (
        tmp_path / "docs_output" / "codedoc.json"
    ).read_text(encoding="utf-8")


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
                "dependencies_analysis": {
                    "external": ["react"],
                    "dependency_refs": ["react"],
                    "catalog_updates": [
                        {
                            "name": "react",
                            "type": "external",
                            "used_for": "Rendering UI components.",
                        }
                    ],
                    "usage_notes": [
                        {"import": "react", "used_for": "Creates this component tree."}
                    ],
                },
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
                "dependencies_analysis": {
                    "external": ["react", "react-router-dom"],
                    "dependency_refs": ["react", "react-router-dom"],
                    "catalog_updates": [
                        {
                            "name": "react-router-dom",
                            "type": "external",
                            "used_for": "Routing screens.",
                        }
                    ],
                    "usage_notes": [
                        {"import": "react", "used_for": "Supports route component rendering."},
                        {"import": "react-router-dom", "used_for": "Defines app routes."},
                    ],
                },
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
    assert payload["dependency_catalog"][0]["name"] == "react"
    assert payload["dependency_catalog"][0]["file_count"] == 2
    assert payload["dependency_catalog"][0]["used_for"] == "Rendering UI components."
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
    assert "## Dependency Catalog" in markdown
    assert "### react" in markdown
    assert "src/" in markdown
    assert "`src/main.tsx` -> `src/router.tsx`" in markdown
