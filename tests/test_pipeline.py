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
    assert '"schema_version": "1.4"' in output
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


def test_public_output_normalizes_external_package_names(tmp_path):
    import json

    from codedoc.core.output import write_project_outputs

    json_path, _ = write_project_outputs(
        [
            {
                "id": "main-hash",
                "hash": "main-hash",
                "file_path": "lib/main.dart",
                "format": "dart",
                "language": "dart",
                "last_processed": "2026-05-13T00:00:00+00:00",
                "documentation": {
                    "file_path": "lib/main.dart",
                    "language": "dart",
                    "description": "Starts the app.",
                    "dependencies_analysis": {
                        "external": [
                            "flutter/material.dart",
                            "provider/provider.dart",
                            "dart:async",
                        ],
                        "dependency_refs": [
                            "flutter/material.dart",
                            "provider/provider.dart",
                        ],
                    },
                    "state": "checked",
                },
            }
        ],
        {"checked": 1, "failed": 0, "skipped": 0},
        tmp_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["files"][0]["links"]["external_dependencies"] == [
        "dart:async",
        "flutter",
        "provider",
    ]


def test_pipeline_no_entry_no_docs_uses_auto_detection(tmp_path):
    """0.8.1: pipeline with no --entry and no existing docs must NOT raise 'No entry point
    specified'.  Instead _resolve_entry_and_docs() returns quietly and lets
    detect_entry_file() handle auto-detection at scan time.
    """
    from codedoc.core.loader import load_config
    from codedoc.pipeline import _resolve_entry_and_docs

    (tmp_path / "main.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    config = load_config(tmp_path, {"output_dir": "docs_output", "output_format": "json",
                                    "propagate_changes": False})
    config["entry_file"] = None  # simulate no --entry flag

    # Must NOT raise ConfigError — leaves entry_file unset for detect_entry_file()
    _resolve_entry_and_docs(tmp_path, config)
    assert config.get("entry_file") is None, (
        "_resolve_entry_and_docs() must leave entry_file unset so detect_entry_file() "
        "can handle auto-detection later in the pipeline"
    )


def test_pipeline_reads_entry_from_existing_json_metadata(tmp_path, monkeypatch):
    import json

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    output_dir = tmp_path / "docs_output"
    output_dir.mkdir()

    # Pre-write a public JSON that includes both the metadata block (for entry_file
    # discovery) and the file's documentation (so _build_documentation_records can
    # recover it without calling the LLM).
    file_hash = compute_file_hash(main)
    (output_dir / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "main.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "main.py",
                    "hash": file_hash,
                    "description": "Resumed from metadata.",
                    "language": "python",
                    "format": "py",
                }
            ],
        }),
        encoding="utf-8",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached metadata resume")

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
    assert "Resumed from metadata." in (
        output_dir / "codedoc.json"
    ).read_text(encoding="utf-8")



def test_pipeline_reuses_identical_file_content_without_llm(tmp_path, monkeypatch):
    import json

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    entry = tmp_path / "entry.py"
    content = "def shared():\n    return 1\n"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    entry.write_text("import first\nimport second\n", encoding="utf-8")

    # Pre-write the public JSON with first.py and entry.py docs and their hashes,
    # so that second.py (identical content to first.py) can be reused by hash.
    docs_output = tmp_path / "docs_output"
    docs_output.mkdir()
    first_hash = compute_file_hash(first)
    entry_hash = compute_file_hash(entry)
    (docs_output / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "entry.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "entry.py",
                    "hash": entry_hash,
                    "description": "Entry module.",
                    "language": "python",
                    "format": "py",
                    "imports": ["first", "second"],
                },
                {
                    "path": "first.py",
                    "hash": first_hash,
                    "description": "Shared helper.",
                    "language": "python",
                    "format": "py",
                },
            ],
        }),
        encoding="utf-8",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for identical cached content")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "entry.py",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["reused"] == 1
    output = (tmp_path / "docs_output" / "codedoc.json").read_text(encoding="utf-8")
    assert '"path": "first.py"' in output
    assert '"path": "second.py"' in output
    assert '"description": "Shared helper."' in output

    # Second run: public JSON still exists, all files are up-to-date, nothing to reuse
    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "entry.py",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert stats["reused"] == 0
    assert (tmp_path / "docs_output" / "codedoc.json").exists()


def test_pipeline_cached_run_honors_markdown_format(tmp_path, monkeypatch):
    import json

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    docs_output = tmp_path / "docs_output"
    docs_output.mkdir()

    file_hash = compute_file_hash(main)

    # Pre-write a valid public JSON so _load_existing_file_docs can recover the docs.
    (docs_output / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "main.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "main.py",
                    "hash": file_hash,
                    "description": "Main entry point.",
                    "language": "python",
                    "format": "py",
                }
            ],
        }),
        encoding="utf-8",
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
    import json

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


def test_python_api_accepts_config_as_first_argument(tmp_path, monkeypatch):
    import json

    from codedoc.core.db import compute_file_hash
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    docs_output = tmp_path / "docs_output"
    docs_output.mkdir()

    file_hash = compute_file_hash(main)

    # Pre-write public JSON so docs can be recovered.
    (docs_output / "codedoc.json").write_text(
        json.dumps({
            "_codedoc": {"entry_file": "main.py", "schema_version": "1.3"},
            "files": [
                {
                    "path": "main.py",
                    "hash": file_hash,
                    "description": "Current directory API.",
                    "language": "python",
                    "format": "py",
                }
            ],
        }),
        encoding="utf-8",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached API output")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)
    monkeypatch.chdir(tmp_path)

    stats = run_pipeline(
        {
            "output_dir": "docs_output",
            "output_format": "md",
            "entry_file": "main.py",
        }
    )

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
        imports = ""
        if index == 0:
            imports = "".join(f"import file_{dep}\n" for dep in range(1, 6))
        (tmp_path / f"file_{index}.py").write_text(
            f"{imports}def func_{index}():\n    return {index}\n",
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
            "entry_file": "file_0.py",
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
            "entry_file": "main.py",
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


# ---------------------------------------------------------------------------
# 0.7.0 regression tests
# ---------------------------------------------------------------------------

def test_md_output_embeds_file_hashes_in_metadata_comment(tmp_path):
    """file_hashes must appear in the <!-- codedoc-ai: ... --> comment so that
    subsequent --format md runs can perform incremental hash checks."""
    import json as _json

    from codedoc.core.output import write_project_outputs

    output_dir = tmp_path / "out"
    records = [
        {
            "hash": "deadbeef01",
            "file_path": "app.py",
            "language": "python",
            "documentation": {
                "file_path": "app.py",
                "language": "python",
                "description": "The app.",
            },
        }
    ]
    _, md_path = write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        output_format="md",
        entry_file="app.py",
    )

    assert md_path is not None
    content = md_path.read_text(encoding="utf-8")
    assert "<!-- codedoc-ai:" in content

    # Extract metadata comment and verify file_hashes is present
    import re
    match = re.search(r"<!-- codedoc-ai: (\{.*?\}) -->", content, re.DOTALL)
    assert match, "metadata comment not found"
    meta = _json.loads(match.group(1))
    assert "file_hashes" in meta
    assert meta["file_hashes"].get("app.py") == "deadbeef01"


def test_md_only_incremental_skips_unchanged_files(tmp_path, monkeypatch):
    """Second --format md run must not call the LLM for unchanged files.
    This verifies that file_hashes from the MD metadata comment are used
    for the incremental hash check when no JSON exists."""
    import json as _json

    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.pipeline import run_pipeline

    # Write a source file
    src = tmp_path / "app.py"
    src.write_text("def app(): pass\n", encoding="utf-8")
    real_hash = compute_file_hash(src)

    output_dir = tmp_path / "docs_output"

    # Simulate what a first MD run would have produced: an MD file with
    # file_hashes embedded in the metadata comment.  No JSON file exists.
    records = [
        {
            "hash": real_hash,
            "file_path": "app.py",
            "language": "python",
            "documentation": {
                "file_path": "app.py",
                "language": "python",
                "description": "The app module.",
            },
        }
    ]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        output_dir,
        output_format="md",
        entry_file="app.py",
    )
    # Confirm: only MD exists, no JSON
    assert (output_dir / "codedoc.md").exists()
    assert not (output_dir / "codedoc.json").exists()

    def fail_if_llm_used(config):
        raise AssertionError("LLM must not be called — file is unchanged")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_used)

    stats = run_pipeline(
        tmp_path,
        {
            "entry_file": "app.py",
            "output_dir": "docs_output",
            "output_format": "md",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert (output_dir / "codedoc.md").exists()
    assert not (output_dir / "codedoc.json").exists()


def test_cross_format_resume_finds_entry_from_md_sibling(tmp_path, monkeypatch):
    """--output docs/claude.json after a previous --format md run that wrote
    docs/claude.md must read the entry from claude.md and resume without error."""
    import json as _json

    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.pipeline import run_pipeline

    src = tmp_path / "main.py"
    src.write_text("def main(): pass\n", encoding="utf-8")
    real_hash = compute_file_hash(src)

    docs_dir = tmp_path / "docs"

    # Simulate a previous --format md run that wrote docs/claude.md
    records = [
        {
            "hash": real_hash,
            "file_path": "main.py",
            "language": "python",
            "documentation": {
                "file_path": "main.py",
                "language": "python",
                "description": "Entry module.",
            },
        }
    ]
    write_project_outputs(
        records,
        {"checked": 1, "failed": 0, "skipped": 0},
        docs_dir,
        output_format="md",
        entry_file="main.py",
        md_filename="claude.md",
    )

    assert (docs_dir / "claude.md").exists()
    assert not (docs_dir / "claude.json").exists()

    def fail_if_llm_used(config):
        raise AssertionError("LLM must not be called — file is unchanged")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_used)

    # Now run with --output docs/claude.json — should find claude.md as sibling
    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs/claude.json",
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 0
    assert (docs_dir / "claude.json").exists()
    json_content = (docs_dir / "claude.json").read_text(encoding="utf-8")
    assert "Entry module." in json_content


def test_select_files_warns_when_entry_not_in_file_map(tmp_path, monkeypatch, caplog):
    """When the entry file exists on disk but is not picked up by the scanner
    (e.g. unsupported extension), a clear warning must be logged instead of
    silently falling back to documenting all files."""
    import logging
    import json as _json

    from codedoc.pipeline import run_pipeline

    # A .py file the scanner will pick up
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
    # Entry file physically exists but has an unsupported extension —
    # scanner will ignore it, so it won't appear in file_map.
    (tmp_path / "entry.txt").write_text("entrypoint\n", encoding="utf-8")

    def fake_provider(config):
        class P:
            provider_name = "fake"
            def complete_json(self, prompt, system=""): return _json.dumps({
                "description": "x", "role_in_system": "x",
                "functions": [], "classes": [], "exports": [],
                "key_concepts": [], "usage_example": "",
                "dependencies_analysis": {"internal": [], "external": []},
            })
            def complete(self, prompt, system="", temperature=0.1):
                return self.complete_json(prompt)
        return P()

    monkeypatch.setattr("codedoc.pipeline.create_provider", fake_provider)

    with caplog.at_level(logging.WARNING, logger="codedoc.pipeline"):
        run_pipeline(
            tmp_path,
            {
                "output_dir": "docs_output",
                "output_format": "json",
                "entry_file": "entry.txt",   # exists but unsupported extension → not in file_map
                "propagate_changes": False,
                "max_parallel_files": 1,
                "parallel_agents": False,
            },
        )

    assert any(
        "not found in the scanned file set" in r.message
        for r in caplog.records
    ), "Expected warning about entry file not found in scanned set"


def test_format_both_with_named_file_raises_config_error(tmp_path):
    """--format both combined with a named output file must raise ConfigError,
    not silently downgrade to a single format."""
    from codedoc.utils.errors import ConfigError
    from codedoc.pipeline import run_pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    try:
        run_pipeline(
            tmp_path,
            {
                "output_dir": "docs/report.md",   # named file
                "output_format": "both",           # conflicts
                "entry_file": "main.py",
            },
        )
        assert False, "Expected ConfigError was not raised"
    except ConfigError as exc:
        assert "both" in str(exc).lower()
        assert "directory" in str(exc).lower()
