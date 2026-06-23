from codedoc.core.record_meta import expected_analysis_identity

# 0.10.0: cache-identity keys a prior single-mode CodeDoc run would persist, so
# reuse fixtures simulate real prior output rather than pre-0.10.0 records.
_PRIOR_RUN_IDENTITY = expected_analysis_identity("single")


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
                    # 0.10.1: public links project from the parser imports.
                    "imports": [
                        "flutter/material.dart",
                        "provider/provider.dart",
                        "dart:async",
                    ],
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
    # 0.9.3: SDK/standard-library names are separated from third-party packages.
    # `dart:async` is a Dart SDK library, so it moves to `sdk_dependencies`.
    links = payload["files"][0]["links"]
    assert links["external_dependencies"] == ["flutter", "provider"]
    assert links["sdk_dependencies"] == ["dart:async"]


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
                    **_PRIOR_RUN_IDENTITY,
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
                    **_PRIOR_RUN_IDENTITY,
                },
                {
                    "path": "first.py",
                    "hash": first_hash,
                    "description": "Shared helper.",
                    "language": "python",
                    "format": "py",
                    **_PRIOR_RUN_IDENTITY,
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
                    **_PRIOR_RUN_IDENTITY,
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
                    **_PRIOR_RUN_IDENTITY,
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
    from pathlib import Path

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

    # The ``run`` alias passes the current working directory unchanged, so the
    # captured root must equal CWD — not any hard-coded checkout directory name.
    assert Path(captured["root"]).resolve() == Path.cwd().resolve()
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
                "imports": ["react", "./router"],
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
                "imports": ["react", "react-router-dom"],
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
            **_PRIOR_RUN_IDENTITY,
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
            **_PRIOR_RUN_IDENTITY,
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


def test_select_files_raises_when_entry_not_in_file_map(tmp_path, monkeypatch):
    """A2: when an explicit entry exists on disk but is not picked up by the
    scanner (e.g. unsupported extension), the run must raise ConfigError instead
    of silently falling back to documenting all files."""
    import json as _json
    import pytest

    from codedoc.utils.errors import ConfigError
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

    with pytest.raises(ConfigError) as exc_info:
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

    assert "scanned file set" in str(exc_info.value)


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


def test_select_files_warns_and_excludes_unreachable_files(tmp_path, caplog):
    """A1 visibility: files not reachable from --entry are excluded, and the
    exclusion is logged loudly (never silent).  Pure, no LLM."""
    import logging

    from codedoc.core.graph import DependencyGraph
    from codedoc.core.loader import load_config
    from codedoc.pipeline import _select_files

    # entry.py imports helper.py; orphan.py is imported by nobody.
    (tmp_path / "entry.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "orphan.py").write_text("y = 2\n", encoding="utf-8")

    graph = DependencyGraph()
    for rel in ("entry.py", "helper.py", "orphan.py"):
        graph.add_file(rel)
    graph.add_dependency("entry.py", "helper.py")

    file_map = {
        rel: {"path": tmp_path / rel, "rel_path": rel, "language": "python", "extension": ".py"}
        for rel in ("entry.py", "helper.py", "orphan.py")
    }

    config = load_config(tmp_path, {"entry_file": "entry.py"})

    with caplog.at_level(logging.WARNING):
        reachable, selected, entry_rel = _select_files(tmp_path, config, graph, file_map)

    assert entry_rel == "entry.py"
    assert selected == {"entry.py", "helper.py"}
    assert reachable == selected
    assert "orphan.py" not in selected
    # The omission must be visible.
    warning_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "orphan.py" in warning_text
    assert "disconnected" in warning_text.lower()


def test_select_files_no_exclusion_when_all_reachable(tmp_path, caplog):
    """No spurious warning when every scanned file is reachable from the entry."""
    import logging

    from codedoc.core.graph import DependencyGraph
    from codedoc.core.loader import load_config
    from codedoc.pipeline import _select_files

    (tmp_path / "entry.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")

    graph = DependencyGraph()
    for rel in ("entry.py", "helper.py"):
        graph.add_file(rel)
    graph.add_dependency("entry.py", "helper.py")

    file_map = {
        rel: {"path": tmp_path / rel, "rel_path": rel, "language": "python", "extension": ".py"}
        for rel in ("entry.py", "helper.py")
    }

    config = load_config(tmp_path, {"entry_file": "entry.py"})

    with caplog.at_level(logging.WARNING):
        reachable, selected, entry_rel = _select_files(tmp_path, config, graph, file_map)

    assert selected == {"entry.py", "helper.py"}
    assert reachable == selected
    warning_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "not reachable" not in warning_text.lower()


class _RateLimitOrch:
    class _LLM:
        provider_name = "fake"
    llm = _LLM()

    def process(self, descriptor, content, imports):
        raise RuntimeError("429 rate limit exceeded")


def test_A4_recorded_this_run_recovers_real_record(tmp_path):
    """A4: a file actually recorded THIS run whose future surfaces a rate-limit
    error is treated as done and recovers the REAL record (not {})."""
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.core.queue import ProcessingQueue
    from codedoc.utils.errors import ErrorReporter

    src = tmp_path / "x.py"
    src.write_text("import os\n", encoding="utf-8")
    descriptor = {"rel_path": "x.py", "path": src, "language": "python", "extension": ".py"}

    recorder = SafeWriter(tmp_path / "codedoc.json", "json", "x.py", {"x.py": descriptor})
    # A worker recorded it during THIS run.
    recorder.record("x.py", {"description": "Fresh desc", "language": "python",
                             "role_in_system": "role"}, "hash123")
    assert recorder.recorded_this_run("x.py")

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0}
    reporter = ErrorReporter(tmp_path / "error.log")

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [descriptor], _RateLimitOrch(), queue, stats, reporter,
        max_workers=2, recorder=recorder, profile=None,
    )

    assert succeeded.get("x.py", {}).get("description") == "Fresh desc"
    assert retry_rate_limited == []


def test_A4_preloaded_stale_record_is_retried_not_restored(tmp_path):
    """A4 (root cause): a changed file whose only record was PRELOADED from a
    prior run (stale) and which then rate-limits must be RETRIED, never silently
    restored from the old documentation or counted as checked."""
    import json as _json
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.safe_writer import SafeWriter
    from codedoc.core.queue import ProcessingQueue
    from codedoc.utils.errors import ErrorReporter

    src = tmp_path / "x.py"
    src.write_text("import os  # changed\n", encoding="utf-8")
    descriptor = {"rel_path": "x.py", "path": src, "language": "python", "extension": ".py"}

    backup = tmp_path / "codedoc.json"
    backup.write_text(_json.dumps({
        "_codedoc": {"status": "in_progress", "schema_version": "1.4"},
        "files": [{"path": "x.py", "language": "python",
                   "description": "STALE", "hash": "old"}],
    }), encoding="utf-8")

    recorder = SafeWriter(backup, "json", "x.py", {"x.py": descriptor})
    recorder.load()  # preloads the stale record but NOT as recorded-this-run
    assert recorder.has_record("x.py")
    assert not recorder.recorded_this_run("x.py")

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0}
    reporter = ErrorReporter(tmp_path / "error.log")

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [descriptor], _RateLimitOrch(), queue, stats, reporter,
        max_workers=2, recorder=recorder, profile=None,
    )

    retried_paths = [d["rel_path"] for d, _exc in retry_rate_limited]
    assert "x.py" in retried_paths, "stale preloaded file must be retried"
    assert "x.py" not in succeeded
    assert stats["checked"] == 0


def test_A2_explicit_entry_with_zero_scanned_files_raises(tmp_path):
    """A2: an explicit entry with no supported files scanned must raise, not exit
    successfully having documented nothing."""
    import pytest
    from codedoc.utils.errors import ConfigError
    from codedoc.pipeline import run_pipeline
    # Empty project (no supported files at all).
    with pytest.raises(ConfigError):
        run_pipeline(tmp_path, {"entry_file": "main.py", "output_format": "json",
                                "parallel_agents": False})


def test_A2_entry_outside_project_root_raises(tmp_path):
    """A2: an explicit entry resolving outside the project root raises ConfigError
    (not a leaked ValueError)."""
    import pytest
    from codedoc.utils.errors import ConfigError
    from codedoc.pipeline import run_pipeline

    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("y = 1\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        run_pipeline(root, {"entry_file": str(outside), "output_format": "json",
                            "parallel_agents": False})
    assert "outside" in str(exc_info.value).lower()


def test_A6_walker_state_independent_when_interleaved(tmp_path):
    """A6: two _Walker generators driven concurrently (interleaved) keep fully
    independent state — proving re-entrancy, which the old function-attribute
    implementation did not provide."""
    from itertools import zip_longest
    from codedoc.core.scanner import _Walker

    r1 = tmp_path / "r1"
    (r1 / "node_modules").mkdir(parents=True)
    (r1 / "node_modules" / "lib.py").write_text("a=1\n")
    (r1 / "keep1.py").write_text("a=1\n")

    r2 = tmp_path / "r2"
    r2.mkdir()
    (r2 / "keep2.py").write_text("b=1\n")

    w1 = _Walker(scan_root=r1, skip_dirs={"node_modules"}, ignore_prefixes=set())
    w2 = _Walker(scan_root=r2, skip_dirs=set(), ignore_prefixes=set())

    seen1, seen2 = [], []
    for a, b in zip_longest(w1.walk(r1), w2.walk(r2)):
        if a is not None:
            seen1.append(a.name)
        if b is not None:
            seen2.append(b.name)

    assert seen1 == ["keep1.py"]
    assert seen2 == ["keep2.py"]
    assert w1.skipped_dirs == 1   # node_modules
    assert w2.skipped_dirs == 0


def test_version_identity_consistent(capsys):
    """Release identity: pyproject, codedoc.__version__, CLI --version, and the
    README 'Current release' all agree."""
    import re
    import pathlib
    import pytest
    import codedoc

    repo_root = pathlib.Path(__file__).resolve().parent.parent

    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    assert match, "version not found in pyproject.toml"
    assert match.group(1) == codedoc.__version__

    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    readme_match = re.search(r'Current release:\s*`([^`]+)`', readme)
    assert readme_match, "'Current release' not found in README.md"
    assert readme_match.group(1) == codedoc.__version__

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit):
        main(["--version"])
    assert codedoc.__version__ in capsys.readouterr().out


def test_A5_interrupt_prints_clean_message_and_exits_130(tmp_path, monkeypatch, capsys):
    """A5: KeyboardInterrupt with no recovery_path attached prints the truthful
    generic message (no recovery file confirmed) and exits 130.

    0.9.8: when the interrupt carries no ``recovery_path`` (interrupted before a
    recovery file was initialized), the CLI must not claim a recovery file
    exists, and must affirm the stable output was left untouched.
    """
    import pytest

    import codedoc.pipeline as pipeline_mod

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(pipeline_mod, "run_pipeline", raise_interrupt)

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path)])

    assert exc_info.value.code == 130
    err = capsys.readouterr().err
    assert "interrupted" in err.lower()
    # Truthful generic wording: no recovery file was created/confirmed, and the
    # stable output was left untouched.  Never the old "Progress has been saved".
    assert "crash-recovery file was created or confirmed" in err
    assert "left untouched" in err
    assert "Progress has been saved" not in err
