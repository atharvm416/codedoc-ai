"""Tests organized by feature ownership."""

from __future__ import annotations

import json

def _make_raw_record(usage_example: str, file_path: str = "lib/widget.dart") -> dict:
    """Build a minimal pipeline-style raw record for _clean_file()."""
    return {
        "hash": "testhash",
        "file_path": file_path,
        "language": "dart",
        "documentation": {
            "file_path": file_path,
            "language": "dart",
            "description": "A widget.",
            "role_in_system": "UI component.",
            "imports": ["package:flutter/material.dart"],
            "functions": [],
            "classes": [{"name": "MyWidget", "description": "Widget."}],
            "exports": [],
            "key_concepts": ["widget"],
            "usage_example": usage_example,
            "dependencies_analysis": {
                "internal": [],
                "external": ["flutter"],
                "dependency_refs": ["flutter"],
                "catalog_updates": [],
                "usage_notes": [],
                "warnings": [],
            },
        },
    }

def _clean(usage_example: str) -> str:
    from codedoc.core.project_view import _clean_file
    result = _clean_file(_make_raw_record(usage_example))
    return result.get("usage_example", "")

def test_B1_your_package_is_removed():
    """B1: 'your_package' placeholder is removed."""
    assert _clean("import 'package:your_package/core/theme.dart';") == ""

def test_B1_your_package_in_python_import_is_removed():
    """B1b: 'your_package' in a Python import is removed."""
    assert _clean("from your_package import MyClass") == ""

def test_B1_your_package_inline_text_is_removed():
    """B1c: 'your_package' anywhere in the example causes removal."""
    assert _clean("# Usage\n# See your_package docs for more info") == ""

def test_B2_your_project_is_removed():
    """B2a: 'your_project' placeholder is removed."""
    assert _clean("from your_project.utils import helper") == ""

def test_B2_your_app_is_removed():
    """B2b: 'your_app' placeholder is removed."""
    assert _clean("import your_app.main") == ""

def test_B2_your_package_name_is_removed():
    """B2c: 'your_package_name' placeholder is removed."""
    assert _clean("import 'package:your_package_name/module.dart';") == ""

def test_B2_example_package_is_removed():
    """B2d: 'example_package' placeholder is removed."""
    assert _clean("from example_package import SomeClass") == ""

def test_B2_my_package_is_removed():
    """B2e: 'my_package' placeholder is removed."""
    assert _clean("import my_package") == ""

def test_B3_dart_package_example_is_removed():
    """B3: Dart-style 'package:example/' prefix is removed."""
    assert _clean("import 'package:example/some_lib.dart';") == ""

def test_B3_dart_package_example_with_path_is_removed():
    """B3b: 'package:example/foo/bar.dart' is removed."""
    assert _clean("import 'package:example/core/theme/app_theme.dart';") == ""

def test_B4_reused_cached_record_sanitized_via_clean_file():
    """B4: A record loaded from cache with a placeholder usage_example is sanitized
    when it passes through _clean_file() during build_project_view()."""
    from codedoc.core.project_view import build_project_view

    # Simulate a cached "public" record re-entering build_project_view via
    # _build_documentation_records() → _public_record_to_doc().
    # In that flow the usage_example ends up in documentation["usage_example"].
    cached_record = {
        "hash": "h1",
        "file_path": "lib/widget.dart",
        "language": "dart",
        "documentation": {
            "file_path": "lib/widget.dart",
            "language": "dart",
            "description": "Widget.",
            "usage_example": "import 'package:your_package/widget.dart';",
            "dependencies_analysis": {},
        },
    }

    view = build_project_view([cached_record], {"checked": 0, "failed": 0, "skipped": 0})
    file_map = {f["path"]: f for f in view["files"]}
    assert file_map["lib/widget.dart"].get("usage_example", "") == "", (
        "Cached record with placeholder usage_example must be sanitized in build_project_view()"
    )

def test_B4_safe_writer_record_sanitized():
    """B4b: SafeWriter.record() calls clean_file_record which sanitizes the example."""
    import tempfile
    from pathlib import Path
    from codedoc.core.safe_writer import SafeWriter

    with tempfile.TemporaryDirectory() as tmp:
        backup = Path(tmp) / "codedoc.json"
        sw = SafeWriter(backup, "json", "lib/main.dart", {})
        sw.record(
            "lib/widget.dart",
            {
                "language": "dart",
                "description": "Widget.",
                "usage_example": "import 'package:my_package/widget.dart';",
                "dependencies_analysis": {},
            },
            file_hash="h1",
        )
        data = json.loads(backup.read_text(encoding="utf-8"))
        files_map = {f["path"]: f for f in data.get("files", [])}
        assert files_map["lib/widget.dart"].get("usage_example", "") == "", (
            "SafeWriter must sanitize placeholder usage_example via clean_file_record()"
        )

def test_B5_real_flutter_import_preserved():
    """B5a: A real Flutter import is NOT removed."""
    example = "import 'package:flutter/material.dart';"
    assert _clean(example) == example

def test_B5_real_python_import_preserved():
    """B5b: A real Python stdlib/library import is NOT removed."""
    example = "from pathlib import Path\n\np = Path('.')\nprint(p.exists())"
    assert _clean(example) == example

def test_B5_real_relative_import_preserved():
    """B5c: A real relative import path is NOT removed."""
    example = "import '../core/theme/app_theme.dart';"
    assert _clean(example) == example

def test_B5_real_package_with_example_in_name_preserved():
    """B5d: A real package whose name contains 'example' as part of a longer word is kept."""
    # 'flutter_example_showcase' — 'example' is not a standalone word
    example = "import 'package:flutter_example_showcase/main.dart';"
    assert _clean(example) == example

def test_B5_code_with_example_word_in_variable_preserved():
    """B5e: A usage example that uses 'example' as a variable name is kept."""
    example = "example = MyClass()\nexample.run()"
    # 'example' alone is not in our pattern — only 'example_package'
    assert _clean(example) == example

def test_B5_real_dart_package_not_matching_example_preserved():
    """B5f: 'package:provider/provider.dart' has no placeholder — preserved."""
    example = "import 'package:provider/provider.dart';"
    assert _clean(example) == example

def test_B6_example_package_as_prefix_not_matched():
    """B6a: 'example_package_utils' is NOT treated as a placeholder."""
    example = "from example_package_utils.helpers import something"
    # 'example_package' ends in 'e', followed by '_' (a word char) → no \b → no match
    assert _clean(example) == example

def test_B6_your_package_as_suffix_not_matched():
    """B6b: 'real_your_package' is NOT treated as a placeholder."""
    # 'your_package' is preceded by '_' (word char) → no leading \b → no match
    example = "from real_your_package import helper"
    assert _clean(example) == example

def test_B6_my_package_extended_not_matched():
    """B6c: 'my_package_manager' is NOT treated as a placeholder."""
    example = "import my_package_manager"
    assert _clean(example) == example

def test_B7_uppercase_your_package_removed():
    """B7a: 'YOUR_PACKAGE' (all caps) is treated as a placeholder."""
    assert _clean("from YOUR_PACKAGE import something") == ""

def test_B7_mixed_case_your_project_removed():
    """B7b: 'Your_Project' (mixed case) is treated as a placeholder."""
    assert _clean("import Your_Project.main") == ""

def test_B7_package_example_uppercase_removed():
    """B7c: 'package:EXAMPLE/' is treated as a placeholder."""
    assert _clean("import 'package:EXAMPLE/module.dart';") == ""

def test_B8_empty_string_unchanged():
    """B8a: An empty string is returned as-is (no crash)."""
    assert _clean("") == ""

def test_B8_none_like_empty_from_result():
    """B8b: A record with no usage_example key produces an empty result (no crash)."""
    from codedoc.core.project_view import _clean_file
    record = {
        "hash": "h1",
        "file_path": "main.py",
        "language": "python",
        "documentation": {
            "file_path": "main.py",
            "language": "python",
            "description": "Entry point.",
            "dependencies_analysis": {},
            # No 'usage_example' key
        },
    }
    result = _clean_file(record)
    # Should not crash and usage_example should not appear (or be "")
    assert result.get("usage_example", "") == ""

def test_B9_sanitizer_is_idempotent():
    """B9: Running _sanitize_usage_example twice gives the same result."""
    from codedoc.core.project_view import _sanitize_usage_example

    placeholder = "from your_package import MyClass"
    first = _sanitize_usage_example(placeholder)
    second = _sanitize_usage_example(first)
    assert first == "" and second == "", "Sanitizer must be idempotent"

def test_B9_valid_example_idempotent():
    """B9b: Running _sanitize_usage_example on a valid example is idempotent."""
    from codedoc.core.project_view import _sanitize_usage_example

    valid = "import 'package:flutter/material.dart';"
    assert _sanitize_usage_example(valid) == valid
    assert _sanitize_usage_example(_sanitize_usage_example(valid)) == valid

def test_B10_placeholder_not_in_regen_json():
    """B10: After Markdown → embedded view → JSON round-trip, placeholders are absent."""
    from codedoc.core.project_view import build_project_view, markdown_from_view, json_from_markdown

    record_with_placeholder = {
        "hash": "h1",
        "file_path": "lib/main.dart",
        "language": "dart",
        "documentation": {
            "file_path": "lib/main.dart",
            "language": "dart",
            "description": "Entry point.",
            "usage_example": "import 'package:your_package/core/app.dart';",
            "dependencies_analysis": {
                "internal": [], "external": ["flutter"],
                "dependency_refs": [], "catalog_updates": [],
                "usage_notes": [], "warnings": [],
            },
        },
    }

    view = build_project_view([record_with_placeholder],
                               {"checked": 1, "failed": 0, "skipped": 0},
                               entry_file="lib/main.dart")
    md = markdown_from_view(view)
    regen = json.loads(json_from_markdown(md))

    file_map = {f["path"]: f for f in regen["files"]}
    usage = file_map["lib/main.dart"].get("usage_example", "")
    assert "your_package" not in usage, (
        f"Placeholder must not appear in regenerated JSON, got: {usage!r}"
    )
    assert usage == "", "Sanitized placeholder must produce an empty usage_example"

def test_B10_valid_usage_survives_round_trip():
    """B10b: A valid usage_example is preserved through the round-trip."""
    from codedoc.core.project_view import build_project_view, markdown_from_view, json_from_markdown

    valid_usage = "import 'package:flutter/material.dart';\n\nvoid main() => runApp(MyApp());"
    record = {
        "hash": "h2",
        "file_path": "lib/main.dart",
        "language": "dart",
        "documentation": {
            "file_path": "lib/main.dart",
            "language": "dart",
            "description": "Entry point.",
            "usage_example": valid_usage,
            "dependencies_analysis": {
                "internal": [], "external": ["flutter"],
                "dependency_refs": [], "catalog_updates": [],
                "usage_notes": [], "warnings": [],
            },
        },
    }

    view = build_project_view([record], {"checked": 1, "failed": 0, "skipped": 0})
    md = markdown_from_view(view)
    regen = json.loads(json_from_markdown(md))

    file_map = {f["path"]: f for f in regen["files"]}
    assert file_map["lib/main.dart"]["usage_example"] == valid_usage, (
        "Valid usage_example must be preserved unchanged through the round-trip"
    )

def test_B11_pipeline_json_has_no_placeholder(tmp_path, monkeypatch):
    """B11a: Pipeline-produced JSON has usage_example sanitized."""
    import json as _json

    (tmp_path / "main.dart").write_text("void main() {}\n")

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": "Entry point.",
                    "role_in_system": "App launcher.",
                    "key_concepts": ["entry point"],
                    "usage_example": "import 'package:your_package/main.dart';",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": [],
                }})
            return _json.dumps({
                "description": "Entry point.",
                "role_in_system": "App launcher.",
                "functions": [], "classes": [], "exports": [],
            })

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: FakeProvider())
    from codedoc.pipeline import run_pipeline

    run_pipeline(tmp_path, {
        "entry_file": "main.dart",
        "output_format": "json",
        "parallel_agents": False,
        "propagate_changes": False,
        "supported_extensions": [".dart"],
    })

    out = _json.loads((tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8"))
    for f in out.get("files", []):
        usage = f.get("usage_example", "")
        assert "your_package" not in usage, (
            f"Placeholder must not appear in pipeline JSON output for {f['path']!r}: {usage!r}"
        )

def test_B11_pipeline_md_has_no_placeholder(tmp_path, monkeypatch):
    """B11b: Pipeline-produced Markdown (and its embedded view) has usage_example sanitized."""
    import json as _json
    from codedoc.core.project_view import read_embedded_view

    (tmp_path / "main.py").write_text("def main(): pass\n")

    class FakeProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            if "key_concepts" in prompt:
                return _json.dumps({
                    "description": "Entry.",
                    "role_in_system": "Main.",
                    "key_concepts": [],
                    "usage_example": "from your_project.utils import helper",
                })
            if "dependencies_analysis" in prompt:
                return _json.dumps({"dependencies_analysis": {
                    "internal": [], "external": [], "dependency_refs": [],
                    "catalog_updates": [], "usage_notes": [], "warnings": [],
                }})
            return _json.dumps({"description": "Entry.", "role_in_system": "Main.",
                                "functions": [], "classes": [], "exports": []})

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _: FakeProvider())
    from codedoc.pipeline import run_pipeline

    run_pipeline(tmp_path, {
        "entry_file": "main.py",
        "output_format": "md",
        "parallel_agents": False,
        "propagate_changes": False,
    })

    md_content = (tmp_path / "codedoc" / "codedoc.md").read_text(encoding="utf-8")

    # Check embedded view
    embedded = read_embedded_view(md_content)
    assert embedded is not None
    for f in embedded.get("files", []):
        usage = f.get("usage_example", "")
        assert "your_project" not in usage, (
            f"Placeholder must not appear in embedded view for {f['path']!r}: {usage!r}"
        )

    # Check visible Markdown text
    assert "your_project" not in md_content, (
        "Placeholder must not appear in visible Markdown output"
    )
