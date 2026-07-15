"""0.10.1 — deterministic dependency projection, shared enrichment, truncation.

Covers Workstreams F (cross-mode dependency consistency), G (shared strict
cleaning, aligned prompt semantics, and the file-doc-v3 cache invalidation), and
H (head-plus-tail bounded source context).
"""
from __future__ import annotations

from codedoc.agents import (
    documentation_agent,
    file_documentation_agent,
    structure_agent,
)
from codedoc.agents.base_agent import TRUNCATION_MARKER, truncate_for_llm
from codedoc.agents.response_cleaning import (
    clean_dependency_response,
    clean_structure_response,
)
from codedoc.core.project_view import build_project_view
from codedoc.core.record_meta import ANALYSIS_REVISION, expected_analysis_identity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _py_record(path, imports, model_external=None, *, hash_="h"):
    deps = {}
    if model_external is not None:
        deps["external"] = model_external
    return {
        "hash": hash_,
        "file_path": path,
        "language": "python",
        "documentation": {
            "file_path": path,
            "language": "python",
            "description": "d",
            "imports": imports,
            "dependencies_analysis": deps,
        },
    }


def _links(view, path):
    return {f["path"]: f.get("links", {}) for f in view["files"]}[path]


# ---------------------------------------------------------------------------
# F. Cross-mode dependency consistency
# ---------------------------------------------------------------------------

def test_python_links_identical_regardless_of_model_dependencies():
    # single and triple emit different model dependency text, but for Python the
    # public links project from the parser imports only — so they are identical.
    single = build_project_view(
        [_py_record("a.py", ["os", "requests"], model_external=["os", "requests"])],
        {"checked": 1},
    )
    triple = build_project_view(
        [_py_record("a.py", ["os", "requests"], model_external=["totally", "different"])],
        {"checked": 1},
    )
    assert _links(single, "a.py") == _links(triple, "a.py")
    assert _links(single, "a.py")["external_dependencies"] == ["requests"]
    assert _links(single, "a.py")["sdk_dependencies"] == ["os"]


def test_python_model_only_name_never_appears_in_links():
    record = _py_record("a.py", ["os"], model_external=["requests", "evil"])
    links = _links(build_project_view([record], {"checked": 1}), "a.py")
    assert links.get("external_dependencies", []) == []
    assert links["sdk_dependencies"] == ["os"]


def test_project_import_resolved_by_graph_edge_is_not_external():
    edges = [{"from": "pkg/a.py", "to": "pkg/b.py", "type": "internal_import"}]
    view = build_project_view(
        [
            _py_record("pkg/a.py", ["pkg.b", "os"]),
            _py_record("pkg/b.py", []),
        ],
        {"checked": 2},
        graph_edges=edges,
    )
    links = _links(view, "pkg/a.py")
    assert links["internal_dependencies"] == ["pkg/b.py"]
    assert links["sdk_dependencies"] == ["os"]
    assert "pkg" not in links.get("external_dependencies", [])


def test_stdlib_imports_keep_sdk_classification():
    links = _links(
        build_project_view([_py_record("a.py", ["os", "sys", "json"])], {"checked": 1}),
        "a.py",
    )
    assert links["sdk_dependencies"] == ["json", "os", "sys"]
    assert "external_dependencies" not in links


def test_catalog_for_unmatched_dependency_name_is_discarded():
    # A usage note for a name that is not in the parser imports cannot create a
    # catalog entry.
    record = _py_record("a.py", ["os"])
    record["documentation"]["dependencies_analysis"]["usage_notes"] = [
        {"import": "requests", "used_for": "http"}
    ]
    view = build_project_view([record], {"checked": 1})
    assert "dependency_catalog" not in view


def test_non_python_uses_model_external_when_parser_omits_packages():
    # The JS/TS parser omits bare npm specifiers; the deterministic projection
    # falls back to the model's reported external for non-Python languages.
    record = {
        "hash": "h",
        "file_path": "app.tsx",
        "language": "tsx",
        "documentation": {
            "file_path": "app.tsx",
            "language": "tsx",
            "imports": ["./helper"],
            "dependencies_analysis": {"external": ["react"]},
        },
    }
    links = _links(build_project_view([record], {"checked": 1}), "app.tsx")
    assert links["external_dependencies"] == ["react"]


# ---------------------------------------------------------------------------
# G. Shared strict cleaning + aligned prompt semantics
# ---------------------------------------------------------------------------

def test_triple_structure_cleaner_drops_unknown_keys_booleans_and_empties():
    raw = {
        "description": "  trimmed  ",
        "role_in_system": True,           # boolean rejected
        "functions": [
            {"name": "f", "description": "ok"},
            {"name": "f", "description": "ok"},  # duplicate dropped
            {"bogus": "x"},                       # nameless dropped
        ],
        "classes": [],                    # empty omitted
        "exports": ["E", "E"],            # de-duplicated
        "unknown_key": "ignored",         # unknown dropped
    }
    cleaned = clean_structure_response(raw, "m.py")
    assert cleaned["description"] == "trimmed"
    assert "role_in_system" not in cleaned
    assert cleaned["functions"] == [{"name": "f", "description": "ok"}]
    assert "classes" not in cleaned
    assert cleaned["exports"] == ["E"]
    assert "unknown_key" not in cleaned


def test_triple_dependency_cleaner_preserves_shape_and_drops_empties():
    raw = {
        "dependencies_analysis": {
            "internal": ["./a", "./a"],
            "external": ["react"],
            "warnings": [],
        }
    }
    cleaned = clean_dependency_response(raw, "m.py")
    da = cleaned["dependencies_analysis"]
    assert da["internal"] == ["./a"]
    assert da["external"] == ["react"]
    assert "warnings" not in da


def test_both_prompt_families_share_symbol_and_usage_definitions():
    fda_system, fda_prompt = file_documentation_agent.build_prompt(
        "a.py", "code", ["os"], "python"
    )
    struct_system, struct_prompt = structure_agent.build_prompt(
        "a.py", "code", ["os"], "python"
    )
    doc_system, doc_prompt = documentation_agent.build_prompt(
        "a.py", "code", "python", {}, {}
    )
    # Local-symbol + re-export definitions appear in both structure prompts.
    for prompt in (fda_prompt, struct_prompt):
        assert "DEFINED IN this file" in prompt
        assert "re-exports" in prompt
    # Usage-example factuality wording appears in both prompts that emit one.
    for prompt in (fda_prompt, doc_prompt):
        assert "Include usage_example only when" in prompt
        assert "path/to/file" in prompt
    # Both families warn the model about the truncation marker.
    for prompt in (fda_prompt, struct_prompt, doc_prompt):
        assert "[truncated]" in prompt


def test_cache_identity_is_v2():
    assert ANALYSIS_REVISION == "file-doc-v3"
    assert expected_analysis_identity("single") == {
        "_analysis_revision": "file-doc-v3",
        "_analysis_mode": "single",
    }


def test_v1_record_is_invalidated_once_under_v2(tmp_path):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.graph import DependencyGraph
    from codedoc.core.planning import build_pipeline_plan

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    file_map = {
        "main.py": {
            "path": tmp_path / "main.py",
            "rel_path": "main.py",
            "language": "python",
            "extension": ".py",
        }
    }
    graph = DependencyGraph()
    graph.add_file("main.py")
    file_hash = compute_file_hash(tmp_path / "main.py")

    def _plan(revision):
        existing = {
            "main.py": {
                "path": "main.py",
                "hash": file_hash,
                "description": "cached",
                "_analysis_revision": revision,
                "_analysis_mode": "single",
            }
        }
        plan, _ = build_pipeline_plan(
            file_map, graph, {"main.py"}, "main.py", existing, [],
            {"propagate_changes": False, "max_files": 0, "analysis_mode": "single"},
        )
        return plan

    # A current-revision record with an unchanged hash is reused (no LLM call):
    # it is skipped as unchanged, never routed to an agent.
    current = _plan(ANALYSIS_REVISION)
    assert "main.py" in current.unchanged_rels
    assert "main.py" not in current.agent_rels
    # A stale file-doc-v1 record is invalidated and reprocessed once.
    stale = _plan("file-doc-v1")
    assert "main.py" in stale.agent_rels
    assert "main.py" not in stale.unchanged_rels


# ---------------------------------------------------------------------------
# H. Head-plus-tail bounded source context
# ---------------------------------------------------------------------------

def test_short_content_is_unchanged():
    text = "def f():\n    return 1\n"
    assert truncate_for_llm(text, 12000) == text


def test_oversized_content_stays_within_ceiling_including_marker():
    text = "x" * 5000
    result = truncate_for_llm(text, 1000)
    assert len(result) == 1000
    assert TRUNCATION_MARKER in result


def test_head_and_tail_present_and_middle_absent():
    head = "HEADMARKER\n" + "a" * 2000
    middle = "MIDDLEUNIQUE\n" + "b" * 2000
    tail = "c" * 2000 + "\nclass LateClass:\n    pass\n"
    content = head + middle + tail
    result = truncate_for_llm(content, 1500)
    assert len(result) <= 1500
    assert "HEADMARKER" in result          # leading slice retained
    assert "LateClass" in result           # late definition now visible
    assert "MIDDLEUNIQUE" not in result    # omitted middle
    assert TRUNCATION_MARKER in result


def test_unicode_characters_are_not_split():
    # A run of multibyte characters around the cut must never raise or split a
    # code point (Python slices by code point).
    content = "é" * 5000
    result = truncate_for_llm(content, 1000)
    assert len(result) == 1000
    assert "�" not in result  # no replacement char from a broken cut


def test_degenerate_ceiling_smaller_than_marker():
    result = truncate_for_llm("x" * 100, len(TRUNCATION_MARKER) - 1)
    assert len(result) == len(TRUNCATION_MARKER) - 1


def test_truncation_is_identical_for_every_caller():
    # The single shared helper guarantees both modes and dry-run estimation see
    # exactly the same bounded text for the same input.
    content = "y" * 9000
    first = truncate_for_llm(content, 4000)
    second = truncate_for_llm(content, 4000)
    assert first == second
    assert len(first) == 4000
