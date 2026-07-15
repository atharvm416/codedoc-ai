"""0.12.0 — cache identity confined to files whose effective block changed."""

from pathlib import PurePosixPath

from codedoc.core.db import compute_file_hash
from codedoc.core.graph import DependencyGraph
from codedoc.core.planning import (
    _identity_matches,
    _record_is_reusable,
    build_pipeline_plan,
)
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    validate_profile,
)

_KNOWN = frozenset({".py", ".js", ".ts", ".rb"})


def _profile(common_desc, per_extension):
    raw = {
        "single": {
            "common": {"requested_shape": {"description": common_desc}},
            "per_extension": per_extension,
        }
    }
    return ResolvedProfile(
        "single",
        validate_profile(raw, active_mode="single", known_extensions=_KNOWN,
                         source="inline", source_path=None),
    )


def _write(tmp_path, rel, content="x = 1\n"):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _descriptor(path, rel):
    return {
        "path": path, "rel_path": rel,
        "language": "generic", "extension": path.suffix.lower(),
    }


def _record(path, rel, resolved):
    rec = {
        "path": rel, "hash": compute_file_hash(path), "description": "cached",
        "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
    }
    digest = resolved.file_digest(PurePosixPath(rel).name.lower())
    if digest != NO_PROMPT_PROFILE_DIGEST:
        rec["_prompt_profile_digest"] = digest
    return rec


def _plan(file_map, existing, resolved, selected=None):
    graph = DependencyGraph()
    for rel in file_map:
        graph.add_file(rel)
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "max_content_chars": 12000, "truncation_head_ratio": 0.70,
    }
    selected = set(selected) if selected is not None else set(file_map)
    plan, _ = build_pipeline_plan(
        file_map, graph, selected, None, existing, [], config,
        resolved_profile=resolved,
    )
    return plan


def test_editing_used_override_invalidates_only_that_extension(tmp_path):
    js = _write(tmp_path, "a.js")
    py = _write(tmp_path, "b.py")
    file_map = {"a.js": _descriptor(js, "a.js"), "b.py": _descriptor(py, "b.py")}

    v1 = _profile("common C", {".js": {"requested_shape": {"description": "J1"}}})
    existing = {
        "a.js": _record(js, "a.js", v1),
        "b.py": _record(py, "b.py", v1),
    }
    # Under v1 both are reusable.
    plan1 = _plan(file_map, existing, v1)
    assert plan1.agent_rels == frozenset()

    # Edit only the .js override; common is unchanged.
    v2 = _profile("common C", {".js": {"requested_shape": {"description": "J2"}}})
    plan2 = _plan(file_map, existing, v2)
    assert "a.js" in plan2.agent_rels          # .js block changed
    assert "b.py" not in plan2.agent_rels       # common unchanged -> still reusable


def test_unreachable_override_changes_nothing(tmp_path):
    py = _write(tmp_path, "b.py")
    js = _write(tmp_path, "a.js")
    file_map = {"b.py": _descriptor(py, "b.py"), "a.js": _descriptor(js, "a.js")}

    v1 = _profile("common C", {".rb": {"requested_shape": {"description": "R1"}}})
    existing = {
        "b.py": _record(py, "b.py", v1),
        "a.js": _record(js, "a.js", v1),
    }
    # Edit the unreachable .rb override; no selected file matches .rb.
    v2 = _profile("common C", {".rb": {"requested_shape": {"description": "R2"}}})
    plan = _plan(file_map, existing, v2)
    assert plan.agent_rels == frozenset()
    # Every selected file kept its prior digest.
    assert v1.file_digest("b.py") == v2.file_digest("b.py")
    assert v1.file_digest("a.js") == v2.file_digest("a.js")


def test_identical_content_reuse_honours_destination_scope(tmp_path):
    # a.js and a.ts have byte-identical content; only .js has an override.
    content = "shared = 1\n"
    js = _write(tmp_path, "a.js", content)
    ts = _write(tmp_path, "a.ts", content)
    assert compute_file_hash(js) == compute_file_hash(ts)

    resolved = _profile("common C", {".js": {"requested_shape": {"description": "Jdesc"}}})
    # A prior record documented a.js under the .js scope.
    existing = {"a.js": _record(js, "a.js", resolved)}
    # Now plan a.ts (identical content) as the destination.
    file_map = {"a.ts": _descriptor(ts, "a.ts")}
    plan = _plan(file_map, existing, resolved, selected={"a.ts"})
    # The a.js record's digest belongs to a different scope, so it is not reused.
    assert "a.ts" in plan.agent_rels
    assert "a.ts" not in plan.identical_reuse_rels


def test_reuse_predicate_rejects_digest_mismatch():
    # _record_is_reusable / _identity_matches remain digest-sensitive.
    base = {
        "hash": "h", "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
        "_prompt_profile_digest": "pp-v1:aaa",
    }
    expected_same = dict(base)
    expected_diff = {**base, "_prompt_profile_digest": "pp-v1:bbb"}
    assert _identity_matches(base, expected_same)
    assert not _identity_matches(base, expected_diff)
    assert _record_is_reusable(base, "h", expected_same)
    assert not _record_is_reusable(base, "h", expected_diff)


def test_common_only_profile_matches_prior_no_extension_digest(tmp_path):
    # A profile with no per_extension yields the same digest for every file,
    # independent of basename (the golden test pins the literal value).
    resolved = _profile("common only", {})
    assert resolved.file_digest("a.py") == resolved.file_digest("a.js")
    assert resolved.file_digest("a.py") == resolved.file_digest("")
