"""Tests organized by feature ownership."""
# ruff: noqa: F811

from tests.support.prompt_profile_runs import project  # noqa: F401, F811


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
from tests.support.profiles import INLINE
from tests.support.providers import SmartFake
from tests.support.prompt_profile_runs import _run
from codedoc.core.record_meta import (
    CACHE_IDENTITY_KEYS,
    expected_ordinary_path_identity,
    normalized_identity_value,
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
        "language": "generic",
        "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
        "_ordinary_path_identity": expected_ordinary_path_identity(rel),
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
        "path": "a.py",
        "hash": "h", "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
        "language": "generic",
        "_prompt_profile_digest": "pp-v1:aaa",
    }
    expected_same = dict(base)
    expected_diff = {**base, "_prompt_profile_digest": "pp-v1:bbb"}
    assert _identity_matches(base, expected_same)
    assert not _identity_matches(base, expected_diff)
    assert _record_is_reusable(base, "h", expected_same, "generic", rel_path="a.py")
    assert not _record_is_reusable(base, "h", expected_diff, "generic", rel_path="a.py")
    assert not _record_is_reusable(base, "h", expected_same, "python", rel_path="a.py")
    # A record whose stored path differs from the destination is refused even
    # when every other field (hash, language, cache identity) matches.
    assert not _record_is_reusable(base, "h", expected_same, "generic", rel_path="b.py")

def test_common_only_profile_matches_prior_no_extension_digest(tmp_path):
    # A profile with no per_extension yields the same digest for every file,
    # independent of basename (the golden test pins the literal value).
    resolved = _profile("common only", {})
    assert resolved.file_digest("a.py") == resolved.file_digest("a.js")
    assert resolved.file_digest("a.py") == resolved.file_digest("")

def test_first_activation_invalidates_then_reuses(monkeypatch, project):
    _run(monkeypatch, project, {"entry_file": "main.py"}, SmartFake())
    fake2 = SmartFake("SAFE")
    s2 = _run(monkeypatch, project, {"entry_file": "main.py", "prompt_profiles": INLINE}, fake2)
    assert s2["checked"] == 1 and fake2.doc_calls == 1  # reprocessed once on activation
    fake3 = SmartFake("SAFE")
    s3 = _run(monkeypatch, project, {"entry_file": "main.py", "prompt_profiles": INLINE}, fake3)
    assert s3["checked"] == 0 and fake3.doc_calls == 0 and fake3.review_calls == 0  # reused
    assert s3["documentation_calls_attempted"] == 0
    assert s3["prompt_customization_security_review_calls_attempted"] == 0

EXTS = frozenset({".py", ".java"})

def _to_envelope(raw):
    """Wrap a legacy flat profile dict into the ``common`` envelope."""
    if not isinstance(raw, dict):
        return raw
    out = {k: v for k, v in raw.items() if k in ("schema_version", "$comment")}
    if isinstance(raw.get("single"), dict) and "common" not in raw["single"]:
        sec = raw["single"]
        common = {k: sec[k] for k in ("fields", "requested_shape") if k in sec}
        new_sec = {"common": common}
        if "per_extension" in sec:
            new_sec["per_extension"] = sec["per_extension"]
        out["single"] = new_sec
    elif "single" in raw:
        out["single"] = raw["single"]
    if isinstance(raw.get("triple"), dict) and "common" not in raw["triple"]:
        sec = raw["triple"]
        agents = ("structure", "dependency", "documentation")
        common, per_ext = {}, {}
        for agent in agents:
            block = sec.get(agent, {})
            common[agent] = {k: block[k] for k in ("fields", "requested_shape") if k in block}
            for ext, ov in (block.get("per_extension") or {}).items():
                per_ext.setdefault(ext, {})[agent] = ov
        new_sec = {"common": common}
        if per_ext:
            new_sec["per_extension"] = per_ext
        out["triple"] = new_sec
    elif "triple" in raw:
        out["triple"] = raw["triple"]
    return out

def _resolved(raw, mode="single"):
    return ResolvedProfile(
        mode,
        validate_profile(_to_envelope(raw), active_mode=mode, known_extensions=EXTS,
                         source="inline", source_path=None),
    )

def test_digest_in_cache_identity_keys():
    assert "_prompt_profile_digest" in CACHE_IDENTITY_KEYS

def test_absent_default_digest_matches_explicit_no_profile():
    base = {"_analysis_revision": "file-doc-v3", "_analysis_mode": "single"}
    explicit = {**base, "_prompt_profile_digest": NO_PROMPT_PROFILE_DIGEST}
    assert _identity_matches(base, explicit)
    assert _identity_matches(explicit, base)
    # normalization is symmetric
    assert normalized_identity_value("_prompt_profile_digest", base) == NO_PROMPT_PROFILE_DIGEST

def test_active_digest_invalidates_cache():
    base = {"_analysis_revision": "file-doc-v3", "_analysis_mode": "single"}
    assert not _identity_matches(base, {**base, "_prompt_profile_digest": "pp-v1:x"})

def test_composition_with_other_identity_keys_through_single_predicate():
    stored = {
        "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
        "_max_context_revision": "truncate-v1:max=12000:head=0.7000",
        "_prompt_profile_digest": "pp-v1:abc",
    }
    assert _identity_matches(stored, dict(stored))
    # Any single differing cache key blocks reuse.
    for key, other in [
        ("_analysis_mode", "triple"),
        ("_max_context_revision", "truncate-v1:max=9000:head=0.7000"),
        ("_prompt_profile_digest", "pp-v1:def"),
    ]:
        assert not _identity_matches(stored, {**stored, key: other})

def test_digest_ignores_comment_and_schema_version_but_tracks_order():
    a = {"schema_version": 1, "single": {"fields": [
        {"key": "description", "type": "string", "instruction": "d"},
        {"key": "exports", "type": "string_list", "instruction": "e"}]}}
    b = {"$comment": "ignored", "single": {"fields": [
        {"key": "description", "type": "string", "instruction": "d"},
        {"key": "exports", "type": "string_list", "instruction": "e"}]}}
    reordered = {"single": {"fields": [
        {"key": "exports", "type": "string_list", "instruction": "e"},
        {"key": "description", "type": "string", "instruction": "d"}]}}
    da = _resolved(a).file_digest("a.py")
    db = _resolved(b).file_digest("a.py")
    dr = _resolved(reordered).file_digest("a.py")
    assert da == db          # $comment + schema_version do not affect the digest
    assert da != dr          # field order does affect the digest
    assert da != NO_PROMPT_PROFILE_DIGEST

def test_extension_local_change_invalidates_only_that_extension():
    raw = {"single": {
        "fields": [{"key": "description", "type": "string", "instruction": "base"}],
        "per_extension": {".py": {"fields": [
            {"key": "description", "type": "string", "instruction": "py only"}]}}}}
    resolved = _resolved(raw)
    # a.java uses the base block; a.py uses the override -> different digests.
    assert resolved.file_digest("a.java") != resolved.file_digest("a.py")

    raw2 = {"single": {
        "fields": [{"key": "description", "type": "string", "instruction": "base"}],
        "per_extension": {".py": {"fields": [
            {"key": "description", "type": "string", "instruction": "py CHANGED"}]}}}}
    resolved2 = _resolved(raw2)
    # a.java digest unchanged; a.py digest changed.
    assert resolved2.file_digest("a.java") == resolved.file_digest("a.java")
    assert resolved2.file_digest("a.py") != resolved.file_digest("a.py")
