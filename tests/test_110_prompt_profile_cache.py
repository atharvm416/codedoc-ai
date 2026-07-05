"""0.11.0 — cache identity, digest stability, and JSON/Markdown conversion.

Covers Tests #16-#20 of the plan (digest composition, reuse/invalidation, and
round-trip conversion of records produced under a profile).
"""

import json

from codedoc.core.markdown_view import json_from_markdown, markdown_from_json
from codedoc.core.planning import _identity_matches
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    validate_profile,
)
from codedoc.core.record_meta import CACHE_IDENTITY_KEYS, normalized_identity_value

LANGS = frozenset({"python", "java"})


def _to_envelope(raw):
    """Wrap a legacy flat profile dict into the 0.11.3 ``common`` envelope."""
    if not isinstance(raw, dict):
        return raw
    out = {k: v for k, v in raw.items() if k in ("schema_version", "$comment")}
    if isinstance(raw.get("single"), dict) and "common" not in raw["single"]:
        sec = raw["single"]
        common = {k: sec[k] for k in ("fields", "requested_shape") if k in sec}
        new_sec = {"common": common}
        if "per_language" in sec:
            new_sec["per_language"] = sec["per_language"]
        out["single"] = new_sec
    elif "single" in raw:
        out["single"] = raw["single"]
    if isinstance(raw.get("triple"), dict) and "common" not in raw["triple"]:
        sec = raw["triple"]
        agents = ("structure", "dependency", "documentation")
        common, per_lang = {}, {}
        for agent in agents:
            block = sec.get(agent, {})
            common[agent] = {k: block[k] for k in ("fields", "requested_shape") if k in block}
            for lang, ov in (block.get("per_language") or {}).items():
                per_lang.setdefault(lang, {})[agent] = ov
        new_sec = {"common": common}
        if per_lang:
            new_sec["per_language"] = per_lang
        out["triple"] = new_sec
    elif "triple" in raw:
        out["triple"] = raw["triple"]
    return out


def _resolved(raw, mode="single"):
    return ResolvedProfile(
        mode,
        validate_profile(_to_envelope(raw), active_mode=mode, known_languages=LANGS,
                         source="inline", source_path=None),
    )


# ---------------------------------------------------------------------------
# Cache identity (Tests #17, #18)
# ---------------------------------------------------------------------------

def test_digest_in_cache_identity_keys():
    assert "_prompt_profile_digest" in CACHE_IDENTITY_KEYS


def test_absent_default_digest_matches_explicit_no_profile():
    base = {"_analysis_revision": "file-doc-v2", "_analysis_mode": "single"}
    explicit = {**base, "_prompt_profile_digest": NO_PROMPT_PROFILE_DIGEST}
    assert _identity_matches(base, explicit)
    assert _identity_matches(explicit, base)
    # normalization is symmetric
    assert normalized_identity_value("_prompt_profile_digest", base) == NO_PROMPT_PROFILE_DIGEST


def test_active_digest_invalidates_cache():
    base = {"_analysis_revision": "file-doc-v2", "_analysis_mode": "single"}
    assert not _identity_matches(base, {**base, "_prompt_profile_digest": "pp-v1:x"})


def test_composition_with_other_identity_keys_through_single_predicate():
    stored = {
        "_analysis_revision": "file-doc-v2", "_analysis_mode": "single",
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


# ---------------------------------------------------------------------------
# Digest stability and scoping (Tests #16, #17)
# ---------------------------------------------------------------------------

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
    da = _resolved(a).file_digest("python")
    db = _resolved(b).file_digest("python")
    dr = _resolved(reordered).file_digest("python")
    assert da == db          # $comment + schema_version do not affect the digest
    assert da != dr          # field order does affect the digest
    assert da != NO_PROMPT_PROFILE_DIGEST


def test_language_local_change_invalidates_only_that_language():
    raw = {"single": {
        "fields": [{"key": "description", "type": "string", "instruction": "base"}],
        "per_language": {"python": {"fields": [
            {"key": "description", "type": "string", "instruction": "py only"}]}}}}
    resolved = _resolved(raw)
    # java uses the base block; python uses the override -> different digests.
    assert resolved.file_digest("java") != resolved.file_digest("python")

    raw2 = {"single": {
        "fields": [{"key": "description", "type": "string", "instruction": "base"}],
        "per_language": {"python": {"fields": [
            {"key": "description", "type": "string", "instruction": "py CHANGED"}]}}}}
    resolved2 = _resolved(raw2)
    # java digest unchanged; python digest changed.
    assert resolved2.file_digest("java") == resolved.file_digest("java")
    assert resolved2.file_digest("python") != resolved.file_digest("python")


# ---------------------------------------------------------------------------
# JSON/Markdown round-trip with reordered/omitted fields (Test #20)
# ---------------------------------------------------------------------------

def _profile_shaped_view():
    # A record as it would be persisted after a profile omitted role_in_system
    # and reordered prompt fields. The public vocabulary is unchanged, so the
    # deterministic embedded-view round-trip must hold.
    return {
        "schema_version": "1.4",
        "project": {"entry_file": "a.py", "file_count": 1,
                    "languages": ["python"], "folders": ["."]},
        "run": {"files_checked": 1, "files_failed": 0, "files_skipped": 0,
                "files_reused": 0, "files_documented": 1},
        "files": [{
            "path": "a.py", "language": "python", "hash": "abc",
            "description": "kept", "exports": ["E"], "key_concepts": ["k"],
            "_prompt_profile_digest": "pp-v1:abc",
            "_analysis_revision": "file-doc-v2", "_analysis_mode": "single",
        }],
    }


def test_json_markdown_round_trip_preserves_profile_shaped_record():
    view = _profile_shaped_view()
    md = markdown_from_json(json.dumps(view))
    back = json.loads(json_from_markdown(md))
    rec = next(f for f in back["files"] if f["path"] == "a.py")
    assert rec["description"] == "kept"
    assert "role_in_system" not in rec
    # Private cache keys survive the embedded-view round-trip.
    assert rec["_prompt_profile_digest"] == "pp-v1:abc"
    assert rec["_analysis_mode"] == "single"
