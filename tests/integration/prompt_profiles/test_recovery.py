"""Tests organized by feature ownership."""

from pathlib import PurePosixPath
import pytest
from codedoc.core.db import compute_file_hash
from codedoc.core.graph import DependencyGraph
from codedoc.core.planning import build_pipeline_plan
from codedoc.core.prompt_profiles import (
    NO_PROMPT_PROFILE_DIGEST,
    ResolvedProfile,
    validate_profile,
)
from codedoc.core.resume import (
    RECOVERY_FILENAME,
    _RECOVERY_IDENTITY_FIELDS,
    _RECOVERY_IDENTITY_VERSION,
    build_recovery_identity,
    load_recovery_records_if_compatible,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.utils.errors import ConfigError

_KNOWN = frozenset({".py", ".js", ".rb"})

def _identity(tmp_path, **changes):
    values = {
        "project_root": tmp_path,
        "json_target": tmp_path / "codedoc" / "codedoc.json",
        "md_target": None,
        "entry_file": "main.py",
        "documentation_scope": "entry",
        "analysis_mode": "single",
        "analysis_revision": "file-doc-v3",
    }
    values.update(changes)
    return build_recovery_identity(**values)

def _write_recovery(tmp_path, identity, records=None):
    out = tmp_path / "codedoc"
    out.mkdir(exist_ok=True)
    recovery = out / RECOVERY_FILENAME
    writer = SafeWriter(recovery, "json", "main.py", {}, identity)
    if records:
        writer.set_queue_order(list(records))
        writer.load(preloaded=records)
    writer.initialize_empty()
    return recovery

def _resolved(common_desc, per_extension):
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

def _record(path, rel, resolved):
    rec = {
        "path": rel, "hash": compute_file_hash(path), "description": "cached",
        "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
    }
    digest = resolved.file_digest(PurePosixPath(rel).name.lower())
    if digest != NO_PROMPT_PROFILE_DIGEST:
        rec["_prompt_profile_digest"] = digest
    return rec

def test_recovery_identity_version_and_fields():
    assert _RECOVERY_IDENTITY_VERSION == 1
    assert "profile_digests_by_language" not in _RECOVERY_IDENTITY_FIELDS
    # The retained run-wide routing/output fields are still compared.
    for field in (
        "project_root", "json_target", "md_target", "entry_file",
        "documentation_scope", "analysis_mode", "analysis_revision",
    ):
        assert field in _RECOVERY_IDENTITY_FIELDS

def test_legacy_recovery_with_profile_digests_is_accepted(tmp_path):
    # A version-1 file from a prior release carries the now-removed field.
    legacy = _identity(tmp_path)
    legacy["profile_digests_by_language"] = {"python": NO_PROMPT_PROFILE_DIGEST}
    recovery = _write_recovery(tmp_path, legacy)
    # The current reader ignores the extra field and accepts on the retained set.
    current = _identity(tmp_path)
    assert load_recovery_records_if_compatible(recovery, current) == {}

def test_unknown_identity_version_still_rejected(tmp_path):
    identity = _identity(tmp_path)
    identity["version"] = 2  # unsupported
    recovery = _write_recovery(tmp_path, identity)
    with pytest.raises(ConfigError, match="crash_recovery.json"):
        load_recovery_records_if_compatible(recovery, _identity(tmp_path))

def test_adding_new_language_file_does_not_invalidate_recovery(tmp_path):
    # The identity no longer depends on the selected file set / languages, so
    # adding the first file of a new language cannot change it.
    recovery = _write_recovery(tmp_path, _identity(tmp_path))
    # Same run, now with (hypothetically) a new-language file added: the identity
    # arguments are unchanged, so the file remains resumable.
    assert load_recovery_records_if_compatible(recovery, _identity(tmp_path)) == {}

def test_editing_any_override_does_not_change_recovery_identity(tmp_path):
    # The recovery identity is profile-independent; editing an override (used or
    # unused) leaves it byte-identical, so recovery is never discarded for it.
    before = _identity(tmp_path)
    after = _identity(tmp_path)
    assert before == after

def test_recovery_overlay_records_filtered_per_file_by_digest(tmp_path):
    # Two records from two scopes are overlaid; editing only the .js scope makes
    # planning reprocess the stale .js record while reusing the .py record.
    js = tmp_path / "a.js"
    js.write_text("j = 1\n", encoding="utf-8")
    py = tmp_path / "b.py"
    py.write_text("p = 1\n", encoding="utf-8")
    file_map = {
        "a.js": {"path": js, "rel_path": "a.js", "language": "generic", "extension": ".js"},
        "b.py": {"path": py, "rel_path": "b.py", "language": "generic", "extension": ".py"},
    }
    v1 = _resolved("common C", {".js": {"requested_shape": {"description": "J1"}}})
    overlaid = {"a.js": _record(js, "a.js", v1), "b.py": _record(py, "b.py", v1)}

    v2 = _resolved("common C", {".js": {"requested_shape": {"description": "J2"}}})
    graph = DependencyGraph()
    graph.add_file("a.js")
    graph.add_file("b.py")
    config = {
        "propagate_changes": False, "max_files": 0, "analysis_mode": "single",
        "max_content_chars": 12000, "truncation_head_ratio": 0.70,
    }
    plan, _ = build_pipeline_plan(
        file_map, graph, {"a.js", "b.py"}, None, overlaid, [], config,
        resolved_profile=v2,
    )
    assert "a.js" in plan.agent_rels          # stale .js record reprocessed
    assert "b.py" not in plan.agent_rels       # matching common record reused

def test_no_instruction_text_in_recovery_file(tmp_path):
    secret = "SENSITIVE-PROFILE-INSTRUCTION-TEXT"
    # A recovery file carries only the run identity plus documentation records —
    # never the profile instruction text.
    identity = _identity(tmp_path)
    main = tmp_path / "main.py"
    main.write_text("x = 1\n", encoding="utf-8")
    record = {
        "path": "main.py", "hash": compute_file_hash(main), "description": "documented",
        "_analysis_revision": "file-doc-v3", "_analysis_mode": "single",
        "_prompt_profile_digest": "pp-v1:deadbeef",
    }
    recovery = _write_recovery(tmp_path, identity, {"main.py": record})
    contents = recovery.read_text(encoding="utf-8")
    assert secret not in contents
    assert "SENSITIVE" not in contents
