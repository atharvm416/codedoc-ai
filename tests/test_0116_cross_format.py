"""0.11.6 regression coverage for safe cross-format incremental reuse."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from codedoc.core.document import read_codedoc_document, records_by_path
from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST
from codedoc.core.record_meta import ANALYSIS_REVISION, CACHE_IDENTITY_KEYS
from codedoc.core.resume import (
    RECOVERY_FILENAME,
    _load_existing_file_docs,
    build_recovery_identity,
)
from codedoc.core.safe_writer import SafeWriter
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError, OutputError
from tests.test_110_prompt_profile_cli import INLINE, SmartFake


def _config(output_format: str, output_dir: str = "docs") -> dict:
    return {
        "entry_file": "main.py",
        "documentation_scope": "all",
        "output_dir": output_dir,
        "output_format": output_format,
        "parallel_agents": False,
        "propagate_changes": False,
    }


def _first_run(tmp_path: Path, monkeypatch, output_format: str, output_dir="docs"):
    fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: fake)
    stats = run_pipeline(tmp_path, _config(output_format, output_dir))
    return fake, stats


def _forbid_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _cfg: pytest.fail("provider must not be constructed for conversion"),
    )


@pytest.mark.parametrize(
    ("first_format", "second_format", "first_name", "second_name"),
    [
        ("md", "json", "codedoc.md", "codedoc.json"),
        ("json", "md", "codedoc.json", "codedoc.md"),
    ],
)
def test_unchanged_format_switch_converts_without_provider(
    tmp_path, monkeypatch, first_format, second_format, first_name, second_name
):
    (tmp_path / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    first_fake, _ = _first_run(tmp_path, monkeypatch, first_format)
    assert first_fake.doc_calls == 1

    out = tmp_path / "docs"
    stable_sibling = out / first_name
    before = stable_sibling.read_bytes()
    _forbid_provider(monkeypatch)
    stats = run_pipeline(tmp_path, _config(second_format))

    assert (out / second_name).exists()
    assert stable_sibling.read_bytes() == before
    assert stats["checked"] == 0
    assert stats["documentation_calls_attempted"] == 0
    assert stats["attempted_calls"] == 0
    assert not (out / RECOVERY_FILENAME).exists()

    old = records_by_path(read_codedoc_document(stable_sibling))["main.py"]
    new = records_by_path(read_codedoc_document(out / second_name))["main.py"]
    assert old["hash"] == new["hash"]
    for key in CACHE_IDENTITY_KEYS:
        assert old.get(key) == new.get(key)


def test_format_switch_processes_only_changed_file(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("y = 1\n", encoding="utf-8")
    first_fake, _ = _first_run(tmp_path, monkeypatch, "json")
    assert first_fake.doc_calls == 2

    (tmp_path / "helper.py").write_text("y = 2\n", encoding="utf-8")
    second_fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: second_fake)
    stats = run_pipeline(tmp_path, _config("md"))

    assert second_fake.doc_calls == 1
    assert stats["checked"] == 1
    assert stats["skipped"] == 1
    assert (tmp_path / "docs" / "codedoc.md").exists()


def test_profile_identity_change_still_reprocesses_fallback(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")

    fake = SmartFake("SAFE")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: fake)
    config = {**_config("md"), "prompt_profiles": INLINE}
    stats = run_pipeline(tmp_path, config)

    assert fake.review_calls == 1
    assert fake.doc_calls == 1
    assert stats["checked"] == 1


def test_analysis_mode_change_reprocesses_cross_format_fallback(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")

    fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: fake)
    stats = run_pipeline(tmp_path, {**_config("md"), "analysis_mode": "triple"})

    assert stats["checked"] == 1
    assert fake.doc_calls == 3
    record = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert record["_analysis_mode"] == "triple"


def test_truncation_identity_change_reprocesses_cross_format_fallback(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x" * 2000, encoding="utf-8")
    first_fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: first_fake)
    run_pipeline(tmp_path, {**_config("json"), "max_content_chars": 1000})
    assert first_fake.doc_calls == 1

    second_fake = SmartFake()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: second_fake)
    stats = run_pipeline(tmp_path, {**_config("md"), "max_content_chars": 1500})

    assert stats["checked"] == 1
    assert second_fake.doc_calls == 1
    record = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert record["_max_context_revision"] == "truncate-v1:max=1500:head=0.7000"


def test_profile_filtered_fields_and_private_metadata_survive_conversion(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    first_fake = SmartFake("SAFE")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: first_fake)
    config = {**_config("json"), "prompt_profiles": INLINE}
    run_pipeline(tmp_path, config)

    original = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.json")
    )["main.py"]
    assert "functions" not in original
    assert "role_in_system" not in original
    assert "_prompt_profile_digest" in original

    _forbid_provider(monkeypatch)
    stats = run_pipeline(tmp_path, {**config, "output_format": "md"})
    converted = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]

    assert stats["checked"] == 0
    assert "functions" not in converted
    assert "role_in_system" not in converted
    for key in CACHE_IDENTITY_KEYS:
        assert converted.get(key) == original.get(key)


def test_existing_selected_target_is_authoritative_and_sibling_is_not_read(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    foreign = tmp_path / "docs" / "codedoc.md"
    foreign.write_text("not a codedoc document", encoding="utf-8")

    _forbid_provider(monkeypatch)
    stats = run_pipeline(tmp_path, _config("json"))
    assert stats["checked"] == 0
    assert foreign.read_text(encoding="utf-8") == "not a codedoc document"


def test_foreign_fallback_blocks_before_provider(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "docs"
    out.mkdir()
    (out / "codedoc.md").write_text("foreign", encoding="utf-8")
    _forbid_provider(monkeypatch)
    with pytest.raises(ConfigError, match="conversion sibling"):
        run_pipeline(tmp_path, _config("json"))
    assert not (out / "codedoc.json").exists()


def test_missing_both_candidates_returns_no_existing_records(tmp_path):
    assert _load_existing_file_docs(
        tmp_path / "missing.json", tmp_path / "missing.md", "json"
    ) == {}
    assert _load_existing_file_docs(
        tmp_path / "missing.json", tmp_path / "missing.md", "md"
    ) == {}


def test_cross_format_dry_run_is_provider_free_and_read_only(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    _forbid_provider(monkeypatch)

    stats = run_pipeline(tmp_path, {**_config("md"), "dry_run": True})
    out = tmp_path / "docs"
    assert stats["would_call_llm_for"] == 0
    assert stats["estimated_calls"] == 0
    assert not (out / "codedoc.md").exists()
    assert not (out / RECOVERY_FILENAME).exists()


@pytest.mark.parametrize(
    ("first_output", "second_output", "first_name", "second_name"),
    [
        ("docs/report.md", "docs/report.json", "report.md", "report.json"),
        ("docs/report.json", "docs/report.md", "report.json", "report.md"),
    ],
)
def test_named_output_switch_uses_same_stem_sibling(
    tmp_path, monkeypatch, first_output, second_output, first_name, second_name
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, first_name.rsplit(".", 1)[1], first_output)
    unrelated = tmp_path / "docs" / (
        "codedoc.json" if second_name.endswith(".json") else "codedoc.md"
    )
    unrelated.write_text("foreign unrelated file", encoding="utf-8")

    _forbid_provider(monkeypatch)
    stats = run_pipeline(
        tmp_path, _config(second_name.rsplit(".", 1)[1], second_output)
    )
    assert stats["checked"] == 0
    assert (tmp_path / "docs" / first_name).exists()
    assert (tmp_path / "docs" / second_name).exists()
    assert unrelated.read_text(encoding="utf-8") == "foreign unrelated file"


def _write_compatible_md_recovery(tmp_path: Path, replacement: str) -> Path:
    out = tmp_path / "docs"
    stable = out / "codedoc.json"
    record = deepcopy(records_by_path(read_codedoc_document(stable))["main.py"])
    record["description"] = replacement
    identity = build_recovery_identity(
        project_root=tmp_path,
        json_target=None,
        md_target=out / "codedoc.md",
        entry_file="main.py",
        documentation_scope="all",
        analysis_mode="single",
        analysis_revision=ANALYSIS_REVISION,
        profile_digests_by_language={"python": NO_PROMPT_PROFILE_DIGEST},
    )
    recovery = out / RECOVERY_FILENAME
    writer = SafeWriter(recovery, "md", "main.py", {}, identity)
    writer.set_queue_order(["main.py"])
    writer.load(preloaded={"main.py": record})
    writer.initialize_empty()
    return recovery


def test_compatible_recovery_overlays_fallback_and_is_deleted_after_conversion(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    recovery = _write_compatible_md_recovery(tmp_path, "newer recovery record")
    _forbid_provider(monkeypatch)

    stats = run_pipeline(tmp_path, _config("md"))
    final = records_by_path(
        read_codedoc_document(tmp_path / "docs" / "codedoc.md")
    )["main.py"]
    assert stats["checked"] == 0
    assert stats["resumed"] == 1
    assert final["description"] == "newer recovery record"
    assert not recovery.exists()


def test_incompatible_recovery_blocks_even_with_complete_fallback(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    recovery = _write_compatible_md_recovery(tmp_path, "newer recovery record")
    raw = recovery.read_text(encoding="utf-8")
    recovery.write_text(
        raw.replace('"analysis_mode": "single"', '"analysis_mode": "triple"'),
        encoding="utf-8",
    )
    _forbid_provider(monkeypatch)

    with pytest.raises(ConfigError, match="analysis_mode"):
        run_pipeline(tmp_path, _config("md"))
    assert recovery.exists()
    assert not (tmp_path / "docs" / "codedoc.md").exists()


def test_zero_call_conversion_write_failure_preserves_sibling_and_recovery(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    _first_run(tmp_path, monkeypatch, "json")
    stable = tmp_path / "docs" / "codedoc.json"
    before = stable.read_bytes()
    recovery = _write_compatible_md_recovery(tmp_path, "newer recovery record")
    _forbid_provider(monkeypatch)

    import codedoc.pipeline as pipeline_mod

    def fail_write(*_args, **_kwargs):
        raise OutputError(str(tmp_path / "docs" / "codedoc.md"), "forced failure")

    monkeypatch.setattr(pipeline_mod, "write_project_outputs", fail_write)
    with pytest.raises(OutputError, match="forced failure"):
        run_pipeline(tmp_path, _config("md"))
    assert stable.read_bytes() == before
    assert recovery.exists()
    assert not (tmp_path / "docs" / "codedoc.md").exists()


def test_loader_accepts_versionless_and_legacy_fallback_documents(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    legacy_json = fixtures / "legacy_codedoc_13.json"
    legacy_md = fixtures / "legacy_codedoc_14.md"
    assert _load_existing_file_docs(tmp_path / "missing.json", legacy_md, "json")
    assert _load_existing_file_docs(legacy_json, tmp_path / "missing.md", "md")
