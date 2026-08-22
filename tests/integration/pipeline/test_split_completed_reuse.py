"""Current same-path completed split reuse (section 8)."""

from __future__ import annotations

import json

import pytest

from codedoc.core.file_division import SplitCapacityBlocked
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import ConfigError
from tests.support.providers import SmartFake


def _large_source() -> str:
    return "".join(
        f"def function_{index}():\n    return {index}\n\n"
        for index in range(220)
    )


def _config(**overrides) -> dict:
    return {
        "entry_file": "main.py",
        "analysis_mode": "single",
        "large_file_strategy": "split",
        "max_content_chars": 2000,
        "parallel_agents": False,
        "max_parallel_files": 1,
        "propagate_changes": False,
        **overrides,
    }


def _establish_completed_split(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text(
        _large_source(), encoding="utf-8", newline=""
    )
    providers: list[SmartFake] = []

    def factory(_config):
        provider = SmartFake()
        providers.append(provider)
        return provider

    monkeypatch.setattr("codedoc.pipeline.create_provider", factory)
    first = run_pipeline(tmp_path, _config())
    assert first["checked"] == 1
    assert len(providers) == 1
    return providers


def test_unchanged_completed_split_is_zero_call_same_path_reuse(
    tmp_path, monkeypatch
) -> None:
    providers = _establish_completed_split(tmp_path, monkeypatch)

    second = run_pipeline(tmp_path, _config())

    assert len(providers) == 1
    assert second["checked"] == 0
    assert second["skipped"] == 1
    assert second["total_calls_planned"] == 0
    assert second["split_completed_files_reused"] == 1
    assert second["split_partial_files_resumed"] == 0
    assert second["split_unpaid_nodes"] == 0
    payload = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    assert payload["last_run"]["split_completed_files_reused"] == 1
    assert "_split_reuse_contract" not in payload["files"][0]
    assert payload["files"][0]["_large_file_identity"].startswith("large-file-v3:")


def test_completed_split_reuse_is_excluded_from_paid_file_caps(
    tmp_path, monkeypatch
) -> None:
    providers = _establish_completed_split(tmp_path, monkeypatch)

    # A real run under caps of one must not trip either cap: the reused file
    # never enters the paid-file candidate population or the call manifest, so
    # planning raises no ConfigError and constructs no provider.
    second = run_pipeline(
        tmp_path,
        _config(max_files=1, max_planned_calls=1),
    )

    # Had the reused file been counted, its 12 leaf calls plus final synthesis
    # would have exceeded max_planned_calls=1 and aborted planning with a
    # ConfigError before any provider was constructed.
    assert len(providers) == 1
    assert second["checked"] == 0
    assert second["skipped"] == 1
    assert second["total_calls_planned"] == 0
    assert second["max_planned_calls_exceeded"] is False
    assert second["split_completed_files_reused"] == 1
    assert second["split_unpaid_nodes"] == 0


def test_completed_split_reuse_is_excluded_from_the_max_files_population(
    tmp_path, monkeypatch
) -> None:
    """Section 8: a reused completed split record is excluded from `max_files`.

    This is the discriminating arm of the paid-file cap contract, and a
    single-file project cannot supply it.  `max_files_exceeded` is
    `len(agent_candidate_rels) > max_files`, so with only one candidate the
    comparison is false for every positive cap and the run succeeds whether or
    not the reused file was counted.  Two selected files are required.

    Here `main.py` is a reusable completed split record and `helper.py` is
    newly selected work that genuinely needs a provider call, so the correct
    candidate population is exactly one.  Were the reused file incorrectly
    included, there would be two documentation-call candidates against
    `max_files=1`; the paid-file safety cap in `codedoc/pipeline.py` would then
    raise `ConfigError` before writer initialisation and before provider
    construction, and this test would fail at `run_pipeline`.  Completing the
    run at all is therefore the proof, and the counts below pin down which file
    was paid for.

    `max_planned_calls` is deliberately left unset so that nothing here depends
    on the call cap; that contract is asserted separately by
    `test_completed_split_reuse_is_excluded_from_paid_file_caps`.
    """
    providers = _establish_completed_split(tmp_path, monkeypatch)

    # A second, newly selected ordinary file that genuinely requires paid work.
    (tmp_path / "helper.py").write_text(
        "def helper_one():\n    return 1\n", encoding="utf-8", newline=""
    )

    second = run_pipeline(
        tmp_path,
        _config(max_files=1, documentation_scope="all"),
    )

    # Exactly one newly paid/documented file, and the split record reused.
    assert second["checked"] == 1
    assert second["skipped"] == 1
    assert second["split_completed_files_reused"] == 1
    assert second["split_partial_files_resumed"] == 0
    assert second["split_unpaid_nodes"] == 0
    # One provider for the first run, one for this run's single paid file.
    assert len(providers) == 2

    payload = json.loads(
        (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    )
    records = {record["path"]: record for record in payload["files"]}
    assert set(records) == {"main.py", "helper.py"}
    # main.py kept its completed split identity rather than being re-executed.
    assert records["main.py"]["_large_file_identity"].startswith("large-file-v3:")
    assert payload["last_run"]["split_completed_files_reused"] == 1


def test_explicit_force_bypasses_completed_split_reuse(tmp_path, monkeypatch) -> None:
    providers = _establish_completed_split(tmp_path, monkeypatch)

    forced = run_pipeline(tmp_path, _config(force_files=["main.py"]))

    assert len(providers) == 2
    assert forced["checked"] == 1
    assert forced["split_completed_files_reused"] == 0
    assert forced["split_unpaid_nodes"] > 0


def test_new_capacity_block_is_checked_before_completed_reuse(
    tmp_path, monkeypatch
) -> None:
    _establish_completed_split(tmp_path, monkeypatch)
    stable_path = tmp_path / "codedoc" / "codedoc.json"
    stable_before = stable_path.read_bytes()

    def blocked_plan(**_kwargs):
        raise SplitCapacityBlocked("main.py", "chunk-cap")

    monkeypatch.setattr("codedoc.core.planning.build_division_plan", blocked_plan)
    monkeypatch.setattr(
        "codedoc.pipeline.create_provider",
        lambda _config: pytest.fail("capacity-blocked reuse created a provider"),
    )

    with pytest.raises(ConfigError, match="chunk-cap"):
        run_pipeline(tmp_path, _config())

    assert stable_path.read_bytes() == stable_before
    assert not (tmp_path / "codedoc" / "crash_recovery.json").exists()
