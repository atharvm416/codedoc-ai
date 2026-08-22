"""Source-level regressions for the installed-artifact smoke harness.

These catch the two known harness mistakes -- an unsupported config key sent
to a peer that predates it, and a hard import of a peer function that may not
exist -- without needing a real installed peer environment or copying
historical source. Installed predecessor matrices are retained as optional
capabilities (plan section 6.4), not as post-commit release gates.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.contract.package import installed_artifact_smoke as harness


# Top-level configuration keys that official PyPI `0.13.1` accepts, which is
# the Matrix A peer.  That release defines 51 top-level keys in its own
# `DEFAULTS` and rejects every other key through
# `_reject_unknown_keys(data, source=_CONFIG_FILENAME)`, so any key the
# harness writes that is absent from its DEFAULTS aborts the peer run with a
# ConfigError before it does any work.  Listed literally here (plan section
# 6.4) so this stays history-independent: no copied source, no fixture, no
# `git show`, no CI checkout-depth change.  `large_file_strategy` is
# deliberately absent -- it was introduced in 0.14.0.
_MATRIX_A_PEER_ACCEPTED_KEYS = frozenset({
    "entry_file",
    "documentation_scope",
    "output_dir",
    "output_format",
    "analysis_mode",
    "parallel_agents",
    "max_parallel_files",
    "file_retry_attempts",
    "propagate_changes",
    "max_content_chars",
})


def test_matrix_a_ordinary_project_config_is_accepted_by_the_0_13_1_peer(tmp_path):
    """Plan section 6.4: assert the whole emitted key set, not just today's
    known offender. A test that only forbids `large_file_strategy` closes
    this defect and nothing else -- the next post-0.13.1 key added to the
    Matrix A config builder would pass here and fail late when the optional
    installed peer is next exercised. That expensive failure is what this test
    exists to prevent. Extending the allowlist is then a deliberate, reviewable
    edit."""
    project = harness._new_ordinary_project(tmp_path, "matrix-a-check")
    config = json.loads((project / "codedoc.config.json").read_text(encoding="utf-8"))

    unsupported = sorted(set(config) - _MATRIX_A_PEER_ACCEPTED_KEYS)
    assert not unsupported, (
        "Matrix A config contains keys official 0.13.1 does not define and "
        f"would reject as unknown: {unsupported}"
    )
    # The specific key this regression was written for, asserted explicitly so
    # a future allowlist edit can never quietly re-admit it.
    assert "large_file_strategy" not in config


def test_matrix_a_output_format_legs_are_accepted_by_the_0_13_1_peer(tmp_path):
    """Matrix A step 4 builds its own configs per output format, so the
    key-set contract must hold on that path too, not only for the ordinary
    project builder."""
    for fmt in ("json", "md", "both"):
        project = tmp_path / f"fmt-{fmt}"
        project.mkdir()
        project.joinpath("main.py").write_text("def helper(): pass\n", encoding="utf-8")
        harness._write_config(project, output_format=fmt)
        config = json.loads(
            (project / "codedoc.config.json").read_text(encoding="utf-8")
        )
        unsupported = sorted(set(config) - _MATRIX_A_PEER_ACCEPTED_KEYS)
        assert not unsupported, (
            f"Matrix A {fmt} leg config contains keys official 0.13.1 would "
            f"reject: {unsupported}"
        )


def test_factory_skips_attest_provider_execution_when_peer_lacks_it(monkeypatch):
    """attest_provider_execution is itself a newer addition than some
    supported peers. The child-process factory must feature-detect it and
    return the provider unattested rather than raising ImportError/
    AttributeError -- a peer that predates the function also predates the
    execution-attestation verification it exists to satisfy."""
    import codedoc.llm.factory as factory_module

    provider = harness._FrozenProvider()
    factory = harness._factory_for(provider)

    monkeypatch.delattr(factory_module, "attest_provider_execution", raising=True)

    result = factory({"llm_provider": "openai", "model_name": "gpt-4o-mini"})
    assert result is provider


def test_cross_version_requires_exact_candidate_version(tmp_path):
    with pytest.raises(harness.SmokeFailure, match="candidate-version"):
        harness.main(
            [
                "--scenario",
                "cross-version",
                "--peer-python",
                str(tmp_path / "peer-python"),
                "--peer-version",
                "0.14.4",
                "--work",
                str(tmp_path / "work"),
            ]
        )


def test_output_transition_table_is_exhaustive():
    assert set(harness._FORMAT_TRANSITIONS) == {
        ("json", "json"),
        ("md", "md"),
        ("both", "both"),
        ("json", "md"),
        ("md", "json"),
        ("json", "both"),
        ("md", "both"),
    }


def test_transition_inventory_table_covers_every_leg():
    """Every leg the format table walks must declare an exact inventory, and
    no extra leg may be declared, so the two structures cannot drift apart."""
    assert set(harness._TRANSITION_INVENTORIES) == set(harness._FORMAT_TRANSITIONS)


@pytest.mark.parametrize(("start_fmt", "end_fmt"), harness._FORMAT_TRANSITIONS)
def test_transition_inventory_table_matches_real_pipeline_behavior(
    tmp_path, monkeypatch, start_fmt, end_fmt
):
    """Plan section 6.4: the reader-leg inventory is *not* "the file for the
    requested format". A same-stem format switch preserves the previous
    opposite-format sibling, so `json -> md` leaves both files behind.

    Expecting only the requested format made all four opposite-single legs of
    the cross-version format table fail against a real installed peer -- a
    failure no source-suite test could reach, because that table had only
    ever been exercised through the installed matrices. Driving the real
    pipeline in-process and provider-free pins the harness's table to actual
    production behavior, so it cannot drift again silently.
    """
    from codedoc.pipeline import run_pipeline
    from tests.support.providers import SmartFake

    (tmp_path / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    def _config(output_format: str) -> dict:
        return {
            "entry_file": "main.py",
            "documentation_scope": "all",
            "output_dir": "docs",
            "output_format": output_format,
            "parallel_agents": False,
            "propagate_changes": False,
        }

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: SmartFake())
    run_pipeline(tmp_path, _config(start_fmt))
    harness._assert_output_inventory(
        tmp_path, harness._FRESH_FORMAT_INVENTORIES[start_fmt], "creator"
    )

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _cfg: SmartFake())
    run_pipeline(tmp_path, _config(end_fmt))
    harness._assert_output_inventory(
        tmp_path,
        harness._TRANSITION_INVENTORIES[(start_fmt, end_fmt)],
        "reader",
    )


def test_intentional_child_interrupt_has_portable_exit_status():
    def interrupted(_args):
        raise KeyboardInterrupt

    assert harness._invoke_child_cli(interrupted, ["."]) == 130


@pytest.mark.parametrize(
    "diagnostic",
    [
        "error: unrecognized arguments: --not-a-real-option",
        "unknown configuration key: not_a_real_key",
        "ModuleNotFoundError: No module named 'codedoc'",
        "entry file was not found: main.py",
        "invalid configuration syntax",
    ],
)
def test_unrelated_exit_two_cannot_satisfy_recovery_refusal(tmp_path, diagnostic):
    project = tmp_path / "unrelated-exit-two"
    recovery = project / "docs" / "crash_recovery.json"
    recovery.parent.mkdir(parents=True)
    recovery.write_text(harness._legacy_recovery(99), encoding="utf-8")
    project.joinpath(".codedoc-smoke-calls.json").write_text("0", encoding="utf-8")
    before = harness._snapshot(project)
    result = subprocess.CompletedProcess(
        args=["codedoc", "--not-a-real-option"],
        returncode=2,
        stdout="",
        stderr=diagnostic,
    )

    with pytest.raises(harness.SmokeFailure, match="unrelated-exit-2"):
        harness._assert_recovery_specific_refusal(
            project, result, 0, before, "negative-control"
        )


def test_missing_call_count_sidecar_cannot_be_interpreted_as_zero(
    tmp_path, monkeypatch
):
    project = tmp_path / "missing-sidecar"
    project.mkdir()

    def child_deleting_sidecar(
        _project,
        _cli_args,
        *,
        python_exe,
        interrupt_after=None,
        call_count_path=None,
    ):
        del python_exe, interrupt_after
        assert call_count_path is not None
        call_count_path.unlink()
        return subprocess.CompletedProcess(
            args=["codedoc"], returncode=2, stdout="", stderr="recovery schema mismatch"
        )

    monkeypatch.setattr(harness, "_run_child", child_deleting_sidecar)

    with pytest.raises(harness.SmokeFailure, match="sidecar-missing"):
        harness._run_cli(project, ["--version"], python_exe="python")
