"""Source-level regressions for the installed-artifact smoke harness.

These catch the two known harness mistakes -- an unsupported config key sent
to a peer that predates it, and a hard import of a peer function that may not
exist -- without needing a real installed peer environment or copying
historical source. Installed predecessor matrices are retained as optional
capabilities (plan section 6.4), not as post-commit release gates.

They also pin the narrow harness retarget: the origin-bound ``codedoc-ai``
distribution selection, the ``--candidate-version`` requirement for
``--scenario all``, the fixture-aligned serialized-signature acceptance
boundary with its separate 600-character prompt-hint contract, and the
frozen live-validation fixture dry-run scenario (plan sections 7.2.2 / 9.3).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# Origin-bound `codedoc-ai` distribution selection (plan sections 7.2.2 / 9.3;
# hardening for the "a repo-local egg-info shadows the real install" hazard).
# `_prove_installed_origin` runs before any chdir and cannot complete from the
# checkout, so every case below fakes a self-consistent installed environment.
# ---------------------------------------------------------------------------


class _FakeDist:
    """Minimal stand-in for `importlib.metadata.PathDistribution`."""

    def __init__(self, name: str, version: str, dist_info: Path) -> None:
        self._name = name
        self.version = version
        self._path = Path(dist_info)

    @property
    def metadata(self) -> dict:
        return {"Name": self._name}

    def locate_file(self, rel: str) -> Path:
        # Mirrors PathDistribution.locate_file: relative to the metadata
        # directory's parent, i.e. the site-packages root for a real install.
        return self._path.parent / rel


def _fake_installed_env(
    monkeypatch,
    tmp_path: Path,
    *,
    module_version: str = "1.2.3",
    dists=(("codedoc-ai", "1.2.3", "site-packages/codedoc_ai-1.2.3.dist-info"),),
    console_version: str | None = None,
    console_rel: str = "env/codedoc.exe",
    console_missing: bool = False,
    repo_rel: str = "repo",
) -> tuple[Path, Path]:
    """Wire `_prove_installed_origin`'s whole world to a throwaway tree.

    ``dists`` entries are ``(name, version, path-relative-to-tmp_path)``; the
    metadata directory is created so ``_path`` resolution is realistic.
    Returns ``(package_path, console_path)``.
    """
    site_root = tmp_path / "site-packages"
    package_path = (site_root / "codedoc" / "__init__.py")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_text("__version__ = %r\n" % module_version, encoding="utf-8")
    (tmp_path / repo_rel).mkdir(parents=True, exist_ok=True)
    env_bin = tmp_path / "env"
    env_bin.mkdir(parents=True, exist_ok=True)
    (env_bin / "python.exe").write_text("", encoding="utf-8")
    console_path = tmp_path / console_rel
    console_path.parent.mkdir(parents=True, exist_ok=True)
    console_path.write_text("", encoding="utf-8")

    fake_dists = []
    for name, version, rel in dists:
        info_dir = tmp_path / rel
        info_dir.mkdir(parents=True, exist_ok=True)
        fake_dists.append(_FakeDist(name, version, info_dir))

    fake_codedoc = SimpleNamespace(
        __file__=str(package_path), __version__=module_version
    )
    monkeypatch.setitem(sys.modules, "codedoc", fake_codedoc)
    monkeypatch.setattr(harness, "_repository_root", lambda: tmp_path / repo_rel)
    monkeypatch.setattr(
        harness.site, "getsitepackages", lambda: [str(site_root)]
    )
    monkeypatch.setattr(
        harness.importlib_metadata, "distributions", lambda: iter(list(fake_dists))
    )
    monkeypatch.setattr(
        harness.shutil,
        "which",
        lambda _name: None if console_missing else str(console_path),
    )
    monkeypatch.setattr(harness.sys, "executable", str(env_bin / "python.exe"))

    def _fake_run(cmd, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=f"codedoc {console_version or module_version}\n",
            stderr="",
        )

    monkeypatch.setattr(harness.subprocess, "run", _fake_run)
    return package_path.resolve(), console_path.resolve()


def test_prove_installed_origin_binds_distribution_and_forwards_candidate(
    monkeypatch, tmp_path
):
    """Happy path: one co-located `codedoc-ai` distribution that owns the
    imported package, versions agree, and the supplied candidate version is
    honoured."""
    package_path, console_path = _fake_installed_env(monkeypatch, tmp_path)
    assert harness._prove_installed_origin("1.2.3") == (package_path, console_path)


def test_prove_installed_origin_rejects_repository_egg_info_shadow(
    monkeypatch, tmp_path
):
    """A repository-local `codedoc_ai.egg-info` (the real hazard in this
    environment) does not own the imported package and is not under the
    package's site root, so it cannot stand in for the real distribution."""
    _fake_installed_env(
        monkeypatch,
        tmp_path,
        module_version="1.2.3",
        dists=(("codedoc-ai", "9.9.9", "repo/codedoc/codedoc_ai.egg-info"),),
    )
    with pytest.raises(
        harness.SmokeFailure, match="codedoc-ai-distribution-origin-mismatch"
    ):
        harness._prove_installed_origin("1.2.3")


def test_prove_installed_origin_prefers_co_located_distribution_over_shadow(
    monkeypatch, tmp_path
):
    """With both a repo shadow (wrong version) and the real co-located
    distribution present, selection is origin-bound: the co-located one wins
    and the shadow's version never enters the comparison."""
    package_path, console_path = _fake_installed_env(
        monkeypatch,
        tmp_path,
        module_version="1.2.3",
        dists=(
            ("codedoc-ai", "9.9.9", "repo/codedoc/codedoc_ai.egg-info"),
            ("codedoc-ai", "1.2.3", "site-packages/codedoc_ai-1.2.3.dist-info"),
        ),
    )
    assert harness._prove_installed_origin("1.2.3") == (package_path, console_path)


def test_prove_installed_origin_rejects_wrong_site_root_distribution(
    monkeypatch, tmp_path
):
    _fake_installed_env(
        monkeypatch,
        tmp_path,
        dists=(("codedoc-ai", "1.2.3", "other-site/codedoc_ai-1.2.3.dist-info"),),
    )
    with pytest.raises(
        harness.SmokeFailure, match="codedoc-ai-distribution-origin-mismatch"
    ):
        harness._prove_installed_origin("1.2.3")


def test_prove_installed_origin_rejects_ambiguous_distributions(
    monkeypatch, tmp_path
):
    _fake_installed_env(
        monkeypatch,
        tmp_path,
        dists=(
            ("codedoc-ai", "1.2.3", "site-packages/codedoc_ai-1.2.3.dist-info"),
            ("codedoc-ai", "1.2.3", "site-packages/codedoc_ai-1.2.3.egg-info"),
        ),
    )
    with pytest.raises(
        harness.SmokeFailure, match="codedoc-ai-distribution-ambiguous"
    ):
        harness._prove_installed_origin("1.2.3")


def test_prove_installed_origin_rejects_missing_distribution(monkeypatch, tmp_path):
    _fake_installed_env(monkeypatch, tmp_path, dists=())
    with pytest.raises(
        harness.SmokeFailure, match="codedoc-ai-distribution-not-found"
    ):
        harness._prove_installed_origin("1.2.3")


@pytest.mark.parametrize("dist_name", ["codedoc_ai", "Codedoc-AI", "codedoc.ai"])
def test_prove_installed_origin_normalizes_distribution_name(
    monkeypatch, tmp_path, dist_name
):
    package_path, console_path = _fake_installed_env(
        monkeypatch,
        tmp_path,
        dists=((dist_name, "1.2.3", "site-packages/codedoc_ai-1.2.3.dist-info"),),
    )
    assert harness._prove_installed_origin("1.2.3") == (package_path, console_path)


def test_prove_installed_origin_rejects_distribution_version_disagreement(
    monkeypatch, tmp_path
):
    _fake_installed_env(
        monkeypatch,
        tmp_path,
        module_version="1.2.3",
        dists=(("codedoc-ai", "2.2.2", "site-packages/codedoc_ai-2.2.2.dist-info"),),
    )
    with pytest.raises(
        harness.SmokeFailure, match="candidate-module-metadata-version-mismatch"
    ):
        harness._prove_installed_origin(None)


def test_prove_installed_origin_rejects_candidate_version_mismatch(
    monkeypatch, tmp_path
):
    _fake_installed_env(monkeypatch, tmp_path, module_version="1.2.3")
    with pytest.raises(harness.SmokeFailure, match="candidate-version-mismatch"):
        harness._prove_installed_origin("9.9.9")


def test_prove_installed_origin_still_rejects_a_checkout_import(
    monkeypatch, tmp_path
):
    """The pre-existing checkout-import guard is untouched: a package resolved
    from inside the repository root fails before any distribution lookup."""
    _fake_installed_env(monkeypatch, tmp_path)
    monkeypatch.setattr(harness, "_repository_root", lambda: tmp_path)
    with pytest.raises(harness.SmokeFailure, match="installed-origin-check-failed"):
        harness._prove_installed_origin("1.2.3")


def test_prove_installed_origin_still_rejects_a_console_outside_the_environment(
    monkeypatch, tmp_path
):
    """The pre-existing console-origin guard is untouched."""
    _fake_installed_env(
        monkeypatch, tmp_path, console_rel="somewhere-else/codedoc.exe"
    )
    with pytest.raises(
        harness.SmokeFailure, match="console-script-environment-mismatch"
    ):
        harness._prove_installed_origin("1.2.3")


# ---------------------------------------------------------------------------
# `--candidate-version` is required for the complete harness, not only
# `--scenario cross-version`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["--scenario", "all"], []])
def test_scenario_all_requires_candidate_version(argv):
    with pytest.raises(harness.SmokeFailure, match="candidate-version"):
        harness.main(argv)


def test_run_all_signature_requires_a_candidate_version():
    params = inspect.signature(harness._run_all).parameters
    assert list(params) == ["candidate_version"]
    assert params["candidate_version"].default is inspect.Parameter.empty


def test_scenario_all_forwards_candidate_version_to_prove_installed_origin(
    monkeypatch
):
    seen: list[str | None] = []

    def _stub(expected_version=None):
        seen.append(expected_version)
        raise harness.SmokeFailure("stub-origin-reached")

    monkeypatch.setattr(harness, "_prove_installed_origin", _stub)
    with pytest.raises(harness.SmokeFailure, match="stub-origin-reached"):
        harness.main(["--scenario", "all", "--candidate-version", "3.1.4"])
    assert seen == ["3.1.4"]


def test_child_run_dispatch_is_not_blocked_by_the_candidate_requirement(
    monkeypatch, tmp_path
):
    """`--child-run` is dispatched before the scenario logic, so the new
    `--scenario all` requirement must not reach it."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        harness,
        "_child_run",
        lambda project, cli_args: calls.append((project, cli_args)) or 0,
    )
    rc = harness.main(
        ["--child-run", "--project", str(tmp_path), "--", "--version"]
    )
    assert rc == 0
    assert calls == [(tmp_path.resolve(), ["--version"])]


def test_cross_version_candidate_requirement_is_unchanged(tmp_path):
    """Regression guard: the cross-version path still requires its own four
    arguments and is not affected by the `--scenario all` change."""
    with pytest.raises(harness.SmokeFailure, match="candidate-version"):
        harness.main(
            [
                "--scenario",
                "cross-version",
                "--peer-python",
                str(tmp_path / "peer"),
                "--peer-version",
                "0.14.4",
                "--work",
                str(tmp_path / "work"),
            ]
        )


# ---------------------------------------------------------------------------
# Fixture-aligned serialized-signature acceptance boundary, with the separate
# 600-character leaf-prompt hint contract.
# ---------------------------------------------------------------------------


def test_signature_bound_matrix_is_fixture_aligned():
    """The old 552 / 600 / 601 matrix is gone; the boundary now tracks the
    frozen fixture's 1,520-character declaration and the exact hard bound."""
    assert harness._SIGNATURE_BOUND_MATRIX == (
        (1520, False),
        (2000, False),
        (2001, True),
    )
    accepted = {chars for chars, fails in harness._SIGNATURE_BOUND_MATRIX if not fails}
    rejected = {chars for chars, fails in harness._SIGNATURE_BOUND_MATRIX if fails}
    assert accepted == {1520, 2000}
    assert rejected == {2001}
    assert 552 not in accepted and 601 not in rejected

    from codedoc.core.file_division import MAX_LEAF_SYMBOL_SIGNATURE_CHARS

    assert max(accepted) == MAX_LEAF_SYMBOL_SIGNATURE_CHARS
    assert min(rejected) == MAX_LEAF_SYMBOL_SIGNATURE_CHARS + 1


def test_signature_bound_scenario_runs_clean_end_to_end(tmp_path):
    """Behavioural: the retargeted scenario passes against the real
    provider-free pipeline. It would fail with the old matrix because a
    601-character signature is now accepted under the 2,000 bound."""
    harness._scenario_signature_bound(tmp_path)


def test_prompt_signature_hint_constant_is_600_with_its_own_diagnostic(monkeypatch):
    harness._assert_prompt_signature_hint_is_600()

    import codedoc.core.file_division as fd

    monkeypatch.setattr(fd, "MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS", 599)
    with pytest.raises(
        harness.SmokeFailure, match="prompt-signature-hint-chars-not-600"
    ) as excinfo:
        harness._assert_prompt_signature_hint_is_600()
    # Distinct from every serialized-signature-boundary diagnostic.
    assert "did-not-fail-closed" not in str(excinfo.value)
    assert "was-unexpectedly-rejected" not in str(excinfo.value)


def test_signature_scenario_reports_the_hint_regression_before_the_boundary(
    monkeypatch, tmp_path
):
    """The 600-hint check gates the scenario and is not folded into the
    serialized-acceptance loop."""
    import codedoc.core.file_division as fd

    monkeypatch.setattr(fd, "MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS", 601)
    with pytest.raises(
        harness.SmokeFailure, match="prompt-signature-hint-chars-not-600"
    ):
        harness._scenario_signature_bound(tmp_path)


# ---------------------------------------------------------------------------
# Frozen live-validation fixture + the section 9.3 installed dry-run scenario.
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    Path(harness.__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "live_validation"
    / "oversized_signature.py"
)


def test_load_frozen_live_fixture_matches_repository_bytes():
    raw = harness._load_frozen_live_fixture()
    assert raw == _FIXTURE_PATH.read_bytes()
    assert len(raw) == harness._LIVE_FIXTURE_BYTES == 2317
    assert hashlib.sha256(raw).hexdigest() == harness._LIVE_FIXTURE_SHA256
    assert harness._LIVE_FIXTURE_SHA256 == (
        "cfbf8716bcab26e996d6d467997eeb57fe53b172d21a9831d8fd42f623b24da1"
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"", "live-fixture-size-mismatch"),
        (b"x" * harness._LIVE_FIXTURE_BYTES, "live-fixture-hash-mismatch"),
    ],
)
def test_load_frozen_live_fixture_rejects_drift(monkeypatch, tmp_path, payload, reason):
    fake = tmp_path / "tests" / "fixtures" / "live_validation"
    fake.mkdir(parents=True)
    (fake / "oversized_signature.py").write_bytes(payload)
    monkeypatch.setattr(harness, "_repository_root", lambda: tmp_path)
    with pytest.raises(harness.SmokeFailure, match=reason):
        harness._load_frozen_live_fixture()


def test_load_frozen_live_fixture_rejects_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_repository_root", lambda: tmp_path)
    with pytest.raises(harness.SmokeFailure, match="live-fixture-missing"):
        harness._load_frozen_live_fixture()


def test_live_fixture_dry_run_scenario_runs_clean(tmp_path):
    harness._scenario_live_fixture_dry_run(tmp_path)
    written = json.loads(
        (tmp_path / "live-fixture-dry-run" / "codedoc.config.json").read_text(
            encoding="utf-8"
        )
    )
    assert "response_correction_enabled" not in written
    assert written["max_content_chars"] == 1000
    assert written["large_file_strategy"] == "split"
    assert written["dry_run"] is True
    assert written["max_planned_calls"] == 5
    # A dry run leaves no output or recovery behind.
    assert not (tmp_path / "live-fixture-dry-run" / "docs").exists()


def test_live_fixture_dry_run_topology_matches_real_pipeline(tmp_path, monkeypatch):
    """Pins `_LIVE_FIXTURE_DRY_RUN_TOPOLOGY` to actual production behaviour by
    driving the real config-file loading + planning path independently, with
    the response-correction key absent and provider construction forbidden."""
    from codedoc import pipeline

    project = tmp_path / "independent"
    project.mkdir()
    project.joinpath("main.py").write_bytes(harness._load_frozen_live_fixture())
    harness._write_config(
        project,
        large_file_strategy="split",
        max_content_chars=1000,
        allow_partial=False,
        dry_run=True,
        max_planned_calls=5,
    )

    def _forbidden(_cfg):
        pytest.fail("dry-run planning constructed a provider")

    monkeypatch.setattr(pipeline, "create_provider", _forbidden)
    stats = pipeline.run_pipeline(project, {})

    assert stats.get("response_correction_enabled") is True
    for key, expected in harness._LIVE_FIXTURE_DRY_RUN_TOPOLOGY.items():
        assert stats.get(key, "<missing>") == expected, key


def test_live_fixture_dry_run_scenario_is_wired_into_run_all():
    source = inspect.getsource(harness._run_all)
    assert "_scenario_live_fixture_dry_run(neutral_root)" in source


def test_retargeted_harness_helpers_hardcode_no_release_version():
    """No bare release triple is baked into the reusable helpers, comments,
    docstrings, or error strings this retarget touched. The frozen fixture
    SHA-256 is content-addressed, not a version, and lives in a module
    constant rather than any of these bodies."""
    for symbol in (
        "_run_all",
        "_scenario_signature_bound",
        "_assert_prompt_signature_hint_is_600",
        "_scenario_live_fixture_dry_run",
        "_load_frozen_live_fixture",
        "_origin_bound_distribution",
        "_canonical_distribution_name",
        "_prove_installed_origin",
        "main",
    ):
        body = inspect.getsource(getattr(harness, symbol))
        assert not re.search(r"\b\d+\.\d+\.\d+\b", body), symbol
