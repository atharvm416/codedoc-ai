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
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.contract.package import installed_artifact_smoke as harness
from tests.support.structure_extra import requires_structure_pack


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


def test_run_all_signature_requires_a_candidate_version_and_structure_profile():
    params = inspect.signature(harness._run_all).parameters
    assert list(params) == ["candidate_version", "structure_profile"]
    assert all(
        param.default is inspect.Parameter.empty for param in params.values()
    )


def test_scenario_all_forwards_candidate_version_to_prove_installed_origin(
    monkeypatch
):
    seen: list[str | None] = []

    def _stub(expected_version=None):
        seen.append(expected_version)
        raise harness.SmokeFailure("stub-origin-reached")

    monkeypatch.setattr(harness, "_prove_installed_origin", _stub)
    with pytest.raises(harness.SmokeFailure, match="stub-origin-reached"):
        harness.main(
            [
                "--scenario",
                "all",
                "--candidate-version",
                "3.1.4",
                "--structure-profile",
                "structure",
            ]
        )
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


def test_live_fixture_dry_run_scenario_runs_clean_structure_profile(tmp_path):
    """The developer/CI interpreter pins ``tree-sitter-language-pack==0.13.0``
    (the ``dev``/``structure`` extra), so the in-process live-fixture scenario
    certifies the structure profile here. The base profile is certified in an
    isolated parser-absent subprocess below."""
    harness._scenario_live_fixture_dry_run(tmp_path, "structure")
    written = json.loads(
        (tmp_path / "live-fixture-dry-run-structure" / "codedoc.config.json").read_text(
            encoding="utf-8"
        )
    )
    assert "response_correction_enabled" not in written
    assert written["max_content_chars"] == 1000
    assert written["large_file_strategy"] == "split"
    assert written["dry_run"] is True
    # The structure profile authorizes five initial calls.
    assert written["max_planned_calls"] == 5
    # A dry run leaves no output or recovery behind.
    assert not (tmp_path / "live-fixture-dry-run-structure" / "docs").exists()


def test_live_fixture_structure_topology_matches_real_pipeline(tmp_path, monkeypatch):
    """Pins the structure-profile expectation to actual production behaviour by
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
    expected = harness._live_fixture_topology("structure")
    for key, value in expected.items():
        assert stats.get(key, "<missing>") == value, key
    assert (
        stats.get("call_manifest_digest")
        == harness._LIVE_FIXTURE_PROFILE_PLAN_DIGEST["structure"]
    )


def test_live_fixture_dry_run_scenario_is_wired_into_run_all():
    source = inspect.getsource(harness._run_all)
    assert "_scenario_live_fixture_dry_run(neutral_root, structure_profile)" in source
    assert "_verify_structure_profile(structure_profile)" in source


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


# ===========================================================================
# F-1 -- explicit base/structure installed-harness certification profiles
# (plan sections 1.1, 3.5, 4.1-4.2, 5.1-5.3, 8.A-8.C, 9.1 items 1-6/9,
# 11 items 1-6, 12).
# ===========================================================================

_REPO_ROOT = Path(harness.__file__).resolve().parents[3]
_HARNESS_FILE = Path(harness.__file__).resolve()

# The five provider-key variable names the harness requires to be absent for
# any executable candidate run (plan section 9.3).
_PROVIDER_KEY_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "LLM_API_KEY",
)


def _provider_free_env(extra_path_dir: Path | None = None) -> dict[str, str]:
    """A copy of the current environment with every provider-key variable
    removed *by name only* (their values are never read, printed, or stored)
    and inherited ``PYTHONPATH`` dropped. Optionally prepend an interpreter's
    own bin dir(s) to ``PATH`` (``extra_path_dir`` and, if present, its
    ``Scripts``/``bin`` sibling) so that interpreter's own console script
    resolves ahead of any unrelated one on the ambient path."""
    env = {k: os.environ[k] for k in os.environ if k not in _PROVIDER_KEY_NAMES}
    env.pop("PYTHONPATH", None)
    if extra_path_dir is not None:
        prepend = [str(extra_path_dir)]
        for sub in ("Scripts", "bin"):
            candidate = Path(extra_path_dir) / sub
            if candidate.is_dir():
                prepend.append(str(candidate))
        env["PATH"] = os.pathsep.join([*prepend, env.get("PATH", "")])
    return env


# --- controlled (SIMULATED) parser isolation ------------------------------
#
# This preamble does NOT create a genuinely parser-absent installed
# environment. It is process-level isolation only: a ``sys.meta_path`` import
# blocker plus an ``importlib.metadata`` veto, so production
# ``PARSER_PACKAGE_VERSION`` computes ``not-installed`` at first import inside
# ONE child interpreter that otherwise still has the distribution on disk.
# Genuine base/structure package-state evidence comes from the explicit,
# non-collected harness commands in plan section 10 -- never from this.
_SIMULATED_PARSER_ABSENCE_PREAMBLE = """
import importlib.abc as _iab
import importlib.metadata as _im
import re as _re
import sys as _sys

_BLOCKED = {"tree_sitter_language_pack", "tree_sitter", "tree_sitter_languages"}


class _ParserBlocker(_iab.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".", 1)[0] in _BLOCKED:
            raise ModuleNotFoundError(name, name=name)
        return None


for _m in [m for m in list(_sys.modules) if m.split(".", 1)[0] in _BLOCKED]:
    del _sys.modules[_m]
_sys.meta_path.insert(0, _ParserBlocker())

_real_version = _im.version
_real_distribution = _im.distribution


def _norm(value):
    return _re.sub(r"[-_.]+", "-", str(value).strip()).lower()


def _blocked_version(name):
    if _norm(name) == "tree-sitter-language-pack":
        raise _im.PackageNotFoundError(name)
    return _real_version(name)


def _blocked_distribution(name):
    if _norm(name) == "tree-sitter-language-pack":
        raise _im.PackageNotFoundError(name)
    return _real_distribution(name)


_im.version = _blocked_version
_im.distribution = _blocked_distribution
"""


def _run_simulated_isolation_subprocess(
    body: str, *, simulate_parser_absence: bool
) -> subprocess.CompletedProcess[str]:
    """Run *body* in a fresh interpreter with the repository root on
    ``sys.path`` (so ``import tests.contract.package.installed_artifact_smoke``
    works) and, optionally, the SIMULATED parser-absence preamble first.

    This is deterministic source-contract coverage, not installed-candidate
    evidence: it deliberately imports ``codedoc`` from the checkout and never
    calls ``_prove_installed_origin``. It proves ``_verify_structure_profile``
    and the live-fixture scenario logic against a controlled process state.
    """
    preamble = (
        _SIMULATED_PARSER_ABSENCE_PREAMBLE if simulate_parser_absence else ""
    )
    script = preamble + "\n" + textwrap.dedent(body)
    env = {**_provider_free_env(), "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="backslashreplace",
        env=env,
        cwd=str(_REPO_ROOT),
        check=False,
    )


# --- self-certification capability probe --------------------------------
#
# This decides ONLY whether the CURRENT source-test interpreter can execute
# the real redirected-verbose / exit-fidelity `--child-run` branch (its
# `codedoc` must resolve outside the checkout). It is NOT candidate
# certification: it asserts NO release version, supplies NO expected version
# to `_prove_installed_origin`, and produces NO installed-candidate evidence.
# Genuine base/structure installed runs are non-collected harness commands
# whose interpreter paths belong in execution evidence, never in this file
# (plan section 10, lines 1112-1118).
_SELF_CERT_PROBE = textwrap.dedent(
    """
    import importlib.util, json
    _spec = importlib.util.spec_from_file_location(
        "iasmoke", {harness_file!r}
    )
    _h = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_h)
    out = {{}}
    try:
        # No expected version: internal origin consistency only.
        pkg, _console = _h._prove_installed_origin()
        out["origin_ok"] = True
        out["package_origin"] = str(pkg)
    except _h.SmokeFailure as exc:
        out["origin_ok"] = False
        out["origin_error"] = str(exc)
    print(json.dumps(out))
    """
)


def _interpreter_own_console(interpreter: Path) -> Path:
    """The console script that belongs to *interpreter*'s environment: beside
    it (venv layout) or under its ``Scripts``/``bin`` sibling (base layout).
    Falls back to the beside-it path when none is present on disk yet. This is
    derived from the interpreter's location, never from an ambient PATH
    lookup, so a foreign `codedoc.exe` can never be selected."""
    name = "codedoc.exe" if os.name == "nt" else "codedoc"
    for directory in (
        interpreter.parent,
        interpreter.parent / "Scripts",
        interpreter.parent / "bin",
    ):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return interpreter.parent / name


def _current_interpreter_can_self_certify() -> bool:
    """True only when a neutral subprocess under ``sys.executable`` -- its own
    bin first on PATH, no inherited PYTHONPATH, cwd outside the repo -- passes
    ``_prove_installed_origin()`` (no expected version) and its ``codedoc``
    resolves outside the checkout. False for the ordinary source interpreter,
    whose ``codedoc`` resolves into the repository."""
    script = _SELF_CERT_PROBE.format(harness_file=str(_HARNESS_FILE))
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="backslashreplace",
            env=_provider_free_env(Path(sys.executable).parent),
            cwd=str(_REPO_ROOT.parent),
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0 or not result.stdout.strip():
        return False
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False
    if not payload.get("origin_ok"):
        return False
    return not harness._is_within(
        Path(payload["package_origin"]).resolve(), _REPO_ROOT
    )


# --- argument contract -----------------------------------------------------


def test_scenario_all_requires_structure_profile():
    """Omitting the profile for ``--scenario all`` is a stable fail-closed
    harness error, raised before any installed-origin or planning work."""
    with pytest.raises(
        harness.SmokeFailure, match="scenario-all-requires-structure-profile"
    ):
        harness.main(["--scenario", "all", "--candidate-version", "1.2.3"])


def test_scenario_all_rejects_an_unknown_structure_profile():
    with pytest.raises(SystemExit):
        harness.main(
            [
                "--scenario",
                "all",
                "--candidate-version",
                "1.2.3",
                "--structure-profile",
                "hybrid",
            ]
        )


@pytest.mark.parametrize("profile", ["base", "structure"])
def test_structure_profile_accepts_exactly_the_closed_pair(profile):
    """The two valid choices parse; each then fails later at the
    installed-origin / profile checks, never at argument parsing."""
    assert set(harness._STRUCTURE_PROFILES) == {"base", "structure"}
    with pytest.raises(harness.SmokeFailure):
        harness.main(
            [
                "--scenario",
                "all",
                "--candidate-version",
                "1.2.3",
                "--structure-profile",
                profile,
            ]
        )


def test_child_run_is_not_blocked_by_the_structure_profile_requirement(
    monkeypatch, tmp_path
):
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


def test_cross_version_path_ignores_the_structure_profile_selector(tmp_path):
    """``--structure-profile`` is a ``--scenario all`` concern; the
    cross-version branch still requires only its own four arguments."""
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
                "--structure-profile",
                "base",
            ]
        )


# --- profile verification (mismatch fails before planning) ----------------


def test_verify_structure_profile_rejects_unknown_value():
    with pytest.raises(harness.SmokeFailure, match="unknown-structure-profile"):
        harness._verify_structure_profile("dev")


def test_verify_structure_profile_accepts_structure_on_the_dev_interpreter():
    """The dev/CI interpreter has the pinned extra, so 'structure' verifies
    and 'base' fails closed here -- the reverse holds in the parser-absent
    subprocess proof below."""
    facts = harness._verify_structure_profile("structure")
    assert facts == {
        "parser_distribution": "tree-sitter-language-pack",
        "structure_profile": "structure",
        "parser_identity": harness._STRUCTURE_PROFILE_PARSER_VERSION,
    }
    with pytest.raises(
        harness.SmokeFailure, match="structure-profile-base-requires-absent-parser"
    ):
        harness._verify_structure_profile("base")


def test_verify_structure_profile_records_only_bounded_parser_facts():
    """The probe reports parser distribution name, requested profile, and the
    normalized identity -- nothing else, no unrelated package/env enumeration."""
    facts = harness._verify_structure_profile("structure")
    assert set(facts) == {
        "parser_distribution",
        "structure_profile",
        "parser_identity",
    }


def test_verify_structure_profile_rejects_a_lied_about_parser_version(monkeypatch):
    """Plan section 11.3: claiming the wrong installed parser version must be
    rejected, in both directions."""
    monkeypatch.setattr(
        harness, "_installed_parser_identity", lambda: ("0.12.9", "0.12.9", True)
    )
    with pytest.raises(
        harness.SmokeFailure, match="structure-profile-requires-pinned-parser"
    ):
        harness._verify_structure_profile("structure")

    monkeypatch.setattr(
        harness, "_installed_parser_identity", lambda: ("not-installed", None, True)
    )
    with pytest.raises(
        harness.SmokeFailure, match="structure-profile-base-requires-absent-parser"
    ):
        harness._verify_structure_profile("base")

    monkeypatch.setattr(
        harness,
        "_installed_parser_identity",
        lambda: (harness._STRUCTURE_PROFILE_PARSER_VERSION, None, False),
    )
    with pytest.raises(
        harness.SmokeFailure, match="structure-profile-requires-pinned-parser"
    ):
        harness._verify_structure_profile("structure")


# --- SIMULATED parser isolation (deterministic source-contract coverage) --
#
# These use `_SIMULATED_PARSER_ABSENCE_PREAMBLE`: process-level import/metadata
# isolation, NOT a genuinely parser-absent installed environment. They import
# `codedoc` from the checkout and never call `_prove_installed_origin`. Their
# value is deterministic proof of `_verify_structure_profile` behaviour and the
# live-fixture scenario logic (plan section 5.3.1). Genuine parser-absent /
# parser-pinned package-state evidence comes from the non-collected harness
# commands recorded in execution evidence (plan section 10), not from here.


def test_base_profile_verifies_and_plans_topology_under_simulated_absence():
    """`_verify_structure_profile("base")` accepts, `"structure"` fails closed,
    and the base four-chunk / six-call topology plans end to end when the
    parser is made unavailable at the process level."""
    result = _run_simulated_isolation_subprocess(
        """
        import tempfile
        from pathlib import Path

        import tests.contract.package.installed_artifact_smoke as h

        parser_identity, distribution_version, module_importable = (
            h._installed_parser_identity()
        )
        assert parser_identity == "not-installed", parser_identity
        assert distribution_version is None, distribution_version
        assert module_importable is False, module_importable

        h._verify_structure_profile("base")
        try:
            h._verify_structure_profile("structure")
        except h.SmokeFailure:
            pass
        else:
            raise SystemExit("structure verified under simulated parser absence")

        h._scenario_live_fixture_dry_run(Path(tempfile.mkdtemp()), "base")
        print("BASE_SIMULATED_OK")
        """,
        simulate_parser_absence=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BASE_SIMULATED_OK" in result.stdout


@requires_structure_pack
def test_structure_profile_verifies_and_plans_topology_in_a_subprocess():
    """The structure profile in a fresh process that really has the pinned
    parser -- exact version, module importable, and the three-chunk /
    five-call topology planned end to end."""
    result = _run_simulated_isolation_subprocess(
        """
        import tempfile
        from pathlib import Path

        import tests.contract.package.installed_artifact_smoke as h

        parser_identity, distribution_version, module_importable = (
            h._installed_parser_identity()
        )
        assert parser_identity == h._STRUCTURE_PROFILE_PARSER_VERSION, parser_identity
        assert distribution_version == h._STRUCTURE_PROFILE_PARSER_VERSION
        assert module_importable is True

        h._verify_structure_profile("structure")
        try:
            h._verify_structure_profile("base")
        except h.SmokeFailure:
            pass
        else:
            raise SystemExit("base verified in a parser-present process")

        h._scenario_live_fixture_dry_run(Path(tempfile.mkdtemp()), "structure")
        print("STRUCTURE_SUBPROCESS_OK")
        """,
        simulate_parser_absence=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STRUCTURE_SUBPROCESS_OK" in result.stdout


@requires_structure_pack
def test_base_topology_plans_under_simulated_absence_with_parser_present_parent():
    """Plan section 7.2: the base topology must plan correctly even though the
    outer interpreter has the structure extra -- the child's simulated absence
    is what the run measures, not the parent's parser."""
    from codedoc.parser.tree_sitter_structure import PARSER_PACKAGE_VERSION

    assert PARSER_PACKAGE_VERSION == harness._STRUCTURE_PROFILE_PARSER_VERSION
    result = _run_simulated_isolation_subprocess(
        """
        import tempfile
        from pathlib import Path

        import tests.contract.package.installed_artifact_smoke as h

        h._scenario_live_fixture_dry_run(Path(tempfile.mkdtemp()), "base")
        print("BASE_UNDER_PARSER_PARENT_OK")
        """,
        simulate_parser_absence=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BASE_UNDER_PARSER_PARENT_OK" in result.stdout


# --- frozen per-profile topology -----------------------------------------


def test_live_fixture_topology_is_shared_invariants_plus_a_closed_delta():
    base = harness._live_fixture_topology("base")
    structure = harness._live_fixture_topology("structure")

    for key, value in harness._LIVE_FIXTURE_SHARED_TOPOLOGY.items():
        assert base[key] == value == structure[key], key

    # Exact frozen numbers -- plan section 5.2, not derived from the product.
    assert (base["split_lexical_files"], base["split_syntax_files"]) == (1, 0)
    assert base["split_chunks"] == 4
    assert base["unit_documentation_calls_planned"] == 4
    assert base["initial_provider_calls_planned"] == 6
    assert base["initial_documentation_calls_planned"] == 6
    assert base["documentation_calls_planned"] == 6
    assert base["total_calls_planned"] == 6
    assert base["max_planned_calls"] == 6
    assert base["max_planned_calls_exceeded"] is False
    assert base["correction_calls_possible_max"] == 6
    assert base["provider_calls_max_before_retries"] == 12

    assert (structure["split_lexical_files"], structure["split_syntax_files"]) == (0, 1)
    assert structure["split_chunks"] == 3
    assert structure["unit_documentation_calls_planned"] == 3
    assert structure["initial_provider_calls_planned"] == 5
    assert structure["initial_documentation_calls_planned"] == 5
    assert structure["documentation_calls_planned"] == 5
    assert structure["total_calls_planned"] == 5
    assert structure["max_planned_calls"] == 5
    assert structure["max_planned_calls_exceeded"] is False
    assert structure["correction_calls_possible_max"] == 5
    assert structure["provider_calls_max_before_retries"] == 10

    assert base != structure


def test_live_fixture_topology_rejects_an_unknown_profile():
    with pytest.raises(harness.SmokeFailure, match="unknown-structure-profile"):
        harness._live_fixture_topology("dev")


def test_live_fixture_profile_plan_digests_are_frozen_and_distinct():
    digests = harness._LIVE_FIXTURE_PROFILE_PLAN_DIGEST
    assert set(digests) == {"base", "structure"}
    assert digests["base"] != digests["structure"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in digests.values())


def test_selecting_expectations_never_reads_the_actual_stats():
    """Plan section 11.4: the harness picks the frozen expectation from the
    validated profile; it never infers the profile or the counts from the
    planned stats."""
    scenario_src = inspect.getsource(harness._scenario_live_fixture_dry_run)
    assert "_verify_structure_profile(structure_profile)" in scenario_src
    assert "_live_fixture_topology(structure_profile)" in scenario_src
    mismatch_src = inspect.getsource(harness._live_fixture_topology_mismatch)
    # The frozen expectation drives the comparison, not the actual stats.
    assert "expected.items()" in mismatch_src


def test_base_stats_shape_does_not_satisfy_the_structure_table_or_vice_versa():
    """Plan section 11 items 1-2, at the comparison-helper level."""
    base_expected = harness._live_fixture_topology("base")
    structure_expected = harness._live_fixture_topology("structure")

    assert not harness._live_fixture_topology_mismatch(
        dict(base_expected), base_expected
    )
    assert not harness._live_fixture_topology_mismatch(
        dict(structure_expected), structure_expected
    )
    assert harness._live_fixture_topology_mismatch(
        dict(base_expected), structure_expected
    )
    assert harness._live_fixture_topology_mismatch(
        dict(structure_expected), base_expected
    )


def test_base_run_against_the_structure_table_fails_closed_simulated():
    """Plan section 11 item 1: a base plan checked against the structure
    expectation must raise a topology mismatch, not pass. Simulated isolation."""
    result = _run_simulated_isolation_subprocess(
        """
        import tempfile
        from pathlib import Path

        import tests.contract.package.installed_artifact_smoke as h

        h._LIVE_FIXTURE_PROFILE_TOPOLOGY["base"] = dict(
            h._LIVE_FIXTURE_PROFILE_TOPOLOGY["structure"]
        )
        h._LIVE_FIXTURE_PROFILE_PLAN_DIGEST["base"] = (
            h._LIVE_FIXTURE_PROFILE_PLAN_DIGEST["structure"]
        )
        try:
            h._scenario_live_fixture_dry_run(Path(tempfile.mkdtemp()), "base")
        except h.SmokeFailure as exc:
            assert "topology-mismatch" in str(exc), exc
            print("BASE_VS_STRUCTURE_MISMATCH_OK")
        else:
            raise SystemExit("base plan satisfied the structure table")
        """,
        simulate_parser_absence=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BASE_VS_STRUCTURE_MISMATCH_OK" in result.stdout


def test_base_profile_with_the_structure_call_cap_fails_closed_simulated():
    """Plan section 11 item 5: forcing ``max_planned_calls=5`` into the base
    invocation reproduces the original F-1 failure -- the cap is exceeded and
    the frozen ``max_planned_calls_exceeded=False`` no longer holds. Simulated
    isolation."""
    result = _run_simulated_isolation_subprocess(
        """
        import tempfile
        from pathlib import Path

        import tests.contract.package.installed_artifact_smoke as h

        h._LIVE_FIXTURE_PROFILE_TOPOLOGY["base"] = {
            **h._LIVE_FIXTURE_PROFILE_TOPOLOGY["base"],
            "max_planned_calls": 5,
        }
        try:
            h._scenario_live_fixture_dry_run(Path(tempfile.mkdtemp()), "base")
        except h.SmokeFailure as exc:
            assert "topology-mismatch" in str(exc), exc
            assert "max_planned_calls_exceeded" in str(exc), exc
            print("BASE_CAP5_FAILS_CLOSED_OK")
        else:
            raise SystemExit("base plan accepted the structure call cap")
        """,
        simulate_parser_absence=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BASE_CAP5_FAILS_CLOSED_OK" in result.stdout


def test_run_all_verifies_the_profile_and_threads_it_to_the_live_fixture(
    tmp_path, monkeypatch
):
    """Plan section 9.1 item 9 / 11 item 4: ``_run_all`` verifies the selected
    profile up front and passes it to the live-fixture scenario, while still
    invoking every other scenario exactly once."""
    recorded: dict[str, object] = {}
    seen: list[str] = []
    monkeypatch.setattr(
        harness, "_prove_installed_origin", lambda version=None: (Path("pkg"), Path("con"))
    )
    # The neutral-project temp dir must not resolve under the (real) repo root;
    # neutralize that guard so this structural test can reach every scenario.
    monkeypatch.setattr(
        harness, "_repository_root", lambda: tmp_path / "outside-any-repo"
    )
    monkeypatch.setattr(
        harness,
        "_verify_structure_profile",
        lambda profile: recorded.setdefault("verified", profile),
    )
    for name in (
        "_scenario_truncate",
        "_scenario_fresh_split",
        "_scenario_signature_bound",
        "_scenario_redirected_verbose",
        "_scenario_completed_reuse",
        "_scenario_interrupt_resume",
        "_scenario_imports_only",
        "_scenario_preserve_first",
    ):
        monkeypatch.setattr(
            harness, name, lambda root, _n=name: seen.append(_n)
        )
    monkeypatch.setattr(
        harness,
        "_scenario_live_fixture_dry_run",
        lambda root, profile: recorded.setdefault("live_fixture_profile", profile),
    )
    monkeypatch.setattr(
        harness,
        "_scenario_exit_fidelity",
        lambda root, console: seen.append("_scenario_exit_fidelity"),
    )

    assert harness._run_all("9.9.9", "base") == 0
    assert recorded["verified"] == "base"
    assert recorded["live_fixture_profile"] == "base"
    assert "_scenario_imports_only" in seen
    assert len(seen) == len(set(seen))


# ===========================================================================
# F-9 -- `_scenario_imports_only` reduction-tree reconstruction repair
# (plan sections 1.1, 3.5.1, 4.2.1, 5.3.1, 9.1 item 7, 11 item 19).
# ===========================================================================


def test_imports_only_scenario_runs_clean(tmp_path):
    """Before the repair this raised
    ``KeyError: node_<64 hex>`` on a reducer node id production never
    checkpointed (the comparison tree was built from the source ceiling, not
    the carried automatic synthesis budget)."""
    harness._scenario_imports_only(tmp_path)


def test_imports_only_builds_its_comparison_tree_from_the_carried_budget():
    """Plan section 9.1 item 7: the comparison tree, final manifest, and
    recovery validation all ride the carried automatic synthesis budget, never
    the public source ceiling; the deprecated ``max_content_chars`` tree alias
    is gone."""
    src = inspect.getsource(harness._scenario_imports_only)
    assert "MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS" in src
    assert "synthesis_manifest_chars=synthesis_manifest_chars" in src
    assert "max_chars=tree.synthesis_manifest_chars" in src
    # The only surviving `max_content_chars=` is the public source ceiling on
    # the written config, and it is bound to the named budget, never a literal.
    assert "max_content_chars=source_budget_chars" in src
    assert "max_content_chars=2000" not in src
    assert "max_content_chars=_IMPORTS_ONLY_SOURCE_BUDGET_CHARS" not in src


def test_imports_only_breaks_when_the_comparison_tree_uses_the_source_ceiling(
    tmp_path, monkeypatch
):
    """Plan section 11 item 19: rebuilding the comparison tree from the source
    ceiling instead of the carried synthesis budget must restore the exact
    ``KeyError`` on a non-existent reducer-node id -- not a softer failure.

    Only the scenario's own local ``build_reduction_tree`` call is redirected
    (it re-imports the name per call); production planning bound the symbol at
    import and still checkpoints the real 12000-budget topology, so the
    locally invented reducer level has no matching recovered node."""
    import codedoc.core.file_division as fd

    real_build = fd.build_reduction_tree

    def _from_source_ceiling(plan, *, synthesis_manifest_chars=None, max_content_chars=None, **kw):
        return real_build(
            plan,
            max_content_chars=harness._IMPORTS_ONLY_SOURCE_BUDGET_CHARS,
            **kw,
        )

    monkeypatch.setattr(fd, "build_reduction_tree", _from_source_ceiling)
    with pytest.raises(KeyError) as excinfo:
        harness._scenario_imports_only(tmp_path)
    assert re.search(r"node_[0-9a-f]{64}", str(excinfo.value))


def test_imports_only_source_budget_constant_is_the_public_ceiling():
    assert harness._IMPORTS_ONLY_SOURCE_BUDGET_CHARS == 2000
    from codedoc.core.file_division import MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS

    # The carried synthesis budget the scenario must use is strictly larger.
    assert (
        max(harness._IMPORTS_ONLY_SOURCE_BUDGET_CHARS, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS)
        == MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
        > harness._IMPORTS_ONLY_SOURCE_BUDGET_CHARS
    )


# ===========================================================================
# Canonical-gate scenario coverage
# (plan sections 5.3.1, 9.1 items 8-9, 11 item 20, 12).
# ===========================================================================

# Every scenario `_run_all` invokes, mapped to the collected, deterministic,
# provider-free contract test(s) in THIS module that execute its real path.
# `redirected_verbose` and `exit_fidelity` drive a real `codedoc`
# console-script / `--child-run` subprocess: their collected tests run the
# real path when `sys.executable` can self-certify, and otherwise a controlled
# double at the installed-origin boundary while still exercising every helper
# and assertion the scenario owns (plan section 5.3.1). Genuine installed
# base/structure runs are non-collected harness commands (plan section 10) and
# are deliberately NOT referenced here.
_SCENARIO_COLLECTED_TESTS: dict[str, tuple[str, ...]] = {
    "_scenario_truncate": ("test_scenario_truncate_executes_provider_free",),
    "_scenario_fresh_split": ("test_scenario_fresh_split_executes_provider_free",),
    "_scenario_signature_bound": (
        "test_signature_bound_scenario_runs_clean_end_to_end",
    ),
    "_scenario_live_fixture_dry_run": (
        "test_live_fixture_dry_run_scenario_runs_clean_structure_profile",
        "test_base_profile_verifies_and_plans_topology_under_simulated_absence",
    ),
    "_scenario_redirected_verbose": ("test_scenario_redirected_verbose_executes",),
    "_scenario_completed_reuse": (
        "test_scenario_completed_reuse_executes_provider_free",
    ),
    "_scenario_interrupt_resume": (
        "test_scenario_interrupt_resume_executes_provider_free",
    ),
    "_scenario_imports_only": ("test_imports_only_scenario_runs_clean",),
    "_scenario_preserve_first": (
        "test_scenario_preserve_first_executes_provider_free",
    ),
    "_scenario_exit_fidelity": ("test_scenario_exit_fidelity_executes",),
}

# R4: the exact frozen canonical order. Independent of `harness`'s own
# authority tuple so a bad edit to EITHER is caught.
_EXPECTED_RUN_ALL_ORDER: tuple[str, ...] = (
    "_scenario_truncate",
    "_scenario_fresh_split",
    "_scenario_signature_bound",
    "_scenario_live_fixture_dry_run",
    "_scenario_redirected_verbose",
    "_scenario_completed_reuse",
    "_scenario_interrupt_resume",
    "_scenario_imports_only",
    "_scenario_preserve_first",
    "_scenario_exit_fidelity",
)


def _parsed_run_all_order() -> tuple[str, ...]:
    """The scenario calls in ``_run_all``'s body, in source order. Matches only
    the ``_scenario_X(neutral_root`` call lines -- not the ``[scenario]
    reached`` markers, not the module-level authority tuple."""
    source = inspect.getsource(harness._run_all)
    return tuple(
        re.findall(r"^\s+(_scenario_[a-z_]+)\(neutral_root", source, re.M)
    )


def _run_all_order_violations(parsed: "tuple[str, ...] | list[str]") -> list[str]:
    """Every way *parsed* deviates from the frozen canonical order. Empty means
    an exact, position-wise match with no missing/stale/duplicate scenario."""
    parsed = tuple(parsed)
    problems: list[str] = []
    if parsed != _EXPECTED_RUN_ALL_ORDER:
        problems.append("order-or-membership-mismatch")
    if len(parsed) != len(set(parsed)):
        problems.append("duplicate-scenario")
    missing = sorted(set(_EXPECTED_RUN_ALL_ORDER) - set(parsed))
    stale = sorted(set(parsed) - set(_EXPECTED_RUN_ALL_ORDER))
    if missing:
        problems.append(f"missing:{missing}")
    if stale:
        problems.append(f"stale:{stale}")
    return problems


def _run_all_reached_scenarios() -> list[str]:
    return sorted(set(_parsed_run_all_order()))


def _coverage_gaps(reached: list[str], registry: dict[str, tuple[str, ...]]) -> list[str]:
    return sorted(s for s in reached if not registry.get(s))


def _coverage_stale_entries(
    reached: list[str], registry: dict[str, tuple[str, ...]]
) -> list[str]:
    return sorted(set(registry) - set(reached))


def test_every_run_all_scenario_has_a_collected_test_that_executes_it():
    """Plan section 9.1 item 8: no scenario reachable from ``--scenario all``
    may be certified only by an installed run that never reaches it."""
    reached = _run_all_reached_scenarios()
    assert reached, "parsed no scenarios out of _run_all"
    assert _coverage_gaps(reached, _SCENARIO_COLLECTED_TESTS) == []
    assert _coverage_stale_entries(reached, _SCENARIO_COLLECTED_TESTS) == []

    module = sys.modules[__name__]
    for scenario, test_names in _SCENARIO_COLLECTED_TESTS.items():
        assert test_names, scenario
        for test_name in test_names:
            assert callable(getattr(module, test_name, None)), (
                f"{scenario} coverage names a missing test: {test_name!r}"
            )


def test_coverage_contract_flags_a_dropped_scenario_test():
    """Plan section 11 item 20: removing a scenario's collected test must fail
    the coverage contract, not silently shrink the gate."""
    reached = _run_all_reached_scenarios()
    holey = {k: v for k, v in _SCENARIO_COLLECTED_TESTS.items() if k != "_scenario_imports_only"}
    assert _coverage_gaps(reached, holey) == ["_scenario_imports_only"]


def test_coverage_contract_flags_a_scenario_removed_from_run_all():
    """Plan section 11 item 20 / section 12: deleting a scenario from
    ``--scenario all`` leaves a registry entry with nothing to execute."""
    trimmed = [s for s in _run_all_reached_scenarios() if s != "_scenario_preserve_first"]
    assert _coverage_stale_entries(trimmed, _SCENARIO_COLLECTED_TESTS) == [
        "_scenario_preserve_first"
    ]


def test_run_all_scenario_order_is_frozen_exactly():
    """R4: the parsed ``_run_all`` sequence equals the frozen tuple exactly,
    and the harness's own authority tuple agrees. A position-wise comparison,
    not a set membership check."""
    parsed = _parsed_run_all_order()
    assert parsed == _EXPECTED_RUN_ALL_ORDER
    assert harness._CANONICAL_SCENARIO_ORDER == _EXPECTED_RUN_ALL_ORDER
    assert _run_all_order_violations(parsed) == []
    assert tuple(_SCENARIO_COLLECTED_TESTS) == _EXPECTED_RUN_ALL_ORDER


def test_run_all_order_contract_detects_a_swap():
    """R4: swapping two scenarios (same membership, different order) fails --
    the old ``sorted(set(...), key=index)`` check could not see this."""
    swapped = list(_EXPECTED_RUN_ALL_ORDER)
    swapped[3], swapped[7] = swapped[7], swapped[3]
    assert set(swapped) == set(_EXPECTED_RUN_ALL_ORDER)
    assert swapped != list(_EXPECTED_RUN_ALL_ORDER)
    assert "order-or-membership-mismatch" in _run_all_order_violations(swapped)


def test_run_all_order_contract_detects_deletion():
    """R4: dropping a scenario is flagged as missing."""
    violations = _run_all_order_violations(_EXPECTED_RUN_ALL_ORDER[1:])
    assert any(v.startswith("missing:") for v in violations)
    assert "order-or-membership-mismatch" in violations


def test_run_all_order_contract_detects_duplication():
    """R4: a repeated scenario is flagged as a duplicate."""
    dup = _EXPECTED_RUN_ALL_ORDER + (_EXPECTED_RUN_ALL_ORDER[0],)
    violations = _run_all_order_violations(dup)
    assert "duplicate-scenario" in violations
    assert "order-or-membership-mismatch" in violations


def test_structure_profile_is_not_a_public_codedoc_cli_or_config_key(tmp_path):
    """Plan sections 1.2 / 5.1 / 9.1 item 5 / 11 item 4: the selector lives on
    the release harness only -- neither a ``codedoc`` CLI argument nor a
    configuration key, and never written into a project config."""
    from codedoc.cli import cli as codedoc_cli
    from codedoc.core.loader import DEFAULTS

    cli_src = inspect.getsource(codedoc_cli)
    assert "structure-profile" not in cli_src and "structure_profile" not in cli_src
    assert "structure_profile" not in DEFAULTS

    written = harness._write_config(tmp_path, large_file_strategy="split")
    assert "structure_profile" not in written
    on_disk = json.loads(
        (tmp_path / "codedoc.config.json").read_text(encoding="utf-8")
    )
    assert "structure_profile" not in on_disk


# Genuine base/structure `--scenario all` executions are NON-collected harness
# commands (plan section 10, lines 1112-1118). They are run separately with an
# explicit interpreter + explicit `--candidate-version`, and their interpreter
# paths belong only in execution evidence -- never in this file. This module
# therefore contains no external-candidate discovery, no workstation paths, no
# environment-variable interpreter overrides, and no candidate-environment
# skips.


def test_console_scripts_are_derived_from_the_interpreter_not_path_looked_up():
    """R1 regression: a `codedoc` console is only ever taken from an
    interpreter's own environment root (beside it or its Scripts/bin sibling),
    never from an ambient PATH lookup, so an unrelated `codedoc.exe` on PATH
    cannot be selected or paired with `sys.executable`."""
    src = inspect.getsource(_interpreter_own_console)
    assert "shutil.which" not in src and "which(" not in src

    for probe in (sys.executable, r"C:\some\other\env\python.exe"):
        console = _interpreter_own_console(Path(probe))
        assert harness._is_within(console, Path(probe).parent)

    # The console-script collected tests pair only `sys.executable` with its
    # own derived console -- never a PATH result.
    for test in (
        test_scenario_redirected_verbose_executes,
        test_scenario_exit_fidelity_executes,
    ):
        body = inspect.getsource(test)
        assert "shutil.which" not in body
        if "_interpreter_own_console(" in body:
            assert "_interpreter_own_console(Path(sys.executable))" in body


def test_self_certification_probe_asserts_no_release_version():
    """Defect A / requirement 3: the self-certification probe proves internal
    origin consistency only. It passes NO expected version to
    `_prove_installed_origin`, reads no module version, and never builds a
    `--candidate-version`."""
    probe = _SELF_CERT_PROBE
    assert "_prove_installed_origin()" in probe
    assert "_prove_installed_origin(codedoc" not in probe
    assert '_prove_installed_origin("' not in probe
    assert "codedoc.__version__" not in probe
    assert "module_version" not in probe
    assert "--candidate-version" not in probe

    fn = inspect.getsource(_current_interpreter_can_self_certify)
    # The function itself never calls _prove_installed_origin with an argument
    # and never touches a release version.
    assert "_prove_installed_origin(codedoc" not in fn
    assert '_prove_installed_origin("' not in fn
    assert "candidate-version" not in fn
    assert "__version__" not in fn


def test_candidate_version_authority_is_only_the_caller_supplied_value():
    """Defect A / requirement 4: `--candidate-version` is never constructed
    from an installed candidate's own reported version anywhere in this file.
    Every use is a literal/parametrized test value handed to `harness.main`."""
    module_src = Path(__file__).read_text(encoding="utf-8")
    for idx, line in enumerate(module_src.splitlines(), start=1):
        if "--candidate-version" not in line:
            continue
        # The value on the following lines must not come from a probe payload
        # or an installed module's self-report.
        window = "\n".join(module_src.splitlines()[idx - 1 : idx + 2])
        assert "module_version" not in window, (idx, window)
        assert "codedoc.__version__" not in window, (idx, window)
        assert "__version__" not in window, (idx, window)
        assert "env_info[" not in window, (idx, window)
        assert "payload[" not in window, (idx, window)


def test_installed_self_report_cannot_replace_the_caller_expected_version(
    monkeypatch, tmp_path
):
    """Requirement 4 / defect A: an internally-consistent installed environment
    (module version == distribution metadata == console) that reports 2.5.0 is
    still rejected against a different caller-supplied expected version. The
    candidate's own self-report is never the authority."""
    _fake_installed_env(
        monkeypatch,
        tmp_path,
        module_version="2.5.0",
        dists=(("codedoc-ai", "2.5.0", "site-packages/codedoc_ai-2.5.0.dist-info"),),
    )
    with pytest.raises(harness.SmokeFailure, match="candidate-version-mismatch"):
        harness._prove_installed_origin("7.7.7")
    # With NO expected version (the self-certification-probe contract) the same
    # environment passes on internal consistency alone.
    assert harness._prove_installed_origin() == harness._prove_installed_origin(None)


def test_main_forwards_the_exact_caller_candidate_version(monkeypatch):
    """Requirement 4: `main(... --candidate-version X ...)` forwards exactly X
    to `_run_all` / origin validation, unchanged."""
    seen: list[object] = []

    def _capture(candidate_version, structure_profile):
        seen.append((candidate_version, structure_profile))
        return 0

    monkeypatch.setattr(harness, "_run_all", _capture)
    assert (
        harness.main(
            [
                "--scenario",
                "all",
                "--candidate-version",
                "4.2.0-rc1",
                "--structure-profile",
                "base",
            ]
        )
        == 0
    )
    assert seen == [("4.2.0-rc1", "base")]


# --- provider-free scenarios: executed directly --------------------------


def test_scenario_truncate_executes_provider_free(tmp_path):
    harness._scenario_truncate(tmp_path)


def test_scenario_fresh_split_executes_provider_free(tmp_path):
    harness._scenario_fresh_split(tmp_path)


def test_scenario_completed_reuse_executes_provider_free(tmp_path):
    harness._scenario_completed_reuse(tmp_path)


def test_scenario_interrupt_resume_executes_provider_free(tmp_path):
    harness._scenario_interrupt_resume(tmp_path)


def test_scenario_preserve_first_executes_provider_free(tmp_path):
    harness._scenario_preserve_first(tmp_path)


# --- console-script scenarios --------------------------------------------
#
# `_scenario_redirected_verbose` and `_scenario_exit_fidelity` drive a real
# `codedoc` console script / `--child-run` subprocess under `sys.executable`.
# The real-vs-double decision is made UP FRONT from
# `_current_interpreter_can_self_certify()` -- never a try/except that could
# rescue a real failure into the double (plan R3). The real path propagates
# every failure. The double runs only for the pre-established reason "this
# interpreter's codedoc resolves into the repo", and still exercises every
# helper and assertion the scenario owns (plan section 5.3.1). The console is
# always `sys.executable`'s own, never one taken from ambient PATH (plan R1).


def _verbose_double_stub(tmp_path: Path) -> Path:
    stub = tmp_path / "verbose_stub.py"
    stub.write_text(
        "import sys\nsys.stdout.write('redirected verbose double: ok\\n')\n",
        encoding="utf-8",
    )
    return stub


def test_scenario_redirected_verbose_executes(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    if _current_interpreter_can_self_certify():
        # Real path: every failure -- CLI regression, origin error, anything --
        # propagates. No fallback. Put this interpreter's own bin dir(s) first
        # on PATH so the child's own console resolves, never a foreign one.
        monkeypatch.setenv(
            "PATH", _provider_free_env(Path(sys.executable).parent)["PATH"]
        )
        harness._scenario_redirected_verbose(work)
        return

    # Pre-established reason only: this interpreter cannot self-certify. Drive
    # the PowerShell ``2>&1 | Tee-Object`` plumbing, log read-back, and the
    # exit-status / logging-error / size / privacy assertions via a benign
    # stdout double.
    stub = _verbose_double_stub(tmp_path)
    monkeypatch.setattr(
        harness,
        "_child_command",
        lambda project, cli_args, python_exe=sys.executable: [python_exe, str(stub)],
    )
    harness._scenario_redirected_verbose(work)


def test_redirected_verbose_real_path_never_falls_back_on_failure(tmp_path, monkeypatch):
    """R3: once the real path is chosen, an unrelated nonzero child result must
    fail -- the controlled double is only for the 'cannot self-certify' case,
    never a rescue for a real failure. Here the child exits 7 for an unrelated
    reason and the scenario must raise, not silently pass with a stub."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(
        harness,
        "_child_command",
        lambda project, cli_args, python_exe=sys.executable: [
            python_exe,
            "-c",
            "import sys; sys.stderr.write('unrelated regression\\n'); sys.exit(7)",
        ],
    )
    with pytest.raises(harness.SmokeFailure, match="redirected-verbose-exit-status"):
        harness._scenario_redirected_verbose(work)


def test_redirected_verbose_collected_test_has_no_rescue_fallback():
    """R3: the collected test decides real-vs-double up front and has no
    exception handler that could turn a real-run failure into a double pass."""
    src = inspect.getsource(test_scenario_redirected_verbose_executes)
    assert "_current_interpreter_can_self_certify()" in src
    assert "except" not in src


def test_scenario_exit_fidelity_executes(tmp_path, monkeypatch):
    if _current_interpreter_can_self_certify():
        work = tmp_path / "work"
        work.mkdir()
        # This interpreter's OWN console, never one taken from ambient PATH,
        # and its bin dir(s) first on PATH so the `--child-run` child resolves
        # the same one.
        monkeypatch.setenv(
            "PATH", _provider_free_env(Path(sys.executable).parent)["PATH"]
        )
        console = _interpreter_own_console(Path(sys.executable))
        harness._scenario_exit_fidelity(work, console)
        return

    # Pre-established reason only: fake ``subprocess.run`` mapping the two
    # control argument sets to their real exit codes for BOTH the direct and
    # the child invocation, exercising `_child_command`, `_run_child`
    # (env/PATH assembly), and the scenario's status-parity assertion.
    def _fake_run(cmd, **_kwargs):
        args = list(cmd)
        if "--version" in args:
            code = 0
        elif "--analysis-mode" in args:
            code = 2
        else:
            code = 1
        return SimpleNamespace(args=args, returncode=code, stdout="", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", _fake_run)
    work = tmp_path / "work"
    work.mkdir()
    harness._scenario_exit_fidelity(work, tmp_path / "codedoc-double.exe")


# ---------------------------------------------------------------------------
# Console-script environment binding across POSIX symlinked venvs.
#
# `_prove_installed_origin` requires the resolved `codedoc` console script to
# live inside the running interpreter's script directory. Computing that
# directory as `Path(sys.executable).resolve().parent` is wrong on every POSIX
# runner: `venv` symlinks `bin/python` at the interpreter it was created from,
# so resolving the executable lands in the base installation's `bin` while the
# console script is a real file in the venv -- the check then fails with
# `console-script-environment-mismatch`. Windows copies `python.exe` into
# `Scripts\`, so the resolved executable stays inside the environment and the
# defect is invisible there. That asymmetry is why it reached CI unnoticed.
# ---------------------------------------------------------------------------


def _can_symlink(tmp_path) -> bool:
    """Whether this platform/permission set can create a symlink."""
    target = tmp_path / "_probe_target"
    target.mkdir(exist_ok=True)
    link = tmp_path / "_probe_link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        return False
    finally:
        try:
            link.unlink()
        except OSError:
            pass
    return True


def test_environment_bin_never_follows_a_symlinked_interpreter(tmp_path, monkeypatch):
    """A POSIX-shaped venv whose `bin/python` symlinks out to a base
    installation still resolves to its OWN script directory, so a console
    script sitting beside that symlink is correctly recognised as belonging to
    the environment under test."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")

    base_bin = tmp_path / "base" / "bin"
    base_bin.mkdir(parents=True)
    base_python = base_bin / "python3"
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    os.symlink(base_python, venv_python)          # exactly what `venv` does
    console = venv_bin / "codedoc"
    console.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(harness.sys, "executable", str(venv_python))

    resolved = harness._environment_bin()
    assert resolved == venv_bin.resolve(), resolved
    assert resolved != base_bin.resolve()
    # The console script beside the symlinked interpreter belongs to the
    # environment -- this is the exact containment `_prove_installed_origin`
    # asserts, and what the pre-fix computation got wrong.
    assert harness._is_within(console.resolve(), resolved)

    # The superseded computation is proven wrong on this layout, so a
    # regression back to it fails here rather than only on a POSIX runner.
    superseded = Path(harness.sys.executable).resolve().parent
    assert superseded == base_bin.resolve()
    assert not harness._is_within(console.resolve(), superseded)


def test_environment_bin_matches_a_copied_interpreter_layout(tmp_path, monkeypatch):
    """The Windows layout -- `python.exe` copied into the environment rather
    than symlinked -- keeps working, so the fix is not platform-specific."""
    env_bin = tmp_path / "env" / "Scripts"
    env_bin.mkdir(parents=True)
    interpreter = env_bin / "python.exe"
    interpreter.write_text("", encoding="utf-8")
    console = env_bin / "codedoc.exe"
    console.write_text("", encoding="utf-8")

    monkeypatch.setattr(harness.sys, "executable", str(interpreter))

    resolved = harness._environment_bin()
    assert resolved == env_bin.resolve(), resolved
    assert harness._is_within(console.resolve(), resolved)


def test_prove_installed_origin_computes_environment_bin_from_the_helper():
    """The production check routes through `_environment_bin`, so the two
    tests above actually guard `_prove_installed_origin` rather than an
    unused helper."""
    source = inspect.getsource(harness._prove_installed_origin)
    assert "environment_bin = _environment_bin()" in source
    assert "Path(sys.executable).resolve().parent" not in source
