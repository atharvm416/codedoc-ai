"""Tests organized by feature ownership."""

from __future__ import annotations

import io
import os
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

def test_workflow_is_manual_safe_and_packaged_in_metadata():
    repo = Path(__file__).resolve().parents[3]
    workflow = (repo / "codedoc" / "templates" / "github-actions-codedoc.yml").read_text(
        encoding="utf-8"
    )
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (repo / "MANIFEST.in").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "\npush:" not in workflow and "\npull_request:" not in workflow
    assert "contents: read" in workflow
    assert workflow.index("--dry-run") < workflow.index("Generate documentation")
    assert workflow.count('            --max-files "$CODEDOC_MAX_FILES"') == 2
    assert 'args=(' in workflow and '"$CODEDOC_PROJECT_ROOT"' in workflow
    assert '"$CODEDOC_OUTPUT_PATH"' in workflow
    assert workflow.count("${{ inputs.") == 3
    run_blocks = workflow.split("run: |")[1:]
    assert run_blocks
    assert all("${{ inputs." not in block for block in run_blocks)
    assert "git push" not in workflow and "git commit" not in workflow
    assert "path: ${{ env.CODEDOC_OUTPUT_PATH }}" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert 'codedoc = ["templates/*.yml"]' in pyproject
    assert "recursive-include codedoc/templates *.yml" in manifest

_PACKAGED_TEMPLATE = "codedoc/templates/github-actions-codedoc.yml"


def _current_stem() -> str:
    repo = Path(__file__).resolve().parents[3]
    project = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    return f"{project['name'].replace('-', '_')}-{project['version']}"


def _check_artifact_contract(artifact_dir: str) -> None:
    """The artifact-content contract (section 6.1): unset/empty is not
    applicable and passes cleanly; a missing/unreadable directory or a
    wrong wheel/sdist inventory fails; only an exact current-version pair is
    inspected for the packaged workflow template. Any extra/stale
    distribution fails the contract rather than being ignored."""
    if not artifact_dir:
        return

    stem = _current_stem()
    artifact_path = Path(artifact_dir)

    assert artifact_path.is_dir(), (
        f"CODEDOC_ARTIFACT_DIR must be a readable directory: {artifact_dir!r}"
    )
    distribution_files = sorted(
        path
        for path in artifact_path.iterdir()
        if path.is_file()
        and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
    )
    assert len(distribution_files) == 2, (
        f"Expected exactly two distribution files in {artifact_dir!r}, found "
        f"{[path.name for path in distribution_files]!r}"
    )
    wheels = sorted(artifact_path.glob(f"{stem}-*.whl"))
    sdists = sorted(artifact_path.glob(f"{stem}.tar.gz"))
    assert len(wheels) == 1, (
        f"Expected exactly one {stem}-*.whl in {artifact_dir!r}, found {len(wheels)}"
    )
    assert len(sdists) == 1, (
        f"Expected exactly one {stem}.tar.gz in {artifact_dir!r}, found {len(sdists)}"
    )

    with zipfile.ZipFile(wheels[0]) as archive:
        assert _PACKAGED_TEMPLATE in archive.namelist()
    with tarfile.open(sdists[0]) as archive:
        assert any(name.endswith("/" + _PACKAGED_TEMPLATE) for name in archive.getnames())


def test_built_distributions_contain_workflow_template():
    """Inspect release artifacts via the test-only ``CODEDOC_ARTIFACT_DIR``
    environment variable (section 6.1) -- never a product config key, never
    read by ``codedoc.core.loader`` or the CLI. The source suite runs with
    this variable unset, so this contract cannot remain permanently vacuous
    once CI/the local artifact gate sets it; the other three states (section
    12.1 R6) are exercised directly, below, against constructed fixtures so
    they run in the source suite too."""
    _check_artifact_contract(os.environ.get("CODEDOC_ARTIFACT_DIR", ""))


def _write_fake_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(_PACKAGED_TEMPLATE, "on: workflow_dispatch:\n")


def _write_fake_sdist(path: Path, stem: str) -> None:
    with tarfile.open(path, "w:gz") as archive:
        content = b"on: workflow_dispatch:\n"
        info = tarfile.TarInfo(name=f"{stem}/{_PACKAGED_TEMPLATE}")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def test_artifact_contract_missing_directory_fails(tmp_path):
    with pytest.raises(AssertionError, match="must be a readable directory"):
        _check_artifact_contract(str(tmp_path / "does-not-exist"))


def test_artifact_contract_wrong_pair_count_fails(tmp_path):
    stem = _current_stem()
    _write_fake_wheel(tmp_path / f"{stem}-py3-none-any.whl")
    # No sdist at all: wrong pair count.
    with pytest.raises(AssertionError, match="Expected exactly two distribution files"):
        _check_artifact_contract(str(tmp_path))


def test_artifact_contract_exact_pair_passes(tmp_path):
    stem = _current_stem()
    _write_fake_wheel(tmp_path / f"{stem}-py3-none-any.whl")
    _write_fake_sdist(tmp_path / f"{stem}.tar.gz", stem)
    _check_artifact_contract(str(tmp_path))


def test_artifact_contract_exact_pair_plus_stale_extra_fails(tmp_path):
    """A stale distribution beside the current pair must fail the exact
    two-file inventory instead of being ignored."""
    stem = _current_stem()
    _write_fake_wheel(tmp_path / f"{stem}-py3-none-any.whl")
    _write_fake_sdist(tmp_path / f"{stem}.tar.gz", stem)
    stale_stem = stem.rsplit("-", 1)[0] + "-0.0.0"
    _write_fake_wheel(tmp_path / f"{stale_stem}-py3-none-any.whl")
    with pytest.raises(AssertionError, match="Expected exactly two distribution files"):
        _check_artifact_contract(str(tmp_path))
