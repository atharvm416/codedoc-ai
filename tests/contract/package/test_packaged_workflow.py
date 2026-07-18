"""Tests organized by feature ownership."""

from __future__ import annotations

import tarfile
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

def test_built_distributions_contain_workflow_template():
    """Inspect release artifacts when present; the release check builds them."""
    repo = Path(__file__).resolve().parents[3]
    dist = repo / "dist"
    wheels = sorted(dist.glob("codedoc_ai-0.9.2-*.whl"))
    sdists = sorted(dist.glob("codedoc_ai-0.9.2.tar.gz"))
    if not wheels or not sdists:
        pytest.skip("release artifacts have not been built")

    packaged = "codedoc/templates/github-actions-codedoc.yml"
    try:
        with zipfile.ZipFile(wheels[-1]) as archive:
            assert packaged in archive.namelist()
        with tarfile.open(sdists[-1]) as archive:
            assert any(name.endswith("/" + packaged) for name in archive.getnames())
    except PermissionError:
        pytest.skip("sandbox cannot read externally built artifacts")
