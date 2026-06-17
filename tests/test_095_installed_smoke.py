"""0.9.5 — source-tree CLI smoke checks.

These run against the source tree in normal CI: they assert the package imports
and the console entry point reports its version and help without contacting any
provider.  The installed-wheel equivalent of these checks is executed by the CI
workflow after building and installing the wheel in a clean environment; see
``.github/workflows/ci.yml``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_package_imports():
    import codedoc

    assert isinstance(codedoc.__version__, str)
    assert codedoc.__version__


def test_cli_version_matches_package_version():
    import codedoc

    result = subprocess.run(
        [sys.executable, "-m", "codedoc", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert codedoc.__version__ in result.stdout


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "codedoc", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def test_runtime_dependencies_exclude_dormant_http_client():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"requests' not in pyproject


def test_sdist_manifest_includes_public_docs_and_excludes_repository_tests():
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")
    assert "include RUN_FLOW.md" in manifest
    assert "recursive-include tests" not in manifest
    assert "prune tests" in manifest
