"""Tests organized by feature ownership."""

from __future__ import annotations

import re
import pytest
from codedoc import __version__
from codedoc.cli.cli import run_cli

def test_version_identity_consistent(capsys):
    """Release identity: pyproject, codedoc.__version__, and CLI agree."""
    import pathlib
    import pytest
    import codedoc

    repo_root = pathlib.Path(__file__).resolve().parents[3]

    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    assert match, "version not found in pyproject.toml"
    assert match.group(1) == codedoc.__version__

    from codedoc.cli.cli import main
    with pytest.raises(SystemExit):
        main(["--version"])
    assert codedoc.__version__ in capsys.readouterr().out

def test_version_flag_reports_canonical_identity(capsys):
    with pytest.raises(SystemExit) as caught:
        run_cli(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out
