"""Structural guard for the test-suite conventions this release establishes.

Reads test paths and Python ASTs; never imports a collected test module.
Exposes exactly one collected test, ``test_suite_architecture_contract``,
which aggregates and reports every violation of four properties:

1. no module under ``tests/`` imports a module whose basename matches
   ``test_*.py``;
2. ``tests/support/`` contains no ``test_*.py`` file and no class named
   ``Test*``;
3. direct ``time.sleep(...)`` appears only in the allowlisted modules below;
4. each marker registered in ``[tool.pytest.ini_options].markers`` appears
   verbatim in ``CONTRIBUTING.md``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"
_SUPPORT_DIR = _TESTS_DIR / "support"
_CONTRIBUTING = _REPO_ROOT / "CONTRIBUTING.md"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The three concurrency-coordination sites where a deterministic event or
# barrier cannot express the condition, so a real time.sleep is permitted.
_SLEEP_ALLOWLIST = {
    "test_095_persistence_failures.py",
    "test_097_fail_fast.py",
    "test_pipeline.py",
}


def _all_test_tree_files() -> list[Path]:
    return sorted(p for p in _TESTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_module_basenames(tree: ast.Module) -> list[str]:
    """Every module basename this file imports, from both import forms.

    Covers ``import tests.test_x`` (basename from the dotted path) and
    ``from tests import test_x`` / ``from tests.test_x import y`` (basename
    from either the dotted path or a submodule name in the `from` clause).
    """
    basenames: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                basenames.append(alias.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            basenames.append(node.module.rsplit(".", 1)[-1])
            for alias in node.names:
                basenames.append(alias.name)
    return basenames


def _check_no_cross_test_imports() -> list[str]:
    violations = []
    for path in _all_test_tree_files():
        tree = _parse(path)
        for basename in _imported_module_basenames(tree):
            if basename.startswith("test_") and basename != path.stem:
                violations.append(
                    f"{path.relative_to(_REPO_ROOT)}: imports test module '{basename}'"
                )
    return violations


def _check_support_package_uncollectable() -> list[str]:
    violations = []
    if not _SUPPORT_DIR.is_dir():
        return [f"{_SUPPORT_DIR.relative_to(_REPO_ROOT)}: directory is missing"]
    for path in sorted(_SUPPORT_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name.startswith("test_"):
            violations.append(
                f"{path.relative_to(_REPO_ROOT)}: collectible filename under tests/support/"
            )
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                violations.append(
                    f"{path.relative_to(_REPO_ROOT)}: class '{node.name}' looks collectible"
                )
    return violations


def _has_direct_time_sleep(tree: ast.Module) -> bool:
    """True if *tree* calls ``time.sleep(...)`` or a bare ``sleep(...)``
    imported via ``from time import sleep``."""
    bare_sleep_imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "time"
        and any(alias.name == "sleep" for alias in node.names)
        for node in ast.walk(tree)
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "sleep"
            and isinstance(func.value, ast.Name)
            and func.value.id == "time"
        ):
            return True
        if bare_sleep_imported and isinstance(func, ast.Name) and func.id == "sleep":
            return True
    return False


def _check_sleep_allowlist() -> list[str]:
    violations = []
    for path in _all_test_tree_files():
        if _has_direct_time_sleep(_parse(path)) and path.name not in _SLEEP_ALLOWLIST:
            violations.append(
                f"{path.relative_to(_REPO_ROOT)}: direct time.sleep(...) outside the allowlist"
            )
    return violations


def _configured_marker_names() -> list[str]:
    """Marker names declared in ``[tool.pytest.ini_options].markers``.

    Read directly from ``pyproject.toml`` text (not ``pytestconfig.getini``,
    which also returns pytest's and plugins' builtin markers) so only the
    markers this project registers are checked.
    """
    text = _PYPROJECT.read_text(encoding="utf-8")
    section_match = re.search(
        r"^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)", text, re.DOTALL | re.MULTILINE
    )
    if not section_match:
        return []
    markers_match = re.search(r"markers\s*=\s*\[(.*?)\]", section_match.group(1), re.DOTALL)
    if not markers_match:
        return []
    return [
        literal.split(":", 1)[0].strip()
        for literal in re.findall(r'"([^"]*)"', markers_match.group(1))
    ]


def _check_markers_documented() -> list[str]:
    violations = []
    contributing_text = _CONTRIBUTING.read_text(encoding="utf-8")
    for name in _configured_marker_names():
        if name not in contributing_text:
            violations.append(
                f"marker '{name}' is registered but not documented verbatim in CONTRIBUTING.md"
            )
    return violations


def test_suite_architecture_contract():
    violations = [
        *_check_no_cross_test_imports(),
        *_check_support_package_uncollectable(),
        *_check_sleep_allowlist(),
        *_check_markers_documented(),
    ]
    assert not violations, "Test suite architecture violations:\n" + "\n".join(
        f"  - {v}" for v in violations
    )
