"""AST-only enforcement of the test-suite architecture's eleven rules."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = _REPO_ROOT / "tests"
_SUPPORT_DIR = _TESTS_DIR / "support"
_ALLOWED_LEVELS = {"unit", "integration", "contract", "e2e", "platform", "meta"}
_SLEEP_ALLOWLIST = {
    "integration/persistence/test_write_failures.py",
    "integration/pipeline/test_fail_fast.py",
    "integration/pipeline/test_parallel_execution.py",
}
_EXPECTED_GOLDENS = {
    "project_view.json",
    "completed_output.json",
    "completed_output.md",
    "empty_output.json",
    "empty_output.md",
}


def _python_files() -> list[Path]:
    return sorted(
        path
        for path in _TESTS_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _test_modules() -> list[Path]:
    return [path for path in _python_files() if path.name.startswith("test_")]


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_basenames(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.rsplit(".", 1)[-1])
            names.update(alias.name for alias in node.names)
    return names


def _check_no_test_imports() -> list[str]:
    return [
        f"{_rel(path)}: imports test module '{name}'"
        for path in _python_files()
        for name in _imported_basenames(_tree(path))
        if name.startswith("test_")
    ]


def _check_support_is_not_collectable() -> list[str]:
    if not _SUPPORT_DIR.is_dir():
        return ["tests/support: directory is missing"]
    violations = []
    for path in sorted(_SUPPORT_DIR.rglob("*.py")):
        if path.name.startswith("test_"):
            violations.append(f"{_rel(path)}: collectible support filename")
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                violations.append(f"{_rel(path)}: collectible class '{node.name}'")
    return violations


def _has_direct_sleep(tree: ast.Module) -> bool:
    bare_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "time"
        for alias in node.names
        if alias.name == "sleep"
    }
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
        if isinstance(func, ast.Name) and func.id in bare_names:
            return True
    return False


def _check_sleep_calls() -> list[str]:
    return [
        f"{_rel(path)}: direct sleep call outside the allowlist"
        for path in _python_files()
        if _has_direct_sleep(_tree(path))
        and path.relative_to(_TESTS_DIR).as_posix() not in _SLEEP_ALLOWLIST
    ]


def _configured_markers() -> list[str]:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(
        r"^\[tool\.pytest\.ini_options\]\n(.*?)(?=^\[|\Z)",
        text,
        re.DOTALL | re.MULTILINE,
    )
    if not section:
        return []
    markers = re.search(r"markers\s*=\s*\[(.*?)\]", section.group(1), re.DOTALL)
    if not markers:
        return []
    return [
        value.split(":", 1)[0].strip()
        for value in re.findall(r'"([^"]*)"', markers.group(1))
    ]


def _check_marker_docs() -> list[str]:
    text = (_REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    return [
        f"marker '{name}' is not documented verbatim in CONTRIBUTING.md"
        for name in _configured_markers()
        if name not in text
    ]


def _check_root_and_versioned_names() -> list[str]:
    violations = [
        f"{_rel(path)}: collected test remains at tests root"
        for path in sorted(_TESTS_DIR.glob("test_*.py"))
    ]
    violations.extend(
        f"{_rel(path)}: version-numbered test filename"
        for path in _test_modules()
        if re.fullmatch(r"test_\d+_.*\.py", path.name)
    )
    return violations


def _check_levels_and_initializers() -> list[str]:
    violations = []
    for path in _test_modules():
        relative = path.relative_to(_TESTS_DIR)
        if len(relative.parts) < 2 or relative.parts[0] not in _ALLOWED_LEVELS:
            violations.append(f"{_rel(path)}: outside approved test levels")
    for directory in sorted({path.parent for path in _test_modules()}):
        initializer = directory / "__init__.py"
        if not initializer.is_file():
            violations.append(f"{_rel(initializer)}: missing for test directory")
    return violations


def _check_goldens() -> list[str]:
    expected_dir = _TESTS_DIR / "fixtures" / "documents" / "golden"
    violations = []
    for name in sorted(_EXPECTED_GOLDENS):
        expected = expected_dir / name
        if not expected.is_file():
            violations.append(f"{_rel(expected)}: golden fixture is missing")
        violations.extend(
            f"{_rel(path)}: golden fixture is outside its required directory"
            for path in _TESTS_DIR.rglob(name)
            if path != expected
        )
    violations.extend(
        f"{_rel(path)}: golden directory is misplaced"
        for path in _TESTS_DIR.rglob("golden")
        if path.is_dir() and path != expected_dir
    )
    return violations


def _support_imports(tree: ast.Module) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("tests.support."):
                    modules.add(alias.name.split(".")[2])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tests.support":
                modules.update(alias.name for alias in node.names)
            elif node.module.startswith("tests.support."):
                modules.add(node.module.split(".")[2])
    return modules


def _check_support_consumers() -> list[str]:
    consumers = {
        path.stem: set()
        for path in _SUPPORT_DIR.glob("*.py")
        if path.name != "__init__.py"
    }
    for path in _test_modules():
        for module in _support_imports(_tree(path)):
            if module in consumers:
                consumers[module].add(path)
    return [
        f"{_rel(_SUPPORT_DIR / (module + '.py'))}: imported by "
        f"{len(paths)} collected test modules; requires at least 2"
        for module, paths in sorted(consumers.items())
        if len(paths) < 2
    ]


def _is_platform_marker(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "platform"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
    )


def _has_platform_marker(tree: ast.Module) -> bool:
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            values = [(statement.targets, statement.value)]
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            values = [([statement.target], statement.value)]
        else:
            values = []
        for targets, value in values:
            if any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in targets
            ) and any(_is_platform_marker(node) for node in ast.walk(value)):
                return True
    return False


def _check_platform_markers() -> list[str]:
    violations = []
    for path in _test_modules():
        under_platform = path.relative_to(_TESTS_DIR).parts[0] == "platform"
        marked = _has_platform_marker(_tree(path))
        if under_platform and not marked:
            violations.append(f"{_rel(path)}: missing module-level platform marker")
        elif marked and not under_platform:
            violations.append(f"{_rel(path)}: platform marker outside tests/platform")
    return violations


def test_suite_architecture_contract():
    violations = [
        *_check_no_test_imports(),
        *_check_support_is_not_collectable(),
        *_check_sleep_calls(),
        *_check_marker_docs(),
        *_check_root_and_versioned_names(),
        *_check_levels_and_initializers(),
        *_check_goldens(),
        *_check_support_consumers(),
        *_check_platform_markers(),
    ]
    assert not violations, "Test suite architecture violations:\n" + "\n".join(
        f"  - {violation}" for violation in violations
    )
