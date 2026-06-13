"""0.9.3 — pure dependency classifier (codedoc.core.dependency_kind).

Verifies language-aware canonicalization and external/sdk classification.
The classifier is pure, deterministic, filesystem-free, and never returns
``internal``.
"""
from __future__ import annotations

import sys

import pytest

from codedoc.core.dependency_kind import (
    KIND_EXTERNAL,
    KIND_SDK,
    classify_non_project_dependency,
)


# ---------------------------------------------------------------------------
# Dart
# ---------------------------------------------------------------------------

def test_dart_sdk_library_retains_full_identifier():
    dep = classify_non_project_dependency("dart:async", "dart")
    assert dep.kind == KIND_SDK
    assert dep.canonical == "dart:async"


def test_dart_package_subpath_canonicalizes_to_first_segment():
    dep = classify_non_project_dependency("package:flutter/material.dart", "dart")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "flutter"


def test_dart_strips_single_package_prefix():
    dep = classify_non_project_dependency("package:provider/provider.dart", "dart")
    assert dep.canonical == "provider"


def test_dart_bare_package_is_external():
    dep = classify_non_project_dependency("flutter", "dart")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "flutter"


def test_dart_relative_kept_relative_and_external():
    dep = classify_non_project_dependency("../widgets/button.dart", "dart")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "../widgets/button.dart"


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["os", "os.path", "json", "collections.abc"])
def test_python_stdlib_is_sdk(name):
    dep = classify_non_project_dependency(name, "python")
    assert dep.kind == KIND_SDK
    assert dep.canonical == name.split(".", 1)[0]


@pytest.mark.parametrize("name", ["requests", "requests.adapters", "flask"])
def test_python_third_party_is_external(name):
    dep = classify_non_project_dependency(name, "python")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == name.split(".", 1)[0]


def test_python_uses_committed_fallback_when_no_stdlib_names(monkeypatch):
    # Force the 3.9 fallback path by removing the interpreter attribute.
    monkeypatch.delattr(sys, "stdlib_module_names", raising=False)
    assert classify_non_project_dependency("os", "python").kind == KIND_SDK
    assert classify_non_project_dependency("requests", "python").kind == KIND_EXTERNAL


def test_python_relative_dotted_yields_empty_canonical():
    dep = classify_non_project_dependency(".utils", "python")
    assert dep.canonical == ""


# ---------------------------------------------------------------------------
# Node (javascript / jsx / typescript / tsx)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["javascript", "jsx", "typescript", "tsx"])
def test_node_builtin_is_sdk(lang):
    dep = classify_non_project_dependency("fs", lang)
    assert dep.kind == KIND_SDK
    assert dep.canonical == "fs"


def test_node_prefixed_builtin_is_sdk():
    dep = classify_non_project_dependency("node:fs", "javascript")
    assert dep.kind == KIND_SDK
    assert dep.canonical == "fs"


def test_node_builtin_subpath_is_sdk():
    dep = classify_non_project_dependency("fs/promises", "javascript")
    assert dep.kind == KIND_SDK
    assert dep.canonical == "fs"


def test_node_scoped_package_canonicalizes_to_scope_and_name():
    dep = classify_non_project_dependency("@scope/pkg/sub/mod", "typescript")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "@scope/pkg"


def test_node_unscoped_package_subpath_canonicalizes_to_first_segment():
    dep = classify_non_project_dependency("lodash/debounce", "javascript")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "lodash"


def test_node_relative_kept_relative_and_external():
    dep = classify_non_project_dependency("./components/Button", "tsx")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "./components/Button"


# ---------------------------------------------------------------------------
# Unknown language, empty input, idempotence
# ---------------------------------------------------------------------------

def test_unknown_language_is_external():
    dep = classify_non_project_dependency("some_pkg", "ruby")
    assert dep.kind == KIND_EXTERNAL
    assert dep.canonical == "some_pkg"


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_input_yields_empty_canonical(raw):
    dep = classify_non_project_dependency(raw, "python")
    assert dep.canonical == ""


def test_language_tag_is_case_insensitive():
    dep = classify_non_project_dependency("dart:io", "DART")
    assert dep.kind == KIND_SDK


@pytest.mark.parametrize(
    "raw,lang",
    [
        ("package:flutter/material.dart", "dart"),
        ("dart:async", "dart"),
        ("os.path", "python"),
        ("requests.adapters", "python"),
        ("node:fs", "javascript"),
        ("@scope/pkg/sub", "typescript"),
        ("lodash/debounce", "javascript"),
    ],
)
def test_classifying_a_canonical_value_is_idempotent(raw, lang):
    first = classify_non_project_dependency(raw, lang)
    second = classify_non_project_dependency(first.canonical, lang)
    assert second.canonical == first.canonical
    assert second.kind == first.kind
