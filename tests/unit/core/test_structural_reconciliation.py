"""Section 2 (workstreams D/E, findings F-3/F-4/F-5): the one shared
source-backed authority for the public ``functions`` / ``classes`` /
``exports`` arrays.

Every case here uses the smallest synthetic Python / JS / TS / TSX source that
preserves the relevant syntax form -- no real project file is copied in.
"""

from __future__ import annotations

import sys

import pytest

from codedoc.core.result_assembly import flat_combined_result
from codedoc.core.structural_reconciliation import (
    LexicalProof,
    ReconciledArrays,
    StructureTruth,
    apply_structural_reconciliation,
    parser_kind_bucket,
    prove_lexical_declarations,
    reconcile_structural_arrays,
)
from codedoc.parser.language_specs import LANGUAGE_SPECS
from codedoc.parser.tree_sitter_structure import extract_structure
from tests.support.structure_extra import requires_structure_pack


# ---------------------------------------------------------------------------
# Closed kind mapping (section 5.4 item 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        # every class-like declaration node type that appears in LANGUAGE_SPECS
        "class_definition",
        "class_declaration",
        "class_specifier",
        "class",
        "interface_declaration",
        "protocol_declaration",
        "trait_item",
        "enum_declaration",
        "enum_item",
        "enum_specifier",
        "struct_declaration",
        "struct_item",
        "struct_specifier",
        "object_declaration",
        "record_declaration",
    ],
)
def test_class_like_parser_kinds_only_publish_as_classes(kind):
    assert parser_kind_bucket(kind) == "classes"


@pytest.mark.parametrize(
    "kind",
    [
        # every callable declaration node type that appears in LANGUAGE_SPECS
        "function_definition",
        "function_declaration",
        "function_item",
        "function_signature",
        "generator_function_declaration",
        "method_definition",
        "method_declaration",
        "method_signature",
        "method",
        "singleton_method",
    ],
)
def test_function_like_parser_kinds_only_publish_as_functions(kind):
    assert parser_kind_bucket(kind) == "functions"


def test_constructor_declaration_is_a_callable_never_a_class():
    # The retired split predicate mapped ``constructor_declaration`` to a class
    # purely because the string contains "struct"; the closed mapping decides it
    # explicitly as a callable member.
    assert parser_kind_bucket("constructor_declaration") == "functions"


@pytest.mark.parametrize(
    "kind",
    [
        "type_alias_declaration",  # TS `type X = ...`
        "type_declaration",        # Go: alias | struct | interface -- ambiguous
        "namespace_definition",    # C++ namespace container
        "module",                  # Ruby namespace / mixin container
        "impl_item",               # Rust impl block -- not a named type
        "element",                 # HTML markup element
        "script_element",          # HTML <script> container
        "totally_unknown_kind",    # anything the mapping does not know
        "",
    ],
)
def test_container_alias_namespace_markup_and_unknown_kinds_have_no_bucket(kind):
    assert parser_kind_bucket(kind) is None


def test_every_language_spec_declaration_kind_maps_to_a_valid_bucket_or_none():
    seen: set[str] = set()
    for spec in LANGUAGE_SPECS.values():
        for kind in spec.declaration_node_types:
            seen.add(kind)
            bucket = parser_kind_bucket(kind)
            assert bucket in ("functions", "classes", None), (kind, bucket)
    # the mapping is exercised against the real, closed grammar surface
    assert {"function_definition", "class_declaration", "element"} <= seen


def test_split_ledger_predicate_delegates_to_the_shared_closed_mapping():
    # ``file_division._structural_kind_matches`` must agree with the shared
    # authority for every kind -- including the ones the retired substring
    # predicate got wrong (constructor, type alias, namespace, element).
    from codedoc.core.file_division import _structural_kind_matches

    cases = {
        "function_declaration": "functions",
        "class_declaration": "classes",
        "method_definition": "functions",
        "interface_declaration": "classes",
        "constructor_declaration": "functions",
        "enum_declaration": "classes",
        "type_alias_declaration": None,
        "namespace_definition": None,
        "element": None,
        "type_declaration": None,
        "made_up_kind": None,
    }
    for kind, bucket in cases.items():
        assert parser_kind_bucket(kind) == bucket
        assert _structural_kind_matches(kind, "functions") is (bucket == "functions")
        assert _structural_kind_matches(kind, "classes") is (bucket == "classes")


# ---------------------------------------------------------------------------
# Conservative lexical recognizer (section 5.4)
# ---------------------------------------------------------------------------


def _names(proof_bucket):
    return [name for name, _offset in proof_bucket]


def test_python_def_async_def_and_class_are_proven():
    src = (
        "def sync_one():\n    return 1\n\n\n"
        "async def async_two():\n    return 2\n\n\n"
        "class Widget:\n    pass\n"
    )
    proof = prove_lexical_declarations(src, "python")
    assert _names(proof.functions) == ["sync_one", "async_two"]
    assert _names(proof.classes) == ["Widget"]


def test_js_function_and_class_declarations_are_proven():
    src = (
        "export function alpha() {}\n"
        "function beta() {}\n"
        "class Gamma {}\n"
        "export default class Delta {}\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert set(_names(proof.functions)) == {"alpha", "beta"}
    assert set(_names(proof.classes)) == {"Gamma", "Delta"}


def test_arrow_and_function_expression_bindings_are_proven_as_functions():
    src = (
        "const arrowOne = () => 1;\n"
        "let arrowTwo = async (a, b) => a + b;\n"
        "var funcExpr = function () { return 3; };\n"
        "const plainData = [1, 2, 3];\n"
        "const alias = someCall();\n"
    )
    proof = prove_lexical_declarations(src, "typescript")
    assert set(_names(proof.functions)) == {"arrowOne", "arrowTwo", "funcExpr"}
    assert proof.classes == ()


def test_typed_react_fc_binding_is_a_function_never_a_class():
    src = "const DPRTab: React.FC = () => { return null; };\n"
    proof = prove_lexical_declarations(src, "tsx")
    assert _names(proof.functions) == ["DPRTab"]
    assert proof.classes == ()


def test_react_fc_annotation_without_initializer_proves_nothing():
    # No `=` initializer: the binding regex requires one, so this is ambiguous
    # and omitted rather than guessed.
    src = "let DPRTab: React.FC;\n"
    proof = prove_lexical_declarations(src, "tsx")
    assert proof.functions == ()
    assert proof.classes == ()


def test_comments_and_strings_never_prove_a_declaration():
    src = (
        "// def commented_out():\n"
        "/* function blockComment() {} class BlockCls {} */\n"
        'const s = "class NotAClass {}";\n'
        "const t = `function inTemplate() {}`;\n"
        "def real_one():\n    pass\n"
    )
    proof = prove_lexical_declarations(src, "python")
    assert _names(proof.functions) == ["real_one"]
    assert proof.classes == ()

    js = (
        "// function jsComment() {}\n"
        "const banner = 'export function bannerText() {}';\n"
        "export function realJs() {}\n"
    )
    jsp = prove_lexical_declarations(js, "javascript")
    assert _names(jsp.functions) == ["realJs"]


def test_python_triple_quoted_string_body_proves_nothing():
    src = (
        'x = """\n'
        "def inside_string():\n"
        "    pass\n"
        'class AlsoInside:\n'
        '"""\n'
        "def outside():\n    pass\n"
    )
    proof = prove_lexical_declarations(src, "python")
    assert _names(proof.functions) == ["outside"]
    assert proof.classes == ()


def test_import_only_and_call_bindings_are_not_declarations():
    src = (
        "import { useState } from 'react';\n"
        "const lodash = require('lodash');\n"
        "obj.method = () => {};\n"
        "wrap(() => {});\n"
        "export function realExport() {}\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert _names(proof.functions) == ["realExport"]
    assert "useState" not in _names(proof.functions)
    assert "lodash" not in _names(proof.functions)
    assert "method" not in _names(proof.functions)


def test_explicit_named_and_default_exports_are_proven():
    src = (
        "const a = 1;\n"
        "const b = 2;\n"
        "function c() {}\n"
        "export { a, b as renamed };\n"
        "export default c;\n"
        "export const dataArray = [1, 2, 3];\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    exported = set(_names(proof.exports))
    assert {"a", "renamed", "c", "dataArray"} <= exported


def test_recognizer_is_silent_for_unsupported_languages():
    assert prove_lexical_declarations("func Main() {}\n", "go") == LexicalProof()
    assert prove_lexical_declarations("class X {}\n", "java") == LexicalProof()


# ---------------------------------------------------------------------------
# Reconciliation against lexical proof (base install / no symbols)
# ---------------------------------------------------------------------------


def _truth(rel_path, language, source, *, symbols=(), mode="lexical"):
    return StructureTruth(
        rel_path=rel_path,
        language=language,
        source=source,
        symbols=symbols,
        structural_mode=mode,
    )


def test_f3_top_level_node_script_publishes_no_invented_function():
    src = (
        "const fs = require('fs');\n"
        "const pkg = require('./package.json');\n"
        "console.log('bump', pkg.version);\n"
        "process.exit(0);\n"
    )
    out = reconcile_structural_arrays(
        [{"name": "updateEnvVersion", "description": "bumps the env version"}],
        [],
        [],
        truth=_truth("scripts/updateEnvVersion.js", "javascript", src),
    )
    assert out.functions == ()
    assert out.diagnostics["rejected_unmatched"]["functions"] == 1
    assert out.enforced is True


def test_f4_one_arrow_declaration_stays_one_function_when_reported_twice():
    src = "const handleSignUp = async (event) => { event.preventDefault(); };\n"
    out = reconcile_structural_arrays(
        [
            {"name": "handleSignUp", "description": "first"},
            {"name": "handleSignUp", "description": "second"},
        ],
        [],
        [],
        truth=_truth("src/SignUp.tsx", "tsx", src),
    )
    assert out.functions == ({"name": "handleSignUp", "description": "first"},)


def test_f5_react_fc_binding_is_never_published_as_a_class():
    src = "const DPRTab: React.FC = () => { return null; };\n"
    out = reconcile_structural_arrays(
        [],
        [{"name": "DPRTab", "description": "a project tab"}],
        [],
        truth=_truth("src/ProjectList.tsx", "tsx", src),
    )
    assert out.classes == ()
    assert out.functions == ({"name": "DPRTab", "description": "a project tab"},)


def test_react_fc_kept_as_function_only_when_the_binding_is_proven():
    # Same model claim, but the source has no proven DPRTab binding.
    src = "type DPRTab = React.FC;\n"
    out = reconcile_structural_arrays(
        [{"name": "DPRTab", "description": "x"}],
        [{"name": "DPRTab", "description": "y"}],
        [],
        truth=_truth("src/ProjectList.tsx", "tsx", src),
    )
    assert out.functions == ()
    assert out.classes == ()


def test_imported_or_called_name_is_not_a_local_declaration():
    src = (
        "import { doThing } from './thing';\n"
        "const result = compute();\n"
        "def real_local():\n    pass\n"
    )
    out = reconcile_structural_arrays(
        [
            {"name": "doThing", "description": "imported"},
            {"name": "compute", "description": "called"},
            {"name": "real_local", "description": "here"},
        ],
        [],
        [],
        truth=_truth("m.py", "python", src),
    )
    assert out.functions == ({"name": "real_local", "description": "here"},)
    assert out.diagnostics["rejected_unmatched"]["functions"] == 2


def test_filename_is_never_turned_into_a_declaration():
    out = reconcile_structural_arrays(
        [{"name": "updateEnvVersion"}],
        [],
        [],
        truth=_truth("scripts/updateEnvVersion.js", "javascript", "const noop = 1;\n"),
    )
    assert out.functions == ()


def test_parser_or_source_order_wins_over_model_order():
    src = "def gamma():\n    pass\n\n\ndef alpha():\n    pass\n\n\ndef beta():\n    pass\n"
    out = reconcile_structural_arrays(
        [
            {"name": "beta", "description": "b"},
            {"name": "alpha", "description": "a"},
            {"name": "gamma", "description": "g"},
        ],
        [],
        [],
        truth=_truth("m.py", "python", src),
    )
    assert [f["name"] for f in out.functions] == ["gamma", "alpha", "beta"]


def test_language_with_no_syntax_and_no_recognizer_omits_every_unproven_item():
    # P1-A: a structural claim must never survive without source proof. Java has
    # no lexical recognizer here and the call carries no parser symbols, so
    # nothing the model volunteered can be published -- and there is no
    # "passthrough" escape hatch.
    out = reconcile_structural_arrays(
        [{"name": "Main", "description": "d"}, {"name": "ghost", "description": "x"}],
        [{"name": "MainClass", "description": "d"}, {"name": "Ghost", "description": "x"}],
        ["Main", "GhostExport"],
        truth=_truth("Main.java", "java", "class Main { void Main() {} }\n"),
    )
    assert out.enforced is True
    assert out.functions == ()
    assert out.classes == ()
    assert out.exports == ()
    assert "passthrough_reason" not in out.diagnostics
    assert out.diagnostics["rejected_unmatched"] == {
        "functions": 2,
        "classes": 2,
        "exports": 2,
    }


def test_java_source_never_retains_an_invented_symbol_through_apply():
    truth = _truth("Svc.java", "java", "class Svc {\n  void real() {}\n}\n")
    cleaned = {
        "description": "d",
        "functions": [{"name": "ghost", "description": "invented"}],
        "classes": [{"name": "Ghost", "description": "invented"}],
        "exports": ["GhostExport"],
    }
    out = apply_structural_reconciliation(cleaned, truth)
    assert out["functions"] == []
    assert out["classes"] == []
    assert out["exports"] == []
    assert "ghost" not in __import__("json").dumps(out)
    assert "Ghost" not in __import__("json").dumps(out)


# ---------------------------------------------------------------------------
# Reconciliation against parser SymbolFacts (structure install)
# ---------------------------------------------------------------------------


@requires_structure_pack
def test_syntax_mode_rejects_a_model_function_that_matches_no_symbol():
    src = "def real_one():\n    return 1\n"
    result = extract_structure("m.py", "python", src)
    assert result.structural_mode == "syntax"
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [
            {"name": "real_one", "description": "kept"},
            {"name": "ghost_fn", "description": "invented"},
        ],
        [],
        [],
        truth=truth,
    )
    assert out.functions == ({"name": "real_one", "description": "kept"},)
    assert out.diagnostics["rejected_unmatched"]["functions"] == 1


@requires_structure_pack
def test_syntax_mode_keeps_same_name_overloads_as_two_distinct_declarations():
    src = (
        "def process(a):\n    return a\n\n\n"
        "def process(a, b):\n    return a + b\n"
    )
    result = extract_structure("m.py", "python", src)
    assert len(result.symbols) == 2
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [
            {"name": "process", "description": "first"},
            {"name": "process", "description": "second"},
        ],
        [],
        [],
        truth=truth,
    )
    # Both real declarations are preserved; without a disambiguator the model's
    # two same-name reports must NOT be positionally guessed onto them.
    names = [f["name"] for f in out.functions]
    assert names == ["process", "process"]
    assert names.count("process") == 2
    assert all("description" not in f for f in out.functions)


@requires_structure_pack
def test_reversed_model_order_with_signatures_associates_each_overload_in_source_order():
    # P1-B(2): response order must not decide which description attaches to which
    # overload -- an explicit signature does, and output stays in source order.
    src = (
        "def process(a):\n    return a\n\n\n"
        "def process(a, b, c):\n    return a\n"
    )
    result = extract_structure("m.py", "python", src)
    assert len(result.symbols) == 2
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [
            {"name": "process", "description": "three-arg", "signature": "process(a, b, c)"},
            {"name": "process", "description": "one-arg", "signature": "process(a)"},
        ],
        [],
        [],
        truth=truth,
    )
    assert [f["name"] for f in out.functions] == ["process", "process"]
    # first in source order is the one-arg overload
    assert out.functions[0]["description"] == "one-arg"
    assert out.functions[1]["description"] == "three-arg"


@requires_structure_pack
def test_two_signature_matched_overloads_with_identical_description_stay_two_entries():
    # P1-B(3): identical public {name, description} dicts for DIFFERENT
    # declarations must survive the shared authority and the public assembly.
    src = (
        "def render(a):\n    return a\n\n\n"
        "def render(a, b):\n    return a\n"
    )
    result = extract_structure("m.py", "python", src)
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [
            {"name": "render", "description": "renders", "signature": "render(a, b)"},
            {"name": "render", "description": "renders", "signature": "render(a)"},
        ],
        [],
        [],
        truth=truth,
    )
    assert list(out.functions) == [
        {"name": "render", "description": "renders"},
        {"name": "render", "description": "renders"},
    ]
    # ... and through the public flat assembly, which is told the arrays are
    # already source-backed so the two overloads are not collapsed.
    flat = flat_combined_result(
        "m.py",
        "python",
        [],
        {
            "description": "d",
            "functions": list(out.functions),
            "classes": [],
            "exports": [],
        },
        reconciled_structure=True,
    )
    assert [f["name"] for f in flat["functions"]] == ["render", "render"]
    assert flat["functions"] == [
        {"name": "render", "description": "renders"},
        {"name": "render", "description": "renders"},
    ]
    assert flat["structure"]["functions"] == flat["functions"]


@requires_structure_pack
def test_two_reports_about_one_symbol_still_collapse_to_one_entry():
    # The overload preservation must not defeat F-4: one real declaration
    # reported twice is still one published entry.
    src = "def once(a):\n    return a\n"
    result = extract_structure("m.py", "python", src)
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [
            {"name": "once", "description": "first wins"},
            {"name": "once", "description": "second ignored"},
        ],
        [],
        [],
        truth=truth,
    )
    assert list(out.functions) == [{"name": "once", "description": "first wins"}]


@requires_structure_pack
def test_ambiguous_same_name_without_evidence_is_not_assigned_by_guessing():
    src = (
        "def build(a):\n    return a\n\n\n"
        "def build(a, b):\n    return a\n"
    )
    result = extract_structure("m.py", "python", src)
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [{"name": "build", "description": "only one report, two decls"}],
        [],
        [],
        truth=truth,
    )
    # both real declarations are still surfaced; the lone description is not
    # guessed onto either.
    assert [f["name"] for f in out.functions] == ["build", "build"]
    assert all("description" not in f for f in out.functions)


def test_two_same_name_lexical_declarations_preserve_two_identities():
    # P1-B(1): the lexical recognizer's identity is occurrence-based, not
    # name-only, so two real same-name declarations do not collapse.
    src = (
        "def handler(a):\n    return a\n\n\n"
        "def handler(a, b):\n    return a\n"
    )
    proof = prove_lexical_declarations(src, "python")
    assert _names(proof.functions) == ["handler", "handler"]
    offsets = [off for _name, off in proof.functions]
    assert offsets[0] != offsets[1]

    out = reconcile_structural_arrays(
        [{"name": "handler", "description": "d"}],
        [],
        [],
        truth=_truth("m.py", "python", src),
    )
    assert [f["name"] for f in out.functions] == ["handler", "handler"]


@requires_structure_pack
def test_syntax_mode_kind_is_authoritative_over_the_model_claim():
    src = "class Registry:\n    pass\n"
    result = extract_structure("m.py", "python", src)
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [{"name": "Registry", "description": "wrongly a function"}],
        [],
        [],
        truth=truth,
    )
    assert out.functions == ()
    assert out.classes == ({"name": "Registry", "description": "wrongly a function"},)


# ---------------------------------------------------------------------------
# base lexical vs structure syntax compatibility (section 5.5 last bullet)
# ---------------------------------------------------------------------------


@requires_structure_pack
def test_base_lexical_and_structure_syntax_agree_for_the_closed_fixtures(monkeypatch):
    src = (
        "import React from 'react';\n"
        "export function handleSignUp() { return 1; }\n"
        "const DPRTab: React.FC = () => { return null; };\n"
        "class LegacyPanel { render() {} }\n"
    )
    model_functions = [
        {"name": "handleSignUp", "description": "signs up"},
        {"name": "handleSignUp", "description": "dup"},
        {"name": "DPRTab", "description": "tab"},
        {"name": "updateEnvVersion", "description": "invented"},
    ]
    model_classes = [
        {"name": "DPRTab", "description": "not a class"},
        {"name": "LegacyPanel", "description": "a class"},
    ]

    structure_result = extract_structure("src/Panel.tsx", "tsx", src)
    structure_truth = _truth(
        "src/Panel.tsx", "tsx", src,
        symbols=structure_result.symbols,
        mode=structure_result.structural_mode,
    )
    structure_out = reconcile_structural_arrays(
        list(model_functions), list(model_classes), [], truth=structure_truth
    )

    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    lexical_result = extract_structure("src/Panel.tsx", "tsx", src)
    assert lexical_result.structural_mode == "lexical"
    lexical_truth = _truth("src/Panel.tsx", "tsx", src, mode="lexical")
    lexical_out = reconcile_structural_arrays(
        list(model_functions), list(model_classes), [], truth=lexical_truth
    )

    assert {f["name"] for f in structure_out.functions} == {"handleSignUp", "DPRTab"}
    assert {c["name"] for c in structure_out.classes} == {"LegacyPanel"}
    assert {f["name"] for f in lexical_out.functions} == {"handleSignUp", "DPRTab"}
    assert {c["name"] for c in lexical_out.classes} == {"LegacyPanel"}


# ---------------------------------------------------------------------------
# privacy / bounded diagnostics (section 5.4 item 10)
# ---------------------------------------------------------------------------


@requires_structure_pack
def test_reconciliation_never_leaks_internal_evidence_to_public_arrays():
    src = "def kept():\n    '''sensitive-doc'''\n    return 1\n"
    result = extract_structure("m.py", "python", src)
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    cleaned = {
        "description": "d",
        "functions": [
            {
                "name": "kept",
                "description": "ok",
                "signature": "def kept()",
                "_provenance": [{"symbol_id": "symbol_deadbeef"}],
            }
        ],
        "classes": [],
        "exports": [],
    }
    reconciled = apply_structural_reconciliation(cleaned, truth)
    published = reconciled["functions"][0]
    assert set(published) == {"name", "description"}
    assert "signature" not in published
    assert "_provenance" not in published
    assert "symbol_id" not in published


def test_diagnostics_are_bounded_value_safe_counts_only():
    src = "def only():\n    pass\n"
    out = reconcile_structural_arrays(
        [{"name": "only"}, {"name": "ghost", "description": "x" * 5000}],
        [],
        [],
        truth=_truth("m.py", "python", src),
    )
    diag = out.diagnostics
    assert set(diag) >= {"path", "structural_mode", "matched", "rejected_unmatched"}
    for section in ("matched", "rejected_unmatched"):
        for value in diag[section].values():
            assert isinstance(value, int) and 0 <= value <= 10_000
    # No source text, response text, signature, or id anywhere in diagnostics.
    blob = repr(diag)
    assert "ghost" not in blob and "xxxx" not in blob and "only" not in blob


def test_apply_structural_reconciliation_leaves_non_structural_fields_untouched():
    src = "def kept():\n    pass\n"
    cleaned = {
        "description": "keep me",
        "role_in_system": "keep me too",
        "functions": [{"name": "kept", "description": "d"}, {"name": "gone"}],
        "classes": [],
        "exports": [],
        "key_concepts": ["a", "b"],
        "dependencies_analysis": {"external": ["x"]},
    }
    out = apply_structural_reconciliation(
        cleaned, _truth("m.py", "python", src)
    )
    assert out["description"] == "keep me"
    assert out["role_in_system"] == "keep me too"
    assert out["key_concepts"] == ["a", "b"]
    assert out["dependencies_analysis"] == {"external": ["x"]}
    assert [f["name"] for f in out["functions"]] == ["kept"]


def test_reconciled_arrays_is_the_public_contract_shape():
    out = reconcile_structural_arrays([], [], [], truth=_truth("m.py", "python", ""))
    assert isinstance(out, ReconciledArrays)
    assert out.functions == () and out.classes == () and out.exports == ()


# ---------------------------------------------------------------------------
# P2-B: closed export proof -- accepted forms
# ---------------------------------------------------------------------------


def test_esm_export_declaration_and_named_and_default_exports_are_proven():
    src = (
        "export function alpha() {}\n"
        "export const beta = () => 1;\n"
        "const gamma = 3;\n"
        "export { gamma, beta as bAlias };\n"
        "export default alpha;\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert {"alpha", "beta", "gamma", "bAlias"} <= set(_names(proof.exports))


def test_named_re_export_and_namespace_re_export_are_proven():
    src = (
        "export { readFile, writeFile } from './fs-helpers';\n"
        "export * as helpers from './helpers';\n"
        "export * from './everything';\n"  # cannot be named -> omitted
    )
    proof = prove_lexical_declarations(src, "typescript")
    names = set(_names(proof.exports))
    assert {"readFile", "writeFile", "helpers"} <= names
    assert "everything" not in names


def test_commonjs_module_exports_and_exports_property_are_proven():
    src = (
        "function core() {}\n"
        "exports.core = core;\n"
        "module.exports.helper = function () {};\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert {"core", "helper"} <= set(_names(proof.exports))


def test_commonjs_single_line_object_export_is_proven():
    src = "const a = 1;\nconst b = 2;\nmodule.exports = { a, b, renamed: a };\n"
    proof = prove_lexical_declarations(src, "javascript")
    assert {"a", "b", "renamed"} <= set(_names(proof.exports))


def test_python_static_dunder_all_is_proven_as_exports():
    src = (
        "def public_one():\n    pass\n\n\n"
        "def _private():\n    pass\n\n\n"
        "__all__ = ['public_one', 'also_ok']\n"
    )
    proof = prove_lexical_declarations(src, "python")
    assert {"public_one", "also_ok"} <= set(_names(proof.exports))


def test_reconcile_publishes_only_dunder_all_names_for_python():
    src = "def kept():\n    pass\n\n\n__all__ = ['kept']\n"
    out = reconcile_structural_arrays(
        [{"name": "kept", "description": "d"}],
        [],
        ["kept", "not_in_all"],
        truth=_truth("m.py", "python", src),
    )
    assert out.exports == ("kept",)


# ---------------------------------------------------------------------------
# P2-B: closed export proof -- rejected forms
# ---------------------------------------------------------------------------


def test_dynamic_or_computed_exports_prove_nothing():
    src = (
        "const name = 'foo';\n"
        "module.exports[name] = 1;\n"          # computed
        "module.exports = makeExports();\n"    # dynamic call
        "exports['bar'] = 2;\n"                # computed string index
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert proof.exports == ()


def test_property_access_that_is_not_an_export_statement_proves_nothing():
    src = (
        "const config = {};\n"
        "config.value = 1;\n"
        "this.state = 2;\n"
        "obj.exports = 3;\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert proof.exports == ()


def test_dunder_all_with_a_dynamic_element_voids_the_whole_proof():
    src = (
        "def kept():\n    pass\n\n\n"
        "__all__ = ['kept'] + _extra_names()\n"
    )
    proof = prove_lexical_declarations(src, "python")
    assert proof.exports == ()


def test_exports_in_comments_or_strings_prove_nothing():
    src = (
        "// export const commented = 1;\n"
        "const doc = 'module.exports = { fake: 1 };';\n"
        "export const realThing = () => 1;\n"
    )
    proof = prove_lexical_declarations(src, "javascript")
    assert _names(proof.exports) == ["realThing"]


@pytest.mark.parametrize("language", ["java", "go", "dart", "csharp", "ruby"])
def test_reconcile_omits_every_model_export_for_a_language_without_export_proof(language):
    out = reconcile_structural_arrays(
        [],
        [],
        ["someName", "another"],
        truth=_truth(f"m.{language}", language, "whatever source\n"),
    )
    assert out.exports == ()
    assert out.enforced is True


# ---------------------------------------------------------------------------
# P1-A: no model-owned structural passthrough for any language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language, source",
    [
        ("go", "package main\nfunc Real() {}\n"),
        ("rust", "fn real() {}\n"),
        ("dart", "void real() {}\n"),
        ("csharp", "class C { void Real() {} }\n"),
    ],
)
def test_unproven_functions_and_classes_are_omitted_for_every_language(language, source):
    out = reconcile_structural_arrays(
        [{"name": "invented", "description": "d"}],
        [{"name": "Invented", "description": "d"}],
        [],
        truth=_truth(f"m.{language}", language, source),
    )
    assert out.functions == ()
    assert out.classes == ()
    assert out.enforced is True


# ---------------------------------------------------------------------------
# P2: a model claim never publishes an incompatible same-name kind
# ---------------------------------------------------------------------------


_SAME_NAME_FN_AND_CLS = (
    "def Same():\n    return 1\n\n\n"
    "class Same:\n    pass\n"
)


def test_function_only_claim_publishes_only_the_function_not_the_class():
    out = reconcile_structural_arrays(
        [{"name": "Same", "description": "function only"}],
        [],
        [],
        truth=_truth("m.py", "python", _SAME_NAME_FN_AND_CLS),
    )
    assert list(out.functions) == [{"name": "Same", "description": "function only"}]
    assert out.classes == ()


def test_class_only_claim_publishes_only_the_class_not_the_function():
    out = reconcile_structural_arrays(
        [],
        [{"name": "Same", "description": "class only"}],
        [],
        truth=_truth("m.py", "python", _SAME_NAME_FN_AND_CLS),
    )
    assert list(out.classes) == [{"name": "Same", "description": "class only"}]
    assert out.functions == ()


def test_both_kinds_claimed_publishes_both_each_with_its_own_description():
    out = reconcile_structural_arrays(
        [{"name": "Same", "description": "the function"}],
        [{"name": "Same", "description": "the class"}],
        [],
        truth=_truth("m.py", "python", _SAME_NAME_FN_AND_CLS),
    )
    assert list(out.functions) == [{"name": "Same", "description": "the function"}]
    assert list(out.classes) == [{"name": "Same", "description": "the class"}]


def test_reversing_provider_order_does_not_change_incompatible_kind_result():
    a = reconcile_structural_arrays(
        [{"name": "Same", "description": "fn"}],
        [{"name": "Same", "description": "cls"}],
        [],
        truth=_truth("m.py", "python", _SAME_NAME_FN_AND_CLS),
    )
    b = reconcile_structural_arrays(
        [{"name": "z", "description": "x"}, {"name": "Same", "description": "fn"}],
        [{"name": "Same", "description": "cls"}],
        [],
        truth=_truth("m.py", "python", _SAME_NAME_FN_AND_CLS),
    )
    assert a.functions == b.functions == ({"name": "Same", "description": "fn"},)
    assert a.classes == b.classes == ({"name": "Same", "description": "cls"},)


def test_wrong_kind_react_fc_class_claim_still_corrects_to_a_function():
    # F-5: no class candidate, exactly one unambiguous alternative kind (the
    # proven React.FC binding) -- the source kind overrides the model.
    src = "const DPRTab: React.FC = () => { return null; };\n"
    out = reconcile_structural_arrays(
        [],
        [{"name": "DPRTab", "description": "a tab"}],
        [],
        truth=_truth("src/Tab.tsx", "tsx", src),
    )
    assert list(out.functions) == [{"name": "DPRTab", "description": "a tab"}]
    assert out.classes == ()


@requires_structure_pack
def test_wrong_kind_claim_with_multiple_incompatible_candidates_is_omitted():
    # Two real same-name classes, a lone wrong-kind function claim: the kind
    # correction target is ambiguous, so nothing is published rather than
    # guessed.
    src = "class Dup:\n    pass\n\n\nclass Dup:\n    def m(self):\n        return 1\n"
    result = extract_structure("m.py", "python", src)
    assert sum(1 for s in result.symbols if s.kind == "class_definition") == 2
    truth = _truth("m.py", "python", src, symbols=result.symbols, mode="syntax")
    out = reconcile_structural_arrays(
        [{"name": "Dup", "description": "wrongly a function"}],
        [],
        [],
        truth=truth,
    )
    assert out.functions == ()
    assert out.classes == ()


def test_incompatible_kind_selection_is_identical_for_split_and_ordinary():
    truth = _truth("m.py", "python", _SAME_NAME_FN_AND_CLS)
    ordinary = apply_structural_reconciliation(
        {
            "description": "d",
            "functions": [{"name": "Same", "description": "fn only"}],
            "classes": [],
            "exports": [],
        },
        truth,
    )
    split = reconcile_structural_arrays(
        [{"name": "Same", "description": "fn only"}], [], [], truth=truth
    )
    assert ordinary["functions"] == [{"name": "Same", "description": "fn only"}]
    assert ordinary["classes"] == []
    assert list(split.functions) == [{"name": "Same", "description": "fn only"}]
    assert split.classes == ()


# ---------------------------------------------------------------------------
# P2: proven exports keep canonical source order, never alphabetical
# ---------------------------------------------------------------------------


def test_two_names_on_one_export_line_keep_source_order_not_alphabetical():
    out = reconcile_structural_arrays(
        [],
        [],
        ["alpha", "zeta"],  # model order: alphabetical, reversed vs source
        truth=_truth("m.js", "javascript", "export { zeta, alpha };\n"),
    )
    assert out.exports == ("zeta", "alpha")


def test_named_re_export_line_keeps_source_order():
    src = 'export { localZ as publicZ, localA as publicA } from "./module";\n'
    out = reconcile_structural_arrays(
        [], [], ["publicA", "publicZ"],
        truth=_truth("m.ts", "typescript", src),
    )
    assert out.exports == ("publicZ", "publicA")


def test_commonjs_object_export_line_keeps_source_order():
    src = "const value = 1;\nmodule.exports = { zeta, alpha, middle: value };\n"
    out = reconcile_structural_arrays(
        [], [], ["middle", "alpha", "zeta"],
        truth=_truth("m.js", "javascript", src),
    )
    assert out.exports == ("zeta", "alpha", "middle")


def test_python_dunder_all_keeps_declared_order():
    src = 'def a():\n    pass\n\n\n__all__ = ["zeta", "alpha", "middle"]\n'
    out = reconcile_structural_arrays(
        [], [], ["middle", "zeta", "alpha"],
        truth=_truth("m.py", "python", src),
    )
    assert out.exports == ("zeta", "alpha", "middle")


def test_export_order_survives_shuffled_model_input_and_duplicates_collapse():
    src = "export { gamma, beta, alpha };\n"
    out = reconcile_structural_arrays(
        [], [], ["alpha", "beta", "alpha", "gamma", "beta"],
        truth=_truth("m.js", "javascript", src),
    )
    assert out.exports == ("gamma", "beta", "alpha")


def test_export_order_through_public_flat_assembly():
    truth = _truth("m.js", "javascript", "export { zeta, alpha, middle };\n")
    reconciled = apply_structural_reconciliation(
        {
            "description": "d",
            "functions": [],
            "classes": [],
            "exports": ["middle", "alpha", "zeta"],
        },
        truth,
    )
    flat = flat_combined_result("m.js", "javascript", [], reconciled)
    assert flat["exports"] == ["zeta", "alpha", "middle"]
    assert flat["structure"]["exports"] == ["zeta", "alpha", "middle"]


def test_export_order_position_is_not_exposed_publicly():
    truth = _truth("m.js", "javascript", "export { zeta, alpha };\n")
    out = reconcile_structural_arrays([], [], ["alpha", "zeta"], truth=truth)
    assert out.exports == ("zeta", "alpha")
    assert all(isinstance(name, str) for name in out.exports)
    blob = repr(out.diagnostics)
    for token in ("position", "ordinal", "occurrence"):
        assert token not in blob


def test_comments_strings_and_computed_exports_still_rejected_with_ordering_fix():
    src = (
        "// export { commented };\n"
        "const s = 'export { inString };';\n"
        "const k = 'x';\n"
        "module.exports[k] = 1;\n"
        "export { realA, realB };\n"
    )
    out = reconcile_structural_arrays(
        [], [], ["inString", "commented", "realB", "realA"],
        truth=_truth("m.js", "javascript", src),
    )
    assert out.exports == ("realA", "realB")
