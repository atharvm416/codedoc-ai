"""0.12.0 — deterministic extension-scope resolution and memoization."""

from codedoc.core.prompt_profiles import (
    ResolvedProfile,
    build_resolved_profile,
    classify_profile_action,
    documentation_projectable_paths,
    validate_profile,
)


def _resolved(raw, mode="single", known=frozenset({".py", ".js", ".ts", ".cs"})):
    profile = validate_profile(
        raw, active_mode=mode, known_extensions=known, source="inline", source_path=None,
    )
    return ResolvedProfile(mode, profile)


def _single(common_desc, per_extension):
    return {
        "single": {
            "common": {"requested_shape": {"description": common_desc}},
            "per_extension": per_extension,
        }
    }


def test_longest_suffix_wins():
    resolved = _resolved(_single("base", {
        ".ts": {"requested_shape": {"description": "TS override"}},
        ".d.ts": {"requested_shape": {"description": "DTS override"}},
    }))
    assert "DTS override" in resolved.resolve_block("combined", "types.d.ts").text
    ts_text = resolved.resolve_block("combined", "widget.ts").text
    assert "TS override" in ts_text and "DTS override" not in ts_text


def test_case_insensitive_basename_matching():
    resolved = _resolved(_single("base", {
        ".d.ts": {"requested_shape": {"description": "DTS override"}},
    }))
    assert "DTS override" in resolved.resolve_block("combined", "Types.D.TS").text


def test_dotfile_named_exactly_like_suffix_does_not_match():
    resolved = _resolved(_single("base desc", {
        ".ts": {"requested_shape": {"description": "TS override"}},
    }))
    # A file whose entire name is ".ts" is not a ".ts"-suffixed file.
    assert "base desc" in resolved.resolve_block("combined", ".ts").text


def test_precedence_extension_beats_common_beats_builtin():
    resolved = _resolved(_single("common desc", {
        ".js": {"requested_shape": {"description": "js desc"}},
    }))
    js_sel = resolved.resolve_bundle(resolved.scope_for({"rel_path": "a.js"})).selections["combined"]
    py_sel = resolved.resolve_bundle(resolved.scope_for({"rel_path": "a.py"})).selections["combined"]
    assert "js desc" in js_sel.block.text          # extension beats common
    assert "common desc" in py_sel.block.text       # common beats built-in
    assert js_sel.scope == "extension" and js_sel.selector == ".js"
    assert py_sel.scope == "common" and py_sel.selector is None
    # No profile at all -> built-in.
    builtin = ResolvedProfile("single", None)
    builtin_sel = builtin.resolve_bundle(builtin.scope_for({"rel_path": "a.py"})).selections["combined"]
    assert builtin_sel.scope == "built-in"


def test_unused_extension_entry_changes_nothing():
    base = _resolved(_single("common desc", {}))
    with_unused = _resolved(_single("common desc", {
        ".cs": {"requested_shape": {"description": "cs desc"}},
    }))
    # A .py file never touches the .cs override; digest and text are identical.
    assert base.file_digest("a.py") == with_unused.file_digest("a.py")
    assert base.resolve_block("combined", "a.py").text == \
        with_unused.resolve_block("combined", "a.py").text


def test_single_mode_resolves_one_combined_block():
    resolved = _resolved(_single("common desc", {}))
    bundle = resolved.resolve_bundle(resolved.scope_for({"rel_path": "a.py"}))
    assert set(bundle.selections) == {"combined"}


def test_triple_mode_resolves_three_distinct_blocks_in_one_bundle():
    raw = {
        "triple": {
            "common": {
                "structure": {"requested_shape": {"description": "struct desc"}},
                "dependency": {"requested_shape": {
                    "dependencies_analysis": {"warnings": ["dep warn"]}}},
                "documentation": {"requested_shape": {"description": "doc desc"}},
            },
            "per_extension": {},
        }
    }
    resolved = _resolved(raw, mode="triple")
    bundle = resolved.resolve_bundle(resolved.scope_for({"rel_path": "a.py"}))
    assert set(bundle.selections) == {"structure", "dependency", "documentation"}
    texts = {a: sel.block.text for a, sel in bundle.selections.items()}
    # Three distinct blocks, never collapsed.
    assert len(set(texts.values())) == 3
    assert "struct desc" in texts["structure"]
    assert "doc desc" in texts["documentation"]


def test_single_only_triple_projection_includes_per_extension():
    raw = _single("Base doc.", {
        ".js": {"requested_shape": {
            **{"description": "JS doc."},
        }},
    })
    profile = validate_profile(
        raw, active_mode="triple", known_extensions=frozenset({".py", ".js"}),
        source="inline", source_path=None,
    )
    resolved = build_resolved_profile(classify_profile_action(profile, "triple"), "triple")
    # Structure/dependency stay built-in; documentation projects common + per_extension.
    assert not resolved.resolve_block("structure", "a.js").active
    assert not resolved.resolve_block("dependency", "a.js").active
    js_doc = resolved.resolve_block("documentation", "a.js")
    py_doc = resolved.resolve_block("documentation", "a.py")
    assert "JS doc." in js_doc.text
    assert "Base doc." in py_doc.text
    assert set(js_doc.requested_field_paths) <= documentation_projectable_paths()


def test_resolve_bundle_is_memoized_per_selector():
    resolved = _resolved(_single("base", {
        ".js": {"requested_shape": {"description": "js"}},
    }))
    b1 = resolved.resolve_bundle(resolved.scope_for({"rel_path": "one.js"}))
    b2 = resolved.resolve_bundle(resolved.scope_for({"rel_path": "two.js"}))
    b3 = resolved.resolve_bundle(resolved.scope_for({"rel_path": "a.py"}))
    b4 = resolved.resolve_bundle(resolved.scope_for({"rel_path": "b.py"}))
    # Same matched selector -> shared memoized selections object.
    assert b1.selections is b2.selections
    assert b3.selections is b4.selections
    assert b1.selections is not b3.selections
    # At most len(per_extension) + 1 = 2 distinct cores rendered.
    assert len(resolved._core_cache) == 2
