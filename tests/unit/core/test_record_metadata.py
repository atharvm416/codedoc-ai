"""Tests organized by feature ownership."""
# ruff: noqa: F811


from __future__ import annotations

from tests.support.record_metadata_cases import private_key  # noqa: F401, F811

import json
import pytest
from codedoc.core import record_meta
from codedoc.core.project_view import (
    clean_file_record,
)
from tests.support.record_metadata_cases import _record
from codedoc.core.record_meta import PRIVATE_RECORD_KEYS
from tests.support.run_metadata_cases import _view

def test_carry_copies_only_registered_keys(private_key):
    source = {"_secret": "keep", "_other": "drop", "plain": 1}
    target: dict = {}
    record_meta.carry_private_keys(source, target)
    assert target == {"_secret": "keep"}

def test_carry_does_not_mutate_source(private_key):
    source = {"_secret": "keep"}
    snapshot = dict(source)
    record_meta.carry_private_keys(source, {})
    assert source == snapshot

def test_carry_skips_absent_keys(private_key):
    target = {"existing": 1}
    record_meta.carry_private_keys({"plain": 2}, target)
    assert target == {"existing": 1}

@pytest.mark.parametrize("value", [None, "", False, 0, [], {}])
def test_carry_preserves_falsey_values(private_key, value):
    target: dict = {}
    record_meta.carry_private_keys({"_secret": value}, target)
    assert "_secret" in target
    assert target["_secret"] == value

def test_carry_is_idempotent(private_key):
    target: dict = {}
    record_meta.carry_private_keys({"_secret": "v"}, target)
    record_meta.carry_private_keys({"_secret": "v"}, target)
    assert target == {"_secret": "v"}

def test_empty_registry_carries_nothing():
    # The production registry holds the complete private-key contract; neither
    # "_anything" is registered, so nothing is carried.
    target: dict = {}
    record_meta.carry_private_keys({"_secret": "v", "_anything": 1}, target)
    assert target == {}

def test_nested_orchestrator_result_source(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d", "_secret": "from-nested"})
    )
    assert cleaned["_secret"] == "from-nested"

def test_top_level_persisted_source(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d"}, {"_secret": "from-top"})
    )
    assert cleaned["_secret"] == "from-top"

def test_top_level_wins_when_both_present(private_key):
    cleaned = clean_file_record(
        _record(
            {"language": "python", "description": "d", "_secret": "from-nested"},
            {"_secret": "from-top"},
        )
    )
    assert cleaned["_secret"] == "from-top"

def test_falsey_registered_value_survives_pruning(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d", "_secret": ""})
    )
    assert "_secret" in cleaned and cleaned["_secret"] == ""

def test_unknown_underscore_key_is_rejected(private_key):
    cleaned = clean_file_record(
        _record({"language": "python", "description": "d", "_other": "leak"})
    )
    assert "_other" not in cleaned

def test_clean_file_does_not_mutate_source(private_key):
    documentation = {"language": "python", "description": "d", "_secret": "v"}
    record = _record(documentation)
    snapshot = json.dumps(record, sort_keys=True)
    clean_file_record(record)
    assert json.dumps(record, sort_keys=True) == snapshot

def test_underscore_file_keys_are_limited_to_registered_private_keys_and_deps():
    view = _view()
    underscored = {
        key
        for file in view["files"]
        for key in file
        if isinstance(key, str) and key.startswith("_")
    }

    assert underscored == set(PRIVATE_RECORD_KEYS) | {"_deps"}

def test_carry_orders_production_keys_canonically_from_reversed_source():
    target: dict = {}
    record_meta.carry_private_keys(
        {
            "_large_file_identity": "large-file-v1:test",
            "_split_reuse_contract": "fresh-only-v1",
            "_prompt_profile_digest": "profile-digest",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_analysis_mode": "single",
            "_analysis_revision": "file-doc-v3",
        },
        target,
    )
    assert list(target) == [
        "_analysis_revision",
        "_analysis_mode",
        "_max_context_revision",
        "_prompt_profile_digest",
        "_large_file_identity",
        "_split_reuse_contract",
    ]

def test_carry_orders_production_keys_canonically_from_shuffled_source():
    target: dict = {}
    record_meta.carry_private_keys(
        {
            "_analysis_mode": "single",
            "_prompt_profile_digest": "profile-digest",
            "_large_file_identity": "large-file-v1:test",
            "_analysis_revision": "file-doc-v3",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
        },
        target,
    )
    assert list(target) == [
        "_analysis_revision",
        "_analysis_mode",
        "_max_context_revision",
        "_prompt_profile_digest",
        "_large_file_identity",
    ]

def test_partial_key_set_preserves_canonical_relative_order():
    target: dict = {}
    record_meta.carry_private_keys(
        {
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_analysis_revision": "file-doc-v3",
        },
        target,
    )
    assert list(target) == ["_analysis_revision", "_max_context_revision"]

def test_cache_identity_keys_are_a_subset_of_registered_private_keys():
    assert record_meta.CACHE_IDENTITY_KEYS <= record_meta.PRIVATE_RECORD_KEYS
    assert set(record_meta.PRIVATE_RECORD_KEYS) == set(record_meta.PRIVATE_KEY_ORDER)


def test_retired_split_contract_stays_registered_without_an_absent_default():
    key = "_split_reuse_contract"
    assert record_meta.FRESH_SPLIT_REUSE_CONTRACT == "fresh-only-v1"
    assert key in record_meta.CACHE_IDENTITY_KEYS
    assert key in record_meta.PRIVATE_RECORD_KEYS
    # _ordinary_path_identity was appended after this key (0.14.4), so
    # _split_reuse_contract is now second-to-last rather than last.
    assert record_meta.PRIVATE_KEY_ORDER[-2] == key
    assert key not in record_meta._CACHE_KEY_ABSENT_DEFAULTS
    assert record_meta.normalized_identity_value(key, {}) is None
    assert key not in record_meta.expected_analysis_identity("single")


def test_ordinary_path_identity_appended_last_with_no_absent_default():
    key = "_ordinary_path_identity"
    assert record_meta.PRIVATE_KEY_ORDER[-1] == key
    assert key in record_meta.CACHE_IDENTITY_KEYS
    assert key in record_meta.PRIVATE_RECORD_KEYS
    assert key not in record_meta._CACHE_KEY_ABSENT_DEFAULTS
    # An absent key normalizes to None and compares unequal to any expected
    # (non-None) value, so a pre-0.14.4 record lacking it stays invalid.
    assert record_meta.normalized_identity_value(key, {}) is None
    assert key not in record_meta.expected_analysis_identity("single")


def test_ordinary_path_identity_ascii_fixed_vector():
    assert (
        record_meta.expected_ordinary_path_identity("src/app/main.py")
        == "ordinary-path-v1:"
        "fac2a67645b6e215e0a3e0ce32ef161b9034798b829f35fc7aadc2b0b45a4de5"
    )


def test_ordinary_path_identity_unicode_fixed_vector():
    assert (
        record_meta.expected_ordinary_path_identity("src/café/naïve.py")
        == "ordinary-path-v1:"
        "6dfaad6c5c469d25a74be483f6d468120b08d70637c70bce1cf8d4fa337f2d7d"
    )


def test_ordinary_path_identity_differs_by_path():
    assert record_meta.expected_ordinary_path_identity(
        "a.py"
    ) != record_meta.expected_ordinary_path_identity("b.py")


def test_ordinary_path_identity_carried_last_when_present():
    target: dict = {}
    record_meta.carry_private_keys(
        {
            "_split_reuse_contract": "fresh-only-v1",
            "_ordinary_path_identity": "ordinary-path-v1:test",
            "_analysis_revision": "file-doc-v3",
            "_analysis_mode": "single",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_prompt_profile_digest": "profile-digest",
            "_large_file_identity": "large-file-v1:test",
        },
        target,
    )
    assert list(target) == [
        "_analysis_revision",
        "_analysis_mode",
        "_max_context_revision",
        "_prompt_profile_digest",
        "_large_file_identity",
        "_split_reuse_contract",
        "_ordinary_path_identity",
    ]

def test_registered_synthetic_keys_are_carried_in_sorted_order(monkeypatch):
    monkeypatch.setattr(
        record_meta, "PRIVATE_RECORD_KEYS", frozenset({"_b_syn", "_a_syn"})
    )
    target: dict = {}
    record_meta.carry_private_keys({"_b_syn": "b", "_a_syn": "a"}, target)
    assert list(target) == ["_a_syn", "_b_syn"]

def _large_file_identity_kwargs() -> dict:
    return {
        "source_chars": 5000,
        "max_chars": 2000,
        "rel_path": "src/large.py",
        "division_plan_digest": "division-plan:" + "1" * 64,
        "reduction_tree_digest": "reduction-tree:" + "2" * 64,
        "structural_mode": "syntax",
        "imports_digest": "imports:" + "3" * 64,
    }


def test_completed_large_file_identity_binds_leaf_capsule_revision_and_bound(
    monkeypatch,
):
    """Section 5: the completed large-file identity changes whenever the
    bound leaf-capsule revision or its derived canonical-character maximum
    changes -- exactly the two inputs `0.14.4`'s leaf per-kind cap correction
    advances (`leaf-capsule-v6` -> `leaf-capsule-v7`, 200,192 -> 448,672) --
    while the `large-file-v3:` prefix itself never changes, since split
    completed identity is not versioned by the package release."""
    kwargs = _large_file_identity_kwargs()
    current_v7 = record_meta.expected_large_file_identity(**kwargs)
    assert current_v7 is not None
    assert current_v7.startswith("large-file-v3:")
    assert record_meta.LEAF_CAPSULE_SCHEMA_REVISION == "leaf-capsule-v7"
    assert record_meta.MAX_LEAF_CAPSULE_CANONICAL_CHARS == 448672

    monkeypatch.setattr(record_meta, "LEAF_CAPSULE_SCHEMA_REVISION", "leaf-capsule-v6")
    revision_reverted = record_meta.expected_large_file_identity(**kwargs)
    assert revision_reverted != current_v7
    assert revision_reverted.startswith("large-file-v3:")

    monkeypatch.setattr(record_meta, "MAX_LEAF_CAPSULE_CANONICAL_CHARS", 200192)
    bound_also_reverted = record_meta.expected_large_file_identity(**kwargs)
    assert bound_also_reverted != revision_reverted
    assert bound_also_reverted != current_v7
    assert bound_also_reverted.startswith("large-file-v3:")


def test_patched_registry_excludes_unregistered_production_keys(private_key):
    # Ordering must never widen membership: the canonical production keys are
    # unregistered here and must not be carried.
    target: dict = {}
    record_meta.carry_private_keys(
        {
            "_analysis_revision": "file-doc-v3",
            "_analysis_mode": "single",
            "_max_context_revision": "truncate-v1:max=10:head=0.7000",
            "_prompt_profile_digest": "profile-digest",
            "_secret": "keep",
        },
        target,
    )
    assert target == {"_secret": "keep"}
