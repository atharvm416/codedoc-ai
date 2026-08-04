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
    assert record_meta.PRIVATE_KEY_ORDER[-1] == key
    assert key not in record_meta._CACHE_KEY_ABSENT_DEFAULTS
    assert record_meta.normalized_identity_value(key, {}) is None
    assert key not in record_meta.expected_analysis_identity("single")

def test_registered_synthetic_keys_are_carried_in_sorted_order(monkeypatch):
    monkeypatch.setattr(
        record_meta, "PRIVATE_RECORD_KEYS", frozenset({"_b_syn", "_a_syn"})
    )
    target: dict = {}
    record_meta.carry_private_keys({"_b_syn": "b", "_a_syn": "a"}, target)
    assert list(target) == ["_a_syn", "_b_syn"]

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
