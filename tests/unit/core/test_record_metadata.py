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
    # Production default: registry is empty.
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
