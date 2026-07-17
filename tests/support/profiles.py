"""Shared prompt-profile literal fixtures, relocated from
``tests/test_110_prompt_profile_cli.py`` for reuse across collected test
modules without importing one collected test module from another.
"""
from __future__ import annotations

INLINE = {"schema_version": 1, "single": {"common": {"fields": [
    {"key": "description", "type": "string", "instruction": "Describe the file."},
    {"key": "key_concepts", "type": "string_list", "instruction": "List concepts."},
]}}}
