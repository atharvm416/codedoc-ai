from __future__ import annotations

import pytest

from codedoc.core.ignore_manager import (
    END_MARKER,
    START_MARKER,
    generated_ignore_entries,
    update_output_gitignore,
)


def test_generated_entries_are_literal_sorted_and_deduplicated():
    entries = generated_ignore_entries(
        ["z.json", "a[1]*?.md", "trailing  ", "z.json"]
    )
    assert entries == [r"/a\[1\]\*\?.md", r"/trailing\ \ ", "/z.json"]


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b", "a\\b", "bad\nname"])
def test_generated_entries_reject_non_basenames(name):
    with pytest.raises(ValueError):
        generated_ignore_entries([name])


def test_update_output_gitignore_preserves_user_content(tmp_path):
    target = tmp_path / ".gitignore"
    target.write_text("user-rule\n", encoding="utf-8")
    update_output_gitignore(tmp_path, ".gitignore", ["/codedoc.json"])
    text = target.read_text(encoding="utf-8")
    assert text.startswith("user-rule\n\n")
    assert text.count(START_MARKER) == 1
    assert text.count(END_MARKER) == 1
    assert "/codedoc.json" in text
