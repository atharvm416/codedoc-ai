from __future__ import annotations

import os

import pytest

from codedoc.core.block_manager import BlockError, merge_managed_block, write_owned_block

START = "# start"
END = "# end"


def _can_symlink(tmp_path) -> bool:
    target = tmp_path / "_probe_target"
    link = tmp_path / "_probe_link"
    target.write_text("x\n", encoding="utf-8")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError, AttributeError):
        return False
    finally:
        if link.is_symlink() or link.exists():
            link.unlink()
        target.unlink()
    return True


def test_managed_block_create_append_replace_and_idempotence(tmp_path):
    created = merge_managed_block(None, ["/a", "/b"], START, END)
    assert created == "# start\n/a\n/b\n# end\n"

    appended = merge_managed_block("user rule\n", ["/a"], START, END)
    assert appended == "user rule\n\n# start\n/a\n# end\n"
    replaced = merge_managed_block(appended, ["/b"], START, END)
    assert replaced == "user rule\n\n# start\n/b\n# end\n"
    assert merge_managed_block(replaced, ["/b"], START, END) == replaced

    target = tmp_path / ".gitignore"
    write_owned_block(target, ["/b"], START, END)
    assert target.read_text(encoding="utf-8") == "# start\n/b\n# end\n"


def test_managed_block_preserves_crlf():
    existing = "user\r\n\r\n# start\r\n/old\r\n# end\r\n"
    merged = merge_managed_block(existing, ["/new"], START, END)
    assert merged == "user\r\n\r\n# start\r\n/new\r\n# end\r\n"


@pytest.mark.parametrize(
    "text",
    [
        "# start\n/a\n",
        "# end\n/a\n",
        "# end\n# start\n",
        "# start\n# start\n# end\n",
        "# start\n# end\n# end\n",
    ],
)
def test_managed_block_rejects_malformed_markers(text):
    with pytest.raises(BlockError):
        merge_managed_block(text, ["/a"], START, END)


@pytest.mark.parametrize(
    ("lines", "start", "end"),
    [(["bad\nline"], START, END), ([START], START, END), ([], "", END), ([], START, START)],
)
def test_managed_block_rejects_invalid_arguments(lines, start, end):
    with pytest.raises(ValueError):
        merge_managed_block(None, lines, start, end)


def test_write_owned_block_rejects_invalid_utf8_and_directory(tmp_path):
    invalid = tmp_path / "invalid"
    invalid.write_bytes(b"\xff")
    with pytest.raises(BlockError):
        write_owned_block(invalid, ["/a"], START, END)
    assert invalid.read_bytes() == b"\xff"

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(BlockError):
        write_owned_block(directory, ["/a"], START, END)


def test_write_owned_block_rejects_symlink_target(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation unavailable on this platform")
    real = tmp_path / "real_target"
    real.write_text("user\n", encoding="utf-8")
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(BlockError):
        write_owned_block(link, ["/a"], START, END)
    # The link target is never overwritten.
    assert real.read_text(encoding="utf-8") == "user\n"

