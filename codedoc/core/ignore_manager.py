"""Owned `.gitignore` entries for completed codedoc artifacts."""

from __future__ import annotations

from pathlib import Path

from codedoc.core.block_manager import write_owned_block

START_MARKER = "# >>> codedoc-managed (do not edit this block) >>>"
END_MARKER = "# <<< codedoc-managed <<<"


def _literal_pattern(filename: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or any(ord(char) < 32 or ord(char) == 127 for char in filename)
    ):
        raise ValueError("artifact filenames must be non-empty portable basenames")
    escaped = filename.replace("\\", "\\\\")
    for char in ("*", "?", "[", "]"):
        escaped = escaped.replace(char, f"\\{char}")
    trailing_spaces = len(escaped) - len(escaped.rstrip(" "))
    if trailing_spaces:
        escaped = escaped[:-trailing_spaces] + "\\ " * trailing_spaces
    return f"/{escaped}"


def generated_ignore_entries(artifact_filenames: list[str]) -> list[str]:
    """Return sorted, de-duplicated root-anchored literal ignore patterns."""
    if not isinstance(artifact_filenames, list):
        raise ValueError("artifact_filenames must be a list")
    return sorted({_literal_pattern(filename) for filename in artifact_filenames})


def update_output_gitignore(output_dir: Path, filename: str, entries: list[str]) -> None:
    """Replace codedoc's owned block in the configured output ignore file."""
    write_owned_block(Path(output_dir) / filename, entries, START_MARKER, END_MARKER)
