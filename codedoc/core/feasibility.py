"""Deterministic advisories for infeasible cross-file prompt instructions."""

from __future__ import annotations

import re

from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedProfile,
    build_review_units,
    field_index,
)

CROSS_FILE_SIGNALS = (
    ("every_file", "every file", re.compile(r"\bevery\s+files?\b", re.IGNORECASE)),
    ("each_file", "each file", re.compile(r"\beach\s+files?\b", re.IGNORECASE)),
    ("all_files", "all files", re.compile(r"\ball\s+files?\b", re.IGNORECASE)),
    (
        "other_file",
        "other/another file",
        re.compile(r"\b(?:other|another)\s+files?\b", re.IGNORECASE),
    ),
    (
        "different_file",
        "different file",
        re.compile(r"\bdifferent\s+files?\b", re.IGNORECASE),
    ),
    (
        "across_files",
        "across files",
        re.compile(r"\bacross\s+(?:all\s+)?files?\b", re.IGNORECASE),
    ),
    (
        "project_scope",
        "project-wide",
        re.compile(
            r"\b(?:across\s+the\s+project|project[- ]wide|whole\s+project|"
            r"entire\s+project)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "repository_scope",
        "repository/repo-wide",
        re.compile(r"\b(?:repository|repo[- ]wide)\b", re.IGNORECASE),
    ),
    ("codebase_scope", "codebase", re.compile(r"\bcodebase\b", re.IGNORECASE)),
    (
        "other_modules",
        "other modules",
        re.compile(r"\bother\s+modules?\b", re.IGNORECASE),
    ),
    (
        "importers",
        "files that import",
        re.compile(r"\bfiles?\s+that\s+imports?\b", re.IGNORECASE),
    ),
    ("callers", "callers", re.compile(r"\bcallers?\b", re.IGNORECASE)),
    (
        "cross_file_usages",
        "usages across",
        re.compile(r"\busages?\s+across\b", re.IGNORECASE),
    ),
    (
        "all_references",
        "all/every references",
        re.compile(r"\b(?:all|every)\b[^\n]{0,40}\breferences?\b", re.IGNORECASE),
    ),
    (
        "all_imports_from",
        "all imports from",
        re.compile(r"\ball\s+imports?\s+from\b", re.IGNORECASE),
    ),
)

MAX_PROFILE_FEASIBILITY_NOTES = 16
_MAPPING_ERROR = "Internal feasibility mapping mismatch."


def build_feasibility_notes(
    resolved_profile: ResolvedProfile,
    planned_scopes: frozenset[FileScope],
) -> tuple[str, ...]:
    """Return bounded, non-blocking notes for customized cross-file requests."""
    units, _components = build_review_units(resolved_profile, planned_scopes)
    notes: list[str] = []
    seen: set[str] = set()

    for unit in units:
        parts = unit.component.split("/", 2)
        if len(parts) != 3 or any(not part for part in parts):
            raise RuntimeError(_MAPPING_ERROR)
        mode, agent, _scope = parts
        try:
            default = field_index(mode, agent)[unit.field_path]
        except KeyError:
            raise RuntimeError(_MAPPING_ERROR) from None
        if (unit.field_type, unit.instruction) == (
            default.type,
            default.instruction,
        ):
            continue

        for category, display_token, pattern in CROSS_FILE_SIGNALS:
            if not pattern.search(unit.instruction):
                continue
            note = (
                f"{unit.component} {unit.field_path}: instruction requests "
                f"cross-file scope [{category}: {display_token}]; a single-file "
                "pass can only see the file being documented."
            )
            if note not in seen:
                notes.append(note)
                seen.add(note)
            break
        if len(notes) >= MAX_PROFILE_FEASIBILITY_NOTES:
            break

    return tuple(notes)
