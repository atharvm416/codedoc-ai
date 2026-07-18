"""Tests organized by feature ownership."""

import copy
import re
from types import SimpleNamespace
import pytest
from codedoc.core.feasibility import (
    CROSS_FILE_SIGNALS,
    MAX_PROFILE_FEASIBILITY_NOTES,
    build_feasibility_notes,
)
from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedProfile,
    default_prompt_profiles,
    validate_profile,
)
from tests.support.feasibility_cases import _cross_file_profile

_KNOWN = frozenset({".py", ".js"})

def _resolved(raw: dict, mode: str = "single") -> ResolvedProfile:
    return ResolvedProfile(
        mode,
        validate_profile(
            raw,
            active_mode=mode,
            known_extensions=_KNOWN,
            source="inline",
            source_path=None,
        ),
    )

def _scopes(*basenames: str) -> frozenset[FileScope]:
    return frozenset(FileScope(basename=name) for name in basenames)

def test_only_customized_fields_are_checked_and_first_signal_wins():
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["requested_shape"]["description"] = (
        "Describe every file and all files in the project."
    )
    notes = build_feasibility_notes(_resolved(raw), _scopes("main.py"))

    assert len(notes) == 1
    assert "single/combined/* description" in notes[0]
    assert "[every_file: every file]" in notes[0]
    # The unchanged default usage_example mentions "another file"; it must not
    # create a second note merely because its component is active.
    assert "usage_example" not in notes[0]

def test_note_is_bounded_and_does_not_echo_arbitrary_instruction_text():
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["requested_shape"]["description"] = (
        "Inspect the whole project and include PRIVATE-MARKER-741."
    )
    notes = build_feasibility_notes(_resolved(raw), _scopes("main.py"))

    assert notes == (
        "single/combined/* description: instruction requests cross-file scope "
        "[project_scope: project-wide]; a single-file pass can only see the file "
        "being documented.",
    )
    assert "PRIVATE-MARKER-741" not in notes[0]

def test_signal_vocabulary_is_exact_and_originating_phrase_matches():
    assert MAX_PROFILE_FEASIBILITY_NOTES == 16
    assert tuple(
        (category, display, pattern.pattern)
        for category, display, pattern in CROSS_FILE_SIGNALS
    ) == (
        ("every_file", "every file", r"\bevery\s+files?\b"),
        ("each_file", "each file", r"\beach\s+files?\b"),
        ("all_files", "all files", r"\ball\s+files?\b"),
        ("other_file", "other/another file", r"\b(?:other|another)\s+files?\b"),
        ("different_file", "different file", r"\bdifferent\s+files?\b"),
        ("across_files", "across files", r"\bacross\s+(?:all\s+)?files?\b"),
        (
            "project_scope",
            "project-wide",
            r"\b(?:across\s+the\s+project|project[- ]wide|whole\s+project|entire\s+project)\b",
        ),
        (
            "repository_scope",
            "repository/repo-wide",
            r"\b(?:repository|repo[- ]wide)\b",
        ),
        ("codebase_scope", "codebase", r"\bcodebase\b"),
        ("other_modules", "other modules", r"\bother\s+modules?\b"),
        ("importers", "files that import", r"\bfiles?\s+that\s+imports?\b"),
        ("callers", "callers", r"\bcallers?\b"),
        ("cross_file_usages", "usages across", r"\busages?\s+across\b"),
        (
            "all_references",
            "all/every references",
            r"\b(?:all|every)\b[^\n]{0,40}\breferences?\b",
        ),
        ("all_imports_from", "all imports from", r"\ball\s+imports?\s+from\b"),
    )
    assert all(pattern.flags & re.IGNORECASE for _, _, pattern in CROSS_FILE_SIGNALS)
    assert not any(
        pattern.search("callersuffix filecodebase")
        for _, _, pattern in CROSS_FILE_SIGNALS
    )
    notes = build_feasibility_notes(
        _resolved(_cross_file_profile()),
        _scopes("main.py"),
    )
    assert "[different_file: different file]" in notes[0]

def test_unreachable_extension_override_produces_no_note():
    raw = default_prompt_profiles("single")
    raw["single"]["per_extension"] = {
        ".js": {
            "requested_shape": {
                "description": "Describe every file in the repository."
            }
        }
    }
    assert build_feasibility_notes(_resolved(raw), _scopes("main.py")) == ()

def test_notes_are_capped_at_sixteen():
    raw = default_prompt_profiles("triple")

    def customize(value, path="field"):
        if isinstance(value, str):
            return f"Describe all files for {path}."
        if isinstance(value, list):
            item = value[0]
            if isinstance(item, str):
                return [f"Describe all files for {path}."]
            editable = "description" if "description" in item else "used_for"
            item[editable] = f"Describe all files for {path}.{editable}."
            return value
        for key, child in value.items():
            value[key] = customize(child, f"{path}.{key}")
        return value

    common = raw["triple"]["common"]
    for agent in ("structure", "dependency", "documentation"):
        shape = common[agent]["requested_shape"]
        common[agent]["requested_shape"] = customize(shape, agent)
    extension = copy.deepcopy(common)
    extension["structure"]["requested_shape"]["description"] += " Extension."
    raw["triple"]["per_extension"] = {".js": extension}

    notes = build_feasibility_notes(
        _resolved(raw, mode="triple"),
        _scopes("main.py", "main.js"),
    )
    assert len(notes) == MAX_PROFILE_FEASIBILITY_NOTES
    assert len(set(notes)) == len(notes)

@pytest.mark.parametrize(
    "unit",
    [
        SimpleNamespace(
            component="broken",
            field_path="description",
            field_type="string",
            instruction="all files",
        ),
        SimpleNamespace(
            component="single/combined/*",
            field_path="unknown",
            field_type="string",
            instruction="all files",
        ),
    ],
)
def test_internal_mapping_mismatch_is_bounded(monkeypatch, unit):
    monkeypatch.setattr(
        "codedoc.core.feasibility.build_review_units",
        lambda _resolved, _scopes: ([unit], {}),
    )
    with pytest.raises(RuntimeError) as caught:
        build_feasibility_notes(
            ResolvedProfile("single", None),
            frozenset(),
        )
    assert str(caught.value) == "Internal feasibility mapping mismatch."
    assert caught.value.__cause__ is None
