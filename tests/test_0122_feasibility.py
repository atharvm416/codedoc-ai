"""0.12.2 deterministic prompt-profile feasibility advisories."""

import copy
import json
import re
from types import SimpleNamespace

import pytest

from codedoc.cli.cli import (
    _print_feasibility_advisories,
    _print_prompt_profile_dry_run,
    _print_prompt_profile_run,
    run_cli,
)
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
from codedoc.pipeline import run_pipeline
from codedoc.utils.errors import PromptCustomizationValidationError

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


def _cross_file_profile() -> dict:
    raw = default_prompt_profiles("single")
    raw["single"]["common"]["requested_shape"]["description"] = (
        "Add a reference of a different file."
    )
    return raw


class _ReviewFake:
    provider_name = "fake"

    def __init__(self, verdict="SAFE", *, malformed=False):
        self.verdict = verdict
        self.malformed = malformed
        self.review_calls = 0
        self.doc_calls = 0

    def complete_json(self, prompt, system=""):
        if "standards/safety review" in prompt:
            self.review_calls += 1
            if self.malformed:
                return "{}"
            review_id = next(
                line.split(": ", 1)[1]
                for line in prompt.splitlines()
                if line.startswith("review_id: ")
            )
            ordinal, count = next(
                line.split(": ", 1)[1].split("/", 1)
                for line in prompt.splitlines()
                if line.startswith("batch: ")
            )
            return json.dumps(
                {
                    "review_id": review_id,
                    "batch_index": int(ordinal),
                    "batch_count": int(count),
                    "verdict": self.verdict,
                    "reasons": ["blocked"] if self.verdict == "TOO_RISKY" else [],
                    "warnings": ["confirm"] if self.verdict == "RISKY" else [],
                }
            )
        self.doc_calls += 1
        return json.dumps({"description": "Documented file."})

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


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


def test_cli_helper_and_profile_presenters_print_advisory(capsys):
    note = "single/combined/* description: bounded note"
    stats = {
        "prompt_profile_source": "inline",
        "prompt_customization_feasibility_advisories": (note,),
    }

    _print_feasibility_advisories(stats)
    _print_prompt_profile_dry_run(stats)
    _print_prompt_profile_run(stats)

    output = capsys.readouterr().out
    assert output.count("Feasibility advisory (non-blocking):") == 3
    assert output.count(f"- {note}") == 3


def test_cli_helper_ignores_missing_or_non_dict_stats(capsys):
    _print_feasibility_advisories(None)
    _print_feasibility_advisories({})
    assert capsys.readouterr().out == ""


def test_dry_and_real_runs_transport_the_same_advisory_without_persisting_it(
    tmp_path, monkeypatch
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    profile = _cross_file_profile()

    dry = run_pipeline(
        tmp_path,
        {
            "dry_run": True,
            "entry_file": "main.py",
            "prompt_profiles": profile,
        },
    )
    fake = _ReviewFake("SAFE")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    real = run_pipeline(
        tmp_path,
        {
            "entry_file": "main.py",
            "prompt_profiles": profile,
        },
    )

    notes = dry["prompt_customization_feasibility_advisories"]
    assert notes
    assert notes == real["prompt_customization_feasibility_advisories"]
    assert dry["prompt_customization_feasibility_notes"] == len(notes)
    assert real["prompt_customization_feasibility_notes"] == len(notes)
    assert fake.review_calls == 1
    assert fake.doc_calls == 1
    output = (tmp_path / "codedoc" / "codedoc.json").read_text(encoding="utf-8")
    assert "prompt_customization_feasibility" not in output


@pytest.mark.parametrize(
    ("verdict", "malformed", "confirm_risky", "expected_status"),
    [
        ("SAFE", True, None, "failed-closed"),
        ("RISKY", False, lambda _warnings: False, "risky-confirmation-blocked"),
        ("TOO_RISKY", False, None, "too-risky-blocked"),
    ],
)
def test_blocking_review_paths_carry_advisories(
    tmp_path,
    monkeypatch,
    verdict,
    malformed,
    confirm_risky,
    expected_status,
):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = _ReviewFake(verdict, malformed=malformed)
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)

    with pytest.raises(PromptCustomizationValidationError) as caught:
        run_pipeline(
            tmp_path,
            {
                "entry_file": "main.py",
                "prompt_profiles": _cross_file_profile(),
            },
            confirm_risky=confirm_risky,
        )

    assert (
        caught.value.stats["prompt_customization_security_review"]
        == expected_status
    )
    assert caught.value.stats["prompt_customization_feasibility_advisories"]
    assert fake.doc_calls == 0


def test_cli_prints_advisory_on_review_block(tmp_path, monkeypatch, capsys):
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    fake = _ReviewFake("TOO_RISKY")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _config: fake)
    config = {
        "prompt_profiles": _cross_file_profile(),
        "entry_file": "main.py",
    }
    (tmp_path / "codedoc.config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    assert run_cli([str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "Feasibility advisory (non-blocking):" in captured.err
    assert "[different_file: different file]" in captured.err
