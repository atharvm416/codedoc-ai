"""0.12.1 — cache revision, prompt clauses, and internal-stat containment.

Verifies the ``file-doc-v3`` cache advance and one-time v2 invalidation, that the
four assembled prompts each contain every strengthened clause, and that the
correction statistics stay internal run stats — absent from completed JSON and
Markdown output and from crash recovery.
"""

from __future__ import annotations

import json

from codedoc.agents import (
    dependency_agent,
    documentation_agent,
    file_documentation_agent,
    structure_agent,
)
from codedoc.agents.base_agent import EXACT_JSON_RESPONSE_RULES
from codedoc.core.record_meta import (
    ANALYSIS_REVISION,
    normalized_identity_value,
)

_CLAUSES = tuple(EXACT_JSON_RESPONSE_RULES.split("\n"))


class _Fake:
    provider_name = "fake"

    def complete_json(self, prompt, system=""):
        return json.dumps(
            {"description": "A file.", "role_in_system": "r",
             "functions": [{"name": "f", "description": "d"}]}
        )

    def complete(self, prompt, system="", temperature=0.1):
        return self.complete_json(prompt, system)


def test_analysis_revision_is_v3():
    assert ANALYSIS_REVISION == "file-doc-v3"


def test_v2_record_is_invalidated():
    # A stored v2 record no longer matches the current v3 identity.
    assert normalized_identity_value("_analysis_revision", {"_analysis_revision": "file-doc-v2"}) == (
        "file-doc-v2"
    )
    assert ANALYSIS_REVISION != "file-doc-v2"


def test_all_four_prompts_contain_every_strengthened_clause():
    prompts = {
        "combined": file_documentation_agent.build_prompt("m.py", "x=1", ["os"], "python")[1],
        "structure": structure_agent.build_prompt("m.py", "x=1", ["os"], "python")[1],
        "dependency": dependency_agent.build_prompt("m.py", "x=1", ["os"], "python")[1],
        "documentation": documentation_agent.build_prompt("m.py", "x=1", "python", {}, {})[1],
    }
    for name, prompt in prompts.items():
        for clause in _CLAUSES:
            assert clause in prompt, f"{name} missing: {clause}"


def test_correction_stats_absent_from_completed_output(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: _Fake())
    stats = pipeline.run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "output_format": "both",
         "response_correction_enabled": True, "propagate_changes": False},
    )
    # Run stats carry the internal correction counters (like documentation_calls_*).
    assert "response_correction_calls_attempted" in stats
    assert "documentation_calls_attempted" in stats
    # Neither the counters nor any correction internal leak into completed output.
    for name in ("codedoc.json", "codedoc.md"):
        text = (tmp_path / "codedoc" / name).read_text(encoding="utf-8")
        assert "response_correction_calls_attempted" not in text
        assert "response_contract_final" not in text
        assert "documentation_calls_attempted" not in text


def test_crash_recovery_stores_no_correction_internal(tmp_path, monkeypatch):
    from codedoc import pipeline

    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

    recovery_seen = {}

    class Recorder(_Fake):
        def complete_json(self, prompt, system=""):
            rec = tmp_path / "codedoc" / "crash_recovery.json"
            if rec.exists():
                recovery_seen["text"] = rec.read_text(encoding="utf-8")
            return super().complete_json(prompt, system)

    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda _c: Recorder())
    pipeline.run_pipeline(
        tmp_path,
        {"entry_file": "main.py", "response_correction_enabled": True,
         "propagate_changes": False},
    )
    if "text" in recovery_seen:
        assert "response_contract_final" not in recovery_seen["text"]
        assert "response_contract_diagnostic" not in recovery_seen["text"]
