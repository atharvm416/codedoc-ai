"""Tests organized by feature ownership."""

from __future__ import annotations

from tests.support.pipeline_identity import _PRIOR_RUN_IDENTITY
from tests.support.logging_runs import patch_provider
from tests.support.logging_runs import write_py

def test_python_api_accepts_config_as_first_argument(tmp_path, monkeypatch):
    from codedoc.core.db import compute_file_hash
    from codedoc.core.output import write_project_outputs
    from codedoc.pipeline import run_pipeline

    main = tmp_path / "main.py"
    main.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    docs_output = tmp_path / "docs_output"
    docs_output.mkdir()

    file_hash = compute_file_hash(main)

    write_project_outputs(
        [{
            "file_path": "main.py",
            "hash": file_hash,
            "language": "python",
            "documentation": {"description": "Current directory API."},
            **_PRIOR_RUN_IDENTITY,
        }],
        {"checked": 1, "failed": 0, "skipped": 0},
        docs_output,
        output_format="md",
        entry_file="main.py",
    )

    def fail_if_llm_is_created(config):
        raise AssertionError("LLM should not be created for cached API output")

    monkeypatch.setattr("codedoc.pipeline.create_provider", fail_if_llm_is_created)
    monkeypatch.chdir(tmp_path)

    stats = run_pipeline(
        {
            "output_dir": "docs_output",
            "output_format": "md",
            "entry_file": "main.py",
        }
    )

    assert stats["checked"] == 0
    assert (tmp_path / "docs_output" / "codedoc.md").exists()
    assert "Current directory API." in (
        tmp_path / "docs_output" / "codedoc.md"
    ).read_text(encoding="utf-8")

def test_A6_walker_state_independent_when_interleaved(tmp_path):
    """A6: two _Walker generators driven concurrently (interleaved) keep fully
    independent state — proving re-entrancy, which the old function-attribute
    implementation did not provide."""
    from itertools import zip_longest
    from codedoc.core.scanner import _Walker

    r1 = tmp_path / "r1"
    (r1 / "node_modules").mkdir(parents=True)
    (r1 / "node_modules" / "lib.py").write_text("a=1\n")
    (r1 / "keep1.py").write_text("a=1\n")

    r2 = tmp_path / "r2"
    r2.mkdir()
    (r2 / "keep2.py").write_text("b=1\n")

    w1 = _Walker(scan_root=r1, skip_dirs={"node_modules"}, ignore_prefixes=set())
    w2 = _Walker(scan_root=r2, skip_dirs=set(), ignore_prefixes=set())

    seen1, seen2 = [], []
    for a, b in zip_longest(w1.walk(r1), w2.walk(r2)):
        if a is not None:
            seen1.append(a.name)
        if b is not None:
            seen2.append(b.name)

    assert seen1 == ["keep1.py"]
    assert seen2 == ["keep2.py"]
    assert w1.skipped_dirs == 1   # node_modules
    assert w2.skipped_dirs == 0

class TestConfigurableContentTruncation:
    """G6: max_content_chars is configurable and truncation is logged at INFO."""
    def test_pipeline_wires_max_content_chars(self, tmp_path, monkeypatch):
        """max_content_chars from config reaches the agents via the pipeline."""
        received = []

        def capturing_truncate(self, content, file_path=""):
            received.append(self._max_content_chars)
            return content

        from codedoc.agents import base_agent
        monkeypatch.setattr(base_agent.BaseAgent, "_truncate", capturing_truncate)
        patch_provider(monkeypatch)

        src = tmp_path / "src.py"
        write_py(src, "x = 1\n")

        from codedoc.pipeline import run_pipeline
        run_pipeline(
            tmp_path,
            {"entry_file": "src.py", "output_dir": "out", "max_content_chars": 25000},
        )

        assert received, "No _truncate calls recorded"
        assert all(v == 25000 for v in received), f"Expected 25000 but got: {set(received)}"
