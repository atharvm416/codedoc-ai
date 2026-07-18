"""Shared test support extracted from mapped source modules."""

import pytest
import codedoc.pipeline as pipe

def _output(project):
    return project / "codedoc" / "codedoc.json"

def _run(monkeypatch, project, cfg, fake):
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda c: fake)
    return pipe.run_pipeline(project, cfg)

@pytest.fixture
def project(tmp_path):
    (tmp_path / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path
