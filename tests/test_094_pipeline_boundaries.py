"""0.9.4 — pipeline decomposition (Workstream A) boundary characterization.

These tests assert the structural contract of splitting ``codedoc.pipeline``
into ``resume`` / ``discovery`` / ``execution`` without changing behavior:

- the moved private helpers remain importable from ``codedoc.pipeline`` (a
  one-release compatibility shim used by repository tests and integrations);
- compatibility re-exports are the same object as the defining-module
  implementation at import time;
- ``resume`` and ``discovery`` never create providers or schedule agent work;
- ``discovery`` never writes output;
- there is no circular import among the decomposed modules;
- the ``ExecutionOptions`` / ``ExecutionContext`` boundary exists with the
  documented fields and carries no configuration dictionary.

Behavioral equivalence of execution, retries, the rate-limit ladder, resume,
and selection is covered by the full pre-existing suite (test_080/081/092/
pipeline/scenarios); this file guards the *seams* created in 0.9.4.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import pytest

import codedoc.pipeline as pipeline
from codedoc.core import discovery, execution, resume


# ---------------------------------------------------------------------------
# Compatibility re-exports from codedoc.pipeline
# ---------------------------------------------------------------------------

RESUME_REEXPORTS = [
    "_resolve_live_backup_path",
    "_load_existing_file_docs",
    "_load_existing_file_docs_from_md",
    "_public_record_to_doc",
    "_build_documentation_records",
    "_cleanup_stale_build_file",
    "_remove_legacy_db",
]

DISCOVERY_REEXPORTS = [
    "_resolve_entry_and_docs",
    "_build_graph",
    "_select_files",
    "_graph_edges",
]

EXECUTION_REEXPORTS = [
    "_is_rate_limit_error",
    "_detect_limit_type",
    "_build_default_ladder",
    "_parse_retry_after",
    "_process_and_record",
    "_process_descriptor_batch",
    "_process_files_sequentially",
    "_process_one_file",
    "_process_one_file_with_retries",
    "_agent_errors",
    "_safe_file_hash",
    "_cancel_pending",
    "_log_file_progress",
]


@pytest.mark.parametrize(
    "name,module",
    [(n, resume) for n in RESUME_REEXPORTS]
    + [(n, discovery) for n in DISCOVERY_REEXPORTS]
    + [(n, execution) for n in EXECUTION_REEXPORTS],
)
def test_pipeline_reexport_is_defining_implementation(name, module):
    """``codedoc.pipeline._name`` must be the exact object defined in the
    module it moved to, preserving direct-import compatibility."""
    assert getattr(pipeline, name) is getattr(module, name)


def test_moved_helper_monkeypatches_target_defining_module(monkeypatch):
    """Rebinding a compatibility alias cannot rebind the defining module.

    Repository tests that need to intercept moved helper calls must patch the
    defining module, which is the call site used by the extracted code.
    """
    def marker(*_args, **_kwargs):
        return {"patched": True}

    monkeypatch.setattr(execution, "_process_one_file", marker)
    assert execution._process_one_file is marker
    assert pipeline._process_one_file is not marker


def test_pipeline_keeps_provider_and_orchestrator_construction():
    # Provider/orchestrator creation stays in pipeline (heavily monkeypatched
    # as ``codedoc.pipeline.create_provider`` / ``.Orchestrator``).
    assert hasattr(pipeline, "create_provider")
    assert hasattr(pipeline, "Orchestrator")
    assert hasattr(pipeline, "SafeWriter")
    assert callable(pipeline.run_pipeline)


# ---------------------------------------------------------------------------
# Module responsibility boundaries (verified by source inspection)
# ---------------------------------------------------------------------------

def _source(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_resume_does_not_create_providers_or_schedule_agents():
    src = _source(resume)
    assert "create_provider" not in src
    assert "Orchestrator" not in src
    assert "ThreadPoolExecutor" not in src


def test_discovery_does_not_create_providers_or_write_output():
    src = _source(discovery)
    assert "create_provider" not in src
    assert "write_project_outputs" not in src
    assert "Orchestrator" not in src


def test_execution_does_not_receive_configuration_dictionary():
    """execution.py must not pull policy from a raw config dict; it consumes
    ExecutionOptions / ExecutionContext instead."""
    src = _source(execution)
    assert "load_config" not in src
    assert "get_rate_limit_profile" not in src  # profile is built by the pipeline
    assert 'config.get(' not in src


# ---------------------------------------------------------------------------
# No circular imports among the decomposed modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "modname",
    [
        "codedoc.core.resume",
        "codedoc.core.discovery",
        "codedoc.core.execution",
        "codedoc.core.project_view",
        "codedoc.core.markdown_view",
        "codedoc.pipeline",
    ],
)
def test_modules_import_cleanly(modname):
    assert importlib.import_module(modname) is not None


def test_resume_and_discovery_do_not_import_pipeline():
    for module in (resume, discovery, execution):
        assert "import codedoc.pipeline" not in _source(module)
        assert "from codedoc.pipeline" not in _source(module)


# ---------------------------------------------------------------------------
# Execution boundary contract
# ---------------------------------------------------------------------------

def test_execution_options_fields():
    fields = {f.name for f in dataclasses.fields(execution.ExecutionOptions)}
    assert fields == {
        "max_workers",
        "retry_attempts",
        "max_consecutive_failures",
        "rate_limit_adaptive",
        "parallel_ladder",
        "respect_retry_after",
        "retry_after_cap_s",
    }
    # frozen (immutable policy)
    opts = execution.ExecutionOptions(
        max_workers=2,
        retry_attempts=1,
        max_consecutive_failures=5,
        rate_limit_adaptive=True,
        parallel_ladder=None,
        respect_retry_after=True,
        retry_after_cap_s=30,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.max_workers = 3  # type: ignore[misc]


def test_execution_context_fields():
    fields = {f.name for f in dataclasses.fields(execution.ExecutionContext)}
    assert fields == {
        "orchestrator",
        "queue",
        "recorder",
        "error_reporter",
        "rate_limit_profile",
        "stats",
        "new_results",
        "options",
    }


def test_execute_agent_files_is_public_entry():
    assert callable(execution.execute_agent_files)
    # The pipeline wires execution through this facade.
    assert "execute_agent_files(context)" in _source(pipeline)
