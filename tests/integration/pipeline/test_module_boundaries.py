"""Tests organized by feature ownership."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
import pytest
import codedoc.pipeline as pipeline
from codedoc.core import discovery, execution, resume
from codedoc.core import markdown_view, project_view

RESUME_REEXPORTS = [
    "_load_existing_file_docs",
    "_load_existing_file_docs_from_md",
    "_public_record_to_doc",
    "_build_documentation_records",
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

@pytest.mark.parametrize(
    "modname",
    ["codedoc.bootstrap", "codedoc.llm.local_provider"],
)
def test_dormant_runtime_modules_are_removed(modname):
    """The self-install check and the reserved local provider are both gone."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(modname)

def test_pipeline_no_longer_imports_bootstrap():
    src = _source(pipeline)
    assert "bootstrap" not in src
    assert "ensure_codedoc_installed" not in src

def test_resume_and_discovery_do_not_import_pipeline():
    for module in (resume, discovery, execution):
        assert "import codedoc.pipeline" not in _source(module)
        assert "from codedoc.pipeline" not in _source(module)

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
        "execution_requests",
        "division_plans",
        "reduction_trees",
        "provider_identity",
        "split_execution_mode",
    }

def test_execute_agent_files_is_public_entry():
    assert callable(execution.execute_agent_files)
    # The pipeline wires execution through this facade.
    assert "execute_agent_files(context)" in _source(pipeline)

MOVED_NAMES = [
    "markdown_from_view",
    "markdown_to_view",
    "json_from_markdown",
    "markdown_from_json",
    "read_embedded_view",
    "read_embedded_view_result",
    "EmbeddedViewResult",
    "_build_full_view_comment",
    "_public_view_for_embedding",
    "_build_meta_comment",
]

@pytest.mark.parametrize("name", MOVED_NAMES)
def test_moved_names_importable_from_both_modules(name):
    from_mv = getattr(markdown_view, name)
    from_pv = getattr(project_view, name)  # via project_view.__getattr__ shim
    assert from_mv is from_pv

def test_project_view_retains_assembly_api():
    for name in ("build_project_view", "json_from_view", "clean_file_record", "read_codedoc_meta"):
        assert hasattr(project_view, name)

def test_project_view_getattr_rejects_unknown():
    with pytest.raises(AttributeError):
        project_view.this_name_does_not_exist  # noqa: B018

def test_no_circular_import_between_serializer_modules():
    import importlib

    # Importing either first must succeed (markdown_view -> project_view is
    # one-way; project_view -> markdown_view is lazy via __getattr__).
    for first, second in (
        ("codedoc.core.project_view", "codedoc.core.markdown_view"),
        ("codedoc.core.markdown_view", "codedoc.core.project_view"),
    ):
        importlib.import_module(first)
        importlib.import_module(second)
    import codedoc.core.project_view as pv

    src = Path(pv.__file__).read_text(encoding="utf-8")
    # The only reference to markdown_view in project_view is the lazy __getattr__.
    assert src.count("import markdown_view") == 1

def test_project_view_shim_exposes_only_scheduled_names():
    """Unrelated markdown-view internals must not leak through the shim."""
    assert hasattr(markdown_view, "_append_file_markdown")
    with pytest.raises(AttributeError):
        project_view._append_file_markdown

def test_provider_adapters_do_not_own_pipeline_policy():
    from codedoc.llm import api_provider

    source = Path(api_provider.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "codedoc.core.discovery",
        "codedoc.core.planning",
        "codedoc.core.safe_writer",
        "codedoc.core.output",
        "codedoc.pipeline",
    )
    assert not any(name in source for name in forbidden_imports)
