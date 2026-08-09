"""Executable smoke harness for an installed CodeDoc artifact.

This file is intentionally not named ``test_*.py``.  Release verification
launches it explicitly from each wheel/sdist environment; pytest must never
collect it from the source tree.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import tempfile
from typing import Callable, Iterable


_SENTINELS = (
    "source-sentinel-do-not-log-41f8",
    "prompt-sentinel-do-not-log-2a77",
    "request-body-sentinel-do-not-log-9bd1",
    "response-body-sentinel-do-not-log-6c03",
    "endpoint.invalid/private?token=endpoint-sentinel-783e",
    "Authorization: Bearer auth-sentinel-38ac",
    "sk-proj-api-key-sentinel-08c9",
)


class SmokeFailure(RuntimeError):
    """A bounded installed-artifact verification failure."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prove_installed_origin() -> tuple[Path, Path]:
    """Prove product imports and the console script come from this environment.

    This deliberately runs before changing directory or importing project
    content.  The harness itself may live in the checkout; the product under
    test may not.
    """
    import codedoc

    package_path = Path(codedoc.__file__).resolve()
    repository = _repository_root()
    site_roots = {
        Path(value).resolve()
        for value in (*site.getsitepackages(), site.getusersitepackages())
        if value
    }
    if _is_within(package_path, repository):
        raise SmokeFailure("installed-origin-check-failed")
    if not any(_is_within(package_path, root) for root in site_roots):
        raise SmokeFailure("site-packages-origin-check-failed")

    script_name = "codedoc.exe" if os.name == "nt" else "codedoc"
    located = shutil.which(script_name) or shutil.which("codedoc")
    if not located:
        raise SmokeFailure("console-script-not-found")
    console_path = Path(located).resolve()
    environment_bin = Path(sys.executable).resolve().parent
    if _is_within(console_path, repository):
        raise SmokeFailure("console-script-repository-origin")
    if not _is_within(console_path, environment_bin):
        raise SmokeFailure("console-script-environment-mismatch")
    return package_path, console_path


class _FrozenProvider:
    """Network-free deterministic response source for installed scenarios."""

    provider_name = "fake"

    def __init__(
        self,
        *,
        interrupt_after: int | None = None,
        call_count_path: Path | None = None,
        leaf_signature: str | None = None,
    ) -> None:
        self.calls = 0
        self.interrupt_after = interrupt_after
        self.call_count_path = call_count_path
        self.leaf_signature = leaf_signature

    def complete_json(self, prompt: str, system: str = "") -> str:
        import logging

        self.calls += 1
        logging.getLogger("httpcore.http11").debug(" | ".join(_SENTINELS))
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            # This attempt itself never completes or gets checkpointed, so it
            # must not be recorded in the call-count sidecar: that count is
            # read back as "calls that genuinely completed/checkpointed",
            # not "calls attempted", and callers reconcile it exactly against
            # an independent fresh-run baseline.
            raise KeyboardInterrupt
        if self.call_count_path is not None:
            self.call_count_path.write_text(str(self.calls), encoding="utf-8")
        if "standards/safety review" in prompt:
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
                    "verdict": "SAFE",
                    "reasons": [],
                    "warnings": [],
                }
            )
        if "Analyse the imports" in prompt:
            return json.dumps(
                {
                    "dependencies_analysis": {
                        "internal": [],
                        "external": [],
                        "dependency_refs": [],
                        "catalog_updates": [],
                        "usage_notes": [],
                        "warnings": [],
                    }
                }
            )
        if "This is one bounded fragment of a larger" in prompt:
            function: dict = {"name": "f", "description": "does f"}
            if self.leaf_signature is not None:
                function["signature"] = self.leaf_signature
            return json.dumps(
                {
                    "description": "A bounded fragment.",
                    "functions": [function],
                }
            )
        if "Refine one combined narrative from" in prompt:
            return json.dumps({"narrative": "A refined narrative."})
        return json.dumps(
            {
                "description": "A file.",
                "role_in_system": "core",
                "functions": [{"name": "f", "description": "does f"}],
                "key_concepts": ["installed smoke"],
                "usage_example": "import main",
            }
        )

    def complete(
        self, prompt: str, system: str = "", temperature: float = 0.1
    ) -> str:
        return self.complete_json(prompt, system)


def _factory_for(provider: _FrozenProvider) -> Callable[[dict], _FrozenProvider]:
    def factory(resolved_config: dict) -> _FrozenProvider:
        from codedoc.llm.factory import attest_provider_execution

        attest_provider_execution(provider, resolved_config)
        return provider

    return factory


def _run_in_process(
    project: Path,
    config: dict,
    provider: _FrozenProvider | None = None,
    *,
    forbid_provider: bool = False,
) -> tuple[dict, str]:
    import codedoc.pipeline as pipeline

    prior_factory = pipeline.create_provider
    if forbid_provider:
        def forbidden_factory(_config: dict):
            raise SmokeFailure("unexpected-provider-construction")

        pipeline.create_provider = forbidden_factory
    else:
        if provider is None:
            provider = _FrozenProvider()
        pipeline.create_provider = _factory_for(provider)

    output = io.StringIO()
    try:
        with redirect_stdout(output), redirect_stderr(output):
            stats = pipeline.run_pipeline(project, config)
    finally:
        pipeline.create_provider = prior_factory
    return stats, output.getvalue()


def _write_config(project: Path, **overrides: object) -> dict:
    config: dict = {
        "entry_file": "main.py",
        "documentation_scope": "entry",
        "output_dir": "docs",
        "output_format": "json",
        "analysis_mode": "single",
        "parallel_agents": False,
        "max_parallel_files": 1,
        "file_retry_attempts": 0,
        "propagate_changes": False,
    }
    config.update(overrides)
    project.joinpath("codedoc.config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return config


def _large_source() -> str:
    return "".join(
        f"def fn_{index}():\n    return {index}\n\n" for index in range(220)
    )


def _read_output(project: Path) -> dict:
    return json.loads(project.joinpath("docs", "codedoc.json").read_text(encoding="utf-8"))


def _artifact_text(project: Path, extra: Iterable[str] = ()) -> str:
    parts = list(extra)
    for name in ("codedoc.json", "codedoc.md", "crash_recovery.json"):
        for path in project.rglob(name):
            parts.append(path.read_text(encoding="utf-8", errors="backslashreplace"))
    return "\n".join(parts)


def _hash_optional(path: Path) -> str | None:
    """SHA-256 of *path*, or ``None`` if it does not exist -- so "the file is
    absent" and "the file hashes to some value" are never confused."""
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(project: Path) -> dict[str, str | None]:
    """Record digests of every artifact whose byte-identity section 18 cares
    about, so "byte-identical" is a recorded comparison, not an assertion."""
    docs = project / "docs"
    return {
        "crash_recovery.json": _hash_optional(docs / "crash_recovery.json"),
        "codedoc.json": _hash_optional(docs / "codedoc.json"),
    }


def _read_call_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(path.read_text(encoding="utf-8").strip())


def _assert_private(project: Path, *captured: str) -> None:
    combined = _artifact_text(project, captured)
    leaked = [sentinel for sentinel in _SENTINELS if sentinel in combined]
    if leaked:
        raise SmokeFailure("sentinel-leak-detected")


def _scenario_truncate(root: Path) -> None:
    project = root / "truncate"
    project.mkdir()
    project.joinpath("main.py").write_text("value = 1\n" * 400, encoding="utf-8")
    config = _write_config(
        project, large_file_strategy="truncate", max_content_chars=1000
    )
    stats, captured = _run_in_process(project, config)
    if stats["checked"] != 1 or not _read_output(project)["files"]:
        raise SmokeFailure("truncate-scenario-failed")
    _assert_private(project, captured)


def _scenario_fresh_split(root: Path) -> None:
    project = root / "fresh-split"
    project.mkdir()
    project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
    config = _write_config(
        project, large_file_strategy="split", max_content_chars=2000
    )
    stats, captured = _run_in_process(project, config)
    record = _read_output(project)["files"][0]
    if stats["checked"] != 1:
        raise SmokeFailure("fresh-split-not-executed")
    if not record.get("_large_file_identity", "").startswith("large-file-v3:"):
        raise SmokeFailure("fresh-split-identity-mismatch")
    if "_split_reuse_contract" in record:
        raise SmokeFailure("retired-split-contract-stamped")
    if project.joinpath("docs", "crash_recovery.json").exists():
        raise SmokeFailure("clean-split-left-recovery")
    _assert_private(project, captured)


def _scenario_signature_bound(root: Path) -> None:
    """Section 18: a bounded 552-character model leaf signature is accepted
    with response correction disabled, a synthetic parser-aligned
    600-character boundary succeeds, and 601 characters fails closed with no
    truncated public fact -- exercised through the installed artifact, not
    only at the source level."""
    for signature_chars, expect_failure in ((552, False), (600, False), (601, True)):
        project = root / f"signature-{signature_chars}"
        project.mkdir()
        project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
        config = _write_config(
            project,
            large_file_strategy="split",
            max_content_chars=2000,
            response_correction_enabled=False,
        )
        provider = _FrozenProvider(leaf_signature="s" * signature_chars)
        stats, captured = _run_in_process(project, config, provider)
        if expect_failure:
            if stats.get("failed", 0) != 1 or stats.get("checked", 0) != 0:
                raise SmokeFailure(f"signature-{signature_chars}-did-not-fail-closed")
            if project.joinpath("docs", "codedoc.json").exists():
                text = project.joinpath("docs", "codedoc.json").read_text(encoding="utf-8")
                if "signature" in text or ("s" * 100) in text:
                    raise SmokeFailure(f"signature-{signature_chars}-leaked-into-output")
        else:
            if stats.get("checked", 0) != 1 or stats.get("failed", 0) != 0:
                raise SmokeFailure(f"signature-{signature_chars}-was-unexpectedly-rejected")
            output_text = project.joinpath("docs", "codedoc.json").read_text(encoding="utf-8")
            if "signature" in output_text:
                raise SmokeFailure(f"signature-{signature_chars}-private-field-published")
        _assert_private(project, captured)


def _scenario_completed_reuse(root: Path) -> None:
    project = root / "completed-reuse"
    project.mkdir()
    project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
    config = _write_config(
        project, large_file_strategy="split", max_content_chars=2000
    )
    _run_in_process(project, config)
    stats, captured = _run_in_process(project, config, forbid_provider=True)
    if stats["checked"] != 0 or stats["split_completed_files_reused"] != 1:
        raise SmokeFailure("completed-zero-call-reuse-failed")
    _assert_private(project, captured)


def _scenario_interrupt_resume(root: Path) -> None:
    project = root / "interrupt-resume"
    project.mkdir()
    project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
    config = _write_config(
        project, large_file_strategy="split", max_content_chars=2000
    )
    fresh_stats, _captured = _run_in_process(
        project, {**config, "dry_run": True}, forbid_provider=True
    )
    interrupted = _FrozenProvider(interrupt_after=1)
    try:
        _run_in_process(project, config, interrupted)
    except KeyboardInterrupt:
        pass
    else:
        raise SmokeFailure("interruption-not-observed")
    recovery = project / "docs" / "crash_recovery.json"
    if not recovery.exists():
        raise SmokeFailure("interruption-did-not-preserve-recovery")

    resumed = _FrozenProvider()
    stats, captured = _run_in_process(project, config, resumed)
    if stats["split_partial_files_resumed"] != 1:
        raise SmokeFailure("partial-resume-not-counted")
    if resumed.calls >= fresh_stats["total_calls_planned"]:
        raise SmokeFailure("resume-did-not-exclude-restored-work")
    if recovery.exists():
        raise SmokeFailure("clean-resume-left-recovery")
    _assert_private(project, captured)


def _scenario_imports_only(root: Path) -> None:
    """Prove the installed planner schedules only final synthesis when the
    parser-derived imports tuple changes while frozen source bytes do not.

    Parser-derived imports normally follow source bytes, so changing an import
    statement is a source-hash change and correctly invalidates every node.
    This scenario isolates the imports identity exactly as the contract does:
    create live leaf/reducer checkpoints for unchanged source, add the
    corresponding old-import final checkpoint, then validate and plan against
    a different same-length imports tuple.
    """
    from codedoc.core.document import read_codedoc_document
    from codedoc.core.execution_model import build_call_manifest
    from codedoc.core.file_division import (
        SPLIT_PARTIAL_SCHEMA_VERSION,
        SplitTreeState,
        build_division_plan,
        build_fact_ledger,
        build_reduction_tree,
        deterministic_imports_digest,
        final_execution_identity,
        final_input_digest,
        final_synthesis_input,
        load_canonical_json_object,
        provider_execution_identity,
        refine_narrative_inputs,
        tree_node_state,
        validate_recovered_tree,
    )
    from codedoc.core.loader import load_config
    from codedoc.core.prompt_profiles import NO_PROMPT_PROFILE_DIGEST
    from codedoc.core.result_assembly import flat_combined_result

    project = root / "imports-only"
    project.mkdir()
    source = project / "main.py"
    source_text = _large_source()
    source.write_text(source_text, encoding="utf-8")
    config = _write_config(
        project, large_file_strategy="split", max_content_chars=2000
    )
    dry_stats, _captured = _run_in_process(
        project, {**config, "dry_run": True}, forbid_provider=True
    )
    interrupted = _FrozenProvider(
        interrupt_after=dry_stats["total_calls_planned"] - 1
    )
    try:
        _run_in_process(project, config, interrupted)
    except KeyboardInterrupt:
        pass
    else:
        raise SmokeFailure("imports-only-setup-did-not-interrupt-final")
    if not project.joinpath("docs", "crash_recovery.json").exists():
        raise SmokeFailure("imports-only-setup-lost-recovery")

    recovered = read_codedoc_document(
        project / "docs" / "crash_recovery.json",
        include_partial_files=True,
    ).partial_files[0]
    division = build_division_plan(
        rel_path="main.py",
        language="python",
        content=source_text,
        source_budget_chars=2000,
    )
    before_imports = ("alpha",)
    after_imports = ("bravo",)
    tree = build_reduction_tree(
        division,
        max_content_chars=2000,
        language="python",
        imports=before_imports,
    )
    results = {
        node.node_id: load_canonical_json_object(node.result_json)
        for node in recovered.nodes
    }
    final = tree.final_node
    leaf_capsules = [results[chunk.chunk_id] for chunk in division.chunks]
    ledger = build_fact_ledger(
        leaf_capsules,
        language="python",
        chunks=division.chunks,
        symbols=division.symbols,
    )
    root_narratives = tuple(
        results[child_id].get("narrative", results[child_id].get("description", ""))
        for child_id in final.child_ids
    )
    manifest_json = final_synthesis_input(
        rel_path="main.py",
        language="python",
        imports=before_imports,
        root_narratives=refine_narrative_inputs(root_narratives),
        root_coverage_leaf_ids=final.leaf_ids,
        ledger=ledger,
        max_chars=2000,
    )
    resolved_config = load_config(project, config)
    provider_identity = provider_execution_identity(resolved_config)
    before_imports_digest = deterministic_imports_digest(before_imports)
    final_state = tree_node_state(
        node_id=final.node_id,
        node_type="final",
        rel_path="main.py",
        content_hash=recovered.content_hash,
        division_plan_digest=division.plan_digest,
        execution_identity_digest=final_execution_identity(
            rel_path="main.py",
            content_hash=recovered.content_hash,
            division_plan_digest=division.plan_digest,
            reduction_tree_digest=tree.tree_digest,
            provider_identity=provider_identity,
            prompt_profile_digest=NO_PROMPT_PROFILE_DIGEST,
            imports_digest=before_imports_digest,
            node=final,
        ),
        input_digest=final_input_digest(
            imports_digest=before_imports_digest,
            resolved_shape_digest=NO_PROMPT_PROFILE_DIGEST,
            manifest_json=manifest_json,
        ),
        unit_id=None,
        child_ids=final.child_ids,
        coverage_leaf_ids=final.leaf_ids,
        result=flat_combined_result(
            "main.py",
            "python",
            list(before_imports),
            {"description": "A file."},
        ),
    )
    completed_state = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=recovered.content_hash,
        division_plan_digest=division.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=(*recovered.nodes, final_state),
    )
    retained, quarantine = validate_recovered_tree(
        completed_state.nodes,
        plan=division,
        tree=tree,
        content_hash=recovered.content_hash,
        provider_identity=provider_identity,
        prompt_profile_digest=NO_PROMPT_PROFILE_DIGEST,
        imports_digest=deterministic_imports_digest(after_imports),
        imports=after_imports,
        language="python",
        max_content_chars=2000,
    )
    retained_ids = {node.node_id for node in retained}
    expected_retained = {
        *(chunk.chunk_id for chunk in division.chunks),
        *(node.node_id for node in tree.all_intermediate_nodes),
    }
    retained_state = SplitTreeState(
        schema_version=SPLIT_PARTIAL_SCHEMA_VERSION,
        owner="codedoc-ai",
        rel_path="main.py",
        content_hash=recovered.content_hash,
        division_plan_digest=division.plan_digest,
        reduction_tree_digest=tree.tree_digest,
        nodes=retained,
        quarantine=quarantine,
    )
    unpaid = build_call_manifest(
        [],
        {"main.py"},
        "single",
        {"main.py": division},
        {"main.py": tree},
        {"main.py": retained_state},
    )
    if (
        retained_ids != expected_retained
        or tuple(entry.node_id for entry in quarantine) != (final.node_id,)
        or len(unpaid.calls) != 1
        or unpaid.calls[0].category != "file-synthesis"
        or unpaid.calls[0].owner != "main.py"
    ):
        raise SmokeFailure("imports-only-final-rerun-failed")


def _legacy_recovery(schema_version: int) -> str:
    return json.dumps(
        {
            "_crash_safety": "INCOMPLETE RUN - frozen installed smoke state",
            "_codedoc": {
                "status": "in_progress",
                "live_backup": True,
                "partial_files": {
                    "main.py": {
                        "schema_version": schema_version,
                        "owner": "codedoc-ai",
                        "rel_path": "main.py",
                        "nodes": {},
                    }
                },
            },
            "files": [],
        },
        indent=2,
    )


def _scenario_preserve_first(root: Path) -> None:
    from codedoc.utils.errors import ConfigError

    for schema_version in (1, 2, 99):
        project = root / f"preserve-{schema_version}"
        project.mkdir()
        project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
        config = _write_config(
            project, large_file_strategy="split", max_content_chars=2000
        )
        recovery = project / "docs" / "crash_recovery.json"
        recovery.parent.mkdir()
        before = _legacy_recovery(schema_version).encode("utf-8")
        recovery.write_bytes(before)
        try:
            _run_in_process(project, config, forbid_provider=True)
        except ConfigError:
            pass
        else:
            raise SmokeFailure("unsupported-recovery-did-not-block")
        if recovery.read_bytes() != before:
            raise SmokeFailure("unsupported-recovery-was-mutated")


def _child_command(
    project: Path, cli_args: list[str], python_exe: str = sys.executable
) -> list[str]:
    return [
        python_exe,
        str(Path(__file__).resolve()),
        "--child-run",
        "--project",
        str(project),
        "--",
        *cli_args,
    ]


def _run_child(
    project: Path,
    cli_args: list[str],
    *,
    python_exe: str = sys.executable,
    interrupt_after: int | None = None,
    call_count_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if interrupt_after is not None:
        env["CODEDOC_SMOKE_INTERRUPT_AFTER"] = str(interrupt_after)
    if call_count_path is not None:
        env["CODEDOC_SMOKE_CALL_COUNT_PATH"] = str(call_count_path)
    # Prepend *python_exe*'s own environment bin/Scripts directory so the
    # console-script lookup inside the child (`shutil.which`) resolves to
    # that environment's own `codedoc`, never an unrelated one earlier on
    # the inherited ambient PATH (e.g. a separate non-venv install).
    env["PATH"] = os.pathsep.join(
        (str(Path(python_exe).resolve().parent), env.get("PATH", ""))
    )
    return subprocess.run(
        _child_command(project, cli_args, python_exe),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
        env=env,
    )


def _scenario_redirected_verbose(root: Path) -> None:
    project = root / "redirected-verbose"
    project.mkdir()
    project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
    _write_config(project, large_file_strategy="split", max_content_chars=2000)
    cli_args = [
        ".",
        "--entry",
        "main.py",
        "--output",
        "docs",
        "--format",
        "json",
        "--large-file-strategy",
        "split",
        "--no-parallel",
        "--verbose",
    ]
    log_path = project / "verbose.log"
    if os.name == "nt":
        def quote(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        native = " ".join(quote(part) for part in _child_command(project, cli_args))
        command = (
            f"& {native} 2>&1 | Tee-Object -FilePath {quote(str(log_path))}; "
            "exit $LASTEXITCODE"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="backslashreplace",
            check=False,
        )
        captured = result.stdout + result.stderr + log_path.read_text(
            encoding="utf-8", errors="backslashreplace"
        )
    else:
        result = _run_child(project, cli_args)
        captured = result.stdout + result.stderr
        log_path.write_text(captured, encoding="utf-8")
    if result.returncode != 0:
        raise SmokeFailure("redirected-verbose-exit-status")
    if "Logging error" in captured or "--- Logging error ---" in captured:
        raise SmokeFailure("redirected-verbose-logging-error")
    if len(captured.encode("utf-8")) > 1_000_000:
        raise SmokeFailure("redirected-verbose-log-unbounded")
    _assert_private(project, captured)


def _scenario_exit_fidelity(root: Path, console_path: Path) -> None:
    project = root / "exit-fidelity"
    project.mkdir()
    controls = (
        (["--version"], 0),
        ([".", "--analysis-mode", "not-a-mode"], 2),
    )
    for args, expected in controls:
        direct = subprocess.run(
            [str(console_path), *args],
            cwd=project,
            text=True,
            capture_output=True,
            check=False,
        )
        child = _run_child(project, args)
        if direct.returncode != expected or child.returncode != direct.returncode:
            raise SmokeFailure("child-exit-status-mismatch")


_SPLIT_CLI_ARGS = [
    ".", "--entry", "main.py", "--output", "docs", "--format", "json",
    "--large-file-strategy", "split", "--no-parallel",
]
_ORDINARY_CLI_ARGS = [
    ".", "--entry", "main.py", "--output", "docs", "--format", "json",
    "--no-parallel",
]


def _run_cli(
    project: Path,
    cli_args: list[str],
    *,
    python_exe: str,
    interrupt_after: int | None = None,
) -> tuple["subprocess.CompletedProcess[str]", int]:
    """Run the installed CLI as a real subprocess under *python_exe* against
    *project*, returning the completed process and the fake-provider call
    count actually made (read back from a call-count sidecar file inside
    *project*, so the count survives even an interrupted process)."""
    call_count_path = project / ".codedoc-smoke-calls.json"
    if call_count_path.exists():
        call_count_path.unlink()
    result = _run_child(
        project,
        cli_args,
        python_exe=python_exe,
        interrupt_after=interrupt_after,
        call_count_path=call_count_path,
    )
    return result, _read_call_count(call_count_path)


def _scenario_cross_version(work: Path, peer_python: Path) -> None:
    """Section 18 predecessor-artifact compatibility matrix.

    *peer_python* is the exact released `0.14.2` environment's interpreter;
    the current process's own interpreter (`sys.executable`) is the `0.14.3`
    candidate. Two installed versions never run in one environment: every
    step below drives one or the other as a fresh subprocess against a
    shared project directory under *work*, exchanging state only through
    that directory -- neither environment ever imports the other.
    """

    def _new_split_project(name: str) -> Path:
        project = work / name
        project.mkdir(parents=True)
        project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
        _write_config(project, large_file_strategy="split", max_content_chars=2000)
        return project

    def _large_file_identity(project: Path) -> str:
        record = json.loads(
            (project / "docs" / "codedoc.json").read_text(encoding="utf-8")
        )["files"][0]
        identity = record.get("_large_file_identity")
        if not isinstance(identity, str) or not identity.startswith("large-file-v3:"):
            raise SmokeFailure("cross-version-missing-or-malformed-large-file-identity")
        return identity

    def _names_unsupported_schema_cause(result: "subprocess.CompletedProcess[str]") -> bool:
        """Whether *result* names the specific nested cause
        `document.py::_partial_files_from_meta` raises for an unsupported
        split-partial container schema -- not merely a generic recovery or
        configuration message (every generic recovery remedy already
        mentions "current schema-4", which would make a broad "schema" or
        "unsupported" substring check pass for an unrelated failure)."""
        combined = result.stdout + result.stderr
        return "unsupported split-partial schema version" in combined

    # Independent fresh-run baseline: the exact total call count a genuinely
    # uninterrupted run of this file needs, measured once up front so later
    # steps can prove resume did EXACTLY the remaining unpaid work -- not
    # merely "some" work -- by reconciling against it.
    baseline = _new_split_project("baseline")
    baseline_result, fresh_total_calls = _run_cli(
        baseline, _SPLIT_CLI_ARGS, python_exe=sys.executable
    )
    if baseline_result.returncode != 0 or fresh_total_calls <= 1:
        raise SmokeFailure("cross-version-baseline-run-failed-or-too-small")

    # 1. envA (released 0.14.2) produces a completed v5 split record.
    completed = _new_split_project("completed")
    result, calls = _run_cli(completed, _SPLIT_CLI_ARGS, python_exe=str(peer_python))
    if result.returncode != 0:
        raise SmokeFailure(
            f"cross-version-1-envA-completed-failed: {result.stderr[-2000:]}"
        )
    if calls <= 0:
        raise SmokeFailure("cross-version-1-envA-made-no-calls")
    if (completed / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("cross-version-1-envA-left-recovery")
    v5_identity = _large_file_identity(completed)

    # 2. envB (candidate 0.14.3, self) proves it stale and re-executes.
    result, calls = _run_cli(completed, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0:
        raise SmokeFailure(
            f"cross-version-2-envB-rerun-failed: {result.stderr[-2000:]}"
        )
    if calls != fresh_total_calls:
        raise SmokeFailure(
            "cross-version-2-v6-workload-call-count-mismatch: "
            f"{calls} != baseline {fresh_total_calls}"
        )
    v6_identity = _large_file_identity(completed)
    if v6_identity == v5_identity:
        raise SmokeFailure("cross-version-2-identity-did-not-actually-change")
    if (completed / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("cross-version-2-envB-did-not-cleanly-finalize")

    # 3. envA (released 0.14.2) leaves schema-3 recovery. The child must be
    # genuinely interrupted (exit 130), not merely nonzero for some other
    # reason, and exactly one call must have succeeded before that.
    schema3 = _new_split_project("schema3")
    interrupted_result, first_calls = _run_cli(
        schema3, _SPLIT_CLI_ARGS, python_exe=str(peer_python), interrupt_after=1
    )
    if interrupted_result.returncode != 130:
        raise SmokeFailure(
            f"cross-version-3-envA-did-not-cleanly-interrupt: "
            f"exit={interrupted_result.returncode}"
        )
    if first_calls != 1:
        raise SmokeFailure("cross-version-3-envA-unexpected-call-count")
    if not (schema3 / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("cross-version-3-envA-left-no-recovery")
    step3_snapshot = _snapshot(schema3)

    # 4. envB (candidate 0.14.3) blocks byte-identically. Recorded hashes
    # must equal step 3's post-state exactly; expected fake-call count zero;
    # the block must be the named schema/unsupported-generation rejection
    # (exit 2, the ConfigError class), not an unrelated crash.
    result, calls = _run_cli(schema3, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 2:
        raise SmokeFailure(
            f"cross-version-4-envB-did-not-block-as-config-error: "
            f"exit={result.returncode}"
        )
    if not _names_unsupported_schema_cause(result):
        raise SmokeFailure("cross-version-4-block-message-did-not-name-the-cause")
    if calls != 0:
        raise SmokeFailure("cross-version-4-envB-made-an-unexpected-call")
    if _snapshot(schema3) != step3_snapshot:
        raise SmokeFailure("cross-version-4-envB-mutated-schema3-artifact")

    # 4b. Repeat with a released schema-3 container holding more than
    # `MAX_QUARANTINE_ENTRIES_PER_FILE` (32) nodes: the early per-container
    # schema rejection must be reached identically, never the bounded
    # quarantine path -- so a >32-node predecessor container must not raise
    # a quarantine-bound failure instead of the plain schema rejection.
    #
    # The envelope (`recovery_identity` and every container-level field) is
    # genuinely envA-produced -- a real interrupted envA run against its own
    # project, exactly like step 3 -- so this is not distinguishable from a
    # real recovery file by anything other than node count. Only the `nodes`
    # array is then inflated with synthetic siblings to exceed 32; producing
    # 37 real interrupted leaf calls would work identically but far more
    # slowly for no added proof value, since the schema check that must
    # reject this container runs before any node (real or synthetic) is
    # deserialized at all.
    schema3_many = _new_split_project("schema3-many")
    many_interrupted, many_first_calls = _run_cli(
        schema3_many, _SPLIT_CLI_ARGS, python_exe=str(peer_python), interrupt_after=1
    )
    if many_interrupted.returncode != 130:
        raise SmokeFailure(
            f"cross-version-4b-envA-did-not-cleanly-interrupt: "
            f"exit={many_interrupted.returncode}"
        )
    if many_first_calls != 1:
        raise SmokeFailure("cross-version-4b-envA-unexpected-call-count")
    many_recovery_path = schema3_many / "docs" / "crash_recovery.json"
    many_payload = json.loads(many_recovery_path.read_text(encoding="utf-8"))
    many_container = many_payload["_codedoc"]["partial_files"]["main.py"]
    if many_container["schema_version"] != 3:
        raise SmokeFailure("cross-version-4b-envA-did-not-write-schema-3")
    real_node = many_container["nodes"][0]
    while len(many_container["nodes"]) <= 32:
        synthetic = dict(real_node)
        index = len(many_container["nodes"])
        synthetic["node_id"] = f"chunk_{index:064d}"
        synthetic["coverage_leaf_ids"] = [synthetic["node_id"]]
        many_container["nodes"].append(synthetic)
    if len(many_container["nodes"]) <= 32:
        raise SmokeFailure("cross-version-4b-setup-did-not-exceed-quarantine-bound")
    many_recovery_path.write_text(json.dumps(many_payload, indent=2), encoding="utf-8")
    many_before = many_recovery_path.read_bytes()

    result, calls = _run_cli(schema3_many, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 2:
        raise SmokeFailure(
            f"cross-version-4b-envB-did-not-block-many-node-schema3: "
            f"exit={result.returncode}"
        )
    if not _names_unsupported_schema_cause(result):
        raise SmokeFailure("cross-version-4b-block-message-did-not-name-the-cause")
    if "bounded entry count" in (result.stdout + result.stderr).lower():
        raise SmokeFailure("cross-version-4b-reached-quarantine-bound-instead-of-schema-check")
    if calls != 0:
        raise SmokeFailure("cross-version-4b-envB-made-an-unexpected-call")
    if many_recovery_path.read_bytes() != many_before:
        raise SmokeFailure("cross-version-4b-envB-mutated-many-node-schema3-artifact")

    # 5. Deliberate move-aside, then a fresh envB run must complete normally.
    recovery3 = schema3 / "docs" / "crash_recovery.json"
    recovery3.replace(schema3 / "docs" / "crash_recovery.json.moved-aside")
    result, calls = _run_cli(schema3, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0:
        raise SmokeFailure(
            f"cross-version-5-envB-fresh-run-failed: {result.stderr[-2000:]}"
        )
    if calls != fresh_total_calls:
        raise SmokeFailure("cross-version-5-envB-fresh-run-call-count-mismatch")
    if recovery3.exists():
        raise SmokeFailure("cross-version-5-envB-left-recovery-after-completion")

    # 6. envB (candidate 0.14.3) leaves schema-4 recovery. Do NOT copy or
    # relocate: recovery identity binds project_root among other fields, so
    # an artifact examined outside its own project would no longer be the
    # scenario a real downgrading user meets.
    schema4 = _new_split_project("schema4")
    interrupted_result, first_calls = _run_cli(
        schema4, _SPLIT_CLI_ARGS, python_exe=sys.executable, interrupt_after=1
    )
    if interrupted_result.returncode != 130:
        raise SmokeFailure(
            f"cross-version-6-envB-did-not-cleanly-interrupt: "
            f"exit={interrupted_result.returncode}"
        )
    if first_calls != 1:
        raise SmokeFailure("cross-version-6-envB-unexpected-call-count")
    if not (schema4 / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("cross-version-6-envB-left-no-recovery")
    step6_snapshot = _snapshot(schema4)

    # 7. envA (released 0.14.2) blocks on that ORIGINAL recovery, in place --
    # no copy is required precisely because the downgrade attempt must
    # neither mutate nor delete the artifact. envA does not know schema 4,
    # so it must reject it as unsupported/malformed input (exit 2), not
    # crash for an unrelated reason.
    result, calls = _run_cli(schema4, _SPLIT_CLI_ARGS, python_exe=str(peer_python))
    if result.returncode != 2:
        raise SmokeFailure(
            f"cross-version-7-envA-did-not-block-as-config-error: "
            f"exit={result.returncode}"
        )
    if not _names_unsupported_schema_cause(result):
        raise SmokeFailure("cross-version-7-block-message-did-not-name-the-cause")
    if calls != 0:
        raise SmokeFailure("cross-version-7-envA-made-an-unexpected-call")
    if _snapshot(schema4) != step6_snapshot:
        raise SmokeFailure("cross-version-7-envA-mutated-schema4-artifact")

    # 8. envB resumes that same unchanged recovery and completes -- doing
    # EXACTLY the remaining unpaid work, reconciled against the independent
    # fresh-run baseline above, so silently discarding the checkpoint and
    # re-executing from scratch (which would also exit 0 with calls > 0)
    # cannot pass this step.
    result, calls = _run_cli(schema4, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0:
        raise SmokeFailure(
            f"cross-version-8-envB-resume-failed: {result.stderr[-2000:]}"
        )
    if first_calls + calls != fresh_total_calls:
        raise SmokeFailure(
            "cross-version-8-resume-did-not-reconcile-with-fresh-baseline: "
            f"{first_calls} + {calls} != {fresh_total_calls}"
        )
    if (schema4 / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("cross-version-8-envB-left-recovery-after-completion")

    # 9. Ordinary compatibility, both directions -- zero-call reuse and
    # byte-identical PUBLIC DOCUMENTATION. `last_run` is deliberately excluded
    # from the identity check: it truthfully reports what happened in the
    # most recent invocation (documented vs. reused counts), so it always
    # legitimately differs between the creator's run and the reader's
    # zero-call reuse even though the rest of the document is untouched.
    # Everything else -- `files`, `tree`, `folders`, and any other top-level
    # field -- must be byte-for-byte identical.
    def _public_document_excluding_last_run(project: Path) -> dict:
        doc = json.loads((project / "docs" / "codedoc.json").read_text(encoding="utf-8"))
        doc.pop("last_run", None)
        return doc

    def _ordinary_round_trip(name: str, creator: str, reader: str) -> None:
        project = work / name
        project.mkdir()
        project.joinpath("main.py").write_text("def helper(): pass\n", encoding="utf-8")
        _write_config(project, large_file_strategy="truncate")
        result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=creator)
        if result.returncode != 0 or calls <= 0:
            raise SmokeFailure(f"cross-version-9-{name}-setup-failed")
        before = _public_document_excluding_last_run(project)
        result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=reader)
        if result.returncode != 0:
            raise SmokeFailure(f"cross-version-9-{name}-read-failed")
        if calls != 0:
            raise SmokeFailure(f"cross-version-9-{name}-did-not-reuse-zero-call")
        after = _public_document_excluding_last_run(project)
        if after != before:
            raise SmokeFailure(f"cross-version-9-{name}-documentation-changed-on-reuse")

    _ordinary_round_trip("ordinaryA", creator=str(peer_python), reader=sys.executable)
    _ordinary_round_trip("ordinaryB", creator=sys.executable, reader=str(peer_python))

    print("cross-version installed artifact matrix: ok")


def _child_run(project: Path, cli_args: list[str]) -> int:
    _prove_installed_origin()
    os.chdir(project)
    import codedoc.pipeline as pipeline

    interrupt_after_raw = os.environ.get("CODEDOC_SMOKE_INTERRUPT_AFTER")
    call_count_path_raw = os.environ.get("CODEDOC_SMOKE_CALL_COUNT_PATH")
    provider = _FrozenProvider(
        interrupt_after=(
            int(interrupt_after_raw) if interrupt_after_raw is not None else None
        ),
        call_count_path=(
            Path(call_count_path_raw) if call_count_path_raw is not None else None
        ),
    )
    pipeline.create_provider = _factory_for(provider)
    from codedoc.cli.cli import main as cli_main

    cli_main(cli_args)
    return 0


def _run_all() -> int:
    _package_path, console_path = _prove_installed_origin()
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="codedoc-installed-smoke-") as temp_name:
        neutral_root = Path(temp_name).resolve()
        if _is_within(neutral_root, _repository_root()):
            raise SmokeFailure("neutral-project-inside-repository")
        try:
            os.chdir(neutral_root)
            _scenario_truncate(neutral_root)
            _scenario_fresh_split(neutral_root)
            _scenario_signature_bound(neutral_root)
            _scenario_redirected_verbose(neutral_root)
            _scenario_completed_reuse(neutral_root)
            _scenario_interrupt_resume(neutral_root)
            _scenario_imports_only(neutral_root)
            _scenario_preserve_first(neutral_root)
            _scenario_exit_fidelity(neutral_root, console_path)
        finally:
            os.chdir(original_cwd)
    print("installed artifact smoke: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=["all", "cross-version"], default="all"
    )
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument(
        "--peer-python",
        type=Path,
        help="Interpreter of the other installed version, for --scenario cross-version.",
    )
    parser.add_argument(
        "--work",
        type=Path,
        help="Throwaway directory (outside the repository) for --scenario cross-version.",
    )
    args, remainder = parser.parse_known_args(argv)
    if args.child_run:
        if args.project is None:
            raise SmokeFailure("child-project-required")
        if remainder[:1] == ["--"]:
            remainder = remainder[1:]
        return _child_run(args.project.resolve(), remainder)
    if remainder:
        raise SmokeFailure("unexpected-harness-arguments")
    if args.scenario == "cross-version":
        if args.peer_python is None or args.work is None:
            raise SmokeFailure("cross-version-requires-peer-python-and-work")
        work = args.work.resolve()
        if _is_within(work, _repository_root()):
            raise SmokeFailure("cross-version-work-inside-repository")
        _prove_installed_origin()
        _scenario_cross_version(work, args.peer_python.resolve())
        return 0
    return _run_all()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"installed artifact smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
