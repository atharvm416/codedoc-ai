"""Executable smoke harness for an installed CodeDoc artifact.

This file is intentionally not named ``test_*.py``.  Release verification
launches it explicitly from each wheel/sdist environment; pytest must never
collect it from the source tree.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from importlib import metadata as importlib_metadata
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


def _prove_installed_origin(expected_version: str | None = None) -> tuple[Path, Path]:
    """Prove product imports and the console script come from this environment.

    This deliberately runs before changing directory or importing project
    content.  The harness itself may live in the checkout; the product under
    test may not.
    """
    import codedoc

    package_path = Path(codedoc.__file__).resolve()
    module_version = codedoc.__version__
    metadata_version = importlib_metadata.version("codedoc-ai")
    if not isinstance(module_version, str) or not module_version:
        raise SmokeFailure("candidate-module-version-missing")
    if metadata_version != module_version:
        raise SmokeFailure(
            "candidate-module-metadata-version-mismatch: "
            f"{module_version!r} != {metadata_version!r}"
        )
    if expected_version is not None and module_version != expected_version:
        raise SmokeFailure(
            f"candidate-version-mismatch: installed {module_version!r} "
            f"!= expected --candidate-version {expected_version!r}"
        )
    repository = _repository_root()
    site_roots = {
        Path(value).resolve()
        for value in site.getsitepackages()
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
    version_result = subprocess.run(
        [str(console_path), "--version"],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
    )
    expected_output = f"codedoc {module_version}"
    if version_result.returncode != 0 or version_result.stdout.strip() != expected_output:
        raise SmokeFailure(
            "candidate-console-version-mismatch: "
            f"exit={version_result.returncode} output={version_result.stdout.strip()!r} "
            f"expected={expected_output!r}"
        )
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
        # Feature-detected, not imported directly: this child process may be
        # running under an older peer's own installed codedoc (section
        # 12.1 R3, cross-version matrices), and attest_provider_execution is
        # itself a newer addition. A peer that predates it also predates the
        # execution-attestation verification it exists to satisfy, so
        # skipping the call is correct for that peer, not a workaround.
        from codedoc.llm import factory as factory_module

        attest = getattr(factory_module, "attest_provider_execution", None)
        if attest is not None:
            attest(provider, resolved_config)
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
    """Digest the complete project tree except the per-child call sidecar."""
    return {
        path.relative_to(project).as_posix(): _hash_optional(path)
        for path in sorted(project.rglob("*"))
        if path.is_file() and path.name != ".codedoc-smoke-calls.json"
    }


def _read_call_count(path: Path) -> int:
    if not path.exists():
        raise SmokeFailure("provider-call-sidecar-missing")
    raw = path.read_text(encoding="utf-8")
    try:
        count = int(raw)
    except ValueError:
        raise SmokeFailure("provider-call-sidecar-malformed") from None
    if count < 0 or raw != str(count):
        raise SmokeFailure("provider-call-sidecar-noncanonical")
    return count


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
    # R4: absence is not zero.  Initialize a canonical zero before every
    # child and require the sidecar to remain present and parseable afterward.
    call_count_path.write_text("0", encoding="utf-8")
    result = _run_child(
        project,
        cli_args,
        python_exe=python_exe,
        interrupt_after=interrupt_after,
        call_count_path=call_count_path,
    )
    return result, _read_call_count(call_count_path)


_PEER_VERSION_MATRIX = {
    "0.13.1": "a",
    "0.14.4": "b",
}

_PEER_PROBE_SCRIPT = (
    "import json, os, shutil, site, sys, codedoc; "
    "from importlib.metadata import version as metadata_version; "
    "script_name = 'codedoc.exe' if os.name == 'nt' else 'codedoc'; "
    "located = shutil.which(script_name) or shutil.which('codedoc'); "
    "print(json.dumps({"
    "'version': codedoc.__version__, "
    "'metadata_version': metadata_version('codedoc-ai'), "
    "'file': codedoc.__file__, "
    "'site_roots': [v for v in site.getsitepackages() if v], "
    "'console_path': located, "
    "'executable': sys.executable"
    "}))"
)


def _prove_peer_installed_origin(peer_python: Path, expected_version: str) -> Path:
    """Prove the *peer* environment's own installed codedoc module, import
    origin, and console script all resolve outside the repository, into
    that environment's own site-packages/bin, and matches *expected_version*
    -- run before any state changes, exactly like `_prove_installed_origin`
    proves the candidate (section 12.1 R4: module, metadata, import, and
    console origin, all verified before any state creation). This runs as a
    fresh subprocess under *peer_python* so the check reflects that
    environment, never whatever the current process already has imported."""
    result = subprocess.run(
        [str(peer_python), "-c", _PEER_PROBE_SCRIPT],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
    )
    if result.returncode != 0:
        raise SmokeFailure(f"peer-probe-failed: {result.stderr[-2000:]}")
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        raise SmokeFailure("peer-probe-malformed-output") from None

    package_path = Path(payload["file"]).resolve()
    if _is_within(package_path, _repository_root()):
        raise SmokeFailure("peer-installed-origin-repository")
    site_roots = {Path(value).resolve() for value in payload["site_roots"]}
    if not any(_is_within(package_path, root) for root in site_roots):
        raise SmokeFailure("peer-site-packages-origin-check-failed")
    if payload["version"] != expected_version:
        raise SmokeFailure(
            f"peer-version-mismatch: installed {payload['version']!r} "
            f"!= expected --peer-version {expected_version!r}"
        )
    if payload.get("metadata_version") != expected_version:
        raise SmokeFailure(
            f"peer-metadata-version-mismatch: installed "
            f"{payload.get('metadata_version')!r} != expected --peer-version "
            f"{expected_version!r}"
        )

    located_console = payload.get("console_path")
    if not located_console:
        raise SmokeFailure("peer-console-script-not-found")
    console_path = Path(located_console).resolve()
    if _is_within(console_path, _repository_root()):
        raise SmokeFailure("peer-console-script-repository-origin")
    peer_environment_bin = Path(payload["executable"]).resolve().parent
    if not _is_within(console_path, peer_environment_bin):
        raise SmokeFailure("peer-console-script-environment-mismatch")
    version_result = subprocess.run(
        [str(console_path), "--version"],
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
    )
    expected_output = f"codedoc {expected_version}"
    if version_result.returncode != 0 or version_result.stdout.strip() != expected_output:
        raise SmokeFailure(
            "peer-console-version-mismatch: "
            f"exit={version_result.returncode} output={version_result.stdout.strip()!r} "
            f"expected={expected_output!r}"
        )
    return package_path


def _new_split_project(work: Path, name: str) -> Path:
    project = work / name
    project.mkdir(parents=True)
    project.joinpath("main.py").write_text(_large_source(), encoding="utf-8")
    _write_config(project, large_file_strategy="split", max_content_chars=2000)
    return project


def _new_ordinary_project(work: Path, name: str, *, file_count: int = 3) -> Path:
    """A small ordinary (non-split) multi-file project: enough files that an
    interrupted run leaves genuine, non-trivial partial progress."""
    project = work / name
    project.mkdir(parents=True)
    imports = "\n".join(f"import module_{index}" for index in range(1, file_count))
    project.joinpath("main.py").write_text(
        (imports + "\n") if imports else "def helper(): pass\n", encoding="utf-8"
    )
    for index in range(1, file_count):
        project.joinpath(f"module_{index}.py").write_text(
            f"def fn_{index}():\n    return {index}\n", encoding="utf-8"
        )
    # No large_file_strategy override: that config key was introduced in
    # 0.14.0 (section 6.4), so Matrix A's peer (official 0.13.1) predates it
    # entirely and would reject it as unknown. Omitting the key is also
    # behaviorally correct, since truncate-shaped handling is the only
    # behavior either version has for this ordinary (non-split) project.
    _write_config(project, documentation_scope="all")
    return project


def _large_file_identity(project: Path) -> str:
    record = json.loads(
        (project / "docs" / "codedoc.json").read_text(encoding="utf-8")
    )["files"][0]
    identity = record.get("_large_file_identity")
    if not isinstance(identity, str) or not identity.startswith("large-file-v3:"):
        raise SmokeFailure("cross-version-missing-or-malformed-large-file-identity")
    return identity


def _public_document_excluding_last_run(project: Path) -> dict:
    """`last_run` truthfully reports what happened in the most recent
    invocation (documented vs. reused/regenerated counts), so it always
    legitimately differs between runs even when the rest of the document is
    untouched. Everything else -- `files`, `tree`, `folders`, and any other
    top-level field -- must be byte-for-byte identical for two runs to count
    as producing the same semantic baseline."""
    doc = json.loads((project / "docs" / "codedoc.json").read_text(encoding="utf-8"))
    doc.pop("last_run", None)
    return doc


def _ordinary_round_trip(work: Path, name: str, *, creator: str, reader: str) -> None:
    project = work / name
    project.mkdir()
    project.joinpath("main.py").write_text("def helper(): pass\n", encoding="utf-8")
    _write_config(project, large_file_strategy="truncate")
    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=creator)
    if result.returncode != 0 or calls <= 0:
        raise SmokeFailure(f"ordinary-round-trip-{name}-setup-failed")
    before = _public_document_excluding_last_run(project)
    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=reader)
    if result.returncode != 0:
        raise SmokeFailure(f"ordinary-round-trip-{name}-read-failed")
    if calls != 0:
        raise SmokeFailure(f"ordinary-round-trip-{name}-did-not-reuse-zero-call")
    after = _public_document_excluding_last_run(project)
    if after != before:
        raise SmokeFailure(f"ordinary-round-trip-{name}-documentation-changed-on-reuse")


def _output_format_cli_args(fmt: str, *, split: bool) -> list[str]:
    args = [".", "--entry", "main.py", "--output", "docs", "--format", fmt]
    if split:
        args += ["--large-file-strategy", "split"]
    return args + ["--no-parallel"]


_RECOVERY_REFUSAL_FORBIDDEN = (
    "unrecognized argument",
    "unrecognized option",
    "unknown argument",
    "unknown option",
    "unknown config",
    "unknown configuration",
    "unknown key",
    "no module named",
    "modulenotfounderror",
    "importerror",
    "entry file",
    "entry point",
    "project root",
    "source root",
    "malformed configuration",
    "configuration parse",
    "parse configuration",
    "configuration syntax",
    "invalid configuration",
    "malformed config",
    "could not read config",
    "failed to load config",
)


def _assert_recovery_specific_refusal(
    project: Path,
    result: "subprocess.CompletedProcess[str]",
    calls: int,
    before_snapshot: dict[str, str | None],
    token: str,
) -> None:
    """Reject false-green exit-2 results that are unrelated to recovery."""
    if result.returncode != 2:
        raise SmokeFailure(f"{token}-not-bounded-exit-2: exit={result.returncode}")
    if calls != 0:
        raise SmokeFailure(f"{token}-unexpected-provider-calls: {calls}")
    sidecar = project / ".codedoc-smoke-calls.json"
    if sidecar.read_text(encoding="utf-8") != "0":
        raise SmokeFailure(f"{token}-call-sidecar-not-exact-zero")
    if _snapshot(project) != before_snapshot:
        raise SmokeFailure(f"{token}-mutated-recovery-artifacts")

    stderr = result.stderr
    if len(stderr.encode("utf-8")) > 16_000:
        raise SmokeFailure(f"{token}-stderr-unbounded")
    lowered = stderr.lower()
    if any(phrase in lowered for phrase in _RECOVERY_REFUSAL_FORBIDDEN):
        raise SmokeFailure(f"{token}-unrelated-exit-2-diagnostic")
    recovery_specific = "crash_recovery.json" in lowered or (
        "recovery" in lowered
        and any(term in lowered for term in ("compatib", "schema", "identity", "unsupported"))
    )
    if not recovery_specific:
        raise SmokeFailure(f"{token}-non-recovery-exit-2-diagnostic")
    _assert_private(project, result.stdout + stderr)


def _require_clean_run_control(
    project: Path, cli_args: list[str], *, python_exe: str, token: str
) -> None:
    """Prove an interpreter/argument/config/project combination works cleanly."""
    result, calls = _run_cli(project, cli_args, python_exe=python_exe)
    if result.returncode != 0 or calls <= 0:
        raise SmokeFailure(
            f"{token}-clean-run-control-failed: calls={calls} "
            f"exit={result.returncode} {result.stderr[-2000:]}"
        )
    if (project / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure(f"{token}-clean-run-control-left-recovery")


def _semantic_output(project: Path, fmt: str) -> dict:
    """Return the lossless public view while excluding run-specific counters."""
    from codedoc.core.document import read_codedoc_document

    docs = project / "docs"
    selected = docs / ("codedoc.md" if fmt == "md" else "codedoc.json")
    view = json.loads(json.dumps(read_codedoc_document(selected).view))
    view.pop("last_run", None)
    if fmt == "both":
        markdown_view = json.loads(
            json.dumps(read_codedoc_document(docs / "codedoc.md").view)
        )
        markdown_view.pop("last_run", None)
        if markdown_view != view:
            raise SmokeFailure("both-output-semantic-divergence")
    return view


# Exact inventory a *fresh* run of each output format leaves in `docs/`,
# with no pre-existing output of any kind (plan section 6.4's first three
# table rows).
_FRESH_FORMAT_INVENTORIES: dict[str, frozenset[str]] = {
    "json": frozenset({"codedoc.json"}),
    "md": frozenset({"codedoc.md"}),
    "both": frozenset({"codedoc.json", "codedoc.md"}),
}

# Exact inventory after the *reader* leg of a format transition, keyed by
# (start_fmt, end_fmt) -- plan section 6.4's table, which is NOT simply "the
# file for the requested format".  A same-stem format switch preserves the
# previous opposite-format sibling, so `json -> md` leaves BOTH files behind,
# not only `codedoc.md`.
#
# The two single -> `both` legs are the trap: their inventory is identical to
# the two single -> opposite-single legs, but their pre-existing file is
# *rewritten* rather than preserved.  `_run_cross_version_format_table`
# therefore asserts byte-identity on exactly the two opposite-single legs and
# semantic equivalence only on the `both` legs.
#
# `crash_recovery.json` is deliberately absent from every entry: a completed
# run must leave none, and this exact-set comparison is what proves it.
_TRANSITION_INVENTORIES: dict[tuple[str, str], frozenset[str]] = {
    ("json", "json"): frozenset({"codedoc.json"}),
    ("md", "md"): frozenset({"codedoc.md"}),
    ("both", "both"): frozenset({"codedoc.json", "codedoc.md"}),
    ("json", "md"): frozenset({"codedoc.json", "codedoc.md"}),
    ("md", "json"): frozenset({"codedoc.json", "codedoc.md"}),
    ("json", "both"): frozenset({"codedoc.json", "codedoc.md"}),
    ("md", "both"): frozenset({"codedoc.json", "codedoc.md"}),
}


def _assert_output_inventory(
    project: Path, expected: frozenset[str], token: str
) -> None:
    """Assert `docs/` holds exactly *expected*.

    *expected* is passed in rather than derived from the requested format,
    because the correct inventory depends on what was already there: see
    `_FRESH_FORMAT_INVENTORIES` and `_TRANSITION_INVENTORIES`.
    """
    docs = project / "docs"
    if not docs.is_dir():
        raise SmokeFailure(f"{token}-output-directory-missing")
    actual = {path.name for path in docs.iterdir() if path.is_file()}
    if actual != set(expected):
        raise SmokeFailure(f"{token}-inventory-mismatch: {sorted(actual)} != {sorted(expected)}")


def _records_by_path_from_view(view: dict) -> dict[str, dict]:
    records = view.get("files")
    if not isinstance(records, list):
        raise SmokeFailure("format-baseline-files-missing")
    return {
        record["path"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }


def _candidate_can_reuse(creator_view: dict, candidate_view: dict) -> bool:
    """Evaluate compatibility with the candidate's actual identity registry."""
    from codedoc.core.record_meta import CACHE_IDENTITY_KEYS, normalized_identity_value

    creator_records = _records_by_path_from_view(creator_view)
    candidate_records = _records_by_path_from_view(candidate_view)
    if set(creator_records) != set(candidate_records):
        return False
    for path, expected in candidate_records.items():
        stored = creator_records[path]
        if stored.get("hash") != expected.get("hash"):
            return False
        if any(
            normalized_identity_value(key, stored)
            != normalized_identity_value(key, expected)
            for key in CACHE_IDENTITY_KEYS
        ):
            return False
    return True


_FORMAT_TRANSITIONS = (
    ("json", "json"),
    ("md", "md"),
    ("both", "both"),
    ("json", "md"),
    ("md", "json"),
    ("json", "both"),
    ("md", "both"),
)


def _fresh_format_baselines(
    work: Path,
    *,
    matrix: str,
    role: str,
    builder: Callable[[Path, str], Path],
    python_exe: str,
    split: bool,
) -> dict[str, tuple[int, dict]]:
    baselines: dict[str, tuple[int, dict]] = {}
    for fmt in ("json", "md", "both"):
        token = f"matrix-{matrix}-4-{role}-fresh-{fmt}"
        project = builder(work, f"{matrix}4-{role}-fresh-{fmt}")
        result, calls = _run_cli(
            project,
            _output_format_cli_args(fmt, split=split),
            python_exe=python_exe,
        )
        if result.returncode != 0 or calls <= 0:
            raise SmokeFailure(
                f"{token}-failed: calls={calls} exit={result.returncode} "
                f"{result.stderr[-2000:]}"
            )
        _assert_output_inventory(project, _FRESH_FORMAT_INVENTORIES[fmt], token)
        baselines[fmt] = (calls, _semantic_output(project, fmt))
    return baselines


def _run_cross_version_format_table(
    work: Path,
    peer_python: Path,
    *,
    matrix: str,
    builder: Callable[[Path, str], Path],
    split: bool,
) -> None:
    """Exercise both interpreter directions for the complete R5 format table."""
    candidate = _fresh_format_baselines(
        work,
        matrix=matrix,
        role="candidate",
        builder=builder,
        python_exe=sys.executable,
        split=split,
    )
    peer = _fresh_format_baselines(
        work,
        matrix=matrix,
        role="peer",
        builder=builder,
        python_exe=str(peer_python),
        split=split,
    )
    directions = (
        ("peer-to-candidate", str(peer_python), sys.executable, peer, candidate, True),
        ("candidate-to-peer", sys.executable, str(peer_python), candidate, peer, False),
    )
    leak_oracle_ran = False
    for direction, creator_python, reader_python, creator_base, reader_base, candidate_reader in directions:
        for start_fmt, end_fmt in _FORMAT_TRANSITIONS:
            token = f"matrix-{matrix}-4-{direction}-{start_fmt}-to-{end_fmt}"
            project = builder(work, f"{matrix}4-{direction}-{start_fmt}-to-{end_fmt}")
            creator_result, creator_calls = _run_cli(
                project,
                _output_format_cli_args(start_fmt, split=split),
                python_exe=creator_python,
            )
            expected_creator_calls, expected_creator_view = creator_base[start_fmt]
            if creator_result.returncode != 0 or creator_calls != expected_creator_calls:
                raise SmokeFailure(
                    f"{token}-creator-call-count: {creator_calls} != "
                    f"{expected_creator_calls} (exit={creator_result.returncode}) "
                    f"{creator_result.stderr[-2000:]}"
                )
            _assert_output_inventory(
                project, _FRESH_FORMAT_INVENTORIES[start_fmt], f"{token}-creator"
            )
            creator_view = _semantic_output(project, start_fmt)
            if creator_view != expected_creator_view:
                raise SmokeFailure(f"{token}-creator-semantic-baseline-mismatch")

            preserved_path: Path | None = None
            preserved_bytes: bytes | None = None
            if (start_fmt, end_fmt) in (("json", "md"), ("md", "json")):
                preserved_path = project / "docs" / f"codedoc.{start_fmt}"
                preserved_bytes = preserved_path.read_bytes()

            reader_result, reader_calls = _run_cli(
                project,
                _output_format_cli_args(end_fmt, split=split),
                python_exe=reader_python,
            )
            expected_reader_calls, expected_reader_view = reader_base[end_fmt]
            if reader_result.returncode != 0:
                raise SmokeFailure(
                    f"{token}-reader-failed: exit={reader_result.returncode} "
                    f"{reader_result.stderr[-2000:]}"
                )
            if candidate_reader:
                compatible = _candidate_can_reuse(creator_view, expected_reader_view)
                exact_reader_calls = 0 if compatible else expected_reader_calls
                if reader_calls != exact_reader_calls:
                    raise SmokeFailure(
                        f"{token}-candidate-call-count: {reader_calls} != "
                        f"{exact_reader_calls} (compatible={compatible})"
                    )
            elif reader_calls not in (0, expected_reader_calls):
                raise SmokeFailure(
                    f"{token}-peer-partial-call-count: {reader_calls} not in "
                    f"(0, {expected_reader_calls})"
                )

            _assert_output_inventory(
                project,
                _TRANSITION_INVENTORIES[(start_fmt, end_fmt)],
                f"{token}-reader",
            )
            if _semantic_output(project, end_fmt) != expected_reader_view:
                raise SmokeFailure(f"{token}-reader-semantic-baseline-mismatch")
            if preserved_path is not None and preserved_path.read_bytes() != preserved_bytes:
                raise SmokeFailure(f"{token}-opposite-format-sibling-not-preserved")
            if not leak_oracle_ran and direction == "peer-to-candidate" and end_fmt == "both":
                _assert_private(
                    project,
                    creator_result.stdout
                    + creator_result.stderr
                    + reader_result.stdout
                    + reader_result.stderr,
                )
                leak_oracle_ran = True
    if not leak_oracle_ran:
        raise SmokeFailure(f"matrix-{matrix}-4-completed-leak-oracle-not-run")


def _exercise_recovery_refusal_oracle(
    work: Path,
    *,
    matrix: str,
    builder: Callable[[Path, str], Path],
    split: bool,
) -> None:
    """Force one recovery-specific candidate refusal and leak scan per matrix."""
    args = _output_format_cli_args("json", split=split)
    _require_clean_run_control(
        builder(work, f"{matrix}4-refusal-control"),
        args,
        python_exe=sys.executable,
        token=f"matrix-{matrix}-4-refusal",
    )
    project = builder(work, f"{matrix}4-refusal")
    recovery = project / "docs" / "crash_recovery.json"
    recovery.parent.mkdir()
    recovery.write_text(_legacy_recovery(99), encoding="utf-8")
    before_snapshot = _snapshot(project)
    result, calls = _run_cli(project, args, python_exe=sys.executable)
    _assert_recovery_specific_refusal(
        project,
        result,
        calls,
        before_snapshot,
        f"matrix-{matrix}-4-refusal",
    )


# ---------------------------------------------------------------------------
# Matrix A -- ordinary-record predecessor (official PyPI 0.13.1). This peer
# predates `large_file_strategy: split` entirely, so nothing here ever
# exercises split configuration or split-shaped assertions (plan section
# 6.4). Every step drives peer and candidate as fresh subprocesses against a
# shared project directory, exchanging state only through that directory.
# ---------------------------------------------------------------------------


def _ordinary_fresh_baseline(work: Path) -> tuple[int, dict]:
    project = _new_ordinary_project(work, "matrix-a-baseline")
    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0 or calls <= 1:
        raise SmokeFailure("matrix-a-baseline-run-failed-or-too-small")
    return calls, _public_document_excluding_last_run(project)


def _peer_ordinary_fresh_baseline(work: Path, peer_python: Path) -> tuple[int, dict]:
    project = _new_ordinary_project(work, "matrix-a-peer-baseline")
    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=str(peer_python))
    if result.returncode != 0 or calls <= 1:
        raise SmokeFailure("matrix-a-peer-baseline-run-failed-or-too-small")
    return calls, _public_document_excluding_last_run(project)


def _matrix_a_step1_regeneration_required(
    work: Path, peer_python: Path, fresh_calls: int, fresh_document: dict
) -> None:
    project = _new_ordinary_project(work, "a1-regeneration")
    peer_result, peer_calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=str(peer_python))
    if peer_result.returncode != 0 or peer_calls <= 0:
        raise SmokeFailure(f"matrix-a-1-peer-run-failed: {peer_result.stderr[-2000:]}")
    if (project / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("matrix-a-1-peer-left-recovery")

    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0:
        raise SmokeFailure(f"matrix-a-1-candidate-run-failed: {result.stderr[-2000:]}")
    if calls != fresh_calls:
        raise SmokeFailure(
            "matrix-a-1-candidate-did-not-fully-regenerate-every-peer-record: "
            f"{calls} != fresh baseline {fresh_calls}"
        )
    if _public_document_excluding_last_run(project) != fresh_document:
        raise SmokeFailure("matrix-a-1-candidate-output-diverged-from-fresh-baseline")


def _matrix_a_step2_peer_recovery_preserved_or_resumed(
    work: Path, peer_python: Path, fresh_calls: int, fresh_document: dict
) -> None:
    _require_clean_run_control(
        _new_ordinary_project(work, "a2-clean-control"),
        _ORDINARY_CLI_ARGS,
        python_exe=sys.executable,
        token="matrix-a-2-candidate",
    )
    project = _new_ordinary_project(work, "a2-peer-recovery")
    peer_interrupted, peer_first_calls = _run_cli(
        project, _ORDINARY_CLI_ARGS, python_exe=str(peer_python), interrupt_after=1
    )
    if peer_interrupted.returncode != 130:
        raise SmokeFailure(
            f"matrix-a-2-peer-did-not-cleanly-interrupt: exit={peer_interrupted.returncode}"
        )
    if peer_first_calls != 1:
        raise SmokeFailure("matrix-a-2-peer-unexpected-call-count")
    recovery = project / "docs" / "crash_recovery.json"
    if not recovery.exists():
        raise SmokeFailure("matrix-a-2-peer-left-no-recovery")
    before_snapshot = _snapshot(project)

    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=sys.executable)
    if result.returncode == 0:
        if peer_first_calls + calls != fresh_calls:
            raise SmokeFailure(
                "matrix-a-2-candidate-resume-did-not-reconcile-with-fresh-baseline: "
                f"{peer_first_calls} + {calls} != {fresh_calls}"
            )
        if recovery.exists():
            raise SmokeFailure("matrix-a-2-candidate-left-recovery-after-completion")
        if _public_document_excluding_last_run(project) != fresh_document:
            raise SmokeFailure("matrix-a-2-candidate-resumed-output-diverged-from-fresh-baseline")
    elif result.returncode == 2:
        _assert_recovery_specific_refusal(
            project, result, calls, before_snapshot, "matrix-a-2-candidate"
        )
    else:
        raise SmokeFailure(
            f"matrix-a-2-candidate-neither-resumed-nor-blocked-boundedly: exit={result.returncode}"
        )


def _matrix_a_step3_candidate_recovery_not_corrupted(
    work: Path,
    peer_python: Path,
    fresh_calls: int,
    peer_fresh_calls: int,
    peer_fresh_document: dict,
) -> None:
    _require_clean_run_control(
        _new_ordinary_project(work, "a3-clean-control"),
        _ORDINARY_CLI_ARGS,
        python_exe=str(peer_python),
        token="matrix-a-3-peer",
    )
    project = _new_ordinary_project(work, "a3-candidate-recovery")
    candidate_interrupted, candidate_first_calls = _run_cli(
        project, _ORDINARY_CLI_ARGS, python_exe=sys.executable, interrupt_after=1
    )
    if candidate_interrupted.returncode != 130:
        raise SmokeFailure(
            f"matrix-a-3-candidate-did-not-cleanly-interrupt: exit={candidate_interrupted.returncode}"
        )
    if candidate_first_calls != 1:
        raise SmokeFailure("matrix-a-3-candidate-unexpected-call-count")
    recovery = project / "docs" / "crash_recovery.json"
    if not recovery.exists():
        raise SmokeFailure("matrix-a-3-candidate-left-no-recovery")
    before_snapshot = _snapshot(project)

    result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=str(peer_python))
    if result.returncode == 2:
        _assert_recovery_specific_refusal(
            project, result, calls, before_snapshot, "matrix-a-3-peer"
        )
        # The candidate must subsequently resume its own interrupted recovery.
        result, calls = _run_cli(project, _ORDINARY_CLI_ARGS, python_exe=sys.executable)
        if result.returncode != 0:
            raise SmokeFailure(f"matrix-a-3-candidate-resume-failed: {result.stderr[-2000:]}")
        if candidate_first_calls + calls != fresh_calls:
            raise SmokeFailure("matrix-a-3-candidate-resume-call-count-mismatch")
        if recovery.exists():
            raise SmokeFailure("matrix-a-3-candidate-resume-left-recovery")
    elif result.returncode == 0:
        if calls <= 0:
            raise SmokeFailure("matrix-a-3-peer-claimed-completion-with-zero-calls")
        if recovery.exists():
            raise SmokeFailure("matrix-a-3-peer-completed-run-left-recovery")
        if candidate_first_calls + calls != peer_fresh_calls:
            raise SmokeFailure(
                "matrix-a-3-peer-resume-did-not-reconcile-with-its-own-fresh-baseline"
            )
        if _public_document_excluding_last_run(project) != peer_fresh_document:
            raise SmokeFailure(
                "matrix-a-3-peer-resumed-output-diverged-from-its-own-fresh-baseline"
            )
    else:
        raise SmokeFailure(
            f"matrix-a-3-peer-neither-resumed-nor-blocked-boundedly: exit={result.returncode}"
        )


def _matrix_a_step4_output_formats_and_leak_freedom(work: Path, peer_python: Path) -> None:
    """R5: both directions of the exact seven-leg output transition table."""
    def builder(parent: Path, name: str) -> Path:
        return _new_ordinary_project(parent, name, file_count=1)

    _run_cross_version_format_table(
        work, peer_python, matrix="a", builder=builder, split=False
    )
    _exercise_recovery_refusal_oracle(
        work, matrix="a", builder=builder, split=False
    )


def _matrix_a_ordinary_predecessor(work: Path, peer_python: Path) -> None:
    fresh_calls, fresh_document = _ordinary_fresh_baseline(work)
    peer_fresh_calls, peer_fresh_document = _peer_ordinary_fresh_baseline(work, peer_python)
    _matrix_a_step1_regeneration_required(work, peer_python, fresh_calls, fresh_document)
    _matrix_a_step2_peer_recovery_preserved_or_resumed(work, peer_python, fresh_calls, fresh_document)
    _matrix_a_step3_candidate_recovery_not_corrupted(
        work, peer_python, fresh_calls, peer_fresh_calls, peer_fresh_document
    )
    _matrix_a_step4_output_formats_and_leak_freedom(work, peer_python)
    print("matrix A (ordinary predecessor) installed artifact matrix: ok")


# ---------------------------------------------------------------------------
# Matrix B -- split-capable predecessor (TestPyPI 0.14.4). This peer
# supports `large_file_strategy: split`, so it runs the full split matrix
# (plan section 6.4).
# ---------------------------------------------------------------------------


def _split_fresh_baseline(work: Path) -> tuple[int, dict]:
    project = _new_split_project(work, "matrix-b-baseline")
    result, calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0 or calls <= 1:
        raise SmokeFailure("matrix-b-baseline-run-failed-or-too-small")
    return calls, _public_document_excluding_last_run(project)


def _peer_split_fresh_baseline(work: Path, peer_python: Path) -> tuple[int, dict]:
    project = _new_split_project(work, "matrix-b-peer-baseline")
    result, calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=str(peer_python))
    if result.returncode != 0 or calls <= 1:
        raise SmokeFailure("matrix-b-peer-baseline-run-failed-or-too-small")
    return calls, _public_document_excluding_last_run(project)


def _matrix_b_step1_peer_completes_candidate_consumes(
    work: Path, peer_python: Path, fresh_calls: int, fresh_document: dict
) -> None:
    project = _new_split_project(work, "b1-peer-completed")
    peer_result, peer_calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=str(peer_python))
    if peer_result.returncode != 0 or peer_calls <= 0:
        raise SmokeFailure(f"matrix-b-1-peer-run-failed: {peer_result.stderr[-2000:]}")
    if (project / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("matrix-b-1-peer-left-recovery")

    result, calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode != 0:
        raise SmokeFailure(f"matrix-b-1-candidate-consume-failed: {result.stderr[-2000:]}")
    if calls not in (0, fresh_calls):
        raise SmokeFailure(
            "matrix-b-1-candidate-did-neither-zero-call-reuse-nor-full-regenerate: "
            f"{calls} calls (expected 0 or {fresh_calls})"
        )
    if (project / "docs" / "crash_recovery.json").exists():
        raise SmokeFailure("matrix-b-1-candidate-did-not-cleanly-finalize")
    _large_file_identity(project)
    if _public_document_excluding_last_run(project) != fresh_document:
        raise SmokeFailure("matrix-b-1-candidate-output-diverged-from-fresh-baseline")


def _matrix_b_step2_peer_recovery_preserved_or_resumed(
    work: Path, peer_python: Path, fresh_calls: int, fresh_document: dict
) -> None:
    _require_clean_run_control(
        _new_split_project(work, "b2-clean-control"),
        _SPLIT_CLI_ARGS,
        python_exe=sys.executable,
        token="matrix-b-2-candidate",
    )
    project = _new_split_project(work, "b2-peer-recovery")
    peer_interrupted, peer_first_calls = _run_cli(
        project, _SPLIT_CLI_ARGS, python_exe=str(peer_python), interrupt_after=1
    )
    if peer_interrupted.returncode != 130:
        raise SmokeFailure(
            f"matrix-b-2-peer-did-not-cleanly-interrupt: exit={peer_interrupted.returncode}"
        )
    if peer_first_calls != 1:
        raise SmokeFailure("matrix-b-2-peer-unexpected-call-count")
    recovery = project / "docs" / "crash_recovery.json"
    if not recovery.exists():
        raise SmokeFailure("matrix-b-2-peer-left-no-recovery")
    before_snapshot = _snapshot(project)

    result, calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=sys.executable)
    if result.returncode == 0:
        if peer_first_calls + calls != fresh_calls:
            raise SmokeFailure(
                "matrix-b-2-candidate-resume-did-not-reconcile-with-fresh-baseline: "
                f"{peer_first_calls} + {calls} != {fresh_calls}"
            )
        if recovery.exists():
            raise SmokeFailure("matrix-b-2-candidate-left-recovery-after-completion")
        if _public_document_excluding_last_run(project) != fresh_document:
            raise SmokeFailure("matrix-b-2-candidate-resumed-output-diverged-from-fresh-baseline")
    elif result.returncode == 2:
        _assert_recovery_specific_refusal(
            project, result, calls, before_snapshot, "matrix-b-2-candidate"
        )
    else:
        raise SmokeFailure(
            f"matrix-b-2-candidate-neither-resumed-nor-blocked-boundedly: exit={result.returncode}"
        )


def _matrix_b_step3_candidate_recovery_not_corrupted(
    work: Path,
    peer_python: Path,
    fresh_calls: int,
    peer_fresh_calls: int,
    peer_fresh_document: dict,
) -> None:
    _require_clean_run_control(
        _new_split_project(work, "b3-clean-control"),
        _SPLIT_CLI_ARGS,
        python_exe=str(peer_python),
        token="matrix-b-3-peer",
    )
    project = _new_split_project(work, "b3-candidate-recovery")
    candidate_interrupted, candidate_first_calls = _run_cli(
        project, _SPLIT_CLI_ARGS, python_exe=sys.executable, interrupt_after=1
    )
    if candidate_interrupted.returncode != 130:
        raise SmokeFailure(
            f"matrix-b-3-candidate-did-not-cleanly-interrupt: exit={candidate_interrupted.returncode}"
        )
    if candidate_first_calls != 1:
        raise SmokeFailure("matrix-b-3-candidate-unexpected-call-count")
    recovery = project / "docs" / "crash_recovery.json"
    if not recovery.exists():
        raise SmokeFailure("matrix-b-3-candidate-left-no-recovery")
    before_snapshot = _snapshot(project)

    result, calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=str(peer_python))
    if result.returncode == 2:
        _assert_recovery_specific_refusal(
            project, result, calls, before_snapshot, "matrix-b-3-peer"
        )
        result, calls = _run_cli(project, _SPLIT_CLI_ARGS, python_exe=sys.executable)
        if result.returncode != 0:
            raise SmokeFailure(f"matrix-b-3-candidate-resume-failed: {result.stderr[-2000:]}")
        if candidate_first_calls + calls != fresh_calls:
            raise SmokeFailure("matrix-b-3-candidate-resume-call-count-mismatch")
        if recovery.exists():
            raise SmokeFailure("matrix-b-3-candidate-resume-left-recovery")
    elif result.returncode == 0:
        if calls <= 0:
            raise SmokeFailure("matrix-b-3-peer-claimed-completion-with-zero-calls")
        if recovery.exists():
            raise SmokeFailure("matrix-b-3-peer-completed-run-left-recovery")
        if candidate_first_calls + calls != peer_fresh_calls:
            raise SmokeFailure(
                "matrix-b-3-peer-resume-did-not-reconcile-with-its-own-fresh-baseline"
            )
        if _public_document_excluding_last_run(project) != peer_fresh_document:
            raise SmokeFailure(
                "matrix-b-3-peer-resumed-output-diverged-from-its-own-fresh-baseline"
            )
    else:
        raise SmokeFailure(
            f"matrix-b-3-peer-neither-resumed-nor-blocked-boundedly: exit={result.returncode}"
        )


def _matrix_b_step4_output_formats_and_leak_freedom(
    work: Path, peer_python: Path, fresh_calls: int, peer_fresh_calls: int
) -> None:
    """R5: both directions of the exact seven-leg split output table."""
    del fresh_calls, peer_fresh_calls  # Baselines are format-specific and re-proven here.
    _run_cross_version_format_table(
        work, peer_python, matrix="b", builder=_new_split_project, split=True
    )
    _exercise_recovery_refusal_oracle(
        work, matrix="b", builder=_new_split_project, split=True
    )

    _ordinary_round_trip(work, "b4-ordinaryA", creator=str(peer_python), reader=sys.executable)
    _ordinary_round_trip(work, "b4-ordinaryB", creator=sys.executable, reader=str(peer_python))


def _matrix_b_split_predecessor(work: Path, peer_python: Path) -> None:
    fresh_calls, fresh_document = _split_fresh_baseline(work)
    peer_fresh_calls, peer_fresh_document = _peer_split_fresh_baseline(work, peer_python)
    _matrix_b_step1_peer_completes_candidate_consumes(work, peer_python, fresh_calls, fresh_document)
    _matrix_b_step2_peer_recovery_preserved_or_resumed(work, peer_python, fresh_calls, fresh_document)
    _matrix_b_step3_candidate_recovery_not_corrupted(
        work, peer_python, fresh_calls, peer_fresh_calls, peer_fresh_document
    )
    _matrix_b_step4_output_formats_and_leak_freedom(
        work, peer_python, fresh_calls, peer_fresh_calls
    )
    print("matrix B (split-capable predecessor) installed artifact matrix: ok")


def _scenario_cross_version(work: Path, peer_python: Path, peer_version: str) -> None:
    """Section 18 predecessor-artifact compatibility matrices.

    *peer_python* is a genuinely separate installed environment's
    interpreter for the exact released version named by *peer_version*; the
    current process's own interpreter (`sys.executable`) is the candidate
    under test. The two supported predecessors are not interchangeable:
    official `0.13.1` predates `large_file_strategy: split` entirely and
    runs Matrix A, while TestPyPI `0.14.4` is split-capable and runs Matrix
    B. Every step drives one or the other as a fresh subprocess against a
    shared project directory under *work*, exchanging state only through
    that directory -- neither environment ever imports the other.
    """
    matrix = _PEER_VERSION_MATRIX.get(peer_version)
    if matrix is None:
        raise SmokeFailure(
            f"unsupported-peer-version: {peer_version!r} (expected one of "
            f"{sorted(_PEER_VERSION_MATRIX)})"
        )
    _prove_peer_installed_origin(peer_python, peer_version)
    if matrix == "a":
        _matrix_a_ordinary_predecessor(work, peer_python)
    else:
        _matrix_b_split_predecessor(work, peer_python)


def _invoke_child_cli(cli_main: Callable[[list[str]], None], cli_args: list[str]) -> int:
    """Normalize an intentional child interruption to one portable status."""
    try:
        cli_main(cli_args)
    except KeyboardInterrupt:
        # A programmatically raised KeyboardInterrupt exits as 130 on POSIX
        # but 0xC000013A on Windows.  The matrix contract uses one portable,
        # intentional-interruption token rather than treating the Windows
        # process status as an unrelated crash.
        return 130
    return 0


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

    return _invoke_child_cli(cli_main, cli_args)


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
        "--peer-version",
        help=(
            "Exact released version installed under --peer-python, one of "
            f"{sorted(_PEER_VERSION_MATRIX)}. Required for --scenario cross-version."
        ),
    )
    parser.add_argument(
        "--candidate-version",
        help=(
            "Exact candidate version installed under this interpreter. "
            "Required for --scenario cross-version."
        ),
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
        if (
            args.peer_python is None
            or args.peer_version is None
            or args.candidate_version is None
            or args.work is None
        ):
            raise SmokeFailure(
                "cross-version-requires-peer-python-peer-version-candidate-version-and-work"
            )
        work = args.work.resolve()
        if _is_within(work, _repository_root()):
            raise SmokeFailure("cross-version-work-inside-repository")
        if not work.is_dir():
            raise SmokeFailure("cross-version-work-must-be-an-existing-directory")
        if any(work.iterdir()):
            raise SmokeFailure("cross-version-work-must-be-empty")
        _prove_installed_origin(args.candidate_version)
        _scenario_cross_version(work, args.peer_python.resolve(), args.peer_version)
        return 0
    return _run_all()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"installed artifact smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
