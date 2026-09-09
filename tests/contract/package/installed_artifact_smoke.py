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
import re
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


def _canonical_distribution_name(name: object) -> str:
    """PEP 503 name normalization: case-fold and collapse every run of ``-``,
    ``_`` or ``.`` to a single hyphen, so ``codedoc_ai``, ``Codedoc-AI`` and
    ``codedoc..ai`` all compare equal to ``codedoc-ai``."""
    text = name if isinstance(name, str) else ""
    return re.sub(r"[-_.]+", "-", text.strip()).lower()


def _distribution_owns_imported_package(dist: object, package_path: Path) -> bool:
    """True when *dist*'s own metadata says it installed the exact
    ``codedoc/__init__.py`` file that was imported -- the origin binding a bare
    ``importlib.metadata.version("codedoc-ai")`` lookup does not provide."""
    try:
        located = Path(dist.locate_file("codedoc/__init__.py")).resolve()
    except Exception:
        return False
    return located == package_path


def _origin_bound_distribution(
    package_path: Path, package_site_root: Path
) -> object:
    """Return the single installed ``codedoc-ai`` distribution that both owns
    the imported package file and lives under the same site-packages root.

    Distinct, stable failures for the three ways this can go wrong:

    * ``codedoc-ai-distribution-not-found`` -- no ``codedoc-ai`` distribution
      is installed at all;
    * ``codedoc-ai-distribution-origin-mismatch`` -- one or more are installed
      but none is co-located with the imported package (e.g. only a
      repository-local ``codedoc_ai.egg-info`` shadow, or an install under a
      different site root);
    * ``codedoc-ai-distribution-ambiguous`` -- more than one qualifies.
    """
    named = [
        dist
        for dist in importlib_metadata.distributions()
        if _canonical_distribution_name(_distribution_name(dist)) == "codedoc-ai"
    ]
    if not named:
        raise SmokeFailure("codedoc-ai-distribution-not-found")
    origin_bound = [
        dist
        for dist in named
        if _distribution_owns_imported_package(dist, package_path)
        and _distribution_is_under(dist, package_site_root)
    ]
    if not origin_bound:
        raise SmokeFailure("codedoc-ai-distribution-origin-mismatch")
    if len(origin_bound) > 1:
        raise SmokeFailure("codedoc-ai-distribution-ambiguous")
    return origin_bound[0]


def _distribution_name(dist: object) -> object:
    try:
        return dist.metadata["Name"]
    except Exception:
        return getattr(dist, "name", None)


def _distribution_is_under(dist: object, root: Path) -> bool:
    """Whether *dist*'s on-disk metadata directory sits under *root*.  A
    non-path distribution (no ``_path``) is not rejected on this basis alone --
    :func:`_distribution_owns_imported_package` is the authoritative bind."""
    origin = getattr(dist, "_path", None)
    if origin is None:
        return True
    try:
        return _is_within(Path(origin).resolve(), root)
    except Exception:
        return False


def _prove_installed_origin(expected_version: str | None = None) -> tuple[Path, Path]:
    """Prove product imports and the console script come from this environment.

    This deliberately runs before changing directory or importing project
    content.  The harness itself may live in the checkout; the product under
    test may not.  Distribution metadata is bound to the imported package's
    own origin, so a repository-local ``codedoc_ai.egg-info`` (or any other
    unrelated ``codedoc-ai`` install) cannot stand in for the real one.
    """
    import codedoc

    package_path = Path(codedoc.__file__).resolve()
    module_version = codedoc.__version__
    if not isinstance(module_version, str) or not module_version:
        raise SmokeFailure("candidate-module-version-missing")

    repository = _repository_root()
    site_roots = {
        Path(value).resolve()
        for value in site.getsitepackages()
        if value
    }
    if _is_within(package_path, repository):
        raise SmokeFailure("installed-origin-check-failed")
    package_site_root = next(
        (root for root in site_roots if _is_within(package_path, root)),
        None,
    )
    if package_site_root is None:
        raise SmokeFailure("site-packages-origin-check-failed")

    distribution = _origin_bound_distribution(package_path, package_site_root)
    distribution_version = distribution.version
    if not isinstance(distribution_version, str) or not distribution_version:
        raise SmokeFailure("candidate-distribution-version-missing")
    if distribution_version != module_version:
        raise SmokeFailure(
            "candidate-module-metadata-version-mismatch: "
            f"{module_version!r} != {distribution_version!r}"
        )
    if expected_version is not None and module_version != expected_version:
        raise SmokeFailure(
            f"candidate-version-mismatch: installed {module_version!r} "
            f"!= expected --candidate-version {expected_version!r}"
        )

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
    # Compared against the origin-bound distribution version, which the checks
    # above have already tied to codedoc.__version__ and, when supplied, to
    # --candidate-version.
    expected_output = f"codedoc {distribution_version}"
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


#: The installed signature-acceptance boundary. Lengths track the frozen
#: live-validation fixture's real 1,520-character declaration and the exact
#: serialized-response hard bound; one code point past it fails closed. The
#: separate 600-character leaf-prompt hint clamp is asserted on its own, never
#: folded into this matrix.
_SIGNATURE_BOUND_MATRIX: tuple[tuple[int, bool], ...] = (
    (1520, False),
    (2000, False),
    (2001, True),
)


def _assert_prompt_signature_hint_is_600() -> None:
    """The internal leaf-prompt signature hint stays clamped at 600 characters
    no matter how large the serialized-response bound becomes. Its own
    diagnostic reason, distinct from the response-acceptance boundary."""
    from codedoc.core.file_division import MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS

    if MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS != 600:
        raise SmokeFailure(
            "prompt-signature-hint-chars-not-600: "
            f"{MAX_LEAF_PROMPT_SIGNATURE_HINT_CHARS!r}"
        )


def _scenario_signature_bound(root: Path) -> None:
    """The installed artifact accepts a serialized leaf ``signature`` up to the
    hard bound and rejects one past it, fail-closed, with no truncated public
    fact -- exercised end to end, not only at the source level.

    Response correction stays disabled here so the boundary observed is the
    cleaner's, not a repair's. The separate leaf-prompt signature-hint clamp
    is verified independently (:func:`_assert_prompt_signature_hint_is_600`)
    with its own diagnostic.
    """
    _assert_prompt_signature_hint_is_600()
    for signature_chars, expect_failure in _SIGNATURE_BOUND_MATRIX:
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


#: Frozen facts of the single tracked live-validation source fixture
#: (plan section 9.3). Verified before every installed use so a drifted or
#: substituted fixture fails loudly rather than silently changing the topology.
_LIVE_FIXTURE_REL_PARTS = (
    "tests",
    "fixtures",
    "live_validation",
    "oversized_signature.py",
)
_LIVE_FIXTURE_BYTES = 2317
_LIVE_FIXTURE_SHA256 = (
    "cfbf8716bcab26e996d6d467997eeb57fe53b172d21a9831d8fd42f623b24da1"
)

#: A base install has no ``tree-sitter-language-pack``; the optional
#: ``structure`` extra pins exactly this version.  The installed-artifact
#: harness certifies the two installations as two explicit profiles instead of
#: silently inheriting whichever topology the developer environment produces
#: (plan sections 5.1 / 5.2).  This is a release-harness selector only -- it is
#: never a public CodeDoc CLI option or configuration key.
_PARSER_DISTRIBUTION_NAME = "tree-sitter-language-pack"
_STRUCTURE_PROFILE_PARSER_VERSION = "0.13.0"
_STRUCTURE_PROFILES: tuple[str, ...] = ("base", "structure")

#: Fixture-topology facts identical in both profiles: the same byte-frozen
#: fixture and the same public product configuration are used for each, and the
#: only configuration difference is the positive call cap below (plan section
#: 5.2).  Every key names a stat the installed product itself publishes.
_LIVE_FIXTURE_SHARED_TOPOLOGY: dict[str, object] = {
    "dry_run": True,
    "large_file_strategy_resolved": "split",
    "large_file_source_ceiling_chars": 1000,
    "split_internal_manifest_budget_chars": 12000,
    "file_retry_attempts": 0,
    "split_divided_files": 1,
    "split_oversized_units": 1,
    "split_unit_consolidation_calls_planned": 1,
    "split_general_reduction_calls_planned": 0,
    "split_final_synthesis_calls_planned": 1,
    "prompt_review_calls_planned": 0,
    "split_boundary_cuts_balanced_codepoint": 1,
    "split_boundary_cuts_syntax": 0,
    "split_boundary_cuts_physical_line": 0,
    "split_crlf_atomicity_extra_chunks": 0,
    "max_planned_calls_exceeded": False,
    "retries_included_in_ceiling": False,
}

#: Per-profile topology deltas, independently frozen from the real production
#: planning path (plan section 5.2) -- never derived from the product under
#: test.  ``structural_mode`` is a per-file plan/identity field, not a run
#: stat, so the discriminating counters asserted here are the published
#: ``split_lexical_files`` / ``split_syntax_files``.  ``max_planned_calls`` is
#: both a frozen expectation and the sole configuration delta between the two
#: profile invocations.  If a fresh measurement contradicts either number,
#: stop and revise the plan -- do not loosen these to "greater than zero" or
#: recompute them from the product.
_LIVE_FIXTURE_PROFILE_TOPOLOGY: dict[str, dict[str, object]] = {
    "base": {
        "split_lexical_files": 1,
        "split_syntax_files": 0,
        "split_chunks": 4,
        "unit_documentation_calls_planned": 4,
        "initial_provider_calls_planned": 6,
        "initial_documentation_calls_planned": 6,
        "documentation_calls_planned": 6,
        "total_calls_planned": 6,
        "max_planned_calls": 6,
        "correction_calls_possible_max": 6,
        "provider_calls_max_before_retries": 12,
    },
    "structure": {
        "split_lexical_files": 0,
        "split_syntax_files": 1,
        "split_chunks": 3,
        "unit_documentation_calls_planned": 3,
        "initial_provider_calls_planned": 5,
        "initial_documentation_calls_planned": 5,
        "documentation_calls_planned": 5,
        "total_calls_planned": 5,
        "max_planned_calls": 5,
        "correction_calls_possible_max": 5,
        "provider_calls_max_before_retries": 10,
    },
}

#: Provider-free call-manifest digest per profile, independently frozen.  They
#: must stay stable and unequal because the two topologies genuinely differ
#: (plan section 5.3): a profile mismatch has to fail before any digest or
#: call count is accepted.  Each value is the ``call_manifest_digest`` the
#: real provider-free planning path produces for the frozen live fixture under
#: that profile; the per-profile split topology and call counts are unchanged,
#: but the manifest digest is the SHA-256 of the newline-joined initial
#: call ids, so it moves whenever any of those ids move.  Two revisions feed
#: it: ``FINAL_SYNTHESIS_REVISION`` through ``file_synthesis_call_id`` and
#: ``REDUCER_PROMPT_REVISION`` through ``file_reduction_call_id`` (this fixture
#: plans one unit-consolidation reduction node).  The 0.14.9 values below were
#: re-measured after the ``file-reduction-v3`` -> ``file-reduction-v4`` advance
#: of plan section 5.6.2: topology, call counts, and every category/owner/
#: ordinal stayed identical, and the single reduction call id was the only id
#: that changed under the advance (independently reproduced against the
#: patched-back v3 constant, which reproduces the pre-0.14.9 digests exactly).
_LIVE_FIXTURE_PROFILE_PLAN_DIGEST: dict[str, str] = {
    "base": "37dbef2bd3c4ecd88126442bfa3b029d635a8eb62b67ecec1f95024a6a044544",
    "structure": "a9b788ae61e7a10d4257e4b1180b2032922617866d300c2adadc273c8728f2ae",
}


def _live_fixture_topology(structure_profile: str) -> dict[str, object]:
    """Compose the exact frozen expectation for *structure_profile* from the
    shared invariants plus that profile's closed delta mapping.  The harness
    only ever selects an expectation from an already-validated profile -- it
    never derives expected counts from actual stats."""
    try:
        deltas = _LIVE_FIXTURE_PROFILE_TOPOLOGY[structure_profile]
    except KeyError:
        raise SmokeFailure(
            f"unknown-structure-profile: {structure_profile!r}"
        ) from None
    merged = dict(_LIVE_FIXTURE_SHARED_TOPOLOGY)
    merged.update(deltas)
    return merged


def _live_fixture_topology_mismatch(
    stats: dict, expected: dict[str, object]
) -> dict[str, tuple[object, object]]:
    """``{key: (actual, expected)}`` for every frozen key whose planned value
    differs.  Empty means the planned topology matches exactly."""
    return {
        key: (stats.get(key, "<missing>"), value)
        for key, value in expected.items()
        if stats.get(key, "<missing>") != value
    }


def _installed_parser_identity() -> tuple[str, str | None, bool]:
    """Bounded probe of the optional syntax parser only (plan section 5.1):
    the production ``PARSER_PACKAGE_VERSION`` identity, the installed
    distribution version (or ``None`` when the distribution is absent), and
    whether the module can be imported.  No unrelated package or environment
    enumeration."""
    from importlib import util as importlib_util

    from codedoc.parser.tree_sitter_structure import PARSER_PACKAGE_VERSION

    try:
        distribution_version: str | None = importlib_metadata.version(
            _PARSER_DISTRIBUTION_NAME
        )
    except importlib_metadata.PackageNotFoundError:
        distribution_version = None
    try:
        module_importable = (
            importlib_util.find_spec("tree_sitter_language_pack") is not None
        )
    except ModuleNotFoundError:
        # An import blocker (or a missing parent) surfaces here as an
        # exception rather than a ``None`` spec; either way the module is not
        # available to this process.
        module_importable = False
    return PARSER_PACKAGE_VERSION, distribution_version, module_importable


def _verify_structure_profile(structure_profile: str) -> dict[str, object]:
    """Fail closed with a stable :class:`SmokeFailure` unless the running
    environment actually matches *structure_profile*, before any fixture
    planning (plan section 5.1):

    * ``base`` requires ``tree-sitter-language-pack`` absent -- no importable
      module, no distribution metadata -- and production
      ``PARSER_PACKAGE_VERSION`` reporting ``not-installed``;
    * ``structure`` requires the pinned distribution version and the
      production parser identity both reporting exactly that version.

    Records only the parser distribution name, the requested profile, and the
    normalized parser identity.
    """
    if structure_profile not in _STRUCTURE_PROFILES:
        raise SmokeFailure(f"unknown-structure-profile: {structure_profile!r}")
    parser_identity, distribution_version, module_importable = (
        _installed_parser_identity()
    )
    facts: dict[str, object] = {
        "parser_distribution": _PARSER_DISTRIBUTION_NAME,
        "structure_profile": structure_profile,
        "parser_identity": parser_identity,
    }
    if structure_profile == "base":
        if (
            parser_identity != "not-installed"
            or distribution_version is not None
            or module_importable
        ):
            raise SmokeFailure(
                f"structure-profile-base-requires-absent-parser: {facts}"
            )
    else:  # "structure"
        if (
            parser_identity != _STRUCTURE_PROFILE_PARSER_VERSION
            or distribution_version != _STRUCTURE_PROFILE_PARSER_VERSION
            or not module_importable
        ):
            raise SmokeFailure(
                f"structure-profile-requires-pinned-parser: {facts}"
            )
    return facts


def _load_frozen_live_fixture() -> bytes:
    """Read the tracked live-validation fixture from the checkout and verify
    its exact byte length and SHA-256 before any use."""
    import hashlib

    path = _repository_root().joinpath(*_LIVE_FIXTURE_REL_PARTS)
    if not path.is_file():
        raise SmokeFailure("live-fixture-missing")
    raw = path.read_bytes()
    if len(raw) != _LIVE_FIXTURE_BYTES:
        raise SmokeFailure(
            f"live-fixture-size-mismatch: {len(raw)} != {_LIVE_FIXTURE_BYTES}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _LIVE_FIXTURE_SHA256:
        raise SmokeFailure(f"live-fixture-hash-mismatch: {digest}")
    return raw


def _scenario_live_fixture_dry_run(root: Path, structure_profile: str) -> None:
    """Plan sections 5.1-5.3: the frozen live-validation fixture, planned
    through the installed product's normal config-file loading path with the
    ``response_correction_enabled`` key absent, resolves the exact per-profile
    split dry-run topology for *structure_profile* ({base,structure}).

    The environment is verified against the requested profile *before* any
    fixture planning, so a profile/package mismatch fails closed rather than
    silently certifying the wrong topology.  Beyond the frozen counts the
    scenario proves, for the selected profile: correction defaults on, no
    provider or output/recovery is produced, planning is byte/digest
    deterministic across repeats, the two profiles' plan digests are distinct,
    and the canonical source reconstructs exactly once from the ordered chunk
    payloads.
    """
    _verify_structure_profile(structure_profile)
    expected = _live_fixture_topology(structure_profile)
    call_cap = expected["max_planned_calls"]

    raw = _load_frozen_live_fixture()
    project = root / f"live-fixture-dry-run-{structure_profile}"
    project.mkdir()
    project.joinpath("main.py").write_bytes(raw)
    config = _write_config(
        project,
        large_file_strategy="split",
        max_content_chars=1000,
        allow_partial=False,
        dry_run=True,
        # The ONLY configuration difference between the two profile
        # invocations (plan section 5.2): the positive cap the frozen
        # topology requires.
        max_planned_calls=call_cap,
    )
    config_text = project.joinpath("codedoc.config.json").read_text(encoding="utf-8")
    if "response_correction_enabled" in config or "response_correction_enabled" in config_text:
        raise SmokeFailure("live-fixture-config-pins-correction-key")

    before = _snapshot(project)
    # Empty overrides: the resolved configuration comes only from the file plus
    # DEFAULTS, so the absent response-correction key genuinely exercises the
    # default-on resolution.  ``forbid_provider`` turns any provider
    # construction into an immediate failure.
    stats, captured = _run_in_process(project, {}, forbid_provider=True)
    stats_again, _repeat_captured = _run_in_process(
        project, {}, forbid_provider=True
    )

    if stats.get("response_correction_enabled") is not True:
        raise SmokeFailure(
            "live-fixture-correction-default-not-enabled: "
            f"{stats.get('response_correction_enabled')!r}"
        )

    mismatch = _live_fixture_topology_mismatch(stats, expected)
    if mismatch:
        raise SmokeFailure(
            f"live-fixture-dry-run-topology-mismatch[{structure_profile}]: {mismatch}"
        )

    frozen_digest = _LIVE_FIXTURE_PROFILE_PLAN_DIGEST[structure_profile]
    if stats.get("call_manifest_digest") != frozen_digest:
        raise SmokeFailure(
            f"live-fixture-dry-run-plan-digest-mismatch[{structure_profile}]: "
            f"{stats.get('call_manifest_digest')!r} != {frozen_digest!r}"
        )
    other_profile = "structure" if structure_profile == "base" else "base"
    if _LIVE_FIXTURE_PROFILE_PLAN_DIGEST[other_profile] == frozen_digest:
        raise SmokeFailure("live-fixture-profile-plan-digests-not-distinct")
    if (
        stats_again.get("call_manifest_digest") != frozen_digest
        or _live_fixture_topology_mismatch(stats_again, expected)
    ):
        raise SmokeFailure(
            f"live-fixture-dry-run-nondeterministic[{structure_profile}]"
        )

    # Plan section 5.3: the canonical decoded source reconstructs exactly once
    # from the ordered chunks.  `build_division_plan` itself enforces the
    # ordered, gap-free, complete, non-overlapping partition and the per-chunk
    # source ceiling; the join is the "reconstructs exactly once" check, and
    # the count is tied back to the profile's frozen chunk total.
    from codedoc.core.file_division import build_division_plan

    decoded = raw.decode("utf-8")
    division = build_division_plan(
        rel_path="main.py",
        language="python",
        content=decoded,
        source_budget_chars=1000,
    )
    if "".join(chunk.payload for chunk in division.chunks) != decoded:
        raise SmokeFailure(
            f"live-fixture-chunks-do-not-reconstruct-source[{structure_profile}]"
        )
    if len(division.chunks) != expected["split_chunks"]:
        raise SmokeFailure(
            f"live-fixture-chunk-count-mismatch[{structure_profile}]: "
            f"{len(division.chunks)} != {expected['split_chunks']}"
        )
    if any(not chunk.payload for chunk in division.chunks):
        raise SmokeFailure(f"live-fixture-empty-chunk[{structure_profile}]")
    if any(chunk.payload_chars > 1000 for chunk in division.chunks):
        raise SmokeFailure(
            f"live-fixture-chunk-over-source-ceiling[{structure_profile}]"
        )

    if _snapshot(project) != before:
        raise SmokeFailure("live-fixture-dry-run-mutated-project")
    if project.joinpath("docs", "codedoc.json").exists():
        raise SmokeFailure("live-fixture-dry-run-wrote-output")
    if project.joinpath("docs", "crash_recovery.json").exists():
        raise SmokeFailure("live-fixture-dry-run-wrote-recovery")

    fixture_markers = (
        "merge_resolved_configuration",
        "provider: str | None = None, model: str | None = None",
    )
    if any(marker in captured for marker in fixture_markers):
        raise SmokeFailure("live-fixture-source-leaked-into-diagnostics")
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


#: Public source ceiling the imports-only scenario divides its fixture at.
#: Production split planning then carries the *automatic* internal synthesis
#: budget ``max(source_ceiling, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS)`` (plan
#: sections 4.2.1 / 5.3.1); the comparison reduction tree must be built from
#: that carried budget through the preferred ``synthesis_manifest_chars``
#: keyword, never the deprecated ``max_content_chars`` alias -- otherwise it
#: models a reducer level production never checkpointed and the recovered
#: ``final.child_ids`` lookup raises ``KeyError``.
_IMPORTS_ONLY_SOURCE_BUDGET_CHARS = 2000


def _scenario_imports_only(root: Path) -> None:
    """Prove the installed planner schedules only final synthesis when the
    parser-derived imports tuple changes while frozen source bytes do not.

    Parser-derived imports normally follow source bytes, so changing an import
    statement is a source-hash change and correctly invalidates every node.
    This scenario isolates the imports identity exactly as the contract does:
    create live leaf/reducer checkpoints for unchanged source, add the
    corresponding old-import final checkpoint, then validate and plan against
    a different same-length imports tuple.

    The comparison reduction tree, its final manifest, and recovery validation
    all consume the same carried automatic synthesis budget production uses --
    never the public source ceiling -- so the locally derived node identities
    match the topology the product actually checkpoints.
    """
    from codedoc.core.document import read_codedoc_document
    from codedoc.core.execution_model import build_call_manifest
    from codedoc.core.file_division import (
        MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS,
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

    source_budget_chars = _IMPORTS_ONLY_SOURCE_BUDGET_CHARS
    # The run's single effective split-synthesis manifest budget, computed
    # exactly as production planning computes it (plan section 5.7).
    synthesis_manifest_chars = max(
        source_budget_chars, MIN_SPLIT_SYNTHESIS_MANIFEST_CHARS
    )

    project = root / "imports-only"
    project.mkdir()
    source = project / "main.py"
    source_text = _large_source()
    source.write_text(source_text, encoding="utf-8")
    config = _write_config(
        project,
        large_file_strategy="split",
        max_content_chars=source_budget_chars,
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
        # The source ceiling: production divides source at exactly this value.
        source_budget_chars=source_budget_chars,
    )
    before_imports = ("alpha",)
    after_imports = ("bravo",)
    tree = build_reduction_tree(
        division,
        # The carried automatic synthesis budget, exactly as production
        # planning passes it -- not the deprecated max_content_chars alias.
        synthesis_manifest_chars=synthesis_manifest_chars,
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
        # Production bounds the final manifest by the reduction tree's own
        # carried synthesis budget (codedoc.core.execution), never a source
        # ceiling.
        max_chars=tree.synthesis_manifest_chars,
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
        # No source/synthesis ceiling argument: validate_recovered_tree
        # recomputes the final manifest bounded by tree.synthesis_manifest_chars
        # (plan section 5.7), never a separately supplied ceiling.
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


#: The canonical scenario sequence ``--scenario all`` runs, in this exact
#: order (plan sections 5.3.1 / 9.2 / 12). Each scenario call in ``_run_all``
#: below is followed by a ``[scenario] reached N/10: <name>`` marker so a
#: completed run is auditable scenario by scenario, and this tuple is the
#: independent authority a swap / deletion / duplication is checked against.
_CANONICAL_SCENARIO_ORDER: tuple[str, ...] = (
    "_scenario_truncate",
    "_scenario_fresh_split",
    "_scenario_signature_bound",
    "_scenario_live_fixture_dry_run",
    "_scenario_redirected_verbose",
    "_scenario_completed_reuse",
    "_scenario_interrupt_resume",
    "_scenario_imports_only",
    "_scenario_preserve_first",
    "_scenario_exit_fidelity",
)


def _run_all(candidate_version: str, structure_profile: str) -> int:
    _package_path, console_path = _prove_installed_origin(candidate_version)
    # Fail closed before any scenario if the environment does not actually
    # match the certified profile (plan section 5.1).
    _verify_structure_profile(structure_profile)
    total = len(_CANONICAL_SCENARIO_ORDER)

    def _reached(index: int, name: str) -> None:
        print(f"[scenario] reached {index}/{total}: {name}")

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="codedoc-installed-smoke-") as temp_name:
        neutral_root = Path(temp_name).resolve()
        if _is_within(neutral_root, _repository_root()):
            raise SmokeFailure("neutral-project-inside-repository")
        try:
            os.chdir(neutral_root)
            _scenario_truncate(neutral_root)
            _reached(1, "truncate")
            _scenario_fresh_split(neutral_root)
            _reached(2, "fresh_split")
            _scenario_signature_bound(neutral_root)
            _reached(3, "signature_bound")
            _scenario_live_fixture_dry_run(neutral_root, structure_profile)
            _reached(4, "live_fixture_dry_run")
            _scenario_redirected_verbose(neutral_root)
            _reached(5, "redirected_verbose")
            _scenario_completed_reuse(neutral_root)
            _reached(6, "completed_reuse")
            _scenario_interrupt_resume(neutral_root)
            _reached(7, "interrupt_resume")
            _scenario_imports_only(neutral_root)
            _reached(8, "imports_only")
            _scenario_preserve_first(neutral_root)
            _reached(9, "preserve_first")
            _scenario_exit_fidelity(neutral_root, console_path)
            _reached(10, "exit_fidelity")
        finally:
            os.chdir(original_cwd)
    print(f"installed artifact smoke: ok ({structure_profile} profile)")
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
            "Required for --scenario all and --scenario cross-version."
        ),
    )
    parser.add_argument(
        "--structure-profile",
        choices=list(_STRUCTURE_PROFILES),
        help=(
            "Which installation profile to certify for --scenario all: "
            "'base' (no optional syntax parser) or 'structure' (pinned parser "
            "extra). Required for --scenario all. Release-harness selector "
            "only -- never a public CodeDoc CLI option or configuration key."
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
    if args.candidate_version is None:
        raise SmokeFailure("scenario-all-requires-candidate-version")
    if args.structure_profile is None:
        raise SmokeFailure("scenario-all-requires-structure-profile")
    return _run_all(args.candidate_version, args.structure_profile)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as exc:
        print(f"installed artifact smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
