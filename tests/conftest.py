"""Test session setup.

Redirect the temp ROOT into the repository (``.pyt_tmp``) before pytest computes
its base temp directory. This avoids depending on the system temp location
(e.g. Windows ``%TEMP%\\pytest-of-<user>``), which on some machines becomes
permission-locked and makes the whole suite unrunnable.

We deliberately redirect the *root* rather than pinning ``--basetemp`` to a
fixed path: pytest then creates garbage-collected numbered subdirectories
(``pytest-0``, ``pytest-1`` ...) under it and never force-removes a single fixed
directory, so a locked leftover only produces a warning instead of erroring the
run. This addresses the observed locked-system-temp failure; it is not a claim
that every possible environment is covered.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parent.parent / ".pyt_tmp"
_TMP_ROOT.mkdir(exist_ok=True)

# Set before pytest's tmp_path_factory first computes getbasetemp().
os.environ["TMP"] = str(_TMP_ROOT)
os.environ["TEMP"] = str(_TMP_ROOT)
os.environ["TMPDIR"] = str(_TMP_ROOT)
# Force tempfile to recompute against the new environment.
tempfile.tempdir = str(_TMP_ROOT)


@pytest.fixture(autouse=True)
def _attest_injected_pipeline_providers(monkeypatch):
    """Give network-free pipeline doubles the explicit factory attestation.

    Production verification is fail-closed. Tests replace
    ``codedoc.pipeline.create_provider`` extensively, so this test-only
    boundary attaches the resolved non-secret descriptor when a double has no
    descriptor of its own. A deliberately mismatched valid descriptor is left
    untouched and is still rejected by the production verifier.
    """
    from codedoc import pipeline
    from codedoc.core.file_division import verify_provider_execution_identity
    from codedoc.llm.factory import (
        attest_provider_execution,
        constructed_provider_execution,
    )

    def _verify(provider, config, planned_identity):
        if constructed_provider_execution(provider) is None:
            attest_provider_execution(provider, config)
        return verify_provider_execution_identity(
            provider,
            config,
            planned_identity,
        )

    monkeypatch.setattr(
        pipeline,
        "verify_provider_execution_identity",
        _verify,
    )


@pytest.fixture(autouse=True)
def _enable_future_split_recovery(request):
    if request.node.get_closest_marker("future_split_recovery") is None:
        yield
        return
    from codedoc.core import release_policy

    previous = release_policy.CURRENT_SPLIT_RELEASE
    release_policy.CURRENT_SPLIT_RELEASE = release_policy.SplitReleasePolicy(
        execution=True,
        completed_reuse=True,
        partial_recovery=True,
        node_checkpoints=True,
    )
    try:
        yield
    finally:
        release_policy.CURRENT_SPLIT_RELEASE = previous
