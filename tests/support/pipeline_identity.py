"""Shared test support extracted from mapped source modules."""

from codedoc.core.record_meta import (
    expected_analysis_identity,
    expected_ordinary_path_identity,
)


def _prior_run_identity(rel_path: str = "main.py") -> dict:
    """Cache-identity keys a prior CodeDoc run would have persisted for *rel_path*.

    Includes ``_ordinary_path_identity`` (0.14.4) so a fixture record built
    with this helper is same-path reusable; ordinary cross-path reuse is
    refused regardless of this key.
    """
    return {
        **expected_analysis_identity("single"),
        "_ordinary_path_identity": expected_ordinary_path_identity(rel_path),
    }


# Backward-compatible default for every existing "main.py"-path call site.
_PRIOR_RUN_IDENTITY = _prior_run_identity("main.py")
