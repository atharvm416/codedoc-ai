"""codedoc: local-first, LLM-agnostic codebase documentation."""

__version__ = "0.11.5"
__author__ = "Atharv Mannur"


def run_pipeline(*args, **kwargs):
    """Run codedoc's documentation pipeline.

    Imported lazily so ``import codedoc`` stays lightweight and does not require
    optional console/runtime dependencies until the pipeline is actually used.
    """
    from codedoc.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = ["run_pipeline", "__version__"]
