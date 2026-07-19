"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

def _record(path, language, *, external=None, imports=None, catalog_updates=None,
            dependency_refs=None, usage_notes=None):
    # 0.10.1 (Workstream F): public external/sdk links are projected from the
    # parser ``imports``, not from the model ``dependencies_analysis.external``.
    # These tests express the parser-found dependency names; by default the
    # ``external`` argument *is* the parser import list so each case states the
    # deterministic input once.  ``dependencies_analysis.external`` is still set
    # so the (now bounded) model-enrichment path is exercised, but it can no
    # longer create a public link on its own.
    deps = {}
    if external is not None:
        deps["external"] = external
    if catalog_updates is not None:
        deps["catalog_updates"] = catalog_updates
    if dependency_refs is not None:
        deps["dependency_refs"] = dependency_refs
    if usage_notes is not None:
        deps["usage_notes"] = usage_notes
    parser_imports = imports if imports is not None else (external or [])
    return {
        "hash": f"h-{path}",
        "file_path": path,
        "language": language,
        "documentation": {
            "file_path": path,
            "language": language,
            "description": "d",
            "imports": parser_imports,
            "dependencies_analysis": deps,
        },
    }
