"""Shared test support extracted from mapped source modules."""

from __future__ import annotations

import hashlib
from pathlib import Path

from codedoc.core.execution_model import AgentCallContext, FileExecutionRequest
from codedoc.core.prompt_profiles import FileScope, ResolvedProfile


def make_execution_request(
    tmp_path: Path,
    rel_path: str,
    content: str = "x = 1\n",
    *,
    language: str = "python",
    imports: tuple[str, ...] = (),
    analysis_mode: str = "single",
    max_content_chars: int = 12000,
    truncation_head_ratio: float = 0.70,
    write: bool = True,
) -> FileExecutionRequest:
    """Build one frozen ``FileExecutionRequest`` for direct orchestrator/
    execution unit tests that do not go through ``build_pipeline_plan()``.

    Writes *content* to ``tmp_path / rel_path`` (unless *write* is False, for a
    request whose backing file a test deletes or never creates) and derives
    ``content_hash`` from those same bytes, exactly as
    ``codedoc.core.db.read_source_snapshot`` would.
    """
    absolute = tmp_path / rel_path
    if write:
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_text(content, encoding="utf-8", newline="")
    bundle = ResolvedProfile(analysis_mode, None).resolve_bundle(
        FileScope(basename=Path(rel_path).name.lower())
    )
    return FileExecutionRequest(
        rel_path=rel_path,
        language=language,
        imports=tuple(imports),
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        context=AgentCallContext(
            analysis_mode=analysis_mode,
            max_content_chars=max_content_chars,
            truncation_head_ratio=truncation_head_ratio,
            resolved_shape_bundle=bundle,
        ),
    )


def make_execution_requests(
    tmp_path: Path,
    rel_paths: list[str],
    content: str = "x = 1\n",
    **kwargs,
) -> list[FileExecutionRequest]:
    """Build one ``FileExecutionRequest`` per name in *rel_paths*."""
    return [
        make_execution_request(tmp_path, rel_path, content, **kwargs)
        for rel_path in rel_paths
    ]
