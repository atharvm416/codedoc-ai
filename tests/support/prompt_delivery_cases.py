"""Shared test support extracted from mapped source modules."""

import hashlib
from pathlib import Path

from codedoc.core.execution_model import AgentCallContext, FileExecutionRequest
from codedoc.core.prompt_profiles import (
    ResolvedProfile,
    validate_profile,
)

EXTS = frozenset({".py", ".java"})

def _request(
    resolved,
    content="x = 1",
    *,
    path="a.py",
    language="python",
    imports=(),
    mode="single",
    max_content_chars=12000,
    truncation_head_ratio=0.70,
) -> FileExecutionRequest:
    """Build a FileExecutionRequest carrying *resolved*'s bundle for *path*.

    ``resolved=None`` means no profile — mirrors how planning.py falls back to
    ``ResolvedProfile(mode, None)`` when no profile was resolved for the run.
    """
    effective = resolved if resolved is not None else ResolvedProfile(mode, None)
    bundle = effective.resolve_bundle(effective.scope_for({"rel_path": path}))
    return FileExecutionRequest(
        rel_path=path,
        absolute_path=Path(path).resolve(),
        language=language,
        imports=tuple(imports),
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        context=AgentCallContext(
            analysis_mode=mode,
            max_content_chars=max_content_chars,
            truncation_head_ratio=truncation_head_ratio,
            resolved_shape_bundle=bundle,
        ),
    )

def _profile(raw, mode="single"):
    return ResolvedProfile(
        mode,
        validate_profile(
            _to_envelope(raw), active_mode=mode, known_extensions=EXTS,
            source="inline", source_path=None,
        ),
    )

def _to_envelope(raw):
    """Wrap a legacy flat profile dict into the ``common`` envelope."""
    if not isinstance(raw, dict):
        return raw
    out = {k: v for k, v in raw.items() if k in ("schema_version", "$comment")}
    if isinstance(raw.get("single"), dict) and "common" not in raw["single"]:
        sec = raw["single"]
        common = {k: sec[k] for k in ("fields", "requested_shape") if k in sec}
        new_sec = {"common": common}
        if "per_extension" in sec:
            new_sec["per_extension"] = sec["per_extension"]
        out["single"] = new_sec
    elif "single" in raw:
        out["single"] = raw["single"]
    if isinstance(raw.get("triple"), dict) and "common" not in raw["triple"]:
        sec = raw["triple"]
        agents = ("structure", "dependency", "documentation")
        common, per_ext = {}, {}
        for agent in agents:
            block = sec.get(agent, {})
            common[agent] = {k: block[k] for k in ("fields", "requested_shape") if k in block}
            for ext, ov in (block.get("per_extension") or {}).items():
                per_ext.setdefault(ext, {})[agent] = ov
        new_sec = {"common": common}
        if per_ext:
            new_sec["per_extension"] = per_ext
        out["triple"] = new_sec
    elif "triple" in raw:
        out["triple"] = raw["triple"]
    return out
