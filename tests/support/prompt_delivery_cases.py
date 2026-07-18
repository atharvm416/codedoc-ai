"""Shared test support extracted from mapped source modules."""

from codedoc.core.prompt_profiles import (
    ResolvedProfile,
    validate_profile,
)

EXTS = frozenset({".py", ".java"})

def _descriptor(path="a.py", language="python"):
    return {"rel_path": path, "language": language, "extension": ".py"}

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
