"""Deterministic assembly helpers for the public flat file record."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Mapping, Sequence

MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND = 12
MAX_PUBLIC_SYMBOL_NAME_CHARS = 128
MAX_PUBLIC_SYMBOL_DESCRIPTION_CHARS = 300
MAX_PUBLIC_EXPORT_ITEMS = 32
MAX_PUBLIC_EXPORT_ITEM_CHARS = 256


def _public_text(value: object, maximum: int) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned[:maximum] if cleaned else None


def project_public_symbols(
    value: object, *, collapse_identical: bool = True
) -> list[dict]:
    """Project a symbol list to the public ``{name, description?}`` shape.

    ``collapse_identical`` (the default) drops a later item byte-identical to an
    earlier one -- the right behavior for untrusted model output. The shared
    structural authority passes ``collapse_identical=False`` for its already
    source-backed arrays: two genuine same-name overloads legitimately project
    to identical dicts and must both survive. The per-kind cap and the char
    caps always apply.
    """
    if not isinstance(value, list):
        return []
    projected: list[dict] = []
    seen: set[str] = set()
    for position, item in enumerate(value):
        if not isinstance(item, Mapping):
            continue
        name = _public_text(item.get("name"), MAX_PUBLIC_SYMBOL_NAME_CHARS)
        if name is None:
            continue
        symbol = {"name": name}
        description = _public_text(
            item.get("description"),
            MAX_PUBLIC_SYMBOL_DESCRIPTION_CHARS,
        )
        if description is not None:
            symbol["description"] = description
        value_key = json.dumps(
            symbol,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        key = value_key if collapse_identical else f"{position}\x00{value_key}"
        if key in seen:
            continue
        if len(projected) >= MAX_PUBLIC_SYMBOL_ITEMS_PER_KIND:
            break
        seen.add(key)
        projected.append(symbol)
    return projected


def project_public_exports(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    projected: list[str] = []
    seen: set[str] = set()
    for item in value:
        export = _public_text(item, MAX_PUBLIC_EXPORT_ITEM_CHARS)
        if export is None or export in seen:
            continue
        if len(projected) >= MAX_PUBLIC_EXPORT_ITEMS:
            break
        seen.add(export)
        projected.append(export)
    return projected


def extension_of(rel_path: str) -> str:
    """Return the scanner-equivalent lower-cased extension for *rel_path*."""
    return PurePosixPath(rel_path).suffix.lower()


def flat_combined_result(
    file_path: str,
    language: str,
    imports: Sequence[str],
    cleaned: Mapping[str, object],
    *,
    reconciled_structure: bool = False,
) -> dict:
    """Assemble cleaned model fields with deterministic file identity fields.

    ``reconciled_structure`` marks that ``functions`` / ``classes`` already came
    from the shared source-backed authority, so their multiplicity (two genuine
    same-name overloads) is preserved through projection instead of being
    collapsed as if it were untrusted model output.
    """
    description = cleaned.get("description", "")
    role = cleaned.get("role_in_system", "")
    functions = project_public_symbols(
        cleaned.get("functions", []), collapse_identical=not reconciled_structure
    )
    classes = project_public_symbols(
        cleaned.get("classes", []), collapse_identical=not reconciled_structure
    )
    exports = project_public_exports(cleaned.get("exports", []))
    dependencies_analysis = cleaned.get("dependencies_analysis", {})
    key_concepts = cleaned.get("key_concepts", [])
    usage_example = cleaned.get("usage_example", "")
    return {
        "file_path": file_path,
        "language": language,
        "extension": extension_of(file_path),
        "imports": list(imports),
        "description": description,
        "role_in_system": role,
        "functions": functions,
        "classes": classes,
        "exports": exports,
        "structure": {
            "description": description,
            "role_in_system": role,
            "functions": functions,
            "classes": classes,
            "exports": exports,
        },
        "dependencies_analysis": dependencies_analysis,
        "key_concepts": key_concepts,
        "usage_example": usage_example,
        "documentation": {
            "description": description,
            "role_in_system": role,
            "key_concepts": key_concepts,
            "usage_example": usage_example,
        },
        "state": "checked",
    }
