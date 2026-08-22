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


def project_public_symbols(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    projected: list[dict] = []
    seen: set[str] = set()
    for item in value:
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
        key = json.dumps(
            symbol,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
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
) -> dict:
    """Assemble cleaned model fields with deterministic file identity fields."""
    description = cleaned.get("description", "")
    role = cleaned.get("role_in_system", "")
    functions = project_public_symbols(cleaned.get("functions", []))
    classes = project_public_symbols(cleaned.get("classes", []))
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
