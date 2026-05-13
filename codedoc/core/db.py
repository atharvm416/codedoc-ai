"""Incremental memory database for codedoc."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from codedoc.utils.logger import get_logger

logger = get_logger(__name__)

DB_FILENAME = "codedoc_db.json"

def _compute_file_hash(file_path: Path) -> str:
    """Compute a SHA256 hash of file content."""
    hash_sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha.update(chunk)
    return hash_sha.hexdigest()


class CodeDocDB:
    """JSON-backed memory for incremental runs and generated doc metadata."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.db_path = self.root / DB_FILENAME
        self.data: dict = self._load()

    def _load(self) -> dict:
        if not self.db_path.exists():
            return self._empty()

        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Invalid codedoc_db.json; starting fresh: %s", exc)
            return self._empty()

        if "files" in data:
            data.setdefault("history", [])
            return data

        # Backwards compatibility for v1, where rel_path entries lived at root.
        return {
            "files": data,
            "history": [],
        }

    def _empty(self) -> dict:
        return {"files": {}, "history": []}

    def _save(self) -> None:
        try:
            self.db_path.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save codedoc_db.json: %s", exc)

    def needs_processing(self, rel_path: str, file_path: Path) -> bool:
        """Return True if the file has changed or was never processed before."""
        current_hash = _compute_file_hash(file_path)
        entry = self.data.get("files", {}).get(rel_path)
        if not entry or entry.get("hash") != current_hash or not _has_documentation(entry):
            return True
        logger.debug("Skipping unchanged file: %s", rel_path)
        return False

    def reuse_by_content_hash(self, descriptor: dict) -> bool:
        """Reuse cached analysis when another file has identical content."""
        rel_path = descriptor["rel_path"]
        file_path = descriptor["path"]
        current_hash = _compute_file_hash(file_path)
        reusable = self._find_result_by_hash(current_hash)
        if not reusable:
            return False

        source_rel, source_entry = reusable
        result = _result_from_entry(source_entry, source_rel)
        result["file_path"] = rel_path
        result["language"] = descriptor.get("language", result.get("language", ""))
        result["extension"] = descriptor.get("extension", result.get("extension", ""))

        logger.info(
            "Reusing cached documentation for %s from identical content in %s",
            rel_path,
            source_rel,
        )
        self._mark_processed(rel_path, file_path, result, reused_from=source_rel)
        return True

    def mark_processed(self, rel_path: str, file_path: Path, result: dict | None = None) -> None:
        """Mark a file as successfully processed and record useful memory."""
        self._mark_processed(rel_path, file_path, result)

    def _mark_processed(
        self,
        rel_path: str,
        file_path: Path,
        result: dict | None = None,
        reused_from: str | None = None,
    ) -> None:
        result = result or {}
        current_hash = _compute_file_hash(file_path)
        previous = self.data["files"].get(rel_path, {})
        now = datetime.now(timezone.utc).isoformat()
        commit = _git_value(self.root, ["git", "rev-parse", "--short", "HEAD"])
        author = (
            _git_value(self.root, ["git", "config", "user.name"])
            or os.environ.get("GIT_AUTHOR_NAME")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
        )

        change_event = {
            "file_path": rel_path,
            "processed_at": now,
            "previous_hash": previous.get("hash"),
            "hash": current_hash,
            "git_commit": commit,
            "author": author,
            "description": result.get("description", ""),
            "reused_from": reused_from,
        }

        cache_result = _compact_result_for_cache(result)
        entry = {
            "id": current_hash,
            "hash": current_hash,
            "format": file_path.suffix.lower().lstrip("."),
            "path": rel_path,
            "last_processed": now,
            "git_commit": commit,
            "author": author,
            "language": result.get("language", ""),
            "imports": result.get("imports", []),
            "description": result.get("description", ""),
            "role_in_system": result.get("role_in_system", ""),
            "key_concepts": result.get("key_concepts", []),
            "result": cache_result,
            "reused_from": reused_from,
        }
        self.data.pop("version", None)
        self.data["files"][rel_path] = _prune_empty(entry)
        self.data["history"].append(_prune_empty(change_event))
        self.data["history"] = self.data["history"][-500:]
        self._save()

    def _find_result_by_hash(self, content_hash: str) -> tuple[str, dict] | None:
        for rel_path, entry in sorted(self.data.get("files", {}).items()):
            if entry.get("hash") == content_hash and _has_documentation(entry):
                return rel_path, entry
        return None

    def documentation_records(
        self,
        rel_paths: set[str],
        file_map: dict[str, dict] | None = None,
        ordered_paths: list[str] | None = None,
    ) -> list[dict]:
        """Return cached documentation records for selected files."""
        file_map = file_map or {}
        paths = ordered_paths or sorted(rel_paths)
        records: list[dict] = []

        for rel_path in paths:
            if rel_path not in rel_paths:
                continue
            entry = self.data.get("files", {}).get(rel_path)
            if not entry or not _has_documentation(entry):
                continue

            descriptor = file_map.get(rel_path, {})
            descriptor_path = descriptor.get("path")
            if descriptor_path and entry.get("hash") != _compute_file_hash(descriptor_path):
                continue

            result = _result_from_entry(entry, rel_path)
            result.setdefault("file_path", rel_path)
            result.setdefault(
                "language",
                entry.get("language", descriptor.get("language", "")),
            )
            result.setdefault("extension", descriptor.get("extension", ""))
            result.setdefault("imports", entry.get("imports", []))
            result.setdefault("description", entry.get("description", ""))
            result.setdefault("role_in_system", entry.get("role_in_system", ""))
            result.setdefault("key_concepts", entry.get("key_concepts", []))

            records.append(
                {
                    "id": entry.get("id") or entry.get("hash", ""),
                    "hash": entry.get("hash", ""),
                    "file_path": rel_path,
                    "format": entry.get("format")
                    or descriptor.get("extension", "").lstrip("."),
                    "language": result.get("language", ""),
                    "last_processed": entry.get("last_processed", ""),
                    "git_commit": entry.get("git_commit"),
                    "author": entry.get("author"),
                    "documentation": result,
                }
            )

        return records


def _git_value(root: Path, command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None

    value = completed.stdout.strip()
    return value or None


def _compact_result_for_cache(result: dict) -> dict:
    """Remove fields already stored on the cache record and empty sections."""
    compact = dict(result or {})
    for key in (
        "file_path",
        "language",
        "extension",
        "imports",
        "description",
        "role_in_system",
        "key_concepts",
        "state",
    ):
        compact.pop(key, None)

    documentation = compact.get("documentation")
    if isinstance(documentation, dict):
        for key in ("description", "role_in_system", "key_concepts", "usage_example"):
            if documentation.get(key) == result.get(key):
                documentation.pop(key, None)
        if not documentation:
            compact.pop("documentation", None)

    dependencies = compact.get("dependencies_analysis")
    if isinstance(dependencies, dict):
        dependencies = _prune_empty(dependencies)
        if dependencies:
            compact["dependencies_analysis"] = dependencies
        else:
            compact.pop("dependencies_analysis", None)

    compact.pop("structure", None)

    return _prune_empty(compact)


def _has_documentation(entry: dict) -> bool:
    return bool(entry.get("result") or entry.get("description") or entry.get("role_in_system"))


def _result_from_entry(entry: dict, rel_path: str) -> dict:
    result = dict(entry.get("result", {}))
    result.setdefault("file_path", rel_path)
    result.setdefault("language", entry.get("language", ""))
    result.setdefault("imports", entry.get("imports", []))
    result.setdefault("description", entry.get("description", ""))
    result.setdefault("role_in_system", entry.get("role_in_system", ""))
    result.setdefault("key_concepts", entry.get("key_concepts", []))
    result.setdefault("state", "checked")
    return _prune_empty(result)


def _prune_empty(value):
    if isinstance(value, dict):
        pruned = {
            key: _prune_empty(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return {
            key: item
            for key, item in pruned.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            item
            for item in (_prune_empty(item) for item in value)
            if item not in (None, "", [], {})
        ]
    return value
