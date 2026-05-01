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
DB_VERSION = 2


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
            data.setdefault("version", DB_VERSION)
            data.setdefault("history", [])
            return data

        # Backwards compatibility for v1, where rel_path entries lived at root.
        return {
            "version": DB_VERSION,
            "files": data,
            "history": [],
        }

    def _empty(self) -> dict:
        return {"version": DB_VERSION, "files": {}, "history": []}

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
        if not entry or entry.get("hash") != current_hash:
            return True
        logger.debug("Skipping unchanged file: %s", rel_path)
        return False

    def mark_processed(self, rel_path: str, file_path: Path, result: dict | None = None) -> None:
        """Mark a file as successfully processed and record useful memory."""
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
        }

        self.data["files"][rel_path] = {
            "hash": current_hash,
            "last_processed": now,
            "git_commit": commit,
            "author": author,
            "language": result.get("language", ""),
            "imports": result.get("imports", []),
            "description": result.get("description", ""),
            "role_in_system": result.get("role_in_system", ""),
            "key_concepts": result.get("key_concepts", []),
        }
        self.data["history"].append(change_event)
        self.data["history"] = self.data["history"][-500:]
        self._save()


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
