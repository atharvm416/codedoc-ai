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


def compute_file_hash(file_path: Path) -> str:
    return _compute_file_hash(file_path)


class CodeDocDB:
    """JSON-backed memory for incremental runs and generated doc metadata."""

    def __init__(self, root: Path, output_dir: Path | None = None) -> None:
        """
        Args:
            root:       Project root — used for git context and relative paths.
            output_dir: Directory where codedoc_db.json will be stored.
                        Defaults to the project root for backward compatibility,
                        but the standard path is inside the output directory so
                        all generated files stay in one place.
        """
        self.root = Path(root).resolve()
        db_dir = Path(output_dir).resolve() if output_dir else self.root
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / DB_FILENAME
        self._migrate_from_root(db_dir)
        self.data: dict = self._load()

    # ------------------------------------------------------------------
    # Migration
    # ------------------------------------------------------------------

    def _migrate_from_root(self, db_dir: Path) -> None:
        """
        One-time migration: if the DB does not yet exist in db_dir but an
        older codedoc_db.json is present at the project root, move it so
        existing incremental caches are not lost after an upgrade.
        """
        if self.db_path.exists():
            return
        old_path = self.root / DB_FILENAME
        if old_path.exists() and old_path != self.db_path:
            try:
                old_path.rename(self.db_path)
                logger.info(
                    "Migrated codedoc_db.json from project root to %s",
                    self.db_path.parent,
                )
            except Exception as exc:
                logger.warning(
                    "Could not migrate codedoc_db.json from project root: %s", exc
                )

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

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

        # Backward compatibility for v1, where rel_path entries lived at root.
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def needs_processing(self, rel_path: str, file_path: Path) -> bool:
        """Return True if the file has changed or was never processed before."""
        current_hash = _compute_file_hash(file_path)
        entry = self.data.get("files", {}).get(rel_path)
        if not entry or entry.get("hash") != current_hash:
            return True
        logger.debug("Skipping unchanged file: %s", rel_path)
        return False

    def get_entry(self, rel_path: str) -> dict:
        return self.data.get("files", {}).get(rel_path, {})

    def mark_processed(self, rel_path: str, file_path: Path, result: dict | None = None) -> None:
        """Mark a file as successfully processed and record useful metadata."""
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
            "reused_from": reused_from,
        }

        # Extract only dependencies_analysis from the result — all other doc
        # fields live in the public JSON output, not the DB.
        deps_analysis = result.get("dependencies_analysis")

        entry: dict = {
            "hash": current_hash,
            "last_processed": now,
            "git_commit": commit,
            "author": author,
        }
        if deps_analysis:
            entry["dependencies_analysis"] = deps_analysis

        self.data.pop("version", None)
        self.data["files"][rel_path] = _prune_empty(entry)
        self.data["history"].append(_prune_empty(change_event))
        self.data["history"] = self.data["history"][-500:]
        self._save()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
