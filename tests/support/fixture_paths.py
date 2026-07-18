"""Canonical paths for repository-owned test fixtures."""

from pathlib import Path

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT_FIXTURES = FIXTURES_ROOT / "projects"
GOLDEN_DOCUMENT_FIXTURES = FIXTURES_ROOT / "documents" / "golden"
LEGACY_DOCUMENT_FIXTURES = FIXTURES_ROOT / "documents" / "legacy"
SOURCE_FIXTURES = FIXTURES_ROOT / "sources"
