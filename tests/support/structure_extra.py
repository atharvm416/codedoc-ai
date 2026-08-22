"""Skip marker for tests that require the optional pinned syntax extra.

Two distinct test populations exercise the optional
``tree-sitter-language-pack`` boundary, and they must never be conflated:

- **Absence-simulation tests** monkeypatch the package away and assert the
  complete lexical fallback.  They are the supported base-install contract and
  must run — and pass — in every environment.  They never use this marker.
- **Presence-dependent tests** assert syntax-mode extraction results
  (``structural_mode == "syntax"``, grammar-derived symbols, packing shapes
  that depend on declaration atoms).  Without the installed pack those
  assertions describe an environment that intentionally does not exist, so
  they skip with this named reason instead of failing.

The ``dev`` extra pins and installs the pack, so the release-verification
gates always execute the presence-dependent tests; the skip exists for base
installs, minimal CI legs, and downstream packagers running the suite without
optional extras (D15).
"""

from __future__ import annotations

import importlib.util

import pytest

STRUCTURE_PACK_INSTALLED = (
    importlib.util.find_spec("tree_sitter_language_pack") is not None
)

requires_structure_pack = pytest.mark.skipif(
    not STRUCTURE_PACK_INSTALLED,
    reason=(
        "requires the optional 'structure' extra (tree-sitter-language-pack); "
        "base installs use complete lexical fallback, which the "
        "absence-simulation tests cover without this package"
    ),
)
