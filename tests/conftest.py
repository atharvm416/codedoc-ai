"""Test session setup.

Redirect the temp ROOT into the repository (``.pyt_tmp``) before pytest computes
its base temp directory. This avoids depending on the system temp location
(e.g. Windows ``%TEMP%\\pytest-of-<user>``), which on some machines becomes
permission-locked and makes the whole suite unrunnable.

We deliberately redirect the *root* rather than pinning ``--basetemp`` to a
fixed path: pytest then creates garbage-collected numbered subdirectories
(``pytest-0``, ``pytest-1`` ...) under it and never force-removes a single fixed
directory, so a locked leftover only produces a warning instead of erroring the
run. This addresses the observed locked-system-temp failure; it is not a claim
that every possible environment is covered.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP_ROOT = Path(__file__).resolve().parent.parent / ".pyt_tmp"
_TMP_ROOT.mkdir(exist_ok=True)

# Set before pytest's tmp_path_factory first computes getbasetemp().
os.environ["TMP"] = str(_TMP_ROOT)
os.environ["TEMP"] = str(_TMP_ROOT)
os.environ["TMPDIR"] = str(_TMP_ROOT)
# Force tempfile to recompute against the new environment.
tempfile.tempdir = str(_TMP_ROOT)
