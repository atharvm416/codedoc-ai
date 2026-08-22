"""Section 3A: Windows redirected verbose logging must be Unicode-safe and
must never change privacy, exit status, provider calls, or recovery.

Windows-only: the PowerShell ``2>&1 | Tee-Object`` subprocess case skips with
the recorded reason "requires Windows PowerShell redirected output" on every
other platform. Platform-neutral logging-privacy assertions live in
``tests/unit/utils/test_logging_isolation.py`` and must never skip anywhere.

The child process runs the repository's own ``codedoc`` package (not an
installed artifact -- that harness is ``tests/contract/package/
installed_artifact_smoke.py``, used only post-build) with a network-free fake
provider that deliberately emits a third-party DEBUG record embedding request
body, prompt, and adversarial non-CP1252 Unicode text, reproducing the exact
P1/P2 conditions from the installed-library finding without any real provider
contact or credential.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.support.logging_sentinels import (
    SENTINEL_AUTHORIZATION_HEADER,
    SENTINEL_PROMPT_FRAGMENT,
    SENTINEL_REQUEST_BODY,
    assert_no_sentinels_leaked,
)

pytestmark = pytest.mark.platform

_REQUIRES_WINDOWS_REASON = "requires Windows PowerShell redirected output"
_REPO_ROOT = Path(__file__).resolve().parents[3]

_CHILD_SCRIPT = textwrap.dedent(
    '''
    import sys
    sys.path.insert(0, r"{repo_root}")

    import codedoc.pipeline as pipeline_mod


    class _FakeThirdPartyDebugProvider:
        provider_name = "fake"

        def complete_json(self, prompt, system=""):
            import logging
            import json

            logging.getLogger("openai").debug(
                "json_data.messages=%s prompt=%s auth=%s unicode=%s",
                {request_body!r},
                {prompt_fragment!r},
                {auth_header!r},
                "\\u65e5\\u672c\\u8a9e \\U0001F600 \\u00e9\\u00e8\\u00ea",
            )
            if "key_concepts" in prompt:
                return json.dumps(
                    {{"description": "d", "role_in_system": "r",
                      "key_concepts": [], "usage_example": ""}}
                )
            if "dependencies_analysis" in prompt:
                return json.dumps(
                    {{"dependencies_analysis": {{
                        "internal": [], "external": [], "dependency_refs": [],
                        "catalog_updates": [], "usage_notes": [], "warnings": [],
                    }}}}
                )
            return json.dumps(
                {{"description": "d", "role_in_system": "r",
                  "functions": [], "classes": [], "exports": []}}
            )

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)


    from codedoc.llm.factory import attest_provider_execution


    def _create_fake_provider(config):
        provider = _FakeThirdPartyDebugProvider()
        attest_provider_execution(provider, config)
        return provider


    pipeline_mod.create_provider = _create_fake_provider

    from codedoc.cli.cli import main

    main(sys.argv[1:])
    '''
)


def _write_child_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "child_run.py"
    script_path.write_text(
        _CHILD_SCRIPT.format(
            repo_root=str(_REPO_ROOT).replace("\\", "\\\\"),
            request_body=SENTINEL_REQUEST_BODY,
            prompt_fragment=SENTINEL_PROMPT_FRAGMENT,
            auth_header=SENTINEL_AUTHORIZATION_HEADER,
        ),
        encoding="utf-8",
    )
    return script_path


def _run_powershell_redirected(script_path: Path, project_dir: Path, log_path: Path):
    # "chcp 1252" forces the legacy single-byte Windows code page before the
    # redirected run, reproducing the exact console condition the P2 finding
    # was observed under (Rich's legacy Windows renderer encoding through
    # CP1252) rather than relying on whichever default this environment has.
    ps_command = (
        "chcp 1252 | Out-Null; "
        f'& "{sys.executable}" "{script_path}" "{project_dir}" '
        "--entry main.py --verbose --no-parallel 2>&1 | "
        f'Tee-Object -FilePath "{log_path}"; exit $LASTEXITCODE'
    )
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_command],
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
        errors="replace",
    )


@pytest.mark.skipif(sys.platform != "win32", reason=_REQUIRES_WINDOWS_REASON)
def test_redirected_verbose_run_is_unicode_safe_and_leak_free(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("x = 1\n", encoding="utf-8")
    log_path = tmp_path / "captured.log"
    script_path = _write_child_script(tmp_path)

    result = _run_powershell_redirected(script_path, project_dir, log_path)

    combined = (result.stdout or "") + (result.stderr or "")
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    )

    # P2: no logging traceback, no UnicodeEncodeError, correct exit status.
    assert "--- Logging error ---" not in combined
    assert "--- Logging error ---" not in log_text
    assert "UnicodeEncodeError" not in combined
    assert "UnicodeEncodeError" not in log_text
    assert result.returncode == 0, combined

    # A clean successful run reaches codedoc complete: stable output written,
    # recovery file removed.
    assert (project_dir / "codedoc" / "codedoc.json").exists()
    assert not (project_dir / "codedoc" / "crash_recovery.json").exists()

    # P1: no raw request/prompt/credential text anywhere in the redirected
    # stream or the captured log file.
    assert_no_sentinels_leaked(combined, log_text)


@pytest.mark.skipif(sys.platform != "win32", reason=_REQUIRES_WINDOWS_REASON)
def test_interrupted_redirected_run_preserves_stable_output_and_recovery(tmp_path):
    """An interrupted (non-zero exit) redirected run must still leave no
    logging traceback and no leaked sentinel, and must preserve whatever
    recovery state existed -- logging/redirection is observational only."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("x = 1\n", encoding="utf-8")
    (project_dir / "other.py").write_text("y = 2\n", encoding="utf-8")
    log_path = tmp_path / "captured.log"
    script_path = _write_child_script(tmp_path)

    # No --entry: both files documented, plenty of chances for a leak. Run
    # once to establish a clean baseline stable output first.
    ps_command = (
        "chcp 1252 | Out-Null; "
        f'& "{sys.executable}" "{script_path}" "{project_dir}" '
        f'--verbose --no-parallel 2>&1 | Tee-Object -FilePath "{log_path}"; '
        "exit $LASTEXITCODE"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_command],
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
        errors="replace",
    )
    combined = (result.stdout or "") + (result.stderr or "")
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    )
    assert "--- Logging error ---" not in combined
    assert "--- Logging error ---" not in log_text
    assert result.returncode == 0, combined
    assert_no_sentinels_leaked(combined, log_text)
    assert (project_dir / "codedoc" / "codedoc.json").exists()
