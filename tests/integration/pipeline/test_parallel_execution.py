"""Tests organized by feature ownership."""

from __future__ import annotations

import json

def test_pipeline_processes_files_with_bounded_parallelism(tmp_path, monkeypatch):
    import threading
    import time

    from codedoc.pipeline import run_pipeline

    for index in range(6):
        imports = ""
        if index == 0:
            imports = "".join(f"import file_{dep}\n" for dep in range(1, 6))
        (tmp_path / f"file_{index}.py").write_text(
            f"{imports}def func_{index}():\n    return {index}\n",
            encoding="utf-8",
        )

    class SlowProvider:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        @property
        def provider_name(self):
            return "SlowProvider"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                if "key_concepts" in prompt:
                    return json.dumps(
                        {
                            "description": "Documented file.",
                            "role_in_system": "Test role.",
                            "key_concepts": [],
                            "usage_example": "",
                        }
                    )
                if "dependencies_analysis" in prompt:
                    return json.dumps(
                        {
                            "dependencies_analysis": {
                                "internal": [],
                                "external": [],
                                "dependency_refs": [],
                                "catalog_updates": [],
                                "usage_notes": [],
                                "warnings": [],
                            }
                        }
                    )
                return json.dumps(
                    {
                        "description": "Structured file.",
                        "role_in_system": "Test role.",
                        "functions": [],
                        "classes": [],
                        "exports": [],
                    }
                )
            finally:
                with self.lock:
                    self.active -= 1

    provider = SlowProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "file_0.py",
            "parallel_agents": False,
            "max_parallel_files": 3,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 6
    assert stats["failed"] == 0
    assert 1 < provider.max_active <= 3
    assert (tmp_path / "docs_output" / "codedoc.json").exists()

def test_pipeline_retries_failed_file_before_marking_failed(tmp_path, monkeypatch):

    from codedoc.pipeline import run_pipeline

    source = tmp_path / "main.py"
    source.write_text("def main():\n    return 'ok'\n", encoding="utf-8")

    class FlakyProvider:
        def __init__(self):
            self.calls = 0

        @property
        def provider_name(self):
            return "FlakyProvider"

        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt, system)

        def complete_json(self, prompt, system=""):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary provider outage")
            if "key_concepts" in prompt:
                return json.dumps(
                    {
                        "description": "Recovered file.",
                        "role_in_system": "Recovered role.",
                        "key_concepts": [],
                        "usage_example": "",
                    }
                )
            if "dependencies_analysis" in prompt:
                return json.dumps(
                    {
                        "dependencies_analysis": {
                            "internal": [],
                            "external": [],
                            "dependency_refs": [],
                            "catalog_updates": [],
                            "usage_notes": [],
                            "warnings": [],
                        }
                    }
                )
            return json.dumps(
                {
                    "description": "Recovered structure.",
                    "role_in_system": "Recovered role.",
                    "functions": [],
                    "classes": [],
                    "exports": [],
                }
            )

    provider = FlakyProvider()
    monkeypatch.setattr("codedoc.pipeline.create_provider", lambda config: provider)

    stats = run_pipeline(
        tmp_path,
        {
            "output_dir": "docs_output",
            "output_format": "json",
            "entry_file": "main.py",
            "parallel_agents": False,
            "max_parallel_files": 1,
            "file_retry_attempts": 1,
            "propagate_changes": False,
        },
    )

    assert stats["checked"] == 1
    assert stats["failed"] == 0
    assert provider.calls > 1
    assert "Recovered file." in (
        tmp_path / "docs_output" / "codedoc.json"
    ).read_text(encoding="utf-8")

def test_D8_process_batch_returns_exception_with_descriptor(tmp_path, monkeypatch):
    """D8: retry_rate_limited contains (descriptor, exception) tuples."""
    from codedoc.utils.errors import LLMError
    from codedoc.pipeline import _process_descriptor_batch
    from codedoc.core.queue import ProcessingQueue
    from codedoc.core.safe_writer import SafeWriter

    (tmp_path / "main.py").write_text("x=1\n")
    descriptor = {
        "rel_path": "main.py",
        "path": tmp_path / "main.py",
        "language": "python",
    }

    class RateLimitProvider:
        provider_name = "openai"
        def complete_json(self, prompt, system=""):
            raise LLMError("openai", "429 rate_limit_exceeded tpm")
        def complete(self, prompt, system="", temperature=0.1):
            return self.complete_json(prompt)

    from codedoc.agents.orchestrator import Orchestrator
    orch = Orchestrator(RateLimitProvider(), parallel=False)

    queue = ProcessingQueue()
    queue.add(descriptor)
    stats = {"checked": 0, "failed": 0}
    from codedoc.utils.errors import ErrorReporter
    error_reporter = ErrorReporter()

    backup = tmp_path / "codedoc.json"
    sw = SafeWriter(backup, "json", "main.py", {"main.py": descriptor})
    sw.set_queue_order(["main.py"])

    from codedoc.llm.rate_limit_profile import get_rate_limit_profile
    profile = get_rate_limit_profile("openai")

    succeeded, retry_rate_limited, failed = _process_descriptor_batch(
        [descriptor], orch, queue, stats, error_reporter,
        max_workers=1, recorder=sw, profile=profile,
    )

    assert len(retry_rate_limited) == 1, "Rate-limited file must be in retry list"
    desc_back, exc_back = retry_rate_limited[0]
    assert desc_back is descriptor, "Original descriptor must be returned"
    # The exception may be AgentError wrapping LLMError — verify it carries
    # the rate-limit signal so _is_rate_limit_error can detect it via chain walk.
    assert exc_back is not None, "Exception must be preserved (not None)"
    assert "429" in str(exc_back), (
        f"Rate-limit signal must be somewhere in the exception string: {exc_back!r}"
    )
