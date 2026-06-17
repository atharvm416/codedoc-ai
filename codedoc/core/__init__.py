"""Core codedoc components."""

__all__ = [
    "load_config",
    "scan_files",
    "detect_entry_file",
    "ProcessingQueue",
    "DependencyGraph",
    "write_summary",
    "json_from_markdown",
    "markdown_from_json",
    "Checkpoint",
    "SafeWriter",
]


def __getattr__(name: str):
    if name == "load_config":
        from codedoc.core.loader import load_config

        return load_config
    if name in {"scan_files", "detect_entry_file"}:
        from codedoc.core.scanner import detect_entry_file, scan_files

        return {"scan_files": scan_files, "detect_entry_file": detect_entry_file}[name]
    if name == "ProcessingQueue":
        from codedoc.core.queue import ProcessingQueue

        return ProcessingQueue
    if name == "DependencyGraph":
        from codedoc.core.graph import DependencyGraph

        return DependencyGraph
    if name == "write_summary":
        from codedoc.core.output import write_summary

        return write_summary
    if name in {"json_from_markdown", "markdown_from_json"}:
        from codedoc.core.markdown_view import json_from_markdown, markdown_from_json

        return {
            "json_from_markdown": json_from_markdown,
            "markdown_from_json": markdown_from_json,
        }[name]
    if name == "Checkpoint":
        from codedoc.core.checkpoint import Checkpoint

        return Checkpoint
    if name == "SafeWriter":
        from codedoc.core.safe_writer import SafeWriter

        return SafeWriter
    raise AttributeError(name)
