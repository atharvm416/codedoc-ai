"""Core codedoc components."""

__all__ = [
    "load_config",
    "scan_files",
    "detect_entry_file",
    "ProcessingQueue",
    "DependencyGraph",
    "write_outputs",
    "write_summary",
    "CodeDocDB",
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
    if name in {"write_outputs", "write_summary"}:
        from codedoc.core.output import write_outputs, write_summary

        return {"write_outputs": write_outputs, "write_summary": write_summary}[name]
    if name == "CodeDocDB":
        from codedoc.core.db import CodeDocDB

        return CodeDocDB
    raise AttributeError(name)
