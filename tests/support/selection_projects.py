"""Shared test support extracted from mapped source modules."""

from __future__ import annotations
from codedoc.core.graph import DependencyGraph

def _graph_and_map(tmp_path):
    graph = DependencyGraph()
    file_map = {}
    for rel in ("main.py", "helper.py", "orphan.py"):
        graph.add_file(rel)
        file_map[rel] = {
            "path": tmp_path / rel,
            "rel_path": rel,
            "language": "python",
            "extension": ".py",
        }
    graph.add_dependency("main.py", "helper.py")
    return graph, file_map

def _project(tmp_path):
    (tmp_path / "main.py").write_text("import helper\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "orphan.py").write_text("y = 2\n", encoding="utf-8")
