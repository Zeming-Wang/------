"""Preview the real MaAS reproduction RootGraph in MASFactory Visualizer.

This file uses the same graph builder as the executable CLI path. It builds only
the MASFactory topology; it does not initialize runtime objects or call an LLM.
"""

from __future__ import annotations

from maas_reproduction.workflow import build_maas_reproduction_graph


def build_runtime_preview_graph():
    graph = build_maas_reproduction_graph()
    graph.build()
    return graph


graph = build_runtime_preview_graph()
