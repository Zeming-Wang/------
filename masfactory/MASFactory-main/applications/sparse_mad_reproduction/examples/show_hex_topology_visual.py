from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_mad.agent_topology_hex_visual import build_hex_topology_visual_graph, hex_edge_counts


def main() -> None:
    graph = build_hex_topology_visual_graph()
    print(f"Built graph: {graph.name}")
    print(hex_edge_counts())
    print("Open sparse_mad/agent_topology_hex_visual.py with MASFactory Visualizer.")


if __name__ == "__main__":
    main()
