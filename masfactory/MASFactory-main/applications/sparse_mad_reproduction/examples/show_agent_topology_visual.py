from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_mad.agent_topology_visual import build_agent_topology_visual_graph, expected_edge_counts


def main() -> None:
    num_agents = 4
    graph = build_agent_topology_visual_graph(num_agents=num_agents)
    print(f"Built graph: {graph.name}")
    print(expected_edge_counts(num_agents=num_agents))
    print("Open sparse_mad/agent_topology_visual.py with MASFactory Visualizer to inspect explicit Agent nodes and topology edges.")


if __name__ == "__main__":
    main()
