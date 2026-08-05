from __future__ import annotations


TOPOLOGY_INPUT_KEYS = {"question": "placeholder input for topology visualization"}
TOPOLOGY_EDGE_KEYS = {"communication": "undirected communication relation represented by one edge"}
SUMMARY_KEYS = {"visual_summary": "summary of full and neighbor topology graphs"}


def build_hex_topology_visual_graph():
    """Static 6-agent communication-topology graph for Visualizer.

    This graph visualizes the paper topology itself, not the cross-round message
    execution flow. Each Agent is a node. Each edge means the two agents can
    see each other's previous-round answers. For visual clarity, each undirected
    relation is represented with one directed MASFactory edge.
    """
    from masfactory import CustomNode, RootGraph

    graph = RootGraph(name="hex_topology_visual_graph")

    start = graph.create_node(CustomNode, name="TopologyInput", forward=_topology_input)

    full_a1 = graph.create_node(CustomNode, name="Full_Agent_1", forward=_pass_communication)
    full_a2 = graph.create_node(CustomNode, name="Full_Agent_2", forward=_pass_communication)
    full_a3 = graph.create_node(CustomNode, name="Full_Agent_3", forward=_pass_communication)
    full_a4 = graph.create_node(CustomNode, name="Full_Agent_4", forward=_pass_communication)
    full_a5 = graph.create_node(CustomNode, name="Full_Agent_5", forward=_pass_communication)
    full_a6 = graph.create_node(CustomNode, name="Full_Agent_6", forward=_pass_communication)

    neighbor_a1 = graph.create_node(CustomNode, name="Neighbor_Agent_1", forward=_pass_communication)
    neighbor_a2 = graph.create_node(CustomNode, name="Neighbor_Agent_2", forward=_pass_communication)
    neighbor_a3 = graph.create_node(CustomNode, name="Neighbor_Agent_3", forward=_pass_communication)
    neighbor_a4 = graph.create_node(CustomNode, name="Neighbor_Agent_4", forward=_pass_communication)
    neighbor_a5 = graph.create_node(CustomNode, name="Neighbor_Agent_5", forward=_pass_communication)
    neighbor_a6 = graph.create_node(CustomNode, name="Neighbor_Agent_6", forward=_pass_communication)

    summary = graph.create_node(CustomNode, name="TopologySummary", forward=_summary)

    graph.edge_from_entry(start, TOPOLOGY_INPUT_KEYS)

    graph.create_edge(start, full_a1, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(start, neighbor_a1, TOPOLOGY_EDGE_KEYS)

    # Full topology K6: every pair of agents is connected.
    graph.create_edge(full_a1, full_a2, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a1, full_a3, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a1, full_a4, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a1, full_a5, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a1, full_a6, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a2, full_a3, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a2, full_a4, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a2, full_a5, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a2, full_a6, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a3, full_a4, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a3, full_a5, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a3, full_a6, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a4, full_a5, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a4, full_a6, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(full_a5, full_a6, TOPOLOGY_EDGE_KEYS)

    # Neighbor topology C6: only adjacent agents are connected in a ring.
    graph.create_edge(neighbor_a1, neighbor_a2, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(neighbor_a2, neighbor_a3, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(neighbor_a3, neighbor_a4, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(neighbor_a4, neighbor_a5, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(neighbor_a5, neighbor_a6, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(neighbor_a1, neighbor_a6, TOPOLOGY_EDGE_KEYS)

    graph.create_edge(full_a6, summary, TOPOLOGY_EDGE_KEYS)
    graph.create_edge(neighbor_a6, summary, TOPOLOGY_EDGE_KEYS)
    graph.edge_to_exit(summary, SUMMARY_KEYS)

    graph.build()
    return graph


def hex_edge_counts() -> dict[str, int]:
    return {
        "fully_connected_edges": 15,
        "neighbor_ring_edges": 6,
    }


def _topology_input(input_data: dict) -> dict:
    return {"communication": "communication topology visualization"}


def _pass_communication(input_data: dict) -> dict:
    return {"communication": input_data.get("communication", "visible communication relation")}


def _summary(input_data: dict) -> dict:
    return {
        "visual_summary": (
            "Full topology is K6 with 15 undirected edges; "
            "neighbor topology is C6 with 6 undirected ring edges."
        )
    }
