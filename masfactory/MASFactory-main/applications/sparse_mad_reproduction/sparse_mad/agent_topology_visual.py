from __future__ import annotations

from collections.abc import Callable


QUESTION_KEYS = {
    "question": "input question for the debate demo",
}

RESPONSE_KEYS = {
    "visible_response": "one previous-round response visible through a topology edge",
}

CONSENSUS_KEYS = {
    "agent_answer": "final answer sent to consensus",
}


def build_agent_topology_visual_graph(*, num_agents: int = 4):
    """Build a MASFactory graph whose edges directly visualize paper topologies.

    This graph is intentionally visual-first: every debate agent is an explicit
    node, and communication topology is represented by explicit edges from
    round-1 agent nodes to round-2 agent nodes.
    """
    from masfactory import CustomNode, RootGraph

    if num_agents < 3:
        raise ValueError("num_agents must be at least 3 for neighbor topology visualization")

    graph = RootGraph(name="agent_topology_visual_graph")

    prepare = graph.create_node(CustomNode, name="PrepareQuestion", forward=_prepare_question)

    full_initial = [
        graph.create_node(
            CustomNode,
            name=f"Full_R1_Agent_{index}",
            forward=_make_initial_forward(f"Full_R1_Agent_{index}"),
        )
        for index in range(1, num_agents + 1)
    ]
    full_debate = [
        graph.create_node(
            CustomNode,
            name=f"Full_R2_Agent_{index}",
            forward=_make_debate_forward(f"Full_R2_Agent_{index}"),
        )
        for index in range(1, num_agents + 1)
    ]
    full_vote = graph.create_node(CustomNode, name="Full_ConsensusVote", forward=_consensus_forward)

    neighbor_initial = [
        graph.create_node(
            CustomNode,
            name=f"Neighbor_R1_Agent_{index}",
            forward=_make_initial_forward(f"Neighbor_R1_Agent_{index}"),
        )
        for index in range(1, num_agents + 1)
    ]
    neighbor_debate = [
        graph.create_node(
            CustomNode,
            name=f"Neighbor_R2_Agent_{index}",
            forward=_make_debate_forward(f"Neighbor_R2_Agent_{index}"),
        )
        for index in range(1, num_agents + 1)
    ]
    neighbor_vote = graph.create_node(CustomNode, name="Neighbor_ConsensusVote", forward=_consensus_forward)

    compare = graph.create_node(CustomNode, name="CompareAccuracyAndCostSaving", forward=_compare_forward)

    graph.edge_from_entry(prepare, QUESTION_KEYS)

    for node in full_initial + neighbor_initial:
        graph.create_edge(prepare, node, QUESTION_KEYS)

    for source_index, source in enumerate(full_initial):
        for target_index, target in enumerate(full_debate):
            if source_index != target_index:
                graph.create_edge(source, target, RESPONSE_KEYS)

    for source_index, source in enumerate(neighbor_initial):
        left_neighbor = (source_index - 1) % num_agents
        right_neighbor = (source_index + 1) % num_agents
        graph.create_edge(source, neighbor_debate[left_neighbor], RESPONSE_KEYS)
        graph.create_edge(source, neighbor_debate[right_neighbor], RESPONSE_KEYS)

    for node in full_debate:
        graph.create_edge(node, full_vote, CONSENSUS_KEYS)
    for node in neighbor_debate:
        graph.create_edge(node, neighbor_vote, CONSENSUS_KEYS)

    graph.create_edge(full_vote, compare, {"full_result": "fully-connected result"})
    graph.create_edge(neighbor_vote, compare, {"neighbor_result": "neighbor-connected result"})
    graph.edge_to_exit(compare, {"visual_summary": "visual explanation of both topologies"})

    graph.build()
    return graph


def expected_edge_counts(*, num_agents: int) -> dict[str, int]:
    return {
        "fully_connected_directed_edges": num_agents * (num_agents - 1),
        "neighbor_connected_directed_edges": num_agents * 2,
    }


def _prepare_question(input_data: dict) -> dict:
    return {"question": input_data.get("question", "visualization only")}


def _make_initial_forward(agent_name: str) -> Callable[[dict], dict]:
    def forward(input_data: dict) -> dict:
        return {"visible_response": f"{agent_name} initial answer to {input_data.get('question', '')}"}

    return forward


def _make_debate_forward(agent_name: str) -> Callable[[dict], dict]:
    def forward(input_data: dict) -> dict:
        return {"agent_answer": f"{agent_name} updated answer after reading visible neighbors"}

    return forward


def _consensus_forward(input_data: dict) -> dict:
    return {"full_result": "full consensus", "neighbor_result": "neighbor consensus"}


def _compare_forward(input_data: dict) -> dict:
    return {
        "visual_summary": (
            "This graph visualizes the paper SOP: each agent is a node; "
            "edges from R1 agents to R2 agents show which previous answers are visible."
        )
    }
