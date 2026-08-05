from __future__ import annotations


QUESTION_KEYS = {"question": "input question for visualization"}
RESPONSE_KEYS = {"visible_response": "previous-round response visible through this communication edge"}
ANSWER_KEYS = {"agent_answer": "agent final answer for consensus"}
FULL_RESULT_KEYS = {"full_result": "fully-connected consensus result"}
NEIGHBOR_RESULT_KEYS = {"neighbor_result": "neighbor-connected consensus result"}
SUMMARY_KEYS = {"visual_summary": "summary of the static topology visualization"}


def build_static_agent_topology_graph():
    """Static 4-agent topology graph for MASFactory Visualizer.

    No loops are used here on purpose. The Visualizer can warn on dynamic graph
    construction, so this file explicitly declares every node and edge.
    """
    from masfactory import CustomNode, RootGraph

    graph = RootGraph(name="static_agent_topology_visual_graph")

    prepare = graph.create_node(CustomNode, name="PrepareQuestion", forward=_prepare_question)

    full_r1_a1 = graph.create_node(CustomNode, name="Full_R1_Agent_1", forward=_initial_full_a1)
    full_r1_a2 = graph.create_node(CustomNode, name="Full_R1_Agent_2", forward=_initial_full_a2)
    full_r1_a3 = graph.create_node(CustomNode, name="Full_R1_Agent_3", forward=_initial_full_a3)
    full_r1_a4 = graph.create_node(CustomNode, name="Full_R1_Agent_4", forward=_initial_full_a4)

    full_r2_a1 = graph.create_node(CustomNode, name="Full_R2_Agent_1", forward=_debate_full_a1)
    full_r2_a2 = graph.create_node(CustomNode, name="Full_R2_Agent_2", forward=_debate_full_a2)
    full_r2_a3 = graph.create_node(CustomNode, name="Full_R2_Agent_3", forward=_debate_full_a3)
    full_r2_a4 = graph.create_node(CustomNode, name="Full_R2_Agent_4", forward=_debate_full_a4)
    full_vote = graph.create_node(CustomNode, name="Full_ConsensusVote", forward=_full_vote)

    neighbor_r1_a1 = graph.create_node(CustomNode, name="Neighbor_R1_Agent_1", forward=_initial_neighbor_a1)
    neighbor_r1_a2 = graph.create_node(CustomNode, name="Neighbor_R1_Agent_2", forward=_initial_neighbor_a2)
    neighbor_r1_a3 = graph.create_node(CustomNode, name="Neighbor_R1_Agent_3", forward=_initial_neighbor_a3)
    neighbor_r1_a4 = graph.create_node(CustomNode, name="Neighbor_R1_Agent_4", forward=_initial_neighbor_a4)

    neighbor_r2_a1 = graph.create_node(CustomNode, name="Neighbor_R2_Agent_1", forward=_debate_neighbor_a1)
    neighbor_r2_a2 = graph.create_node(CustomNode, name="Neighbor_R2_Agent_2", forward=_debate_neighbor_a2)
    neighbor_r2_a3 = graph.create_node(CustomNode, name="Neighbor_R2_Agent_3", forward=_debate_neighbor_a3)
    neighbor_r2_a4 = graph.create_node(CustomNode, name="Neighbor_R2_Agent_4", forward=_debate_neighbor_a4)
    neighbor_vote = graph.create_node(CustomNode, name="Neighbor_ConsensusVote", forward=_neighbor_vote)

    compare = graph.create_node(CustomNode, name="CompareAccuracyAndCostSaving", forward=_compare)

    graph.edge_from_entry(prepare, QUESTION_KEYS)

    graph.create_edge(prepare, full_r1_a1, QUESTION_KEYS)
    graph.create_edge(prepare, full_r1_a2, QUESTION_KEYS)
    graph.create_edge(prepare, full_r1_a3, QUESTION_KEYS)
    graph.create_edge(prepare, full_r1_a4, QUESTION_KEYS)
    graph.create_edge(prepare, neighbor_r1_a1, QUESTION_KEYS)
    graph.create_edge(prepare, neighbor_r1_a2, QUESTION_KEYS)
    graph.create_edge(prepare, neighbor_r1_a3, QUESTION_KEYS)
    graph.create_edge(prepare, neighbor_r1_a4, QUESTION_KEYS)

    # Fully connected visibility: every R2 agent sees every other R1 agent.
    graph.create_edge(full_r1_a1, full_r2_a2, RESPONSE_KEYS)
    graph.create_edge(full_r1_a1, full_r2_a3, RESPONSE_KEYS)
    graph.create_edge(full_r1_a1, full_r2_a4, RESPONSE_KEYS)
    graph.create_edge(full_r1_a2, full_r2_a1, RESPONSE_KEYS)
    graph.create_edge(full_r1_a2, full_r2_a3, RESPONSE_KEYS)
    graph.create_edge(full_r1_a2, full_r2_a4, RESPONSE_KEYS)
    graph.create_edge(full_r1_a3, full_r2_a1, RESPONSE_KEYS)
    graph.create_edge(full_r1_a3, full_r2_a2, RESPONSE_KEYS)
    graph.create_edge(full_r1_a3, full_r2_a4, RESPONSE_KEYS)
    graph.create_edge(full_r1_a4, full_r2_a1, RESPONSE_KEYS)
    graph.create_edge(full_r1_a4, full_r2_a2, RESPONSE_KEYS)
    graph.create_edge(full_r1_a4, full_r2_a3, RESPONSE_KEYS)

    # Neighbor-connected visibility on a 4-agent ring.
    graph.create_edge(neighbor_r1_a1, neighbor_r2_a2, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a1, neighbor_r2_a4, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a2, neighbor_r2_a1, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a2, neighbor_r2_a3, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a3, neighbor_r2_a2, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a3, neighbor_r2_a4, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a4, neighbor_r2_a1, RESPONSE_KEYS)
    graph.create_edge(neighbor_r1_a4, neighbor_r2_a3, RESPONSE_KEYS)

    graph.create_edge(full_r2_a1, full_vote, ANSWER_KEYS)
    graph.create_edge(full_r2_a2, full_vote, ANSWER_KEYS)
    graph.create_edge(full_r2_a3, full_vote, ANSWER_KEYS)
    graph.create_edge(full_r2_a4, full_vote, ANSWER_KEYS)
    graph.create_edge(neighbor_r2_a1, neighbor_vote, ANSWER_KEYS)
    graph.create_edge(neighbor_r2_a2, neighbor_vote, ANSWER_KEYS)
    graph.create_edge(neighbor_r2_a3, neighbor_vote, ANSWER_KEYS)
    graph.create_edge(neighbor_r2_a4, neighbor_vote, ANSWER_KEYS)

    graph.create_edge(full_vote, compare, FULL_RESULT_KEYS)
    graph.create_edge(neighbor_vote, compare, NEIGHBOR_RESULT_KEYS)
    graph.edge_to_exit(compare, SUMMARY_KEYS)

    graph.build()
    return graph


def _prepare_question(input_data: dict) -> dict:
    return {"question": input_data.get("question", "static visualization")}


def _initial_full_a1(input_data: dict) -> dict:
    return {"visible_response": "Full Agent 1 round 1 answer"}


def _initial_full_a2(input_data: dict) -> dict:
    return {"visible_response": "Full Agent 2 round 1 answer"}


def _initial_full_a3(input_data: dict) -> dict:
    return {"visible_response": "Full Agent 3 round 1 answer"}


def _initial_full_a4(input_data: dict) -> dict:
    return {"visible_response": "Full Agent 4 round 1 answer"}


def _initial_neighbor_a1(input_data: dict) -> dict:
    return {"visible_response": "Neighbor Agent 1 round 1 answer"}


def _initial_neighbor_a2(input_data: dict) -> dict:
    return {"visible_response": "Neighbor Agent 2 round 1 answer"}


def _initial_neighbor_a3(input_data: dict) -> dict:
    return {"visible_response": "Neighbor Agent 3 round 1 answer"}


def _initial_neighbor_a4(input_data: dict) -> dict:
    return {"visible_response": "Neighbor Agent 4 round 1 answer"}


def _debate_full_a1(input_data: dict) -> dict:
    return {"agent_answer": "Full Agent 1 round 2 answer"}


def _debate_full_a2(input_data: dict) -> dict:
    return {"agent_answer": "Full Agent 2 round 2 answer"}


def _debate_full_a3(input_data: dict) -> dict:
    return {"agent_answer": "Full Agent 3 round 2 answer"}


def _debate_full_a4(input_data: dict) -> dict:
    return {"agent_answer": "Full Agent 4 round 2 answer"}


def _debate_neighbor_a1(input_data: dict) -> dict:
    return {"agent_answer": "Neighbor Agent 1 round 2 answer"}


def _debate_neighbor_a2(input_data: dict) -> dict:
    return {"agent_answer": "Neighbor Agent 2 round 2 answer"}


def _debate_neighbor_a3(input_data: dict) -> dict:
    return {"agent_answer": "Neighbor Agent 3 round 2 answer"}


def _debate_neighbor_a4(input_data: dict) -> dict:
    return {"agent_answer": "Neighbor Agent 4 round 2 answer"}


def _full_vote(input_data: dict) -> dict:
    return {"full_result": "Full topology consensus"}


def _neighbor_vote(input_data: dict) -> dict:
    return {"neighbor_result": "Neighbor topology consensus"}


def _compare(input_data: dict) -> dict:
    return {"visual_summary": "Static graph: full topology has 12 visibility edges; neighbor topology has 8."}
