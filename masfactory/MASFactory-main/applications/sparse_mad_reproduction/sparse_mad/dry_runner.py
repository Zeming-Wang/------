from __future__ import annotations

from sparse_mad.topology import build_topology
from sparse_mad.voting import majority_vote


def run_sparse_mad_dry(
    *,
    question: str,
    num_agents: int = 6,
    topology_type: str = "ring",
    max_rounds: int = 3,
    correct_answer: str = "42",
) -> dict:
    """Run a deterministic sparse MAD simulation without calling an LLM."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    topology = build_topology(num_agents=num_agents, topology_type=topology_type)
    responses_by_round: dict[int, dict[str, dict[str, str]]] = {}
    visibility_by_round: dict[int, dict[str, list[str]]] = {}

    responses_by_round[1] = _initial_responses(
        question=question,
        agents=topology.agents,
        correct_answer=correct_answer,
    )

    for round_id in range(2, max_rounds + 1):
        previous = responses_by_round[round_id - 1]
        visibility_by_round[round_id] = topology.neighbors
        responses_by_round[round_id] = {
            agent: _debate_response(
                agent=agent,
                round_id=round_id,
                question=question,
                previous_answer=previous[agent]["answer"],
                neighbor_answers=[previous[neighbor]["answer"] for neighbor in topology.neighbors[agent]],
                correct_answer=correct_answer,
            )
            for agent in topology.agents
        }

    final_round = responses_by_round[max_rounds]
    vote = majority_vote({agent: response["answer"] for agent, response in final_round.items()})

    return {
        "question": question,
        "topology": topology.as_dict(),
        "responses_by_round": responses_by_round,
        "visibility_by_round": visibility_by_round,
        "final_agent_answers": {agent: response["answer"] for agent, response in final_round.items()},
        "final_answer": vote.final_answer,
        "vote_summary": vote.vote_summary,
        "token_cost": 0,
    }


def _initial_responses(*, question: str, agents: list[str], correct_answer: str) -> dict[str, dict[str, str]]:
    responses = {}
    for index, agent in enumerate(agents):
        answer = correct_answer if index % 3 != 0 else str(int(correct_answer) - 1)
        responses[agent] = {
            "reasoning": f"{agent} initial dry reasoning for: {question}",
            "answer": answer,
        }
    return responses


def _debate_response(
    *,
    agent: str,
    round_id: int,
    question: str,
    previous_answer: str,
    neighbor_answers: list[str],
    correct_answer: str,
) -> dict[str, str]:
    votes = neighbor_answers + [previous_answer]
    answer = correct_answer if correct_answer in votes else previous_answer
    return {
        "reasoning": (
            f"{agent} round {round_id} reviewed {len(neighbor_answers)} neighbor answers "
            f"for: {question}"
        ),
        "answer": answer,
    }
