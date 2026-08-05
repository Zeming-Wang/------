from __future__ import annotations

from sparse_mad.llm_runner import LLMClient, run_sparse_mad_llm
from sparse_mad.metrics import is_correct, sum_token_cost


def evaluate_topology(
    *,
    dataset: list[dict[str, str]],
    client: LLMClient,
    topology_type: str,
    num_agents: int = 6,
    max_rounds: int = 3,
) -> dict:
    if not dataset:
        raise ValueError("dataset cannot be empty")

    runs = []
    correct_count = 0
    for sample in dataset:
        run = run_sparse_mad_llm(
            question=sample["question"],
            client=client,
            num_agents=num_agents,
            topology_type=topology_type,
            max_rounds=max_rounds,
        )
        run["expected_answer"] = sample["answer"]
        run["correct"] = is_correct(run["final_answer"], sample["answer"])
        if run["correct"]:
            correct_count += 1
        runs.append(run)

    total_count = len(dataset)
    return {
        "topology_type": topology_type,
        "accuracy": correct_count / total_count,
        "correct_count": correct_count,
        "total_count": total_count,
        "token_cost": sum_token_cost(runs),
        "runs": runs,
    }


def compare_topologies(
    *,
    dataset: list[dict[str, str]],
    client: LLMClient,
    num_agents: int = 6,
    max_rounds: int = 3,
) -> dict:
    fully_connected = evaluate_topology(
        dataset=dataset,
        client=client,
        topology_type="fully_connected",
        num_agents=num_agents,
        max_rounds=max_rounds,
    )
    neighbor_connected = evaluate_topology(
        dataset=dataset,
        client=client,
        topology_type="ring",
        num_agents=num_agents,
        max_rounds=max_rounds,
    )

    full_cost = fully_connected["token_cost"]
    sparse_cost = neighbor_connected["token_cost"]
    cost_saving = 0.0 if full_cost == 0 else 1 - sparse_cost / full_cost

    return {
        "fully_connected": fully_connected,
        "neighbor_connected": neighbor_connected,
        "cost_saving": cost_saving,
        "num_agents": num_agents,
        "max_rounds": max_rounds,
    }
