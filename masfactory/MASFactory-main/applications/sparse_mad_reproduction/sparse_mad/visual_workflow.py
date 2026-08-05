from __future__ import annotations

from sparse_mad.experiment import evaluate_topology
from sparse_mad.llm_runner import LLMClient


VISUAL_INPUT_KEYS = {
    "dataset": "list of samples with question and answer",
    "num_agents": "number of debate agents",
    "max_rounds": "total debate rounds",
}

PREPARED_KEYS = {
    **VISUAL_INPUT_KEYS,
    "experiment_name": "name for this topology comparison experiment",
}

FULL_TOPOLOGY_KEYS = {
    **PREPARED_KEYS,
    "full_topology_type": "fully-connected topology label",
}

FULL_RESULT_KEYS = {
    **PREPARED_KEYS,
    "full_topology_type": "fully-connected topology label",
    "fully_connected": "evaluation result for D=1 topology",
}

NEIGHBOR_TOPOLOGY_KEYS = {
    **FULL_RESULT_KEYS,
    "neighbor_topology_type": "neighbor-connected topology label",
}

BOTH_RESULT_KEYS = {
    **FULL_RESULT_KEYS,
    "neighbor_topology_type": "neighbor-connected topology label",
    "neighbor_connected": "evaluation result for sparse neighbor topology",
}

METRIC_KEYS = {
    **BOTH_RESULT_KEYS,
    "cost_saving": "relative sparse cost saving against fully connected",
}

REPORT_KEYS = {
    **METRIC_KEYS,
    "report": "human-readable experiment report",
}


def build_visual_comparison_graph(*, client: LLMClient):
    """Build a visual, fine-grained MASFactory graph for the paper SOP."""
    from masfactory import CustomNode, RootGraph

    graph = RootGraph(name="visual_sparse_mad_paper_sop")

    prepare_dataset = graph.create_node(CustomNode, name="PrepareDataset", forward=_prepare_dataset)
    build_full_topology = graph.create_node(CustomNode, name="BuildFullTopology_D_equals_1", forward=_build_full_topology)
    run_full_mad = graph.create_node(
        CustomNode,
        name="RunFullyConnectedMAD",
        forward=lambda input_data: _run_full_mad(input_data, client=client),
    )
    build_neighbor_topology = graph.create_node(
        CustomNode,
        name="BuildNeighborTopology",
        forward=_build_neighbor_topology,
    )
    run_neighbor_mad = graph.create_node(
        CustomNode,
        name="RunNeighborConnectedMAD",
        forward=lambda input_data: _run_neighbor_mad(input_data, client=client),
    )
    compute_cost_saving = graph.create_node(CustomNode, name="ComputeAccuracyAndCostSaving", forward=_compute_cost_saving)
    format_report = graph.create_node(CustomNode, name="FormatExperimentReport", forward=_format_report)

    graph.edge_from_entry(prepare_dataset, VISUAL_INPUT_KEYS)
    graph.create_edge(prepare_dataset, build_full_topology, PREPARED_KEYS)
    graph.create_edge(build_full_topology, run_full_mad, FULL_TOPOLOGY_KEYS)
    graph.create_edge(run_full_mad, build_neighbor_topology, FULL_RESULT_KEYS)
    graph.create_edge(build_neighbor_topology, run_neighbor_mad, NEIGHBOR_TOPOLOGY_KEYS)
    graph.create_edge(run_neighbor_mad, compute_cost_saving, BOTH_RESULT_KEYS)
    graph.create_edge(compute_cost_saving, format_report, METRIC_KEYS)
    graph.edge_to_exit(format_report, REPORT_KEYS)

    graph.build()
    return graph


def _prepare_dataset(input_data: dict) -> dict:
    return {
        "dataset": input_data["dataset"],
        "num_agents": int(input_data.get("num_agents", 4)),
        "max_rounds": int(input_data.get("max_rounds", 2)),
        "experiment_name": "Sparse MAD comparison: fully connected vs neighbor connected",
    }


def _build_full_topology(input_data: dict) -> dict:
    return {**input_data, "full_topology_type": "fully_connected"}


def _run_full_mad(input_data: dict, *, client: LLMClient) -> dict:
    fully_connected = evaluate_topology(
        dataset=input_data["dataset"],
        client=client,
        topology_type=input_data["full_topology_type"],
        num_agents=input_data["num_agents"],
        max_rounds=input_data["max_rounds"],
    )
    return {**input_data, "fully_connected": fully_connected}


def _build_neighbor_topology(input_data: dict) -> dict:
    return {**input_data, "neighbor_topology_type": "ring"}


def _run_neighbor_mad(input_data: dict, *, client: LLMClient) -> dict:
    neighbor_connected = evaluate_topology(
        dataset=input_data["dataset"],
        client=client,
        topology_type=input_data["neighbor_topology_type"],
        num_agents=input_data["num_agents"],
        max_rounds=input_data["max_rounds"],
    )
    return {**input_data, "neighbor_connected": neighbor_connected}


def _compute_cost_saving(input_data: dict) -> dict:
    full_cost = input_data["fully_connected"]["token_cost"]
    neighbor_cost = input_data["neighbor_connected"]["token_cost"]
    cost_saving = 0.0 if full_cost == 0 else 1 - neighbor_cost / full_cost
    return {**input_data, "cost_saving": cost_saving}


def _format_report(input_data: dict) -> dict:
    full = input_data["fully_connected"]
    neighbor = input_data["neighbor_connected"]
    report = (
        "Sparse MAD comparison report\n"
        f"Experiment: {input_data['experiment_name']}\n"
        f"Agents: {input_data['num_agents']} | Rounds: {input_data['max_rounds']}\n"
        f"Fully-connected MAD: accuracy={full['accuracy']:.3f}, cost={full['token_cost']}\n"
        f"Neighbor-connected MAD: accuracy={neighbor['accuracy']:.3f}, cost={neighbor['token_cost']}\n"
        f"Cost saving: {input_data['cost_saving']:.1%}"
    )
    return {**input_data, "report": report}
