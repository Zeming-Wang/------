from __future__ import annotations

from sparse_mad.dry_runner import run_sparse_mad_dry
from sparse_mad.experiment import compare_topologies
from sparse_mad.llm_runner import LLMClient, run_sparse_mad_llm


WORKFLOW_KEYS = {
    "question": "input task question",
    "num_agents": "number of debate agents",
    "topology_type": "communication topology type",
    "max_rounds": "total debate rounds",
    "correct_answer": "optional expected answer for dry-run simulation",
}

LLM_WORKFLOW_KEYS = {
    "question": "input task question",
    "num_agents": "number of debate agents",
    "topology_type": "communication topology type",
    "max_rounds": "total debate rounds",
}

COMPARISON_KEYS = {
    "dataset": "list of samples with question and answer",
    "num_agents": "number of debate agents",
    "max_rounds": "total debate rounds",
}

RESULT_KEYS = {
    "question": "input task question",
    "topology": "communication topology information",
    "responses_by_round": "all agent responses grouped by round",
    "visibility_by_round": "neighbor visibility for debate rounds",
    "final_agent_answers": "final answer from each agent",
    "final_answer": "majority-vote final answer",
    "vote_summary": "answer vote counts",
    "token_cost": "tracked input token cost",
}

EXIT_KEYS = {
    **RESULT_KEYS,
    "summary": "human-readable run summary",
}

COMPARISON_RESULT_KEYS = {
    "fully_connected": "evaluation result for D=1 topology",
    "neighbor_connected": "evaluation result for sparse neighbor topology",
    "cost_saving": "relative sparse cost saving against fully connected",
    "num_agents": "number of debate agents",
    "max_rounds": "total debate rounds",
}

COMPARISON_EXIT_KEYS = {
    **COMPARISON_RESULT_KEYS,
    "summary": "human-readable comparison summary",
}


def build_sparse_mad_dry_graph():
    from masfactory import CustomNode, RootGraph

    graph = RootGraph(name="sparse_mad_dry_graph")

    prepare = graph.create_node(CustomNode, name="PrepareQuestion", forward=_prepare_question)
    run_debate = graph.create_node(CustomNode, name="RunSparseMADDry", forward=_run_sparse_mad_dry)
    format_result = graph.create_node(CustomNode, name="FormatResult", forward=_format_result)

    graph.edge_from_entry(prepare, WORKFLOW_KEYS)
    graph.create_edge(prepare, run_debate, WORKFLOW_KEYS)
    graph.create_edge(run_debate, format_result, RESULT_KEYS)
    graph.edge_to_exit(format_result, EXIT_KEYS)

    graph.build()
    return graph


def build_sparse_mad_llm_graph(*, client: LLMClient):
    from masfactory import CustomNode, RootGraph

    graph = RootGraph(name="sparse_mad_llm_graph")

    prepare = graph.create_node(CustomNode, name="PrepareQuestion", forward=_prepare_question)
    run_debate = graph.create_node(
        CustomNode,
        name="RunSparseMADLLM",
        forward=lambda input_data: _run_sparse_mad_llm(input_data, client=client),
    )
    format_result = graph.create_node(CustomNode, name="FormatResult", forward=_format_result)

    graph.edge_from_entry(prepare, LLM_WORKFLOW_KEYS)
    graph.create_edge(prepare, run_debate, LLM_WORKFLOW_KEYS)
    graph.create_edge(run_debate, format_result, RESULT_KEYS)
    graph.edge_to_exit(format_result, EXIT_KEYS)

    graph.build()
    return graph


def build_sparse_mad_comparison_graph(*, client: LLMClient):
    from masfactory import CustomNode, RootGraph

    graph = RootGraph(name="sparse_mad_topology_comparison_graph")

    prepare = graph.create_node(CustomNode, name="PrepareComparison", forward=_prepare_comparison)
    run_comparison = graph.create_node(
        CustomNode,
        name="RunTopologyComparison",
        forward=lambda input_data: _run_topology_comparison(input_data, client=client),
    )
    format_result = graph.create_node(CustomNode, name="FormatComparison", forward=_format_comparison)

    graph.edge_from_entry(prepare, COMPARISON_KEYS)
    graph.create_edge(prepare, run_comparison, COMPARISON_KEYS)
    graph.create_edge(run_comparison, format_result, COMPARISON_RESULT_KEYS)
    graph.edge_to_exit(format_result, COMPARISON_EXIT_KEYS)

    graph.build()
    return graph


def _prepare_question(input_data: dict) -> dict:
    return {
        "question": input_data["question"],
        "num_agents": int(input_data.get("num_agents", 6)),
        "topology_type": input_data.get("topology_type", "ring"),
        "max_rounds": int(input_data.get("max_rounds", 3)),
        "correct_answer": str(input_data.get("correct_answer", "42")),
    }


def _prepare_comparison(input_data: dict) -> dict:
    return {
        "dataset": input_data["dataset"],
        "num_agents": int(input_data.get("num_agents", 6)),
        "max_rounds": int(input_data.get("max_rounds", 3)),
    }


def _run_sparse_mad_dry(input_data: dict) -> dict:
    return run_sparse_mad_dry(
        question=input_data["question"],
        num_agents=input_data["num_agents"],
        topology_type=input_data["topology_type"],
        max_rounds=input_data["max_rounds"],
        correct_answer=input_data["correct_answer"],
    )


def _run_sparse_mad_llm(input_data: dict, *, client: LLMClient) -> dict:
    return run_sparse_mad_llm(
        question=input_data["question"],
        client=client,
        num_agents=input_data["num_agents"],
        topology_type=input_data["topology_type"],
        max_rounds=input_data["max_rounds"],
    )


def _run_topology_comparison(input_data: dict, *, client: LLMClient) -> dict:
    return compare_topologies(
        dataset=input_data["dataset"],
        client=client,
        num_agents=input_data["num_agents"],
        max_rounds=input_data["max_rounds"],
    )


def _format_result(input_data: dict) -> dict:
    topology = input_data["topology"]
    summary = (
        f"Final answer: {input_data['final_answer']} | "
        f"topology density: {topology['density']:.3f} | "
        f"votes: {input_data['vote_summary']}"
    )
    return {**input_data, "summary": summary}


def _format_comparison(input_data: dict) -> dict:
    full = input_data["fully_connected"]
    neighbor = input_data["neighbor_connected"]
    summary = (
        f"Fully-connected accuracy={full['accuracy']:.3f}, cost={full['token_cost']} | "
        f"Neighbor-connected accuracy={neighbor['accuracy']:.3f}, cost={neighbor['token_cost']} | "
        f"cost saving={input_data['cost_saving']:.1%}"
    )
    return {**input_data, "summary": summary}
