"""Static MaAS reproduction path for MASFactory Visualizer.

Open this file with the VS Code MASFactory Visualizer to inspect the phase-1
execution path. This file is visualization-only and does not call an LLM.
"""

from __future__ import annotations

from masfactory import CustomNode, RootGraph


ENTRY_TO_CONFIG_KEYS = {
    "application_root": "MaAS reproduction application root.",
    "dataset": "Dataset name.",
    "mode": "Execution mode.",
    "sample": "MaAS sample count.",
    "batch_size": "Training batch size.",
}

CONFIG_TO_CONTROLLER_KEYS = {
    "settings": "MaAS runtime settings.",
}

CONTROLLER_TO_ARCHITECTURE_KEYS = {
    "problem": "Current problem text.",
    "entry_point": "HumanEval entry point, empty for math datasets.",
    "expected_answer": "Dataset-specific expected answer or test payload.",
    "problem_index": "Current problem index.",
}

ARCHITECTURE_TO_EVALUATOR_KEYS = {
    "problem": "Current problem text.",
    "entry_point": "HumanEval entry point, empty for math datasets.",
    "expected_answer": "Dataset-specific expected answer or test payload.",
    "prediction": "MaAS workflow prediction.",
    "cost": "Cumulative LLM cost returned by MaAS workflow.",
    "logprob": "Selected architecture log probability.",
    "problem_index": "Current problem index.",
}

EVALUATOR_TO_LOSS_KEYS = {
    "score": "Benchmark score for the current problem.",
    "cost": "Cumulative LLM cost returned by MaAS workflow.",
    "logprob": "Selected architecture log probability.",
    "problem_index": "Current problem index.",
}

LOSS_TO_CONTROLLER_KEYS = {
    "result_score": "Current benchmark score.",
    "result_cost": "Current cumulative cost.",
    "result_logprob": "Current architecture log probability.",
    "result_loss": "Batch loss value, or None before update.",
    "result_update_performed": "Whether optimizer.step was executed.",
    "result_problem_index": "Completed problem index.",
}

FINAL_RESULT_KEYS = {
    "average_score": "Average score over all processed problems.",
    "round": "Completed MaAS round.",
    "checkpoint_path": "Controller checkpoint path.",
    "result_path": "Runtime result directory.",
    "runtime_metadata": "Execution metadata.",
}


def build_maas_reproduction_visual_graph() -> RootGraph:
    graph = RootGraph(name="MaASReproductionExpandedPath")

    config = graph.create_node(CustomNode, "ConfigNode", forward=_pass)
    controller = graph.create_node(CustomNode, "TrainingLoop_Controller", forward=_pass)
    architecture = graph.create_node(CustomNode, "ArchitectureExecNode", forward=_pass)
    evaluator = graph.create_node(CustomNode, "EvaluatorNode", forward=_pass)
    loss_update = graph.create_node(CustomNode, "LossUpdateNode", forward=_pass)
    result = graph.create_node(CustomNode, "ResultNode", forward=_pass)

    graph.edge_from_entry(config, ENTRY_TO_CONFIG_KEYS)
    graph.create_edge(config, controller, CONFIG_TO_CONTROLLER_KEYS)
    graph.create_edge(controller, architecture, CONTROLLER_TO_ARCHITECTURE_KEYS)
    graph.create_edge(architecture, evaluator, ARCHITECTURE_TO_EVALUATOR_KEYS)
    graph.create_edge(evaluator, loss_update, EVALUATOR_TO_LOSS_KEYS)
    graph.create_edge(loss_update, controller, LOSS_TO_CONTROLLER_KEYS)
    graph.create_edge(controller, result, FINAL_RESULT_KEYS)
    graph.edge_to_exit(result, FINAL_RESULT_KEYS)

    graph.build()
    return graph


def _pass(input_data: dict) -> dict:
    return dict(input_data)


graph = build_maas_reproduction_visual_graph()
