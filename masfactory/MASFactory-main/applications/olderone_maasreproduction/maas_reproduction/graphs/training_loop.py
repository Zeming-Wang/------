"""TrainingLoop graph builder for the MaAS MASFactory application."""
from __future__ import annotations

import logging

from maas_reproduction.nodes.architecture_exec_node import architecture_exec_forward
from maas_reproduction.nodes.evaluator_node import evaluator_forward
from maas_reproduction.nodes.training_controller import training_controller

logger = logging.getLogger(__name__)

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

TRAINING_LOOP_PUSH_KEYS = {
    "average_score": "Average score over all processed problems.",
    "round": "Completed MaAS round.",
    "checkpoint_path": "Controller checkpoint path.",
    "result_path": "Runtime result directory.",
    "runtime_metadata": "Execution metadata.",
}


def create_training_loop(name: str = "TrainingLoop", max_iterations: int = 100000):
    from masfactory.components.graphs.loop import Loop

    loop = Loop(
        name=name,
        max_iterations=max_iterations,
        terminate_condition_function=training_controller,
        pull_keys=None,
        push_keys=TRAINING_LOOP_PUSH_KEYS,
    )
    attach_training_loop_body(loop)
    return loop


def attach_training_loop_body(loop) -> None:
    from masfactory.components.custom_node import CustomNode
    from maas_reproduction.nodes.loss_update_node import loss_update_forward

    architecture_node = loop.create_node(
        CustomNode,
        "ArchitectureExecNode",
        forward=architecture_exec_forward,
        pull_keys=None,
        push_keys={},
    )
    evaluator_node = loop.create_node(
        CustomNode,
        "EvaluatorNode",
        forward=evaluator_forward,
        pull_keys=None,
        push_keys={},
    )
    loss_update_node = loop.create_node(
        CustomNode,
        "LossUpdateNode",
        forward=loss_update_forward,
        pull_keys=None,
        push_keys={},
    )

    loop.edge_from_controller(architecture_node, CONTROLLER_TO_ARCHITECTURE_KEYS)
    loop.create_edge(architecture_node, evaluator_node, ARCHITECTURE_TO_EVALUATOR_KEYS)
    loop.create_edge(evaluator_node, loss_update_node, EVALUATOR_TO_LOSS_KEYS)
    loop.edge_to_controller(loss_update_node, LOSS_TO_CONTROLLER_KEYS)

    logger.info("Created MaAS TrainingLoop graph with ArchitectureExec -> Evaluator -> LossUpdate")
