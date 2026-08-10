"""RootGraph wiring for the MaAS reproduction application."""

from __future__ import annotations

import logging

from masfactory import CustomNode, Loop, RootGraph

from maas_reproduction.graphs.training_loop import attach_training_loop_body
from maas_reproduction.graphs.training_loop import TRAINING_LOOP_PUSH_KEYS
from maas_reproduction.nodes import config_forward, result_forward
from maas_reproduction.nodes.training_controller import training_controller

logger = logging.getLogger(__name__)

CONFIG_TO_TRAINING_KEYS = {
    "dataset": "Dataset name.",
    "mode": "Execution mode.",
    "sample": "MaAS sample count.",
    "batch_size": "Training batch size.",
    "round": "MaAS optimization round.",
    "model_config": "Model names used by MaAS.",
    "paths": "Resolved runtime paths.",
}

ENTRY_TO_CONFIG_KEYS = {
    "application_root": "MaAS reproduction application root.",
    "dataset": "Dataset name.",
    "mode": "Execution mode.",
    "sample": "MaAS sample count.",
    "round_number": "MaAS optimization round.",
    "batch_size": "Training batch size.",
    "learning_rate": "Controller optimizer learning rate.",
    "is_textgrad": "Whether textgrad is enabled.",
    "opt_model_name": "Optimization model name.",
    "exec_model_name": "Execution model name.",
}


def build_maas_reproduction_graph(name: str = "MaASReproduction") -> RootGraph:
    graph = RootGraph(name=name)
    config_node = graph.create_node(
        CustomNode,
        "ConfigNode",
        forward=config_forward,
        push_keys={"settings": "MaAS runtime settings."},
    )
    training_loop = graph.create_node(
        Loop,
        "TrainingLoop",
        max_iterations=100000,
        terminate_condition_function=training_controller,
        pull_keys=None,
        push_keys=TRAINING_LOOP_PUSH_KEYS,
    )
    attach_training_loop_body(training_loop)
    result_node = graph.create_node(
        CustomNode,
        "ResultNode",
        forward=result_forward,
    )

    graph.edge_from_entry(config_node, ENTRY_TO_CONFIG_KEYS)
    graph.create_edge(config_node, training_loop, CONFIG_TO_TRAINING_KEYS)
    graph.create_edge(training_loop, result_node, TRAINING_LOOP_PUSH_KEYS)
    graph.edge_to_exit(result_node, TRAINING_LOOP_PUSH_KEYS)

    logger.info("Built MaAS reproduction RootGraph wiring")
    return graph
