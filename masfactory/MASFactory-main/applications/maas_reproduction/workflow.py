"""Task 2 RootGraph builders for the native MaAS reproduction."""
from __future__ import annotations

from typing import Any, Callable, Iterable

from masfactory import RootGraph
from applications.maas_reproduction.components.input_split_node import InputSplitNode
from applications.maas_reproduction.components.evaluator_node import EvaluatorNode
from applications.maas_reproduction.components.loss_update_node import LossUpdateNode
from applications.maas_reproduction.components.metrics_node import MetricsNode
from applications.maas_reproduction.components.sample_result_node import SampleResultNode
from applications.maas_reproduction.components.architecture_exec_graph import ArchitectureExecGraph


def _build_root(*, train: bool, scorer: object | None = None,
                batch_accumulator: object | None = None, metrics: object | None = None,
                policy_controller: object | None = None, operator_embeddings: object | None = None,
                operator_catalog: object | None = None, programmer: object | None = None,
                generate: object | None = None, operator_registry: object | None = None,
                dataset: str = "MATH", cost_tracker: object | None = None) -> RootGraph:
    root = RootGraph(name="maas_train" if train else "maas_test")
    split = root.create_node(InputSplitNode, name="input_split")
    architecture = root.create_node(
        ArchitectureExecGraph,
        name="architecture",
        policy_controller=policy_controller,
        operator_embeddings=operator_embeddings,
        operator_catalog=operator_catalog,
        programmer=programmer,
        generate=generate,
        operator_registry=operator_registry,
        dataset=dataset,
        cost_tracker=cost_tracker,
    )
    evaluator = root.create_node(EvaluatorNode, name="evaluator", scorer=scorer, pull_keys={})
    tail = (root.create_node(LossUpdateNode, name="loss_update", batch_accumulator=batch_accumulator, pull_keys={})
            if train else root.create_node(MetricsNode, name="metrics", metrics=metrics, pull_keys={}))
    sample = root.create_node(SampleResultNode, name="sample_result", pull_keys={})
    # RootGraph edges require declared keys; the stable envelope key keeps the
    # invocation payload opaque while InputSplit performs validation/isolation.
    root.edge_from_entry(split, keys={"sample": ""})
    root.create_edge(split, architecture, keys={"architecture_request": ""})
    root.create_edge(split, evaluator, keys={"evaluation_context": ""})
    root.create_edge(architecture, evaluator, keys={"architecture_result": ""})
    root.create_edge(evaluator, tail, keys={"evaluation_result": ""})
    if train:
        root.create_edge(evaluator, sample, keys={"evaluation_result": ""})
        root.create_edge(tail, sample, keys={"update_result": ""})
    else:
        root.create_edge(tail, sample, keys={"evaluation_result": ""})
    root.edge_to_exit(sample, keys={"sample_result": ""})
    root.build()
    return root


def build_train_root_graph(*, scorer: object | None = None,
                           batch_accumulator: object | None = None, **kwargs: Any) -> RootGraph:
    return _build_root(
        train=True,
        scorer=scorer,
        batch_accumulator=batch_accumulator,
        **kwargs,
    )


def build_test_root_graph(*, scorer: object | None = None,
                          metrics: object | None = None, **kwargs: Any) -> RootGraph:
    return _build_root(
        train=False,
        scorer=scorer,
        metrics=metrics,
        **kwargs,
    )


__all__ = ["build_train_root_graph", "build_test_root_graph"]
