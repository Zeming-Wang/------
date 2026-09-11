"""Task 2 RootGraph builders for the native MaAS reproduction."""
from __future__ import annotations

from typing import Any, Callable, Iterable

from masfactory import CustomNode, RootGraph
from maas_reproduction.schemas import ArchitectureResult
from components.input_split_node import InputSplitNode
from components.evaluator_node import EvaluatorNode
from components.loss_update_node import LossUpdateNode
from components.metrics_node import MetricsNode
from components.sample_result_node import SampleResultNode


class _MaASRootGraph(RootGraph):
    """RootGraph accepting either a documented envelope or direct sample fields."""
    def invoke(self, input: dict[str, object], attributes: dict[str, object] | None = None):
        if "sample" not in input:
            input = {"sample": dict(input)}
        return super().invoke(input, attributes)


def _architecture_forward(graph: object | None):
    def forward(message: dict[str, object]) -> dict[str, object]:
        request = message["architecture_request"]
        if graph is None:
            result = ArchitectureResult(None, None, None, "no_architecture_graph", None, False, False)
            return {"architecture_result": result}
        payload = request.__dict__ if hasattr(request, "__dict__") else {
            "problem": request.problem, "problem_index": request.problem_index, "entry_point": request.entry_point
        }
        try:
            output = graph.invoke(payload) if hasattr(graph, "invoke") else graph(payload)
            if isinstance(output, tuple):
                output = output[0]
            result = output.get("architecture_result") if isinstance(output, dict) else output
            return {"architecture_result": result}
        except Exception as exc:
            return {"architecture_result": ArchitectureResult(None, None, None, "architecture_error", None, False, False, execution_metadata={"error": type(exc).__name__})}
    return forward


def _build_root(*, train: bool, architecture_graph: object | None = None, scorer: object | None = None,
                batch_accumulator: object | None = None, metrics: object | None = None) -> RootGraph:
    root = _MaASRootGraph(name="maas_train" if train else "maas_test")
    split = root.create_node(InputSplitNode, name="input_split")
    architecture = root.create_node(CustomNode, name="architecture", forward=_architecture_forward(architecture_graph), pull_keys={})
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


def build_train_root_graph(*, architecture_graph: object | None = None, scorer: object | None = None,
                           batch_accumulator: object | None = None, **kwargs: Any) -> RootGraph:
    return _build_root(train=True, architecture_graph=architecture_graph, scorer=scorer, batch_accumulator=batch_accumulator)


def build_test_root_graph(*, architecture_graph: object | None = None, scorer: object | None = None,
                          metrics: object | None = None, **kwargs: Any) -> RootGraph:
    return _build_root(train=False, architecture_graph=architecture_graph, scorer=scorer, metrics=metrics)


__all__ = ["build_train_root_graph", "build_test_root_graph"]
