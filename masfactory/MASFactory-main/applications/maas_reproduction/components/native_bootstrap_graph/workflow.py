"""Native bootstrap graph for the MaAS execution pipeline.

The graph deliberately has a small interface.  It does not know about policy
selection or evaluation; it only turns a request and an already selected route
into the initial :class:`DispatchState`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from masfactory.components.custom_node import CustomNode
from masfactory.components.graphs.graph import Graph

from maas_reproduction.maas_reproduction.schemas import (
    ArchitectureRequest,
    DispatchState,
    FailureSource,
    OperatorResult,
    RoutePlan,
)

from .components.bootstrap_result_node import BootstrapResultNode


def _invoke(operator: Any, payload: dict[str, Any]) -> Any:
    """Invoke an injected MASFactory graph, node adapter, or plain callable."""
    if operator is None:
        raise RuntimeError("bootstrap operator is not configured")
    if hasattr(operator, "invoke"):
        result = operator.invoke(payload)
        return result[0] if isinstance(result, tuple) else result
    if callable(operator):
        return operator(payload)
    raise TypeError("bootstrap operator must be callable or expose invoke()")


def _as_result(value: Any, operator_name: str) -> OperatorResult:
    """Normalize an operator response without hiding fatal malformed output."""
    if isinstance(value, OperatorResult):
        return value
    if isinstance(value, Mapping) and "operator_result" in value:
        value = value["operator_result"]
    if isinstance(value, OperatorResult):
        return value
    if isinstance(value, str):
        return OperatorResult(operator_name=operator_name, status="success", solution=value)
    if isinstance(value, Mapping):
        data = dict(value)
        data.setdefault("operator_name", operator_name)
        data.setdefault("status", "success")
        return OperatorResult(**{key: data[key] for key in (
            "operator_name", "status", "solution", "candidates", "code",
            "execution_output", "metadata") if key in data})
    raise TypeError(f"{operator_name} returned no structured result")


class NativeBootstrapGraph(Graph):
    """MATH's fixed Programmer → Generate bootstrap stage.

    ``programmer`` and ``generate`` are injected adapters (usually Graphs or
    callables), which keeps model/provider details outside this graph.  A
    bounded retry is applied to each adapter.  Recoverable structured results
    continue; only an inability to produce a usable solution is fatal.
    """

    def __init__(
        self,
        name: str = "NativeBootstrapGraph",
        *,
        programmer: Any = None,
        generate: Any = None,
        dataset: str = "MATH",
        retry_limit: int = 1,
    ) -> None:
        if isinstance(retry_limit, bool) or not isinstance(retry_limit, int) or retry_limit < 0:
            raise ValueError("retry_limit must be a non-negative integer")
        self.programmer = programmer
        self.generate = generate
        self.dataset = dataset
        self.retry_limit = retry_limit
        super().__init__(name)

    def _run(self, operator: Any, operator_name: str, payload: dict[str, Any]) -> OperatorResult:
        last_error: Exception | None = None
        for _attempt in range(self.retry_limit + 1):
            try:
                return _as_result(_invoke(operator, payload), operator_name)
            except Exception as exc:  # retry belongs to bootstrap's control flow
                last_error = exc
        assert last_error is not None
        raise last_error

    def build(self) -> None:
        if self._is_built:
            return

        def programmer_forward(data: dict[str, Any]) -> dict[str, Any]:
            if self.dataset not in ("MATH", "math"):
                return {"request": data.get("request"), "route_plan": data.get("route_plan"),
                        "programmer_result": None, "programmer_error": None}
            request = data.get("request")
            try:
                result = self._run(self.programmer, "Programmer", {
                    "request": request,
                    "problem": request.problem,
                    "entry_point": request.entry_point,
                    "current_solution": "",
                })
            except Exception as exc:
                return {"request": request, "route_plan": data["route_plan"],
                        "programmer_result": None,
                        "programmer_error": f"{type(exc).__name__}: {exc}"}
            return {"request": request, "route_plan": data["route_plan"],
                    "programmer_result": result, "programmer_error": None}

        def generate_forward(data: dict[str, Any]) -> dict[str, Any]:
            result = data.get("programmer_result")
            if data.get("programmer_error"):
                return {**data, "generate_result": None}
            if result is None:
                return {**data, "generate_result": None}
            solution = result.solution or (result.candidates[0] if result.candidates else result.code)
            if not solution:
                raise ValueError("Programmer produced no bootstrap solution")
            try:
                generated = self._run(self.generate, "Generate", {
                    "request": data["request"],
                    "problem": data["request"].problem,
                    "entry_point": data["request"].entry_point,
                    "current_solution": solution,
                    "candidates": (solution,),
                })
            except Exception as exc:
                return {**data, "generate_result": None,
                        "programmer_error": f"{type(exc).__name__}: {exc}"}
            return {**data, "generate_result": generated, "programmer_error": None}

        class _ProgrammerNode(CustomNode):
            def __init__(node_self, name: str) -> None:
                super().__init__(name, forward=programmer_forward, pull_keys={}, push_keys={})

        class _GenerateNode(CustomNode):
            def __init__(node_self, name: str) -> None:
                super().__init__(name, forward=generate_forward, pull_keys={}, push_keys={})

        programmer_node = self.create_node(_ProgrammerNode, "Programmer")
        generate_node = self.create_node(_GenerateNode, "Generate")
        result_node = BootstrapResultNode(
            "BootstrapResultNode", dataset=self.dataset, pull_keys={}, push_keys={}
        )
        self._nodes[result_node.name] = result_node
        result_node._set_owner(self)
        keys = {"request": "", "route_plan": ""}
        self.edge_from_entry(programmer_node, keys)
        self.create_edge(programmer_node, generate_node, {
            "request": "", "route_plan": "", "programmer_result": "", "programmer_error": ""
        })
        self.create_edge(generate_node, result_node, {
            "request": "", "route_plan": "", "programmer_result": "", "generate_result": "", "programmer_error": ""
        })
        self.edge_to_exit(result_node, {"dispatch_state": "", "failure_source": "", "error_state": ""})
        super().build()


def build_native_bootstrap_graph(**kwargs: Any) -> NativeBootstrapGraph:
    """Construct (but do not invoke) a native bootstrap graph."""
    return NativeBootstrapGraph(**kwargs)


__all__ = ["NativeBootstrapGraph", "build_native_bootstrap_graph"]
