"""Native MASFactory operator dispatch loop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from masfactory.components.controls.logic_switch import LogicSwitch
from masfactory.components.graphs.loop import Loop

from .components.invalid_operator_node import InvalidOperatorNode
from .components.logic_switch import operator_route
from .components.route_cursor_node import RouteCursorNode
from .components.state_reducer_node import StateReducerNode

DISPATCH_LOOP_KEYS = {"dispatch_state": "Current complete operator dispatch state."}


def should_terminate(message: dict, attributes: dict) -> bool:
    state = message["dispatch_state"]
    return state.termination_requested or state.route_cursor >= len(state.route_plan.items)


class OperatorDispatchLoop(Loop):
    """Execute registered operators in flattened RoutePlan order."""

    def __init__(
        self,
        name: str = "operator_dispatch_loop",
        *,
        operator_registry: Mapping[str, Mapping[str, Any]] | None = None,
        max_iterations: int = 100,
    ) -> None:
        if isinstance(max_iterations, bool) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        self.operator_registry = dict(operator_registry or {})
        super().__init__(
            name=name,
            max_iterations=max_iterations,
            terminate_condition_function=should_terminate,
            pull_keys={},
            push_keys=DISPATCH_LOOP_KEYS,
        )

    def build(self) -> None:
        if self._is_built:
            return
        cursor = self.create_node(RouteCursorNode, "route_cursor")
        switch = self.create_node(LogicSwitch, "operator_switch", pull_keys={}, push_keys={})
        invalid = self.create_node(InvalidOperatorNode, "invalid_operator")
        reducer = self.create_node(StateReducerNode, "state_reducer")

        self.edge_from_controller(cursor, keys=DISPATCH_LOOP_KEYS)
        self.create_edge(cursor, switch, {"operator_invocation": "Current operator invocation."})

        operator_nodes: dict[str, Any] = {}
        for operator_name, spec in self.operator_registry.items():
            node_class = spec.get("node_class")
            config = dict(spec.get("config") or {})
            if node_class is None:
                raise ValueError(f"operator registry entry {operator_name!r} lacks node_class")
            operator_nodes[operator_name] = self.create_node(node_class, operator_name, **config)
            self.create_edge(
                operator_nodes[operator_name], reducer,
                {"operator_result": "Structured OperatorResult."},
            )
            edge = self.create_edge(
                switch, operator_nodes[operator_name],
                {"operator_invocation": "Current operator invocation."},
            )
            switch.condition_binding(operator_route(operator_name), edge)

        invalid_edge = self.create_edge(
            switch, invalid,
            {"operator_invocation": "Current operator invocation."},
        )
        switch.condition_binding(
            lambda message, attributes: (
                (message["operator_invocation"].operator_name
                 if hasattr(message["operator_invocation"], "operator_name")
                 else message["operator_invocation"]["operator_name"])
                not in operator_nodes
            ),
            invalid_edge,
        )
        self.create_edge(
            invalid, reducer,
            {"operator_result": "Structured OperatorResult."},
        )
        self.edge_to_controller(reducer, keys=DISPATCH_LOOP_KEYS)
        super().build()


__all__ = ["DISPATCH_LOOP_KEYS", "OperatorDispatchLoop", "should_terminate"]
