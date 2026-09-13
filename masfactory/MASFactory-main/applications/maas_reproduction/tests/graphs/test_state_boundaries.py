from __future__ import annotations

from masfactory.components.custom_node import CustomNode

from applications.maas_reproduction.components.architecture_exec_graph.workflow import ArchitectureExecGraph
from applications.maas_reproduction.components.operator_dispatch_loop.workflow import LOOP_CONTROL_KEYS, OperatorDispatchLoop
from applications.maas_reproduction.maas_reproduction.schemas import ArchitectureRequest, OperatorResult, RouteItem, RoutePlan


class FakeOperator(CustomNode):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            forward=lambda _message: {
                "operator_result": OperatorResult(
                    operator_name=name,
                    status="success",
                    solution="answer",
                )
            },
            pull_keys={},
            push_keys={},
        )


def test_dispatch_controller_message_contains_only_control_fields():
    loop = OperatorDispatchLoop(
        operator_registry={"Generate": {"node_class": FakeOperator}},
    )
    loop.build()

    assert loop._controller.input_keys == LOOP_CONTROL_KEYS
    assert loop._controller.output_keys == LOOP_CONTROL_KEYS
    assert "dispatch_state" not in loop._controller.input_keys
    assert "dispatch_state" not in loop._controller.output_keys


def test_architecture_graph_keeps_query_explicit_and_attributes_empty():
    captured: list[str] = []

    class Controller:
        def forward(self, query, _embeddings, _names):
            captured.append(query)
            return ([0.0], [["Generate"]])

    graph = ArchitectureExecGraph(
        policy_controller=Controller(),
        operator_embeddings=object(),
        operator_registry={"Generate": {"node_class": FakeOperator}},
        dataset="GSM8K",
    )
    graph.build()
    assert "evaluator" not in graph._nodes
    graph._pull_attributes({"expected_answer": "SECRET_SENTINEL", "query": "wrong"})

    assert "expected_answer" not in graph.attributes
    graph._forward({"architecture_request": ArchitectureRequest("real query", 3)})
    assert captured == ["real query"]


def test_loop_business_state_stays_in_local_attributes():
    plan = RoutePlan(
        items=(RouteItem(0, 0, 0, "Generate"),),
        policy_log_prob=0.0,
    )
    from applications.maas_reproduction.maas_reproduction.schemas import DispatchState

    state = DispatchState(ArchitectureRequest("q", 0), plan)
    loop = OperatorDispatchLoop(
        operator_registry={"Generate": {"node_class": FakeOperator}},
    )
    loop.build()
    output = loop._forward({"dispatch_state": state})

    assert output["dispatch_state"].route_cursor == 1
    assert "dispatch_state" not in loop._controller.output_keys
