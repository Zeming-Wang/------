from masfactory.components.custom_node import CustomNode

from applications.maas_reproduction.components.operator_dispatch_loop.workflow import (
    LOOP_CONTROL_KEYS,
    OperatorDispatchLoop,
)
from applications.maas_reproduction.maas_reproduction.schemas import OperatorResult


class FakeOperator(CustomNode):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            forward=lambda message: {
                "operator_result": OperatorResult(
                    operator_name=name, status="success", solution="answer"
                )
            },
            pull_keys={},
            push_keys={},
        )


def test_dispatch_loop_builds_controller_routes_and_feedback():
    loop = OperatorDispatchLoop(
        operator_registry={"Generate": {"node_class": FakeOperator, "config": {}}}
    )
    loop.build()

    assert loop.check_built()
    assert loop._controller.input_keys == LOOP_CONTROL_KEYS
    assert loop._controller.output_keys == LOOP_CONTROL_KEYS
    assert "dispatch_state" not in loop._controller.input_keys
    assert "dispatch_state" not in loop._controller.output_keys
    assert {node.name for node in loop._nodes.values()} == {
        "route_cursor",
        "operator_switch",
        "invalid_operator",
        "state_reducer",
        "Generate",
    }


def test_registry_stores_blueprints_and_not_graph_instances():
    registry = {"Generate": {"node_class": FakeOperator, "config": {}}}
    loop = OperatorDispatchLoop(operator_registry=registry)

    assert loop.operator_registry["Generate"]["node_class"] is FakeOperator
    assert not any(hasattr(value, "_nodes") for value in loop.operator_registry.values())
