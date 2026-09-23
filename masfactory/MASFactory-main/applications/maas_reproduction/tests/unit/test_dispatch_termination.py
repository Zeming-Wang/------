import pytest

from applications.maas_reproduction.components.operator_dispatch_loop.components.route_cursor_node import (
    route_cursor_forward,
)
from applications.maas_reproduction.components.operator_dispatch_loop.components.state_reducer_node import (
    reduce_result,
)
from applications.maas_reproduction.components.operator_dispatch_loop.workflow import LoopControllerMessage, should_terminate
from applications.maas_reproduction.maas_reproduction.schemas import (
    ArchitectureRequest,
    DispatchState,
    OperatorResult,
    RouteItem,
    RoutePlan,
)


def make_state(*names: str, cursor: int = 0, termination_requested: bool = False):
    plan = RoutePlan(
        items=tuple(
            RouteItem(i, 0, i, name, is_control_marker=name == "EarlyStop")
            for i, name in enumerate(names)
        ),
        policy_log_prob=object(),
    )
    return DispatchState(
        request=ArchitectureRequest("problem", 0),
        route_plan=plan,
        route_cursor=cursor,
        termination_requested=termination_requested,
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (make_state("Generate"), False),
        (make_state("Generate", termination_requested=True), True),
        (make_state("Generate", cursor=1), True),
        (make_state(), True),
        (make_state("Generate", cursor=2), True),
    ],
)
def test_should_terminate_truth_table(state, expected):
    control = LoopControllerMessage(
        cursor=state.route_cursor,
        iteration=0,
        should_continue=not expected,
    )
    assert should_terminate({"loop_control": control}, {}) is expected


def test_route_cursor_only_creates_an_invocation_snapshot():
    state = make_state("Generate")
    output = route_cursor_forward(
        {"loop_control": LoopControllerMessage(0, 0, True)},
        {"dispatch_state": state},
    )

    invocation = output["operator_invocation"]
    assert invocation.operator_name == "Generate"
    assert invocation.problem == "problem"
    assert state.route_cursor == 0


def test_route_cursor_rejects_end_of_route():
    with pytest.raises(IndexError):
        route_cursor_forward(
            {"loop_control": LoopControllerMessage(0, 0, True)},
            {"dispatch_state": make_state(cursor=0)},
        )


def test_reducer_returns_new_state_and_advances_cursor():
    previous = make_state("Generate", "SelfRefine")
    updated = reduce_result(
        previous,
        OperatorResult(
            operator_name="Generate",
            status="success",
            solution="answer",
            candidates=("alternative", "answer"),
        ),
    )

    assert updated is not previous
    assert updated.route_cursor == 1
    assert updated.current_solution == "answer"
    assert updated.candidates == ("alternative", "answer")
    assert previous.route_cursor == 0
    assert previous.candidates == ()


def test_reducer_marks_early_stop_without_penalty_fields():
    previous = make_state("EarlyStop")
    updated = reduce_result(
        previous, OperatorResult(operator_name="EarlyStop", status="control")
    )

    assert updated.termination_requested is True
    assert updated.route_cursor == 1
    assert not hasattr(updated, "utility")
    assert updated.error_state is None


def test_loop_builds_early_stop_control_branch():
    from applications.maas_reproduction.components.operator_dispatch_loop.workflow import OperatorDispatchLoop

    loop = OperatorDispatchLoop(operator_registry={})
    loop.build()
    assert "early_stop_control" in loop._nodes
    assert "EarlyStop" not in loop.operator_registry
