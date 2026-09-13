import pytest

from applications.maas_reproduction.components.architecture_exec_graph.components.finalize_architecture_result_node import (
    finalize_architecture_result,
)
from applications.maas_reproduction.maas_reproduction.adapters.cost_tracker import CostTracker
from applications.maas_reproduction.maas_reproduction.schemas import (
    ArchitectureRequest,
    DispatchState,
    FailureSource,
    RouteItem,
    RoutePlan,
)


class Manager:
    def __init__(self, value=0.0):
        self.total_cost = value


def state(policy):
    request = ArchitectureRequest("solve", 0)
    plan = RoutePlan((RouteItem(0, 0, 0, "Generate"),), policy)
    return DispatchState(request, plan, current_solution="answer", candidates=("answer",))


def test_cost_tracker_reports_delta_and_reset():
    manager = Manager(1.0)
    tracker = CostTracker(manager)
    before = tracker.snapshot()
    manager.total_cost = 1.25
    result = tracker.delta(before, tracker.snapshot())
    assert result.reliable and result.value == pytest.approx(0.25)
    manager.total_cost = 0.5
    result = tracker.delta(before, tracker.snapshot())
    assert not result.reliable
    assert "decreased" in result.error


def test_finalize_preserves_live_policy_value_and_reports_cost():
    policy = object()
    manager = Manager(2.0)
    tracker = CostTracker(manager)
    before = tracker.snapshot()
    manager.total_cost = 2.4
    output = finalize_architecture_result(
        {"dispatch_state": state(policy)},
        {"cost_tracker": tracker, "cost_before": before},
    )
    result = output["architecture_result"]
    assert result.prediction == "answer"
    assert result.result_valid and result.cost_reliable
    assert result.cost_delta == pytest.approx(0.4)
    assert result.policy_log_prob is policy


def test_finalize_missing_cost_is_unreliable_and_does_not_fabricate_zero():
    result = finalize_architecture_result({"dispatch_state": state(object())}, {})[
        "architecture_result"
    ]
    assert result.result_valid
    assert not result.cost_reliable
    assert result.cost_delta is None
    assert result.failure_source is FailureSource.INFRASTRUCTURE
