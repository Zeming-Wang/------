import json
from dataclasses import FrozenInstanceError, fields

import pytest

from maas_reproduction.schemas import (
    ArchitectureRequest,
    ArchitectureResult,
    CostResult,
    DispatchState,
    EvaluationContext,
    EvaluationResult,
    FailureSource,
    OperatorInvocation,
    OperatorResult,
    RouteItem,
    RoutePlan,
    SampleResult,
    TrainingSignal,
)


POLICY_TENSOR = object()


def route_plan(*items: RouteItem) -> RoutePlan:
    return RoutePlan(items=items, policy_log_prob=POLICY_TENSOR)


def test_architecture_request_is_minimal_and_frozen() -> None:
    request = ArchitectureRequest(problem="2 + 2?", problem_index=0)

    assert [item.name for item in fields(request)] == [
        "problem",
        "problem_index",
        "entry_point",
    ]
    assert not hasattr(request, "expected_answer")
    with pytest.raises(FrozenInstanceError):
        request.problem = "changed"  # type: ignore[misc]


def test_expected_answer_is_confined_to_evaluation_context() -> None:
    context = EvaluationContext(
        problem="2 + 2?",
        problem_index=0,
        expected_answer="4",
    )
    invocation = OperatorInvocation(
        sequence_index=0,
        layer_index=0,
        position=0,
        operator_name="Generate",
        problem=context.problem,
    )

    assert context.expected_answer == "4"
    assert not hasattr(invocation, "expected_answer")


def test_route_plan_requires_contiguous_layer_major_sequence_indices() -> None:
    plan = route_plan(
        RouteItem(0, 0, 0, "Generate"),
        RouteItem(1, 0, 1, "Programmer"),
        RouteItem(2, 1, 0, "SelfRefine"),
    )

    assert tuple(item.sequence_index for item in plan.items) == (0, 1, 2)
    assert plan.policy_log_prob is POLICY_TENSOR

    with pytest.raises(ValueError, match="contiguous"):
        route_plan(RouteItem(1, 0, 0, "Generate"))


def test_early_stop_is_the_only_control_marker() -> None:
    marker = RouteItem(0, 1, 0, "EarlyStop", is_control_marker=True)
    assert marker.is_control_marker

    with pytest.raises(ValueError, match="only control marker"):
        RouteItem(0, 0, 0, "Generate", is_control_marker=True)
    with pytest.raises(ValueError, match="must be marked"):
        RouteItem(0, 0, 0, "EarlyStop")


def test_dispatch_state_has_only_frozen_minimal_fields() -> None:
    state = DispatchState(
        request=ArchitectureRequest("problem", 3),
        route_plan=route_plan(),
        candidates=["candidate"],  # type: ignore[arg-type]
    )

    assert [item.name for item in fields(state)] == [
        "request",
        "route_plan",
        "route_cursor",
        "current_solution",
        "candidates",
        "termination_requested",
        "error_state",
    ]
    assert state.candidates == ("candidate",)
    assert not hasattr(state, "cost")
    assert not hasattr(state, "score")
    assert not hasattr(state, "utility")


def test_recoverable_timeout_is_a_legal_operator_result() -> None:
    result = OperatorResult(
        operator_name="Programmer",
        status="timeout",
        execution_output="timeout",
        metadata={"attempts": 3},
    )

    assert result.solution is None
    assert result.execution_output == "timeout"


def test_cost_result_requires_an_explicit_reliability_explanation() -> None:
    assert CostResult(value=0.2, reliable=True).value == 0.2
    unreliable = CostResult(value=-0.1, reliable=False, error="negative delta")
    assert unreliable.error == "negative delta"

    with pytest.raises(ValueError, match="requires an error"):
        CostResult(value=None, reliable=False)


def test_valid_architecture_result_can_retain_failure_attribution() -> None:
    result = ArchitectureResult(
        prediction="fallback answer",
        cost_delta=0.25,
        policy_log_prob=POLICY_TENSOR,
        status="recoverable_failure",
        failure_source=FailureSource.ROUTE_EXECUTION,
        result_valid=True,
        cost_reliable=True,
    )

    assert result.failure_source is FailureSource.ROUTE_EXECUTION
    assert result.policy_log_prob is POLICY_TENSOR


@pytest.mark.parametrize("cost", [None, -0.01, float("inf"), float("nan")])
def test_reliable_cost_rejects_missing_negative_or_non_finite_delta(cost: float | None) -> None:
    with pytest.raises(ValueError):
        ArchitectureResult(
            prediction="answer",
            cost_delta=cost,
            policy_log_prob=POLICY_TENSOR,
            status="success",
            failure_source=None,
            result_valid=True,
            cost_reliable=True,
        )


def test_unreliable_cost_may_retain_diagnostic_delta() -> None:
    result = ArchitectureResult(
        prediction="answer",
        cost_delta=-0.25,
        policy_log_prob=POLICY_TENSOR,
        status="cost_error",
        failure_source=FailureSource.INFRASTRUCTURE,
        result_valid=True,
        cost_reliable=False,
    )
    assert result.cost_delta == -0.25


def test_reliable_evaluation_requires_valid_result_and_finite_score() -> None:
    with pytest.raises(ValueError, match="finite score"):
        EvaluationResult(
            problem_index=0,
            prediction="answer",
            score=None,
            cost_delta=0.1,
            policy_log_prob=POLICY_TENSOR,
            status="success",
            failure_source=None,
            result_valid=True,
            cost_reliable=True,
            evaluation_reliable=True,
        )


def test_training_signal_states_are_coherent() -> None:
    update = TrainingSignal(should_update=True, utility=0.7, skip_reason=None)
    skip = TrainingSignal(
        should_update=False,
        utility=None,
        skip_reason="unreliable_cost",
    )
    assert update.utility == 0.7
    assert skip.skip_reason == "unreliable_cost"

    with pytest.raises(ValueError, match="cannot carry utility"):
        TrainingSignal(should_update=False, utility=0.7, skip_reason="failure")
    with pytest.raises(ValueError, match="unknown training skip_reason"):
        TrainingSignal(should_update=False, utility=None, skip_reason="failure")


def test_sample_result_is_json_serializable_and_has_no_policy_tensor() -> None:
    result = SampleResult(
        problem_index=9,
        prediction="42",
        score=1,
        cost_delta=0.1,
        status="success",
        failure_source="route_execution",  # type: ignore[arg-type]
        result_valid=True,
        cost_reliable=True,
        evaluation_reliable=True,
        policy_log_prob_value=-0.5,
        utility=0.7,
        update_performed=True,
        skip_reason=None,
        loss_value=0.35,
    )

    payload = result.to_dict()
    assert payload["failure_source"] == "route_execution"
    assert "policy_log_prob" not in payload
    assert json.loads(json.dumps(payload))["policy_log_prob_value"] == -0.5


def test_sample_result_rejects_non_scalar_logprob_value() -> None:
    with pytest.raises(TypeError, match="policy_log_prob_value"):
        SampleResult(
            problem_index=0,
            prediction=None,
            score=None,
            cost_delta=None,
            status="failed",
            failure_source=None,
            result_valid=False,
            cost_reliable=False,
            evaluation_reliable=False,
            policy_log_prob_value=POLICY_TENSOR,  # type: ignore[arg-type]
            utility=None,
            update_performed=False,
            skip_reason="invalid_architecture_result",
            loss_value=None,
        )
