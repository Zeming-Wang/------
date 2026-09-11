"""Finalize a dispatch state into the pre-evaluation architecture contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from masfactory.components.custom_node import CustomNode

from maas_reproduction.schemas import (
    ArchitectureResult,
    CostResult,
    DispatchState,
    FailureSource,
)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def finalize_architecture_result(message: dict[str, Any], attributes: dict[str, Any]) -> dict[str, Any]:
    state = message.get("dispatch_state", attributes.get("dispatch_state"))
    completion = message.get("completion_result")
    tracker = attributes.get("cost_tracker") or message.get("cost_tracker")
    before = attributes.get("cost_before", message.get("cost_before"))

    prediction = _value(completion, "prediction")
    if prediction is None and isinstance(state, DispatchState):
        prediction = state.current_solution or (state.candidates[-1] if state.candidates else None)

    error_state = state.error_state if isinstance(state, DispatchState) else None
    failure = _value(completion, "failure_source") or _value(error_state, "failure_source")
    try:
        failure_source = FailureSource(failure) if failure is not None else None
    except (TypeError, ValueError):
        failure_source = FailureSource.ROUTE_EXECUTION

    result_valid = isinstance(state, DispatchState) and bool(prediction and str(prediction).strip())
    status = _value(completion, "status") or ("success" if result_valid and not error_state else "recoverable_failure" if result_valid else "invalid_result")

    cost = CostResult(None, False, "cost tracker unavailable")
    if tracker is not None and callable(getattr(tracker, "snapshot", None)) and callable(getattr(tracker, "delta", None)):
        try:
            after = tracker.snapshot()
            if before is not None:
                measured = tracker.delta(before, after)
                cost = measured if isinstance(measured, CostResult) else CostResult(float(measured), True)
            else:
                cost = CostResult(None, False, "missing cost_before snapshot")
        except Exception as exc:  # cost failures are data, not graph failures
            cost = CostResult(None, False, f"cost snapshot failed: {type(exc).__name__}: {exc}")
    elif isinstance(message.get("cost_result"), CostResult):
        cost = message["cost_result"]

    if not cost.reliable:
        failure_source = failure_source or FailureSource.INFRASTRUCTURE
        if status == "success":
            status = "cost_error"

    route_metadata: dict[str, Any] = {}
    execution_metadata: dict[str, Any] = {"cost_error": cost.error} if cost.error else {}
    if isinstance(state, DispatchState):
        route_metadata = {
            "route_length": len(state.route_plan.items),
            "route_cursor": state.route_cursor,
            "termination_requested": state.termination_requested,
        }
        if error_state:
            execution_metadata["error_state"] = dict(error_state)
    completion_metadata = _value(completion, "metadata")
    if isinstance(completion_metadata, Mapping):
        execution_metadata.update(dict(completion_metadata))

    result = ArchitectureResult(
        prediction=str(prediction) if prediction is not None else None,
        cost_delta=cost.value,
        policy_log_prob=(state.route_plan.policy_log_prob if isinstance(state, DispatchState) else None),
        status=str(status),
        failure_source=failure_source,
        result_valid=result_valid,
        cost_reliable=cost.reliable,
        route_metadata=route_metadata,
        execution_metadata=execution_metadata,
    )
    return {"architecture_result": result}


class FinalizeArchitectureResultNode(CustomNode):
    """Convert final dispatch state to ``ArchitectureResult`` without detaching policy tensors."""

    def __init__(self, name: str = "finalize_architecture_result", **kwargs: Any) -> None:
        super().__init__(name=name, forward=finalize_architecture_result, pull_keys={}, push_keys={}, **kwargs)


__all__ = ["FinalizeArchitectureResultNode", "finalize_architecture_result"]
