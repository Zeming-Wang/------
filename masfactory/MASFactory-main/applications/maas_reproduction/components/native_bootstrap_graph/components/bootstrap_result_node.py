"""Final deterministic reducer for the native bootstrap stage."""

from __future__ import annotations

from typing import Any

from masfactory.components.custom_node import CustomNode

from maas_reproduction.maas_reproduction.schemas import (
    ArchitectureRequest,
    DispatchState,
    FailureSource,
    OperatorResult,
    RoutePlan,
)


class BootstrapResultNode(CustomNode):
    """Convert bootstrap operator outputs into the minimal ``DispatchState``."""

    def __init__(self, name: str = "BootstrapResultNode", *, dataset: str = "MATH", **kwargs: Any) -> None:
        self.dataset = dataset
        super().__init__(name, forward=self._forward_bootstrap, **kwargs)

    @staticmethod
    def _solution(result: OperatorResult | None) -> str | None:
        if result is None:
            return None
        return result.solution or (result.candidates[0] if result.candidates else result.code)

    def _forward_bootstrap(self, data: dict[str, Any]) -> dict[str, Any]:
        request = data.get("request")
        route_plan = data.get("route_plan")
        if request is None or route_plan is None:
            return {
                "dispatch_state": None,
                "failure_source": FailureSource.BOOTSTRAP.value,
                "error_state": {"error": "missing request or route_plan"},
            }
        programmer = data.get("programmer_result")
        generated = data.get("generate_result")
        if data.get("programmer_error"):
            return {
                "dispatch_state": None,
                "failure_source": FailureSource.BOOTSTRAP.value,
                "error_state": {"error": data["programmer_error"]},
            }
        if isinstance(programmer, dict):
            programmer = None
        if isinstance(generated, dict):
            generated = None
        if self.dataset not in ("MATH", "math"):
            return {
                "dispatch_state": DispatchState(
                    request=request,
                    route_plan=route_plan,
                    route_cursor=0,
                    current_solution=None,
                    candidates=(),
                    termination_requested=False,
                    error_state=None,
                ),
                "failure_source": None,
                "error_state": None,
            }
        solution = self._solution(generated) or self._solution(programmer)
        if not isinstance(request, ArchitectureRequest) or not isinstance(route_plan, RoutePlan) or not solution:
            return {
                "dispatch_state": None,
                "failure_source": FailureSource.BOOTSTRAP.value,
                "error_state": {"error": "bootstrap did not produce a usable solution"},
            }
        state = DispatchState(
            request=request,
            route_plan=route_plan,
            route_cursor=0,
            current_solution=solution,
            candidates=(solution,),
            termination_requested=False,
            error_state=None,
        )
        return {"dispatch_state": state, "failure_source": None, "error_state": None}


__all__ = ["BootstrapResultNode"]
