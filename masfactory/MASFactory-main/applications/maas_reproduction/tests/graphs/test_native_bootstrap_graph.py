from applications.maas_reproduction.components.native_bootstrap_graph import NativeBootstrapGraph
from applications.maas_reproduction.maas_reproduction.schemas import (
    ArchitectureRequest,
    FailureSource,
    RouteItem,
    RoutePlan,
)


def _plan() -> RoutePlan:
    return RoutePlan((RouteItem(0, 0, 0, "Generate"),), policy_log_prob=1.0)


def test_math_bootstrap_builds_initial_dispatch_state() -> None:
    seen: list[str] = []

    def programmer(payload):
        seen.append("Programmer")
        return {"solution": "initial"}

    def generate(payload):
        seen.append("Generate")
        assert payload["current_solution"] == "initial"
        return {"solution": "refined"}

    graph = NativeBootstrapGraph(programmer=programmer, generate=generate)
    graph.build()
    output = graph._forward({"request": ArchitectureRequest("x", 0), "route_plan": _plan()})

    state = output["dispatch_state"]
    assert seen == ["Programmer", "Generate"]
    assert state.current_solution == "refined"
    assert state.candidates == ("refined",)
    assert state.route_cursor == 0
    assert output["failure_source"] is None


def test_bootstrap_retries_and_reports_fatal_failure_without_state() -> None:
    attempts = 0

    def programmer(_payload):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("provider down")

    graph = NativeBootstrapGraph(programmer=programmer, generate=lambda _: {"solution": "x"}, retry_limit=2)
    graph.build()
    output = graph._forward({"request": ArchitectureRequest("x", 0), "route_plan": _plan()})

    assert attempts == 3
    assert output["dispatch_state"] is None
    assert output["failure_source"] == FailureSource.BOOTSTRAP.value


def test_non_math_bootstrap_is_a_noop_and_does_not_call_operators() -> None:
    graph = NativeBootstrapGraph(
        dataset="GSM8K",
        programmer=lambda _: (_ for _ in ()).throw(AssertionError()),
        generate=lambda _: (_ for _ in ()).throw(AssertionError()),
    )
    graph.build()
    output = graph._forward({"request": ArchitectureRequest("x", 0), "route_plan": _plan()})
    assert output["dispatch_state"].current_solution is None
    assert output["failure_source"] is None
