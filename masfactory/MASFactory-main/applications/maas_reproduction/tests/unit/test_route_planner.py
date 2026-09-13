from applications.maas_reproduction.maas_reproduction.schemas import ArchitectureRequest, FailureSource
from applications.maas_reproduction.components.architecture_exec_graph.components.route_planner_node import RoutePlannerNode


CATALOG = ("Generate", "Programmer", "SelfRefine", "EarlyStop")


class Scalar:
    requires_grad = True

    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return Scalar(self.value + (other.value if isinstance(other, Scalar) else other))

    __radd__ = __add__


class Controller:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def forward(self, problem, embeddings, names):
        self.calls += 1
        return self.result


def invoke(node, request):
    return node._forward({"architecture_request": request})


def test_planner_flattens_layers_and_preserves_live_logprob():
    logprob = Scalar(-2.0)
    controller = Controller(([logprob, Scalar(-1.0)], [["Generate", "Programmer"], ["SelfRefine"]]))
    node = RoutePlannerNode(controller, object(), CATALOG)
    output = invoke(node, ArchitectureRequest("problem", 0))
    plan = output["route_plan"]
    assert [item.operator_name for item in plan.items] == ["Generate", "Programmer", "SelfRefine"]
    assert [item.sequence_index for item in plan.items] == [0, 1, 2]
    assert plan.policy_log_prob.requires_grad
    assert controller.calls == 1


def test_first_layer_early_stop_is_generate_with_adjustment():
    lp = Scalar(-0.5)
    node = RoutePlannerNode(Controller(([lp], [["EarlyStop"]])), object(), CATALOG)
    result = invoke(node, ArchitectureRequest("p", 0))["route_plan"]
    assert [x.operator_name for x in result.items] == ["Generate"]
    assert result.policy_log_prob.value == -2.0


def test_later_early_stop_is_marker_and_truncates_following_layers():
    node = RoutePlannerNode(Controller(([Scalar(-1.0)] * 3, [["Generate"], ["EarlyStop"], ["SelfRefine"]])), object(), CATALOG)
    result = invoke(node, ArchitectureRequest("p", 0))["route_plan"]
    assert [(x.operator_name, x.is_control_marker) for x in result.items] == [("Generate", False), ("EarlyStop", True)]


def test_replay_mode_does_not_call_controller():
    controller = Controller(([], []))
    node = RoutePlannerNode(controller, object(), CATALOG, replay_routes={3: (("Generate",), ("EarlyStop",))})
    result = invoke(node, ArchitectureRequest("p", 3))["route_plan"]
    assert [x.operator_name for x in result.items] == ["Generate", "EarlyStop"]
    assert controller.calls == 0


def test_planning_failure_is_structured():
    node = RoutePlannerNode(Controller(([], [["Unknown"]])), object(), CATALOG)
    result = invoke(node, ArchitectureRequest("p", 0))
    assert result["route_plan"] is None
    assert result["failure_source"] is FailureSource.PLANNING
    assert result["result_valid"] is False
