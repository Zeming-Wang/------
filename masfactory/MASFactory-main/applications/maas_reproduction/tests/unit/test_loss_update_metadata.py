from applications.maas_reproduction.components.loss_update_node import LossUpdateNode
from applications.maas_reproduction.components.sample_result_node import SampleResultNode
from applications.maas_reproduction.components.architecture_exec_graph.components.finalize_architecture_result_node import finalize_architecture_result
from applications.maas_reproduction.maas_reproduction.schemas import ArchitectureRequest, DispatchState, RouteItem, RoutePlan
from applications.maas_reproduction.maas_reproduction.adapters.cost_tracker import CostTracker


class _Manager:
    total_cost = 0.0


class Accumulator:
    def add(self, *, policy_log_prob, utility):
        assert policy_log_prob == "live-policy"
        assert utility == 0.7
        return {"update_performed": True, "loss_value": 1.4}


def test_loss_update_preserves_utility_after_accumulator_step():
    node = LossUpdateNode(batch_accumulator=Accumulator())
    output = node._update(
        {
            "evaluation_result": {
                "result_valid": True,
                "cost_reliable": True,
                "evaluation_reliable": True,
                "policy_log_prob": "live-policy",
                "score": 1.0,
                "cost_delta": 0.1,
            }
        }
    )
    assert output["update_result"]["utility"] == 0.7
    assert output["update_result"]["update_performed"] is True


def test_sample_result_exposes_cost_reliability_error():
    output = SampleResultNode._convert(
        {
            "evaluation_result": {
                "problem_index": 0,
                "prediction": "42",
                "score": 1.0,
                "cost_delta": None,
                "status": "recoverable_failure",
                "failure_source": "infrastructure",
                "result_valid": True,
                "cost_reliable": False,
                "evaluation_reliable": True,
                "policy_log_prob": None,
                "execution_metadata": {
                    "cost_error": "model pricing is not configured"
                },
            },
            "update_result": {
                "update_performed": False,
                "skip_reason": "unreliable_cost",
            },
        }
    )
    assert output["sample_result"]["failure_detail"] == "model pricing is not configured"


def test_recoverable_operator_error_keeps_route_provenance():
    state = DispatchState(
        ArchitectureRequest("solve", 0),
        RoutePlan((RouteItem(0, 0, 0, "Generate"),), object()),
        current_solution="42",
        error_state={"operator_name": "Programmer", "status": "failed"},
    )
    tracker = CostTracker(_Manager())
    result = finalize_architecture_result(
        {"dispatch_state": state},
        {"cost_tracker": tracker, "cost_before": tracker.snapshot()},
    )["architecture_result"]
    assert result.result_valid
    assert result.failure_source.value == "route_execution"
    public = SampleResultNode._convert({
        "evaluation_result": {
            "problem_index": 0,
            "prediction": result.prediction,
            "score": 1.0,
            "cost_delta": None,
            "status": result.status,
            "failure_source": result.failure_source,
            "result_valid": True,
            "cost_reliable": False,
            "evaluation_reliable": True,
            "execution_metadata": result.execution_metadata,
        },
        "update_result": {"update_performed": False},
    })
    assert "Programmer" in public["sample_result"]["failure_detail"]
