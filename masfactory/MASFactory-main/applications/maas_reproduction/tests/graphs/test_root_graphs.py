from __future__ import annotations

from applications.maas_reproduction.maas_reproduction.runtime.dataset_runner import DatasetRunner
from applications.maas_reproduction.maas_reproduction.schemas import ArchitectureResult
from applications.maas_reproduction.workflow import build_test_root_graph, build_train_root_graph


class FakeTensor:
    def detach(self): return self
    def cpu(self): return self
    def item(self): return -0.5


class FakeArchitecture:
    def __init__(self):
        self.requests = []

    def invoke(self, request):
        self.requests.append(dict(request))
        return {"architecture_result": ArchitectureResult(
            prediction=request["problem"], cost_delta=0.1,
            policy_log_prob=FakeTensor(), status="ok",
            failure_source=None, result_valid=True, cost_reliable=True,
        )}


class Accumulator:
    def add(self, *, policy_log_prob, utility):
        return {"update_performed": True, "utility": utility, "skip_reason": None,
                "loss_value": 0.35}


def test_test_graph_isolates_expected_answer_and_does_not_update():
    architecture = FakeArchitecture()
    graph = build_test_root_graph(architecture_graph=architecture, scorer=lambda prediction, expected: float(prediction == expected))
    output, _ = graph.invoke({"sample": {"problem": "42", "problem_index": 7, "expected_answer": "42"}})
    assert "expected_answer" not in architecture.requests[0]
    assert output["sample_result"]["score"] == 1.0
    assert output["sample_result"]["update_performed"] is False
    assert output["sample_result"]["policy_log_prob_value"] == -0.5


def test_train_graph_produces_detached_public_sample_result():
    graph = build_train_root_graph(architecture_graph=FakeArchitecture(), scorer=lambda *_: 1.0,
                                   batch_accumulator=Accumulator())
    output, _ = graph.invoke({"sample": {"problem": "x", "expected_answer": "x"}})
    result = output["sample_result"]
    assert result["utility"] == 0.7
    assert isinstance(result["policy_log_prob_value"], float)
    assert all(not isinstance(value, FakeTensor) for value in result.values())


def test_dataset_runner_invokes_in_order_with_isolated_samples():
    architecture = FakeArchitecture()
    graph = build_test_root_graph(architecture_graph=architecture, scorer=lambda *_: 1.0)
    samples = [{"problem": "a", "expected_answer": "a"}, {"problem": "b", "expected_answer": "b"}]
    results = DatasetRunner(graph=graph, dataset=samples).run()
    assert [request["problem"] for request in architecture.requests] == ["a", "b"]
    assert [result["problem_index"] for result in results] == [0, 1]
    assert samples == [{"problem": "a", "expected_answer": "a"}, {"problem": "b", "expected_answer": "b"}]
