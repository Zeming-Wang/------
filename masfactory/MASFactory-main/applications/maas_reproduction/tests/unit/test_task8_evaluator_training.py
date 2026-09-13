import pytest
from applications.maas_reproduction.maas_reproduction.benchmarks import GSM8KBenchmark, MATHBenchmark
from applications.maas_reproduction.maas_reproduction.schemas import ArchitectureResult, EvaluationContext, FailureSource
from applications.maas_reproduction.maas_reproduction.training.training_signal import build_training_signal
from applications.maas_reproduction.components.evaluator_node import EvaluatorNode


def test_scorers_follow_dataset_rules():
    assert GSM8KBenchmark().score("work; answer 1,000", "#### 1000") == 1.0
    assert MATHBenchmark().score("therefore \\boxed{2}", "\\boxed{2.0}") == 1.0


def test_evaluator_failure_is_unreliable_and_training_skips():
    node = EvaluatorNode(scorer=GSM8KBenchmark())
    result = node._forward({
        "architecture_result": ArchitectureResult("answer", 0.1, object(), "success", None, True, True),
        "evaluation_context": EvaluationContext("question", 0, "not-a-number"),
    })["evaluation_result"]
    assert not result.evaluation_reliable
    assert result.failure_source is FailureSource.EVALUATION
    signal = build_training_signal(result)
    assert not signal.should_update
    assert signal.skip_reason == "unreliable_evaluation"


def test_failure_source_does_not_change_utility():
    common = {"result_valid": True, "cost_reliable": True, "evaluation_reliable": True, "score": 1.0, "cost_delta": 0.2, "policy_log_prob": object()}
    a = build_training_signal({**common, "failure_source": FailureSource.ROUTE_EXECUTION})
    b = build_training_signal({**common, "failure_source": FailureSource.BOOTSTRAP})
    assert a.utility == b.utility
    assert a.utility == pytest.approx(0.4)
