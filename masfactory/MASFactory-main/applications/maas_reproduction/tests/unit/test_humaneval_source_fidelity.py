from __future__ import annotations

import json

import pytest

from applications.maas_reproduction.components.operators.humaneval_test_graph import (
    HumanEvalTestOutcome,
    HumanEvalTestProtocol,
)
from applications.maas_reproduction.components.operators.registry import OPERATOR_REGISTRY
from applications.maas_reproduction.components.benchmark_completion_graph.humaneval_completion_graph.workflow import (
    HumanEvalCompletionGraph,
)
from applications.maas_reproduction.maas_reproduction.schemas import (
    ArchitectureRequest,
    DispatchState,
    RoutePlan,
)
from applications.maas_reproduction.maas_reproduction.source_compat import (
    HumanEvalPublicTestRepository,
    PublicTestResult,
    parse_solution_letter_xml,
    sanitize,
)


class FakeExecutor:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def run_public_tests(self, solution, cases, entry_point):
        self.calls.append((solution, cases, entry_point))
        return next(self.results)


class FakeRepository:
    def get(self, entry_point):
        assert entry_point == "solve"
        return ("assert candidate() == 1",)


class Reflections:
    def __init__(self):
        self.calls = []

    def __call__(self, payload):
        self.calls.append(dict(payload))
        return {"reflection_and_solution": f"repair-{len(self.calls)}"}


def result(
    passed: bool,
    feedback: str = "failure",
    *,
    execution_succeeded: bool = True,
) -> PublicTestResult:
    return PublicTestResult(
        passed,
        feedback,
        () if passed else (feedback,),
        execution_succeeded=execution_succeeded,
    )


def test_test_protocol_passes_without_reflection() -> None:
    executor = FakeExecutor([result(True, "no error")])
    reflections = Reflections()
    protocol = HumanEvalTestProtocol(
        repository=FakeRepository(), executor=executor,
        reflection_adapter=reflections, max_repairs=3,
    )

    outcome = protocol.run(problem="p", entry_point="solve", solution="initial")

    assert outcome.passed is True
    assert outcome.solution == "initial"
    assert outcome.public_test_runs == 1
    assert outcome.repair_attempts == 0
    assert reflections.calls == []


def test_test_protocol_runs_three_repairs_and_a_fourth_final_test() -> None:
    executor = FakeExecutor([result(False), result(False), result(False), result(True, "no error")])
    reflections = Reflections()
    protocol = HumanEvalTestProtocol(
        repository=FakeRepository(), executor=executor,
        reflection_adapter=reflections, max_repairs=3,
    )

    outcome = protocol.run(problem="p", entry_point="solve", solution="initial")

    assert outcome.passed is True
    assert outcome.solution == "repair-3"
    assert outcome.public_test_runs == 4
    assert outcome.repair_attempts == 3
    assert len(reflections.calls) == 3
    assert [call[0] for call in executor.calls] == ["initial", "repair-1", "repair-2", "repair-3"]


@pytest.mark.parametrize(
    ("public_result", "exec_pass", "test_fail"),
    [
        (result(False, "assertion detail"), "executed successfully", "assertion detail"),
        (
            result(False, "NameError: missing", execution_succeeded=False),
            "executed unsuccessfully, error: \n NameError: missing",
            "executed unsucessfully",
        ),
    ],
)
def test_test_reflection_preserves_source_execution_feedback(
    public_result, exec_pass, test_fail
) -> None:
    executor = FakeExecutor([public_result, result(True, "no error")])
    reflections = Reflections()
    protocol = HumanEvalTestProtocol(
        repository=FakeRepository(), executor=executor,
        reflection_adapter=reflections, max_repairs=1,
    )

    protocol.run(problem="p", entry_point="solve", solution="initial")

    assert reflections.calls[0]["exec_pass"] == exec_pass
    assert reflections.calls[0]["test_fail"] == test_fail


def test_humaneval_test_is_registered_as_a_real_operator() -> None:
    assert OPERATOR_REGISTRY["Test"]["node_class"].__name__ == "HumanEvalTestGraph"


def test_public_repository_preserves_source_string_iteration(tmp_path) -> None:
    path = tmp_path / "public.jsonl"
    path.write_text(
        json.dumps({"entry_point": "custom", "test": "ab"}) + "\n",
        encoding="utf-8",
    )
    repository = HumanEvalPublicTestRepository(path)

    assert repository.get("custom") == ("a", "b")
    assert repository.get("add") == ()


def test_sc_ensemble_accepts_only_source_xml_field() -> None:
    assert parse_solution_letter_xml("<thought>x</thought><solution_letter>b</solution_letter>", 3) == "B"
    with pytest.raises(ValueError, match="lacks"):
        parse_solution_letter_xml('{"solution_letter": "A"}', 3)
    with pytest.raises(ValueError, match="lacks"):
        parse_solution_letter_xml("A", 3)
    with pytest.raises(ValueError, match="range"):
        parse_solution_letter_xml("<solution_letter>D</solution_letter>", 3)


def test_source_sanitize_entrypoint_difference_is_preserved() -> None:
    source = "def helper():\n    return 1\n\ndef solve():\n    return helper()\n\ndef unrelated():\n    return 2"

    entrypoint_filtered = sanitize(source, "solve")
    unfiltered = sanitize(source, None)

    assert "def helper" in entrypoint_filtered
    assert "def solve" in entrypoint_filtered
    assert "def unrelated" not in entrypoint_filtered
    assert "def unrelated" in unfiltered


class FixedProtocol:
    def __init__(self, *, passed: bool):
        self.passed = passed
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        return HumanEvalTestOutcome(
            solution="tested-solution",
            passed=self.passed,
            feedback="last feedback",
            repair_attempts=0 if self.passed else 3,
            public_test_runs=1 if self.passed else 4,
        )


def dispatch_state() -> DispatchState:
    return DispatchState(
        request=ArchitectureRequest("problem", 0, "solve"),
        route_plan=RoutePlan(items=(), policy_log_prob=0.0),
        current_solution="route-solution",
    )


def test_fixed_completion_always_tests_and_returns_repaired_solution() -> None:
    protocol = FixedProtocol(passed=True)
    fallback_calls = []
    graph = HumanEvalCompletionGraph(
        protocol=protocol,
        fallback_adapter=lambda payload: fallback_calls.append(payload),
    )

    result = graph._complete({"dispatch_state": dispatch_state()})

    assert len(protocol.calls) == 1
    assert fallback_calls == []
    assert result["completion_result"]["prediction"] == "tested-solution"
    assert result["completion_result"]["metadata"]["fallback_used"] is False


def test_fixed_completion_uses_one_unretested_fallback_after_failure() -> None:
    protocol = FixedProtocol(passed=False)
    fallback_calls = []

    def fallback(payload):
        fallback_calls.append(dict(payload))
        return {"solution": "fallback-solution"}

    graph = HumanEvalCompletionGraph(protocol=protocol, fallback_adapter=fallback)

    result = graph._complete({"dispatch_state": dispatch_state()})

    assert len(protocol.calls) == 1
    assert len(fallback_calls) == 1
    assert result["completion_result"]["prediction"] == "fallback-solution"
    assert result["completion_result"]["metadata"]["fallback_used"] is True
