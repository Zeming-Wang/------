from applications.maas_reproduction.components.operators import (
    GenerateGraph, MultiGenerateCoTGraph, ProgrammerGraph, ScEnsembleGraph,
    SelfRefineGraph,
)
from applications.maas_reproduction.maas_reproduction.schemas import OperatorInvocation


def invocation(name="Generate", solution="", candidates=()):
    return OperatorInvocation(0, 0, 0, name, "solve 1+1", current_solution=solution,
                              candidates=candidates)


def run(graph, item):
    graph.build()
    return graph._forward({"operator_invocation": item})["operator_result"]


def test_shared_native_graphs_return_structured_results():
    for cls, name in ((GenerateGraph, "Generate"), (SelfRefineGraph, "SelfRefine")):
        result = run(cls(operator=lambda _: {"response": "answer"}), invocation(name))
        assert result.operator_name == name
        assert result.solution == "answer"


def test_multi_generate_cot_calls_adapter_three_times():
    calls = []
    result = run(MultiGenerateCoTGraph(operator=lambda _: calls.append(1) or {"solution": str(len(calls))}),
                 invocation("MultiGenerateCoT"))
    assert len(calls) == 3
    assert result.candidates == ("1", "2", "3")


def test_sc_ensemble_invalid_letter_uses_structured_fallback():
    result = run(ScEnsembleGraph(operator=lambda _: {"solution_letter": "Z"}),
                 invocation("ScEnsemble", candidates=("A", "B")))
    assert result.status == "fallback"
    assert result.solution == "A"


def test_programmer_retries_inside_operator_and_returns_timeout_result():
    attempts = []
    graph = ProgrammerGraph(
        code_generator=lambda payload: attempts.append(payload["attempt"]) or {"code": "solve()"},
        executor=lambda _: {"success": False, "error": "timeout"},
        max_attempts=3,
    )
    result = run(graph, invocation("Programmer"))
    assert attempts == [1, 2, 3]
    assert result.operator_name == "Programmer"
    assert result.status == "timeout"
    assert result.metadata["attempts"] == 3


def test_programmer_graph_exposes_internal_retry_control_flow():
    graph = ProgrammerGraph(code_generator=lambda _: {"code": "solve()"},
                            executor=lambda _: "ok")
    graph.build()
    retry_loop = graph._nodes["ProgrammerRetryLoop"]
    assert retry_loop._nodes.keys() >= {
        "CodeGenerationAgent", "CodeParseNode", "ProgrammerExecutionNode",
        "RetryDecisionNode", "ProgrammerLogicSwitch", "ProgrammerResultNode",
    }


def test_missing_adapter_is_a_structured_operator_failure():
    result = run(GenerateGraph(), invocation("Generate"))
    assert result.operator_name == "Generate"
    assert result.status == "failed"
    assert result.metadata["error_type"] == "RuntimeError"
