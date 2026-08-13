from pathlib import Path
from types import SimpleNamespace
import asyncio
import unittest

from maas_reproduction.nodes.architecture_exec_node import (
    _MAAS_WORKFLOW_MAX_RETRIES,
    _MAAS_WORKFLOW_TIMEOUT_SECONDS,
    architecture_exec_forward,
)


class MathWorkflow:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, problem: str) -> tuple[str, float, float]:
        self.calls.append(problem)
        return "42", 1.5, 0.25


class HumanEvalWorkflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(
        self,
        problem: str,
        entry_point: str,
        log_path: str,
    ) -> tuple[str, float, float]:
        self.calls.append((problem, entry_point, log_path))
        return "def add(a, b): return a + b", 2.0, 0.5


class FlakyWorkflow:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, problem: str) -> tuple[str, float, float]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return problem, 0.0, 0.1


class FailingWorkflow:
    async def __call__(self, problem: str) -> tuple[str, float, float]:
        raise RuntimeError("permanent failure")


class ArchitectureExecNodeTest(unittest.TestCase):
    def test_uses_bounded_smoke_timeout_and_retry_defaults(self) -> None:
        self.assertEqual(_MAAS_WORKFLOW_TIMEOUT_SECONDS, 200)
        self.assertEqual(_MAAS_WORKFLOW_MAX_RETRIES, 2)

    def test_runs_math_workflow_and_preserves_problem_fields(self) -> None:
        workflow = MathWorkflow()
        attributes = {
            "settings": SimpleNamespace(dataset="GSM8K", run_directory=Path("runs/gsm8k")),
            "architecture_workflow": workflow,
        }
        result = architecture_exec_forward(
            {
                "problem": "What is 40 + 2?",
                "entry_point": "",
                "expected_answer": "42",
                "problem_index": 3,
            },
            attributes,
        )

        self.assertEqual(workflow.calls, ["What is 40 + 2?"])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["cost"], 1.5)
        self.assertEqual(result["logprob"], 0.25)
        self.assertEqual(result["expected_answer"], "42")
        self.assertEqual(result["problem_index"], 3)

    def test_runs_humaneval_workflow_with_entry_point_and_log_path(self) -> None:
        workflow = HumanEvalWorkflow()
        attributes = {
            "settings": SimpleNamespace(dataset="HumanEval", run_directory=Path("runs/humaneval")),
            "architecture_workflow": workflow,
        }
        result = architecture_exec_forward(
            {
                "problem": "def add(a, b):",
                "entry_point": "add",
                "expected_answer": "tests",
                "problem_index": 1,
            },
            attributes,
        )

        self.assertEqual(
            workflow.calls,
            [("def add(a, b):", "add", str(Path("runs") / "humaneval"))],
        )
        self.assertEqual(result["prediction"], "def add(a, b): return a + b")
        self.assertEqual(result["cost"], 2.0)
        self.assertEqual(result["logprob"], 0.5)

    def test_retries_temporary_workflow_failure(self) -> None:
        workflow = FlakyWorkflow()
        attributes = {
            "settings": SimpleNamespace(dataset="GSM8K", run_directory=Path("runs/gsm8k")),
            "architecture_workflow": workflow,
        }
        result = architecture_exec_forward(
            {
                "problem": "retry me",
                "entry_point": "",
                "expected_answer": "retry me",
                "problem_index": 0,
            },
            attributes,
        )

        self.assertEqual(workflow.calls, 2)
        self.assertEqual(result["prediction"], "retry me")
        self.assertEqual(result["logprob"], 0.1)

    def test_marks_permanent_workflow_failure_as_zero_cost_result(self) -> None:
        attributes = {
            "settings": SimpleNamespace(dataset="GSM8K", run_directory=Path("runs/gsm8k")),
            "architecture_workflow": FailingWorkflow(),
        }
        with self.assertLogs("maas_reproduction.nodes.architecture_exec_node", "ERROR"):
            result = architecture_exec_forward(
                {
                    "problem": "fail me",
                    "entry_point": "",
                    "expected_answer": "42",
                    "problem_index": 9,
                },
                attributes,
            )

        self.assertEqual(result["prediction"], "permanent failure")
        self.assertEqual(result["cost"], 0.0)
        self.assertEqual(result["logprob"], 0.0)

    def test_running_inside_event_loop_raises_context_error(self) -> None:
        workflow = MathWorkflow()
        attributes = {
            "settings": SimpleNamespace(dataset="GSM8K", run_directory=Path("runs/gsm8k")),
            "architecture_workflow": workflow,
        }

        async def run_node() -> None:
            architecture_exec_forward(
                {
                    "problem": "What is 40 + 2?",
                    "entry_point": "",
                    "expected_answer": "42",
                    "problem_index": 0,
                },
                attributes,
            )

        with self.assertRaises(RuntimeError):
            asyncio.run(run_node())


if __name__ == "__main__":
    unittest.main()
