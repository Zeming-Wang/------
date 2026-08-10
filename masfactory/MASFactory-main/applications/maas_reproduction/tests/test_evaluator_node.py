from types import SimpleNamespace
from pathlib import Path
import unittest

from maas_reproduction.nodes.evaluator_node import evaluator_forward


class EvaluatorNodeTest(unittest.TestCase):
    def test_scores_gsm8k_prediction(self) -> None:
        result = evaluator_forward(
            {
                "problem": "What is 40 + 2?",
                "entry_point": "",
                "expected_answer": "#### 42",
                "prediction": "The answer is 42.",
                "cost": 1.0,
                "logprob": 0.25,
                "problem_index": 0,
            },
            {"settings": SimpleNamespace(dataset="GSM8K")},
        )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["cost"], 1.0)
        self.assertEqual(result["logprob"], 0.25)

    def test_scores_math_prediction(self) -> None:
        result = evaluator_forward(
            {
                "problem": "Compute.",
                "entry_point": "",
                "expected_answer": "\\boxed{7}",
                "prediction": "\\boxed{7}",
                "cost": 2.0,
                "logprob": 0.5,
                "problem_index": 1,
            },
            {"settings": SimpleNamespace(dataset="MATH")},
        )

        self.assertEqual(result["score"], 1)

    def test_scores_humaneval_prediction(self) -> None:
        result = evaluator_forward(
            {
                "problem": "def add(a, b):",
                "entry_point": "add",
                "expected_answer": "def check(fn):\n    assert fn(1, 2) == 3",
                "prediction": "def add(a, b):\n    return a + b",
                "cost": 3.0,
                "logprob": 0.75,
                "problem_index": 2,
            },
            {"settings": SimpleNamespace(dataset="HumanEval", run_directory=Path("."))},
        )

        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
