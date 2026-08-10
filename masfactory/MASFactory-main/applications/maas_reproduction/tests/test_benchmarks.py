import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from maas_reproduction.benchmarks.gsm8k import GSM8KScorer
from maas_reproduction.benchmarks.humaneval import HumanEvalScorer
from maas_reproduction.benchmarks.math import MATHScorer


class BenchmarkScorersTest(unittest.TestCase):
    def test_gsm8k_scores_last_extracted_number(self) -> None:
        scorer = GSM8KScorer()

        score, extracted = scorer.calculate_score(42.0, "First 41, therefore the answer is 42.")

        self.assertEqual(score, 1.0)
        self.assertEqual(extracted, 42.0)

    def test_math_scores_boxed_answer(self) -> None:
        scorer = MATHScorer()

        score, extracted = scorer.calculate_score("We get \\boxed{7}.", "Thus \\boxed{7}.")

        self.assertEqual(score, 1)
        self.assertEqual(extracted, "7")

    def test_humaneval_executes_check_function(self) -> None:
        scorer = HumanEvalScorer()
        solution = "def add(a, b):\n    return a + b"
        test = "def check(fn):\n    assert fn(1, 2) == 3"

        result = scorer.check_solution(solution, test, "add")

        self.assertEqual(result[0], scorer.PASS)

    def test_humaneval_writes_error_log(self) -> None:
        with TemporaryDirectory() as tmpdir:
            scorer = HumanEvalScorer(log_path=Path(tmpdir))

            result = scorer.check_solution("def wrong():\n    return 1", "def check(fn): pass", "missing")

            self.assertEqual(result[0], scorer.FAIL)
            self.assertTrue((Path(tmpdir) / "error.log").exists())


if __name__ == "__main__":
    unittest.main()
