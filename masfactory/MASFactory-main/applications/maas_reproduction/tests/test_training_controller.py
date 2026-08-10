from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest

from maas_reproduction.nodes.training_controller import training_controller

try:
    import torch
except ModuleNotFoundError:
    torch = None


class TrainingControllerTest(unittest.TestCase):
    def test_dispatches_problems_and_writes_final_result(self) -> None:
        settings = SimpleNamespace(
            dataset="GSM8K",
            mode="Graph",
            optimizer=SimpleNamespace(sample=1, round_number=2),
            paths=SimpleNamespace(
                controller_checkpoint=lambda dataset, round_number, sample: Path("checkpoint.pth")
            ),
            run_directory=Path("runs"),
        )
        attrs = {
            "settings": settings,
            "problems": [
                {"question": "q1", "answer": "1"},
                {"question": "q2", "answer": "2"},
            ],
            "problem_index": 0,
            "all_scores": [1.0, 0.0],
            "controller": torch.nn.Linear(1, 1) if torch is not None else None,
        }
        message = {"result_score": 1.0}

        self.assertFalse(training_controller(message, attrs))
        self.assertEqual(message["problem"], "q1")
        self.assertNotIn("result_score", message)

        self.assertFalse(training_controller(message, attrs))
        self.assertEqual(message["problem"], "q2")

        self.assertTrue(training_controller(message, attrs))
        self.assertEqual(message["average_score"], 0.5)
        self.assertEqual(message["round"], 2)
        self.assertEqual(message["checkpoint_path"], "checkpoint.pth")
        self.assertEqual(message["result_path"], str(Path("runs")))

    def test_maps_humaneval_problem_fields(self) -> None:
        settings = SimpleNamespace(dataset="HumanEval", optimizer=SimpleNamespace(sample=1, round_number=1))
        attrs = {
            "settings": settings,
            "problems": [{"prompt": "def add(a, b):", "entry_point": "add", "test": "def check(fn): pass"}],
            "problem_index": 0,
        }
        message = {}

        terminated = training_controller(message, attrs)

        self.assertFalse(terminated)
        self.assertEqual(message["problem"], "def add(a, b):")
        self.assertEqual(message["entry_point"], "add")
        self.assertEqual(message["expected_answer"], "def check(fn): pass")

    def test_resets_problem_index_for_next_repetition(self) -> None:
        settings = SimpleNamespace(dataset="GSM8K", optimizer=SimpleNamespace(sample=2, round_number=1))
        attrs = {
            "settings": settings,
            "problems": [{"question": "q1", "answer": "1"}],
            "problem_index": 0,
        }
        message = {}

        self.assertFalse(training_controller(message, attrs))
        self.assertEqual(message["problem"], "q1")
        self.assertEqual(attrs["problem_index"], 1)

        self.assertFalse(training_controller(message, attrs))
        self.assertEqual(message["problem"], "q1")
        self.assertEqual(attrs["repetition"], 2)
        self.assertEqual(attrs["problem_index"], 1)

    def test_saves_graph_mode_controller_checkpoint_on_final_result(self) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is required for checkpoint save test")

        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "round_1" / "GSM8K_controller_sample1.pth"
            settings = SimpleNamespace(
                dataset="GSM8K",
                mode="Graph",
                optimizer=SimpleNamespace(sample=1, round_number=1),
                paths=SimpleNamespace(
                    controller_checkpoint=lambda dataset, round_number, sample: checkpoint_path
                ),
                run_directory=Path(tmpdir) / "runs",
            )
            attrs = {
                "settings": settings,
                "problems": [{"question": "q1", "answer": "1"}],
                "problem_index": 1,
                "all_scores": [1.0],
                "controller": torch.nn.Linear(1, 1),
            }
            message = {}

            self.assertTrue(training_controller(message, attrs))
            self.assertTrue(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
