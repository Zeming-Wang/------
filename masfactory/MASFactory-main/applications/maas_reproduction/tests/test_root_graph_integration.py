from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

from maas_reproduction.workflow import build_maas_reproduction_graph


class RootGraphIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is required for MaAS graph integration tests")

    def test_root_graph_invokes_training_loop_with_runtime_attributes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = _settings(Path(tmpdir))
            attrs = {
                "settings": settings,
                "architecture_workflow": FakeWorkflow(),
                "problems": [{"question": "1+1?", "answer": "#### 2"}],
                "problem_index": 0,
                "repetition": 1,
                "batch_size": 1,
                "batch_logprobs": [],
                "batch_scores": [],
                "batch_costs": [],
                "all_scores": [],
                "current_repetition_scores": [],
                "sample_results": [],
                "result_columns": ["question", "prediction", "expected_output", "score", "cost", "logprob"],
                "previous_cost": [0.0],
                "previous_repetition_score": None,
                "optimizer": OptimizerSpy(),
                "controller": torch.nn.Linear(1, 1),
                "device": torch.device("cpu"),
            }
            graph = build_maas_reproduction_graph()
            graph.build()

            with patch(
                "maas_reproduction.nodes.config_node.resolve_model_configs",
                return_value=({"model": "opt"}, {"model": "exec"}),
            ):
                output, runtime_attrs = graph.invoke(
                    {
                        "application_root": Path(tmpdir),
                        "dataset": "GSM8K",
                        "mode": "Graph",
                        "sample": 1,
                        "round_number": 1,
                        "batch_size": 1,
                        "learning_rate": 0.01,
                        "is_textgrad": False,
                        "opt_model_name": "opt",
                        "exec_model_name": "exec",
                    },
                    attributes=attrs,
                )

            self.assertEqual(output["average_score"], 1.0)
            self.assertEqual(output["runtime_metadata"]["processed_problems"], 1)
            self.assertTrue(Path(output["checkpoint_path"]).exists())
            self.assertEqual(runtime_attrs["all_scores"], [1.0])
            self.assertEqual(attrs["optimizer"].steps, 1)

    def test_root_graph_runs_two_consecutive_batches(self) -> None:
        with TemporaryDirectory() as tmpdir:
            settings = _settings(Path(tmpdir))
            controller = torch.nn.Linear(1, 1)
            attrs = {
                "settings": settings,
                "architecture_workflow": ControllerLogprobWorkflow(controller),
                "problems": [
                    {"question": "1+1?", "answer": "#### 2"},
                    {"question": "1+1?", "answer": "#### 2"},
                    {"question": "1+1?", "answer": "#### 2"},
                    {"question": "1+1?", "answer": "#### 2"},
                ],
                "problem_index": 0,
                "repetition": 1,
                "batch_size": 2,
                "batch_logprobs": [],
                "batch_scores": [],
                "batch_costs": [],
                "all_scores": [],
                "current_repetition_scores": [],
                "sample_results": [],
                "result_columns": ["question", "prediction", "expected_output", "score", "cost", "logprob"],
                "previous_cost": [0.0],
                "previous_repetition_score": None,
                "optimizer": OptimizerSpy(),
                "controller": controller,
                "device": torch.device("cpu"),
            }
            graph = build_maas_reproduction_graph()
            graph.build()

            with patch(
                "maas_reproduction.nodes.config_node.resolve_model_configs",
                return_value=({"model": "opt"}, {"model": "exec"}),
            ):
                output, runtime_attrs = graph.invoke(
                    {
                        "application_root": Path(tmpdir),
                        "dataset": "GSM8K",
                        "mode": "Graph",
                        "sample": 1,
                        "round_number": 1,
                        "batch_size": 2,
                        "learning_rate": 0.01,
                        "is_textgrad": False,
                        "opt_model_name": "opt",
                        "exec_model_name": "exec",
                    },
                    attributes=attrs,
                )

            self.assertEqual(output["average_score"], 1.0)
            self.assertEqual(runtime_attrs["all_scores"], [1.0, 1.0, 1.0, 1.0])
            self.assertEqual(attrs["optimizer"].steps, 2)
            self.assertEqual(attrs["batch_logprobs"], [])


class FakeWorkflow:
    async def __call__(self, problem: str):
        return "2", 0.1, torch.tensor(0.2, requires_grad=True)


class ControllerLogprobWorkflow:
    def __init__(self, controller) -> None:
        self._controller = controller
        self._calls = 0

    async def __call__(self, problem: str):
        self._calls += 1
        return "2", 0.1 * self._calls, self._controller.weight.sum() * self._calls


class OptimizerSpy:
    def __init__(self) -> None:
        self.steps = 0
        self.zeroes = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self) -> None:
        self.zeroes += 1


def _settings(root: Path):
    return SimpleNamespace(
        dataset="GSM8K",
        mode="Graph",
        optimizer=SimpleNamespace(sample=1, round_number=1),
        paths=SimpleNamespace(
            controller_checkpoint=lambda dataset, round_number, sample: root
            / "round_1"
            / "GSM8K_controller_sample1.pth"
        ),
        run_directory=root / "runs",
    )


if __name__ == "__main__":
    unittest.main()
