from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import csv
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

from maas_reproduction.nodes.training_controller import _run_textgrad_update, _summarize_costs, training_controller

try:
    import torch
except ModuleNotFoundError:
    torch = None


class TrainingControllerTest(unittest.TestCase):
    def test_dispatches_problems_and_writes_final_result(self) -> None:
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            run_directory = Path(tmpdir) / "runs"
            settings = SimpleNamespace(
                dataset="GSM8K",
                mode="Graph",
                optimizer=SimpleNamespace(sample=1, round_number=2),
                paths=SimpleNamespace(
                    controller_checkpoint=lambda dataset, round_number, sample: checkpoint_path
                ),
                run_directory=run_directory,
            )
            attrs = {
                "settings": settings,
                "problems": [
                    {"question": "q1", "answer": "1"},
                    {"question": "q2", "answer": "2"},
                ],
                "problem_index": 0,
                "all_scores": [1.0, 0.0],
                "current_repetition_scores": [],
                "sample_results": [
                    ["q1", "pred1", 1.0, 1.0, 0.25, 0.2],
                    ["q2", "pred2", 2.0, 0.0, 0.5, 0.4],
                ],
                "result_columns": ["question", "prediction", "expected_output", "score", "cost", "logprob"],
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
            self.assertEqual(message["checkpoint_path"], str(checkpoint_path))
            self.assertEqual(message["result_path"], str(run_directory))
            self.assertIn("csv_path", message["runtime_metadata"])
            self.assertIn("results_json_path", message["runtime_metadata"])
            self.assertEqual(message["runtime_metadata"]["avg_cost"], 0.25)
            self.assertEqual(message["runtime_metadata"]["total_cost"], 0.5)

    def test_maps_humaneval_problem_fields(self) -> None:
        settings = SimpleNamespace(dataset="HumanEval", optimizer=SimpleNamespace(sample=1, round_number=1))
        attrs = {
            "settings": settings,
            "problems": [{"prompt": "def add(a, b):", "entry_point": "add", "test": "def check(fn): pass"}],
            "problem_index": 0,
            "current_repetition_scores": [],
            "sample_results": [],
            "result_columns": ["inputs", "prediction", "expected_output", "score", "cost", "logprob"],
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
            "current_repetition_scores": [],
            "sample_results": [],
            "result_columns": ["question", "prediction", "expected_output", "score", "cost", "logprob"],
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
                "current_repetition_scores": [],
                "controller": torch.nn.Linear(1, 1),
            }
            message = {}

            self.assertTrue(training_controller(message, attrs))
            self.assertTrue(checkpoint_path.exists())

    def test_writes_author_style_csv_and_train_results_json(self) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is required for checkpoint save test")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "optimized" / "GSM8K" / "train" / "round_1" / "GSM8K_controller_sample2.pth"
            train_root = root / "optimized" / "GSM8K" / "train"
            settings = SimpleNamespace(
                dataset="GSM8K",
                mode="Graph",
                optimizer=SimpleNamespace(sample=1, round_number=1),
                paths=SimpleNamespace(
                    controller_checkpoint=lambda dataset, round_number, sample: checkpoint_path
                ),
                run_directory=train_root / "round_1",
                architecture_root=train_root,
            )
            attrs = {
                "settings": settings,
                "problems": [{"question": "q1", "answer": "1"}],
                "problem_index": 1,
                "all_scores": [1.0],
                "current_repetition_scores": [],
                "result_columns": ["question", "prediction", "expected_output", "score", "cost", "logprob"],
                "sample_results": [["q1", "1", 1.0, 1.0, 0.3, 0.2]],
                "architecture_workflow": SimpleNamespace(
                    llm=SimpleNamespace(
                        cost_manager=SimpleNamespace(
                            total_prompt_tokens=120,
                            total_completion_tokens=30,
                            total_cost=0.3,
                        )
                    )
                ),
                "textgrad_events": [
                    {
                        "applied": True,
                        "prompt_name": "GENERATE_PROMPT",
                        "prompt_path": str(train_root / "template" / "op_prompt.py"),
                    }
                ],
                "controller": torch.nn.Linear(1, 1),
            }
            message = {}

            self.assertTrue(training_controller(message, attrs))

            csv_path = Path(message["runtime_metadata"]["csv_path"])
            self.assertTrue(csv_path.exists())
            self.assertEqual(csv_path.parent, settings.run_directory)
            with csv_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertNotIn("cost", rows[0])
            self.assertEqual(rows[0]["question"], "q1")

            results_data = json.loads((train_root / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(results_data[0]["round"], 1)
            self.assertEqual(results_data[0]["score"], 1.0)
            self.assertEqual(results_data[0]["avg_cost"], 0.3)
            self.assertEqual(results_data[0]["total_cost"], 0.3)
            self.assertEqual(results_data[0]["token"], 150)
            self.assertEqual(results_data[0]["prompt_tokens"], 120)
            self.assertEqual(results_data[0]["completion_tokens"], 30)
            self.assertTrue(results_data[0]["textgrad_applied"])
            self.assertEqual(message["runtime_metadata"]["total_tokens"], 150)
            self.assertTrue(message["runtime_metadata"]["textgrad_applied"])

    def test_summarizes_cumulative_sample_costs_as_deltas(self) -> None:
        avg_cost, total_cost = _summarize_costs(
            [
                ["q1", "1", 1.0, 1.0, 0.1, 0.2],
                ["q2", "2", 2.0, 1.0, 0.25, 0.3],
            ]
        )

        self.assertAlmostEqual(avg_cost, 0.125)
        self.assertAlmostEqual(total_cost, 0.25)

    def test_triggers_textgrad_after_repetition_score_drops(self) -> None:
        settings = SimpleNamespace(
            dataset="GSM8K",
            mode="Graph",
            optimizer=SimpleNamespace(sample=3, round_number=1, is_textgrad=True),
            run_directory=Path("runs") / "round_1",
        )
        attrs = {
            "settings": settings,
            "problems": [{"question": "q1", "answer": "1"}],
            "problem_index": 1,
            "repetition": 2,
            "current_repetition_scores": [0.0],
            "previous_repetition_score": 1.0,
        }
        message = {}

        update_event = {
            "applied": True,
            "prompt_name": "GENERATE_PROMPT",
            "prompt_path": "assets/optimized/GSM8K/train/template/op_prompt.py",
        }
        with patch(
            "maas_reproduction.nodes.training_controller._run_textgrad_update",
            return_value=update_event,
        ) as run_textgrad:
            self.assertFalse(training_controller(message, attrs))

        run_textgrad.assert_called_once_with(settings)
        self.assertTrue(attrs["textgrad_consumed"])
        self.assertTrue(attrs["textgrad_applied"])
        self.assertEqual(attrs["textgrad_events"], [update_event])

    def test_textgrad_uses_configured_optimizer_llm_config(self) -> None:
        with TemporaryDirectory() as tmpdir:
            round_dir = Path(tmpdir) / "round_1"
            template_dir = Path(tmpdir) / "template"
            template_dir.mkdir(parents=True)
            (template_dir / "op_prompt.py").write_text(
                'GENERATE_PROMPT = """Solve {problem}."""\n',
                encoding="utf-8",
            )
            configured_llm_config = {"model": "deepseek-chat"}
            settings = SimpleNamespace(
                dataset="GSM8K",
                run_directory=round_dir,
                optimizer=SimpleNamespace(round_number=1, sample=2),
                opt_llm_config=configured_llm_config,
            )
            filled_node = Mock()
            filled_node.instruct_content.model_dump.return_value = {"prompt": "Carefully solve {problem}."}
            action_factory = Mock()
            action_factory.fill.return_value = _ImmediateCoroutine(filled_node)

            fake_action_node_module = types.SimpleNamespace(
                ActionNode=types.SimpleNamespace(from_pydantic=Mock(return_value=action_factory))
            )
            fake_llm_module = types.SimpleNamespace(create_llm_instance=Mock(return_value="llm"))

            with patch.dict(
                sys.modules,
                {
                    "maas.actions.action_node": fake_action_node_module,
                    "maas.provider.llm_provider_registry": fake_llm_module,
                },
            ):
                _run_textgrad_update(settings)

        fake_llm_module.create_llm_instance.assert_called_once_with(configured_llm_config)


class _ImmediateCoroutine:
    def __init__(self, value):
        self._value = value

    def send(self, value):
        raise StopIteration(self._value)

    def throw(self, typ, val=None, tb=None):
        if val is None:
            val = typ()
        raise val

    def close(self):
        pass

    def __await__(self):
        if False:
            yield None
        return self._value


if __name__ == "__main__":
    unittest.main()
