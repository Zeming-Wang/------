from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from maas_reproduction.nodes.config_node import config_forward


class ConfigNodeTest(unittest.TestCase):
    def test_builds_runtime_settings_from_input(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch(
                "maas_reproduction.nodes.config_node.resolve_model_configs",
                return_value=({"model": "opt"}, {"model": "exec"}),
            ):
                result = config_forward(
                    {
                        "dataset": "GSM8K",
                        "mode": "Graph",
                        "application_root": Path(tmpdir),
                        "sample": 2,
                        "round_number": 3,
                        "batch_size": 5,
                        "learning_rate": 0.02,
                        "is_textgrad": True,
                        "opt_model_name": "opt-model",
                        "exec_model_name": "exec-model",
                    },
                    {},
                )

        settings = result["settings"]
        self.assertEqual(settings.dataset, "GSM8K")
        self.assertEqual(settings.mode, "Graph")
        self.assertEqual(settings.optimizer.sample, 2)
        self.assertEqual(settings.optimizer.round_number, 3)
        self.assertEqual(settings.optimizer.batch_size, 5)
        self.assertEqual(settings.optimizer.learning_rate, 0.02)
        self.assertEqual(settings.operators[0], "Generate")
        self.assertEqual(result["dataset"], "GSM8K")
        self.assertEqual(result["round"], 3)

    def test_uses_maas_cli_defaults(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch(
                "maas_reproduction.nodes.config_node.resolve_model_configs",
                return_value=({"model": "opt"}, {"model": "exec"}),
            ):
                result = config_forward(
                    {
                        "dataset": "MATH",
                        "application_root": Path(tmpdir),
                    },
                    {},
                )

        settings = result["settings"]
        self.assertEqual(settings.mode, "Graph")
        self.assertEqual(settings.optimizer.sample, 4)
        self.assertEqual(settings.optimizer.round_number, 1)
        self.assertEqual(settings.optimizer.batch_size, 4)
        self.assertEqual(settings.optimizer.learning_rate, 0.01)
        self.assertEqual(settings.optimizer.opt_model_name, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
