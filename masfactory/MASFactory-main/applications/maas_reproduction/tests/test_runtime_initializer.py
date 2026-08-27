import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

from maas_reproduction.config import MaASPaths, MaASRuntimeSettings, OptimizerSettings


class RuntimeInitializerTest(unittest.TestCase):
    def setUp(self) -> None:
        if torch is None:
            raise unittest.SkipTest("torch is required for MaAS runtime initialization tests")

    def test_builds_training_loop_attributes(self) -> None:
        from maas_reproduction.runtime.initializer import build_runtime_attributes

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_minimal_assets(root)
            settings = _settings(root, mode="Graph")

            class Workflow:
                def __init__(self, name, llm_config, dataset, controller, operator_embeddings):
                    self.name = name
                    self.llm_config = llm_config
                    self.dataset = dataset
                    self.controller = controller
                    self.operator_embeddings = operator_embeddings

            with patch(
                "maas_reproduction.runtime.initializer.get_sentence_embedding",
                side_effect=lambda description: torch.ones(384),
            ):
                attrs = build_runtime_attributes(settings, workflow_class=Workflow)

        self.assertIs(attrs["settings"], settings)
        self.assertEqual(len(attrs["problems"]), 1)
        self.assertEqual(attrs["batch_size"], 2)
        self.assertEqual(attrs["problem_index"], 0)
        self.assertEqual(attrs["repetition"], 1)
        self.assertEqual(attrs["previous_cost"], [0.0])
        self.assertEqual(tuple(attrs["operator_embeddings"].shape), (2, 384))
        self.assertEqual(attrs["architecture_workflow"].dataset, "GSM8K")
        self.assertIs(attrs["architecture_workflow"].controller, attrs["controller"])
        self.assertIs(attrs["architecture_workflow"].operator_embeddings, attrs["operator_embeddings"])

    def test_load_workflow_class_supports_relative_template_imports(self) -> None:
        from maas_reproduction.runtime.initializer import load_workflow_class

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = _settings(root, mode="Graph")
            package_root = root / "assets" / "optimized"
            train_root = package_root / "GSM8K" / "train"
            template_root = train_root / "template"
            template_root.mkdir(parents=True)
            for package_dir in (
                package_root,
                package_root / "GSM8K",
                train_root,
                template_root,
            ):
                (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (template_root / "marker.py").write_text("VALUE = 'local-template'\n", encoding="utf-8")
            (train_root / "graph.py").write_text(
                "from .template import marker\n\n"
                "class Workflow:\n"
                "    marker = marker.VALUE\n",
                encoding="utf-8",
            )

            workflow_class = load_workflow_class(settings)

        self.assertEqual(workflow_class.marker, "local-template")

    def test_load_workflow_class_imports_migrated_optimized_assets(self) -> None:
        from maas_reproduction.runtime.initializer import load_workflow_class

        application_root = Path(__file__).resolve().parents[1]
        for dataset in ("GSM8K", "MATH", "HumanEval"):
            for mode in ("Graph", "Test"):
                settings = _settings(application_root, mode=mode, dataset=dataset)

                workflow_class = load_workflow_class(settings)

                self.assertEqual(workflow_class.__name__, "Workflow")


def _settings(root: Path, mode: str, dataset: str = "GSM8K") -> MaASRuntimeSettings:
    return MaASRuntimeSettings(
        dataset=dataset,
        mode=mode,
        paths=MaASPaths.from_application_root(root),
        optimizer=OptimizerSettings(
            sample=1,
            round_number=1,
            batch_size=2,
            learning_rate=0.01,
            is_textgrad=False,
            opt_model_name="opt",
            exec_model_name="exec",
        ),
        question_type="math",
        operators=("Generate", "EarlyStop"),
        opt_llm_config={"model": "opt"},
        exec_llm_config={"model": "exec"},
    )


def _write_minimal_assets(root: Path) -> None:
    data_root = root / "assets" / "data"
    data_root.mkdir(parents=True)
    (data_root / "gsm8k_train.jsonl").write_text(
        json.dumps({"question": "1+1?", "answer": "#### 2"}) + "\n",
        encoding="utf-8",
    )
    template_root = root / "assets" / "optimized" / "GSM8K" / "train" / "template"
    template_root.mkdir(parents=True)
    (template_root / "operator.json").write_text(
        json.dumps(
            {
                "Generate": {"description": "Generate an answer", "interface": "input"},
                "EarlyStop": {"description": "Stop execution", "interface": "none"},
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
