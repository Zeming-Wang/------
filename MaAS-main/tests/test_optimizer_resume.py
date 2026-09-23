import argparse
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace


class _Controller:
    def __init__(self, **_kwargs):
        self.loaded = None

    def to(self, _device):
        return self

    def load_state_dict(self, state):
        self.loaded = state

    def parameters(self):
        return []


class _Adam:
    def __init__(self, _parameters, lr):
        self.lr = lr
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state


def _load_optimizer_module(monkeypatch, payload):
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: name
    torch.optim = SimpleNamespace(Adam=_Adam)
    torch.load = lambda *args, **kwargs: payload
    torch.stack = lambda values: values
    monkeypatch.setitem(sys.modules, "torch", torch)

    evaluator = types.ModuleType("maas.ext.maas.scripts.evaluator")
    evaluator.DatasetType = str
    monkeypatch.setitem(sys.modules, evaluator.__name__, evaluator)

    for module_name, class_name in (
        ("maas.ext.maas.scripts.optimizer_utils.data_utils", "DataUtils"),
        ("maas.ext.maas.scripts.optimizer_utils.experience_utils", "ExperienceUtils"),
        ("maas.ext.maas.scripts.optimizer_utils.evaluation_utils", "EvaluationUtils"),
        ("maas.ext.maas.scripts.optimizer_utils.graph_utils", "GraphUtils"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, class_name, lambda *args, **kwargs: object())
        monkeypatch.setitem(sys.modules, module_name, module)

    logs = types.ModuleType("maas.logs")
    logs.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, logs.__name__, logs)

    model_utils = types.ModuleType("maas.ext.maas.models.utils")
    model_utils.get_sentence_embedding = lambda value: value
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)

    controller = types.ModuleType("maas.ext.maas.models.controller")
    controller.MultiLayerController = _Controller
    monkeypatch.setitem(sys.modules, controller.__name__, controller)

    path = Path(__file__).parents[1] / "maas" / "ext" / "maas" / "scripts" / "optimizer.py"
    spec = importlib.util.spec_from_file_location("optimizer_resume_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optimizer_loads_requested_resume_state(tmp_path, monkeypatch):
    payload = {
        "controller": {"weight": 7},
        "optimizer": {"state": {"step": 4}},
        "repetition": 2,
        "next_batch_idx": 3,
        "total_cost": 1.5,
    }
    module = _load_optimizer_module(monkeypatch, payload)
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")

    optimizer = module.Optimizer(
        dataset="MATH",
        question_type="math",
        opt_llm_config=object(),
        exec_llm_config=object(),
        operators=[],
        sample=4,
        optimized_path="optimized",
        resume=checkpoint,
    )

    assert optimizer.controller.loaded == payload["controller"]
    assert optimizer.optimizer.loaded == payload["optimizer"]
    assert optimizer.resume_repetition == 2
    assert optimizer.resume_next_batch_idx == 3
    assert optimizer.resume_total_cost == 1.5


def test_cli_accepts_resume(monkeypatch):
    models_config = types.ModuleType("maas.configs.models_config")
    models_config.ModelsConfig = object
    monkeypatch.setitem(sys.modules, models_config.__name__, models_config)

    optimizer_module = types.ModuleType("maas.ext.maas.scripts.optimizer")
    optimizer_module.Optimizer = object
    monkeypatch.setitem(sys.modules, optimizer_module.__name__, optimizer_module)

    experiment_configs = types.ModuleType("maas.ext.maas.benchmark.experiment_configs")
    experiment_configs.EXPERIMENT_CONFIGS = {"MATH": object()}
    monkeypatch.setitem(sys.modules, experiment_configs.__name__, experiment_configs)

    path = Path(__file__).parents[1] / "examples" / "maas" / "optimize.py"
    spec = importlib.util.spec_from_file_location("optimize_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["optimize.py", "--dataset", "MATH", "--resume", "checkpoints/latest.pt"],
    )

    args = module.parse_args()

    assert isinstance(args, argparse.Namespace)
    assert args.resume == "checkpoints/latest.pt"
