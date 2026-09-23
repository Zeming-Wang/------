import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


class _Movable:
    def to(self, _device):
        return self

    def state_dict(self):
        return {"weight": 1}


class _Optimizer:
    def __init__(self):
        self.steps = 0

    def state_dict(self):
        return {"state": {"step": 2}}

    def step(self):
        self.steps += 1

    def zero_grad(self):
        return None


class _Tensor:
    requires_grad = True

    def to(self, _device):
        return self

    def __mul__(self, _other):
        return self

    def __rmul__(self, _other):
        return self

    def __sub__(self, _other):
        return self

    def __neg__(self):
        return self

    def mean(self):
        return self

    def backward(self):
        return None

    def item(self):
        return 0.0


def _load_benchmark_module(monkeypatch):
    packages = (
        "maas",
        "maas.actions",
        "maas.configs",
        "maas.provider",
        "maas.utils",
        "maas.ext",
        "maas.ext.maas",
        "maas.ext.maas.scripts",
        "maas.ext.maas.scripts.textgrad",
    )
    for name in packages:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.nn = SimpleNamespace(Module=object)
    torch.optim = SimpleNamespace(Optimizer=object)
    torch.float32 = "float32"
    torch.stack = lambda values: _Tensor()
    torch.tensor = lambda *args, **kwargs: _Tensor()
    torch.save = lambda payload, path: Path(path).write_bytes(b"checkpoint")
    monkeypatch.setitem(sys.modules, "torch", torch)

    monkeypatch.setitem(sys.modules, "aiofiles", types.ModuleType("aiofiles"))

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    tqdm = types.ModuleType("tqdm")
    tqdm.__path__ = []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    tqdm_asyncio_module = types.ModuleType("tqdm.asyncio")

    async def gather(*tasks, **_kwargs):
        return await asyncio.gather(*tasks)

    tqdm_asyncio_module.tqdm_asyncio = SimpleNamespace(gather=gather)
    monkeypatch.setitem(sys.modules, "tqdm.asyncio", tqdm_asyncio_module)

    action_node = types.ModuleType("maas.actions.action_node")
    action_node.ActionNode = object
    monkeypatch.setitem(sys.modules, action_node.__name__, action_node)

    models_config = types.ModuleType("maas.configs.models_config")
    models_config.ModelsConfig = object
    monkeypatch.setitem(sys.modules, models_config.__name__, models_config)

    registry = types.ModuleType("maas.provider.llm_provider_registry")
    registry.create_llm_instance = lambda config: None
    monkeypatch.setitem(sys.modules, registry.__name__, registry)

    logs = types.ModuleType("maas.logs")
    logs.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, logs.__name__, logs)

    common = types.ModuleType("maas.utils.common")
    common.write_json_file = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, common.__name__, common)

    script_utils = types.ModuleType("maas.ext.maas.scripts.utils")
    script_utils.extract_random_prompt = lambda *args, **kwargs: ("", "")
    script_utils.update_prompt_in_file = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, script_utils.__name__, script_utils)

    textgrad = types.ModuleType("maas.ext.maas.scripts.textgrad.textual_gradient")
    textgrad.TEXT_GRAD_PROMPT = "{dataset}{prompt_name}{prompt_content}"
    monkeypatch.setitem(sys.modules, textgrad.__name__, textgrad)

    path = Path(__file__).parents[1] / "maas" / "ext" / "maas" / "benchmark" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_resume_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_benchmark(module, log_path):
    class Benchmark(module.BaseBenchmark):
        def __init__(self):
            super().__init__(
                name="MATH",
                file_path="unused.jsonl",
                log_path=str(log_path),
                batch_size=2,
                controller=_Movable(),
                operator_embeddings=_Movable(),
                optimizer=_Optimizer(),
            )
            self.calls = []

        async def evaluate_problem(self, problem, graph):
            self.calls.append(problem["__maas_problem_id"])
            value = problem["value"]
            return (value, value, value, 1.0, 0.1, 0.0)

        def calculate_score(self, expected_output, prediction):
            return 1.0, prediction

        def get_result_columns(self):
            return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

    return Benchmark()


def test_resume_position_moves_to_next_repetition_without_off_by_one(monkeypatch):
    module = _load_benchmark_module(monkeypatch)

    assert module._normalize_resume_position(2, 1, 3) == (2, 1)
    assert module._normalize_resume_position(2, 3, 3) == (3, 0)
    assert module._normalize_resume_position(2, 4, 3) == (3, 0)


def test_training_resume_after_last_batch_starts_next_repetition(tmp_path, monkeypatch):
    module = _load_benchmark_module(monkeypatch)
    benchmark = _make_benchmark(module, tmp_path / "train" / "round_1")
    positions = []
    benchmark.save_training_checkpoint = lambda graph, repetition, next_batch_idx, filename="latest.pt": positions.append(
        (repetition, next_batch_idx, filename)
    )
    graph = SimpleNamespace(
        llm=SimpleNamespace(get_costs=lambda: SimpleNamespace(total_cost=0.0))
    )
    data = [
        {"__maas_problem_id": f"MATH:{index}", "value": str(index)}
        for index in range(5)
    ]

    asyncio.run(
        benchmark.evaluate_all_problems(
            data,
            graph,
            repetitions=3,
            start_repetition=2,
            start_batch_idx=3,
        )
    )

    assert benchmark.calls == [f"MATH:{index}" for index in range(5)]
    assert positions == [
        (3, 1, "latest.pt"),
        (3, 2, "latest.pt"),
        (3, 3, "latest.pt"),
    ]


def test_only_first_resumed_repetition_uses_next_batch_idx(tmp_path, monkeypatch):
    module = _load_benchmark_module(monkeypatch)
    benchmark = _make_benchmark(module, tmp_path / "train" / "round_1")
    positions = []
    benchmark.save_training_checkpoint = lambda graph, repetition, next_batch_idx, filename="latest.pt": positions.append(
        (repetition, next_batch_idx)
    )
    graph = SimpleNamespace(
        llm=SimpleNamespace(get_costs=lambda: SimpleNamespace(total_cost=0.0))
    )
    data = [
        {"__maas_problem_id": f"MATH:{index}", "value": str(index)}
        for index in range(5)
    ]

    asyncio.run(
        benchmark.evaluate_all_problems(
            data,
            graph,
            repetitions=3,
            start_repetition=2,
            start_batch_idx=1,
        )
    )

    assert benchmark.calls == [
        "MATH:2", "MATH:3", "MATH:4",
        "MATH:0", "MATH:1", "MATH:2", "MATH:3", "MATH:4",
    ]
    assert positions == [(2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]


def test_keyboard_interrupt_saves_current_unfinished_batch(tmp_path, monkeypatch):
    module = _load_benchmark_module(monkeypatch)

    class InterruptBenchmark(module.BaseBenchmark):
        def __init__(self):
            super().__init__(
                name="MATH",
                file_path="unused.jsonl",
                log_path=str(tmp_path / "train" / "round_1"),
                batch_size=1,
                controller=_Movable(),
                operator_embeddings=_Movable(),
                optimizer=_Optimizer(),
            )
            self.positions = []

        async def evaluate_problem(self, problem, graph):
            raise KeyboardInterrupt("simulated Ctrl+C")

        def calculate_score(self, expected_output, prediction):
            return 0.0, prediction

        def get_result_columns(self):
            return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

        def save_training_checkpoint(self, graph, repetition, next_batch_idx, filename="latest.pt"):
            self.positions.append((repetition, next_batch_idx, filename))

    benchmark = InterruptBenchmark()
    graph = SimpleNamespace(
        llm=SimpleNamespace(get_costs=lambda: SimpleNamespace(total_cost=0.0))
    )

    with pytest.raises(KeyboardInterrupt, match="simulated Ctrl.C"):
        asyncio.run(
            benchmark.evaluate_all_problems(
                [{"__maas_problem_id": "MATH:0"}],
                graph,
                repetitions=1,
            )
        )

    assert benchmark.positions == [(1, 0, "interrupted.pt")]


def test_training_checkpoint_contains_only_required_resume_fields(tmp_path, monkeypatch):
    module = _load_benchmark_module(monkeypatch)
    benchmark = _make_benchmark(module, tmp_path / "train" / "round_1")
    captured = {}

    def save(payload, path):
        captured.update(payload)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"checkpoint")

    module.torch.save = save
    graph = SimpleNamespace(
        llm=SimpleNamespace(
            get_costs=lambda: SimpleNamespace(total_cost=1.25),
        )
    )

    output = benchmark.save_training_checkpoint(
        graph,
        repetition=2,
        next_batch_idx=3,
    )

    assert output == tmp_path / "train" / "round_1" / "checkpoints" / "latest.pt"
    assert set(captured) == {
        "controller",
        "optimizer",
        "repetition",
        "next_batch_idx",
        "total_cost",
    }
    assert captured["repetition"] == 2
    assert captured["next_batch_idx"] == 3
    assert captured["total_cost"] == 1.25


def test_test_results_are_appended_per_problem_and_skipped_on_restart(tmp_path, monkeypatch):
    module = _load_benchmark_module(monkeypatch)
    log_path = tmp_path / "test" / "round_1"
    data = [
        {"__maas_problem_id": "MATH:0", "value": "zero"},
        {"__maas_problem_id": "MATH:1", "value": "one"},
    ]

    first = _make_benchmark(module, log_path)
    first_results = asyncio.run(first.evaluate_all_problems_test(data, graph=object()))

    result_path = log_path / "test_progress.jsonl"
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert first.calls == ["MATH:0", "MATH:1"]
    assert [row["problem_id"] for row in rows] == ["MATH:0", "MATH:1"]
    assert len(first_results) == 2

    restarted = _make_benchmark(module, log_path)
    restarted_results = asyncio.run(restarted.evaluate_all_problems_test(data, graph=object()))

    assert restarted.calls == []
    assert restarted_results == first_results
    assert len(result_path.read_text(encoding="utf-8").splitlines()) == 2


def test_legacy_results_jsonl_is_migrated_to_named_test_progress(tmp_path, monkeypatch):
    module = _load_benchmark_module(monkeypatch)
    log_path = tmp_path / "test" / "round_1"
    log_path.mkdir(parents=True)
    legacy_path = log_path / "results.jsonl"
    legacy_path.write_text(
        json.dumps({
            "problem_id": "MATH:0",
            "question": "zero",
            "prediction": "zero",
            "expected_output": "zero",
            "score": 1.0,
            "cost": 0.1,
            "logprob": 0.0,
        }) + "\n",
        encoding="utf-8",
    )
    benchmark = _make_benchmark(module, log_path)

    results = asyncio.run(
        benchmark.evaluate_all_problems_test(
            [{"__maas_problem_id": "MATH:0", "value": "zero"}],
            graph=object(),
        )
    )

    assert benchmark.calls == []
    assert len(results) == 1
    assert not legacy_path.exists()
    assert (log_path / "test_progress.jsonl").exists()
