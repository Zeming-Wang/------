import asyncio
import csv
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

class _Movable:
    def to(self, _device):
        return self

    def state_dict(self):
        return {}


def _load_base_benchmark(monkeypatch):
    """Load the target module without importing MaAS's optional full stack."""
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
    torch.save = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch)

    aiofiles = types.ModuleType("aiofiles")
    monkeypatch.setitem(sys.modules, "aiofiles", aiofiles)

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    tqdm = types.ModuleType("tqdm")
    tqdm.__path__ = []
    monkeypatch.setitem(sys.modules, "tqdm", tqdm)
    tqdm_asyncio_module = types.ModuleType("tqdm.asyncio")
    tqdm_asyncio_module.tqdm_asyncio = SimpleNamespace(gather=None)
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
    logs.logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
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

    benchmark_path = Path(__file__).parents[1] / "maas" / "ext" / "maas" / "benchmark" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_cost_summary_under_test", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseBenchmark


def _make_benchmark_class(base_benchmark):
    class _Benchmark(base_benchmark):
        def __init__(self, log_path, *, fail=False):
            super().__init__(
                name="MATH",
                file_path="unused.jsonl",
                log_path=str(log_path),
                batch_size=1,
                controller=_Movable(),
                operator_embeddings=_Movable(),
                optimizer=None,
            )
            self.fail = fail

        async def load_data(self, specific_indices=None):
            return [{"problem": "one"}, {"problem": "two"}]

        async def evaluate_problem(self, problem, graph):
            raise NotImplementedError

        def calculate_score(self, expected_output, prediction):
            raise NotImplementedError

        def get_result_columns(self):
            return ["question", "prediction", "expected_output", "score", "cost", "logprob"]

        def save_results_to_csv(self, results, columns):
            return sum(result[3] for result in results) / len(results)

        async def evaluate_all_problems_test(self, data, graph, max_concurrent_tasks=10):
            if self.fail:
                raise RuntimeError("run failed")
            return [
                ("one", "1", "1", 1.0, 0.01, 0.0),
                ("two", "0", "1", 0.0, 0.02, 0.0),
            ]

        async def evaluate_all_problems(
            self,
            data,
            graph,
            max_concurrent_tasks=30,
            repetitions=4,
            is_textgrad=False,
            start_repetition=1,
            start_batch_idx=0,
            resume_total_cost=0.0,
        ):
            return [
                ("one", "1", "1", 1.0, 0.01, 0.0),
                ("two", "0", "1", 0.0, 0.02, 0.0),
            ] * repetitions

    return _Benchmark


class _LLM:
    def get_costs(self):
        return SimpleNamespace(
            total_prompt_tokens=120,
            total_completion_tokens=30,
            total_cost=0.012345678,
        )


def _read_summary(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_complete_test_run_appends_one_main_graph_summary(tmp_path, monkeypatch):
    log_path = tmp_path / "optimized" / "MATH" / "test" / "round_1"
    log_path.mkdir(parents=True)
    benchmark_class = _make_benchmark_class(_load_base_benchmark(monkeypatch))
    benchmark = benchmark_class(log_path)
    graph = SimpleNamespace(llm=_LLM())

    score = asyncio.run(
        benchmark.run_evaluation(graph, va_list=None, is_test=True, sample=3)
    )

    rows = _read_summary(tmp_path / "optimized" / "MATH" / "cost_token_summary.csv")
    assert score == pytest.approx(0.5)
    assert len(rows) == 1
    assert rows[0] == {
        "timestamp": rows[0]["timestamp"],
        "mode": "test",
        "dataset": "MATH",
        "sample_count": "3",
        "problem_count": "2",
        "average_score": "0.50000",
        "prompt_tokens": "120",
        "completion_tokens": "30",
        "total_tokens": "150",
        "total_cost": "0.01234568",
        "status": "success",
    }


def test_failed_run_appends_failed_row_and_preserves_exception(tmp_path, monkeypatch):
    log_path = tmp_path / "optimized" / "MATH" / "test" / "round_1"
    log_path.mkdir(parents=True)
    benchmark_class = _make_benchmark_class(_load_base_benchmark(monkeypatch))
    benchmark = benchmark_class(log_path, fail=True)
    graph = SimpleNamespace(llm=_LLM())

    with pytest.raises(RuntimeError, match="run failed"):
        asyncio.run(
            benchmark.run_evaluation(graph, va_list=None, is_test=True, sample=3)
        )

    rows = _read_summary(tmp_path / "optimized" / "MATH" / "cost_token_summary.csv")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["problem_count"] == "2"
    assert rows[0]["average_score"] == ""
    assert rows[0]["prompt_tokens"] == "120"
    assert rows[0]["completion_tokens"] == "30"


def test_complete_train_run_records_loaded_problem_count_not_repetitions(tmp_path, monkeypatch):
    log_path = tmp_path / "optimized" / "MATH" / "train" / "round_1"
    log_path.mkdir(parents=True)
    benchmark_class = _make_benchmark_class(_load_base_benchmark(monkeypatch))
    benchmark = benchmark_class(log_path)
    graph = SimpleNamespace(llm=_LLM())

    score = asyncio.run(
        benchmark.run_evaluation(graph, va_list=None, is_test=False, sample=3)
    )

    rows = _read_summary(tmp_path / "optimized" / "MATH" / "cost_token_summary.csv")
    assert score == pytest.approx(0.5)
    assert len(rows) == 1
    assert rows[0]["mode"] == "train"
    assert rows[0]["sample_count"] == "3"
    assert rows[0]["problem_count"] == "2"
    assert rows[0]["status"] == "success"
