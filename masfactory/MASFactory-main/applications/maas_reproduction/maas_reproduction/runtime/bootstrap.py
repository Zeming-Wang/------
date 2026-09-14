"""Explicit runtime dependency assembly for MaAS."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .settings import RuntimeSettings
from .seed import seed_everything
from ..models.model_factory import create_shared_model
from ..models.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from ..adapters.dataset_loader import DatasetLoader
from ..benchmarks import GSM8KScorer, MATHScorer, HumanEvalScorer
from ..adapters.cost_tracker import CostTracker
from ..adapters.artifact_store import ArtifactStore
from ..adapters.code_executor import CodeExecutor
from .checkpoint_manager import CheckpointManager
from ..training.batch_accumulator import BatchAccumulator
from .. import contracts
from applications.maas_reproduction.components.operators.registry import OPERATOR_REGISTRY
from applications.maas_reproduction.components.operators.programmer_graph.programmer_core import ProgrammerCore
from applications.maas_reproduction.workflow import build_train_root_graph, build_test_root_graph
from .dependencies import RuntimeDependencies
from .prompt_loader import PromptLoader

RuntimeContext = RuntimeDependencies

class _UsageManager:
    """Small injected cost counter used when no external manager is supplied."""
    def __init__(self, *, input_rate: float | None, output_rate: float | None) -> None:
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.cost_reliable = input_rate is not None and output_rate is not None
        self.cost_reliability_error = (
            None
            if self.cost_reliable
            else "model pricing is not configured; set input/output cost per 1k tokens"
        )
        self.input_rate = input_rate
        self.output_rate = output_rate

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        prompt_tokens = max(int(prompt_tokens), 0)
        completion_tokens = max(int(completion_tokens), 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        if self.cost_reliable:
            self.total_cost += (
                prompt_tokens / 1000.0 * float(self.input_rate)
                + completion_tokens / 1000.0 * float(self.output_rate)
            )

def build_runtime(settings: RuntimeSettings, *, fake: bool = False, manager: Any = None) -> RuntimeContext:
    seed_everything(settings.seed)
    model = create_shared_model(settings.model, fake=fake)
    embedder = FakeEmbeddingProvider() if fake else EmbeddingProvider(settings.embedding_model)
    catalog = contracts.operator_catalog_for(settings.dataset)
    embeddings = embedder.encode_many(catalog)
    from ..models.controller import MultiLayerController
    controller = MultiLayerController(embedding_provider=embedder)
    registry = {name: dict(spec) for name, spec in OPERATOR_REGISTRY.items() if name in catalog}
    usage_manager = manager if manager is not None else _UsageManager(
        input_rate=settings.model.input_cost_per_1k_tokens,
        output_rate=settings.model.output_cost_per_1k_tokens,
    )
    if not callable(getattr(usage_manager, "record_usage", None)):
        raise TypeError("usage manager must provide record_usage(prompt_tokens, completion_tokens)")
    prompt_loader = PromptLoader(Path(__file__).parents[2] / "assets" / "prompts")
    programmer_instruction = prompt_loader.load("Programmer", settings.dataset)

    def _problem_from_payload(payload: dict[str, Any]) -> str:
        problem = payload.get("problem")
        if problem:
            return str(problem)
        invocation = payload.get("operator_invocation")
        if isinstance(invocation, dict):
            return str(invocation.get("problem", ""))
        return str(getattr(invocation, "problem", ""))

    def _invoke_model(payload: dict[str, Any], instruction: str, *, user_content: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        problem = _problem_from_payload(payload)
        feedback = payload.get("feedback", "")
        if user_content is None:
            user_content = problem if not feedback else f"Problem:\n{problem}\n\nFeedback:\n{feedback}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
        ]
        tracker = getattr(model, "token_tracker", None)
        before_input = int(getattr(tracker, "total_input_usage", 0) or 0)
        before_output = int(getattr(tracker, "total_output_usage", 0) or 0)
        response = model.invoke(messages, tools=None)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        # MASFactory OpenAIModel records usage on token_tracker and does not
        # put it in the returned response mapping.  Prefer explicit provider
        # usage when present, otherwise use the tracker delta.
        explicit_input = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        explicit_output = usage.get("completion_tokens") if isinstance(usage, dict) else None
        after_input = int(getattr(tracker, "total_input_usage", before_input) or 0)
        after_output = int(getattr(tracker, "total_output_usage", before_output) or 0)
        prompt_tokens = int(explicit_input) if explicit_input is not None else after_input - before_input
        completion_tokens = int(explicit_output) if explicit_output is not None else after_output - before_output
        usage = dict(usage) if isinstance(usage, dict) else {}
        usage.setdefault("prompt_tokens", prompt_tokens)
        usage.setdefault("completion_tokens", completion_tokens)
        usage.setdefault("total_tokens", prompt_tokens + completion_tokens)
        usage_manager.record_usage(prompt_tokens, completion_tokens)
        return response, usage

    def adapter(payload):
        response, usage = _invoke_model(payload, str(payload.get("instructions", "")))
        return {"solution": response.get("content", ""), "metadata": {"usage": usage}}

    def generate_cot_adapter(payload):
        response, usage = _invoke_model(payload, prompt_loader.load("GenerateCoT", settings.dataset))
        return {"solution": response.get("content", ""), "metadata": {"usage": usage}}

    def self_refine_adapter(payload):
        problem = _problem_from_payload(payload)
        solution = payload.get("current_solution", "")
        content = f"Problem:\n{problem}\n\nCurrent solution:\n{solution}"
        response, usage = _invoke_model(payload, prompt_loader.load("SelfRefine", settings.dataset), user_content=content)
        return {"solution": response.get("content", ""), "metadata": {"usage": usage}}

    def bootstrap_generate_adapter(payload):
        problem = _problem_from_payload(payload)
        code_output = payload.get("current_solution", "")
        content = f"Problem:\n{problem}\n\nCode output:\n{code_output}"
        instruction = "Use the executed code output to produce the final mathematical solution and answer."
        response, usage = _invoke_model(payload, instruction, user_content=content)
        return {"solution": response.get("content", ""), "metadata": {"usage": usage}}

    def sc_ensemble_adapter(payload):
        problem = _problem_from_payload(payload)
        candidates = tuple(payload.get("candidates", ()))
        labeled = "\n\n".join(f"{chr(65 + i)}:\n{candidate}" for i, candidate in enumerate(candidates))
        content = f"Problem:\n{problem}\n\nCandidate solutions:\n{labeled}"
        response, usage = _invoke_model(payload, prompt_loader.load("ScEnsemble", settings.dataset), user_content=content)
        text = str(response.get("content", "")).strip().upper()
        letter = next((char for char in text if "A" <= char <= "Z"), "")
        return {"solution_letter": letter, "metadata": {"usage": usage}}

    def programmer_code_adapter(payload):
        strict_programmer_instruction = (
            f"{programmer_instruction}\n\n"
            "You must output ONLY complete executable Python source code.\n"
            "The output must define a callable zero-argument function named solve.\n"
            "The output must call solve() and print its returned answer.\n"
            "Do not output Markdown fences, prose, explanations, labels, or analysis.\n"
            "Do not return a numeric answer or mathematical solution text directly; return Python code only."
        )
        response, usage = _invoke_model(payload, strict_programmer_instruction)
        return {"code": response.get("content", ""), "metadata": {"usage": usage}}

    executor = CodeExecutor()
    def programmer_executor(payload):
        code = payload.get("code", "")
        result = executor.execute(str(code))
        if result.success:
            return result.stdout
        return {
            "success": False,
            "error": result.error or result.stderr or "program execution failed",
        }

    def bootstrap_programmer_adapter(payload):
        core = ProgrammerCore(generator=programmer_code_adapter, executor=programmer_executor, max_attempts=3)
        result = core.run(payload)
        if result.get("error"):
            return {"operator_name": "Programmer", "status": "failed", "code": result.get("code"),
                    "execution_output": result.get("output"), "metadata": {"attempts": result.get("attempts")}}
        return {"operator_name": "Programmer", "status": "success", "solution": result.get("output"),
                "code": result.get("code"), "execution_output": result.get("output"),
                "metadata": {"attempts": result.get("attempts")}}
    for name, spec in registry.items():
        config = spec.setdefault("config", {})
        # Reuse existing Operator interfaces: ProgrammerGraph names its
        # injected generator ``programmer``; agent graphs use ``operator``.
        if name == "Programmer":
            config["code_generator"] = programmer_code_adapter
            config["executor"] = programmer_executor
        elif name == "GenerateCoT":
            config["operator"] = generate_cot_adapter
            config["instructions"] = prompt_loader.load(name, settings.dataset)
        elif name == "SelfRefine":
            config["operator"] = self_refine_adapter
            config["instructions"] = prompt_loader.load(name, settings.dataset)
        elif name == "ScEnsemble":
            config["operator"] = sc_ensemble_adapter
        elif name == "MultiGenerateCoT":
            config["operator"] = generate_cot_adapter
        else:
            config["operator"] = adapter
            config["instructions"] = prompt_loader.load(name, settings.dataset)
    scorer = {"GSM8K": GSM8KScorer(), "MATH": MATHScorer(), "HumanEval": HumanEvalScorer()}[settings.dataset]
    optimizer = accumulator = None
    if settings.mode == "train":
        try:
            optimizer = __import__("torch").optim.Adam(controller.parameters(), lr=settings.learning_rate)
        except ImportError as exc: raise RuntimeError("training requires PyTorch") from exc
        accumulator = BatchAccumulator(optimizer, settings.batch_size)
    tracker = CostTracker(usage_manager)
    # Resolve the default relative output path against this reproduction's
    # own assets directory, not the process working directory.  This keeps
    # checkpoints and results under applications/maas_reproduction/assets.
    output_root = settings.output_root
    if not output_root.is_absolute() and output_root.as_posix() == "assets/output":
        output_root = Path(__file__).parents[2] / "assets" / "output"
    checkpoint = CheckpointManager(output_root / "checkpoints")
    artifact_store = ArtifactStore(output_root / "results")
    kwargs = dict(policy_controller=controller, operator_embeddings=embeddings, operator_catalog=catalog,
                  operator_registry=registry, dataset=settings.dataset, generate=bootstrap_generate_adapter, programmer=bootstrap_programmer_adapter,
                  scorer=scorer, cost_tracker=tracker, batch_accumulator=accumulator)
    root = build_train_root_graph(**kwargs) if settings.mode == "train" else build_test_root_graph(**kwargs)
    from .dataset_runner import DatasetRunner
    data_root = Path(__file__).parents[2] / "assets" / "data"
    loader = DatasetLoader(data_root)
    from dataclasses import asdict
    try:
        dataset = [asdict(x) for x in loader.load(settings.dataset, settings.split)]
    except FileNotFoundError:
        if settings.mode != "smoke":
            raise
        # Smoke mode is intentionally network- and data-independent.  Keep
        # the evaluator context shape identical to a real sample while using
        # a deterministic fixture when benchmark files are not installed.
        dataset = [{
            "problem_index": 0,
            "dataset": settings.dataset,
            "problem": "What is 6 multiplied by 7?",
            "expected_answer": "42",
            "entry_point": "",
            "test": None,
            "canonical_solution": None,
        }]
    runner = DatasetRunner(graph=root, dataset=dataset, epochs=settings.epochs,
                           batch_accumulator=accumulator, checkpoint_manager=checkpoint,
                           controller=controller if settings.mode == "train" else None,
                           optimizer=optimizer, operator_catalog=catalog,
                           artifact_store=artifact_store)
    return RuntimeContext(settings, model, embedder, controller, embeddings, tuple(catalog), registry, scorer, tracker, checkpoint, optimizer, accumulator, root, runner, artifact_store)

__all__ = ["RuntimeContext", "build_runtime"]
