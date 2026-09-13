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
from .checkpoint_manager import CheckpointManager
from ..training.batch_accumulator import BatchAccumulator
from .. import contracts
from applications.maas_reproduction.components.operators.registry import OPERATOR_REGISTRY
from applications.maas_reproduction.workflow import build_train_root_graph, build_test_root_graph
from .dependencies import RuntimeDependencies

RuntimeContext = RuntimeDependencies

class _UsageManager:
    """Small injected cost counter used when no external manager is supplied."""
    def __init__(self) -> None:
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

def build_runtime(settings: RuntimeSettings, *, fake: bool = False, manager: Any = None) -> RuntimeContext:
    seed_everything(settings.seed)
    model = create_shared_model(settings.model, fake=fake)
    embedder = FakeEmbeddingProvider() if fake else EmbeddingProvider(settings.embedding_model)
    catalog = contracts.operator_catalog_for(settings.dataset)
    embeddings = embedder.encode_many(catalog)
    from ..models.controller import MultiLayerController
    controller = MultiLayerController(embedding_provider=embedder)
    registry = {name: dict(spec) for name, spec in OPERATOR_REGISTRY.items() if name in catalog}
    usage_manager = manager if manager is not None else _UsageManager()
    def adapter(payload):
        problem = payload.get("problem", "")
        instruction = payload.get("instructions", "")
        response = model.invoke([{"role":"system","content":instruction}, {"role":"user","content":str(problem)}], tools=None)
        usage = response.get("usage", {})
        usage_manager.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        usage_manager.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        usage_manager.total_cost += float(usage.get("total_tokens", 0) or 0) / 10000.0
        return {"solution": response.get("content", ""), "metadata": {"usage": usage}}
    for name, spec in registry.items():
        config = spec.setdefault("config", {})
        # Reuse existing Operator interfaces: ProgrammerGraph names its
        # injected generator ``programmer``; agent graphs use ``operator``.
        config["programmer" if name == "Programmer" else "operator"] = adapter
    scorer = {"GSM8K": GSM8KScorer(), "MATH": MATHScorer(), "HumanEval": HumanEvalScorer()}[settings.dataset]
    optimizer = accumulator = None
    if settings.mode == "train":
        try:
            optimizer = __import__("torch").optim.Adam(controller.parameters(), lr=settings.learning_rate)
        except ImportError as exc: raise RuntimeError("training requires PyTorch") from exc
        accumulator = BatchAccumulator(optimizer, settings.batch_size)
    tracker = CostTracker(usage_manager)
    checkpoint = CheckpointManager(settings.output_root / "checkpoints")
    artifact_store = ArtifactStore(settings.output_root / "results")
    kwargs = dict(policy_controller=controller, operator_embeddings=embeddings, operator_catalog=catalog,
                  operator_registry=registry, dataset=settings.dataset, generate=adapter, programmer=adapter,
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
