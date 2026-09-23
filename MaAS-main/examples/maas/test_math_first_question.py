"""Run exactly the first MATH test problem with a trained controller.

This is an isolated test harness.  It does not modify the original MaAS
implementation or write a converted checkpoint.  The controller state is
read from the MASFactory checkpoint at runtime, while the execution graph,
operators, and model client remain the original MaAS ones.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "maas" / "ext" / "maas" / "data" / "math_test.json"
DEFAULT_CHECKPOINT = (
    REPO_ROOT.parent
    / "masfactory"
    / "MASFactory-main"
    / "applications"
    / "maas_reproduction"
    / "assets"
    / "output"
    / "run_20260917_135036_948611"
    / "checkpoints"
    / "latest.pt"
)
INPUT_RATE_PER_1K = 0.00015
OUTPUT_RATE_PER_1K = 0.0006


def _install_semantic_kernel_import_shim() -> None:
    """Keep this test runnable when the old optional import is incompatible.

    The original ``maas._compat`` imports ``sk_function`` although this test
    path does not use Semantic Kernel.  The shim lives only in this process;
    no project source file is changed.
    """

    try:
        from semantic_kernel.orchestration import sk_function as _unused

        del _unused
        return
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(
            f"[MaAS-TEST] semantic_kernel import skipped: {type(exc).__name__}",
            flush=True,
        )

    semantic_kernel = ModuleType("semantic_kernel")
    orchestration = ModuleType("semantic_kernel.orchestration")
    orchestration.sk_function = lambda *args, **kwargs: None
    semantic_kernel.orchestration = orchestration
    sys.modules["semantic_kernel"] = semantic_kernel
    sys.modules["semantic_kernel.orchestration"] = orchestration


def _load_reproduction_env() -> None:
    """Load the existing reproduction .env without displaying credentials."""

    env_path = (
        REPO_ROOT.parent
        / "masfactory"
        / "MASFactory-main"
        / "applications"
        / "maas_reproduction"
        / ".env"
    )
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # The command-line environment is still supported without python-dotenv.
        return


def _get_model_config(model_config_name: str):
    from maas.configs.models_config import ModelsConfig

    models_config = ModelsConfig.default()
    llm_config = models_config.get(model_config_name)
    if llm_config is None:
        raise ValueError(
            f"Model config '{model_config_name}' was not found. "
            "Use the existing deepseek-chat entry in MaAS config2.yaml."
        )

    updates = {
        "api_key": os.getenv("OPENAI_API_KEY", llm_config.api_key).strip(),
        "base_url": os.getenv("BASE_URL", llm_config.base_url).strip(),
        "calc_usage": True,
    }
    if hasattr(llm_config, "model_copy"):
        return llm_config.model_copy(update=updates)

    for key, value in updates.items():
        setattr(llm_config, key, value)
    return llm_config


def _read_first_problem(data_path: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"MATH data file not found: {data_path}")

    with data_path.open("r", encoding="utf-8") as data_file:
        first_line = data_file.readline().strip()
    if not first_line:
        raise ValueError(f"MATH data file is empty: {data_path}")

    record = json.loads(first_line)
    if not record.get("problem"):
        raise ValueError("The first MATH record has no 'problem' field")
    return record


def _load_controller(checkpoint_path: Path, device):
    import torch
    from maas.ext.maas.models.controller import MultiLayerController

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}\n"
            "Pass another compatible checkpoint with --checkpoint."
        )

    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch has no weights_only argument
        payload = torch.load(checkpoint_path, map_location=device)

    state_dict = payload.get("controller", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a controller state dictionary")

    controller = MultiLayerController(device=device).to(device)
    controller.load_state_dict(state_dict, strict=True)
    controller.eval()
    return controller


def _build_graph(llm_config, controller, device):
    import torch
    from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils
    from maas.ext.maas.scripts.optimized.MATH.test.template.operator_registry import operator_names
    from maas.ext.maas.models.utils import get_sentence_embedding
    from maas.utils.cost_manager import CostManager

    root_path = "maas/ext/maas/scripts/optimized/MATH"
    graph_utils = GraphUtils(root_path)
    descriptions = graph_utils.load_operators_description_maas(operator_names)
    operator_embeddings = torch.stack([get_sentence_embedding(item) for item in descriptions])

    graph_class = graph_utils.load_graph_maas(f"{root_path}/test")
    graph = graph_class(
        name="MATH",
        llm_config=llm_config,
        dataset="MATH",
        controller=controller,
        operator_embeddings=operator_embeddings,
    )

    # The original graph creates its default CostManager internally.  Replace
    # only that runtime object so billing is identical to maas_reproduction.
    model_names = {"deepseek-chat", "deepseek-flash", str(llm_config.model)}
    token_costs = {
        name: {"prompt": INPUT_RATE_PER_1K, "completion": OUTPUT_RATE_PER_1K}
        for name in model_names
        if name and name != "None"
    }
    graph.llm.cost_manager = CostManager(token_costs=token_costs)
    graph.controller = graph.controller.to(device)

    # DeepSeek may return a valid natural-language answer without the XML
    # tag required by the original ScEnsemble parser.  Keep the API request
    # and its cost, but make this one-problem harness fail soft by retaining
    # the first candidate when only structured-output parsing fails.
    safe_ensemble = _SafeScEnsemble(graph.sc_ensemble)
    graph.sc_ensemble = safe_ensemble
    graph.selection_operator_instances["ScEnsemble"] = safe_ensemble
    return graph


class _SafeScEnsemble:
    def __init__(self, delegate):
        self._delegate = delegate

    async def __call__(self, solutions, problem):
        from pydantic import ValidationError

        try:
            return await self._delegate(solutions=solutions, problem=problem)
        except (ValidationError, KeyError, ValueError) as exc:
            if not solutions:
                raise
            print(
                "[MaAS-TEST] ScEnsemble structured-output parse failed; "
                f"using first candidate ({type(exc).__name__})",
                flush=True,
            )
            return {"response": solutions[0]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the first original MaAS MATH test problem"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Compatible controller checkpoint; defaults to the current MASFactory latest.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Original MaAS MATH JSONL-style data file",
    )
    parser.add_argument(
        "--model-config",
        default="deepseek-chat",
        help="Existing key in MaAS config2.yaml",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="Whole single-problem timeout in seconds",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    import torch

    record = _read_first_problem(args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_config = _get_model_config(args.model_config)
    controller = _load_controller(args.checkpoint, device)
    graph = _build_graph(llm_config, controller, device)

    print(
        f"[MaAS-TEST] dataset=MATH sample_index=0 device={device} "
        f"model={llm_config.model}"
    )
    print(
        "[MaAS-BILLING] input=$0.00015/1K tokens "
        "output=$0.0006/1K tokens"
    )

    try:
        result = await asyncio.wait_for(graph(record["problem"]), timeout=args.timeout)
        prediction, _total_cost, _sum_log_prob = result
        print(f"[MaAS-TEST] first_problem_level={record.get('level', '<unknown>')}")
        print(f"[MaAS-TEST] prediction={prediction}")
    finally:
        costs = graph.llm.get_costs()
        print(
            f"[MaAS-COST] prompt_tokens={costs.total_prompt_tokens} "
            f"completion_tokens={costs.total_completion_tokens} "
            f"total_cost=${float(costs.total_cost):.8f}",
            flush=True,
        )


def main() -> None:
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _install_semantic_kernel_import_shim()
    _load_reproduction_env()
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
