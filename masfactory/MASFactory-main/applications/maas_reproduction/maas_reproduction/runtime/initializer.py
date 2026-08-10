"""Runtime object initialization for the MaAS TrainingLoop."""

from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Iterable

import torch

from maas_reproduction.models.controller import MultiLayerController
from maas_reproduction.models.utils import get_sentence_embedding
from maas_reproduction.runtime.data_loader import load_problems

logger = logging.getLogger(__name__)


def build_runtime_attributes(
    settings,
    specific_indices: Iterable[int] | None = None,
    workflow_class=None,
) -> dict[str, object]:
    """Create the state objects consumed through TrainingLoop attributes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = MultiLayerController(device=device).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=settings.optimizer.learning_rate)
    operator_embeddings = _load_operator_embeddings(settings, device)

    if settings.mode == "Test":
        checkpoint_path = settings.paths.controller_checkpoint(
            settings.dataset,
            settings.optimizer.round_number,
            settings.optimizer.sample,
        )
        controller.load_state_dict(torch.load(checkpoint_path, map_location=device))
        controller.eval()

    workflow_type = workflow_class or load_workflow_class(settings)
    architecture_workflow = workflow_type(
        name=settings.dataset,
        llm_config=settings.exec_llm_config,
        dataset=settings.dataset,
        controller=controller,
        operator_embeddings=operator_embeddings,
    )
    settings.run_directory.mkdir(parents=True, exist_ok=True)
    problems = load_problems(settings, specific_indices=specific_indices)

    logger.info(
        "Initialized MaAS runtime objects: dataset=%s mode=%s problems=%s",
        settings.dataset,
        settings.mode,
        len(problems),
    )
    return {
        "settings": settings,
        "controller": controller,
        "optimizer": optimizer,
        "operator_embeddings": operator_embeddings,
        "architecture_workflow": architecture_workflow,
        "problems": problems,
        "problem_index": 0,
        "repetition": 1,
        "batch_index": 0,
        "batch_logprobs": [],
        "batch_scores": [],
        "batch_costs": [],
        "all_scores": [],
        "previous_cost": 0.0,
        "batch_size": settings.optimizer.batch_size,
        "device": device,
        "run_directory": settings.run_directory,
    }


def load_workflow_class(settings):
    maas_project_root = os.getenv("METAGPT_PROJECT_ROOT")
    if maas_project_root:
        if maas_project_root in sys.path:
            sys.path.remove(maas_project_root)
        sys.path.insert(0, maas_project_root)
    application_root = str(settings.paths.application_root)
    if application_root in sys.path:
        sys.path.remove(application_root)
    sys.path.insert(0, application_root)
    split = "test" if settings.mode == "Test" else "train"
    module_name = f"assets.optimized.{settings.dataset}.{split}.graph"
    _clear_assets_modules()
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module.Workflow


def _clear_assets_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "assets" or module_name.startswith("assets."):
            del sys.modules[module_name]


def _load_operator_embeddings(settings, device: torch.device) -> torch.Tensor:
    descriptions = _load_operator_descriptions(settings)
    return torch.stack([get_sentence_embedding(description) for description in descriptions]).to(device)


def _load_operator_descriptions(settings) -> list[str]:
    operator_file = settings.paths.optimized_split_root(settings.dataset, "train") / "template" / "operator.json"
    with Path(operator_file).open("r", encoding="utf-8") as file:
        operator_data = json.load(file)

    descriptions = []
    for index, operator_name in enumerate(settings.operators):
        operator_info = operator_data[operator_name]
        descriptions.append(
            f"{index}. {operator_name}: {operator_info['description']}, with interface {operator_info['interface']}."
        )
    return descriptions
