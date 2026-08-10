"""Config node forward function for the MaAS reproduction workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from maas_reproduction.config import (
    MaASPaths,
    MaASRuntimeSettings,
    OptimizerSettings,
    resolve_model_configs,
)

logger = logging.getLogger(__name__)


def config_forward(input_data: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
    application_root = Path(input_data["application_root"]).expanduser().resolve()
    dataset = str(input_data["dataset"])
    mode = str(input_data.get("mode", "Graph"))
    opt_model_name = str(input_data.get("opt_model_name", "gpt-4o-mini"))
    exec_model_name = str(input_data.get("exec_model_name", "gpt-4o-mini"))

    optimizer = OptimizerSettings(
        sample=int(input_data.get("sample", 4)),
        round_number=int(input_data.get("round_number", 1)),
        batch_size=int(input_data.get("batch_size", 4)),
        learning_rate=float(input_data.get("learning_rate", 0.01)),
        is_textgrad=bool(input_data.get("is_textgrad", False)),
        opt_model_name=opt_model_name,
        exec_model_name=exec_model_name,
    )
    opt_llm_config, exec_llm_config = resolve_model_configs(opt_model_name, exec_model_name)
    settings = MaASRuntimeSettings.from_experiment(
        dataset=dataset,
        mode=mode,
        paths=MaASPaths.from_environment(application_root),
        optimizer=optimizer,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
    )

    logger.info(
        "Configured MaAS reproduction: dataset=%s mode=%s round=%s sample=%s batch_size=%s",
        settings.dataset,
        settings.mode,
        settings.optimizer.round_number,
        settings.optimizer.sample,
        settings.optimizer.batch_size,
    )
    return {
        "settings": settings,
        "dataset": settings.dataset,
        "mode": settings.mode,
        "sample": settings.optimizer.sample,
        "batch_size": settings.optimizer.batch_size,
        "round": settings.optimizer.round_number,
        "model_config": {
            "opt_model_name": settings.optimizer.opt_model_name,
            "exec_model_name": settings.optimizer.exec_model_name,
        },
        "paths": {
            "dataset_file": str(settings.dataset_file),
            "architecture_root": str(settings.architecture_root),
            "run_directory": str(settings.run_directory),
            "checkpoint_path": str(
                settings.paths.controller_checkpoint(
                    settings.dataset,
                    settings.optimizer.round_number,
                    settings.optimizer.sample,
                )
            ),
        },
    }
