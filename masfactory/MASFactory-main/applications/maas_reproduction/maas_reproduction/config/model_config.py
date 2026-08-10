"""Model configuration resolution for the migrated MaAS application."""

from __future__ import annotations

import os
import sys


def resolve_model_configs(opt_model_name: str, exec_model_name: str) -> tuple[object, object]:
    """Resolve optimization and execution model configs by name.

    This keeps the behavior of the original MaAS CLI: model names are resolved through
    ``ModelsConfig.default()``, and missing names are reported as explicit configuration errors.
    """
    _ensure_maas_project_root()
    from maas.configs.models_config import ModelsConfig

    models_config = ModelsConfig.default()
    opt_llm_config = models_config.get(opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{opt_model_name}' was not found in the models config."
        )

    exec_llm_config = models_config.get(exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{exec_model_name}' was not found in the models config."
        )

    return opt_llm_config, exec_llm_config


def _ensure_maas_project_root() -> None:
    project_root = os.getenv("METAGPT_PROJECT_ROOT")
    if project_root:
        if project_root in sys.path:
            sys.path.remove(project_root)
        sys.path.insert(0, project_root)
