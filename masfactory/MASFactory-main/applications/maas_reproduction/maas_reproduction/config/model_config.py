"""Model configuration resolution for the migrated MaAS application."""

from __future__ import annotations


def resolve_model_configs(opt_model_name: str, exec_model_name: str) -> tuple[object, object]:
    """Resolve optimization and execution model configs by name.

    This keeps the behavior of the original MaAS CLI: model names are resolved through
    ``ModelsConfig.default()``, and missing names are reported as explicit configuration errors.
    """
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

