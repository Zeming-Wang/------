from pathlib import Path
from masfactory import Agent, NodeTemplate
from openai import OpenAI
import yaml
import json
from dotenv import load_dotenv

from ddo.services.policy_sampler import PolicyCandidateSampler
from ddo.services.token_usage import TokenUsageRecorder
from ddo.services.utils_data import get_disease_index_dict, get_disease_knowledge, get_symptom_index_dict
from .app_runtime import app_runtime
from .dataset_context import DatasetContext
from .config_types import APIGroupConfig, RuntimeConfig, APIConfigResolver
from .openai_model_compat import ChatCompletionsModelCompat, OpenAIResponsesModelCompat
from ddo.paths import CONFIG_DIR, get_policy_run_dir

load_dotenv()

def _load_yaml(path: str) -> dict:
    path = Path(path)

    if not path.is_absolute():
        path = CONFIG_DIR / path

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_best_settings(dataset_name: str) -> dict:
    settings_path = get_policy_run_dir(dataset_name) / "best_settings.json"

    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_runtime_config() -> RuntimeConfig:
    basic_cfg = _load_yaml("config.yaml")

    api_cfg_resolver = APIConfigResolver()

    api_cfg = APIGroupConfig(
        for_normal_agents=api_cfg_resolver.resolve(
            provider=basic_cfg["llm_for_normal_agents"]["provider"],
            model_name=basic_cfg["llm_for_normal_agents"]["model_name"]
        ),
        for_confidence_agent=api_cfg_resolver.resolve(
            provider=basic_cfg["llm_for_confidence_agent"]["provider"],
            model_name=basic_cfg["llm_for_confidence_agent"]["model_name"]
        )
    )

    return RuntimeConfig(
        basic_cfg=basic_cfg,
        api_cfg=api_cfg,
        best_settings=_load_best_settings(basic_cfg["dataset"]["name"]),
    )

def _build_dataset_context(dataset_name: str) -> DatasetContext:
    disease_index_dict = get_disease_index_dict(dataset_name)
    symptom_index_dict = get_symptom_index_dict(dataset_name)

    candidate_diseases = tuple(disease_index_dict.keys())
    disease_knowledge = get_disease_knowledge(
        dataset_name=dataset_name,
        candidate_diseases=list(candidate_diseases),
    )

    return DatasetContext(
        dataset_name=dataset_name,
        candidate_diseases=candidate_diseases,
        disease_knowledge=disease_knowledge,
        disease_index_dict=disease_index_dict,
        symptom_index_dict=symptom_index_dict,
    )

def bootstrap_app_runtime() -> RuntimeConfig:

    cfg = _build_runtime_config()
    dataset_ctx = _build_dataset_context(cfg.dataset_name)

    sampler = PolicyCandidateSampler(
        dataset_name=dataset_ctx.dataset_name,
        device=cfg.device,
        window_size=cfg.window_size,
        num_samples=cfg.num_samples,
        importance_threshold=cfg.importance_threshold,
    )

    client = OpenAI(
        api_key=cfg.api_cfg.for_confidence_agent.api_key,
        base_url=cfg.api_cfg.for_confidence_agent.base_url,
        # Application-level retries are logged; do not hide SDK retries below them.
        max_retries=0,
        timeout=cfg.request_timeout_seconds,
    )

    model_cls = (
        OpenAIResponsesModelCompat
        if cfg.api_cfg.for_normal_agents.provider == "openai"
        else ChatCompletionsModelCompat
    )

    model = model_cls(
        api_key=cfg.api_cfg.for_normal_agents.api_key,
        base_url=cfg.api_cfg.for_normal_agents.base_url,
        model_name=cfg.api_cfg.for_normal_agents.model_name,
        max_retries=0,
        timeout=cfg.request_timeout_seconds,
    )

    base_agent = NodeTemplate(
        Agent,
        model=model,
        model_settings={
            "temperature": cfg.normal_model_temperature,
        },
    )

    token_usage = TokenUsageRecorder()

    app_runtime.configure(
        cfg=cfg,
        client=client,
        model=model,
        sampler=sampler,
        base_agent=base_agent,
        dataset_ctx=dataset_ctx,
        token_usage=token_usage,
    )

    return cfg
