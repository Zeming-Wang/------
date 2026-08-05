from dataclasses import dataclass
import os

@dataclass(frozen=True)
class ResolvedAPIConfig:
    provider:str
    api_key:str
    base_url:str
    model_name:str

class APIConfigResolver:
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")

    def resolve(self, provider, model_name) -> ResolvedAPIConfig:
        provider = provider.lower().strip()

        if provider == "openrouter":
            api_key = self.openrouter_api_key
            base_url = self.openrouter_base_url
        elif provider == "openai":
            api_key = self.openai_api_key
            base_url = self.openai_base_url
        elif provider == "deepseek":
            api_key = self.deepseek_api_key
            base_url = self.deepseek_base_url
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return ResolvedAPIConfig(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

@dataclass(frozen=True)
class APIGroupConfig:
    for_normal_agents: ResolvedAPIConfig
    for_confidence_agent: ResolvedAPIConfig

@dataclass(frozen=True)
class RuntimeConfig:
    basic_cfg: dict
    api_cfg: APIGroupConfig
    best_settings: dict

    @property
    def dataset_name(self) -> str:
        return self.basic_cfg["dataset"]["name"]

    @property
    def device(self) -> str:
        return self.basic_cfg["general"]["device"]
    
    @property
    def max_turns(self) -> int:
        return int(self.best_settings["max_turns"])

    @property
    def retry(self) -> int:
        return int(self.best_settings["retry"])

    @property
    def max_retry(self) -> int:
        return self.retry

    @property
    def floor_turns(self) -> int:
        return int(self.best_settings["floor_turns"])

    @property
    def window_size(self) -> int:
        return int(self.best_settings["window_size"])
    
    @property
    def top_k(self) -> int:
        return int(self.best_settings["top_k"])
    
    @property
    def reserve_k(self) -> int:
        return int(self.best_settings["top_k"]) + 4

    @property
    def num_samples(self) -> int:
        return int(self.best_settings["num_samples"])

    @property
    def importance_threshold(self) -> float:
        return float(self.best_settings["importance_threshold"])

    @property
    def confidence_model_temperature(self) -> float:
        return float(
            self.basic_cfg["llm_for_confidence_agent"]["temperature"]
        )

    @property
    def normal_model_temperature(self) -> float:
        return float(
            self.basic_cfg["llm_for_normal_agents"]["temperature"]
        )

    @property
    def btp_temperature(self) -> float:
        return float(
            self.basic_cfg["confidence_estimation"]["btp_temperature"]
        )

    @property
    def disease_temperature(self) -> float:
        return float(
            self.basic_cfg["confidence_estimation"]["disease_temperature"]
        )

    @property
    def fallback_margin(self) -> float:
        return float(
            self.basic_cfg["confidence_estimation"]["fallback_margin"]
        )

    @property
    def symptom_status_threshold(self) -> float:
        return float(
            self.best_settings["symptom_status_threshold"]
        )

    @property
    def frequency_threshold(self) -> float:
        return self.symptom_status_threshold

    @property
    def normal_agent_max_attempts(self) -> int:
        return int(self.basic_cfg.get("api_reliability", {}).get("normal_agent_max_attempts", 6))

    @property
    def confidence_agent_max_attempts(self) -> int:
        return int(self.basic_cfg.get("api_reliability", {}).get("confidence_agent_max_attempts", 8))

    @property
    def retry_base_delay_seconds(self) -> float:
        return float(self.basic_cfg.get("api_reliability", {}).get("retry_base_delay_seconds", 1.0))

    @property
    def retry_max_delay_seconds(self) -> float:
        return float(self.basic_cfg.get("api_reliability", {}).get("retry_max_delay_seconds", 30.0))

    @property
    def request_timeout_seconds(self) -> float:
        return float(self.basic_cfg.get("api_reliability", {}).get("request_timeout_seconds", 120.0))
