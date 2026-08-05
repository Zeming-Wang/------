from dataclasses import dataclass
from typing import Any
from .config_types import RuntimeConfig
from .dataset_context import DatasetContext

@dataclass
class AppRuntime:
    cfg: RuntimeConfig | None = None
    dataset_ctx: DatasetContext | None = None
    client: Any | None = None
    model: Any | None = None
    sampler: Any | None = None
    base_agent: Any | None = None
    token_usage: Any | None = None

    def configure(
        self,
        *,
        cfg: RuntimeConfig,
        dataset_ctx: DatasetContext,
        client: Any,
        model: Any,
        sampler: Any,
        base_agent: Any,
        token_usage: Any,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.model = model
        self.sampler = sampler
        self.base_agent = base_agent
        self.dataset_ctx = dataset_ctx
        self.token_usage = token_usage

app_runtime = AppRuntime()
