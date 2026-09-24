from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from masfactory.adapters.model import Model

from masfactory.compatibility.dify.models import openai_compatible_model_from_dify
from masfactory.compatibility.dify.options import _StubDifyModel


@dataclass
class LlmCompileOptions:
    """Shared LLM compile options for ChatDev / Langflow (and similar importers)."""

    model_factory: Callable[[dict[str, Any]], Model] | None = None
    use_stub_llm: bool = True
    llm_stub_text: str = "stub-llm-response"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    def resolve_model(self, model_config: dict[str, Any]) -> Model:
        if self.model_factory is not None:
            return self.model_factory(model_config)
        if self.use_stub_llm:
            return _StubDifyModel(model_config, default_text=self.llm_stub_text)
        return openai_compatible_model_from_dify(
            model_config,
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
        )


@dataclass
class ChatDevCompileOptions(LlmCompileOptions):
    llm_stub_text: str = "stub-chatdev-response"


@dataclass
class LangflowCompileOptions(LlmCompileOptions):
    llm_stub_text: str = "stub-langflow-response"
