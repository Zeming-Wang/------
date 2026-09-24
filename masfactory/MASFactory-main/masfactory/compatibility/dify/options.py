from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from masfactory.adapters.model import Model

from masfactory.compatibility.dify.models import openai_compatible_model_from_dify
from masfactory.compatibility.errors import CompatibilityImportError


@dataclass
class DifyCompileOptions:
    """Options for compiling a Dify DSL export into an executable MASFactory graph.

    By default, `llm` nodes use a **real** `OpenAIModel` (see `dify_models.openai_compatible_model_from_dify`).
    Set `use_stub_llm=True` for unit tests without network or API keys.
    """

    model_factory: Callable[[dict[str, Any]], Model] | None = None
    use_stub_llm: bool = False
    llm_stub_text: str = "stub-llm-response"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    tool_executor: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    http_client: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    knowledge_retriever: Callable[[dict[str, Any], str], str] | None = None

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


class _StubDifyModel(Model):
    def __init__(self, model_config: dict[str, Any], *, default_text: str):
        super().__init__(model_name=str(model_config.get("name") or "stub"))
        self._default_text = default_text
        self.last_messages: list[dict] | None = None

    def invoke(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        settings: dict | None = None,
        **kwargs,
    ) -> dict:
        from masfactory.adapters.model.base import ModelResponseType

        self.last_messages = messages
        return {"type": ModelResponseType.CONTENT, "content": self._default_text}
