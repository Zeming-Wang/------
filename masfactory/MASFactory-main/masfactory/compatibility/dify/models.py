"""Map Dify `model` blocks to MASFactory `Model` adapters (real API by default)."""

from __future__ import annotations

import os
from typing import Any

from masfactory.adapters.model import Model
from masfactory.adapters.model.openai import OpenAIModel

from masfactory.compatibility.errors import CompatibilityImportError

_NON_OPENAI_SDK_PROVIDERS = frozenset(
    {"anthropic", "google", "gemini", "vertex_ai", "bedrock", "cohere", "azure"}
)


def openai_compatible_model_from_dify(
    model_config: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Model:
    """Build an `OpenAIModel` from a Dify node `model` dict."""
    prov = str(model_config.get("provider") or "openai").lower()
    if prov in _NON_OPENAI_SDK_PROVIDERS:
        raise CompatibilityImportError(
            f"Dify LLM provider {model_config.get('provider')!r} needs a dedicated adapter. "
            "Pass `DifyCompileOptions(model_factory=...)`."
        )

    key = api_key if api_key not in (None, "") else os.getenv("OPENAI_API_KEY")
    if not key:
        raise CompatibilityImportError(
            "Real Dify LLM execution requires `OPENAI_API_KEY` or "
            "`DifyCompileOptions(openai_api_key=...)` / `use_stub_llm=True` for tests."
        )

    url = base_url if base_url not in (None, "") else (
        os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or None
    )
    if url == "":
        url = None

    name = model_config.get("name") or os.getenv("OPENAI_MODEL_NAME") or "gpt-4o-mini"
    completion = model_config.get("completion_params")
    invoke_settings = dict(completion) if isinstance(completion, dict) else None

    return OpenAIModel(
        model_name=str(name),
        api_key=key,
        base_url=url,
        invoke_settings=invoke_settings,
    )
