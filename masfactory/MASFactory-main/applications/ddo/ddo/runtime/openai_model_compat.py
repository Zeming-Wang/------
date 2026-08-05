import json

from masfactory.adapters.model.legacy_openai import LegacyOpenAIModel
from masfactory.adapters.model.openai import OpenAIModel

from ddo.runtime.openai_retry import call_with_openai_retry


class ChatCompletionsModelCompat(LegacyOpenAIModel):
    """OpenAI-compatible chat-completions model for non-OpenAI providers."""

    def invoke(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        settings: dict | None = None,
        **kwargs,
    ) -> dict:
        from masfactory.adapters.model.common import canonical_tool_calls, content_to_text

        tools_dict = [{"type": "function", "function": tool} for tool in tools] if tools else None
        max_attempts, base_delay, max_delay = _normal_agent_retry_settings(kwargs)

        chat_messages = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                chat_messages.append(
                    {
                        "role": "tool",
                        "content": content_to_text(message.get("content")),
                        "tool_call_id": message.get("tool_call_id"),
                    }
                )
                continue

            tool_calls = canonical_tool_calls(message)
            if role == "assistant" and tool_calls:
                chat_messages.append(
                    {
                        "role": "assistant",
                        "content": content_to_text(message.get("content")) or None,
                        "tool_calls": [
                            {
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tool_call.get("name"),
                                    "arguments": json.dumps(tool_call.get("arguments", {}), ensure_ascii=False),
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    }
                )
                continue

            chat_messages.append(
                {
                    "role": role,
                    "content": self._encode_chat_content(message.get("content")),
                }
            )

        def request():
            request_kwargs = {
                "model": self.model_name,
                "messages": chat_messages,
                **self._parse_settings(settings),
                **kwargs,
            }
            if tools_dict is not None:
                request_kwargs["tools"] = tools_dict
            self._disable_deepseek_thinking_if_needed(request_kwargs)
            return self._client.chat.completions.create(**request_kwargs)

        response = call_with_openai_retry(
            request,
            operation_name="normal_agent.chat_completions",
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_unknown_status=True,
        )
        self._record_token_usage(response)
        return self._parse_response(response)

    def _disable_deepseek_thinking_if_needed(self, request_kwargs: dict) -> None:
        try:
            from ddo.runtime.app_runtime import app_runtime

            if app_runtime.cfg is None:
                return

            if app_runtime.cfg.api_cfg.for_normal_agents.provider != "deepseek":
                return

            extra_body = request_kwargs.setdefault("extra_body", {})
            extra_body["thinking"] = {"type": "disabled"}
        except Exception:
            return

    def _record_token_usage(self, response) -> None:
        try:
            from ddo.runtime.app_runtime import app_runtime

            if app_runtime.token_usage is None:
                return

            provider = "openai"
            if app_runtime.cfg is not None:
                provider = app_runtime.cfg.api_cfg.for_normal_agents.provider

            app_runtime.token_usage.add_usage(
                component="normal_agents",
                provider=provider,
                model=self.model_name,
                usage=getattr(response, "usage", None),
            )
        except Exception:
            return


class OpenAIResponsesModelCompat(OpenAIModel):
    """MASFactory OpenAIModel variant that omits null tools for Responses API."""

    def invoke(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        settings: dict | None = None,
        **kwargs,
    ) -> dict:
        tools_dict = [{"type": "function", **tool} for tool in tools] if tools else None
        max_attempts, base_delay, max_delay = _normal_agent_retry_settings(kwargs)

        def request():
            request_kwargs = {
                "model": self.model_name,
                "input": self._encode_responses_input(messages),
                **self._parse_settings(settings),
                **kwargs,
            }
            if tools_dict is not None:
                request_kwargs["tools"] = tools_dict
            self._disable_deepseek_thinking_if_needed(request_kwargs)
            return self._client.responses.create(**request_kwargs)

        response = call_with_openai_retry(
            request,
            operation_name="normal_agent.responses",
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_unknown_status=True,
        )
        self._record_token_usage(response)
        return self._parse_response(response)

    def _disable_deepseek_thinking_if_needed(self, request_kwargs: dict) -> None:
        try:
            from ddo.runtime.app_runtime import app_runtime

            if app_runtime.cfg is None:
                return

            if app_runtime.cfg.api_cfg.for_normal_agents.provider != "deepseek":
                return

            extra_body = request_kwargs.setdefault("extra_body", {})
            extra_body["thinking"] = {"type": "disabled"}
        except Exception:
            return

    def _record_token_usage(self, response) -> None:
        try:
            from ddo.runtime.app_runtime import app_runtime

            if app_runtime.token_usage is None:
                return

            provider = "openai"
            if app_runtime.cfg is not None:
                provider = app_runtime.cfg.api_cfg.for_normal_agents.provider

            app_runtime.token_usage.add_usage(
                component="normal_agents",
                provider=provider,
                model=self.model_name,
                usage=getattr(response, "usage", None),
            )
        except Exception:
            return


def _normal_agent_retry_settings(kwargs: dict) -> tuple[int, float, float]:
    override_attempts = kwargs.pop("max_attempts", kwargs.pop("max_retries", None))
    override_base_delay = kwargs.pop("retry_base_delay", None)
    override_max_delay = kwargs.pop("retry_max_delay", None)

    default_attempts = 6
    default_base_delay = 1.0
    default_max_delay = 30.0
    try:
        from ddo.runtime.app_runtime import app_runtime

        if app_runtime.cfg is not None:
            default_attempts = app_runtime.cfg.normal_agent_max_attempts
            default_base_delay = app_runtime.cfg.retry_base_delay_seconds
            default_max_delay = app_runtime.cfg.retry_max_delay_seconds
    except Exception:
        pass

    return (
        int(override_attempts if override_attempts is not None else default_attempts),
        float(override_base_delay if override_base_delay is not None else default_base_delay),
        float(override_max_delay if override_max_delay is not None else default_max_delay),
    )
