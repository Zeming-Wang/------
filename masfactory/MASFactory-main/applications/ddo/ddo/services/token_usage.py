from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _get_usage_value(usage: Any, *names: str) -> int:
    if usage is None:
        return 0

    if not isinstance(usage, dict):
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage = usage.dict()

    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)

        if isinstance(value, (int, float)):
            return int(value)

    return 0


@dataclass
class TokenUsageBucket:
    provider: str
    model: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, usage: Any) -> None:
        input_tokens = _get_usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _get_usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = _get_usage_value(usage, "total_tokens")

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        self.requests += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TokenUsageRecorder:
    buckets: dict[str, TokenUsageBucket] = field(default_factory=dict)

    def add_usage(self, *, component: str, provider: str, model: str, usage: Any) -> None:
        bucket = self.buckets.get(component)
        if bucket is None:
            bucket = TokenUsageBucket(provider=provider, model=model)
            self.buckets[component] = bucket

        bucket.add_usage(usage)

    def snapshot(self) -> dict[str, Any]:
        by_component = {
            component: bucket.snapshot()
            for component, bucket in self.buckets.items()
        }

        total = {
            "requests": sum(bucket.requests for bucket in self.buckets.values()),
            "input_tokens": sum(bucket.input_tokens for bucket in self.buckets.values()),
            "output_tokens": sum(bucket.output_tokens for bucket in self.buckets.values()),
            "total_tokens": sum(bucket.total_tokens for bucket in self.buckets.values()),
        }

        return {
            "total": total,
            "by_component": by_component,
        }
