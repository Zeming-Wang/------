"""Small adapter helpers shared by Programmer nodes."""
from __future__ import annotations
from typing import Any


def call_adapter(adapter: Any, payload: dict[str, Any]) -> Any:
    if adapter is None:
        raise RuntimeError("operator adapter is not configured")
    if hasattr(adapter, "invoke"):
        value = adapter.invoke(payload)
        return value[0] if isinstance(value, tuple) else value
    if callable(adapter):
        return adapter(payload)
    raise TypeError("operator adapter must expose invoke() or be callable")


def extract_code(value: Any) -> str | None:
    if isinstance(value, str): return value
    if isinstance(value, dict): return value.get("code") or value.get("solution") or value.get("response")
    return getattr(value, "code", None) or getattr(value, "solution", None) or getattr(value, "response", None)
