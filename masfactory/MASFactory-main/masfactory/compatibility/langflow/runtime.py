from __future__ import annotations

import json
import re
from typing import Any

_HANDLE_NAME = re.compile(r'"(?:name|fieldName)"\s*:\s*"([^"]+)"')


from masfactory.compatibility.common.runtime import message_text

__all__ = [
    "message_text",
    "langflow_handle_name",
    "langflow_template_fields",
    "langflow_field_value",
]


def langflow_handle_name(handle: Any) -> str | None:
    if handle is None:
        return None
    if isinstance(handle, dict):
        name = handle.get("name") or handle.get("fieldName")
        return str(name) if name else None
    if not isinstance(handle, str) or not handle.strip():
        return None
    text = handle.replace("œ", '"').replace("\u0153", '"')
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("fieldName")
            return str(name) if name else None
    except json.JSONDecodeError:
        pass
    match = _HANDLE_NAME.search(text)
    return match.group(1) if match else None


def langflow_template_fields(raw: dict[str, Any]) -> dict[str, Any]:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    template = node.get("template")
    return template if isinstance(template, dict) else {}


def langflow_field_value(template: dict[str, Any], name: str, default: Any = None) -> Any:
    field = template.get(name)
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return default
