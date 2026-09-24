from __future__ import annotations

from typing import Any

_MESSAGE_KEYS = ("text", "content", "message", "answer", "output", "input_value")


def message_text(payload: dict[str, Any], *, scan_all_string_values: bool = False) -> str:
    for key in _MESSAGE_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    if scan_all_string_values:
        for val in payload.values():
            if isinstance(val, str) and val.strip():
                return val
    return ""
