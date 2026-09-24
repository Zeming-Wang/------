from __future__ import annotations

from typing import Any

from masfactory.compatibility.common.conditions import is_always_true_condition
from masfactory.compatibility.common.runtime import message_text as _message_text

__all__ = ["message_text", "is_always_true_condition", "evaluate_chatdev_condition"]


def message_text(payload: dict[str, Any]) -> str:
    return _message_text(payload, scan_all_string_values=True)


def evaluate_chatdev_condition(
    cond: Any,
    *,
    message: dict[str, Any],
    attributes: dict[str, object],
) -> bool:
    if is_always_true_condition(cond):
        return True
    if not isinstance(cond, dict):
        return False
    ctype = str(cond.get("type") or "")
    cfg = cond.get("config") if isinstance(cond.get("config"), dict) else {}
    text = message_text({**dict(attributes), **message})
    if ctype == "keyword":
        any_kw = cfg.get("any") or cfg.get("keywords") or []
        if isinstance(any_kw, str):
            any_kw = [any_kw]
        if any_kw and any(str(k) in text for k in any_kw):
            return True
        all_kw = cfg.get("all")
        if isinstance(all_kw, list) and all_kw:
            return all(str(k) in text for k in all_kw)
        none_kw = cfg.get("none")
        if isinstance(none_kw, list) and none_kw:
            return not any(str(k) in text for k in none_kw)
        return not any_kw
    return False
