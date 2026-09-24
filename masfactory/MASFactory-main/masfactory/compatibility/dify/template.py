from __future__ import annotations

import re
from typing import Any

from masfactory.compatibility.dify.runtime import DifyRuntimeState

_DIFY_REF = re.compile(r"\{\{#([^#]+?)#\}\}")


def _apply_filter(value: Any, filt: str | None) -> Any:
    if not filt:
        return value
    name = filt.strip().lower()
    if name == "length":
        if value is None:
            return 0
        if isinstance(value, (list, tuple, set, frozenset)):
            return len(value)
        if isinstance(value, dict):
            return len(value)
        if isinstance(value, str):
            return len(value)
        return len(str(value))
    return value


def render_dify_template(template: str, state: DifyRuntimeState) -> str:
    if not template:
        return ""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        filt: str | None = None
        if "|" in inner:
            path_part, filt = inner.rsplit("|", 1)
            path_part = path_part.strip()
            filt = filt.strip()
        else:
            path_part = inner
        path = [part.strip() for part in path_part.split(".") if part.strip()]
        value = state.resolve_selector(path)
        value = _apply_filter(value, filt)
        if value is None:
            return ""
        return str(value)

    return _DIFY_REF.sub(_replace, template)
