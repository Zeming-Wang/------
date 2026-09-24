from __future__ import annotations

import re
from typing import Any

_VAR_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_langflow_template(text: str, variables: dict[str, Any]) -> str:
    if not text:
        return ""

    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in variables:
            return match.group(0)
        val = variables[key]
        return "" if val is None else str(val)

    return _VAR_PATTERN.sub(_repl, text)
