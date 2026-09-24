from __future__ import annotations

import re
from typing import Any

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_chatdev_template(text: str, variables: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    merged = {**(extra or {}), **variables}

    def _repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in merged:
            return match.group(0)
        return str(merged[key])

    return _VAR_PATTERN.sub(_repl, text)
