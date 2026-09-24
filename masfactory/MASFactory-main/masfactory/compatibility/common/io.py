from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _require_yaml():
    import yaml  # type: ignore[import-untyped]

    return yaml


def load_json_text(text: str) -> Any:
    return json.loads(text)


def load_yaml_text(text: str) -> Any:
    yaml = _require_yaml()
    return yaml.safe_load(text)


def load_document_auto(path: str | Path) -> Any:
    """Load JSON or YAML based on file suffix."""
    p = Path(path)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    if suffix in {".json"}:
        return load_json_text(text)
    if suffix in {".yaml", ".yml"}:
        return load_yaml_text(text)
    # Heuristic: try JSON first, then YAML.
    try:
        return load_json_text(text)
    except json.JSONDecodeError:
        return load_yaml_text(text)
