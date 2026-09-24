from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from masfactory.compatibility.common.io import load_document_auto, load_json_text, load_yaml_text
from masfactory.compatibility.errors import CompatibilityImportError


def load_mapping_source(
    source: str | Path | bytes,
    *,
    label: str = "document",
) -> tuple[dict[str, Any], Path | None]:
    """Load a YAML/JSON mapping from a path, inline text, or bytes."""
    if isinstance(source, bytes):
        text = source.decode("utf-8")
        try:
            doc = load_json_text(text)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            doc = load_yaml_text(text)
        return _require_mapping(doc, label=label), None

    if isinstance(source, Path):
        doc = load_document_auto(source)
        return _require_mapping(doc, label=label), source

    path = Path(source)
    if path.exists():
        doc = load_document_auto(path)
        return _require_mapping(doc, label=label), path

    try:
        doc = load_json_text(str(source))
    except (json.JSONDecodeError, ValueError):
        doc = load_yaml_text(str(source))
    return _require_mapping(doc, label=label), None


def _require_mapping(doc: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise CompatibilityImportError(f"{label} must parse to a mapping at the root.")
    return doc


