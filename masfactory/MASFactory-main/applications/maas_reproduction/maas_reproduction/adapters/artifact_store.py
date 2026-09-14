"""Safe public artifact persistence."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

class ArtifactStore:
    def __init__(self, root: str | Path): self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def _write(self, name: str, value: Mapping[str, Any]) -> Path:
        _reject_expected_answer(value)
        text = json.dumps(dict(value), ensure_ascii=False, indent=2, default=str)
        path = self.root / name; path.write_text(text, encoding="utf-8"); return path
    def write_config(self, value: Mapping[str, Any]) -> Path: return self._write("config.json", value)
    def write_sample_result(self, value: Mapping[str, Any]) -> Path: return self._write(f"sample_{value.get('problem_index', 0)}.json", value)
    def write_metrics(self, value: Mapping[str, Any]) -> Path: return self._write("metrics.json", value)


def _reject_expected_answer(value: Any) -> None:
    """Reject only the evaluator-only field, recursively, by exact key."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() == "expected_answer":
                raise ValueError("forbidden private data in artifact: expected_answer")
            _reject_expected_answer(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_expected_answer(child)

__all__ = ["ArtifactStore"]
