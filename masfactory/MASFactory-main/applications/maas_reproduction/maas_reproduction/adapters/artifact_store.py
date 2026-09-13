"""Safe public artifact persistence."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping

class ArtifactStore:
    def __init__(self, root: str | Path): self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
    def _write(self, name: str, value: Mapping[str, Any]) -> Path:
        text = json.dumps(dict(value), ensure_ascii=False, indent=2, default=str)
        forbidden = ("expected_answer", "api_key", "prompt")
        if any(key in text.lower() for key in forbidden): raise ValueError("forbidden private data in artifact")
        path = self.root / name; path.write_text(text, encoding="utf-8"); return path
    def write_config(self, value: Mapping[str, Any]) -> Path: return self._write("config.json", value)
    def write_sample_result(self, value: Mapping[str, Any]) -> Path: return self._write(f"sample_{value.get('problem_index', 0)}.json", value)
    def write_metrics(self, value: Mapping[str, Any]) -> Path: return self._write("metrics.json", value)

__all__ = ["ArtifactStore"]
