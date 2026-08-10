"""Dataset loading helpers for MaAS runtime state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl_data(file_path: str | Path, specific_indices: Iterable[int] | None = None) -> list[dict]:
    path = Path(file_path)
    data = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    if specific_indices is not None:
        return [data[index] for index in specific_indices if index < len(data)]
    return data


def load_problems(settings, specific_indices: Iterable[int] | None = None) -> list[dict]:
    return load_jsonl_data(settings.dataset_file, specific_indices=specific_indices)
