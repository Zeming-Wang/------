from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BUILTIN_DATASETS = {
    "mini_math": DATA_DIR / "mini_math.jsonl",
}


def load_jsonl_dataset(path: str | Path, *, limit: int | None = None) -> list[dict[str, str]]:
    dataset_path = Path(path)
    samples: list[dict[str, str]] = []
    with dataset_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "question" not in record or "answer" not in record:
                raise ValueError(f"Line {line_number} must contain question and answer fields")
            samples.append({"question": str(record["question"]), "answer": str(record["answer"])})
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_builtin_dataset(name: str, *, limit: int | None = None) -> list[dict[str, str]]:
    try:
        path = BUILTIN_DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BUILTIN_DATASETS))
        raise ValueError(f"Unknown built-in dataset '{name}'. Available: {available}") from exc
    return load_jsonl_dataset(path, limit=limit)
