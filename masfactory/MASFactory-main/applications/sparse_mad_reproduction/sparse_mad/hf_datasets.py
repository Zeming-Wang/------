from __future__ import annotations

import os
import random
import re
import tarfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


DatasetLoader = Callable[[str, str, str], Iterable[dict[str, Any]]]
DEFAULT_DEEPMIND_MATH_SUBSET = "algebra__linear_1d_composed"
_DEEPMIND_MATH_PARQUET_REVISION = "ce2e4d517071326389869c0a9e2cbaa123aee6cc"
_DEEPMIND_MATH_ARCHIVE_URL = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"
_DEEPMIND_SPLIT_DIRS = {
    "test": ("interpolate",),
    "train": ("train-easy", "train-medium", "train-hard"),
}


def load_hf_dataset(
    dataset_name: str,
    *,
    split: str = "test",
    limit: int | None = None,
    math_subset: str = "algebra",
    math_level: int | None = None,
    deepmind_math_subset: str = DEFAULT_DEEPMIND_MATH_SUBSET,
    shuffle: bool = False,
    seed: int = 42,
    loader: DatasetLoader | None = None,
    deepmind_loader: DatasetLoader | None = None,
) -> list[dict[str, str]]:
    default_loader = loader or _load_dataset_from_huggingface
    dataset_name = dataset_name.lower()

    if dataset_name == "gsm8k":
        records = default_loader("openai/gsm8k", "main", split)
        samples = (_map_gsm8k_record(record) for record in records)
        return _select_samples(samples, limit=limit, shuffle=shuffle, seed=seed)

    if dataset_name in {"math", "hendrycks_math"}:
        records = default_loader("EleutherAI/hendrycks_math", math_subset, split)
        records = _filter_math_records_by_level(records, math_level)
        samples = (_map_math_record(record) for record in records)
        return _select_samples(samples, limit=limit, shuffle=shuffle, seed=seed)

    if dataset_name in {"deepmind_math", "math_dataset"}:
        records = (loader or deepmind_loader or _load_deepmind_math_from_huggingface)(
            "deepmind/math_dataset", deepmind_math_subset, split
        )
        samples = (_map_deepmind_math_record(record) for record in records)
        return _select_samples(samples, limit=limit, shuffle=shuffle, seed=seed)

    raise ValueError("dataset_name must be 'gsm8k', 'math', or 'deepmind_math'")


def parse_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        answer = answer_text.rsplit("####", 1)[1]
    else:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
        answer = numbers[-1] if numbers else answer_text
    return _clean_answer(answer)


def parse_math_answer(solution: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", solution)
    if boxed:
        return _clean_answer(boxed[-1])

    numbers = re.findall(r"-?\d+(?:\.\d+)?", solution)
    if numbers:
        return _clean_answer(numbers[-1])
    return _clean_answer(solution)


def _map_gsm8k_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": str(record["question"]),
        "answer": parse_gsm8k_answer(str(record["answer"])),
    }


def _map_math_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": str(record["problem"]),
        "answer": parse_math_answer(str(record["solution"])),
    }


def _map_deepmind_math_record(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": str(record["question"]).strip(),
        "answer": _clean_answer(str(record["answer"])),
    }


def _filter_math_records_by_level(records: Iterable[dict[str, Any]], math_level: int | None):
    if math_level is None:
        yield from records
        return

    for record in records:
        if _parse_math_level(record.get("level")) == math_level:
            yield record


def _parse_math_level(level: Any) -> int | None:
    if level is None:
        return None
    match = re.search(r"\d+", str(level))
    return int(match.group(0)) if match else None


def _select_samples(
    samples: Iterable[dict[str, str]],
    *,
    limit: int | None,
    shuffle: bool,
    seed: int,
) -> list[dict[str, str]]:
    if shuffle:
        result = list(samples)
        random.Random(seed).shuffle(result)
        return result[:limit] if limit is not None else result
    return _take_limit(samples, limit)


def _take_limit(samples: Iterable[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for sample in samples:
        result.append(sample)
        if limit is not None and len(result) >= limit:
            break
    return result


def _load_deepmind_math_from_huggingface(path: str, name: str, split: str):
    try:
        return _load_deepmind_math_from_converted_parquet(path, name, split)
    except Exception as parquet_exc:
        try:
            return _load_deepmind_math_from_original_archive(path, name, split)
        except Exception as archive_exc:
            raise RuntimeError(
                "Could not load DeepMind Mathematics Dataset. Tried Hugging Face converted parquet "
                "and the original DeepMind archive."
            ) from archive_exc


def _load_deepmind_math_from_converted_parquet(path: str, name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is not installed. "
            "Run `python -m pip install datasets` or reinstall requirements.txt."
        ) from exc

    parquet_url = _deepmind_math_parquet_url(path, name, split)
    return load_dataset("parquet", data_files={split: parquet_url}, split=split)


def _deepmind_math_parquet_url(path: str, name: str, split: str) -> str:
    return f"hf://datasets/{path}@{_DEEPMIND_MATH_PARQUET_REVISION}/{name}/{split}-*.parquet"


def _load_deepmind_math_from_original_archive(path: str, name: str, split: str):
    del path
    archive_path = _ensure_deepmind_math_archive()
    return _iter_deepmind_math_tar_records(archive_path, name, split)


def _ensure_deepmind_math_archive() -> Path:
    cache_root = Path(os.environ.get("SPARSE_MAD_CACHE_DIR", Path.home() / ".cache" / "sparse_mad"))
    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / "mathematics_dataset-v1.0.tar.gz"
    if not archive_path.exists():
        urllib.request.urlretrieve(_DEEPMIND_MATH_ARCHIVE_URL, archive_path)
    return archive_path


def _iter_deepmind_math_tar_records(archive_path: str | Path, subset: str, split: str):
    split_dirs = _DEEPMIND_SPLIT_DIRS.get(split)
    if split_dirs is None:
        available = ", ".join(sorted(_DEEPMIND_SPLIT_DIRS))
        raise ValueError(f"DeepMind Mathematics split must be one of: {available}")

    expected_members = {
        f"mathematics_dataset-v1.0/{split_dir}/{subset}.txt" for split_dir in split_dirs
    }
    found_member = False
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in expected_members:
                continue
            found_member = True
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            yield from _iter_question_answer_lines(extracted)

    if not found_member:
        expected = ", ".join(sorted(expected_members))
        raise ValueError(f"DeepMind Mathematics subset file not found in archive: {expected}")


def _iter_question_answer_lines(handle):
    question: str | None = None
    for raw_line in handle:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        if question is None:
            question = line
        else:
            yield {"question": question, "answer": line}
            question = None


def _load_dataset_from_huggingface(path: str, name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face datasets package is not installed. "
            "Run `python -m pip install datasets` or reinstall requirements.txt."
        ) from exc

    return load_dataset(path, name, split=split)


def _clean_answer(answer: str) -> str:
    return answer.strip().replace(",", "").strip("$ ").rstrip(".")
