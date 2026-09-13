"""Author-compatible result artifacts for the MaAS reproduction."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def author_round_directory(settings) -> Path:
    """Return the MaAS-compatible round directory for result artifacts."""
    architecture_root = getattr(settings, "architecture_root", None)
    if architecture_root is not None:
        return Path(architecture_root) / f"round_{settings.optimizer.round_number}"

    run_directory = getattr(settings, "run_directory", None)
    if run_directory is not None:
        return Path(run_directory)

    paths = getattr(settings, "paths", None)
    if paths is not None and hasattr(paths, "controller_checkpoint"):
        try:
            return Path(
                paths.controller_checkpoint(
                    settings.dataset,
                    settings.optimizer.round_number,
                    settings.optimizer.sample,
                )
            ).parent
        except Exception:
            pass

    return Path(settings.run_directory)


def append_mismatch_log(
    log_directory: Path,
    problem: str,
    expected_output: Any,
    prediction: str,
    extracted_output: Any,
    extract_answer_code: str = "None",
) -> None:
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / "log.json"
    if log_file.exists():
        try:
            data = json.loads(log_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(
        {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
    )
    log_file.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")


def write_results_csv(
    log_directory: Path,
    columns: list[str],
    rows: list[list[Any]],
    average_score: float,
) -> Path:
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_directory / f"{average_score:.5f}_{timestamp}.csv"
    cost_index = columns.index("cost") if "cost" in columns else None
    output_columns = [column for column in columns if column != "cost"]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(output_columns)
        for row in rows:
            output_row = [value for index, value in enumerate(row) if index != cost_index]
            writer.writerow(output_row)

    return csv_path


def append_round_summary(
    results_file: Path,
    round_number: int,
    score: float,
    avg_cost: float = 0.0,
    total_cost: float = 0.0,
    token: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    textgrad_events: list[dict[str, Any]] | None = None,
) -> Path:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(
        {
            "round": round_number,
            "score": score,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "token": token,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": token,
            "textgrad_applied": any(event.get("applied", False) for event in textgrad_events or []),
            "textgrad_events": textgrad_events or [],
            "time": datetime.now().isoformat(sep=" ", timespec="seconds"),
        }
    )
    results_file.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return results_file
