"""Loop controller function for MaAS training and evaluation."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def training_controller(input_data: dict[str, object], attributes: dict[str, object]) -> bool:
    for key in list(input_data.keys()):
        if key.startswith("result_"):
            del input_data[key]

    settings = attributes["settings"]
    problems = attributes["problems"]
    problem_index = int(attributes.get("problem_index", 0))
    repetition = int(attributes.get("repetition", 1))
    max_repetitions = int(settings.optimizer.sample)

    if problem_index >= len(problems):
        if repetition >= max_repetitions:
            _write_final_result(input_data, attributes)
            logger.info("MaAS training loop finished with average score %.5f", input_data["average_score"])
            return True
        repetition += 1
        problem_index = 0
        attributes["repetition"] = repetition
        logger.info("MaAS training loop starting repetition %s/%s", repetition, max_repetitions)

    problem = problems[problem_index]
    _write_problem_message(input_data, settings.dataset, problem, problem_index)
    attributes["problem_index"] = problem_index + 1
    attributes["repetition"] = repetition
    return False


def _write_problem_message(
    input_data: dict[str, object],
    dataset: str,
    problem: dict[str, object],
    problem_index: int,
) -> None:
    if dataset == "GSM8K":
        input_data["problem"] = problem["question"]
        input_data["entry_point"] = ""
        input_data["expected_answer"] = problem["answer"]
    elif dataset == "MATH":
        input_data["problem"] = problem["problem"]
        input_data["entry_point"] = ""
        input_data["expected_answer"] = problem["solution"]
    elif dataset == "HumanEval":
        input_data["problem"] = problem["prompt"]
        input_data["entry_point"] = problem["entry_point"]
        input_data["expected_answer"] = problem["test"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    input_data["problem_index"] = problem_index


def _write_final_result(input_data: dict[str, object], attributes: dict[str, object]) -> None:
    settings = attributes["settings"]
    all_scores = attributes.get("all_scores", [])
    average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    input_data["average_score"] = average_score
    input_data["round"] = settings.optimizer.round_number
    input_data["checkpoint_path"] = str(
        settings.paths.controller_checkpoint(
            settings.dataset,
            settings.optimizer.round_number,
            settings.optimizer.sample,
        )
    )
    input_data["result_path"] = str(settings.run_directory)
    input_data["runtime_metadata"] = {
        "dataset": settings.dataset,
        "mode": settings.mode,
        "processed_problems": len(all_scores),
    }
