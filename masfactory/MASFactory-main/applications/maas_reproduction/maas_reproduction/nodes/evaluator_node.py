"""Evaluator node for MaAS benchmark scoring."""

from __future__ import annotations

import logging

from maas_reproduction.benchmarks import GSM8KScorer, HumanEvalScorer, MATHScorer

logger = logging.getLogger(__name__)


def evaluator_forward(
    input_data: dict[str, object],
    attributes: dict[str, object],
) -> dict[str, object]:
    settings = attributes["settings"]
    dataset = settings.dataset
    prediction = str(input_data["prediction"])
    expected_answer = str(input_data["expected_answer"])

    if dataset == "GSM8K":
        scorer = GSM8KScorer()
        expected_number = scorer.extract_number(expected_answer)
        score, _ = scorer.calculate_score(expected_number, prediction)
    elif dataset == "MATH":
        scorer = MATHScorer()
        score, _ = scorer.calculate_score(expected_answer, prediction)
    elif dataset == "HumanEval":
        scorer = HumanEvalScorer(log_path=settings.run_directory)
        result = scorer.check_solution(prediction, expected_answer, str(input_data["entry_point"]))
        score = 1.0 if result[0] == scorer.PASS else 0.0
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    logger.info("MaAS evaluator scored problem %s on %s as %s", input_data["problem_index"], dataset, score)
    return {
        "score": score,
        "cost": input_data["cost"],
        "logprob": input_data["logprob"],
        "problem_index": input_data["problem_index"],
    }
