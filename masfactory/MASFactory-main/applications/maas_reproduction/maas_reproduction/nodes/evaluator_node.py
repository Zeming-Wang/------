"""Evaluator node for MaAS benchmark scoring."""

from __future__ import annotations

import logging

from maas_reproduction.benchmarks import GSM8KScorer, HumanEvalScorer, MATHScorer
from maas_reproduction.nodes.artifact_writer import append_mismatch_log, author_round_directory

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
        score, extracted_output = scorer.calculate_score(expected_number, prediction)
        expected_output = expected_number
        columns = ["question", "prediction", "expected_output", "score", "cost", "logprob"]
    elif dataset == "MATH":
        scorer = MATHScorer()
        score, extracted_output = scorer.calculate_score(expected_answer, prediction)
        expected_output = expected_answer
        columns = ["question", "prediction", "expected_output", "score", "cost", "logprob"]
    elif dataset == "HumanEval":
        scorer = HumanEvalScorer(log_path=settings.run_directory)
        result = scorer.check_solution(prediction, expected_answer, str(input_data["entry_point"]))
        score = 1.0 if result[0] == scorer.PASS else 0.0
        extracted_output = score
        expected_output = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else expected_answer
        columns = ["inputs", "prediction", "expected_output", "score", "cost", "logprob"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if score == 0:
        append_mismatch_log(
            author_round_directory(settings),
            str(input_data["problem"]),
            expected_output,
            prediction,
            extracted_output,
            extract_answer_code=_extract_answer_code(dataset),
        )

    result_row = [
        str(input_data["problem"]),
        prediction,
        expected_output,
        score,
        input_data["cost"],
        input_data["logprob"],
    ]
    attributes.setdefault("result_columns", columns)
    attributes.setdefault("sample_results", []).append(result_row)

    logger.info("MaAS evaluator scored problem %s on %s as %s", input_data["problem_index"], dataset, score)
    return {
        "score": score,
        "cost": input_data["cost"],
        "logprob": input_data["logprob"],
        "problem_index": input_data["problem_index"],
        "result_columns": columns,
        "result_row": result_row,
    }


def _extract_answer_code(dataset: str) -> str:
    if dataset != "MATH":
        return "None"
    return (
        "def extract_model_answer(self, text: str) -> str:\n"
        '    pattern = r"\\\\boxed{((?:[^{}]|{[^{}]*})*)}"\n'
        "    boxed_matches = re.findall(pattern, text, re.DOTALL)\n"
        "    if boxed_matches:\n"
        "        return boxed_matches[-1].strip()\n"
        "    sentence_end_pattern = r\"(?<!\\\\d)[.!?]\\\\s+\"\n"
        "    sentences = re.split(sentence_end_pattern, text)\n"
        "    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]\n"
        "    return sentences[-1] if sentences else \"\""
    )
