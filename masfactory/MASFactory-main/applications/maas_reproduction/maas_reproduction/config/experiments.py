"""Fixed MaAS experiment definitions.

The operator tuples below are the MaAS search space for each supported dataset. They are
copied from the original implementation and must not be changed by framework migration work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QuestionType = Literal["math", "code", "qa"]

#定义实验配置
@dataclass(frozen=True)
class ExperimentConfig:
    """Dataset-specific MaAS search configuration."""

    dataset: str
    question_type: QuestionType
    operators: tuple[str, ...]


EXPERIMENT_CONFIGS: dict[str, ExperimentConfig] = {
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=(
            "Generate",
            "GenerateCoT",
            "MultiGenerateCoT",
            "ScEnsemble",
            "Programmer",
            "SelfRefine",
            "EarlyStop",
        ),
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=(
            "Generate",
            "GenerateCoT",
            "MultiGenerateCoT",
            "ScEnsemble",
            "Programmer",
            "SelfRefine",
            "EarlyStop",
        ),
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=(
            "Generate",
            "GenerateCoT",
            "MultiGenerateCoT",
            "ScEnsemble",
            "Test",
            "SelfRefine",
            "EarlyStop",
        ),
    ),
}

