"""Result node forward function for the MaAS reproduction workflow."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def result_forward(input_data: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
    result = {
        "average_score": input_data["average_score"],
        "round": input_data["round"],
        "checkpoint_path": input_data["checkpoint_path"],
        "result_path": input_data["result_path"],
        "runtime_metadata": input_data["runtime_metadata"],
    }
    logger.info("MaAS reproduction finished: average_score=%s round=%s", result["average_score"], result["round"])
    return result
