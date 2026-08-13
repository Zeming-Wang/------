"""GSM8K answer extraction and scoring."""

from __future__ import annotations

import re

#即整个gsm8k的评测原则，提取最后一个数字然后判分
class GSM8KScorer:
    def extract_number(self, text: str) -> float | None:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        return None

    def calculate_score(self, expected_output: float, prediction: str | float | None) -> tuple[float, float | None]:
        predicted_number = self.extract_number(str(prediction)) if prediction is not None else None
        if predicted_number is None:
            return 0.0, predicted_number
        return 1.0 if abs(expected_output - predicted_number) <= 1e-6 else 0.0, predicted_number
